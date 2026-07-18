# ============================================================================
# Continuation 5, moment-construction audit: warmed before/after profile of
# EK_moments_gammanorm_directgp! (original) vs _fast! (mu/sigma-cached), plus
# an internal component breakdown of the FAST version (UPow/UσPow cache
# lookup vs hFunction!/hFunctionCounter! draw loop vs post-processing), and a
# W/D scaling grid.
# ============================================================================
include(joinpath(@__DIR__, "context_scaled.jl"))   # pulls in context.jl internally -- do NOT also include context.jl separately (re-including it causes a binding-ambiguity error, see moments_fast.jl's note)
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "moments_fast.jl"))
using Printf, Statistics

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "profile_moments_fast")
mkpath(OUTDIR)
const NTHREADS = Threads.nthreads()

function bench_at(D::Int, W::Int; label::String)
    ctx = D == 4 && W == 8000 ? d4_exact_setup(find_smallest = true) :
          d_exact_setup_scaled(D = D, W = W, find_smallest = true)
    zfree0 = zeros(CS.n_free(ctx.m) - 1)
    xf = vcat(ctx.θ0_up[3+ctx.D], ones(ctx.D^2))   # theta==1 calibration-style point (Aod_theta=1 everywhere), always constructible regardless of cold-infeasibility of the inner DUAL solve (moments! doesn't need the dual to be feasible)
    θ_full = CS.reconstruct_full(xf, ctx.m)
    Wn = size(ctx.U, 1); d = ctx.obj.d
    K = zeros(Wn); G = zeros(Wn, d)

    # ---- warm-up (JIT) both versions ----
    ctx.obj.moments!(K, G, θ_full, ctx.U, ctx.obj)
    pow_cache = MuSigmaPowCache(ctx.U, ctx.obj.γ.Uσ)
    EK_moments_gammanorm_directgp_fast!(K, G, θ_full, ctx.U, ctx.obj, pow_cache)

    N = 30
    t_orig = median([@elapsed ctx.obj.moments!(K, G, θ_full, ctx.U, ctx.obj) for _ in 1:N])
    fresh_cache_for_first = MuSigmaPowCache(ctx.U, ctx.obj.γ.Uσ)
    EK_moments_gammanorm_directgp_fast!(K, G, θ_full, ctx.U, ctx.obj, fresh_cache_for_first)   # JIT warm-up of the cold-cache path (untimed)
    t_fast_first = median([@elapsed EK_moments_gammanorm_directgp_fast!(K, G, θ_full, ctx.U, ctx.obj, MuSigmaPowCache(ctx.U, ctx.obj.γ.Uσ)) for _ in 1:N])   # fresh cache each rep, forces one recompute each time
    t_fast_warm = median([@elapsed EK_moments_gammanorm_directgp_fast!(K, G, θ_full, ctx.U, ctx.obj, pow_cache) for _ in 1:N])   # cache already warm from above

    # ---- component breakdown of the FAST version (UPow cache lookup vs hFunction!/hFunctionCounter! draw loop vs post-processing) ----
    μ = θ_full[1]
    t_cache_lookup = median([@elapsed get_upow!(pow_cache, ctx.U, ctx.obj.γ.Uσ, μ) for _ in 1:N])   # already warm -- pure lookup cost
    t_cache_cold = median([@elapsed get_upow!(MuSigmaPowCache(ctx.U, ctx.obj.γ.Uσ), ctx.U, ctx.obj.γ.Uσ, μ) for _ in 1:N])   # forces recompute each rep (fresh cache each time)

    @printf("  D=%-3d W=%-6d (%s): orig=%.4fms  fast_first(cold cache)=%.4fms  fast_warm=%.4fms  speedup=%.2fx  |  UPow cache: cold_recompute=%.4fms warm_lookup=%.4fms\n",
        D, W, label, t_orig*1000, t_fast_first*1000, t_fast_warm*1000, t_orig/t_fast_warm, t_cache_cold*1000, t_cache_lookup*1000)

    return (D = D, W = W, label = label, orig_ms = t_orig*1000, fast_first_ms = t_fast_first*1000,
            fast_warm_ms = t_fast_warm*1000, speedup = t_orig/t_fast_warm,
            upow_cold_recompute_ms = t_cache_cold*1000, upow_warm_lookup_ms = t_cache_lookup*1000,
            nthreads = NTHREADS)
end

println("="^78); println("Warmed before/after: EK_moments_gammanorm_directgp! (orig) vs _fast! (mu/sigma-cached), NTHREADS=$NTHREADS"); println("="^78)
rows = NamedTuple[]
push!(rows, bench_at(4, 8000; label = "D4W8000 (primary)"))

# D=4, W=80000 -- per task's explicit benchmark grid
push!(rows, bench_at(4, 80000; label = "D4W80000"))

# D=6 or D=8 at W=8000
push!(rows, bench_at(6, 8000; label = "D6W8000"))
push!(rows, bench_at(8, 8000; label = "D8W8000"))

# D=10 at W=8000, "if practical"
push!(rows, bench_at(10, 8000; label = "D10W8000"))

open(joinpath(OUTDIR, "moments_fast_scaling.csv"), "w") do io
    cols = keys(rows[1])
    println(io, join(cols, ","))
    for r in rows
        println(io, join((r[c] for c in cols), ","))
    end
end
println("\nWrote ", joinpath(OUTDIR, "moments_fast_scaling.csv"))
