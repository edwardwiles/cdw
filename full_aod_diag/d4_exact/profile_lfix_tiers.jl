# ============================================================================
# Phase 2: profile full-rebuild, block-local, incremental, and O(1)-winner
# tiers SEPARATELY (per explicit task/user instruction -- do not report only
# theoretical speedups). Also benchmarks thread-parallel FD probing for the
# incremental/O(1) tiers (safe to parallelize: unlike fixed_dual_L, these
# tiers never touch ctx.obj's mutable state or call obj.moments! at all --
# each probe allocates its own local scratch, no nested-threading
# oversubscription risk with EK_moments_gammanorm_directgp!'s own internal
# Threads.@threads, which the full-rebuild tier WOULD hit if parallelized --
# not attempted here for that reason, noted not silently skipped).
#
# Run: julia --project=. full_aod_diag/d4_exact/profile_lfix_tiers.jl
# For the parallel scaling table, re-run with JULIA_NUM_THREADS=1,2,4,8,16.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
using Printf

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "profile_lfix_tiers")
mkpath(OUTDIR)
const NTHREADS = Threads.nthreads()

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
xf0 = x_free_from_w(w_up40)
base = solve_base_state(xf0, ctx)
cache = build_lfix_base_cache(xf0, ctx, base)
n = length(w_up40)
const HFD = 0.01

function grad_full_rebuild(w0)
    g = zeros(n)
    for i in 1:n
        wp = copy(w0); wp[i] += HFD; wm = copy(w0); wm[i] -= HFD
        g[i] = (fixed_dual_L(x_free_from_w(wp), ctx, base) - fixed_dual_L(x_free_from_w(wm), ctx, base)) / (2 * HFD)
    end
    return g
end
function grad_tier(w0, tier::Symbol)
    g = zeros(n)
    for i in 1:n
        Lp = lfix_incremental_at(cache, ctx, pe, w0, i, w0[i] + HFD; tier = tier)
        Lm = lfix_incremental_at(cache, ctx, pe, w0, i, w0[i] - HFD; tier = tier)
        g[i] = (Lp - Lm) / (2 * HFD)
    end
    return g
end
function grad_tier_threaded(w0, tier::Symbol)
    g = zeros(n)
    Threads.@threads for i in 1:n
        Lp = lfix_incremental_at(cache, ctx, pe, w0, i, w0[i] + HFD; tier = tier)
        Lm = lfix_incremental_at(cache, ctx, pe, w0, i, w0[i] - HFD; tier = tier)
        g[i] = (Lp - Lm) / (2 * HFD)
    end
    return g
end

# ---- gradient-VALUE cross-check: all 4 methods must agree on the actual gradient (not just the
#      per-perturbation L_fix scalar, already checked in test_lfix_incremental.jl) ----
g_full = grad_full_rebuild(w_up40)
g_bl = grad_tier(w_up40, :block_local)
g_inc = grad_tier(w_up40, :incremental)
g_o1 = grad_tier(w_up40, :incremental_o1)
g_o1_threaded = grad_tier_threaded(w_up40, :incremental_o1)
println("gradient cross-check (max abs diff vs full-rebuild):")
println("  block_local:        ", maximum(abs.(g_bl .- g_full)))
println("  incremental:         ", maximum(abs.(g_inc .- g_full)))
println("  incremental_o1:      ", maximum(abs.(g_o1 .- g_full)))
println("  incremental_o1(thr): ", maximum(abs.(g_o1_threaded .- g_full)))
gradient_ok = maximum(abs.(g_bl .- g_full)) < 1e-8 && maximum(abs.(g_inc .- g_full)) < 1e-8 &&
              maximum(abs.(g_o1 .- g_full)) < 1e-8 && maximum(abs.(g_o1_threaded .- g_full)) < 1e-8
println("gradient cross-check: ", gradient_ok ? "PASS" : "FAIL")
gradient_ok || error("profile_lfix_tiers.jl: gradient mismatch across tiers -- do not trust timing below")

# ---- warm-up (JIT) ----
grad_full_rebuild(w_up40); grad_tier(w_up40, :block_local); grad_tier(w_up40, :incremental)
grad_tier(w_up40, :incremental_o1); grad_tier_threaded(w_up40, :incremental_o1)

# ---- timed N reps, single-threaded ----
const N_REPS = 20
function time_reps(f, args...)
    times = Float64[]
    for _ in 1:N_REPS
        t0 = time_ns(); f(args...); push!(times, (time_ns() - t0) / 1e9)
    end
    sort!(times)
    return (median = times[N_REPS÷2+1], min = times[1], mean = sum(times) / N_REPS)
end

t_full = time_reps(grad_full_rebuild, w_up40)
t_bl = time_reps(grad_tier, w_up40, :block_local)
t_inc = time_reps(grad_tier, w_up40, :incremental)
t_o1 = time_reps(grad_tier, w_up40, :incremental_o1)
t_o1_thr = time_reps(grad_tier_threaded, w_up40, :incremental_o1)

println("\n" * "="^78); println("SINGLE-THREADED TIER COMPARISON (N=$N_REPS full 16-dim gradients, D=$D/W=$(size(ctx.U,1)))"); println("="^78)
rows = NamedTuple[]
for (label, t) in (("full_rebuild (fixed_dual_L, baseline)", t_full), ("block_local (Tier 1)", t_bl),
                    ("incremental (Tier 2)", t_inc), ("incremental_o1 (Tier 3)", t_o1),
                    ("incremental_o1_threaded (nthreads=$NTHREADS)", t_o1_thr))
    speedup = t_full.median / t.median
    @printf("  %-42s median=%.6fs  min=%.6fs  speedup_vs_full=%.2fx\n", label, t.median, t.min, speedup)
    push!(rows, (label = label, median_s = t.median, min_s = t.min, mean_s = t.mean, speedup_vs_full = speedup, nthreads = NTHREADS))
end
write_csv_rows(joinpath(OUTDIR, "profile_lfix_tiers_nthreads$(NTHREADS).csv"), rows)

open(joinpath(OUTDIR, "summary_nthreads$(NTHREADS).txt"), "w") do io
    println(io, "D=", D, " W=", size(ctx.U,1), " N_REPS=", N_REPS, " NTHREADS=", NTHREADS)
    println(io, "gradient cross-check: PASS (all tiers agree with full-rebuild to <1e-8)")
    for r in rows
        println(io, r.label, ": median=", r.median_s, "s  speedup_vs_full=", r.speedup_vs_full, "x")
    end
end
println("\nWrote ", OUTDIR, "/profile_lfix_tiers_nthreads$(NTHREADS).csv")
