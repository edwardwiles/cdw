# ============================================================================
# Phase 1C: baseline D/W scaling matrix. COMPUTATIONAL BENCHMARK ONLY -- see
# context_scaled.jl's header. Does NOT run long outer optimizations (per the
# task's explicit instruction for this subphase); measures per-evaluation and
# per-gradient cost only, at each config's own theta0 (calibration point for
# that config's own synthetic economy).
# ============================================================================
include(joinpath(@__DIR__, "context_scaled.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "derivative_methods.jl"))
using Printf

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "profile_D_W_scaling")
mkpath(OUTDIR)
const H = 0.01

# (D, W, n_eval_reps, n_grad_reps) -- fewer reps at larger/more expensive configs, documented not hidden
# CONFIGS_SUBSET env var (e.g. "D8", "D10", "W20000", "W80000") lets this be resumed one config at a
# time -- added after repeated background-job termination made a single long run unreliable in this
# environment (see docs/fullA_next_handoff.md).
const _ALL_CONFIGS = Dict(
    "D4"     => (4, 8000, 10, 5),
    "D6"     => (6, 8000, 10, 5),
    "D8"     => (8, 8000, 8, 3),
    "D10"    => (10, 8000, 5, 3),
    "W20000" => (4, 20000, 8, 3),
    "W80000" => (4, 80000, 5, 2),
)
const CONFIGS = if haskey(ENV, "CONFIGS_SUBSET")
    [_ALL_CONFIGS[k] for k in split(ENV["CONFIGS_SUBSET"], ",")]
else
    [_ALL_CONFIGS[k] for k in ("D4","D6","D8","D10","W20000","W80000")]
end
const OUT_SUFFIX = get(ENV, "CONFIGS_SUBSET", "all")

function median_stat(times)
    s = sort(times); n = length(s)
    return (median = s[n÷2+1], min = s[1], max = s[end], mean = sum(s)/n)
end

function bench_config(D::Int, W::Int, n_eval_reps::Int, n_grad_reps::Int)
    println("="^78); println("CONFIG D=$D W=$W"); println("="^78); flush(stdout)

    t_ctx0 = time_ns()
    ctx = d_exact_setup_scaled(D = D, W = W)
    t_ctx_build = (time_ns() - t_ctx0) / 1e9
    println("  ctx build time: $(round(t_ctx_build,digits=2))s")

    pe = build_pivot_elimination(ctx)
    x0 = CS.pack_free(ctx.θ0_up, ctx.m)
    n_free = length(x0)

    # ---- exact hard value: warm-up then N reps, warm and cold ----
    evaluate_fullA(x0, ctx; cache = nothing, warm = true)   # JIT warm-up (discarded)
    warm_times = Float64[]; cold_times = Float64[]
    for _ in 1:n_eval_reps
        t0 = time_ns(); evaluate_fullA(x0, ctx; cache = nothing, warm = true); push!(warm_times, (time_ns()-t0)/1e9)
    end
    for _ in 1:n_eval_reps
        t0 = time_ns(); evaluate_fullA(x0, ctx; cache = nothing, warm = false); push!(cold_times, (time_ns()-t0)/1e9)
    end
    sw = median_stat(warm_times); sc = median_stat(cold_times)
    println("  exact eval (warm): median=$(round(sw.median,digits=4))s  (cold): median=$(round(sc.median,digits=4))s")

    # ---- full gradients: pathwise AD, Q_adj FD, L_fix FD, Delta FD ----
    base = solve_base_state(x0, ctx)
    alloc0 = Base.gc_bytes()
    t0 = time_ns(); for _ in 1:n_grad_reps; method_A_pathwise_ad(x0, ctx, base); end
    t_pathwise = ((time_ns()-t0)/1e9) / n_grad_reps
    alloc_pathwise = (Base.gc_bytes() - alloc0) / n_grad_reps

    function fd_grad_time(f)
        alloc_a = Base.gc_bytes()
        t0 = time_ns()
        for _ in 1:n_grad_reps
            g = zeros(n_free)
            for i in 1:n_free
                xp = copy(x0); xp[i] += H; xm = copy(x0); xm[i] -= H
                g[i] = (f(xp) - f(xm)) / (2H)
            end
        end
        return ((time_ns()-t0)/1e9)/n_grad_reps, (Base.gc_bytes()-alloc_a)/n_grad_reps
    end
    t_qadj, a_qadj = fd_grad_time(x -> frozen_adjoint_Q(x, ctx, base))
    t_lfix, a_lfix = fd_grad_time(x -> fixed_dual_L(x, ctx, base))
    t_delta, a_delta = fd_grad_time(x -> optimized_Delta(x, ctx; warm=true))

    println("  full gradients (n_free=$n_free): pathwise=$(round(t_pathwise,digits=3))s  Q_adj_FD=$(round(t_qadj,digits=3))s  L_fix_FD=$(round(t_lfix,digits=3))s  Delta_FD=$(round(t_delta,digits=3))s")

    return (D = D, W = W, n_free = n_free, nTotalMoments = ctx.nTotalMoments, ctx_build_s = t_ctx_build,
            eval_warm_median_s = sw.median, eval_warm_min_s = sw.min,
            eval_cold_median_s = sc.median, eval_cold_min_s = sc.min,
            grad_pathwise_s = t_pathwise, grad_pathwise_alloc_bytes = alloc_pathwise,
            grad_Qadj_FD_s = t_qadj, grad_Qadj_FD_alloc_bytes = a_qadj,
            grad_Lfix_FD_s = t_lfix, grad_Lfix_FD_alloc_bytes = a_lfix,
            grad_Delta_FD_s = t_delta, grad_Delta_FD_alloc_bytes = a_delta,
            n_eval_reps = n_eval_reps, n_grad_reps = n_grad_reps)
end

rows = NamedTuple[]
for (D, W, ner, ngr) in CONFIGS
    try
        push!(rows, bench_config(D, W, ner, ngr))
    catch e
        println("  FAILED at D=$D W=$W: ", e)
        push!(rows, (D = D, W = W, n_free = -1, nTotalMoments = -1, ctx_build_s = NaN,
            eval_warm_median_s = NaN, eval_warm_min_s = NaN, eval_cold_median_s = NaN, eval_cold_min_s = NaN,
            grad_pathwise_s = NaN, grad_pathwise_alloc_bytes = NaN, grad_Qadj_FD_s = NaN, grad_Qadj_FD_alloc_bytes = NaN,
            grad_Lfix_FD_s = NaN, grad_Lfix_FD_alloc_bytes = NaN, grad_Delta_FD_s = NaN, grad_Delta_FD_alloc_bytes = NaN,
            n_eval_reps = ner, n_grad_reps = ngr))
    end
    flush(stdout)
end

open(joinpath(OUTDIR, "profile_D_W_scaling_$(OUT_SUFFIX).csv"), "w") do io
    cols = keys(rows[1])
    println(io, join(cols, ","))
    for r in rows
        println(io, join((r[c] for c in cols), ","))
    end
end
println("\nWrote ", joinpath(OUTDIR, "profile_D_W_scaling_$(OUT_SUFFIX).csv"))
