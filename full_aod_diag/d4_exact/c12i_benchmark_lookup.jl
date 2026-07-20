# Part B.4: end-to-end benchmark. Compares, at L in {10,20,50}:
#   (i)   dense BLAS matmul baseline (existing production PsiObjectiveBundleImplicit callable,
#         via a real live KNITRO inner solve -- oracle.jl::evaluate_fullA, UNMODIFIED)
#   (ii)  cumulative-basis suffix-sum lookup, wired into a real live KNITRO inner solve
#   (iii) interval-basis lookup, wired into a real live KNITRO inner solve
# Each timed as a FULL inner KNITRO dual solve (not an isolated microkernel), repeated `nrep`
# times cold (warm=false, so KNITRO always does a comparable number of iterations rather than
# reusing a cached near-optimal start) after JIT-warming every code path once. Also separately
# benchmarks the O(D) lookup FG kernel in isolation (a realistic N=n_iters evaluations, matching
# the ACTUAL number of KNITRO FG calls the live solve used) as a secondary, lower-noise data
# point, and the threaded histogram builder (Part B.2) at 1/4/8/16 threads.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "cm_lookup_live_knitro.jl"))
using Printf, Statistics, Dates

println("nproc reported by Julia (Sys.CPU_THREADS) = ", Sys.CPU_THREADS, "   JULIA_NUM_THREADS = ", Threads.nthreads())

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
x_free_calib = ctx.θ0_up[ctx.free_idx]

function time_calls(f::Function, nrep::Int)
    # warm-up (JIT) -- not timed
    f()
    ts = Vector{Float64}(undef, nrep)
    for i in 1:nrep
        t0 = time_ns()
        f()
        ts[i] = (time_ns() - t0) / 1e9
    end
    return (mean = mean(ts), median = median(ts), min = minimum(ts), max = maximum(ts), ts = ts)
end

const NREP = 15

println("\n" * "="^100)
println("PART B.4: end-to-end full-inner-KNITRO-solve timing, dense vs suffix-lookup vs interval-lookup")
println("="^100)

results = NamedTuple[]
for L in (10, 20, 50)
    println("\n--- L=$L ---")
    aug_dense = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
    aug_int   = build_cm_augmented_obj_interval(ctx, CS; L = L, contrasts = :anchored)
    aug_suffix = merge(aug_dense, (bins = aug_int.bins,))
    ctx_dense = merge(ctx, (obj = aug_dense.obj_cm,))

    # ---- (i) dense baseline: real production live KNITRO solve via evaluate_fullA ----
    # NOTE: `evaluate_fullA`'s returned `inner_iters` field is `CS.INNER_ITERS_TOTAL[]`, a
    # CUMULATIVE session-wide counter (see oracle.jl) -- NOT a per-solve count. Take an
    # explicit before/after delta to get the true per-solve iteration count (an earlier version
    # of this script printed the raw cumulative value directly, which grows across every solve in
    # this whole script run and is not comparable to the lookup path's per-solve `n_fg_calls`;
    # caught before being reported).
    iters0 = CS.INNER_ITERS_TOTAL[]
    r_dense_once = evaluate_fullA(x_free_calib, ctx_dense; use_cache = false, warm = false)
    n_iters_dense = CS.INNER_ITERS_TOTAL[] - iters0
    tdense = time_calls(() -> evaluate_fullA(x_free_calib, ctx_dense; use_cache = false, warm = false), NREP)

    # ---- (ii) suffix-sum lookup, live KNITRO ----
    r_suf_once = evaluate_fullA_cmlookup(x_free_calib, ctx, aug_suffix; method = :suffix, nthreads_use = 1, warm = false)
    tsuffix = time_calls(() -> evaluate_fullA_cmlookup(x_free_calib, ctx, aug_suffix; method = :suffix, nthreads_use = 1, warm = false), NREP)

    # ---- (iii) interval lookup, live KNITRO ----
    r_int_once = evaluate_fullA_cmlookup(x_free_calib, ctx, aug_int; method = :interval, nthreads_use = 1, warm = false)
    tinterval = time_calls(() -> evaluate_fullA_cmlookup(x_free_calib, ctx, aug_int; method = :interval, nthreads_use = 1, warm = false), NREP)

    @printf("  n_var=%3d  n_fg_calls: dense(n_iters)=%s suffix=%d interval=%d\n",
            aug_dense.obj_cm.outer_constr_index, string(n_iters_dense), r_suf_once.n_fg_calls, r_int_once.n_fg_calls)
    @printf("  FULL INNER SOLVE (median of %d, cold each time):\n", NREP)
    @printf("    dense    : median=%.5fs  mean=%.5fs  min=%.5fs  max=%.5fs\n", tdense.median, tdense.mean, tdense.min, tdense.max)
    @printf("    suffix   : median=%.5fs  mean=%.5fs  min=%.5fs  max=%.5fs   speedup(median vs dense)=%.3fx\n", tsuffix.median, tsuffix.mean, tsuffix.min, tsuffix.max, tdense.median / tsuffix.median)
    @printf("    interval : median=%.5fs  mean=%.5fs  min=%.5fs  max=%.5fs   speedup(median vs dense)=%.3fx\n", tinterval.median, tinterval.mean, tinterval.min, tinterval.max, tdense.median / tinterval.median)

    push!(results, (L = L, n_iters_dense = n_iters_dense, n_fg_suffix = r_suf_once.n_fg_calls, n_fg_interval = r_int_once.n_fg_calls,
                     dense_median = tdense.median, suffix_median = tsuffix.median, interval_median = tinterval.median,
                     speedup_suffix = tdense.median / tsuffix.median, speedup_interval = tdense.median / tinterval.median))
end

println("\n" * "="^100)
println("SUMMARY TABLE (full inner KNITRO solve, median of $NREP cold repeats)")
println("="^100)
@printf("%4s  %10s  %10s  %10s  %10s  %10s  %10s\n", "L", "dense(s)", "suffix(s)", "interval(s)", "spdup_suf", "spdup_int", "n_iters")
for r in results
    @printf("%4d  %10.5f  %10.5f  %10.5f  %10.3f  %10.3f  %10s\n",
            r.L, r.dense_median, r.suffix_median, r.interval_median, r.speedup_suffix, r.speedup_interval, string(r.n_iters_dense))
end

# ================================================================================================
# Secondary, lower-noise data point: isolated FG-kernel-only timing (bypassing KNITRO's own
# overhead -- solver setup, KN_new/KN_free, internal linear algebra -- to isolate JUST the
# objective/gradient evaluation cost), evaluated a REALISTIC number of times (= the actual
# n_fg_calls the live solve used above) at a battery of x points captured from... in the absence
# of trajectory capture, repeated at the converged optimum + jittered neighbors (same battery
# construction as c12i_validate_lookup_fg.jl) so the timed calls are not literally identical
# calls (which could let the CPU cache misleadingly warm/branch-predict in a way a real solve
# would not).
# ================================================================================================
println("\n" * "="^100)
println("SECONDARY: isolated FG-kernel-only timing (dense obj(x,g) vs lookup(x,g)), N = actual n_fg_calls")
println("="^100)

using Random
function fg_battery(n_var, n; seed = 99)
    rng = MersenneTwister(seed)
    return [0.01 .* randn(rng, n_var) for _ in 1:n]
end

for (ri, L) in enumerate((10, 20, 50))
    aug_dense = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
    aug_int   = build_cm_augmented_obj_interval(ctx, CS; L = L, contrasts = :anchored)
    aug_suffix = merge(aug_dense, (bins = aug_int.bins,))

    θ_full = CS.reconstruct_full(x_free_calib, ctx.m)
    W = size(ctx.U, 1)

    # dense obj: materialize H once
    aug_dense.obj_cm.moments!(@view(aug_dense.obj_cm.H[:, 1]), CS.select_G_from_H(aug_dense.obj_cm, aug_dense.obj_cm.H), θ_full, ctx.U, aug_dense.obj_cm)
    aug_dense.obj_cm.H[:, 2] .= 1.0
    aug_int.obj_cm.moments!(@view(aug_int.obj_cm.H[:, 1]), CS.select_G_from_H(aug_int.obj_cm, aug_int.obj_cm.H), θ_full, ctx.U, aug_int.obj_cm)
    aug_int.obj_cm.H[:, 2] .= 1.0

    st_suffix = CMLookupState(aug_dense.obj_cm, aug_dense.ncore, aug_dense.ncm, L, aug_dense.origins, aug_dense.refIndex1, aug_int.bins, nothing; method = :suffix, nthreads_use = 1)
    st_interval = CMLookupState(aug_int.obj_cm, aug_int.ncore, aug_int.ncm, L, aug_int.origins, aug_int.refIndex1, aug_int.bins, nothing; method = :interval, nthreads_use = 1)

    n_var = aug_dense.obj_cm.outer_constr_index
    n_calls = results[ri].n_fg_interval   # actual live-solve FG-call count from the main loop above
    npts = max(n_calls, 10)
    battery = fg_battery(n_var, npts)
    g = zeros(n_var)

    function run_dense_battery()
        for x in battery
            aug_dense.obj_cm(x, g)
        end
    end
    function run_suffix_battery()
        for x in battery
            st_suffix(x, g)
        end
    end
    function run_interval_battery()
        for x in battery
            st_interval(x, g)
        end
    end

    td = time_calls(run_dense_battery, NREP)
    ts = time_calls(run_suffix_battery, NREP)
    ti = time_calls(run_interval_battery, NREP)
    @printf("  L=%2d  npts=%3d  dense=%.5fs  suffix=%.5fs (%.3fx)  interval=%.5fs (%.3fx)   [per-call: dense=%.2fus suffix=%.2fus interval=%.2fus]\n",
            L, npts, td.median, ts.median, td.median/ts.median, ti.median, td.median/ti.median,
            1e6*td.median/npts, 1e6*ts.median/npts, 1e6*ti.median/npts)
end

# ================================================================================================
# Part B.2: threaded weighted-histogram benchmark, W=8000, D=4, at L in {10,20,50} (nbins=L+1),
# nthreads in {1,4,8,16} (host has Sys.CPU_THREADS reported above; JULIA_NUM_THREADS caps what's
# actually usable -- report both).
# ================================================================================================
println("\n" * "="^100)
println("PART B.2: threaded weighted-histogram builder benchmark")
println("="^100)
println("Sys.CPU_THREADS=$(Sys.CPU_THREADS)  Threads.nthreads()=$(Threads.nthreads()) -- thread counts above Threads.nthreads() cannot actually run in parallel within this process.")

Random.seed!(55)
for L in (10, 20, 50)
    aug_int = build_cm_augmented_obj_interval(ctx, CS; L = L, contrasts = :anchored)
    bins = aug_int.bins
    W = size(bins, 1)
    weights = rand(W) .+ 0.1   # positive, arg1-like weights
    nbins = L + 1
    println("\n  L=$L (nbins=$nbins, W=$W):")
    for nt in (1, 2, 4, 8, 16)
        if nt > Threads.nthreads()
            println("    nthreads_use=$nt > Threads.nthreads()=$(Threads.nthreads()) -- SKIPPED (cannot actually parallelize)")
            continue
        end
        f = () -> build_weighted_histogram(bins, weights, D, nbins; nthreads_use = nt)
        t = time_calls(f, 20)
        @printf("    nthreads_use=%2d  median=%.2fus  mean=%.2fus\n", nt, 1e6*t.median, 1e6*t.mean)
    end
end

println("\nDone. Run at: $(now())")
