# Continuation session (2026-07-23, "make the optimized architecture scalable in memory and
# D") Section 10: BLAS inner-solver scaling, redone correctly -- the prior session's own
# D=10 table (docs/melitz_optimization_report_2026-07-23_continuation2.md, Section 11) had
# its `threads=1` row inflated by first-call JIT compilation, disclosed but not re-run
# cleanly. This script: (1) warms the COMPLETE inner path once (any BLAS thread count) so
# every subsequently timed call is post-JIT, (2) benchmarks repeated trials in RANDOMIZED
# thread-count order (not monotonic 1,2,4,...), (3) reports median AND minimum warm time.
#
# D=20 was previously DELIBERATELY NOT attempted at production W (80,000) by this script --
# see this session's own memory audit (docs/melitz_optimization_report_2026-07-23_continuation3.md
# Section F): `PsiObjectiveBundleDelta`'s constructor (`cc_algo/PsiObjectiveBundle.jl` line
# ~425, `build_melitz_psi_bundle`'s own bundle type) unconditionally allocated a dense
# `jac_h::Array{Float64,3}` sized `(N, d+2, l)` REGARDLESS of whether the outer-gradient
# machinery that consumes it is ever invoked -- `2*W*D^4` elements at this problem's scale,
# ~51.5GB at D=20/W=20,000 and ~205.8GB at D=20/W=80,000 for a SINGLE object.
#
# FIXED (continuation4 session): `PsiObjectiveBundleDelta` now mirrors `PsiObjectiveBundleImplicit`'s
# own `needs_outer_moment_jacobian` escape hatch (`cc_algo/PsiObjectiveBundle.jl`). A call-graph
# audit confirmed jac_h is NEVER read on this bundle type's only actual use pattern in this repo
# (fixed-theta inner CC dual solves via `inner_loop_KNITRO`, which never passes a nonempty θ to the
# functor) -- so `needs_outer_moment_jacobian=false` is safe here and unblocks D=20/W=80,000.
#
# Usage: julia --project=. scripts/melitz_blas_scaling_clean.jl

using Printf, Random, Statistics
include(joinpath(@__DIR__, "melitz_finite_delta_campaign.jl"))

function bench_blas_clean(D, W; blas_threads=(1, 2, 4, 8, 16, 20), n_trials=5,
                           min_participation_prob=(D <= 4 ? 0.01 : 0.002))
    println("\n", "="^100)
    @printf("D=%d  W=%d  K=%d economic moments  (min_participation_prob=%.4f)\n", D, W, D^2 + 1,
        min_participation_prob)
    println("="^100)
    data = generate_fake_melitz_data(; D=D, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=W,
        min_participation_prob=min_participation_prob)
    inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")
    obj, theta0 = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt, needs_outer_moment_jacobian=false)
    CS = CounterfactualSensitivity

    # Warm the COMPLETE inner path once, post-JIT, before any timed trial.
    BLAS.set_num_threads(1)
    obj.use_cached_x = false
    obj.x .= NaN
    CS.inner_loop_internal(obj, theta0)
    println("  (warm-up call complete, JIT absorbed)")

    # Randomized trial order: n_trials repeats of every thread count, shuffled as one flat list.
    rng = MersenneTwister(31)
    trial_order = shuffle(rng, repeat(collect(blas_threads), n_trials))
    times_by_threads = Dict(nb => Float64[] for nb in blas_threads)
    for nb in trial_order
        BLAS.set_num_threads(nb)
        obj.use_cached_x = false
        obj.x .= NaN
        t0 = time()
        _, x, nStatus = CS.inner_loop_internal(obj, theta0)
        wall = time() - t0
        push!(times_by_threads[nb], wall)
    end
    BLAS.set_num_threads(1)

    println("  threads   median(s)   min(s)   max(s)   n")
    rows = NamedTuple[]
    for nb in blas_threads
        ts = times_by_threads[nb]
        @printf("  %6d   %8.4f   %6.4f   %6.4f   %d\n", nb, median(ts), minimum(ts), maximum(ts), length(ts))
        push!(rows, (D=D, W=W, blas_threads=nb, median=median(ts), min=minimum(ts), max=maximum(ts), n=length(ts)))
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    all_rows = NamedTuple[]
    append!(all_rows, bench_blas_clean(4, 20_000))
    append!(all_rows, bench_blas_clean(10, 20_000))
    append!(all_rows, bench_blas_clean(10, 80_000))
    append!(all_rows, bench_blas_clean(20, 20_000))
    append!(all_rows, bench_blas_clean(20, 80_000))
    println("\n", "="^100)
    println("SUMMARY (median warm wall time, randomized-order trials)")
    println("="^100)
    @printf("%-6s %-8s %-14s %-10s %-10s\n", "D", "W", "blas_threads", "median(s)", "min(s)")
    for r in all_rows
        @printf("%-6d %-8d %-14d %-10.4f %-10.4f\n", r.D, r.W, r.blas_threads, r.median, r.min)
    end
end
