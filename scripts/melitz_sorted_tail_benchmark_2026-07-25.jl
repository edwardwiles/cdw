# 2026-07-25 sorted-tail moment-construction optimization session, Phases 0/12/13 benchmark.
# See docs/melitz_sorted_tail_optimization_2026-07-25.md for the full report.
#
# Isolated, real-D20/W=80,000 moment-construction wall-clock comparison: dense reference
# vs sorted-tail serial vs sorted-tail parallel, post-JIT (each backend warmed up once
# before timing), at several representative outer points (calibrated reference, a
# near-boundary finite point, points with engineered high/low cutoffs). Does NOT run a
# full outer KNITRO campaign (Phase 12's "short matched campaign" sub-goal) -- see the
# report's own honest scope-accounting section for why.
#
# Usage: julia --project=. -t 16 scripts/melitz_sorted_tail_benchmark_2026-07-25.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS
using Random

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
const W = parse(Int, get(ENV, "MELITZ_BENCH_W", "80000"))

function load_calibration()
    lambdaData = readdlm(joinpath(REAL_DIR, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(REAL_DIR, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(REAL_DIR, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(REAL_DIR, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    return calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate,
        focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
end

function timed(f, label; nrep=3)
    f() # warm up / JIT
    times = Float64[]
    for _ in 1:nrep
        t0 = time_ns()
        f()
        push!(times, (time_ns() - t0) / 1e9)
    end
    @printf("  %-28s  min=%.4fs  median=%.4fs  max=%.4fs  (n=%d)\n", label, minimum(times), sort(times)[div(nrep+1,2)], maximum(times), nrep)
    return times
end

function build_point(calib, z, g_shift::Float64)
    D = calib.D
    p, eq, cf, ctx = melitz_calibration_outer_ctx(calib; outer_parameterization=:logf)
    theta = melitz_reduce_theta(p, ctx)
    theta[1] += g_shift
    A, f, gamma_prime_j, f_jj = melitz_expand_theta(theta, ctx)
    primitives = MelitzPrimitives(D, ctx.sigma, ctx.theta_star, ctx.target_country, ctx.tau, ctx.w, A, f, gamma_prime_j)
    cutoff = melitz_baseline_cutoff(A, f, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma)
    eq2 = MelitzEquilibrium(ctx.expenditure, ones(D), cutoff, ctx.X_data)
    expenditure_prime = ctx.w_prime * ctx.L[ctx.target_country]
    cf2 = MelitzCounterfactual(ctx.target_country, ctx.w_prime, expenditure_prime, 1.0, expenditure_prime)
    return primitives, eq2, cf2
end

function main()
    println("Melitz sorted-tail moment construction benchmark -- 2026-07-25")
    println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", BLAS.get_num_threads())
    BLAS.set_num_threads(1)
    calib = load_calibration()
    D = calib.D
    layout = MelitzMomentLayout(D)
    println("D=$D  W=$W")

    z = pareto_draws(W, D, calib.theta_star; seed=1)
    t_sort0 = time_ns()
    sctx = build_melitz_sorted_tail_context(z, calib.sigma; theta_star=calib.theta_star)
    t_sort = (time_ns() - t_sort0) / 1e9
    @printf("sorted-tail context construction: %.4fs (one-time, amortized across every callback)\n", t_sort)

    points = [
        ("calibrated reference (g_shift=0)", 0.0),
        ("near-boundary (g_shift=-0.0027)", -0.0027),
        ("high-cutoff direction (g_shift=+0.02)", 0.02),
        ("low-cutoff direction (g_shift=-0.02)", -0.02),
    ]

    K = zeros(W)
    G = zeros(W, layout.num_moments)

    for (label, shift) in points
        println("="^90)
        println("POINT: ", label)
        p, eq, cf = build_point(calib, z, shift)
        diag = melitz_sorted_tail_diagnostics(eq, sctx, layout)
        @printf("  active_fraction: min=%.4f  median=%.4f  max=%.4f\n",
            minimum(diag.active_fraction), sort(vec(diag.active_fraction))[div(end,2)], maximum(diag.active_fraction))

        timed(() -> melitz_moments!(K, G, p, eq, cf, z, layout), "dense_reference")
        timed(() -> melitz_moments_sorted_tail!(K, G, p, eq, cf, sctx, layout), "sorted_tail_serial")
        timed(() -> melitz_moments_sorted_tail_parallel!(K, G, p, eq, cf, sctx, layout), "sorted_tail_parallel(t=$(Threads.nthreads()))")
    end
    flush(stdout)
end

main()
