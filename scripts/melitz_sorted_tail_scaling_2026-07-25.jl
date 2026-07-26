# 2026-07-25 sorted-tail session, Phase 13 scaling follow-up (deferred sub-items).
# Deliberately narrow: ONE representative outer point (the calibrated reference), so each
# process launch is cheap enough to repeat across a thread-count sweep (Julia's own thread
# count is fixed at process launch -- this script is meant to be invoked once per `-t N`)
# and a W sweep (via the MELITZ_BENCH_W env var). See
# docs/melitz_sorted_tail_optimization_2026-07-25.md Section G for the full results table.
#
# Usage:
#   MELITZ_BENCH_W=80000 julia --project=. -t N scripts/melitz_sorted_tail_scaling_2026-07-25.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS

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

function timed(f; nrep=5)
    f()
    times = Float64[]
    for _ in 1:nrep
        t0 = time_ns()
        f()
        push!(times, (time_ns() - t0) / 1e9)
    end
    sort!(times)
    return (min=times[1], median=times[div(nrep + 1, 2)], max=times[end])
end

function main()
    BLAS.set_num_threads(1)
    calib = load_calibration()
    D = calib.D
    layout = MelitzMomentLayout(D)
    z = pareto_draws(W, D, calib.theta_star; seed=1)

    t_sort0 = time_ns()
    sctx = build_melitz_sorted_tail_context(z, calib.sigma; theta_star=calib.theta_star)
    t_sort = (time_ns() - t_sort0) / 1e9

    p, eq, cf, ctx = melitz_calibration_outer_ctx(calib; outer_parameterization=:logf)
    theta = melitz_reduce_theta(p, ctx)
    A, f, gamma_prime_j, f_jj = melitz_expand_theta(theta, ctx)
    primitives = MelitzPrimitives(D, ctx.sigma, ctx.theta_star, ctx.target_country, ctx.tau, ctx.w, A, f, gamma_prime_j)
    cutoff = melitz_baseline_cutoff(A, f, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma)
    eq2 = MelitzEquilibrium(ctx.expenditure, ones(D), cutoff, ctx.X_data)
    expenditure_prime = ctx.w_prime * ctx.L[ctx.target_country]
    cf2 = MelitzCounterfactual(ctx.target_country, ctx.w_prime, expenditure_prime, 1.0, expenditure_prime)

    K = zeros(W); G = zeros(W, layout.num_moments)
    r_dense = timed(() -> melitz_moments!(K, G, primitives, eq2, cf2, z, layout))
    r_serial = timed(() -> melitz_moments_sorted_tail!(K, G, primitives, eq2, cf2, sctx, layout))
    r_par = timed(() -> melitz_moments_sorted_tail_parallel!(K, G, primitives, eq2, cf2, sctx, layout))

    @printf("RESULT threads=%d W=%d sort_ctx=%.4fs dense_med=%.4fs serial_med=%.4fs par_med=%.4fs speedup_serial=%.2fx speedup_par=%.2fx\n",
        Threads.nthreads(), W, t_sort, r_dense.median, r_serial.median, r_par.median,
        r_dense.median / r_serial.median, r_dense.median / r_par.median)
    flush(stdout)
end

main()
