# Continuation session (2026-07-24), Phase 3 QMC-seed sensitivity check at the FIXED
# (theta_star=:estimate) real D=20 calibration.
#
# Usage: julia --project=. scripts/melitz_real_d20_seed_sensitivity_2026-07-24.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")

function main()
    lambdaData = readdlm(joinpath(REAL_DIR, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(REAL_DIR, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(REAL_DIR, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(REAL_DIR, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)

    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate,
        focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)

    inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")
    BLAS.set_num_threads(16)

    println("W=80000 seed sweep:")
    for seed in 1:8
        obj, theta0 = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=seed, inner_loop_opt=inner_opt)
        diag = melitz_conditioning_diagnostics(
            MelitzPrimitives(calib.D, calib.sigma, calib.theta_star, calib.target_country, calib.tau, calib.w, calib.A, calib.f, calib.gamma_prime_target),
            MelitzEquilibrium(calib.E, ones(calib.D), calib.q, calib.X),
            MelitzCounterfactual(calib.target_country, calib.w_prime, calib.w_prime * calib.L[calib.target_country], 1.0, calib.w_prime * calib.L[calib.target_country]),
            80_000; seed=seed)
        t0 = time()
        lfd = melitz_recover_lfd(obj, theta0)
        wall = time() - t0
        @printf("  seed=%d  rank=%d/401  cond=%.3e  wall=%.2fs  nStatus=%-4d  Delta=%.4e  lfd_ok=%s\n",
            seed, diag.moment_matrix_rank, diag.condition_number, wall, lfd.nStatus, lfd.Delta, lfd.lfd_ok)
        flush(stdout)
    end

    println("\nW=150000 for the seeds that failed at W=80000:")
    for seed in (2,)
        for W in (120_000, 150_000)
            obj, theta0 = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=seed, inner_loop_opt=inner_opt)
            diag = melitz_conditioning_diagnostics(
                MelitzPrimitives(calib.D, calib.sigma, calib.theta_star, calib.target_country, calib.tau, calib.w, calib.A, calib.f, calib.gamma_prime_target),
                MelitzEquilibrium(calib.E, ones(calib.D), calib.q, calib.X),
                MelitzCounterfactual(calib.target_country, calib.w_prime, calib.w_prime * calib.L[calib.target_country], 1.0, calib.w_prime * calib.L[calib.target_country]),
                W; seed=seed)
            t0 = time()
            lfd = melitz_recover_lfd(obj, theta0)
            wall = time() - t0
            @printf("  seed=%d W=%d  rank=%d/401  cond=%.3e  wall=%.2fs  nStatus=%-4d  Delta=%.4e  lfd_ok=%s\n",
                seed, W, diag.moment_matrix_rank, diag.condition_number, wall, lfd.nStatus, lfd.Delta, lfd.lfd_ok)
            flush(stdout)
        end
    end
    BLAS.set_num_threads(1)
    println("\nDONE.")
end

main()
