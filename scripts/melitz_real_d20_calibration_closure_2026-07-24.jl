# Continuation session (2026-07-24), Phases 1-3: real D=20 calibration closure +
# W=80,000 fixed-point revalidation, under the FIXED (theta_star=:estimate, authoritative
# direct wage solve, explicit share/tau policies) calibration pipeline.
#
# Usage: julia --project=. scripts/melitz_real_d20_calibration_closure_2026-07-24.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")

function main()
    println("="^100)
    println("PHASE 1: real D=20 data-only calibration closure")
    println("="^100)

    lambdaData = readdlm(joinpath(REAL_DIR, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(REAL_DIR, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(REAL_DIR, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(REAL_DIR, "countries.csv"), ',', String))
    D = length(countries)
    focal = findfirst(==("fra"), countries)
    @printf("D=%d, focal_country=%s (index %d)\n", D, countries[focal], focal)

    # --- Addendum Section 1: raw share diagnostics BEFORE any policy ---
    raw_diag = melitz_share_diagnostics(Matrix{Float64}(lambdaData))
    println("\n-- Raw share diagnostics (pi.csv, BEFORE any policy) --")
    @printf("  min cell = %.6e, max cell = %.6e\n", raw_diag.min_cell, raw_diag.max_cell)
    @printf("  max |column sum - 1| = %.3e\n", raw_diag.max_abs_column_sum_deviation)
    @printf("  n_zero_or_negative = %d, n_nonfinite = %d\n", raw_diag.n_zero_or_negative, raw_diag.n_nonfinite)

    println("\n-- Raw tau diagonal (BEFORE any policy) --")
    for o in 1:D
        if !isapprox(tauData[o, o], 1.0; atol=0)
            @printf("  tau[%s,%s] = %.10f  (raw, non-bit-exact)\n", countries[o], countries[o], tauData[o, o])
        end
    end

    # --- Construct the FROZEN MelitzObservedData (default policies: as_supplied / normalize_to_one) ---
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    println("\n-- MelitzObservedData constructed --")
    println("  share_policy = ", observed.share_policy)
    println("  tau_diagonal_policy = ", observed.tau_diagonal_policy)
    println("  lambda === raw (as_supplied)? ", observed.lambda == Matrix{Float64}(lambdaData))

    # --- Phase 1.1: theta estimated from the SAME frozen observed object ---
    theta_hat = melitz_estimate_theta_hat(observed.lambda, observed.tau)
    @printf("\ntheta_hat (estimated from FROZEN observed.lambda/observed.tau) = %.6f\n", theta_hat)

    # For comparison: the OLD (buggy) ordering -- theta estimated on RAW pre-policy tauData.
    theta_hat_raw_order = melitz_estimate_theta_hat(Matrix{Float64}(lambdaData), Matrix{Float64}(tauData))
    @printf("theta_hat (OLD ordering: estimated on RAW pre-policy tauData) = %.6f\n", theta_hat_raw_order)
    @printf("difference = %.3e\n", theta_hat - theta_hat_raw_order)

    # --- Phase 2: authoritative direct wage solve vs. damped-Jacobi vs. Perron ---
    println("\n-- Phase 2: wage solve comparison --")
    wc = calibrate_melitz_wages(observed.lambda, observed.L; tol=1e-8, max_iter=200_000)
    @printf("  direct-solve market-clearing residual = %.3e\n", wc.market_clearing_residual)
    @printf("  damped-Jacobi residual                = %.3e (in %d iterations)\n", wc.damped_residual, wc.damped_iterations)
    @printf("  Perron eigenvector residual            = %.3e (eigenvalue=%.10f)\n", wc.perron_residual, wc.perron_eigenvalue)
    @printf("  max|w_direct - w_damped| / max|w|      = %.3e\n", wc.damped_vs_direct_residual)
    @printf("  wage range = [%.4f, %.4f]\n", wc.wage_range...)

    # --- Phase 1.1/1.3: calibrate with theta_star=:estimate ---
    println("\n-- calibrate_melitz_pareto(theta_star=:estimate) --")
    calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate,
        focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
    @printf("  theta_star used = %.6f  (grav0.theta_hat=%.6f, gap=%.3e, compatible=%s)\n",
        calib.theta_star, calib.gravity.theta_hat, calib.gravity.theta_gap, calib.gravity.compatible)
    @printf("  rhs_A = %.10f, rhs_f = %.10f, |rhs_A - rhs_f| = %.3e\n",
        calib.cutoff_calibration.gravity_rhs_A, calib.cutoff_calibration.gravity_rhs_f,
        abs(calib.cutoff_calibration.gravity_rhs_A - calib.cutoff_calibration.gravity_rhs_f))
    @printf("  gravity_residual_A = %.3e\n", calib.equilibrium_check.gravity_residual_A)
    @printf("  gravity_residual_f = %.3e\n", calib.equilibrium_check.gravity_residual_f)
    @printf("  max|share residual| = %.3e\n", maximum(abs.(calib.equilibrium_check.residual_shares)))
    @printf("  max|cutoff reconstruction residual| = %.3e\n", maximum(abs.(calib.equilibrium_check.residual_cutoff_reconstruction)))
    @printf("  min_support = %.6f, min_export_minus_domestic = %.6f\n", calib.equilibrium_check.min_support, calib.equilibrium_check.min_export_minus_domestic)
    @printf("  f_E range = [%.4f, %.4f]\n", extrema(calib.equilibrium_check.f_E)...)
    @printf("  gamma_prime_target (focal autarky price power) = %.6f\n", calib.gamma_prime_target)
    @printf("  free_entry_link_residual = %.3e\n", calib.free_entry_link_residual)
    @printf("  A range = [%.4e, %.4e], f range = [%.4e, %.4e]\n", extrema(calib.A)..., extrema(calib.f)...)

    p_calib = MelitzPrimitives(calib.D, calib.sigma, calib.theta_star, calib.target_country,
        calib.tau, calib.w, calib.A, calib.f, calib.gamma_prime_target)
    eq_calib = MelitzEquilibrium(calib.E, ones(calib.D), calib.q, calib.X)
    cf_calib = MelitzCounterfactual(calib.target_country, calib.w_prime,
        calib.w_prime * calib.L[calib.target_country], 1.0, calib.w_prime * calib.L[calib.target_country])
    GT_model = melitz_gains_from_trade(p_calib, cf_calib)
    _, GT_ACR = acr_gains_from_trade(p_calib, eq_calib)
    @printf("  GT_model = %.10f, GT_ACR = %.10f, |diff| = %.3e\n", GT_model, GT_ACR, abs(GT_model - GT_ACR))

    # --- Phase 1.4: outer-coordinate roundtrip ---
    println("\n-- Phase 1.4: outer-coordinate roundtrip check --")
    rt = melitz_calibration_roundtrip_check(calib)
    @printf("  max_abs_A_diff = %.3e, max_rel_A_diff = %.3e\n", rt.max_abs_A_diff, rt.max_rel_A_diff)
    @printf("  max_abs_f_diff = %.3e, max_rel_f_diff = %.3e\n", rt.max_abs_f_diff, rt.max_rel_f_diff)
    @printf("  gamma_prime_diff = %.3e, f_jj_diff = %.3e\n", rt.gamma_prime_diff, rt.f_jj_diff)
    @printf("  max_abs_cutoff_diff = %.3e, max_abs_share_diff = %.3e\n", rt.max_abs_cutoff_diff, rt.max_abs_share_diff)
    @printf("  focal_link_residual_reexpanded = %.3e\n", rt.focal_link_residual_reexpanded)

    println("\n" * "="^100)
    println("PHASE 3: real D=20/W=80,000 inner-solve revalidation (fixed calibration)")
    println("="^100)

    BLAS.set_num_threads(16)
    inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")
    for (W, seed) in [(80_000, 1), (80_000, 2)]
        obj, theta0 = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=seed, inner_loop_opt=inner_opt)
        ctx = obj.γ
        diag = melitz_conditioning_diagnostics(p_calib, eq_calib,
            MelitzCounterfactual(calib.target_country, calib.w_prime, calib.w_prime * calib.L[calib.target_country], 1.0, calib.w_prime * calib.L[calib.target_country]),
            W; seed=seed)
        @printf("\n  [seed=%d] moment matrix rank = %d/%d, condition number = %.4e, min active draws = %d (worst cell %s)\n",
            seed, diag.moment_matrix_rank, diag.moment_matrix_size[2], diag.condition_number, diag.min_active_count, diag.worst_cell)

        t0 = time()
        lfd = melitz_recover_lfd(obj, theta0)
        wall = time() - t0
        @printf("  [seed=%d] wall=%.2fs  nStatus=%d  Delta(theta*)=%.7e  lfd_ok=%s\n",
            seed, wall, lfd.nStatus, lfd.Delta, lfd.lfd_ok)
        @printf("  [seed=%d] kkt_opt_error=%.3e  kkt_feas_error=%.3e  max_weighted_moment_residual=%.3e\n",
            seed, lfd.kkt_opt_error, lfd.kkt_feas_error, lfd.maximum_weighted_moment_residual)
        @printf("  [seed=%d] primal_divergence=%.7e  dual_divergence=%.7e  primal_dual_gap=%.3e\n",
            seed, lfd.primal_divergence, lfd.dual_divergence, lfd.primal_dual_gap)

        check = check_profiled_melitz_equilibrium(p_calib, eq_calib, cf_calib, obj.U, lfd.weights)
        @printf("  [seed=%d] residual_autarky_cutoff=%.3e  N_prime_diff_rel=%.3e\n",
            seed, check.residual_autarky_cutoff, check.N_prime_diff_rel)
    end
    BLAS.set_num_threads(1)

    println("\nDONE.")
    flush(stdout)
    return calib
end

main()
