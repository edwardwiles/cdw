# Continuation session (2026-07-24), Phase 4: real D=20 kernel profiling at the FIXED
# (theta_star=:estimate) calibrated reference point. Covers:
#   4.1 inner successful solve (BLAS thread sweep, Julia coordinate threading inactive)
#   4.2 fast rejected solve (a Delta>1 point's screening/rejection timing)
#   4.3 full outer gradient (:B_direct_argument_serial/_parallel, Julia thread sweep, BLAS=1)
#
# Usage: julia --project=. -t 20 scripts/melitz_real_d20_kernel_profile_2026-07-24.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
const INNER_OPT = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")

function load_calibration()
    lambdaData = readdlm(joinpath(REAL_DIR, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(REAL_DIR, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(REAL_DIR, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(REAL_DIR, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate,
        focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
    return calib
end

function phase41_inner_solve(calib; W=80_000, seed=1, blas_threads=(1, 4, 8, 16, 20))
    println("="^100); println("PHASE 4.1: inner successful solve, BLAS thread sweep (Julia coord threading inactive)")
    println("="^100)
    for nb in blas_threads
        BLAS.set_num_threads(nb)
        obj, theta0 = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=seed, inner_loop_opt=INNER_OPT)
        obj.use_cached_x = false; obj.x .= NaN
        CS = CounterfactualSensitivity
        t0 = time()
        bytes = @allocated begin
            global _lfd_ = melitz_recover_lfd(obj, theta0)
        end
        wall = time() - t0
        lfd = _lfd_
        @printf("  BLAS=%3d  wall=%7.3fs  nStatus=%d  Delta=%.4e  lfd_ok=%s  bytes=%d\n",
            nb, wall, lfd.nStatus, lfd.Delta, lfd.lfd_ok, bytes)
        flush(stdout)
    end
    BLAS.set_num_threads(1)
end

function phase42_rejected_solve(calib; W=80_000, seed=1, inner_opt_capped::String=INNER_OPT)
    println("\n" * "="^100); println("PHASE 4.2: rejection classification (cheap deterministic screen vs. cold full inner solve)")
    println("="^100)
    obj, theta0 = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=seed, inner_loop_opt=INNER_OPT)
    ctx = obj.γ

    # (a) CHEAP deterministic cutoff/export-selection screen (Section 1.3, NO Monte Carlo,
    # NO KNITRO) -- a point far enough to violate export-selection/support is rejected in
    # microseconds. This is the genuinely "fast rejection" path for points that are cutoff-
    # infeasible; it never reaches the inner CC solve at all.
    theta_cutoff_infeasible = copy(theta0); theta_cutoff_infeasible[1] += 5.0
    t0 = time()
    state_bad = melitz_outer_state(theta_cutoff_infeasible, ctx)
    wall_cheap = time() - t0
    @printf("  (a) cheap cutoff screen at a wildly displaced point: wall=%.6fs  min_slack=%.4f  feasible=%s\n",
        wall_cheap, state_bad.min_slack, state_bad.feasible)

    # (b) A MODEST displacement that stays cutoff-FEASIBLE but genuinely raises the
    # divergence budget requirement -- classify whether the cold full inner solve (no prior
    # warm dual/threshold reference exists for a never-before-seen point, so the FC/GA
    # stored-dual/live-threshold screens inside melitz_build_finite_delta_callbacks cannot
    # apply here) converges cleanly to a large Delta (BudgetInfeasible) or genuinely fails
    # numerically (NumericalFailure). Uses the SAME W/seed, a maxit-capped inner option file
    # (2000 vs. the production 10000) so a genuinely non-convergent point fails in bounded
    # time for this diagnostic rather than grinding the full production budget.
    theta_moderate = copy(theta0); theta_moderate[1] += 0.5
    state_mod = melitz_outer_state(theta_moderate, ctx)
    @printf("  (b) moderately displaced point: min_slack=%.4f  feasible=%s\n", state_mod.min_slack, state_mod.feasible)
    obj_capped, _ = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=seed, inner_loop_opt=inner_opt_capped)
    obj_capped.use_cached_x = false; obj_capped.x .= NaN
    BLAS.set_num_threads(16)
    t0 = time()
    lfd_mod = melitz_recover_lfd(obj_capped, theta_moderate)
    wall_mod = time() - t0
    classification = lfd_mod.nStatus == 0 && lfd_mod.Delta > 1.0 ? "BudgetInfeasible (converged, Delta>1)" :
                      lfd_mod.nStatus == 0 ? "converged, Delta<=1 (would NOT be rejected)" :
                      "NumericalFailure (nStatus=$(lfd_mod.nStatus))"
    @printf("  (b) cold full inner solve (maxit=2000 cap): wall=%7.3fs  nStatus=%d  Delta=%.4e  => %s\n",
        wall_mod, lfd_mod.nStatus, lfd_mod.Delta, classification)
    BLAS.set_num_threads(1)
    return theta_moderate
end

function phase43_outer_gradient(calib; W=80_000, seed=1, jthreads_available=Threads.nthreads())
    println("\n" * "="^100); println("PHASE 4.3: full outer gradient, Julia thread sweep (BLAS=1)")
    println("="^100)
    @printf("  Threads.nthreads() at process start = %d (thread sweep below is INFORMATIONAL only --\n", jthreads_available)
    println("   actually changing Julia's thread pool requires a process restart with -t N; this")
    println("   script reports the compiled-in thread count and profiles serial vs. parallel dispatch.")
    BLAS.set_num_threads(1)
    obj, theta0 = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=seed, inner_loop_opt=INNER_OPT)
    ctx = obj.γ
    obj.use_cached_x = false; obj.x .= NaN
    lfd = melitz_recover_lfd(obj, theta0)
    @printf("  reference inner solve: nStatus=%d  Delta=%.4e\n", lfd.nStatus, lfd.Delta)
    x = lfd.dual_x
    n = length(theta0)
    @printf("  n_theta (free outer coordinates) = %d\n", n)

    grad_serial! = make_melitz_gradient_delta_direct_serial(1e-4)
    grad_parallel! = make_melitz_gradient_delta_direct_parallel(1e-4)
    g_serial = zeros(n); g_parallel = zeros(n)

    t0 = time()
    bytes_s = @allocated grad_serial!(g_serial, theta0, ctx, obj, x)
    wall_serial = time() - t0
    @printf("  serial:   wall=%7.3fs  bytes=%d  per-coord=%.4fs\n", wall_serial, bytes_s, wall_serial / n)

    t0 = time()
    bytes_p = @allocated grad_parallel!(g_parallel, theta0, ctx, obj, x)
    wall_parallel = time() - t0
    @printf("  parallel: wall=%7.3fs  bytes=%d  per-coord=%.4fs  (Threads.nthreads()=%d)\n",
        wall_parallel, bytes_p, wall_parallel / n, Threads.nthreads())

    agree = maximum(abs.(g_serial .- g_parallel)) / max(1.0, maximum(abs.(g_serial)))
    @printf("  serial/parallel max relative disagreement = %.3e\n", agree)

    n_grads_in_600s_serial = 600.0 / wall_serial
    n_grads_in_600s_parallel = 600.0 / wall_parallel
    @printf("  => at this wall time, a 600s budget affords ~%.1f serial gradients or ~%.1f parallel gradients\n",
        n_grads_in_600s_serial, n_grads_in_600s_parallel)
    return g_serial, g_parallel, x, theta0, ctx, obj
end

function main()
    calib = load_calibration()
    mode = get(ENV, "MELITZ_PHASE4_MODE", "all")
    if mode in ("all", "41_42")
        phase41_inner_solve(calib)
        inner_opt_capped = get(ENV, "MELITZ_INNER_OPT_CAPPED", INNER_OPT)
        phase42_rejected_solve(calib; inner_opt_capped=inner_opt_capped)
    end
    if mode in ("all", "43")
        phase43_outer_gradient(calib)
    end
    println("\nDONE.")
end

main()
