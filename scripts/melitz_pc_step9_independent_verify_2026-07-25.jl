# 2026-07-25: independent verification of the predictor-corrector probe's own step-9
# accepted point (docs/melitz_real_d20_local_geometry_and_continuation_2026-07-25.md Section F)
# -- re-runs the IDENTICAL deterministic driver truncated to 9 steps (steps 1-9 do not
# depend on step 10, so this reproduces the same theta), then independently cold-verifies
# that exact theta a SECOND time and reports full feasibility diagnostics (gravity residuals,
# cutoff min_slack, LFD ok) via melitz_classify_outer_feasibility -- never trusting the
# driver's own internal accept/reject bookkeeping alone for a headline claim.
#
# Usage: julia --project=. -t 16 scripts/melitz_pc_step9_independent_verify_2026-07-25.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS, norm

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
const INNER_OPT = joinpath(dirname(@__DIR__), "melitz_inner_loop_options_capped_2026-07-24.opt")
const OUTER_OPT = joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt")
const OUTER_BUDGET_DELTA = 1.0
const DELTA_EVALUATION_CAP = 10.0
const CAMPAIGN_START_G = -0.497333
const W = 80_000
const KAPPA_FIXED_REFERENCE = 0.92939627   # docs/melitz_real_d20_outer_benchmark_2026-07-24.md Phase 6 incumbent

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

function main()
    println("="^100)
    println("Independent re-verification of predictor-corrector step 9")
    println("="^100)
    BLAS.set_num_threads(16)
    calib = load_calibration()
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ
    theta0 = copy(theta_calib); theta0[1] = CAMPAIGN_START_G

    radii = MelitzBlockTrustRadii(1e-4, 3e-6)
    result = melitz_predictor_corrector_continuation(ctx, obj_inner, theta0;
        delta=OUTER_BUDGET_DELTA, cap=DELTA_EVALUATION_CAP, radii=radii, dg_init=1e-4,
        n_steps=9, inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT,
        min_slack_floor=0.0, prediction_error_tol=0.5, expand_after=2)

    theta9 = result.final_theta
    @printf("\nReplayed to step 9: g=%.8f  Delta(driver)=%.8e  kappa(driver)=%.8f\n",
        theta9[1], result.final_Delta, result.final_kappa)

    # SECOND, fully independent cold re-verification of the exact same theta.
    r_indep = evaluate_melitz_delta(theta9, ctx, obj_inner; cold=true, store_G=false)
    cls = melitz_classify_outer_feasibility(r_indep, OUTER_BUDGET_DELTA)
    @printf("\nIndependent cold re-verification:\n")
    @printf("  Delta=%.8e  nStatus=%d  verified=%s\n", r_indep.Delta, r_indep.nStatus, r_indep.verified)
    @printf("  inner_verified=%s  inner_moment_feasible(lfd_ok)=%s  cutoff_feasible=%s  gravity_feasible=%s  budget_feasible=%s  outer_feasible=%s\n",
        cls.inner_verified, cls.inner_moment_feasible, cls.cutoff_feasible, cls.gravity_feasible,
        cls.budget_feasible, cls.outer_feasible)
    if r_indep.equilibrium_check !== nothing
        @printf("  gravity_residual_A=%.3e  gravity_residual_f=%.3e\n",
            r_indep.equilibrium_check.gravity_residual_A, r_indep.equilibrium_check.gravity_residual_f)
    end
    state = melitz_outer_state(theta9, ctx)
    @printf("  min_slack=%.6f\n", state.min_slack)

    kappa9 = exp(theta9[1])^(1/(ctx.sigma-1)) * (ctx.w_prime / ctx.w[ctx.target_country])
    @printf("\nkappa at step 9 = %.8f   vs. kappa_fixed_reference = %.8f   improvement = %.3e (%.5f%% relative)\n",
        kappa9, KAPPA_FIXED_REFERENCE, KAPPA_FIXED_REFERENCE - kappa9,
        100*(KAPPA_FIXED_REFERENCE - kappa9)/KAPPA_FIXED_REFERENCE)
    @printf("Delta=%.6e <= delta=%.1f ? %s   (within-budget genuinely improving point? %s)\n",
        r_indep.Delta, OUTER_BUDGET_DELTA, r_indep.Delta <= OUTER_BUDGET_DELTA,
        cls.outer_feasible && kappa9 < KAPPA_FIXED_REFERENCE)

    println("\n" * "="^100)
    println("Independent verification complete.")
    println("="^100)
end

main()
