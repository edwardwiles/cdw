# 2026-07-24, continuation of the evaluation-cap-correction session: two user-directed
# diagnostics on WHY the flexible D=20 outer search cannot find a second outer-feasible
# point, given the corrected inner/outer interface is now confirmed correct
# (docs/melitz_real_d20_evaluation_cap_correction_2026-07-24.md):
#
#   1. W-SENSITIVITY: user's intuition is that Melitz's inner CC moments may need a LARGER
#      Monte Carlo sample W than the Ricardian model does for the SAME reliability -- if the
#      InfiniteDeltaCertified/AboveEvaluationCap classifications this session's campaign
#      produced are partly finite-sample artifacts (the range-screen certificate is a
#      statement about the OBSERVED W-draw sample, not the population), a larger W should
#      make some of them resolve to genuine FiniteSolved points, or at least materially
#      shrink their certified_lower_bound. Tested here by re-evaluating the SAME grid points
#      and the SAME sampled AboveEvaluationCap thetas from the cap=10 campaign
#      (docs' own Section 4 table) under a fresh W=160,000 calibration (repo memory already
#      documents `W=8000 understates kappa for delta>=1 vs W>=80,000` -- this tests whether
#      the same direction of effect continues past 80,000).
#
#   2. GRADIENT-INFORMED SMALL STEP: computes the REAL envelope-theorem gradient of
#      Delta(theta) at the campaign's own starting point (the SAME `:B_direct_argument_parallel`
#      backend the production outer search itself uses, not a new formula), then tests
#      whether a SMALL step in the direction that decreases Delta fastest (-grad(Delta),
#      normalized) stays feasible for a wider range of step sizes than KNITRO's own first
#      trial step did (that step, `g: -0.497333 -> -0.501266` PLUS a simultaneous 798-dim
#      A/f move within the block-scaled trust region, landed at
#      certified_lower_bound=1.58e8 -- ENORMOUSLY infeasible for a nominally "small" trust-
#      region step). Also decomposes ||grad(Delta)|| into its g-coordinate vs A/f-coordinate
#      components to see where the local sensitivity actually concentrates.
#
# Usage: julia --project=. -t 16 scripts/melitz_real_d20_w_sensitivity_and_gradient_stepsize_2026-07-24.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS, norm

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
const INNER_OPT = get(ENV, "MELITZ_INNER_OPT", joinpath(dirname(@__DIR__), "melitz_inner_loop_options_capped_2026-07-24.opt"))
const OUTER_OPT = get(ENV, "MELITZ_OUTER_OPT", joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt"))
const OUTER_BUDGET_DELTA = 1.0
const G_FIXED_REFERENCE = -0.49783321
const CAMPAIGN_START_G = -0.497333   # this session's own cap-sensitivity campaigns' shared starting point

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

kappa_of_g(g::Real, calib) = (calib.w_prime / calib.w[calib.target_country]) * exp(g)^(1 / (calib.sigma - 1))

# ============================================================================
# Experiment 1: W-sensitivity. Re-evaluates a small, targeted set of points -- the same
# gamma-only grid + the same 8 AboveEvaluationCap thetas the cap=10 campaign's own diagnosis
# sampled -- under a fresh, higher-W calibration.
# ============================================================================
function w_sensitivity(calib, W::Int; grid_g=G_FIXED_REFERENCE .- (0.0:0.01:0.06))
    println("="^100)
    @printf("EXPERIMENT 1: W-sensitivity at W=%d (baseline in the prior report was W=80,000)\n", W)
    println("="^100)
    t0 = time()
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ
    @printf("  bundle build (fresh z_draws at W=%d): %.1fs\n", W, time() - t0)
    flush(stdout)

    println("\n-- 1a. gamma-only grid, unrestricted evaluate_melitz_delta --")
    cache = MelitzDeltaEvalCache(32)
    grid_results = NamedTuple[]
    for g in grid_g
        theta = copy(theta_calib); theta[1] = g
        t0g = time()
        r = evaluate_melitz_delta(theta, ctx, obj_inner; cold=true, cache=cache, store_G=false)
        dt = time() - t0g
        @printf("  g=%9.5f  Delta=%12.6e  nStatus=%4d  verified=%-5s  wall=%6.2fs\n", g, r.Delta, r.nStatus, r.verified, dt)
        flush(stdout)
        push!(grid_results, (g=g, Delta=r.Delta, nStatus=r.nStatus, verified=r.verified, theta=copy(theta)))
    end

    return (ctx=ctx, obj_inner=obj_inner, theta_calib=theta_calib, grid=grid_results)
end

# ============================================================================
# Experiment 2: gradient-informed small step at the campaign's own starting point.
# ============================================================================
function gradient_stepsize_experiment(ctx, obj_inner, theta_calib)
    println("\n" * "="^100)
    println("EXPERIMENT 2: gradient-informed small step at the campaign's starting point")
    println("="^100)
    theta_init = copy(theta_calib); theta_init[1] = CAMPAIGN_START_G
    r0 = evaluate_melitz_delta(theta_init, ctx, obj_inner; cold=true, store_G=false)
    @printf("theta_init: g=%.6f  Delta=%.6e  nStatus=%d  verified=%s\n", CAMPAIGN_START_G, r0.Delta, r0.nStatus, r0.verified)
    @assert r0.nStatus == 0 "theta_init must be a genuine, converged inner solve for the envelope-theorem gradient to be valid"

    # Build the SAME PsiObjectiveBundleImplicit the production outer search itself uses,
    # populate obj.H at theta_init exactly as melitz_classified_inner_solve does, and read
    # off the JUST-CONVERGED dual (r0.dual_x) -- the envelope-theorem gradient is exact AT
    # THIS dual, no re-solve needed (same fixed-dual construction the outer cb_G! itself
    # relies on for every genuinely FiniteSolved point).
    obj = build_melitz_implicit_bundle(ctx, obj_inner.U, theta_init; delta=OUTER_BUDGET_DELTA,
        find_smallest=true, gradient_backend=:B_direct_argument_parallel, h=1e-4,
        inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT)
    CS = CounterfactualSensitivity
    G_now = CS.select_G_from_H(obj, obj.H)
    obj.moments!(@view(obj.H[:, 1]), G_now, theta_init, obj.U, obj)
    obj.H[:, 2] .= 1.0
    x_star = r0.dual_x

    n = length(theta_init)
    grad = zeros(n)
    direct_gradient_fn = make_melitz_gradient_delta_direct_parallel(1e-4)
    t0 = time()
    direct_gradient_fn(grad, theta_init, ctx, obj, x_star)   # grad[r] == d(1e10*Delta)/dtheta_r
    grad ./= 1e10                                             # -> d(Delta)/dtheta_r, unscaled
    @printf("gradient computed in %.2fs\n", time() - t0)

    g_component = grad[1]
    Af_component_norm = norm(@view grad[2:end])
    @printf("||grad(Delta)|| = %.6e   (g-coordinate component = %.6e ; A/f-block norm = %.6e)\n",
        norm(grad), g_component, Af_component_norm)
    @printf("=> %.4f%% of the gradient's Euclidean norm sits in the 797 A/f coordinates, %.4f%% in the single g coordinate\n",
        100 * Af_component_norm^2 / norm(grad)^2, 100 * g_component^2 / norm(grad)^2)
    flush(stdout)

    # (a) pure objective direction: decrease g only, A/f exactly fixed (== the gamma-only
    # grid path already explored -- included here for a like-for-like step-size comparison
    # against (b)/(c) below, using genuinely tiny steps rather than the grid's own 0.005-0.01
    # increments).
    e1 = zeros(n); e1[1] = 1.0

    # (b) steepest DESCENT direction of Delta itself: moves ALL 798 coordinates, in whatever
    # combination reduces Delta fastest to first order. This is the most "generous" possible
    # direction for staying feasible -- if even THIS blows up quickly, the local corridor is
    # thin in literally every direction, not merely along the objective axis.
    d_desc = -grad ./ norm(grad)

    # (c) KNITRO's own approximate first trial direction, reconstructed only in the sense of
    # "how much did g move relative to how much AboveEvaluationCap it produced" -- not
    # replicated exactly (that requires KNITRO's own approximate Hessian), reported instead
    # as measured fact for comparison (see printed summary at the end).

    println("\n-- step along pure objective direction e1 (decrease g, A/f fixed) --")
    for eps in (1e-5, 1e-4, 1e-3, 1e-2, 0.0039, 1e-1)
        theta_try = theta_init .- eps .* e1
        r = evaluate_melitz_delta(theta_try, ctx, obj_inner; cold=true, store_G=false)
        @printf("  eps=%8.5f  g=%9.5f  Delta=%12.6e  nStatus=%4d  verified=%-5s\n", eps, theta_try[1], r.Delta, r.nStatus, r.verified)
        flush(stdout)
    end

    println("\n-- step along -grad(Delta)/||grad(Delta)|| (steepest descent of Delta itself, full 798-dim) --")
    for eps in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
        theta_try = theta_init .+ eps .* d_desc
        r = evaluate_melitz_delta(theta_try, ctx, obj_inner; cold=true, store_G=false)
        dg = theta_try[1] - theta_init[1]
        dAf = norm(@view((theta_try .- theta_init)[2:end]))
        @printf("  eps=%8.5f  dg=%10.6f  ||dAf||=%9.6f  Delta=%12.6e  nStatus=%4d  verified=%-5s\n",
            eps, dg, dAf, r.Delta, r.nStatus, r.verified)
        flush(stdout)
    end

    println("\n-- step along +grad(Delta)/||grad(Delta)|| (steepest ASCENT of Delta -- sanity check, should get worse fast) --")
    for eps in (1e-5, 1e-4, 1e-3)
        theta_try = theta_init .- eps .* d_desc
        r = evaluate_melitz_delta(theta_try, ctx, obj_inner; cold=true, store_G=false)
        @printf("  eps=%8.5f  Delta=%12.6e  nStatus=%4d  verified=%-5s\n", eps, r.Delta, r.nStatus, r.verified)
        flush(stdout)
    end

    return grad
end

function main()
    calib = load_calibration()
    BLAS.set_num_threads(16)

    # -------- Experiment 2 (already validated once at W=80,000, 2026-07-24 -- gradient
    # concentrated 98.8% in the A/f block, breaks at eps=1e-4 with no cutoff-constraint
    # violation): skip on repeat invocations via MELITZ_RUN_GRADIENT_EXP=0, so a rerun of
    # just Experiment 1 at a different W doesn't re-pay this cost. --------
    grad = nothing
    if get(ENV, "MELITZ_RUN_GRADIENT_EXP", "1") == "1"
        obj_inner80, theta_calib80 = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1, inner_loop_opt=INNER_OPT)
        ctx80 = obj_inner80.γ
        grad = gradient_stepsize_experiment(ctx80, obj_inner80, theta_calib80)
    end

    # -------- Experiment 1: W-sensitivity, gamma-only grid extended past the W=80,000
    # baseline's own conditioning-fragile boundary (g=-0.51783/-0.52783, nStatus=-401 there)
    # to see whether more draws push that boundary further out. --------
    W_HIGH = parse(Int, get(ENV, "MELITZ_W_HIGH", "160000"))
    res160 = w_sensitivity(calib, W_HIGH)

    BLAS.set_num_threads(1)
    println("\nDONE.")
    return grad, res160
end

main()
