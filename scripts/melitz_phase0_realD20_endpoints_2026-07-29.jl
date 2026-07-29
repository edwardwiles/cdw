# 2026-07-29 real-D20 fixed-A/f gamma-profile campaign, Phase 0: establish the exact
# calibration and theoretical endpoints, per the governing prompt. Cheap script -- no
# outer/inner KNITRO campaign, only the calibration build + a handful of direct fixed-A/f
# evaluations near/at each candidate endpoint to check whether it is an ordinary evaluable
# point or a genuine open (Delta->infinity or otherwise singular) limit.
#
# Resolves one live ambiguity before the main campaign is built: a 2026-07-24 session
# (docs/melitz_real_d20_evaluation_cap_correction_2026-07-24.md Section 10.1) used
# g_floor=0.0 (gamma_prime_target=1) as a convenient symmetric-box edge and LABELED it
# "GT=0%", but that session's own :upper-direction search never actually needed or tested the
# upper bound -- it was non-binding by construction. This script derives the true analytical
# ceiling on kappa_ratio from first principles (kappa_ratio <= 1, the ordinary "gains from
# trade cannot be negative" restriction -- the complement of the paper's OWN proven
# kappa_ratio >= lambda_dd^(1/(sigma-1)) ceiling on GT) and checks numerically whether
# kappa_max=1 coincides with gamma_prime=1 (g=0) at this real-D20 fixture, or whether they are
# two different points -- reporting the true, freshly-derived kappa_max=1 endpoint either way,
# not merely reusing the 2026-07-24 session's own g=0 convenience choice.

using Pkg
REPO = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
Pkg.activate(REPO)
using Printf, DelimitedFiles, LinearAlgebra
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
include(joinpath(REPO, "scripts", "melitz_regression_fixtures_2026-07-28.jl"))
using KNITRO

println("Julia threads: ", Threads.nthreads(), "   BLAS threads: ", BLAS.get_num_threads())
BLAS.set_num_threads(1)
flush(stdout)

kappa_of_g(g::Real, wratio::Real, sigma::Real) = wratio * exp(g)^(1 / (sigma - 1))
g_of_kappa(kappa::Real, wratio::Real, sigma::Real) = log((kappa / wratio)^(sigma - 1))

function wage_ratio_and_lambda_jj(theta0, ctx)
    state = melitz_outer_state(theta0, ctx)
    j = ctx.target_country
    lambda_jj = state.equilibrium.trade_flow[j, j] / state.equilibrium.expenditure[j]
    wratio = 1.0 / ctx.w[j]
    return wratio, lambda_jj
end

function main()
    println("="^100)
    println("PHASE 0: real-D20 calibration + theoretical gamma_d_prime endpoints")
    println("="^100)

    real_dir = joinpath(REPO, "real_data", "noah_D20")
    @assert isdir(real_dir) "real_data/noah_D20 not found"
    lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    @assert focal !== nothing "France ('fra') not found in countries.csv"
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal,
        p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)

    @printf("\nfocal country      = %s (index %d of %d)\n", countries[focal], focal, calib.D)
    @printf("sigma               = %.6f\n", calib.sigma)
    @printf("theta_star          = %.6f\n", calib.theta_star)
    @printf("calibrated gamma_prime_target (calib.gamma_prime_target) = %.10f\n", calib.gamma_prime_target)
    @printf("calibrated w_prime (autarky counterfactual wage, normalized) = %.10f\n", calib.w_prime)
    @printf("calibrated w[target] = %.10f\n", calib.w[focal])
    @printf("QMC: W=80000 seed=1 (production convention)\n")

    CAP = 10.0
    policy = CappedEvaluation(CAP)
    inner_opt_capped = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
    obj20, theta0_20 = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
        inner_loop_opt=inner_opt_capped, forbid_dense_fallback=true, policy=policy)
    ctx20 = obj20.γ
    r0 = evaluate_melitz_delta(theta0_20, ctx20, obj20; cold=true, store_G=false)
    @assert r0.verified "real-D20 base point failed to verify"

    wratio, lambda_jj = wage_ratio_and_lambda_jj(theta0_20, ctx20)
    g_pareto = theta0_20[1]
    kappa_pareto = kappa_of_g(g_pareto, wratio, calib.sigma)
    GT_pareto = 1 - kappa_pareto

    @printf("\n-- Fréchet calibration --\n")
    @printf("g_pareto            = %.10f\n", g_pareto)
    @printf("gamma_prime_pareto  = %.10f  (cross-check vs calib.gamma_prime_target: diff=%.3e)\n",
        exp(g_pareto), exp(g_pareto) - calib.gamma_prime_target)
    @printf("wage_ratio (w_prime/w[target]) = %.10f\n", wratio)
    @printf("kappa_pareto        = %.10f\n", kappa_pareto)
    @printf("GT_pareto           = %.6f%%\n", 100 * GT_pareto)
    @printf("DeltaStar_pareto (from base-point solve) = %.6e  nStatus=%d  verified=%s\n",
        r0.Delta, r0.nStatus, r0.verified)

    println("\n" * "="^100)
    println("THEORETICAL MINIMUM gamma_d_prime (branch A: calibration -> down)")
    println("="^100)
    kappa_min = lambda_jj^(1 / (calib.sigma - 1))
    g_ceiling = g_of_kappa(kappa_min, wratio, calib.sigma)
    @printf("lambda_jj (domestic trade share at theta0, target=%s) = %.10f\n", countries[focal], lambda_jj)
    @printf("kappa_min = lambda_jj^(1/(sigma-1))        = %.10f\n", kappa_min)
    @printf("g_ceiling = g_of_kappa(kappa_min)           = %.10f\n", g_ceiling)
    @printf("gamma_d_prime_min_theory = exp(g_ceiling)   = %.10f\n", exp(g_ceiling))
    @printf("GT at this endpoint (=1-kappa_min)          = %.6f%%  <- the paper's proven GT UPPER bound\n", 100 * (1 - kappa_min))
    # Cross-check 2: map GT_max through the live kappa/gamma relationship back to gamma.
    kappa_from_GTmax = 1 - (1 - kappa_min)
    g_from_GTmax = g_of_kappa(kappa_from_GTmax, wratio, calib.sigma)
    @printf("cross-check (via GT->kappa->g): g=%.10f  (diff vs g_ceiling: %.3e)\n",
        g_from_GTmax, g_from_GTmax - g_ceiling)

    println("\n" * "="^100)
    println("THEORETICAL MAXIMUM gamma_d_prime (branch B: calibration -> up)")
    println("="^100)
    kappa_max = 1.0   # ordinary "gains from trade cannot be negative" bound: kappa_ratio <= 1
    g_floor_kappamax = g_of_kappa(kappa_max, wratio, calib.sigma)
    @printf("kappa_max = 1 (GT cannot be negative)       = %.10f\n", kappa_max)
    @printf("g_floor (via kappa_max=1)                    = %.10f\n", g_floor_kappamax)
    @printf("gamma_d_prime_max_theory = exp(g_floor)      = %.10f\n", exp(g_floor_kappamax))
    @printf("GT at this endpoint (=1-kappa_max)           = %.6f%%  <- zero gains from trade (autarky-like reference)\n", 100 * (1 - kappa_max))
    @printf("\n[ambiguity check] literal gamma_prime=1 (g=0) point:\n")
    @printf("  kappa_of_g(0) = wage_ratio = %.10f   GT(g=0) = %.6f%%\n", kappa_of_g(0.0, wratio, calib.sigma), 100 * (1 - kappa_of_g(0.0, wratio, calib.sigma)))
    @printf("  |g_floor(kappa_max=1) - 0| = %.6e  (this session's own prior convention was g_floor=0 exactly)\n", abs(g_floor_kappamax))
    if abs(g_floor_kappamax) < 1e-6
        println("  => g_floor(kappa_max=1) coincides with g=0 to high precision: wage_ratio==1 essentially exactly at this fixture.")
    else
        println("  => g_floor(kappa_max=1) DOES NOT coincide with g=0 -- wage_ratio != 1 at this fixture.")
        println("     Using the freshly-derived kappa_max=1 endpoint (g_floor_kappamax) as the true theoretical")
        println("     maximum, NOT the 2026-07-24 session's g=0 convenience choice, per this session's own")
        println("     'derive analytically, cross-check via kappa/gamma' mandate.")
    end

    println("\n" * "="^100)
    println("OPEN/SINGULAR ENDPOINT CHECK -- attempt direct solves very near each endpoint")
    println("="^100)
    bank_probe = MelitzDualBank(8)
    session_probe = MelitzInnerSession(obj20, ctx20, policy; bank=bank_probe)

    function probe(label, g)
        theta = copy(theta0_20); theta[1] = g
        t0 = time()
        r = solve_melitz_delta!(session_probe, theta, policy; origin_block_screen=true, warm_start_source=:previous)
        wall = time() - t0
        kind = r isa FiniteSolved ? "FiniteSolved" : r isa AboveEvaluationCap ? "AboveEvaluationCap" :
               r isa InfiniteDeltaCertified ? "InfiniteDeltaCertified" : "NumericalFailure"
        Delta = r isa FiniteSolved ? r.Delta : NaN
        @printf("  %-28s g=%.6f  kappa=%.6f  GT=%.6f%%  -> %-24s Delta=%.4e  wall=%.2fs\n",
            label, g, kappa_of_g(g, wratio, calib.sigma), 100*(1-kappa_of_g(g, wratio, calib.sigma)), kind, Delta, wall)
        flush(stdout)
        return r
    end

    println("\n-- near g_ceiling (theoretical minimum gamma) --")
    probe("g_ceiling exact", g_ceiling)
    probe("t=0.995 toward ceiling", g_pareto + 0.995*(g_ceiling - g_pareto))
    probe("t=0.999 toward ceiling", g_pareto + 0.999*(g_ceiling - g_pareto))

    println("\n-- near g_floor (theoretical maximum gamma, kappa_max=1) --")
    probe("g_floor exact", g_floor_kappamax)
    probe("t=0.995 toward floor", g_pareto + 0.995*(g_floor_kappamax - g_pareto))
    probe("t=0.999 toward floor", g_pareto + 0.999*(g_floor_kappamax - g_pareto))

    println("\nDONE Phase 0.")
    return (calib=calib, theta0=theta0_20, ctx=ctx20, obj=obj20, g_pareto=g_pareto, wratio=wratio,
            lambda_jj=lambda_jj, kappa_pareto=kappa_pareto, kappa_min=kappa_min, g_ceiling=g_ceiling,
            kappa_max=kappa_max, g_floor=g_floor_kappamax)
end

main()
