# 2026-07-24, continuation of the evaluation-cap-correction session (user-directed):
# replaces the outer search's ad hoc symmetric g-radius (0.05, purely a numerical
# well-posedness device, no economic content) with the THEORETICALLY EXACT range on
# g = log(gamma_prime_target):
#
#   g_ceiling = log((lambda_dd^(1/(sigma-1)) / (w_prime/w_target))^(sigma-1))   [Delta->infinity limit, upper GT bound]
#   g_floor   = 0   [GT=0, gamma_prime_target=1]
#
# so g in [g_ceiling, g_floor] -- KNITRO can never be asked to evaluate a point that is
# provably infeasible for ANY finite Delta (the theoretical GT ceiling the user's paper
# proves is lambda_dd^(1/(sigma-1)), converted here to the corresponding bound on the outer
# coordinate itself, per the user's own explicit request).
#
# For this session's :upper-direction search (minimizing g), only the LOWER bound is ever
# binding; the theta_box mechanism (solve_melitz_finite_delta_bound) builds a SYMMETRIC box
# theta_init .+- radius, so the g_radius used here is chosen to reproduce the EXACT lower
# bound (g_ceiling) for this direction -- the resulting upper bound is not the true g_floor=0
# (it is tighter, but harmless/non-binding for a search that only ever decreases g).
# A/f bounds are LEFT UNCHANGED (still the ad hoc 0.15 radius) -- this is a single-variable
# test, per the user's own "try gamma_d' first, then A/f" plan.
#
# Usage: julia --project=. -t 16 scripts/melitz_real_d20_theoretical_gamma_bound_2026-07-24.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS, norm

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
const INNER_OPT = get(ENV, "MELITZ_INNER_OPT_CAMPAIGN", joinpath(dirname(@__DIR__), "melitz_inner_loop_options_capped_2026-07-24.opt"))
const OUTER_OPT_CAMPAIGN = get(ENV, "MELITZ_OUTER_OPT_CAMPAIGN", joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt"))
const CAMPAIGN_SECONDS = parse(Float64, get(ENV, "MELITZ_CAMPAIGN_SECONDS", "180"))
const DELTA_EVALUATION_CAP = 10.0
const OUTER_BUDGET_DELTA = 1.0
const LOWER_LIMIT_GUARD = 1e-6
const G_FIXED_REFERENCE = -0.49783321
const CAMPAIGN_START_G = -0.497333

function load_calibration()
    lambdaData = readdlm(joinpath(REAL_DIR, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(REAL_DIR, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(REAL_DIR, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(REAL_DIR, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate,
        focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
    return calib, lambdaData, focal
end

kappa_of_g(g::Real, calib) = (calib.w_prime / calib.w[calib.target_country]) * exp(g)^(1 / (calib.sigma - 1))

function theoretical_g_ceiling(calib, lambdaData, focal)
    lambda_dd = lambdaData[focal, focal]
    kappa_min = lambda_dd^(1 / (calib.sigma - 1))
    g_ceiling = log((kappa_min / (calib.w_prime / calib.w[calib.target_country]))^(calib.sigma - 1))
    return g_ceiling, kappa_min
end

function block_scaled_theta_box(ctx; g_radius, A_radius=0.15, f_radius=0.15)
    n = 1 + (ctx.D^2 - 1) + (length(ctx.f_free_lin) - 1)
    nA = ctx.D^2 - 1
    box = zeros(n)
    box[1] = g_radius
    box[2:1+nA] .= A_radius
    box[2+nA:end] .= f_radius
    return box
end

mutable struct EvalCapLogger
    t0::Float64
    n_fc::Int
    events::Vector{NamedTuple}
    best_kappa::Float64
    best_theta::Union{Nothing,Vector{Float64}}
    above_cap_thetas::Vector{Vector{Float64}}
end
EvalCapLogger() = EvalCapLogger(time(), 0, NamedTuple[], Inf, nothing, Vector{Float64}[])

function log_event!(logger::EvalCapLogger, calib, theta, result)
    logger.n_fc += 1
    t = time() - logger.t0
    g = theta[1]
    kappa = kappa_of_g(g, calib)
    kind = result isa FiniteSolved ? :FiniteSolved :
           result isa AboveEvaluationCap ? :AboveEvaluationCap :
           result isa InfiniteDeltaCertified ? :InfiniteDeltaCertified : :NumericalFailure
    Delta_solved = result isa FiniteSolved ? result.Delta : NaN
    within_budget = result isa FiniteSolved && Delta_solved <= OUTER_BUDGET_DELTA
    if within_budget && kappa < logger.best_kappa
        logger.best_kappa = kappa
        logger.best_theta = copy(theta)
    end
    result isa AboveEvaluationCap && push!(logger.above_cap_thetas, copy(theta))
    push!(logger.events, (t=t, n_fc=logger.n_fc, kind=kind, Delta_solved=Delta_solved, g=g, kappa=kappa, GT=1 - kappa))
    @printf("[LIVE gamma-bound] n_fc=%4d t=%7.2fs kind=%-20s g=%9.5f kappa=%9.6f GT=%8.5f  Delta=%12.6e\n",
        logger.n_fc, t, kind, g, kappa, 1 - kappa, Delta_solved)
    flush(stdout)
    return nothing
end

function main()
    calib, lambdaData, focal = load_calibration()
    g_ceiling, kappa_min = theoretical_g_ceiling(calib, lambdaData, focal)
    @printf("theoretical g_ceiling = %.6f  (kappa_min=%.6f, GT_ceiling=%.6f)\n", g_ceiling, kappa_min, 1 - kappa_min)

    BLAS.set_num_threads(16)
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ

    theta_init = copy(theta_calib); theta_init[1] = CAMPAIGN_START_G
    theta_fixed_af = copy(theta_calib); theta_fixed_af[1] = G_FIXED_REFERENCE

    # g_radius chosen so theta_init[1] - g_radius == g_ceiling EXACTLY (see file header for
    # why the upper side of this symmetric box is not the true g_floor=0, and why that's
    # harmless for this :upper-direction, g-decreasing search).
    g_radius = theta_init[1] - g_ceiling
    @printf("g_radius = theta_init[1] - g_ceiling = %.6f - (%.6f) = %.6f\n", theta_init[1], g_ceiling, g_radius)
    @printf("  => effective lower bound on g: %.6f (== g_ceiling, exact)\n", theta_init[1] - g_radius)
    @printf("  => effective upper bound on g: %.6f (tighter than the true g_floor=0.0, non-binding for this direction)\n", theta_init[1] + g_radius)
    theta_box = block_scaled_theta_box(ctx; g_radius=g_radius, A_radius=0.15, f_radius=0.15)

    logger = EvalCapLogger()
    on_result(theta, result) = log_event!(logger, calib, theta, result)
    melitz_profile_reset!()
    MELITZ_PROFILE[] = true

    println("\n" * "="^100)
    @printf("Gamma-bound campaign: delta=%.1f, delta_evaluation_cap=%.1f, g in [%.6f, %.6f] (theoretical), A/f radius unchanged (0.15)\n",
        OUTER_BUDGET_DELTA, DELTA_EVALUATION_CAP, theta_init[1] - g_radius, theta_init[1] + g_radius)
    println("="^100)
    flush(stdout)

    t0 = time()
    result = solve_melitz_finite_delta_bound(ctx, obj_inner, theta_init; delta=OUTER_BUDGET_DELTA,
        direction=:upper, delta_evaluation_cap=DELTA_EVALUATION_CAP,
        gradient_backend=:B_direct_argument_parallel, h=1e-4, theta_box=theta_box,
        cutoff_constraint_backend=:linear,
        inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT_CAMPAIGN,
        on_inner_result=on_result, dual_bank_max_size=8, lower_limit_guard=LOWER_LIMIT_GUARD,
        external_incumbent=theta_fixed_af)
    wall = time() - t0
    MELITZ_PROFILE[] = false

    println("\n" * "="^100)
    println("CAMPAIGN RESULT [gamma-bound]")
    println("="^100)
    @printf("wall=%.2fs  nStatus=%d  n_fc_calls=%d  n_ga_calls=%d  n_inner_solved=%d\n",
        wall, result.nStatus, result.n_fc_calls, result.n_ga_calls, result.n_inner_solved)
    @printf("n_infinite_delta_reject=%d  n_above_cap_reject=%d  n_numerical_failure_reject=%d\n",
        result.n_infinite_delta_reject, result.n_above_cap_reject, result.n_numerical_failure_reject)
    for (label, cand) in (("initial_incumbent", result.initial_incumbent),
                          ("best_live_incumbent", result.best_live_incumbent),
                          ("cold_verified_incumbent (THE ANSWER)", result.cold_verified_incumbent))
        if cand === nothing
            @printf("  %-38s: nothing\n", label)
        else
            g = cand.eval.theta_free[1]
            kappa = kappa_of_g(g, calib)
            @printf("  %-38s: g=%.6f  kappa=%.6f  GT=%.6f  Delta=%.6e  source=%s\n",
                label, g, kappa, 1 - kappa, cand.eval.Delta, cand.source)
        end
    end

    evs = logger.events
    println("\nFull trajectory (min g reached, max g reached, best FiniteSolved-within-budget kappa):")
    @printf("  min g visited = %.6f  (%.2f%% of the way to g_ceiling from theta_init)\n",
        minimum(e.g for e in evs), 100 * (theta_init[1] - minimum(e.g for e in evs)) / g_radius)
    @printf("  n FiniteSolved = %d, n AboveEvaluationCap = %d, n InfiniteDeltaCertified = %d\n",
        count(e -> e.kind == :FiniteSolved, evs), count(e -> e.kind == :AboveEvaluationCap, evs),
        count(e -> e.kind == :InfiniteDeltaCertified, evs))

    println("\n" * "="^100)
    println("WALL-CLOCK DECOMPOSITION")
    println("="^100)
    melitz_profile_report(stdout; trajectory_total_s=wall)

    BLAS.set_num_threads(1)
    println("\nDONE.")
    return result, logger
end

main()
