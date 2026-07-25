# 2026-07-24, continuation of the evaluation-cap-correction session (user-directed, step 2
# of "try the box constraint on gamma_d' first, then also add them on A/f"):
#
# Keeps the theoretical g-bound from melitz_real_d20_theoretical_gamma_bound_2026-07-24.jl
# (that run alone did NOT change the outcome -- still falls back to the external fixed-A/f
# incumbent), and ADDITIONALLY tightens the A/f block radius, informed directly by the
# gradient-magnitude experiment (melitz_real_d20_w_sensitivity_and_gradient_stepsize_2026-07-24.jl,
# Experiment 2): at theta_init, stepping along the FULL 798-dim steepest-descent direction of
# Delta(theta) (i.e. the most FAVORABLE possible combined direction) was still fine at a
# combined A/f displacement ||dAf||=1e-5, but broke completely (nStatus=-401) at ||dAf||=1e-4.
#
# No exact theoretical formula exists for a per-cell A/f bound (unlike gamma_d', which has
# the closed-form GT-ceiling result) -- A_radius=f_radius=1e-4 is chosen here as the SAME
# order of magnitude as that empirically-observed combined-norm breakdown scale, used
# per-coordinate rather than as a combined bound (so it is not a worst-case sqrt(797)-scaled
# bound, which would be far tighter still and would essentially freeze A/f outright -- this
# is a deliberately pragmatic middle choice, not a derived guarantee).
#
# Usage: julia --project=. -t 16 scripts/melitz_real_d20_gradient_informed_af_radius_2026-07-24.jl

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
const DELTA_EVALUATION_CAP = 10.0
const OUTER_BUDGET_DELTA = 1.0
const LOWER_LIMIT_GUARD = 1e-6
const G_FIXED_REFERENCE = -0.49783321
const CAMPAIGN_START_G = -0.497333
const AF_RADIUS = parse(Float64, get(ENV, "MELITZ_AF_RADIUS", "1e-4"))

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

function block_scaled_theta_box(ctx; g_radius, A_radius, f_radius)
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
    @printf("[LIVE af-radius] n_fc=%4d t=%7.2fs kind=%-20s g=%9.5f kappa=%9.6f GT=%8.5f  Delta=%12.6e\n",
        logger.n_fc, t, kind, g, kappa, 1 - kappa, Delta_solved)
    flush(stdout)
    return nothing
end

function main()
    calib, lambdaData, focal = load_calibration()
    g_ceiling, kappa_min = theoretical_g_ceiling(calib, lambdaData, focal)
    @printf("theoretical g_ceiling = %.6f  (kappa_min=%.6f, GT_ceiling=%.6f)\n", g_ceiling, kappa_min, 1 - kappa_min)
    @printf("A/f radius (gradient-informed) = %.2e\n", AF_RADIUS)

    BLAS.set_num_threads(16)
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ

    theta_init = copy(theta_calib); theta_init[1] = CAMPAIGN_START_G
    theta_fixed_af = copy(theta_calib); theta_fixed_af[1] = G_FIXED_REFERENCE

    g_radius = theta_init[1] - g_ceiling
    theta_box = block_scaled_theta_box(ctx; g_radius=g_radius, A_radius=AF_RADIUS, f_radius=AF_RADIUS)
    @printf("g in [%.6f, %.6f]  (theoretical lower bound exact)\n", theta_init[1] - g_radius, theta_init[1] + g_radius)
    @printf("A/f in [-%.2e, +%.2e] around calibration (per-coordinate, log space)\n", AF_RADIUS, AF_RADIUS)

    logger = EvalCapLogger()
    on_result(theta, result) = log_event!(logger, calib, theta, result)
    melitz_profile_reset!()
    MELITZ_PROFILE[] = true

    println("\n" * "="^100)
    println("Gradient-informed A/f-radius campaign")
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
    println("CAMPAIGN RESULT [af-radius]")
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
    println("\nFull trajectory summary:")
    @printf("  min g visited = %.6f\n", minimum(e.g for e in evs))
    @printf("  n FiniteSolved = %d, n AboveEvaluationCap = %d, n InfiniteDeltaCertified = %d, n NumericalFailure = %d\n",
        count(e -> e.kind == :FiniteSolved, evs), count(e -> e.kind == :AboveEvaluationCap, evs),
        count(e -> e.kind == :InfiniteDeltaCertified, evs), count(e -> e.kind == :NumericalFailure, evs))
    n_within_budget_solved = count(e -> e.kind == :FiniteSolved && e.Delta_solved <= OUTER_BUDGET_DELTA, evs)
    n_over_budget_solved = count(e -> e.kind == :FiniteSolved && e.Delta_solved > OUTER_BUDGET_DELTA, evs)
    @printf("  of the FiniteSolved points: %d within budget, %d over budget but under cap\n", n_within_budget_solved, n_over_budget_solved)

    println("\n" * "="^100)
    println("WALL-CLOCK DECOMPOSITION")
    println("="^100)
    melitz_profile_report(stdout; trajectory_total_s=wall)

    BLAS.set_num_threads(1)
    println("\nDONE.")
    return result, logger
end

main()
