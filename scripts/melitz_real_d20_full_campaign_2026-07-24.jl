# Continuation session (2026-07-24), Phase 7: 10-minute full A/f outer campaign at the
# real D=20/W=80,000/seed=1 calibrated reference, delta=1, upper-GT-bound direction.
#
# Usage: MELITZ_G_START=<g> julia --project=. -t 16 scripts/melitz_real_d20_full_campaign_2026-07-24.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS, norm

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
# CRITICAL (found live, 2026-07-24): the production inner_loop_opt (maxit=10000) lets a
# SINGLE nested inner CC solve run unbounded inside one outer callback -- the outer NLP's
# own `maxtime_real` cannot preempt a call already in flight (KNITRO only checks its time
# budget BETWEEN callback returns), and Phase 6 already showed points near the Delta~1
# boundary here are conditioning-fragile enough to grind for many minutes without
# converging. A first attempt at this campaign hung for 55+ minutes on exactly this (killed
# manually). Every inner solve in this campaign -- including the initial cold evaluation --
# now uses a maxit-capped inner_loop_opt so no single callback can run unbounded, bounding
# the WORST CASE total wall-clock even if maxtime_real's own between-call check is delayed.
const INNER_OPT = get(ENV, "MELITZ_INNER_OPT_CAMPAIGN",
    joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt"))
const OUTER_OPT_CAMPAIGN = get(ENV, "MELITZ_OUTER_OPT_CAMPAIGN",
    joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt"))

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

mutable struct TrajectoryLogger
    t0::Float64
    n_fc::Int
    n_ga::Int
    events::Vector{NamedTuple}
    best_kappa::Float64
    best_theta::Union{Nothing,Vector{Float64}}
    first_fc_time::Union{Nothing,Float64}
    first_grad_time::Union{Nothing,Float64}
end
TrajectoryLogger() = TrajectoryLogger(time(), 0, 0, NamedTuple[], Inf, nothing, nothing, nothing)

function log_inner_result!(logger::TrajectoryLogger, calib, theta, result)
    logger.n_fc += 1
    logger.first_fc_time === nothing && (logger.first_fc_time = time() - logger.t0)
    t = time() - logger.t0

    kind = result isa InnerSolved ? :InnerSolved :
           result isa BudgetInfeasible ? :BudgetInfeasible :
           result isa MomentInfeasible ? :MomentInfeasible :
           result isa NumericalFailure ? :NumericalFailure : :Other
    Delta = result isa InnerSolved ? result.Delta :
            result isa BudgetInfeasible ? result.lower_bound : NaN
    g = theta[1]
    gamma_prime = exp(g)
    kappa = (calib.w_prime / calib.w[calib.target_country]) * gamma_prime^(1 / (calib.sigma - 1))
    accepted = kind == :InnerSolved && Delta <= 1.0
    if accepted && kappa < logger.best_kappa
        logger.best_kappa = kappa
        logger.best_theta = copy(theta)
    end
    push!(logger.events, (t=t, n_fc=logger.n_fc, kind=kind, Delta=Delta, g=g, kappa=kappa,
        GT=1 - kappa, accepted=accepted, theta_norm=norm(theta)))
    # LIVE progress line + flush, every single event -- so an early check-in during the
    # 600s campaign can see genuine progress (or its absence) rather than flying blind
    # until the run finishes or is killed (the exact gap that let the first campaign
    # attempt hang undetected for 55+ minutes).
    @printf("[LIVE] n_fc=%4d  t=%7.2fs  kind=%-16s  Delta=%10.4e  g=%9.5f  kappa=%9.6f  accepted=%s\n",
        logger.n_fc, t, kind, Delta, g, kappa, accepted)
    flush(stdout)
    return nothing
end

function main()
    calib = load_calibration()
    BLAS.set_num_threads(16)
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ

    g_start = parse(Float64, get(ENV, "MELITZ_G_START", string(theta_calib[1] - 0.45)))
    theta_init = copy(theta_calib); theta_init[1] = g_start
    r0 = evaluate_melitz_delta(theta_init, ctx, obj_inner; cold=true)
    @printf("Starting point: g=%.6f  Delta=%.6e  nStatus=%d  verified=%s  min_slack=%.4f\n",
        g_start, r0.Delta, r0.nStatus, r0.verified, r0.min_slack)
    kappa0 = (calib.w_prime / calib.w[calib.target_country]) * exp(g_start)^(1 / (calib.sigma - 1))
    @printf("kappa0=%.6f  GT0=%.6f\n\n", kappa0, 1 - kappa0)
    flush(stdout)

    melitz_profile_reset!()
    MELITZ_PROFILE[] = true

    logger = TrajectoryLogger()
    on_result(theta, result) = log_inner_result!(logger, calib, theta, result)

    # lower_limit_guard=49.0 => lower_limit = -(delta+guard) = -(1.0+49.0) = -50, matching
    # the Ricardian model's own hardcoded lower_limit=-50 convention (cc_algo/ccInner.jl,
    # ccOuter.jl) -- a KNITRO-NATIVE mid-solve early bailout (`if f <= lower_limit; return
    # -KN_INFINITY`) that Melitz's own bundle construction never wired up before this session
    # (build_melitz_implicit_bundle's own docstring: default `nothing` leaves it permanently
    # disabled, `-KNITRO.KN_INFINITY`). Without this, a single nested inner CC solve inside
    # one outer callback has NO fast-exit once it has clearly wandered far past any usable
    # delta budget, and runs all the way to the (comparatively very expensive) maxit/
    # convergence-tolerance criteria instead -- confirmed live: the first attempt at this
    # campaign (without this guard) hung for 55+ minutes before being killed manually.
    t0 = time()
    result = solve_melitz_finite_delta_bound(ctx, obj_inner, theta_init; delta=1.0, direction=:upper,
        gradient_backend=:B_direct_argument_parallel, h=1e-4, theta_box=2.0,
        inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT_CAMPAIGN,
        on_inner_result=on_result, dual_bank_max_size=8, lower_limit_guard=49.0)
    wall = time() - t0
    MELITZ_PROFILE[] = false

    println("\n" * "="^100)
    println("CAMPAIGN RESULT")
    println("="^100)
    @printf("wall=%.2fs  nStatus=%d  n_fc_calls=%d  n_ga_calls=%d  n_inner_solved=%d\n",
        wall, result.nStatus, result.n_fc_calls, result.n_ga_calls, result.n_inner_solved)
    @printf("n_moment_infeasible_reject=%d  n_budget_infeasible_reject=%d  n_numerical_failure_reject=%d  inner_eval_failures=%d\n",
        result.n_moment_infeasible_reject, result.n_budget_infeasible_reject,
        result.n_numerical_failure_reject, result.inner_eval_failures)

    for (label, cand) in (("initial_incumbent", result.initial_incumbent),
                          ("best_live_incumbent", result.best_live_incumbent),
                          ("cold_verified_incumbent", result.cold_verified_incumbent))
        if cand === nothing
            @printf("  %-24s: nothing\n", label)
        else
            g = cand.eval.theta_free[1]
            gamma_prime = exp(g)
            kappa = (calib.w_prime / calib.w[calib.target_country]) * gamma_prime^(1 / (calib.sigma - 1))
            @printf("  %-24s: objective=%.6f  g=%.6f  kappa=%.6f  GT=%.6f  Delta=%.6e  min_slack=%.4f  source=%s\n",
                label, cand.objective, g, kappa, 1 - kappa, cand.eval.Delta, cand.eval.min_slack, cand.source)
        end
    end

    println("\n" * "="^100)
    println("TRAJECTORY (first 40 and last 40 FC events)")
    println("="^100)
    @printf("%6s %10s %10s %14s %12s %12s %10s %8s\n", "n_fc", "t(s)", "kind", "Delta", "g", "kappa", "GT", "accepted")
    evs = logger.events
    show_idx = length(evs) <= 80 ? (1:length(evs)) : vcat(1:40, (length(evs)-39):length(evs))
    for i in show_idx
        e = evs[i]
        @printf("%6d %10.2f %10s %14.4e %12.6f %12.6f %10.6f %8s\n",
            e.n_fc, e.t, e.kind, e.Delta, e.g, e.kappa, e.GT, e.accepted)
    end
    @printf("\ntotal FC events logged=%d  first_fc_time=%.3fs\n", length(evs), something(logger.first_fc_time, NaN))
    @printf("best (accepted, min kappa) seen live: kappa=%.6f  GT=%.6f\n", logger.best_kappa, 1 - logger.best_kappa)

    println("\n" * "="^100)
    println("WALL-CLOCK DECOMPOSITION (melitz_profile_report)")
    println("="^100)
    melitz_profile_report(stdout; trajectory_total_s=wall)

    BLAS.set_num_threads(1)
    println("\nDONE.")
    return result, calib, ctx, obj_inner, logger
end

main()
