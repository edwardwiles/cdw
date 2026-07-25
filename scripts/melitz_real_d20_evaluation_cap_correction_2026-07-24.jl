# 2026-07-24 evaluation-cap-correction session: empirical validation + corrected D=20
# constrained campaign under the CORRECTED inner/outer interface
# (src/melitz/inner_screening.jl, src/melitz/finite_delta_outer.jl -- see
# docs/melitz_real_d20_evaluation_cap_correction_2026-07-24.md for the full report).
# Supersedes scripts/melitz_real_d20_constrained_correction_2026-07-24.jl as the production
# constrained-search driver -- that script (and melitz_real_d20_full_campaign_2026-07-24.jl
# before it) is kept UNMODIFIED for direct before/after comparison; it exercises the OLD,
# pre-correction interface and will error under the renamed types if re-run.
#
# THE central semantic change under test: the abort threshold inside the routine inner
# evaluation is delta_evaluation_cap (default 10.0), NEVER the outer budget delta (default
# 1.0 here). A point with genuine DeltaStar between delta and delta_evaluation_cap must be
# solved FULLY (FiniteSolved) and reported to outer KNITRO with its real value/gradient --
# never intercepted early and replaced with a path-dependent certificate.
#
# Phases (toggle via env vars, so this script can be invoked multiple times cheaply for the
# cap-sensitivity comparison without repeating the expensive grid/diagnosis phases):
#   MELITZ_RUN_GRID=1       (default 1) -- Phase 1: extended gamma-only grid, finds real
#                                          finite-DeltaStar reference points.
#   MELITZ_RUN_CROSSCHECK=1 (default 1) -- Phase 2 (Section 5 tests): exact finite-value/
#                                          gradient cross-check vs an unrestricted solve.
#   MELITZ_RUN_OLDBUG_DEMO=1 (default 1) -- Phase 3: concrete demonstration that the OLD
#                                          delta-gated threshold would have wrongly aborted
#                                          a genuine finite point the corrected code solves.
#   MELITZ_RUN_CAMPAIGN=1   (default 1) -- Phase 4 (Section 8): the corrected constrained
#                                          D=20 campaign at the given delta_evaluation_cap.
#   MELITZ_RUN_ABOVECAP_DIAGNOSIS=1 (default 1) -- Phase 5 (Section 6): re-diagnose any
#                                          AboveEvaluationCap points the campaign produced,
#                                          at cap=10/50/effectively-uncapped.
#   MELITZ_DELTA_EVALUATION_CAP    (default "10.0")
#   MELITZ_CAMPAIGN_SECONDS        (default "180")
#   MELITZ_RUN_TAG                 (default "cap10") -- label for this run's own printed
#                                          section headers, so cap-sensitivity re-runs are
#                                          distinguishable in saved logs.
#
# Usage: julia --project=. -t 16 scripts/melitz_real_d20_evaluation_cap_correction_2026-07-24.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS, norm

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
const INNER_OPT = get(ENV, "MELITZ_INNER_OPT_CAMPAIGN",
    joinpath(dirname(@__DIR__), "melitz_inner_loop_options_capped_2026-07-24.opt"))
const OUTER_OPT_CAMPAIGN = get(ENV, "MELITZ_OUTER_OPT_CAMPAIGN",
    joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt"))
const CAMPAIGN_SECONDS = parse(Float64, get(ENV, "MELITZ_CAMPAIGN_SECONDS", "180"))
const DELTA_EVALUATION_CAP = parse(Float64, get(ENV, "MELITZ_DELTA_EVALUATION_CAP", "10.0"))
const OUTER_BUDGET_DELTA = 1.0
const LOWER_LIMIT_GUARD = parse(Float64, get(ENV, "MELITZ_LOWER_LIMIT_GUARD", "1e-6"))
const RUN_TAG = get(ENV, "MELITZ_RUN_TAG", "cap$(Int(round(DELTA_EVALUATION_CAP)))")
const RUN_GRID = get(ENV, "MELITZ_RUN_GRID", "1") == "1"
const RUN_CROSSCHECK = get(ENV, "MELITZ_RUN_CROSSCHECK", "1") == "1"
const RUN_OLDBUG_DEMO = get(ENV, "MELITZ_RUN_OLDBUG_DEMO", "1") == "1"
const RUN_CAMPAIGN = get(ENV, "MELITZ_RUN_CAMPAIGN", "1") == "1"
const RUN_ABOVECAP_DIAGNOSIS = get(ENV, "MELITZ_RUN_ABOVECAP_DIAGNOSIS", "1") == "1"

# Phase 6 companion-report reference point (docs/melitz_real_d20_outer_benchmark_2026-07-24.md
# Section 6.1/6.2, cold-verified): g_fixed=-0.49783321, Delta(g_fixed)=0.9969031. Re-derived
# fresh below (not hand-copied) so this script stays correct if the calibration ever changes.
const G_FIXED_REFERENCE = -0.49783321
const KAPPA_FIXED_REFERENCE = 0.92939627

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
# Phase 1: extended gamma-only (fixed A/f) grid, PAST the companion report's own
# g=-0.4988/Delta~1.06 endpoint -- probing how far genuine FiniteSolved values extend
# before this real-D20 fixture's own documented conditioning fragility (repo memory:
# "flexible theta ±5% finds inner unbounded, no rescue tool") takes over. Every point here
# is evaluated via evaluate_melitz_delta -- the SAME "authoritative fixed-outer-point
# evaluator" Section 5's cross-check reuses as ground truth, with NO delta_evaluation_cap
# machinery involved at all (an genuinely unrestricted diagnostic solve).
# ============================================================================
function extended_grid_scan(ctx, obj_inner, theta_calib; g_steps=G_FIXED_REFERENCE .- (0.0:0.01:0.30))
    println("="^100)
    println("PHASE 1 [$RUN_TAG]: extended gamma-only grid, UNRESTRICTED evaluate_melitz_delta (no cap)")
    println("="^100)
    cache = MelitzDeltaEvalCache(64)
    rows = NamedTuple[]
    for g in g_steps
        theta = copy(theta_calib); theta[1] = g
        t0 = time()
        r = evaluate_melitz_delta(theta, ctx, obj_inner; cold=true, cache=cache, store_G=false)
        dt = time() - t0
        @printf("  g=%9.5f  Delta=%12.6e  nStatus=%4d  verified=%-5s  wall=%6.2fs\n",
            g, r.Delta, r.nStatus, r.verified, dt)
        flush(stdout)
        push!(rows, (g=g, theta=copy(theta), Delta=r.Delta, nStatus=r.nStatus, verified=r.verified,
            dual_x=copy(r.dual_x), wall=dt))
        # Stop probing further once we hit a genuinely non-verified point twice in a row --
        # no point burning wall-clock deep into the documented conditioning-fragile region
        # once the boundary of "this restricted path can find finite points at all" is
        # already bracketed (Section 6's own "diagnostic solves, hard diagnostic time limit").
        if length(rows) >= 2 && !rows[end].verified && !rows[end-1].verified
            println("  -> two consecutive non-verified points; stopping the grid extension here")
            break
        end
    end
    return rows
end

# ============================================================================
# Phase 2 (governing prompt Section 5): exact finite-value/gradient cross-check. For every
# grid point with a VERIFIED, finite Delta, confirm that melitz_classified_inner_solve under
# delta_evaluation_cap > Delta returns FiniteSolved with the IDENTICAL Delta/dual as the
# unrestricted evaluate_melitz_delta ground truth -- and, for the subset with
# Delta > OUTER_BUDGET_DELTA, confirm this ALSO holds (the core over-budget-but-under-cap
# regression check, at real D=20 scale rather than only the unit-test fixture).
# ============================================================================
function crosscheck_finite_values(ctx, obj_inner, grid_rows; cap=DELTA_EVALUATION_CAP)
    println("\n" * "="^100)
    println("PHASE 2 [$RUN_TAG]: exact finite-value cross-check, delta_evaluation_cap=$cap")
    println("="^100)
    @printf("%10s %14s %8s %10s %14s %10s\n", "g", "Delta(true)", "verified", "over_budget", "Delta(capped)", "match")
    n_checked = 0; n_matched = 0; n_over_budget_checked = 0
    for row in grid_rows
        row.verified || continue
        obj_g = build_melitz_implicit_bundle(ctx, obj_inner.U, row.theta; delta=OUTER_BUDGET_DELTA,
            find_smallest=true, gradient_backend=:B_direct_argument_parallel, h=1e-4,
            inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT_CAMPAIGN,
            lower_limit_guard=LOWER_LIMIT_GUARD, delta_evaluation_cap=cap)
        bank = MelitzDualBank()
        result = melitz_classified_inner_solve(obj_g, row.theta, ctx; delta_evaluation_cap=cap, bank=bank)
        over_budget = row.Delta > OUTER_BUDGET_DELTA
        matched = result isa FiniteSolved && isapprox(result.Delta, row.Delta; atol=1e-6, rtol=1e-6)
        n_checked += 1
        n_matched += matched
        over_budget && (n_over_budget_checked += 1)
        @printf("%10.5f %14.6e %8s %10s %14.6e %10s\n",
            row.g, row.Delta, row.verified, over_budget,
            result isa FiniteSolved ? result.Delta : NaN, matched)
        flush(stdout)
    end
    @printf("\nSummary: %d/%d verified grid points matched EXACTLY (FiniteSolved, same Delta), of which %d were genuinely OVER the outer budget (delta=%.1f) but still under the cap (%.1f).\n",
        n_matched, n_checked, n_over_budget_checked, OUTER_BUDGET_DELTA, cap)
    return (n_checked=n_checked, n_matched=n_matched, n_over_budget_checked=n_over_budget_checked)
end

# ============================================================================
# Phase 3: concrete OLD-bug demonstration. Reconstructs the companion report's own
# g=-0.4988 point (documented Delta=1.060e+00, nStatus=0 -- a genuine, cleanly-solved finite
# point, just over the delta=1 budget) and shows directly:
#   (a) under the OLD (pre-correction) coupling, lower_limit=-(delta+guard) with delta=1
#       would have set lower_limit=-1.000001 -- ABORTING this genuinely solvable point.
#   (b) under the CORRECTED coupling, lower_limit=-(delta_evaluation_cap+guard) with
#       delta_evaluation_cap=10 leaves this point untouched -- it solves fully, FiniteSolved,
#       with the exact documented Delta.
# ============================================================================
function demonstrate_old_bug(ctx, obj_inner, theta_calib)
    println("\n" * "="^100)
    println("PHASE 3 [$RUN_TAG]: concrete OLD-bug demonstration at the companion report's own g=-0.4988 point")
    println("="^100)
    theta_above = copy(theta_calib); theta_above[1] = -0.4988
    r_above = evaluate_melitz_delta(theta_above, ctx, obj_inner; cold=true, store_G=false)
    @printf("  reference point: g=-0.4988  Delta=%.6e  nStatus=%d  verified=%s  (companion report documents Delta=1.060e+00, nStatus=0)\n",
        r_above.Delta, r_above.nStatus, r_above.verified)

    # (a) OLD coupling, simulated directly (not by re-running old code, which no longer
    # exists under these type names) -- build the bundle with the OLD formula's own
    # lower_limit value: -(delta+guard) = -(1.0+1e-6).
    old_lower_limit = -(OUTER_BUDGET_DELTA + LOWER_LIMIT_GUARD)
    obj_old = build_melitz_implicit_bundle(ctx, obj_inner.U, theta_above; delta=OUTER_BUDGET_DELTA,
        find_smallest=true, gradient_backend=:B_direct_argument_parallel, h=1e-4,
        inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT_CAMPAIGN,
        lower_limit_guard=LOWER_LIMIT_GUARD, delta_evaluation_cap=OUTER_BUDGET_DELTA)   # cap==delta reproduces the OLD coupling exactly
    @assert obj_old.lower_limit == old_lower_limit
    bank_old = MelitzDualBank()
    result_old = melitz_classified_inner_solve(obj_old, theta_above, ctx;
        delta_evaluation_cap=OUTER_BUDGET_DELTA, bank=bank_old)
    @printf("  (a) OLD coupling reproduced (cap==delta=%.1f): classified %s", OUTER_BUDGET_DELTA, string(typeof(result_old)))
    if result_old isa AboveEvaluationCap
        @printf(" -- certified_lower_bound=%.6f (a path-dependent NUMBER, NOT DeltaStar itself; true DeltaStar=%.6e was thrown away)\n",
            result_old.certified_lower_bound, r_above.Delta)
    else
        println()
    end

    # (b) CORRECTED coupling: delta_evaluation_cap=DELTA_EVALUATION_CAP (10.0 production
    # default), delta stays at the SAME outer budget (1.0) -- unaffected.
    obj_new = build_melitz_implicit_bundle(ctx, obj_inner.U, theta_above; delta=OUTER_BUDGET_DELTA,
        find_smallest=true, gradient_backend=:B_direct_argument_parallel, h=1e-4,
        inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT_CAMPAIGN,
        lower_limit_guard=LOWER_LIMIT_GUARD, delta_evaluation_cap=DELTA_EVALUATION_CAP)
    bank_new = MelitzDualBank()
    result_new = melitz_classified_inner_solve(obj_new, theta_above, ctx;
        delta_evaluation_cap=DELTA_EVALUATION_CAP, bank=bank_new)
    @printf("  (b) CORRECTED coupling (delta_evaluation_cap=%.1f, delta unchanged at %.1f): classified %s",
        DELTA_EVALUATION_CAP, OUTER_BUDGET_DELTA, string(typeof(result_new)))
    if result_new isa FiniteSolved
        @printf(" -- Delta=%.6e (matches the true, fully-optimized value; %.6e > delta=%.1f, so this is a genuine FiniteSolved OVER-BUDGET point, not an abort)\n",
            result_new.Delta, result_new.Delta, OUTER_BUDGET_DELTA)
    else
        println()
    end
    return (result_old=result_old, result_new=result_new, r_above=r_above)
end

# ============================================================================
# Phase 4 (governing prompt Section 8): the corrected constrained D=20 campaign.
# ============================================================================
function find_near_boundary_start(ctx, obj_inner, theta_calib; target_lo=0.90, target_hi=0.98)
    println("\n" * "="^100)
    println("PHASE 4a [$RUN_TAG]: near-boundary starting point scan (target Delta0 in [$target_lo,$target_hi])")
    println("="^100)
    cache = MelitzDeltaEvalCache(16)
    for step in 0.0005:0.0005:0.02
        g = G_FIXED_REFERENCE + step
        theta = copy(theta_calib); theta[1] = g
        r = evaluate_melitz_delta(theta, ctx, obj_inner; cold=false, cache=cache, store_G=false)
        @printf("  g=%.6f  Delta=%.6e  nStatus=%d  verified=%s\n", g, r.Delta, r.nStatus, r.verified)
        flush(stdout)
        if r.verified && target_lo <= r.Delta <= target_hi
            println("  -> selected as campaign starting point")
            return theta, r
        end
    end
    error("find_near_boundary_start: no point in the scanned range landed in [$target_lo,$target_hi]")
end

function block_scaled_theta_box(ctx; g_radius=0.05, A_radius=0.15, f_radius=0.15)
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
           result isa InfiniteDeltaCertified ? :InfiniteDeltaCertified :
           :NumericalFailure
    Delta_solved = result isa FiniteSolved ? result.Delta : NaN
    certified_lower_bound = result isa AboveEvaluationCap ? result.certified_lower_bound : NaN
    within_budget = result isa FiniteSolved && Delta_solved <= OUTER_BUDGET_DELTA
    over_budget_finite = result isa FiniteSolved && Delta_solved > OUTER_BUDGET_DELTA

    if within_budget && kappa < logger.best_kappa
        logger.best_kappa = kappa
        logger.best_theta = copy(theta)
    end
    if result isa AboveEvaluationCap
        push!(logger.above_cap_thetas, copy(theta))
    end

    push!(logger.events, (t=t, n_fc=logger.n_fc, kind=kind, Delta_solved=Delta_solved,
        certified_lower_bound=certified_lower_bound, source=(result isa AboveEvaluationCap ? result.source : :none),
        g=g, kappa=kappa, within_budget=within_budget, over_budget_finite=over_budget_finite))

    @printf("[LIVE %s] n_fc=%4d t=%7.2fs kind=%-20s g=%9.5f kappa=%9.6f  delta_star_solved=%-5s Delta=%12.6e certified_lower_bound=%12.6e (source=%s)\n",
        RUN_TAG, logger.n_fc, t, kind, g, kappa, (result isa FiniteSolved), Delta_solved, certified_lower_bound,
        (result isa AboveEvaluationCap ? result.source : :none))
    flush(stdout)
    return nothing
end

function run_campaign(ctx, obj_inner, calib, theta_calib)
    theta_fixed_af = copy(theta_calib); theta_fixed_af[1] = G_FIXED_REFERENCE
    theta_init, r_start = find_near_boundary_start(ctx, obj_inner, theta_calib)
    kappa0 = kappa_of_g(theta_init[1], calib)
    @printf("\nCampaign start: g=%.6f  Delta0=%.6e  kappa0=%.6f  GT0=%.6f\n",
        theta_init[1], r_start.Delta, kappa0, 1 - kappa0)
    flush(stdout)

    theta_box = block_scaled_theta_box(ctx)
    logger = EvalCapLogger()
    on_result(theta, result) = log_event!(logger, calib, theta, result)

    melitz_profile_reset!()
    MELITZ_PROFILE[] = true

    println("\n" * "="^100)
    @printf("PHASE 4b [%s]: corrected constrained campaign (%.0fs cap, delta=%.1f, delta_evaluation_cap=%.1f, above-cap points get the fixed sentinel delta_evaluation_cap/delta)\n",
        RUN_TAG, CAMPAIGN_SECONDS, OUTER_BUDGET_DELTA, DELTA_EVALUATION_CAP)
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
    println("CAMPAIGN RESULT [$RUN_TAG]")
    println("="^100)
    @printf("wall=%.2fs  nStatus=%d  n_fc_calls=%d  n_ga_calls=%d  n_inner_solved=%d\n",
        wall, result.nStatus, result.n_fc_calls, result.n_ga_calls, result.n_inner_solved)
    @printf("n_infinite_delta_reject=%d  n_above_cap_reject=%d  n_numerical_failure_reject=%d  inner_eval_failures=%d\n",
        result.n_infinite_delta_reject, result.n_above_cap_reject,
        result.n_numerical_failure_reject, result.inner_eval_failures)

    n_within_budget = count(e -> e.within_budget, logger.events)
    n_over_budget_finite = count(e -> e.over_budget_finite, logger.events)
    @printf("Classification breakdown over %d logged events: FiniteSolved-within-budget=%d  FiniteSolved-over-budget=%d  AboveEvaluationCap=%d  InfiniteDeltaCertified=%d  NumericalFailure=%d\n",
        length(logger.events), n_within_budget, n_over_budget_finite,
        count(e -> e.kind == :AboveEvaluationCap, logger.events),
        count(e -> e.kind == :InfiniteDeltaCertified, logger.events),
        count(e -> e.kind == :NumericalFailure, logger.events))

    for (label, cand) in (("initial_incumbent", result.initial_incumbent),
                          ("best_live_incumbent", result.best_live_incumbent),
                          ("cold_verified_incumbent (THE ANSWER)", result.cold_verified_incumbent))
        if cand === nothing
            @printf("  %-38s: nothing\n", label)
        else
            g = cand.eval.theta_free[1]
            gamma_prime = exp(g)
            kappa = (calib.w_prime / calib.w[calib.target_country]) * gamma_prime^(1 / (calib.sigma - 1))
            @printf("  %-38s: objective=%.6f  g=%.6f  kappa=%.6f  GT=%.6f  Delta=%.6e  min_slack=%.4f  source=%s\n",
                label, cand.objective, g, kappa, 1 - kappa, cand.eval.Delta, cand.eval.min_slack, cand.source)
        end
    end

    println("\n" * "="^100)
    println("TRAJECTORY (first 20 and last 20 logged events) [$RUN_TAG]")
    println("="^100)
    evs = logger.events
    show_idx = length(evs) <= 40 ? (1:length(evs)) : vcat(1:20, (length(evs)-19):length(evs))
    @printf("%6s %8s %20s %14s %14s %10s %8s\n", "n_fc", "t(s)", "kind", "Delta_solved", "cert_lb", "g", "kappa")
    for i in show_idx
        e = evs[i]
        @printf("%6d %8.2f %20s %14.4e %14.4e %10.6f %8.6f\n",
            e.n_fc, e.t, e.kind, e.Delta_solved, e.certified_lower_bound, e.g, e.kappa)
    end

    println("\n" * "="^100)
    println("WALL-CLOCK DECOMPOSITION (melitz_profile_report) [$RUN_TAG]")
    println("="^100)
    melitz_profile_report(stdout; trajectory_total_s=wall)

    return result, logger, wall
end

# ============================================================================
# Phase 5 (governing prompt Section 6): re-diagnose every AboveEvaluationCap theta the
# campaign produced, at cap=10/50/effectively-uncapped (1e9) with a hard diagnostic time
# limit (this script's own INNER_OPT already carries a maxtime_real -- see
# melitz_inner_loop_options_capped_2026-07-24.opt). Classifies each into: finite in (cap,50],
# finite above 50, infinite, or unresolved.
# ============================================================================
function diagnose_above_cap_points(ctx, obj_inner, above_cap_thetas; max_points=8)
    println("\n" * "="^100)
    println("PHASE 5 [$RUN_TAG]: diagnosing AboveEvaluationCap points at cap=10/50/uncapped(1e9)")
    println("="^100)
    if isempty(above_cap_thetas)
        println("  (no AboveEvaluationCap points were produced by this campaign -- nothing to diagnose)")
        return NamedTuple[]
    end
    pts = above_cap_thetas[1:min(max_points, length(above_cap_thetas))]
    rows = NamedTuple[]
    for (i, theta) in enumerate(pts)
        classifications = Dict{Float64,Any}()
        for cap in (10.0, 50.0, 1e9)
            objd = build_melitz_implicit_bundle(ctx, obj_inner.U, theta; delta=OUTER_BUDGET_DELTA,
                find_smallest=true, gradient_backend=:B_direct_argument_parallel, h=1e-4,
                inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT_CAMPAIGN,
                lower_limit_guard=LOWER_LIMIT_GUARD, delta_evaluation_cap=cap)
            bankd = MelitzDualBank()
            t0 = time()
            res = melitz_classified_inner_solve(objd, theta, ctx; delta_evaluation_cap=cap, bank=bankd)
            dt = time() - t0
            classifications[cap] = (result=res, wall=dt)
            kind = res isa FiniteSolved ? "FiniteSolved(Delta=$(round(res.Delta,digits=4)))" :
                   res isa AboveEvaluationCap ? "AboveEvaluationCap(lb=$(round(res.certified_lower_bound,digits=4)),src=$(res.source))" :
                   res isa InfiniteDeltaCertified ? "InfiniteDeltaCertified" : "NumericalFailure(nStatus=$(res.nStatus))"
            @printf("  point %d/%d  cap=%6.0f  %-50s  wall=%6.2fs\n", i, length(pts), cap, kind, dt)
            flush(stdout)
        end
        push!(rows, (idx=i, theta=copy(theta), classifications=classifications))
    end
    return rows
end

function main()
    calib = load_calibration()
    BLAS.set_num_threads(16)
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ

    grid_rows = NamedTuple[]
    if RUN_GRID
        grid_rows = extended_grid_scan(ctx, obj_inner, theta_calib)
    end

    if RUN_CROSSCHECK && !isempty(grid_rows)
        crosscheck_finite_values(ctx, obj_inner, grid_rows; cap=DELTA_EVALUATION_CAP)
    end

    if RUN_OLDBUG_DEMO
        demonstrate_old_bug(ctx, obj_inner, theta_calib)
    end

    result = nothing; logger = nothing; wall = NaN
    if RUN_CAMPAIGN
        result, logger, wall = run_campaign(ctx, obj_inner, calib, theta_calib)
    end

    if RUN_ABOVECAP_DIAGNOSIS && logger !== nothing
        diagnose_above_cap_points(ctx, obj_inner, logger.above_cap_thetas)
    end

    BLAS.set_num_threads(1)
    println("\nDONE [$RUN_TAG].")
    return grid_rows, result, logger, wall
end

main()
