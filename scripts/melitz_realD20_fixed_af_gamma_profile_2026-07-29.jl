# 2026-07-29: high-resolution real-D20 fixed-A/f gamma_d_prime profile, bidirectional
# (Frechet calibration -> theoretical gamma minimum, and calibration -> theoretical gamma
# maximum). Governing prompt: user's real-D20 fixed-A/f gamma profile campaign, continuing
# from the Melitz inner-solver architecture consolidation
# (docs/melitz_inner_solver_architecture_consolidation_2026-07-28.md) and post-consolidation
# validation (docs/melitz_post_consolidation_validation_2026-07-28.md).
#
# PHASE 0 (this session, scripts/melitz_phase0_realD20_endpoints_2026-07-29.jl) established,
# live at this exact fixture (real D=20, noah_D20, W=80,000, seed=1, focal=France, sigma=2.5):
#
#   g_pareto   = -0.4187707128   kappa_pareto = 0.9796971896   GT_pareto = 2.030281%
#   g_ceiling  = -0.5675172401   kappa_min    = 0.8872077596   GT_ceiling = 11.279224%
#                (branch A target: lambda_jj^(1/(sigma-1)), the paper's own proven GT upper
#                 bound -- a Delta->infinity OPEN LIMIT, confirmed live: InfiniteDeltaCertified
#                 at the exact point AND at t=0.995/0.999 approach points)
#   g_floor    = -0.3880030949   kappa_max    = 1.0            GT_floor = 0.0%
#                (branch B target: kappa_ratio<=1, the ordinary "gains from trade cannot be
#                 negative" bound -- an ORDINARY finite theoretical point, NOT a limit by
#                 theory, unlike branch A; found live to ALSO be InfiniteDeltaCertified in its
#                 immediate neighborhood at this fixture -- a genuine, reported, model-specific
#                 finding, not assumed and not used to justify moving the endpoint inward)
#
# NOTE on the 2026-07-24 session's own g_floor=0 convention
# (docs/melitz_real_d20_evaluation_cap_correction_2026-07-24.md Section 10.1): that session
# used gamma_prime_target=1 (g=0) as a convenient symmetric outer-search BOX EDGE for a search
# that only ever needed the LOWER bound (:upper-direction, g-decreasing) -- non-binding by
# their own construction, not a claim that g=0 is the true kappa_ratio<=1 theoretical ceiling.
# At this fixture wage_ratio=1.2952 != 1, so kappa_of_g(0)=wage_ratio=1.2952 > 1 -- g=0 is
# actually PAST the true kappa_max=1 boundary (which sits at g_floor=-0.3880). This session
# uses the freshly-derived, cross-checked g_floor=-0.3880 (kappa_max=1) as the true
# theoretical maximum, per this session's own "derive analytically, cross-check via the live
# kappa/gamma relationship" mandate -- not the 2026-07-24 session's non-binding convenience
# value.
#
# Grid construction (Phase 1): t_i=(i/40)^2, i=0..40 (quadratic -- denser near calibration,
# the economically relevant region), kappa interpolated LINEARLY between the calibration and
# each branch's own theoretical kappa target, mapped back to g via the closed-form
# g_of_kappa. Branch A's literal i=40 endpoint (t=1.0, the established open Delta->infinity
# limit) is REPLACED by three explicit near-limit points t=0.99/0.995/0.999 (Phase 1's own
# "open or singular endpoints" instruction) -- branch A therefore has 40+3=43 evaluated grid
# points. Branch B's literal i=40 endpoint is KEPT (ordinary finite theoretical point by
# theory, even though numerically hard) -- 41 evaluated grid points. Total unique predetermined
# points = 43+41-1 (calibration counted once) = 83, exceeding the required minimum of 81.
#
# Policy: CappedEvaluation(10.0) throughout (raw KNITRO lower_limit=-10.0), the consolidated
# public API (solve_melitz_delta! on a MelitzInnerSession) exclusively -- never a legacy
# direct call. TWO INDEPENDENT sessions/fixtures (own obj/ctx/bank), one per branch, so no
# warm-start state or KNITRO instance is ever shared across branches (this repo's own
# MelitzInnerSession docstring, Section on "never share a session/bank across...").
# Continuation warm-start is IMPLEMENTED EXPLICITLY (not merely via warm_start_source=:previous
# left as a no-op): `last_good_x` is updated ONLY on a FiniteSolved result and is manually
# copied into obj.x/obj.use_cached_x immediately BEFORE every solve attempt -- guarantees no
# point is ever warm-started from an AboveEvaluationCap/InfiniteDeltaCertified/NumericalFailure
# attempt's leftover KNITRO state, per the governing prompt's own explicit prohibition.
#
# Usage: julia --project=. -t 20 scripts/melitz_realD20_fixed_af_gamma_profile_2026-07-29.jl

using Pkg
REPO = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
Pkg.activate(REPO)
using Printf, DelimitedFiles, LinearAlgebra, Dates
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
include(joinpath(REPO, "scripts", "melitz_regression_fixtures_2026-07-28.jl"))
using KNITRO

const OUTDIR = joinpath(REPO, "docs", "key_results")
mkpath(OUTDIR)
const RESDIR = joinpath(REPO, "results")
mkpath(RESDIR)

println("Julia threads: ", Threads.nthreads(), "   BLAS threads: ", BLAS.get_num_threads())
BLAS.set_num_threads(1)
flush(stdout)

const CAP = 10.0
const POLICY = CappedEvaluation(CAP)

kappa_of_g(g::Real, wratio::Real, sigma::Real) = wratio * exp(g)^(1 / (sigma - 1))
g_of_kappa(kappa::Real, wratio::Real, sigma::Real) = log((kappa / wratio)^(sigma - 1))

function wage_ratio_and_lambda_jj(theta0, ctx)
    state = melitz_outer_state(theta0, ctx)
    j = ctx.target_country
    lambda_jj = state.equilibrium.trade_flow[j, j] / state.equilibrium.expenditure[j]
    wratio = 1.0 / ctx.w[j]
    return wratio, lambda_jj
end

# ============================================================================
# Row schema
# ============================================================================
mutable struct ProfileRow
    branch::String
    grid_index::Int
    fraction_to_endpoint::Float64
    gamma_d_prime::Float64
    kappa_ratio::Float64
    GT_pct::Float64
    A_hash::UInt64
    f_hash::UInt64
    classification::String
    delta_star::Float64
    certified_lower_bound::Float64
    exact_infeasibility_kind::String
    exact_infeasibility_block::Int
    policy::String
    evaluation_cap::Float64
    raw_lower_limit::Float64
    solver_status::Int
    iterations::Int
    wall_seconds::Float64
    warm_start_source::String
    primal_dual_gap::Float64
    normalization_residual::Float64
    weighted_moment_residual::Float64
    kkt_residual::Float64
    primal_divergence::Float64
    dual_divergence::Float64
    min_slack::Float64
    gravity_residual_A::Float64
    gravity_residual_f::Float64
    min_density_ratio::Float64
    max_density_ratio::Float64
    cold_replay_status::String
end

function write_csv(path, rows::Vector{ProfileRow})
    open(path, "w") do io
        println(io, join(["branch","grid_index","fraction_to_theoretical_endpoint","gamma_d_prime",
            "kappa_ratio","gains_from_trade_pct","A_hash","f_hash","classification","delta_star",
            "certified_lower_bound","exact_infeasibility_kind","exact_infeasibility_block","policy",
            "evaluation_cap","raw_lower_limit","solver_status","iterations","wall_seconds",
            "warm_start_source","primal_dual_gap","normalization_residual","weighted_moment_residual",
            "kkt_residual","primal_divergence","dual_divergence","min_slack","gravity_residual_A",
            "gravity_residual_f","min_density_ratio","max_density_ratio","cold_replay_status"], ","))
        for r in rows
            @printf(io, "%s,%d,%.8f,%.10f,%.10f,%.8f,%s,%s,%s,%.8e,%.8e,%s,%d,%s,%.4f,%.4f,%d,%d,%.4f,%s,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%s\n",
                r.branch, r.grid_index, r.fraction_to_endpoint, r.gamma_d_prime, r.kappa_ratio, r.GT_pct,
                string(r.A_hash, base=16), string(r.f_hash, base=16), r.classification, r.delta_star,
                r.certified_lower_bound, r.exact_infeasibility_kind, r.exact_infeasibility_block,
                replace(r.policy, "," => ";"), r.evaluation_cap, r.raw_lower_limit, r.solver_status, r.iterations, r.wall_seconds,
                r.warm_start_source, r.primal_dual_gap, r.normalization_residual, r.weighted_moment_residual,
                r.kkt_residual, r.primal_divergence, r.dual_divergence, r.min_slack, r.gravity_residual_A,
                r.gravity_residual_f, r.min_density_ratio, r.max_density_ratio, r.cold_replay_status)
        end
    end
end

# ============================================================================
# One solve + full verification, returns a ProfileRow
# ============================================================================
function solve_and_record(session, theta0, theta, ctx, sigma, wratio, branch::String, grid_index::Int,
                           frac::Float64, A_hash::UInt64, f_hash::UInt64, warm_label::String;
                           origin_block_screen::Bool=true)
    obj = session.obj
    c0 = melitz_backend_counters_snapshot()
    t0 = time()
    r = solve_melitz_delta!(session, theta, POLICY; origin_block_screen=origin_block_screen,
                             warm_start_source=:previous)
    wall = time() - t0
    c1 = melitz_backend_counters_snapshot()
    iters = c1.matrix_free_objective_calls - c0.matrix_free_objective_calls

    g = theta[1]
    kappa = kappa_of_g(g, wratio, sigma)
    GT = 1 - kappa

    row = ProfileRow(branch, grid_index, frac, exp(g), kappa, 100*GT, A_hash, f_hash,
        "NumericalFailure", NaN, NaN, "", -1, string(POLICY), CAP, obj.lower_limit, -1, iters, wall,
        warm_label, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, "not_replayed")

    if r isa FiniteSolved
        ev = evaluate_melitz_delta_from_solution(theta, ctx, obj, r.Delta, r.x, r.nStatus; store_G=false)
        row.classification = "FiniteSolved"
        row.delta_star = r.Delta
        row.solver_status = r.nStatus
        row.primal_dual_gap = ev.primal_dual_gap
        row.weighted_moment_residual = maximum(abs, ev.moment_residuals)
        row.kkt_residual = max(abs(ev.kkt_opt_error), abs(ev.kkt_feas_error))
        row.primal_divergence = ev.primal_divergence
        row.dual_divergence = ev.dual_divergence
        row.min_slack = ev.min_slack
        if ev.equilibrium_check !== nothing
            row.gravity_residual_A = ev.equilibrium_check.gravity_residual_A
            row.gravity_residual_f = ev.equilibrium_check.gravity_residual_f
        end
        W = length(ev.weights)
        row.min_density_ratio = minimum(ev.weights) * W
        row.max_density_ratio = maximum(ev.weights) * W
        row.normalization_residual = abs(sum(ev.weights) - 1.0)
        @printf("  [%s idx=%3d frac=%.6f] g=%9.5f kappa=%.6f GT=%7.4f%%  FiniteSolved  Delta=%.4e  nStatus=%d  wall=%.2fs\n",
            branch, grid_index, frac, g, kappa, 100*GT, r.Delta, r.nStatus, wall)
    elseif r isa AboveEvaluationCap
        row.classification = "AboveEvaluationCap"
        row.certified_lower_bound = r.certified_lower_bound
        row.exact_infeasibility_kind = string(r.source)
        @printf("  [%s idx=%3d frac=%.6f] g=%9.5f kappa=%.6f GT=%7.4f%%  AboveEvaluationCap  lb=%.4e  source=%s  wall=%.2fs\n",
            branch, grid_index, frac, g, kappa, 100*GT, r.certified_lower_bound, r.source, wall)
    elseif r isa InfiniteDeltaCertified
        row.classification = "InfiniteDeltaCertified"
        row.exact_infeasibility_kind = string(r.kind)
        row.exact_infeasibility_block = r.column
        @printf("  [%s idx=%3d frac=%.6f] g=%9.5f kappa=%.6f GT=%7.4f%%  InfiniteDeltaCertified  col=%d kind=%s  wall=%.2fs\n",
            branch, grid_index, frac, g, kappa, 100*GT, r.column, r.kind, wall)
    else # NumericalFailure
        row.solver_status = r.nStatus
        @printf("  [%s idx=%3d frac=%.6f] g=%9.5f kappa=%.6f GT=%7.4f%%  NumericalFailure  nStatus=%d  wall=%.2fs\n",
            branch, grid_index, frac, g, kappa, 100*GT, r.nStatus, wall)
    end
    flush(stdout)
    return row, r
end

# ============================================================================
# Branch runner with explicit last-good-x continuation
# ============================================================================
function run_branch(session, theta0, ctx, sigma, wratio, branch::String, ts::Vector{Float64},
                     kappa_target::Float64, kappa_pareto::Float64, A_hash::UInt64, f_hash::UInt64)
    obj = session.obj
    rows = ProfileRow[]
    last_good_x = copy(obj.x)
    last_good_idx = 0
    nA = ctx.D^2 - 1
    for (i, t) in enumerate(ts)
        g = t == 0.0 ? theta0[1] : g_of_kappa(kappa_pareto + t*(kappa_target - kappa_pareto), wratio, sigma)
        theta = copy(theta0); theta[1] = g
        @assert theta[2:end] == theta0[2:end] "free A/f coordinates must never be perturbed"
        @assert hash(theta[2:1+nA]) == A_hash "A_free hash drifted from calibration"
        @assert hash(theta[2+nA:end]) == f_hash "f_free hash drifted from calibration"

        obj.x .= last_good_x
        obj.use_cached_x = true
        warm_label = last_good_idx == 0 ? "calibration_seed" : "previous_finite_idx$(last_good_idx)"

        row, r = solve_and_record(session, theta0, theta, ctx, sigma, wratio, branch, i-1, t, A_hash, f_hash, warm_label)
        push!(rows, row)
        if r isa FiniteSolved
            last_good_x = copy(r.x)
            last_good_idx = i - 1
        end
    end
    return rows
end

function main()
    println("="^100)
    println("PHASE 0/1: fixtures, endpoints, grid construction")
    println("="^100)

    fixA = build_realD20_fixture(; W=80_000, seed=1, policy=POLICY)
    fixB = build_realD20_fixture(; W=80_000, seed=1, policy=POLICY)
    @assert fixA.theta0 == fixB.theta0 "two independent fixtures must reproduce the identical calibration"
    theta0 = fixA.theta0
    ctxA, objA = fixA.ctx, fixA.obj
    ctxB, objB = fixB.ctx, fixB.obj
    sigma = ctxA.sigma

    wratio, lambda_jj = wage_ratio_and_lambda_jj(theta0, ctxA)
    g_pareto = theta0[1]
    kappa_pareto = kappa_of_g(g_pareto, wratio, sigma)

    kappa_min = lambda_jj^(1 / (sigma - 1))
    g_ceiling = g_of_kappa(kappa_min, wratio, sigma)
    kappa_max = 1.0
    g_floor = g_of_kappa(kappa_max, wratio, sigma)

    @printf("g_pareto=%.10f kappa_pareto=%.10f GT_pareto=%.6f%%\n", g_pareto, kappa_pareto, 100*(1-kappa_pareto))
    @printf("branch A target: kappa_min=%.10f g_ceiling=%.10f GT_ceiling=%.6f%% (OPEN Delta->inf limit)\n",
        kappa_min, g_ceiling, 100*(1-kappa_min))
    @printf("branch B target: kappa_max=%.10f g_floor=%.10f GT_floor=%.6f%% (ordinary finite theoretical point)\n",
        kappa_max, g_floor, 100*(1-kappa_max))
    flush(stdout)

    nA = ctxA.D^2 - 1
    A_hash = hash(theta0[2:1+nA])
    f_hash = hash(theta0[2+nA:end])
    @printf("A_hash=%s  f_hash=%s  (must be IDENTICAL at every grid point on both branches)\n",
        string(A_hash, base=16), string(f_hash, base=16))

    ts_base = [(i/40)^2 for i in 0:40]
    ts_A = vcat(ts_base[1:40], [0.99, 0.995, 0.999])   # drop literal t=1.0 (open limit); 43 points
    ts_B = ts_base                                      # keep literal t=1.0 (ordinary finite point); 41 points
    @printf("branch A: %d grid points  branch B: %d grid points  total unique (calib shared once) = %d\n",
        length(ts_A), length(ts_B), length(ts_A) + length(ts_B) - 1)
    flush(stdout)

    sessionA = MelitzInnerSession(objA, ctxA, POLICY)
    sessionB = MelitzInnerSession(objB, ctxB, POLICY)

    println("\n" * "="^100)
    println("PHASE 2/3: branch A (calibration -> theoretical minimum gamma_d_prime)")
    println("="^100)
    rowsA = run_branch(sessionA, theta0, ctxA, sigma, wratio, "A_toward_min_gamma", ts_A, kappa_min, kappa_pareto, A_hash, f_hash)
    write_csv(joinpath(OUTDIR, "melitz_realD20_fixed_af_gamma_profile_branchA_2026-07-29.csv"), rowsA)

    println("\n" * "="^100)
    println("PHASE 2/3: branch B (calibration -> theoretical maximum gamma_d_prime)")
    println("="^100)
    rowsB = run_branch(sessionB, theta0, ctxB, sigma, wratio, "B_toward_max_gamma", ts_B, kappa_max, kappa_pareto, A_hash, f_hash)
    write_csv(joinpath(OUTDIR, "melitz_realD20_fixed_af_gamma_profile_branchB_2026-07-29.csv"), rowsB)

    all_rows = vcat(rowsA, rowsB)
    write_csv(joinpath(RESDIR, "melitz_realD20_fixed_Af_gamma_profile_2026-07-29.csv"), all_rows)
    println("\nMain grid CSV written: results/melitz_realD20_fixed_Af_gamma_profile_2026-07-29.csv")
    flush(stdout)

    return (fixA=fixA, fixB=fixB, sessionA=sessionA, sessionB=sessionB, theta0=theta0, sigma=sigma,
            wratio=wratio, lambda_jj=lambda_jj, g_pareto=g_pareto, kappa_pareto=kappa_pareto,
            kappa_min=kappa_min, g_ceiling=g_ceiling, kappa_max=kappa_max, g_floor=g_floor,
            A_hash=A_hash, f_hash=f_hash, rowsA=rowsA, rowsB=rowsB, all_rows=all_rows)
end

RESULT = main()
println("\nPHASE 1-3 DONE. Rows: ", length(RESULT.all_rows))
