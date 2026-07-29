# Reduced-q-subspace outer-search backend (2026-07-29 continuation session), Phase 12:
# bounded D4 outer comparison. A SMOKE TEST, not a frontier campaign (governing prompt's own
# framing) -- identical starting economic state, identical evaluation cap, comparable bounded
# runtime/evaluation limits across THREE backends:
#
#   1. production (A,f)                       -- gradient_backend=:auto, :logf
#   2. full-coordinate experimental (A,q)      -- gradient_backend=:B_direct_argument_aq_experimental, :logcutoff
#   3. NEW sequential reduced-q-subspace       -- melitz_run_reduced_q_sequential_search, :logcutoff
#
# DISCLOSED (matching the established precedent in
# docs/melitz_aq_q_bandwidth_convergence_2026-07-29.md's own Phase 12): D4's fixed-A/f
# gamma-profile corridor tops out at Delta~0.572 (Phase 3 of that session), so
# delta in {1,2} would never bind on this corridor (the divergence constraint is trivially
# slack at every reachable point) -- not independently informative for a SHORT smoke test.
# Budgets tested here are 0.1 and 0.5 (both genuinely binding/near-binding), matching the
# prior session's own disclosed scope reduction, not a new gap.
#
# CORRECTED reporting convention (governing prompt's own correction #2): "better"/"more
# extreme" incumbent comparison uses `melitz_reduced_q_more_extreme_kappa` (`reduced_q_controller.jl`)
# -- upper bound: SMALLER kappa is better; lower bound: LARGER kappa is better -- never the
# informal "higher kappa is better" narrative the prior session's own Phase 12 report text
# used (backwards for the :upper case, see this session's own report doc).

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles
println("Julia threads: ", Threads.nthreads()); flush(stdout)

const OUTDIR = joinpath(REPO, "docs", "key_results")
inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
outer_opt = joinpath(REPO, "melitz_outer_finite_delta_alg_direct_2026-07-27.opt")

data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)

results = NamedTuple[]

function run_production_or_full_aq(label::AbstractString, gradient_backend::Symbol, outer_parameterization::Symbol,
                                    delta::Real, direction::Symbol)
    obj, theta0 = build_melitz_psi_bundle(data; outer_parameterization=outer_parameterization,
        policy=CappedEvaluation(10.0), backend=:matrix_free, forbid_dense_fallback=true)
    ctx = obj.γ
    t0 = time()
    res = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=delta, direction=direction,
        gradient_backend=gradient_backend, backend=:matrix_free,
        policy=CappedEvaluation(10.0), inner_loop_opt=inner_opt, outer_loop_opt=outer_opt)
    wall = time() - t0
    inc = res.cold_verified_incumbent
    g_best = inc === nothing ? theta0[1] : inc.eval.theta_free[1]
    kappa_best = kappa_ratio_of_g(g_best, ctx)
    delta_best = inc === nothing ? NaN : inc.eval.Delta
    gt0 = 100 * (1 - kappa_ratio_of_g(theta0[1], ctx))
    gt_best = 100 * (1 - kappa_best)
    @printf("[%-28s] delta=%.1f dir=%s  nStatus=%d  wall=%.1fs  inner_solves=%d (fail=%d)\n",
            label, delta, direction, res.nStatus, wall, res.inner_solve_count, res.inner_eval_failures)
    @printf("    start GT%%=%.4f  best GT%%=%.4f  Delta_best=%.6e  within_budget=%s  n_stages=1(single KNITRO run)  n_FiniteSolved=%d n_AboveCap=%d n_Infinite=%d n_Fail=%d\n",
            gt0, gt_best, delta_best, string(!isnan(delta_best) && delta_best<=delta),
            res.n_inner_solved, res.n_above_cap_reject, res.n_infinite_delta_reject, res.n_numerical_failure_reject)
    flush(stdout)
    push!(results, (label=label, delta=delta, direction=direction, nStatus=res.nStatus, wall_s=wall,
                     start_gt_pct=gt0, best_gt_pct=gt_best, kappa_best=kappa_best, delta_best=delta_best,
                     within_budget=(!isnan(delta_best) && delta_best<=delta),
                     n_accepted_stages=1, n_accepted_steps=res.inner_solve_count,
                     n_finite_solved=res.n_inner_solved, n_above_cap=res.n_above_cap_reject,
                     n_infinite_delta=res.n_infinite_delta_reject, n_numerical_failure=res.n_numerical_failure_reject,
                     n_screened=0, wall_inner_s=NaN, termination_status=string(res.nStatus)))
end

function run_reduced_q(label::AbstractString, delta::Real, direction::Symbol)
    obj, theta0 = build_melitz_psi_bundle(data; outer_parameterization=:logcutoff,
        policy=CappedEvaluation(10.0), backend=:matrix_free, forbid_dense_fallback=true)
    ctx = obj.γ
    t0 = time()
    result = melitz_run_reduced_q_sequential_search(ctx, obj, theta0; delta=delta, direction=direction,
        policy=CappedEvaluation(10.0), n_stages_max=5, max_iterations_per_stage=40, max_seconds_per_stage=120.0,
        outer_loop_opt=outer_opt)
    wall = time() - t0
    kappa_best = kappa_ratio_of_g(result.incumbent.objective * (direction==:upper ? 1.0 : -1.0), ctx)
    gt0 = 100 * (1 - kappa_ratio_of_g(theta0[1], ctx))
    gt_best = 100 * (1 - kappa_best)
    n_finite = sum(s.n_finite_solved for s in result.stages)
    n_above = sum(s.n_above_cap for s in result.stages)
    n_inf = sum(s.n_infinite_delta for s in result.stages)
    n_fail = sum(s.n_numerical_failure for s in result.stages)
    n_screened = sum(s.n_cap_screened for s in result.stages)
    n_steps = sum(length(s.trials) for s in result.stages)
    # Disclosed: per-trial one-sided predicted-sign accounting (governing prompt Phase 12's
    # own "percentage of proposed q moves with correct-sign prediction" metric) is reported
    # directly by the Phase 11 shakedown script instead of being re-derived per KNITRO trial
    # here -- this smoke test focuses on incumbent/classification/timing outcomes.
    @printf("[%-28s] delta=%.1f dir=%s  stopped=%s  wall=%.1fs  n_stages=%d  n_steps=%d\n",
            label, delta, direction, result.stopped_reason, wall, length(result.stages), n_steps)
    @printf("    start GT%%=%.4f  best GT%%=%.4f  Delta_best=%.6e  within_budget=%s  n_FiniteSolved=%d n_AboveCap=%d n_Infinite=%d n_Fail=%d n_screened=%d\n",
            gt0, gt_best, result.incumbent.Delta, string(result.incumbent.Delta<=delta),
            n_finite, n_above, n_inf, n_fail, n_screened)
    flush(stdout)
    push!(results, (label=label, delta=delta, direction=direction, nStatus=-1, wall_s=wall,
                     start_gt_pct=gt0, best_gt_pct=gt_best, kappa_best=kappa_best, delta_best=result.incumbent.Delta,
                     within_budget=(result.incumbent.Delta<=delta),
                     n_accepted_stages=length(result.stages), n_accepted_steps=n_steps,
                     n_finite_solved=n_finite, n_above_cap=n_above, n_infinite_delta=n_inf,
                     n_numerical_failure=n_fail, n_screened=n_screened, wall_inner_s=NaN,
                     termination_status=string(result.stopped_reason)))
end

for delta in (0.1, 0.5), direction in (:upper, :lower)
    run_production_or_full_aq("production_(A,f)", :auto, :logf, delta, direction)
    run_production_or_full_aq("full_experimental_(A,q)", :B_direct_argument_aq_experimental, :logcutoff, delta, direction)
    run_reduced_q("sequential_reduced_q", delta, direction)
end

open(joinpath(OUTDIR, "melitz_reducedq_phase12_d4_comparison_2026-07-29.csv"), "w") do io
    println(io, "label,delta,direction,nStatus,wall_s,start_gt_pct,best_gt_pct,kappa_best,delta_best,within_budget,n_accepted_stages,n_accepted_steps,n_finite_solved,n_above_cap,n_infinite_delta,n_numerical_failure,n_screened,termination_status")
    for r in results
        println(io, join([r.label, r.delta, r.direction, r.nStatus, r.wall_s, r.start_gt_pct, r.best_gt_pct,
                           r.kappa_best, r.delta_best, r.within_budget, r.n_accepted_stages, r.n_accepted_steps,
                           r.n_finite_solved, r.n_above_cap, r.n_infinite_delta, r.n_numerical_failure,
                           r.n_screened, r.termination_status], ","))
    end
end

println("\n" * "="^100)
println("CORRECTED most-extreme-incumbent comparison per (delta,direction), using melitz_reduced_q_more_extreme_kappa:")
for delta in (0.1, 0.5), direction in (:upper, :lower)
    rows = filter(r -> r.delta==delta && r.direction==direction, results)
    best = rows[1]
    for r in rows[2:end]
        if melitz_reduced_q_more_extreme_kappa(direction, r.kappa_best, best.kappa_best)
            best = r
        end
    end
    @printf("  delta=%.1f dir=%-6s -> MOST EXTREME: %-28s  GT%%=%.4f  kappa=%.6f\n", delta, direction, best.label, best.best_gt_pct, best.kappa_best)
end
println("\nPhase 12 comparison complete. Rows: ", length(results))
flush(stdout)
