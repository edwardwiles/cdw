# Gate 2 (2026-07-29 reduced-q validation session): matched-effort D4 ablation at delta=0.5,
# upper and lower, comparing Method A (welfare + exact A, q frozen), Method B (sequential
# reduced q), Method C (sequential production (A,f)) from a COMMON verified starting state.
#
# Common starting state: Gate 1 already established (this session's own
# `melitz_gate1_d4_w_replay_2026-07-29.csv`) that the reduced-q Phase-12 delta=0.5 upper/lower
# incumbents (`theta_full`, `:logcutoff` space) are `FiniteSolved` and within budget at
# W=80,000 -- exactly the governing prompt's own required check ("if the previous W=20,000
# reduced winner is not within budget at W=80,000, do not use it... use the best common
# verified incumbent available"). Both ARE within budget, so both are used as the common start
# for every method at this cell -- the "best common verified incumbent available before the
# competing searches begin" is literally this session's own already-completed Gate 1/Phase 12
# result. For Method C (native `:logf`), the IDENTICAL economic point is reconstructed via
# `melitz_expand_theta`/`reduce_to_free_theta` (round-trip verified to machine precision, not
# assumed) -- both methods start from the same displaced economic state, not merely "close".
#
# Matched effort (governing prompt "Matched effort" section): n_stages_max=5,
# max_iterations_per_stage=40 (KNITRO major-iteration cap, the SAME per-stage effort knob this
# codebase's own Phase 12 precedent uses -- a disclosed, not hidden, stand-in for an exact
# "100 classified evaluations" hard mid-solve cutoff, which this codebase's KNITRO wiring has
# no existing mechanism to enforce mid-solve without risking correctness of the callback
# control flow; the realized per-stage classified-evaluation counts are reported directly so
# the reader can see how close the realized effort landed to the nominal target), stop after 2
# consecutive no-improvement stages, and a POST-STAGE check that skips starting a new stage
# once cumulative classified evaluations already reached 500.

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Printf, Serialization, LinearAlgebra
println("Julia threads: ", Threads.nthreads()); flush(stdout)

const OUTDIR = joinpath(REPO, "docs", "key_results")
const CAP = 10.0
const POLICY = CappedEvaluation(CAP)
const BASE_SEED = 29
const W_GATE2 = 80_000
const N_STAGES_MAX = 5
const MAX_ITER_PER_STAGE = 40
const MAX_SECONDS_PER_STAGE = 120.0
const MAX_EVALS_TOTAL = 500
inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
outer_opt = joinpath(REPO, "melitz_outer_finite_delta_alg_direct_2026-07-27.opt")

ckpt_path = joinpath(OUTDIR, "melitz_reducedq_phase12_incumbent_checkpoints_2026-07-29.jls")
checkpoints = open(deserialize, ckpt_path)

data0 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=BASE_SEED, W=20_000)
z80 = pareto_draws(W_GATE2, data0.primitives.D, data0.primitives.theta_star; seed=BASE_SEED, mode=:halton)
data80 = MelitzSyntheticData(data0.primitives, data0.equilibrium, data0.counterfactual, data0.L, z80, data0.seed)

obj_q, _ = build_melitz_psi_bundle(data80; outer_parameterization=:logcutoff,
    policy=POLICY, backend=:matrix_free, forbid_dense_fallback=true)
ctx_q = obj_q.γ
obj_f, _ = build_melitz_psi_bundle(data80; outer_parameterization=:logf,
    policy=POLICY, backend=:matrix_free, forbid_dense_fallback=true)
ctx_f = obj_f.γ

function common_start(delta::Float64, direction::Symbol)
    cp = checkpoints[(delta, direction)]
    theta_q = collect(Float64.(cp.theta_full))
    obj_q.use_cached_x = false; obj_q.x .= NaN
    lfd_q = melitz_recover_lfd(obj_q, theta_q)
    @assert lfd_q.lfd_ok && lfd_q.Delta <= delta "common start not FiniteSolved/within-budget at W=80,000 for delta=$delta dir=$direction -- Gate 1 should have caught this"

    A, f, gamma_prime_j, f_jj = melitz_expand_theta(theta_q, ctx_q)
    p = MelitzPrimitives(ctx_f.D, ctx_f.sigma, ctx_f.theta_star, ctx_f.target_country, ctx_f.tau, ctx_f.w, A, f, gamma_prime_j)
    theta_f = melitz_reduce_theta(p, ctx_f)
    obj_f.use_cached_x = false; obj_f.x .= NaN
    lfd_f = melitz_recover_lfd(obj_f, theta_f)
    @assert lfd_f.lfd_ok "round-tripped :logf point failed to verify"
    reldiff = abs(lfd_f.Delta - lfd_q.Delta) / max(abs(lfd_q.Delta), 1e-12)
    @printf("common_start delta=%.1f dir=%s: Delta(:logcutoff)=%.8f Delta(:logf)=%.8f reldiff=%.2e\n",
            delta, direction, lfd_q.Delta, lfd_f.Delta, reldiff)
    @assert reldiff < 1e-6 "cross-parameterization round-trip mismatch exceeds tolerance -- NOT the same economic point"
    flush(stdout)
    return theta_q, theta_f, lfd_q.Delta
end

kappa_gt(g, ctx) = 100 * (1 - kappa_ratio_of_g(g, ctx))

stage_rows = NamedTuple[]
summary_rows = NamedTuple[]

function record_stage_rows!(method::String, delta::Float64, direction::Symbol, search_result, anchor_gt0::Float64)
    cum = 0
    for (i, sr) in enumerate(search_result.stages)
        counters = melitz_typed_counters_from_reduced_q_stage(sr)
        melitz_validate_typed_counters(counters; n_trials=length(sr.trials))
        cum += length(sr.trials)
        best_gt = sr.best_incumbent === nothing ? NaN : kappa_gt(sr.best_incumbent.objective * (direction==:upper ? 1.0 : -1.0), ctx_q)
        push!(stage_rows, (method=method, delta=delta, direction=string(direction), stage_id=i,
              anchor_gt_pct=anchor_gt0, n_trials=length(sr.trials), n_finite_solved=counters.n_finite_solved,
              n_above_cap=counters.n_above_cap_evaluated, n_screened=counters.n_screened_above_cap,
              n_infinite_delta=counters.n_infinite_delta, n_numerical_failure=counters.n_numerical_failure,
              best_incumbent_gt_pct=best_gt, cumulative_evals=cum, wall_s=sr.wall_s,
              q_basis_norm=norm(sr.stage.q_basis_free), stage_fingerprint=sr.stage.fingerprint))
    end
end

function run_cell(delta::Float64, direction::Symbol)
    theta_q0, theta_f0, Delta0 = common_start(delta, direction)
    # BUG FIX (found post-hoc, disclosed in the main report doc's Gate 2 section): theta_q0[1]
    # is the RAW welfare coordinate g, not a direction-signed optimization objective -- it must
    # NOT be multiplied by the direction sign (that sign convention applies only to the
    # already-encoded `.objective` fields elsewhere in this script, e.g. `resA.incumbent.objective`,
    # which correctly get sign-flipped back to raw g by the same multiplication). The original
    # version of this line incorrectly applied that same sign flip to theta_q0[1] itself,
    # reporting a wrong "start GT%" for direction=:lower (7.49% instead of the true 0.0562%,
    # since the common start already IS the extreme incumbent for that cell -- see the report's
    # own Gate 2 section for the corrected numbers and interpretation). This does NOT affect
    # `gtA`/`gtB`/`gtC` (all `.objective`-derived) or the matched-effort winner determination.
    gt0 = kappa_gt(theta_q0[1], ctx_q)

    println("\n" * "="^100)
    @printf("Gate 2 cell: delta=%.1f direction=%s  start GT%%=%.4f Delta0=%.6f\n", delta, direction, gt0, Delta0)
    println("="^100); flush(stdout)

    # ---- Method A: welfare + exact A, q frozen ----
    tA0 = time()
    resA = melitz_run_welfare_plus_a_sequential_search(ctx_q, obj_q, theta_q0; delta=delta, direction=direction,
        policy=POLICY, n_stages_max=N_STAGES_MAX, max_iterations_per_stage=MAX_ITER_PER_STAGE,
        max_seconds_per_stage=MAX_SECONDS_PER_STAGE, outer_loop_opt=outer_opt)
    wallA = time() - tA0
    record_stage_rows!("Method_A_welfare_plus_A", delta, direction, resA, gt0)
    gtA = kappa_gt(resA.incumbent.objective * (direction==:upper ? 1.0 : -1.0), ctx_q)
    nA_total = sum(length(s.trials) for s in resA.stages)
    @printf("Method A (welfare+exact-A, q frozen): stages=%d evals=%d wall=%.1fs stopped=%s  GT%%: %.4f -> %.4f\n",
            length(resA.stages), nA_total, wallA, resA.stopped_reason, gt0, gtA)
    flush(stdout)

    # ---- Method B: sequential reduced q ----
    tB0 = time()
    resB = melitz_run_reduced_q_sequential_search(ctx_q, obj_q, theta_q0; delta=delta, direction=direction,
        policy=POLICY, n_stages_max=N_STAGES_MAX, max_iterations_per_stage=MAX_ITER_PER_STAGE,
        max_seconds_per_stage=MAX_SECONDS_PER_STAGE, outer_loop_opt=outer_opt)
    wallB = time() - tB0
    record_stage_rows!("Method_B_reduced_q", delta, direction, resB, gt0)
    gtB = kappa_gt(resB.incumbent.objective * (direction==:upper ? 1.0 : -1.0), ctx_q)
    nB_total = sum(length(s.trials) for s in resB.stages)
    @printf("Method B (sequential reduced q): stages=%d evals=%d wall=%.1fs stopped=%s  GT%%: %.4f -> %.4f\n",
            length(resB.stages), nB_total, wallB, resB.stopped_reason, gt0, gtB)
    flush(stdout)

    # ---- Method C: sequential production (A,f) ----
    tC0 = time()
    resC_outer, stageC = melitz_run_production_stage_sequential_search(ctx_f, obj_f, theta_f0; delta=delta, direction=direction,
        policy=POLICY, n_stages_max=N_STAGES_MAX, max_iterations_per_stage=MAX_ITER_PER_STAGE,
        max_seconds_per_stage=MAX_SECONDS_PER_STAGE, inner_loop_opt=inner_opt, outer_loop_opt=outer_opt)
    wallC = time() - tC0
    cumC = 0
    for sc in stageC
        cumC += sc.n_trials
        best_gt = isnan(sc.best_Delta) ? NaN : kappa_gt(sc.best_objective, ctx_f)
        push!(stage_rows, (method="Method_C_production_Af", delta=delta, direction=string(direction),
              stage_id=sc.stage_id, anchor_gt_pct=gt0, n_trials=sc.n_trials, n_finite_solved=sc.n_finite_solved,
              n_above_cap=sc.n_above_cap, n_screened=sc.n_cap_screened, n_infinite_delta=sc.n_infinite_delta,
              n_numerical_failure=sc.n_numerical_failure, best_incumbent_gt_pct=best_gt, cumulative_evals=cumC,
              wall_s=sc.wall_s, q_basis_norm=0.0, stage_fingerprint=UInt64(0)))
    end
    gtC = kappa_gt(resC_outer.incumbent.objective * (direction==:upper ? 1.0 : -1.0), ctx_f)
    nC_total = sum(sc.n_trials for sc in stageC)
    @printf("Method C (sequential production A,f): stages=%d evals=%d wall=%.1fs stopped=%s  GT%%: %.4f -> %.4f\n",
            length(stageC), nC_total, wallC, resC_outer.stopped_reason, gt0, gtC)
    flush(stdout)

    more_extreme(a, b) = melitz_reduced_q_more_extreme_kappa(direction,
        kappa_ratio_of_g(a[2] * (direction==:upper ? 1.0 : -1.0), a[3]), kappa_ratio_of_g(b[2] * (direction==:upper ? 1.0 : -1.0), b[3]))
    candidates = [("Method_A_welfare_plus_A", resA.incumbent.objective, ctx_q, gtA, nA_total, wallA),
                  ("Method_B_reduced_q", resB.incumbent.objective, ctx_q, gtB, nB_total, wallB),
                  ("Method_C_production_Af", resC_outer.incumbent.objective, ctx_f, gtC, nC_total, wallC)]
    winner = candidates[1]
    for c in candidates[2:end]
        if more_extreme((c[1],c[2],c[3]), (winner[1],winner[2],winner[3]))
            winner = c
        end
    end
    @printf("  MATCHED-EFFORT WINNER delta=%.1f dir=%-6s: %-28s GT%%=%.4f (evals=%d wall=%.1fs)\n",
            delta, string(direction), winner[1], winner[4], winner[5], winner[6])
    flush(stdout)

    for c in candidates
        push!(summary_rows, (delta=delta, direction=string(direction), method=c[1], start_gt_pct=gt0,
              best_gt_pct=c[4], n_evals_total=c[5], wall_s=c[6], is_winner=(c[1]==winner[1])))
    end
end

for direction in (:upper, :lower)
    run_cell(0.5, direction)
end

stage_header = ["method", "delta", "direction", "stage_id", "anchor_gt_pct", "n_trials",
                 "n_finite_solved", "n_above_cap", "n_screened", "n_infinite_delta",
                 "n_numerical_failure", "best_incumbent_gt_pct", "cumulative_evals", "wall_s",
                 "q_basis_norm", "stage_fingerprint"]
melitz_write_typed_counter_csv(joinpath(OUTDIR, "melitz_gate2_d4_matched_effort_stages_2026-07-29.csv"), stage_header, stage_rows)

summary_header = ["delta", "direction", "method", "start_gt_pct", "best_gt_pct", "n_evals_total", "wall_s", "is_winner"]
melitz_write_typed_counter_csv(joinpath(OUTDIR, "melitz_gate2_d4_matched_effort_summary_2026-07-29.csv"), summary_header, summary_rows)

println("\nGate 2 matched-effort comparison complete. Stage rows: ", length(stage_rows), "  Summary rows: ", length(summary_rows))
flush(stdout)
