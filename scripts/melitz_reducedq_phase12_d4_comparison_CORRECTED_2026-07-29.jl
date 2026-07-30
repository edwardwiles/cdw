# CORRECTED Phase 12 D4 outer comparison (2026-07-29 reduced-q validation session, Phase 0).
#
# Supersedes `scripts/melitz_reducedq_phase12_d4_comparison_2026-07-29.jl` and its CSV/doc
# output. Three problems were found in the original output:
#
#   1. CSV labels containing a literal comma ("production_(A,f)", "full_experimental_(A,q)")
#      were written via an unquoted `join([...], ",")` -- any standard CSV parser misaligns
#      every column from `direction` onward for those 8 of 12 rows.
#   2. The accompanying markdown doc's own prose ("the Phase 7 cap screen fired... 91/157/111
#      screened evaluations") misread the `n_numerical_failure` column as `n_screened` -- the
#      CSV's own `n_screened` field is `0` in every one of the 12 original rows; the cap screen
#      never fired in that smoke test.
#   3. A THIRD, more serious problem found while attempting to repair (1)/(2) without a rerun
#      (`scripts/melitz_phase0_diagnose_reducedq_counter_invariant_2026-07-29.jl`): for every
#      `sequential_reduced_q` row, the doc/CSV's own reported `n_finite_solved`+`n_above_cap`
#      total exceeds the recorded step count (`n_accepted_steps`) by a consistent ~35-40%,
#      even though NEITHER `MelitzReducedQTrialRecord` duplication NOR the per-stage
#      classification/`length(trials)` invariant (`sr.n_finite_solved+sr.n_above_cap+
#      sr.n_infinite_delta+sr.n_numerical_failure+sr.n_cap_screened == length(sr.trials)`)
#      reproduces under a byte-identical rerun of the SAME cell on the SAME (unmodified)
#      `reduced_q_controller.jl` -- the rerun's per-stage AND total invariant holds EXACTLY
#      (494==494 for the 0.1/upper cell), its `n_numerical_failure` total (91) and total trial
#      count (494) match the ORIGINAL CSV's own `n_accepted_steps`, but its `n_finite_solved`
#      (369) and `n_above_cap` (34) totals do NOT match the original doc's reported 568/40.
#      Since the exact underlying per-trial records from the ORIGINAL run were never persisted
#      to disk, there is no way to determine whether the original doc's own printed 568/40 came
#      from a stale/duplicated in-memory `result.stages` at print time or a hand-transcription
#      error -- either way, the ORIGINAL numbers cannot be trusted, and per the governing
#      prompt's own instruction ("do not infer classifications from numeric values... reporting
#      scripts", "reproduce... if exact trial records are available" -- they were not), this is
#      a genuine, disclosed case where a rerun IS required, not merely a formatting repair.
#
# This script reruns all 12 original cells with IDENTICAL settings (same D4 fixture, same
# CappedEvaluation(10.0), same outer-loop option file, same n_stages_max/max_iterations_per_stage/
# max_seconds_per_stage), validates `melitz_validate_typed_counters(...; n_trials=...)` against
# an INDEPENDENTLY recorded trial count immediately after every run (any future recurrence of
# problem 3 aborts the script rather than silently exporting bad data), and writes BOTH the CSV
# and the markdown table from the SAME typed-counter records via `typed_eval_counters.jl`.

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Printf
using Serialization: serialize
println("Julia threads: ", Threads.nthreads()); flush(stdout)

const OUTDIR = joinpath(REPO, "docs", "key_results")
const D4_SEED = 29
const D4_W = 20_000

# Gate 1 needs the EXACT economic state of each sequential_reduced_q incumbent (theta_full,
# Delta, dual x, GT%) to replay at larger W without reoptimizing. The original session never
# persisted these to disk (only summary scalars were written to the CSV) -- this run persists a
# full checkpoint for every sequential_reduced_q cell as it goes.
const INCUMBENT_CHECKPOINTS = Dict{Tuple{Float64,Symbol},NamedTuple}()
inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
outer_opt = joinpath(REPO, "melitz_outer_finite_delta_alg_direct_2026-07-27.opt")

data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)

rows = NamedTuple[]

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
    counters = melitz_typed_counters_from_direct_result(res)
    # Independent trial-count cross-check: res.inner_solve_count is recorded by
    # finite_delta_outer.jl's own callback machinery, entirely separately from
    # n_inner_solved/n_above_cap_reject/etc -- a genuine independent check, not tautological.
    melitz_validate_typed_counters(counters; n_trials=res.inner_solve_count)
    @printf("[%-28s] delta=%.1f dir=%s  nStatus=%d  wall=%.1fs  inner_solves=%d\n",
            label, delta, direction, res.nStatus, wall, res.inner_solve_count)
    @printf("    start GT%%=%.4f  best GT%%=%.4f  Delta_best=%.6e  within_budget=%s  n_FiniteSolved=%d n_AboveCap=%d n_Infinite=%d n_Fail=%d\n",
            gt0, gt_best, delta_best, string(!isnan(delta_best) && delta_best<=delta),
            counters.n_finite_solved, counters.n_above_cap_evaluated, counters.n_infinite_delta, counters.n_numerical_failure)
    flush(stdout)
    push!(rows, (label=label, delta=delta, direction=string(direction), nStatus=res.nStatus, wall_s=wall,
                 start_gt_pct=gt0, best_gt_pct=gt_best, kappa_best=kappa_best, delta_best=delta_best,
                 within_budget=(!isnan(delta_best) && delta_best<=delta),
                 n_accepted_stages=1, n_accepted_steps=res.inner_solve_count,
                 n_finite_solved=counters.n_finite_solved, n_above_cap_evaluated=counters.n_above_cap_evaluated,
                 n_screened_above_cap=counters.n_screened_above_cap, n_above_cap_total=melitz_total_above_cap(counters),
                 n_infinite_delta=counters.n_infinite_delta, n_numerical_failure=counters.n_numerical_failure,
                 n_affine_excluded=counters.n_affine_excluded, termination_status=string(res.nStatus),
                 counters=counters, n_trials=res.inner_solve_count))
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
    counters = melitz_typed_counters_from_reduced_q_stages(result.stages)
    n_steps_independent = sum(length(s.trials) for s in result.stages)
    # Validate EVERY stage individually (not just the aggregate) -- this is exactly the check
    # that would have caught problem 3 above before it ever reached a CSV or a doc.
    for s in result.stages
        melitz_validate_typed_counters(melitz_typed_counters_from_reduced_q_stage(s); n_trials=length(s.trials))
    end
    melitz_validate_typed_counters(counters; n_trials=n_steps_independent)
    @printf("[%-28s] delta=%.1f dir=%s  stopped=%s  wall=%.1fs  n_stages=%d  n_steps=%d\n",
            label, delta, direction, result.stopped_reason, wall, length(result.stages), n_steps_independent)
    @printf("    start GT%%=%.4f  best GT%%=%.4f  Delta_best=%.6e  within_budget=%s  n_FiniteSolved=%d n_AboveCap=%d n_Infinite=%d n_Fail=%d n_screened=%d\n",
            gt0, gt_best, result.incumbent.Delta, string(result.incumbent.Delta<=delta),
            counters.n_finite_solved, counters.n_above_cap_evaluated, counters.n_infinite_delta,
            counters.n_numerical_failure, counters.n_screened_above_cap)
    flush(stdout)
    push!(rows, (label=label, delta=delta, direction=string(direction), nStatus=-1, wall_s=wall,
                 start_gt_pct=gt0, best_gt_pct=gt_best, kappa_best=kappa_best, delta_best=result.incumbent.Delta,
                 within_budget=(result.incumbent.Delta<=delta),
                 n_accepted_stages=length(result.stages), n_accepted_steps=n_steps_independent,
                 n_finite_solved=counters.n_finite_solved, n_above_cap_evaluated=counters.n_above_cap_evaluated,
                 n_screened_above_cap=counters.n_screened_above_cap, n_above_cap_total=melitz_total_above_cap(counters),
                 n_infinite_delta=counters.n_infinite_delta, n_numerical_failure=counters.n_numerical_failure,
                 n_affine_excluded=counters.n_affine_excluded, termination_status=string(result.stopped_reason),
                 counters=counters, n_trials=n_steps_independent))

    # Gate 1 checkpoint: full economic state of this incumbent, for later replay at larger W
    # WITHOUT reoptimizing (Gate 1's own "hold the economic state fixed" requirement).
    INCUMBENT_CHECKPOINTS[(delta, direction)] = (
        label=label, delta=delta, direction=direction,
        theta_full=copy(result.incumbent.theta), dual_x=copy(result.incumbent.x),
        Delta_W20000=result.incumbent.Delta, objective=result.incumbent.objective,
        gt_pct_W20000=gt_best, kappa_W20000=kappa_best,
        D=4, sigma=2.5, theta_star=6.8, target_country=1,
        qmc_seed=D4_SEED, W_original=D4_W, outer_parameterization=:logcutoff,
        n_stages=length(result.stages), stopped_reason=result.stopped_reason,
        start_gt_pct=gt0, wall_s=wall)
end

for delta in (0.1, 0.5), direction in (:upper, :lower)
    run_production_or_full_aq("production_(A,f)", :auto, :logf, delta, direction)
    run_production_or_full_aq("full_experimental_(A,q)", :B_direct_argument_aq_experimental, :logcutoff, delta, direction)
    run_reduced_q("sequential_reduced_q", delta, direction)
end

header = ["label", "delta", "direction", "nStatus", "wall_s", "start_gt_pct", "best_gt_pct",
          "kappa_best", "delta_best", "within_budget", "n_accepted_stages", "n_accepted_steps",
          "n_finite_solved", "n_above_cap_evaluated", "n_screened_above_cap", "n_above_cap_total",
          "n_infinite_delta", "n_numerical_failure", "n_affine_excluded", "termination_status"]

csv_path = joinpath(OUTDIR, "melitz_reducedq_phase12_d4_comparison_2026-07-29.csv")
melitz_write_typed_counter_csv(csv_path, header, rows)
println("Wrote corrected, validated, standards-compliant CSV: ", csv_path)

md_path = joinpath(OUTDIR, "melitz_reducedq_phase12_d4_comparison_CORRECTED_2026-07-29.md")
note = """
CORRECTED Phase 12 D4 outer comparison (2026-07-29 reduced-q validation session, Phase 0). This
table supersedes the original `docs/melitz_reduced_q_subspace_search_2026-07-29.md` Phase 12
table and its CSV: (1) the CSV is now properly quoted/standards-compliant; (2) `n_screened_above_cap`
is now its own explicit column, genuinely `0` throughout -- the Phase 7 cap screen did not fire
in this smoke test at any cell, and the ORIGINAL doc's prose attributing 91/157/111 "screened"
evaluations to the cap screen was a misread of the `n_numerical_failure` column; (3) EVERY row
here was produced by a FRESH rerun (not the original run's numbers) because the original
`sequential_reduced_q` rows' own `n_finite_solved+n_above_cap` totals failed a basic
trial-count cross-check that this script's `melitz_validate_typed_counters` now enforces before
any export -- see this script's own header comment for the full diagnosis. Every row below
therefore passed `melitz_validate_typed_counters(...; n_trials=<independently recorded trial
count>)` at write time."""
melitz_write_typed_counter_markdown(md_path, "Phase 12 D4 outer comparison (CORRECTED, rerun)", header, rows; note=note)
println("Wrote corrected markdown table: ", md_path)

println("\n" * "="^100)
println("CORRECTED most-extreme-incumbent comparison per (delta,direction), using melitz_reduced_q_more_extreme_kappa:")
for delta in (0.1, 0.5), direction in (:upper, :lower)
    cellrows = filter(r -> r.delta==delta && r.direction==string(direction), rows)
    best = cellrows[1]
    for r in cellrows[2:end]
        if melitz_reduced_q_more_extreme_kappa(direction, r.kappa_best, best.kappa_best)
            best = r
        end
    end
    @printf("  delta=%.1f dir=%-6s -> MOST EXTREME: %-28s  GT%%=%.4f  kappa=%.6f\n", delta, direction, best.label, best.best_gt_pct, best.kappa_best)
end
println("\nPhase 12 CORRECTED comparison complete. Rows: ", length(rows))

ckpt_path = joinpath(OUTDIR, "melitz_reducedq_phase12_incumbent_checkpoints_2026-07-29.jls")
open(ckpt_path, "w") do io
    serialize(io, INCUMBENT_CHECKPOINTS)
end
println("Wrote Gate 1 incumbent-state checkpoints (", length(INCUMBENT_CHECKPOINTS), " cells): ", ckpt_path)
flush(stdout)
