# Melitz-owned per-run outer-solve diagnostics (2026-07-27 continuation).
#
# `solve_melitz_finite_delta_bound` previously read `CounterfactualSensitivity.INNER_SOLVE_COUNT[]`/
# `INNER_INFEAS_COUNT[]`/`INNER_ITERS_TOTAL[]` unconditionally after `KN_solve` -- a top-level
# Melitz outer solve therefore threw `UndefVarError: CounterfactualSensitivity not defined`
# unless `cc_algo` had already been loaded (found live while running the 2026-07-27 Stage 10B
# campaign standalone; see docs/melitz_outer_parameterization_comparison_2026-07-26.md Section C).
#
# Beyond breaking standalone use, those CS counters were also SILENTLY WRONG for the
# matrix-free path: `CS.INNER_SOLVE_COUNT`/`INNER_INFEAS_COUNT` are incremented ONLY inside
# `cc_algo/inner_loop_functions.jl`'s own dense `inner_loop`/`inner_loop_KNITRO` (confirmed by
# grep) -- `MelitzCCBundle`'s matrix-free inner solve (`cc_bundle.jl`'s own KNITRO driver) never
# touches them. Any matrix-free `solve_melitz_finite_delta_bound` call therefore always reported
# `inner_solve_count == 0`/`inner_infeas_count == 0`, regardless of how many inner solves
# actually happened, whenever `cc_algo` happened to be loaded (never crashing, just wrong).
#
# `MelitzRunDiagnostics` replaces both problems with one Melitz-owned, backend-agnostic object,
# built entirely from counters this codebase ALREADY owns per-call:
#   - `cbset`'s own local `Ref`s (`n_fc_calls`, `n_ga_calls`, `n_inner_solved`,
#     `n_infinite_delta_reject`, `n_above_cap_reject`, `n_numerical_failure_reject`) -- created
#     fresh by `melitz_build_finite_delta_callbacks` on every call (genuinely per-run, not
#     process-global), and already bundle-agnostic: `melitz_classified_inner_solve`
#     (`inner_screening.jl`) resolves every trial point to exactly one of these four outcomes
#     for EITHER backend (comment at `finite_delta_outer.jl`'s own `MelitzFiniteDeltaOuterResult`
#     docstring, Section 616).
#   - `backend_config.jl`'s existing process-global `MELITZ_ALL_COUNTER_REFS` (cache hits,
#     dense-fallback-site calls, matrix-free objective/gradient/Hessian calls), snapshotted
#     immediately before and after `KN_solve` via the already-existing
#     `melitz_backend_counters_snapshot()` and diffed -- turning genuinely process-global
#     counters into a genuinely per-run figure without threading a new argument through every
#     dispatch site that increments them (the same "snapshot + diff" pattern this codebase
#     already uses for ablation studies, per `backend_config.jl`'s own Phase 10 comment).
#
# No `isdefined(CounterfactualSensitivity)` guard is used anywhere in this file: the CS
# dependency is removed entirely, not merely wrapped.

"""
    MelitzRunDiagnostics

Melitz-owned, backend-agnostic per-run outer-solve diagnostics, returned as
`MelitzFiniteDeltaOuterResult.diagnostics`. Every field is a genuine per-call count (a
before/after delta for the global `MELITZ_ALL_COUNTER_REFS` counters, a direct read for the
per-call `cbset` counters) -- never a raw, unreset process-global value.

- `inner_solve_attempts`: every trial point resolved by `melitz_classified_inner_solve`
  (dense or matrix-free), i.e. `inner_solved + infinite_delta_reject + above_cap_reject +
  numerical_failure_reject`.
- `inner_solved`: resolved to a genuine finite, within-budget `DeltaStar`.
- `inner_infinite_delta_reject`: certified `InfiniteDeltaCertified`.
- `inner_above_cap_reject`: `AboveEvaluationCap` (fast-classified, not fully re-derived).
- `inner_numerical_failure_reject`: a genuine inner-solver numerical/convergence failure --
  the closest backend-agnostic analog to `CounterfactualSensitivity.INNER_INFEAS_COUNT`'s own
  "the inner KNITRO solve itself reported infeasible" semantics (which only ever measured the
  dense path).
- `n_fc_calls`/`n_ga_calls`: total `cb_F!`/`cb_G!` invocations.
- `exact_point_cache_hits`/`fc_to_ga_cache_hits`: exact-point cache reuse.
- `dense_fallback_calls`: sum of every dense-path counter (`dense_moment_calls`,
  `dense_outer_gradient_calls`, `dense_inner_objective/gradient/hessian_calls`,
  `dense_G_materializations`, `production_dense_screen_calls`,
  `diagnostic_dense_screen_calls`) -- always `0` under `forbid_dense_fallback=true`.
- `matrix_free_objective_calls`/`matrix_free_gradient_calls`/`matrix_free_hessian_calls`:
  matrix-free callback invocations.
"""
struct MelitzRunDiagnostics
    inner_solve_attempts::Int
    inner_solved::Int
    inner_infinite_delta_reject::Int
    inner_above_cap_reject::Int
    inner_numerical_failure_reject::Int
    n_fc_calls::Int
    n_ga_calls::Int
    exact_point_cache_hits::Int
    fc_to_ga_cache_hits::Int
    dense_fallback_calls::Int
    matrix_free_objective_calls::Int
    matrix_free_gradient_calls::Int
    matrix_free_hessian_calls::Int
end

const MELITZ_DENSE_FALLBACK_COUNTER_KEYS = (
    :dense_moment_calls, :dense_outer_gradient_calls, :dense_inner_objective_calls,
    :dense_inner_gradient_calls, :dense_inner_hessian_calls, :dense_G_materializations,
    :production_dense_screen_calls, :diagnostic_dense_screen_calls)

"""
    melitz_run_diagnostics(cbset, counters_before::NamedTuple, counters_after::NamedTuple)
        -> MelitzRunDiagnostics

Builds the per-run diagnostics object. `counters_before`/`counters_after` must both be
`melitz_backend_counters_snapshot()` results bracketing the solve; `cbset` must be the
`NamedTuple` returned by `melitz_build_finite_delta_callbacks` for the SAME solve.
"""
function melitz_run_diagnostics(cbset, counters_before::NamedTuple, counters_after::NamedTuple)
    Δ(key) = counters_after[key] - counters_before[key]
    n_solved = cbset.n_inner_solved[]
    n_inf = cbset.n_infinite_delta_reject[]
    n_cap = cbset.n_above_cap_reject[]
    n_num = cbset.n_numerical_failure_reject[]
    dense_fallback_calls = sum(Δ(k) for k in MELITZ_DENSE_FALLBACK_COUNTER_KEYS)
    return MelitzRunDiagnostics(
        n_solved + n_inf + n_cap + n_num,
        n_solved, n_inf, n_cap, n_num,
        cbset.n_fc_calls[], cbset.n_ga_calls[],
        Δ(:exact_point_cache_hits), Δ(:FC_to_GA_cache_hits),
        dense_fallback_calls,
        Δ(:matrix_free_objective_calls), Δ(:matrix_free_gradient_calls),
        Δ(:matrix_free_hessian_calls))
end
