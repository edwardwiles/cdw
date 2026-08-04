# REDUCED verifier check — m_min false-negative bug (2026-08-04)

## Question asked

A separate Claude session found and fixed a silent false-rejection bug in FULL's post-inner-solve
verification gate (`classify_inner_result`, `oracle.jl`): the strict `m_min > tol.m_min_floor`
check rejected mathematically valid solves whenever a far-tail Monte Carlo draw's recovered weight
`m = dPsi(r) = exp(r)` (for `r<=1`) underflowed to exactly Float64 `0.0` (happens for `r` roughly
below `-745`), even though `dPsi!` is provably strictly positive on both its branches for any
finite `r` — so `m<=0` was never real evidence of a bad solve. Fixed on `production/fullA-exact`
(commit `546feff`): relax to `m_min >= tol.m_min_floor`, gate on a new `m_weights_all_finite`
field instead (`all(isfinite, m_weights)`, the real safeguard against genuine divergence).
Question: does REDUCED have its own, independently-implemented verifier with the same bug?

## Finding: no — REDUCED shares the exact same verifier code

`reduced_operator_verification_2026-08-01.jl::verify_inner_solution_reduced_profiled!` only
recomputes `(r, f, kkt_resid)` from REDUCED's own kernels (`reduced_homogeneous_dual_contraction`/
`reduced_homogeneous_transpose_contraction!`), then feeds that straight into the SAME
`verify_namedtuple_from_operator` (`operator_verification.jl`) and (downstream, wherever a caller
classifies the result) the SAME `classify_inner_result` (`oracle.jl`) that FULL uses — confirmed by
that file's own docstring ("Feeds the SAME `verify_namedtuple_from_operator`... unchanged") and by
direct read: there is no REDUCED-specific reimplementation of either function anywhere in the tree.

Every REDUCED family evaluator (`profiled_outer_evaluator_2026-08-01.jl` for unrestricted,
`profiled_restricted_family_adapters_2026-08-02.jl` for flexible_cm/common_frechet,
`profiled_zc_lane_point_evaluators_2026-08-02.jl`/`profiled_zc_free_eta_2026-08-04.jl` for
origin_zc/cm_meanzc) uses `result = merge(verify, (...))`, merging the FULL `verify` NamedTuple
wholesale — unlike FULL's own `compressed_live.jl::evaluate_fullA_fast_compressed` (unrestricted
family), which hand-built its own partial NamedTuple and had a real, separate gap (silently
dropping `m_weights_all_finite`, found and fixed in the SAME production commit). **REDUCED has no
such gap** — confirmed by grep across all 5 REDUCED point evaluators before writing the
confirmation test below.

**Practical consequence**: REDUCED's own CURRENT production KNITRO search loops
(`run_profiled_upper_constrained`/`run_profiled_upper_constrained_free_nu`,
`profiled_production_outer_constrained_2026-08-02.jl`/
`profiled_zc_free_nu_production_driver_2026-08-04.jl`) gate live accept/reject purely on KNITRO's
own `inner_status`, not on `is_verified_success`/`classify_inner_result` — so the specific "silent
mid-campaign rejection" failure mode does not currently manifest in REDUCED's own live search loop
the way it did in FULL's `cm_checkpoint.jl`/`cm_outer_driver.jl`. The underlying `m_min`/`verify`
data REDUCED computes and stores IS subject to the identical bug, though, and any current or future
code that calls `is_verified_success`/`classify_inner_result` on REDUCED's results (post-hoc
analysis, a future re-introduction of the currently-unused `verify_fn` hook in
`profiled_production_outer_runner_2026-08-01.jl`) would be affected without this fix.

## Fix applied

Ported commit `546feff` from `production/fullA-exact` onto a new branch,
`fix/reduced-verifier-underflow-2026-08-04`, based on the current canonical prototype
(`prototype/profiled-destination-scales` @ `b6ac1c6`). Clean cherry-pick, no conflicts:
`oracle.jl` and `compressed_live.jl` are byte-identical between the two branches at the relevant
point; `operator_verification.jl` differs only in unrelated Phase-10 per-block KKT-residual
diagnostics added to different functions, so the fix's own target
(`verify_namedtuple_from_operator`) merged cleanly.

No REDUCED-specific source change was needed — the shared-file fix alone closes REDUCED's own
exposure, confirmed empirically below rather than assumed.

## Verification

1. **Ported production regression test** (`test_verifier_underflow_fix_2026-08-04.jl`, FULL-side):
   **45/45 PASS** on this branch, confirming the port applied correctly to the prototype lineage.
2. **New REDUCED-specific confirmation test** (`test_reduced_verifier_underflow_fix_2026-08-04.jl`),
   real D4 KNITRO context, all 5 REDUCED families:
   - `m_weights_all_finite`/`m_weights_all_nonnegative`/`underflow_zero_count`/`r_min`/`r_max` are
     present in every family's real evaluated `result` (not silently dropped).
   - Every family's real calibration-point solve is correctly accepted
     (`classify_inner_result == VerifiedSolved`, `is_verified_success == true`).
   - A synthetic benign-underflow scenario built from REDUCED's own real `m_weights` (mirroring the
     production bug report's own `r=-999.4` example) is correctly **ACCEPTED**.
   - A synthetic genuine-NaN scenario is correctly **REJECTED** with reason
     `:m_weights_not_all_finite`.
   - **13/13 checks PASS.**

## Disposition

- Merged into `prototype/profiled-destination-scales` (fast-forward from `b6ac1c6`), pushed.
- Cherry-picked into `performance/profiled-outer-ab-completion-2026-08-04` (the outer-readiness
  continuation branch), pushed, per explicit request to have it present there too.
- `fix/reduced-verifier-underflow-2026-08-04` branch/worktree removed after merge.
