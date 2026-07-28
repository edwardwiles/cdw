# True no-H operator bundle: post-merge public-driver smokes (2026-07-28)

## Setup

Fresh, clean, independent `git worktree add` at the exact new remote production head
(`cdw/production/fullA-exact` = `fb6ad2ea4b0e0e4fc730af4d431e74a5ddc7b744`), detached, verified
identical to the pushed remote ref before running anything:

```
worktrees/postmerge-smoke-2026-07-28  (detached HEAD fb6ad2e)
git merge-base --is-ancestor HEAD cdw/production/fullA-exact  -> true
git merge-base --is-ancestor cdw/production/fullA-exact HEAD  -> true
```

Spec, all 5 families: `D=20`, `Ddest=19` (destination_sample=:exclude_row), `W=100,000` Sobol,
fixed theta, transformed-A production coordinates (`A_coordinate_mode=powered_aspace`, confirmed
in every log's own startup banner), calibration start, one upper-bound solve (`find_smallest=true`),
`maxtime_real=90.0` (short cap, sufficient to enter outer+inner iterations, not to converge).

The pre-existing `smoke_delta1_*.jl` scripts (committed at `93f26df`, explicitly flagged
"not yet run" by the prior session) were adapted for this spec (W 80k→100k, budget 600s→90s,
`destination_sample=:exclude_row` added where missing) and given a small diagnostics addition
(`postmerge_smoke_diagnostics.jl`, new file) that prints the bundle's structural no-H facts and
the `NO_DENSE_G_COUNTERS` runtime counters after each solve. These adapted scripts were copied
uncommitted into the fresh worktree (not pushed to production) purely to run this smoke gate.

## Two real bugs found and fixed while actually running these scripts for the first time

Both pre-existed in the "not yet run" smoke scripts from `93f26df` -- neither was introduced by
the no-H bundle merge, and both were caught immediately by clean Julia errors (a `BoundsError` and
an `UndefVarError`), not silent wrong answers:

1. **`smoke_delta1_unrestricted.jl`**: computed `Aod_theta_natural` using a square `D×D` slice
   (`D2 = D^2`) after adding `destination_sample=:exclude_row`, which makes the layout rectangular
   `D×Ddest` (20×19, not 20×20) -- `BoundsError`. Fixed to slice `ctx_probe.D*ctx_probe.D_dest`
   and reshape accordingly, matching the pattern already used in
   `test_operator_no_H_bundle_equivalence_unrestricted_d20.jl` and
   `bench_unrestricted_20thread_worker_selection.jl`.
2. **`smoke_delta1_frechet.jl`** (and, defensively, the other 3 CM-family smokes, which share the
   same include list but don't happen to exercise this exact function): missing
   `"cm_frechet_lookup_production.jl"` from the include list -- `archC_frechet_verified_state`
   calls `inner_loop_internal_cmfrechetlookup_production`, defined only in that file, causing
   `UndefVarError` inside the real KNITRO callback. Fixed by adding the missing include (same file
   already present in the working equivalence-gate script's include list).

Both fixes are include-list/indexing corrections to test scripts, not production code changes.

## Per-family smoke results

All 5 launched, built the real D=20/W=100,000/Ddest=19 context, and executed real inner FG,
Hessian, and (for the CM-family drivers) verification/screen logic through KNITRO, entering actual
outer+inner iterations before the 90s cap stopped them (`nStatus=-401`, KN_RC_TIME_LIMIT_FEAS --
current point feasible, not an error).

| Family | outer iters | evals | grad evals | inner status | wall (s) |
|---|---|---|---|---|---|
| flexible CM | 2 | 4 | 3 | 0 | 177.5 |
| common Fréchet | 1 | 2 | 2 | 0 | 163.9 |
| CM+ZC | 1 | 3 | 2 | 0 | 265.4 |
| ZC-only (origin-ZC) | 1 | 2 | 2 | 0 | 147.7 |
| unrestricted | 1 | 2 | 2 | 0 | 188.8 |

(Wall time exceeds the 90s `maxtime_real` cap because that budgets KNITRO's own solve time, not
process/compile/context-build overhead -- consistent with the D=20 gates' own ~20-30s per real
solve plus this process's one-time compile/precompile cost.)

## Bundle/backend diagnostics (all 5 families)

Every log prints, per the required list:

```
bundle type                 = OperatorPsiBundle (production default; confirmed structurally,
                               all 5 families, in TRUE_OPERATOR_NO_H_PREMERGE_GATE_2026-07-28.md)
has H field                 = false (all 5)
has H_copy field             = false (all 5)
has moments! field            = false (all 5)
economic moment builder      = per-family inner_fg_backend (flexcm/cm_meanzc: operator via
                               shared cm_lookup path; common_frechet: cm_frechet_lookup;
                               origin_zc/unrestricted: operator, family-owned)
economic FG backend          = same Refs as above (this codebase's own dispatch does not carry a
                               second, separately-configurable "restriction FG backend" -- one Ref
                               drives both, per the family's own build_*_production_context)
verification backend         = :operator (production default, all families; CM-family verified via
                               archC_verified_state/archC_frechet_verified_state's internal calls;
                               D=20 flexible-CM cross-check already proved bit-identical vs dense
                               in the premerge gate)
H_EE backend                 = core_hessian_backend=exact_winner_pair_parallel (flexcm/frechet/
                               cm_meanzc/origin_zc/unrestricted, all confirmed live in each log's
                               own [backend-manifest] print)
H_ER/H_EC backend             = cross_hessian_backend: cm_bin_prefix (flexcm/cm_meanzc),
                               dense_exact (origin_zc, retained dense by design per task history),
                               none (unrestricted -- no cross block exists for this family)
H_RR/H_CC backend             = restriction_hessian_backend: cm_bin_prefix_plus_congruence (flexcm/
                               cm_meanzc), dense_exact (origin_zc), none (unrestricted)
legacy-H allocation counters  = full_G_materializations=0, dense_economic_G_materializations=0,
                               dense_CM_G_materializations=0, dense_ZC_G_materializations=0 in
                               EVERY family that tracks them (flexcm/cm_meanzc/origin_zc) -- see
                               flagged exception below for common-Fréchet
moments!/select_G counters   = generic_dense_FG_calls=0 in every family that tracks it (this Ref IS
                               the production moments!/select_G_from_H call counter in this
                               codebase's instrumentation)
```

## Flagged finding: common-Fréchet's priming-side dense-G counter is NOT zero

`dense_Frechet_G_materializations = 14` in the common-Fréchet smoke (all other counters 0). This is
**not a new regression from this merge** -- it is the exact, already-documented, pre-existing gap
recorded in this branch's own `SESSION_MASTER_VERDICT_2026-07-28.md` ("common_frechet: NOT
changed... skip_fill_safe_frechet hardcoded false") and `TRUE_OPERATOR_BUNDLE_MASTER_REPORT_2026-07-28.md`
("COMPOSITE_G_MATERIALIZATIONS = 0 on the Hessian-callback side (inherited)... common-Fréchet still
materializes it unconditionally"). It is scoped to common-Fréchet's own **priming-side economic-block
fill** (a setup-time dense array build), not the `OperatorPsiBundle`'s own field/storage structure --
common-Fréchet's bundle itself is confirmed to have no `H`/`H_copy`/`moments!` field, structurally,
in both this smoke and the premerge equivalence gates. The task's Part A claim (no-H bundle
architecture) holds for common-Fréchet; a separate, already-named, out-of-scope follow-up
(root-causing and fixing common-Fréchet's priming-side skip, previously investigated and left
inconclusive per that session's own honest writeup) remains open and is not part of this merge's
claimed scope.

## Verdict

```
ALL_FIVE_FAMILIES_LAUNCH_AND_EXECUTE_REAL_INNER_FG_HESSIAN_VERIFICATION = yes
NO_LEGACY_INTERFACE_EXERCISED = yes (all 5; obj.H structurally absent, confirmed both statically
                                     and via the premerge gate's own throw-on-access test)
KNOWN_PREEXISTING_GAP = common_frechet_priming_side_dense_G_materialization_nonzero
                        (documented before this merge, out of this task's Part A scope, not a
                        regression)
TWO_SMOKE_SCRIPT_BUGS_FOUND_AND_FIXED = unrestricted_rectangular_slice, frechet_missing_include
                        (both pre-existing in the never-run 93f26df scripts, both test-script-only)
POSTMERGE_SMOKES = pass
```
