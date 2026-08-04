# profiled-outer-ab-readiness-2026-08-04 — current-state snapshot

Written at session start, before any Section 3+ work, to give the next continuation (or the user)
a single trustworthy pointer instead of having to re-derive state from prose scattered across the
prior closeout docs.

## Canonical prototype

```
branch: prototype/profiled-destination-scales
HEAD:   395dec3e1e68844128cc98c16be17e91bc9b6603
tag:    profiled-functional-ready-2026-08-04  (points at HEAD, confirmed via `git tag --points-at HEAD`)
```

Verified live against `origin` via `git fetch --all --prune` at session start (no divergence).

## All-five functional readiness

Per `docs/audits/profiled-functional-readiness-closeout-2026-08-03/CONTINUATION_2026-08-04.md`'s
own final verdict block (`FUNCTIONAL_READY`), confirmed genuinely merged into the canonical
prototype (see the POST-HOC CORRECTION added to that file this session — the doc's own closing
paragraph said "awaiting confirmation" for the merge/tag/push/cleanup sequence, which was stale;
`git merge-base --is-ancestor` confirms the doc's own start commit `5a1a647` is a straight-line
ancestor of current canonical HEAD via a clean 20-commit chain):

```
FUNCTIONAL_READY =
    unrestricted:    yes
    flexible_CM:     yes
    common_frechet:  yes
    origin_ZC:       yes  (free-nu evaluator+gradient D4/D20w20k/D20w100k verified)
    CM_plus_ZC:      yes  (free-nu evaluator+gradient D4/D20w20k/D20w100k verified)
```

`FamilyRegistry.jl`'s `production_ready` field is `true` for all 5 REDUCED rows and all 5 FULL
rows. `free_nu_supported` is `true` for origin_zc/cm_meanzc in BOTH formulations (this session
corrected a stale module-docstring claim that it was "deliberately still false" for the REDUCED
pair — the actual capability-row booleans were already correct; only the prose describing them was
stale. See the docstring edit in this session's commit 1.).

## Known remaining outer-performance gaps (this task's actual mission, sections 3-13 below)

These are NOT closed by the functional-readiness work above — that work proved the REDUCED
evaluators/gradients are *correct*; it did not address fair-comparison *performance* infrastructure:

1. **Canonical-runner free-nu parity (task §3)**: not yet independently verified that
   `bin/run_profiled_model.jl`'s FULL origin-ZC path dispatches to a genuine free-nu production
   path rather than a fixed-nu-near-one diagnostic wrapper, side by side with REDUCED's
   `run_profiled_upper_constrained_free_nu`. `FamilyCapability` rows record the drivers by name but
   nothing in the registry certifies the two arms of an A/B use *matched* nu policy.
2. **Powered profiled-relative A coordinates (task §4)**: REDUCED's only coordinate mode is native
   `:profiled_pivot_anchor_relative` (`coordinate_modes` field, all 5 REDUCED rows). FULL uses
   `:powered_aspace`. `:profiled_powered_relative_A` does not exist anywhere in the tree —
   confirmed by the registry's own docstring (line ~150, unedited this session, still accurate) and
   a repo grep. No derivation of the transform exists yet.
3. **Matched gradient instrumentation (task §5)**: `threaded_outer_gradient=false` and
   `bandwidth_cache=false` for every REDUCED row in the registry — there is no REDUCED-side timer
   breakdown to compare against FULL's at all yet, let alone a matched one.
4. **REDUCED gradient threading (task §6)**: `threaded_outer_gradient=false` for all 5 REDUCED
   rows (registry, unedited this session). FULL rows show `true`. Not ported.
5. **REDUCED bandwidth-search reuse (task §7)**: `bandwidth_cache=false` for all 5 REDUCED rows.
   No cache exists to key/invalidate.
6. **Decoded-state / short outer-search A/Bs (task §8-10)**: not run this session as of this
   snapshot — blocked behind items 1-5 above per the task's own gate ordering (§10 "run only after
   all preceding gates pass").
7. **Coordinate-mode tournament (task §11)**: blocked on item 2 (nothing to tournament against
   native REDUCED yet).

## Relationship to the parallel fixed-state inner A/B task

Per this task's own §12, this session does not rerun or duplicate
`benchmark/profiled-fixed-state-inner-ab-2026-08-04` (a separate Claude's ownership). As of this
snapshot that branch/report was not yet available to read; this doc will be updated (not silently
assumed) once it lands.
