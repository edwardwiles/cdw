# profiled-outer-ab-readiness-2026-08-04 — current-state snapshot

> **Superseded 2026-08-04**: see `docs/audits/profiled-outer-ab-completion-2026-08-04/CURRENT_STATE_MATRIX.md`
> and `MASTER.md` (branch `performance/profiled-outer-ab-completion-2026-08-04`) for the
> up-to-date, gate-verified status. This file's own final section ("Known remaining outer-
> performance gaps") is accurate as a historical snapshot but is not a substitute for the
> superseding document's real gates.

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

## Progress this session (Sections 1-4 closed; see companion docs)

1. **Canonical-runner free-nu parity (task §3) — CLOSED, both origin_zc and cm_meanzc.** Traced
   `bin/run_profiled_model.jl`'s dispatch end to end (task §3.1) and found BOTH arms were
   fixed-nu, not just FULL: REDUCED's `_run_reduced` called the 3-arg fixed-nu `evaluate_fn` for
   origin_zc/cm_meanzc (nu pinned at `pes` construction, never re-searched) despite
   `FamilyRegistry.free_nu_supported=true`; FULL's `_run_full_originzc` explicitly pinned
   `nu_bounds` to a ~1e-8-wide box, copied from a diagnostic script. Fixed both (commit
   `b40de6f`): REDUCED now dispatches origin_zc/cm_meanzc through
   `run_profiled_upper_constrained_free_nu` (new `_run_reduced_zc_free_nu` function); FULL now
   uses `run_originzc_upper_checkpointed`'s own genuine `originzc_default_nu_bounds` default
   instead of pinning. Verified live, real D20/W=20,000 CLI smokes for origin_zc (both
   formulations) and cm_meanzc (REDUCED): both origin_zc arms wrote `nu_policy="free"` with
   IDENTICAL `nu_bounds` summary `[-15.222177262685872, 3.9134314695358654]` in their
   `run_manifest.json` (same draw_seed/K_mean/K_pair, as expected), and `eta_nu` genuinely moved
   across outer KNITRO evaluations in both logs (not held at its start value). `ABComparability.jl`
   already hard-fails on a `nu_policy`/`nu_bounds` mismatch between arms (lines 122-123,
   pre-existing, unmodified) — task §3.3's "must hard-fail if one arm is fixed/other free"
   requirement was already met by existing infrastructure, no fix needed there.
2. **Powered profiled-relative A coordinates (task §4) — derivation done, mode implemented,
   NOT wired as default.** See
   `POWERED_PROFILED_COORDINATE_DERIVATION_2026-08-04.md` (this directory) for the full
   derivation: `:profiled_powered_relative_A` is a per-coordinate affine reparametrization of
   REDUCED's existing native `r_free`, using the SAME `(logX,logY,theta)` constants FULL's own
   `:powered_aspace` mode already uses, and is a provable bijection (composition of invertible
   affine maps). Implemented in `full_aod_diag/d4_exact/profiled_powered_relative_a_2026-08-04.jl`
   (commit `6af41f1`), additive only, native mode remains every family's default. Verified via a
   synthetic round-trip + FD gradient chain-rule check (not yet a full production-context D4/D20
   gate — see that file's own commit message for exactly what was and wasn't checked).

## Known remaining outer-performance gaps (sections 5-13, NOT started this session)

3. **Matched gradient instrumentation (task §5)**: `threaded_outer_gradient=false` and
   `bandwidth_cache=false` for every REDUCED row in the registry — there is no REDUCED-side timer
   breakdown to compare against FULL's at all yet, let alone a matched one.
4. **REDUCED gradient threading (task §6)**: `threaded_outer_gradient=false` for all 5 REDUCED
   rows (registry, unedited this session). FULL rows show `true`. Not ported.
5. **REDUCED bandwidth-search reuse (task §7)**: `bandwidth_cache=false` for all 5 REDUCED rows.
   No cache exists to key/invalidate.
6. **Decoded-state / short outer-search A/Bs (task §8-10)**: not run this session as of this
   snapshot — blocked behind items 3-5 above per the task's own gate ordering (§10 "run only after
   all preceding gates pass").
7. **Coordinate-mode tournament (task §11)**: `:profiled_powered_relative_A` exists (item 2 above)
   but has not cleared the full production-context gate list, so a tournament against native
   REDUCED is not yet appropriate to run.

## Relationship to the parallel fixed-state inner A/B task

Per this task's own §12, this session does not rerun or duplicate
`benchmark/profiled-fixed-state-inner-ab-2026-08-04` (a separate Claude's ownership). As of this
snapshot that branch/report was not yet available to read; this doc will be updated (not silently
assumed) once it lands.
