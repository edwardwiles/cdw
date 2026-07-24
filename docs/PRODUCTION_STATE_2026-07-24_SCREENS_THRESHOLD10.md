# Production state — 2026-07-24 (screens restoration + threshold-10 early-abort)

Canonical record of the Part B (exact infeasibility screens restored to CM/CM+mean-ZC/origin-ZC)
+ Part C (delta_auto_reject_threshold=10 certified early-abort) release into
`production/fullA-exact`. Everything in `docs/CM_PRODUCTION_STATE_2026-07-23.md` and
`docs/CM_PRODUCTION_STATE_2026-07-22.md` about the CM driver/checkpoint mechanics/gradient backend
is unaffected and still accurate — this release is additive on top of that state.

## Release identity

- Release branch: `release/fullA-screens-threshold10-now-2026-07-24`, fast-forwarded into
  `production/fullA-exact` on both `cdw` (canonical remote) and local.
- Release tag: `screens-threshold10-production-ready-2026-07-24` (annotated), pointing at the same
  commit as `production/fullA-exact` after this merge. Use `git rev-parse production/fullA-exact`
  (or the tag) for the authoritative current hash.
- Provenance: selectively ported (cherry-picked, not merged wholesale) from
  `release/fullA-omit-row-restore-screens-2026-07-23@acae5c7` — that branch/worktree itself was
  left untouched (separate, concurrent omit-ROW work; see
  `docs/POST_OMIT_ROW_SCREEN_THRESHOLD_HANDOFF_2026-07-24.md`).
- Julia: 1.12.6 (juliaup). KNITRO: 13.0.1. D=20 real-data layout, D-by-D square (pre-omit-ROW).

## What changed

- **Screens** (`full_aod_diag/d4_exact/cm_screen_bridge.jl`): exact pairwise/hard-winner
  certificates now checked before every restricted-family (CM, CM+mean-ZC, origin-ZC) inner
  KNITRO solve, at every `run_cm_upper_checkpointed`/`run_originzc_upper_checkpointed` call site
  and both stage-runner preflight checks. Witness certificate remains opt-in
  (`use_witness=false` default; its cost/benefit benchmark has not been run). Live counters
  (`with_screen_counters`) now actually attached and threaded through every production call site,
  printed as a startup banner + end-of-run summary.
- **Threshold-10** (`cc_algo/threshold_early_abort.jl`): any inner-solve iterate whose canonical
  dual lower bound reaches 10 terminates KNITRO immediately via a clean `KN_RC_USER_TERMINATION`,
  producing a typed `CertifiedDivergenceLowerBound`. Active (`resolved_active_threshold=10.0`) for
  all current campaign deltas (0.1/0.5/1/2); auto-disables for `delta >= 9`. The pre-existing
  `lower_limit=-50` objective-poisoning backstop is unchanged and coexists. **Fixed this release**:
  `threshold_state` was silently dropped to `Inf` (disabling the feature) on all seven
  restricted-family objective-bundle rebuild sites — now correctly forwarded everywhere.
- **Active-layout accessors** (`cc_algo/active_layout.jl`): `active_origins`/`active_destinations`/
  `active_od_cells`, additive, zero behavior change today (fall through to `1:ctx.D`) — exists so
  the omit-ROW work can populate a reduced destination set without touching screen call sites.

## Startup diagnostics (new)

Every production entry point now prints, at the start of each run:
```
[screen-stack] mode=<unrestricted|cm_flexible|cm_plus_meanzc|origin_zc> enabled=true
[screen-stack] ordered active screens: <list>
[threshold-config] mode=<mode> requested_delta=<d> resolved_active_threshold=<t> stored_in_objective_bundle=<t>
```
and, at the end of each `run_*_upper_checkpointed` call:
```
[screen-summary] <label> calls=<n> pairwise_hits=<n> hard_winner_hits=<n> witness_hits=<n> points_passed=<n> inner_solves_avoided=<n> screen_wall_s=<t>
```

## Verification (real D=20/W=80,000 unless noted)

Full detail: `docs/SCREEN_STACK_FINAL_AUDIT_2026-07-24.md`. Summary: all four production
supervisor smokes pass (unrestricted, flexible CM, CM+mean-ZC via the real setsid/pgid supervisor
with checkpoint/kill/resume/cold-verify, origin-ZC); real-D=20 screen-restoration and
threshold-propagation regression tests pass; D=4 no-regression battery green except one confirmed
**pre-existing, unrelated** test-file bug (`test_cm_verified_success.jl`, stale include list,
reproduces identically on the untouched pre-release `production/fullA-exact@fd21f9f2`).

## Known pre-existing gaps found, not fixed (out of this release's bounded scope)

- `test_cm_verified_success.jl` (D=4): `UndefVarError: meanzc_resolve_K not defined`, stale
  include list predating `cm_meanzc_config.jl`.
- Four non-production diagnostic scripts (`c33_phase4_cm_shakedown_{control,interrupt,resume}.jl`,
  `cm_cplus_matched_trajectory.jl`) call `run_cm_upper_checkpointed` without including
  `cm_screen_bridge.jl` — will `UndefVarError` if run as-is. Production entry points are
  unaffected.
