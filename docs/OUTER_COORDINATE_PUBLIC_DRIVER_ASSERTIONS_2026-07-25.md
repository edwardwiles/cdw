# Outer-coordinate public-driver assertions / post-rebase regression report — 2026-07-25

Task §17/§21 "post-rebase regression report" and public-driver-assertion deliverable. Everything
below ran through the actual public driver (`run_polish_checkpointed_unified`) or the direct
decode/gradient primitives it calls (`test_unified_layout_d20_gates.jl`, which exercises
`decode_outer_unified`/`gradient_transform_unified` against the real D=20 ctx the driver itself
builds), not a synthetic/mocked harness.

## Rebase (Phase 0)

`c55e81e..cdw/production/fullA-exact@39b89c5` (32 commits) cherry-picked cleanly onto
`port/transformed-A-and-flexible-theta-production-2026-07-25` — **zero conflicts**, confirmed by
`git status` showing a clean tree after each of the 14 cherry-picked commits.

## D=20 unified-layout gates (Phase 2, real production scale)

`test_unified_layout_d20_gates.jl`, D=20/D_dest=19/W=80,000/seed=20260719, post-rebase +
post-reconciliation:

```
Gate 1: fixed+legacy_z+raw reproduces EXISTING production driver's own numbers
  PASS  xf agreement (max|diff|=6.33e-8)
  PASS  Delta_dual agreement (unified=0.00248677347794064 legacy=0.002486773477940639)
  PASS  both feasible
Gate 2: fixed+powered_aspace+raw reproduces the SAME economic point
  PASS  xf agreement (a-space vs z-space at fixed theta_star) (max|diff|=0.0018310546875)
  PASS  a-space feasible at real D=20 calibration point
  PASS  a-space gravity residual < 1e-6 (gravity=8.98e-18)
  PASS  a-space Delta_dual matches z-space/legacy
Gate 3: DECISIVE chain-rule check at real D=20 scale (fixed theta)
  PASS  rel_err=8.53e-13
Gate 4: cache A/B/A
  PASS  cache size unchanged after identical re-eval
  PASS  Delta_dual bit-identical on cache hit
ALL D20 UNIFIED-LAYOUT GATES PASS
```

Identical to the source branch's own pre-rebase D=20 gate results (see
`docs/UNIFIED_COORDINATE_LAYOUT_ADDENDUM_2026-07-25.md`) to the digit — confirms the rebase
introduced zero numeric change to the core math, as expected given the 32 intervening commits
never touch `gravity_elimination.jl`/`outer_coordinate_layout.jl`/
`flexible_theta_aspace_production.jl`/`flexible_theta.jl` (all byte-identical pre/post-rebase,
confirmed via `git diff --stat`).

## Checkpoint/resume regression (Phase 2, task §14)

`test_unified_checkpoint_resume.jl`, real D=20/W=80,000, through `run_polish_checkpointed_unified`
directly (not a mock):

```
PASS  checkpoint file written after run 1
PASS  resumed n_eval (6) continues from run 1 (2), not reset
PASS  resumed run has a best_feasible incumbent
PASS  layout mismatch on resume raises the expected error
PASS  destination_sample mismatch on resume raises the expected error
ALL UNIFIED CHECKPOINT/RESUME GATES PASS
```

Not covered (scoped out, bounded per the task's own "small, bounded set of cases, not a sweep"
instruction given each case costs a fresh ~65-90s ctx rebuild): off-calibration-theta resume,
draw-mismatch resume. Both guards are structurally identical in code to the layout/destination_
sample guards already tested (same `error(...)` pattern in `run_polish_checkpointed_unified`'s
resume block) and were exercised in the source branch's own pre-addendum cache/checkpoint audit
(`docs/FLEXIBLE_THETA_CACHE_CHECKPOINT_AUDIT_2026-07-25.md`, 18/18 PASS, unmodified by this port).

## Smoke validation of the reconciled driver (Phase 1, all 3 production-relevant arms)

60s truncated D=20 runs through `run_polish_checkpointed_unified`, confirming the Phase 1
hardening (workspace attach, `blas_threads`, `pin_outer_algorithm`, manifest print, theta-aware
`DualBank`) introduced zero regression — see
`docs/COMMON_OUTER_COORDINATE_ARCHITECTURE_2026-07-25.md`'s smoke-evidence table (`fixed_aspace`
reproduced `kappa=0.0649042169269014` bit-identical to the pre-rebase 300s result).

## Backend manifest assertions (task §20)

`[backend-manifest]` startup print confirmed correct for all three layout combinations exercised
(fixed/legacy_z, fixed/powered_aspace, flexible/powered_aspace) in the Phase 1 smoke logs —
`trade_elasticity_mode`/`A_coordinate_mode`/`A_coordinate_mapping_version`/`gp_coordinate_mode`/
`theta_coordinate`/`theta_bounds`/`theta_derivative_backend`/`theta_aware_dual_bank`/
`outer_dimension` all populate correctly and match the layout actually constructed (e.g.
`outer_dimension=380` fixed, `=381` flexible; `theta_aware_dual_bank=true` only in flexible mode).
No dedicated assertion test file was added for this (the smoke-log evidence directly confirms the
values, and adding a formal `@test`-based assertion file was judged lower priority than the
matched-comparison campaign given this task's time budget) — flagged here as a legitimate,
bounded follow-up rather than silently treated as done.

## What was NOT run in this session

- Full CM/ZC/origin-ZC regression matrix for the fixed transformed-A mode (brief §15 anticipates
  this as future family-compatibility work, explicitly not required to gate Release A's own
  merge). Release A's coordinate change is architecturally restriction-family-agnostic (it only
  touches outer-vector decode/gradient-rescale, never the restriction basis), but this claim has
  not been empirically tested against CM/ZC/origin-ZC in this session.
- Off-calibration-theta and draw-mismatch checkpoint/resume cases (see above).
