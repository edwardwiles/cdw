# CM production state — 2026-07-22

Snapshot taken ahead of the internal D=20 common-marginal (CM) production campaign.

**Superseded pointer, updated by the launcher-closure session**: the canonical commit/tag below
are now stale. See `docs/SESSION_SUMMARY_2026-07-22_cm_production_readiness.md` for the current
canonical commit hash and tag (`cm-production-ready-2026-07-22-r3`) — that document is the single
source of truth for "what commit/tag do I launch from," reconciled directly against
`git rev-parse`/`git ls-remote` output, not carried forward by hand. Everything else on this page
(driver, schema, cache behavior, checkpoint semantics) is unchanged and still accurate.

## Canonical commit (as of the ORIGINAL 2026-07-22 CM-closure session — see note above for current)

- Exact commit: `ac710bc346eadf3c5adf60d483c4afe51846a367`
- = local `production/fullA-exact`
- = `cdw/production/fullA-exact` (remote already matched local before this session touched anything)
- = `remediation/fullA-exact-2026-07-22` tip
- `origin/production/fullA-exact` (`habibiscoding/Trade-Model-Robustness`, the retired repo) is
  **stale**, at `670eac4` — 20+ commits behind. `cdw` (`edwardwiles/cdw`) is the canonical remote
  per prior-session decision; this campaign does not depend on `origin` at all.
- Tags at this commit: `post-remediation-2026-07-22/fullA-exact` (pre-existing),
  `cm-production-ready-2026-07-22` (created this session, pushed to `cdw`).

## Branch cleanliness

`gravity-production-fullA-exact` and `gravity-remediation-fullA-exact` worktrees: clean
(`git status --short` empty). No production-relevant uncommitted work found in any worktree
across the whole `trade_robustness_modular` worktree family — see
`docs/REPOSITORY_STATE_AFTER_CM_CLOSURE_2026-07-22.md` for the full inventory. One prior
uncommitted stopgap (`full_gamma_range` kwarg) was found and stashed by the previous session
(documented in `docs/closure_2026-07-22/REMEDIATION_CLOSURE_AND_PROMOTION_REPORT_2026-07-22.md`,
"Discovered and handled" section) before this session began; its capability is already
unconditional production behavior via commit `a69fb21`, so the stash is very likely obsolete but
was preserved, not discarded.

## Active CM driver and production entry point

- Driver: `run_cm_upper_checkpointed` in `full_aod_diag/d4_exact/cm_checkpoint.jl:143`.
- Only wired direction today: `find_smallest=true` is hardcoded inside the function
  (`cm_checkpoint.jl:170`) regardless of what's passed — i.e. only the "cm_upper" direction is a
  real production path. The 3-chain / 4-delta campaign in this brief runs this direction on all
  chains.
- Context builder: `d20_real_setup` / `build_ad_context_real_d20` in `context_real_d20.jl`, D=20
  real-data economy, France focal (`baseIndex=2`).

## Checkpoint schema

Two independent schemas coexist in the same file tree, gated separately:
- `D20Checkpoint` (unrestricted path, `c10_d20_production_driver.jl`): `CHECKPOINT_SCHEMA = 3`.
- `CMCheckpoint` (CM path, `cm_checkpoint.jl`): `CM_CHECKPOINT_SCHEMA = 2`. Schema-1 CM
  checkpoints are rejected with an actionable error (`load_cm_checkpoint`); no legacy-schema
  resume path exists (confirmed by reading `load_cm_checkpoint`, `cm_checkpoint.jl:90-102`).

## Active value and gradient functions (CM path)

- Value + verification: `cm_production_value_verified` (`cm_production_bundle.jl:231`) — returns
  `(_, base, verify)`; `verify.Delta_dual` is the **canonical** `Delta_dual =
  -(mean(Psi(q*))+zeta*)` (fixed at `ab1c74f`, F1). `cb_F!` reads `Δ = verify.Delta_dual`
  directly (`cm_checkpoint.jl:293`) — confirmed by direct code read, not just report text.
- Gradient: `cm_production_gradient` (`cm_production_bundle.jl:202`), called from `cb_G!` with
  `base` reused from `cb_F!`'s `last_F_state[]` when the point matches exactly
  (`cm_checkpoint.jl:322-323`), else recomputed.
- Exception contract: inner-solve failures raise the typed `CMExpectedSolveFailure` (not a bare
  `ErrorException`) at both `cb_F!` and the terminal-point verification; both catch sites narrow
  to `e isa CMExpectedSolveFailure || rethrow()`, so a genuine programming bug propagates instead
  of being silently swallowed (Phase 3B of the closure report, live-tested 7/7).

## Active cache behavior (CM path) — verified by direct code read

- `bandwidth_cache` (a `BandwidthCachePolicy`): caches the finite-difference bandwidth `h` used by
  `cm_production_gradient`'s envelope calculation across calls at the same point. This is the
  **only** cache on the CM path.
- **No exact-point value cache and no successful-dual "bank" exist on the CM path** —
  `cm_checkpoint.jl` has neither `exact_cache` nor `bank` identifiers anywhere (confirmed by
  grep). This is unlike the unrestricted path (`c10_d20_production_driver.jl`'s
  `run_profile_checkpointed`/`run_polish_checkpointed`), which does thread an `exact_cache` and a
  successful-dual `bank` through `screened_eval`. The CM inner solve is cold at every distinct
  outer point, by the current architecture, not by omission — consistent with
  `gravity-robustness-warm-start-map` (memory): the inner CC dual solve is never warm-started in
  this family of drivers by design.
- **`cb_newpt!` / accepted-point checkpoint reuse does not exist on the CM path at all.**
  `run_cm_upper_checkpointed` registers only `cb_F!`/`cb_G!` (`cm_checkpoint.jl:333-334`) — there
  is no `KN_set_newpt_callback` call anywhere in `cm_checkpoint.jl`. Checkpoints are written
  inline inside `cb_F!` itself (`:new_best` when a verified-better incumbent is found,
  `:wall_interval` on a time trigger, `:stage_complete`/`:stage_complete_unverified` at the very
  end) — never as a separate post-hoc re-solve. **This means the "redundant inner solve purely to
  checkpoint" problem that the accepted-point-reuse optimization exists to fix (present on the
  sibling unrestricted-path drivers, `c10_d20_production_driver.jl`, counter
  `n_checkpoint_reuse_hits`, verified 10/10 in Phase 3F of the closure report) has no counterpart
  to fix on the CM driver — there is nothing to reuse into, because nothing redundant happens.**
  See `docs/REPOSITORY_STATE_AFTER_CM_CLOSURE_2026-07-22.md` for the full writeup of why no
  code change was made here this session.

## Known remaining operational concern

A real interrupt→resume shakedown of the CM checkpoint path (Phase 4 of the closure report, real
D=20/W=80,000/L=50/delta=1) produced a genuine hang: the process caught `SIGTERM`, printed
Julia's standard signal-handler stack dump, then remained alive and unresponsive for 100+ more
seconds, requiring a manual `SIGKILL`. Root-caused **away from** the known AUD-02
`par_concurrent_evals` regression (the `.opt` file in use still has `par_concurrent_evals yes`;
the captured stack lacks AUD-02's specific lock signature). Classified as the separately
documented "KNITRO driver runs can hang past their own declared timeout" behavior (heavy GC
activity — 151 cycles — recorded at the point of the dump is consistent with a severe GC stall,
though not proven as the single root cause). **Checkpoint/cold-verify correctness itself is not
affected** — both the control and interrupted-then-resumed checkpoints reverify to machine
precision (`|diff|` 3.3e-15 and 6.9e-18). Operational recommendation, carried into this session's
supervisor design: run under an external supervisory process that can detect (via heartbeat/CPU/
process-state polling, not a bare `timeout` wrapper) and restart a hung run — see
`docs/CM_PRODUCTION_LAUNCHER_2026-07-22.md`.

## Tests passed

Full detail and exact commands in
`docs/closure_2026-07-22/REMEDIATION_CLOSURE_AND_PROMOTION_REPORT_2026-07-22.md` Phase 6:
14/14 fullA-exact regression suites pass, plus Phase 1-5 live CM validations
(`c30`/`c24`/`test_cm_expected_solve_failure_typed`/`c31`/`c32`/`c33`/`c34`, all passing per the
counts quoted in that report). Not re-run in full this session (would cost real KNITRO wall time
for no new information); spot-verified instead by direct code read of the exact lines the report
cites (`Delta_dual` usage, schema constants, `cb_newpt!`/cache presence) — all confirmed accurate
against the current `ac710bc` tree.

## Production launcher command

See `docs/CM_PRODUCTION_LAUNCHER_2026-07-22.md` for the full supervisor/launcher and exact
3-chain launch commands.

## Launcher-closure update (same day, later session)

A follow-on session closed several launcher/bookkeeping gaps found in the supervisor built above
(not the CM mathematics, which this document's "Active value and gradient functions" /
"Active cache behavior" sections above are still the correct description of, unchanged):
return-path safety (`slog` moved to stderr so log lines cannot contaminate the checkpoint path
returned via command substitution), a proper 4-way stage outcome classification
(`clean_solver_completion` / `wall_budget_exhausted` / `probable_stall` /
`unexpected_process_failure` — wall-budget exhaustion is no longer misclassified as a fatal
failure), per-attempt `STAGE_DONE` sentinel scoping (a stale sentinel from an earlier restart or
campaign can no longer be mistaken for the current attempt's own completion), a nonempty-campaign-
directory guard, a reproducible (non-`hash()`-based) chain-perturbation seed, cross-delta
seed-provenance assertions, and a contrast-basis decision (`:orthonormal`, not `:anchored` —
see `docs/fullA_cm_conditioning_and_adaptive_grid_report.md` and
`docs/fullA_cm_hessian_architecture_report.md`). Full detail, exact final commit/tag, and the
supervisor smoke-test results are in
`docs/SESSION_SUMMARY_2026-07-22_cm_production_readiness.md`.
