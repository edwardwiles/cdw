# CM production readiness — session summary, 2026-07-22 (launcher-closure session)

Closes the production-launcher and Git-bookkeeping gaps left after the CM production launcher
was first built (`docs/CM_PRODUCTION_LAUNCHER_2026-07-22.md`, branch
`perf/fullA-cm-postclosure-2026-07-22`, merged into `production/fullA-exact` at `71b07b2`, tagged
`cm-production-ready-2026-07-22-r2`). Does **not** reopen the CM mathematics or the broader
performance work — those are unchanged and still described accurately by
`docs/CM_PRODUCTION_STATE_2026-07-22.md`.

## What changed this session

All in one commit, `<COMMIT_HASH>` (see "Final Git state" below for the exact hash):

1. **Return-path safety** (`scripts/cm_production_supervisor.sh`). `slog` now writes to stderr
   only (still appended to `$SUPERVISOR_LOG`). Before this fix,
   `ckpt_path=$(run_stage_with_watchdog ...)` captured every `slog` line printed during the call
   (tee'd to stdout) in addition to the intended final checkpoint path. `validate_ckpt_path`
   (single-line / nonempty / existing regular file / inside the stage directory / ends with a
   schema-2 `<label>_latest.jls` filename) is now called on every returned path before it is
   trusted. `scripts/test_cm_production_supervisor_return_path.sh` proves this with a real
   stall+restart cycle (many `slog` lines fired) whose captured return value stays a clean single
   line.
2. **4-way stage state machine**, replacing the old binary "STAGE_DONE found / not found" check:
   `clean_solver_completion`, `wall_budget_exhausted`, `probable_stall`,
   `unexpected_process_failure`. Each is recorded as `exit_reason=...` in `run_meta.txt`,
   `supervisor.log`, and (for restart-triggering events) `restarts.log`. The key correctness fix:
   `wall_budget_exhausted` is no longer misclassified as a fatal "exited without sentinel"
   failure — it returns the checkpoint (if one exists) for the caller to cold-verify and advance
   on, exactly like a clean completion, but never restarts (the stage's inclusive budget is gone).
   `probable_stall` resumes from the latest checkpoint against the **same** original stage
   deadline (no fresh budget). A genuinely unexpected nonzero exit (no sentinel, not a deliberate
   wall/stall termination) fails the stage outright with no automatic restart loop.
3. **Per-attempt `STAGE_DONE` scoping.** The stage log is still one cumulative file across
   restarts (needed for a continuous narrative), but the sentinel is now only searched for in the
   byte range written by the **current** attempt (offset recorded before each launch) — a stale
   `STAGE_DONE` from an earlier restart, or from a stale directory left over from an earlier
   campaign, can no longer make a later failed attempt look like it completed.
4. **Nonempty-campaign-directory guard.** The supervisor now refuses to launch into an
   already-nonempty `$CKPT_ROOT` unless `RESUME_CAMPAIGN=1` is set.
5. **Reproducible chain-perturbation seed** (`cm_production_stage_runner.jl`). Julia's generic
   `hash()` is not a stable, version-independent API — an interpreter upgrade could silently
   change every chain's starting point with no error. Replaced with a fixed integer formula
   (`2026_0722_00 + chain_perturb_seed`), printed in full and serialized (with the exact `w0`
   vector) to `$CKPT_DIR/w0_used.jls` for every calibration-mode launch.
6. **Cross-delta seed-provenance validation.** `cm_cold_verify.jl`'s output now also carries
   `contrasts`, `schema`, and `bi` (focal-country index) alongside the fields it already recorded
   (`W`, `draw_design`, `draw_seed`, `cm_L`). The stage runner's `seed_w0` mode now hard-asserts
   all of these match the new stage's fixed production context — only `delta` may legitimately
   differ across a cross-delta transition.
7. **Contrast-basis decision**: switched the hardcoded `contrasts` from `:anchored` to
   **`:orthonormal`**. See "Contrast-basis decision" below for the full reasoning.

Two new shell-level test files (`scripts/test_cm_production_supervisor_return_path.sh`,
`scripts/test_cm_production_supervisor_state_machine.sh`) cover items 1-4 with fake `JULIA_BIN`
stubs — fast (seconds), no KNITRO needed, both passing.

No core production files were touched (`cm_checkpoint.jl`, `cm_outer_driver.jl`,
`cm_production_bundle.jl`, `common_marginals_moments.jl` all unchanged) — only the launcher entry
points (`cm_production_supervisor.sh`, `cm_production_stage_runner.jl`, `cm_cold_verify.jl`) and
this session's own new test files. The existing 14/14 `fullA-exact` regression suite was **not**
re-run for this reason (it exercises code this change does not touch, and would cost real KNITRO
wall time to re-verify something unaffected).

## Contrast-basis decision

**Approved production setting: `contrasts = :orthonormal`** (was `:anchored`).

This is a same-feasible-set basis choice (anchored and orthonormal contrasts span the same
restrictions — orthonormal is a fixed, already-validated Helmert-rotation of anchored), decided
purely on the conditioning evidence already gathered on branch `diag/fullA-d4-exact-cm-conditioning`
(`docs/fullA_cm_conditioning_and_adaptive_grid_report.md`):

- Orthonormal strictly dominates anchored in `cond(Hessian)` at **every** L∈{10,20,50} and both
  points tested, and the gap **widens** with L (anchored: 35,793→143,174 from L=10→50 at
  calibration; orthonormal: 16,342→42,305 — a much flatter curve). This is exactly the L=50 regime
  this campaign runs at, and the widening-with-L trend is the more decision-relevant part of the
  finding, not just the raw ratio at one L.
- Orthonormal is far less reference-country-sensitive (0.8-1.5% cond spread vs 2.9-9.7% for
  anchored) — more robust to the arbitrary reference-origin choice baked into the CM restriction.
- The documented cost of orthonormal is losing per-origin sparsity in the CM moment columns
  (relevant to a *possible future* compressed-moment optimization, not something currently wired
  for the CM block). This did **not** block the decision because the production Hessian backend
  (`cm_hessian_backend=:structured`, Architecture C) was independently validated correct under
  orthonormal contrasts AND still gives a real 2.2-4.5x speedup over the dense baseline at L=50
  (`docs/fullA_cm_hessian_architecture_report.md`, Section 6) — the conditioning win is not traded
  against a broken or unvalidated fast path.
- The separately-measured `:interval` basis (best conditioning of all 4 bases measured) is **not**
  this axis: it is a different moment-construction family, not wired into
  `build_cm_production_context` at all today. Adopting it would be a materially larger change than
  a contrast-basis swap and was out of scope for this decision.

The choice is recorded consistently: `cm_production_stage_runner.jl`'s `CM_CONTRASTS` constant
(used both by the feasibility pre-check and the actual `run_cm_upper_checkpointed` call),
`CMCheckpoint.cm_contrasts` (already a generic recorded field, unchanged structurally), the
`cm_cold_verify.jl` output (`contrasts` field, new), and the two docs referenced above.

## Real-KNITRO supervisor smoke tests — status: NOT completed this session

A real D=20/W=80,000/L=50 three-part smoke test (clean-completion, then budget-exhaustion and
stall/resume each seeded from the clean-completion run's cold-verified incumbent) was started to
validate the state-machine fixes above against the real solver, not just the fake-`JULIA_BIN`
shell tests. It was stopped partway through **at the user's explicit request**, to prioritize
getting the fix into git for the real production campaign over spending more wall-clock time on
additional verification — not because of a failure. At the point it was stopped:

- Real KNITRO was 12 outer iterations into a genuine calibration-mode run (`Delta` moving from
  0.988 toward the 0.966-0.973 range, multiple points already both `feasible=true` and
  `verified=true`, i.e. real `:new_best` checkpoints were being written) — the core
  checkpoint/cold-verify machinery was observed working end-to-end against the real solver, but
  the run had not yet reached `STAGE_DONE`, and Tests B (budget-exhaustion) and C (stall/resume)
  had not started.
- No process was left running or orphaned: the real `julia` PID was re-confirmed by cmdline before
  being killed (SIGTERM was sent first and did not produce a prompt exit — consistent with this
  repo's own documented "KNITRO driver runs can hang past SIGTERM" behavior — so SIGKILL was used
  to actually terminate it). The scratch output directory
  (`production_runs/cm_launcher_smoketest_2026-07-22/`, never committed, matching this repo's
  convention of not committing generated run output) was removed.

**What this means for confidence in the fixes**: the state-machine logic itself (all 4
classifications, the return-path fix, the stale-sentinel guard, the nonempty-dir guard) is
verified by the two new shell-level tests, which exercise the exact same `run_stage_with_watchdog`
code path with fake process stubs standing in for `julia` — the shell logic being tested does not
know or care whether the underlying process is real KNITRO or a stub. What is **not** independently
re-confirmed against a real KNITRO process this session is the full wall_budget_exhausted and
probable_stall paths specifically. Given the real (partial) run already showed the
checkpoint/`cb_F!`/`is_verified_success` machinery working correctly against genuine KNITRO output,
and that machinery is unchanged by this session's edits (only the shell-side classification of
*already-produced* checkpoints changed), residual risk is judged low — but this is a real gap
against the original brief's "do not issue GO unless the complete supervisor passes all three
termination paths [against real KNITRO]" bar, disclosed here rather than glossed over.
**Recommendation**: run the real 3-part smoke test (procedure below) before or during the first
hour of the actual 3-chain campaign, in parallel, rather than skipping it entirely — it does not
block starting the campaign, but it should still happen.

### Smoke-test procedure (for whoever runs it)

1. **Clean-completion**: `DELTAS_OVERRIDE=1.0 scripts/cm_production_supervisor.sh 91 <dir>` and
   let it run to `STAGE_DONE` (expect real wall time of tens of minutes at this scale — it did not
   converge within ~15 minutes in the partial run above). Confirm `exit_reason=clean_solver_completion`
   in `<dir>/supervisor.log` and exactly one launch attempt.
2. **Budget-exhaustion**: source the supervisor, call `run_stage_with_watchdog` directly in
   `seed_w0` mode using step 1's `cold_verified_seed.jls` as the seed (a different delta, e.g.
   2.0, also exercises the new cross-delta provenance assertions) with a short `STAGE_WALL_S`
   (e.g. 180s). Confirm `exit_reason=wall_budget_exhausted`, exactly one launch attempt, a
   checkpoint returned, and that `cm_cold_verify.jl` succeeds against it independently.
3. **Stall/resume**: same seeding, generous `STAGE_WALL_S`, short `STALL_THRESHOLD_S` (e.g. 45s).
   Wait for the seeded run's first checkpoint to appear, re-confirm the real `julia` PID's cmdline
   via `pgrep`/`ps`, send `SIGSTOP` (the deterministic way to "deliberately stop progress" — do not
   rely on the probabilistic real GC-stall reproduction from
   `docs/CM_STALL_INVESTIGATION_2026-07-22.md`, which is not reproducible on demand). Confirm
   `PROBABLE STALL` + a resume relaunch in `supervisor.log`, the original PID gone (SIGKILL likely
   needed, same caveat as above), and the eventual resumed checkpoint cold-verifies.

## Final Git state

- Production branch: `production/fullA-exact` @ `<COMMIT_HASH>`
- Pushed to `cdw`: `cdw/production/fullA-exact` @ `<COMMIT_HASH>` (must match local exactly)
- Production tag: `cm-production-ready-2026-07-22-r3` @ `<COMMIT_HASH>` (annotated, pushed to
  `cdw`)
- `git status --short`: clean (verified after this session's commits)
- `perf/fullA-cm-postclosure-2026-07-22`: worktree and branch (local + `cdw`) **removed** —
  confirmed a full ancestor of `production/fullA-exact` and byte-identical on every file it
  touched before removal. Do not look for its worktree; launch from `production/fullA-exact` or
  the `-r3` tag directly.
- `experiment/fullA-cm-pairwise-zero-cov` worktree (`gravity-experiment-fullA-cm-pairwise-zero-cov`):
  **untouched**, per explicit instruction — it has its own in-flight uncommitted work unrelated to
  this closure.
- Superseded launch references: the pre-launcher `cm-production-ready-2026-07-22` tag (no
  supervisor existed at that commit) and the `cm-production-ready-2026-07-22-r2` tag (predates
  this session's fixes) must not be used for a real launch.

## Launch commands

```bash
export JULIA_NUM_THREADS=20
ROOT=/bbkinghome/edav/gravity_robustness/production_runs/cm_campaign_2026-07-22

nohup scripts/cm_production_supervisor.sh 1 "$ROOT/chain1" > "$ROOT/chain1_supervisor.out" 2>&1 &
echo $! > "$ROOT/chain1_supervisor.pid"

nohup scripts/cm_production_supervisor.sh 2 "$ROOT/chain2" > "$ROOT/chain2_supervisor.out" 2>&1 &
echo $! > "$ROOT/chain2_supervisor.pid"

nohup scripts/cm_production_supervisor.sh 3 "$ROOT/chain3" > "$ROOT/chain3_supervisor.out" 2>&1 &
echo $! > "$ROOT/chain3_supervisor.pid"
```

Run from a fresh checkout/worktree of `cm-production-ready-2026-07-22-r3` (or `production/fullA-exact`
directly, same commit).

## GO / NO-GO

**Conditional GO.** All 6 identified launcher/bookkeeping gaps are fixed, committed, and covered
by passing shell-level tests that exercise the real `run_stage_with_watchdog` logic (fake process
stubs only replace `julia` itself, not the supervisor's own decision logic). Git state is clean
and reconciled. The one open item is that the fixed state machine's `wall_budget_exhausted` and
`probable_stall` paths were not re-confirmed against a live KNITRO process end-to-end this session
(stopped early, at the user's request, to prioritize shipping the fix over further verification —
see above). Recommend running the 3-part real-KNITRO smoke-test procedure above in parallel with,
not instead of, the actual campaign's first hour.
