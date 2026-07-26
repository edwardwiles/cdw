# Shared winner-pair core-Hessian: release completion (2026-07-25/26)

## Release states reached

1. `IMPLEMENTED_ON_FEATURE_BRANCH` — `port/shared-winner-pair-core-hessian-production-2026-07-25`, commits `aac8ed3`/`e65d335`.
2. `VALIDATED_THROUGH_PUBLIC_ENTRY_POINTS` — reached by the prior continuation session.
3. `FINAL_RESTRICTED_GATES_PASSED` — reached this session: worker policy, both matched outer A/Bs, both checkpoint/resumes, full regression sweep.
4. `FAST_FORWARDED_TO_CANONICAL_PRODUCTION` — `production/fullA-exact`: `39b89c5` → `b40e0f4` (fast-forward, verified ancestor).
5. `TAGGED` — `shared-winner-pair-core-hessian-production-ready-2026-07-25`, pushed.
6. `POST_MERGE_SMOKE_PASSED` — all four families, real public drivers, `dense_core_fallback_calls=0`.

## Provenance

- Canonical repo: `github.com/edwardwiles/cdw` (`origin`), matches `reference-cdw-repo-going-forward`.
- Canonical production tip **before** this release: `production/fullA-exact` @ `39b89c5` — confirmed
  unchanged from the fork point the feature branch was built on (local == `origin/production/fullA-exact`
  both `39b89c5` at session start), so no rebase/selective-cherry-pick was structurally necessary:
  the two validated feature-branch commits (`aac8ed3`, `0732951` after cherry-pick;
  `e65d335`/`ce0e5e4`) were already direct children of the canonical tip.
- Release branch: `release/shared-winner-pair-final-merge-2026-07-25`, built via
  `git worktree add -b ... 39b89c5` + `git cherry-pick aac8ed3 e65d335`, then this session's own
  4 additional commits (worker policy, harness fixes, test fixes, counter instrumentation).
- Fast-forward performed via `git update-ref refs/heads/production/fullA-exact
  release/shared-winner-pair-final-merge-2026-07-25` in the canonical repo (NOT `git checkout` +
  `git merge`) — the canonical repo's working tree had unrelated uncommitted work from a different,
  concurrent session (`port/fixed-frechet-cdf-production-2026-07-25`, modified
  `cm_frechet_checkpoint.jl`/`run_frechet_upper_cdf_only.jl` + several untracked docs/scripts); that
  work was left completely untouched, and the fast-forward never checked out a different branch in
  that working tree.
- Ancestry verified: `git merge-base --is-ancestor 39b89c5 production/fullA-exact` → true, both
  before (`--is-ancestor <release_commit> production/fullA-exact` at the release branch's own tip)
  and after the update.
- Pushed: `git push origin production/fullA-exact` (`39b89c5..b40e0f4`) and
  `git push origin shared-winner-pair-core-hessian-production-ready-2026-07-25` (new tag). Both
  confirmed accepted by `origin` (no rejection, no force needed — genuine fast-forward).
- User explicitly pre-authorized this push mid-session ("if the gates and checks etc all pass then
  you should go ahead and merge into production... you don't need to ask me for permission"),
  superseding the general confirm-before-pushing default for this specific release.

## What changed this session (on top of the prior `PORT_READY_NOT_MERGED` state)

1. **Dynamic worker-count policy** (task §3) — see `WINNER_PAIR_DYNAMIC_WORKER_POLICY_2026-07-25.md`.
   Replaced hard-coded `workers=10` (all 3 families) with a piecewise resolver defaulting to 20
   when `>=20` Julia threads are available; re-confirmed via a fresh 20-thread rerun that 20 is
   never worse than 10 at either of two independent points.
2. **CM+mean/ZC matched outer A/B** (task §4) — found and fixed a real bug in the pre-existing
   benchmark harness (missing `eta_nu` in the calibration start point, non-meanZC-aware
   cold-verify) that made every meanZC arm fail with zero evaluations before this session; then ran
   the actual A/B. See `CM_MEANZC_WINNER_PAIR_OUTER_AB_2026-07-25.md`. **PASS.**
3. **Origin-ZC matched outer A/B** (task §5) — new harness adapted from an existing sibling script;
   ran the actual A/B. See `ORIGIN_ZC_WINNER_PAIR_OUTER_AB_2026-07-25.md`. **PASS** (clean win).
4. **Flexible-CM fresh-process checkpoint/resume** (task §6) — through the real public driver,
   with a genuine SIGKILL of the whole process group mid-run. See
   `FLEXIBLE_CM_WINNER_PAIR_CHECKPOINT_RESUME_2026-07-25.md`. **PASS.**
5. **Origin-ZC fresh-process checkpoint/resume** (task §7) — same procedure. See
   `ORIGIN_ZC_WINNER_PAIR_CHECKPOINT_RESUME_2026-07-25.md`. **PASS**, with runtime counters
   captured directly (`winner_pair_hessian_calls=51`, `dense_core_fallback_calls=0`).
6. **Runtime counter instrumentation added to all 3 public stage-runner CLIs** (a gap the
   checkpoint/resume gates themselves exposed): `cm_production_stage_runner.jl`,
   `originzc_production_stage_runner.jl`, `unrestricted_stage_runner.jl` now all call
   `print_core_hessian_counters()` before their `STAGE_DONE` sentinel, so every real production run
   reports whether the winner-pair backend actually executed, not just that it was configured.
7. **Full fixed-mode regression sweep** (task §8) — see `WINNER_PAIR_FINAL_REGRESSION_SWEEP_2026-07-25.md`.
   Found and fixed 2 stale hard-coded `workers=10` test assertions (an expected consequence of
   item 1, not a regression); found and disclosed 1 pre-existing, unrelated, out-of-scope test bug
   (byte-identical to the pre-port canonical tip, so not introduced by this work).
8. **Checkpoint backend-fingerprint policy** (task §9) — explicitly decided NOT to bump any
   checkpoint schema; see `CHECKPOINT_BACKEND_FINGERPRINT_POLICY_2026-07-25.md`.
9. **`CM_WINNER_BIN_CROSS` disclosed as a follow-up, not implemented** (task §10) — see
   `CM_WINNER_BIN_CROSS_FOLLOWUP_2026-07-25.md`.
10. **Canonical merge, tag, and post-merge public-driver smokes** (task §11) — see
    `POST_MERGE_WINNER_PAIR_PUBLIC_SMOKE_2026-07-25.md`. All four families, `dense_core_fallback_calls=0`.

## Final verdict

```
SHARED_WINNER_PAIR_H_EE = MERGED_ALL_FAMILIES

WINNER_PAIR_WORKER_POLICY = nthreads>=20 -> 20 | nthreads in [10,20) -> 10 | nthreads<10 -> nthreads
WINNER_PAIR_WORKERS_AT_20_THREADS = 20

CM_MEANZC_DENSE_FALLBACK_CALLS = 0   (winner-pair arm; dense arm's own 54 calls are the expected :debug_reference_requested reason)
ORIGIN_ZC_DENSE_FALLBACK_CALLS = 0   (winner-pair arm; dense arm's own 119 calls are the expected :debug_reference_requested reason)

FLEXIBLE_CM_CHECKPOINT_RESUME = pass
ORIGIN_ZC_CHECKPOINT_RESUME = pass

CANONICAL_ANCESTRY_VERIFIED = yes
PRODUCTION_TAG_CREATED = yes
POST_MERGE_SMOKE = pass

CM_WINNER_BIN_CROSS = justified_future_task
```

Commit range now on `production/fullA-exact`: `39b89c5..b40e0f4`.
Tag: `shared-winner-pair-core-hessian-production-ready-2026-07-25` (pushed).

## Deliverables in this package

- `SHARED_WINNER_PAIR_RELEASE_COMPLETION_2026-07-25.md` (this file)
- `WINNER_PAIR_DYNAMIC_WORKER_POLICY_2026-07-25.md`
- `CM_MEANZC_WINNER_PAIR_OUTER_AB_2026-07-25.md`
- `ORIGIN_ZC_WINNER_PAIR_OUTER_AB_2026-07-25.md`
- `FLEXIBLE_CM_WINNER_PAIR_CHECKPOINT_RESUME_2026-07-25.md`
- `ORIGIN_ZC_WINNER_PAIR_CHECKPOINT_RESUME_2026-07-25.md`
- `WINNER_PAIR_FINAL_REGRESSION_SWEEP_2026-07-25.md`
- `POST_MERGE_WINNER_PAIR_PUBLIC_SMOKE_2026-07-25.md`
- `CM_WINNER_BIN_CROSS_FOLLOWUP_2026-07-25.md`
- `CHECKPOINT_BACKEND_FINGERPRINT_POLICY_2026-07-25.md`
- raw logs, CSVs, and a SHA256 manifest (see `provenance.txt`/`key_results/` in the pushed Dropbox package)
