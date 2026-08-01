# Source snapshot — profiled economic block, all-families port (2026-08-01)

## Repo / branch provenance

- Main repo: `/bbkinghome/edav/cdw` (remote `origin` = `git@github.com:edwardwiles/cdw.git`)
- Source diagnostic branch: `diagnostic/profiled-scales-unrestricted-outer-ab-2026-08-01`
  - Source worktree: `/bbkinghome/edav/gravity_robustness/worktrees/diagnostic-profiled-scales-unrestricted-outer-ab-2026-08-01`
  - HEAD at snapshot time: `f439109` "Corrected A/B results after gp-drift bugfix: profiled +4-8% at matched iterations"
  - Working tree at snapshot time: clean except one untracked dir, `results/profiled_ab_2026-08-01/` (bulky `.jls` A/B run artifacts, not tracked, left in place, not ported)
- `production/fullA-exact` HEAD at snapshot time: `cd17235` "Update real_data/noah_D20 pi/tau to 2018 (goods-adjusted), replacing WITS-era data"
- Confirmed: `production/fullA-exact` (`cd17235`) is a direct ancestor of the diagnostic branch — `git merge-base` of the two equals `cd17235` exactly, and 0 commits exist on `production/fullA-exact` that are not already on the diagnostic branch. The diagnostic branch is production plus exactly 32 additional commits of validated profiled-unrestricted work. No rebase or cherry-pick reconciliation is needed.

## New port branch

- New branch: `architecture/profiled-economic-block-all-families-2026-08-01`
- New worktree: `/bbkinghome/edav/gravity_robustness/worktrees/architecture-profiled-economic-block-all-families-2026-08-01`
- Created via `git worktree add ... -b architecture/profiled-economic-block-all-families-2026-08-01 diagnostic/profiled-scales-unrestricted-outer-ab-2026-08-01`
- Base commit: `f439109` (identical tree to the diagnostic branch tip)

## 32 commits carried from the diagnostic branch (production + profiled-unrestricted work)

```
f439109 Corrected A/B results after gp-drift bugfix: profiled +4-8% at matched iterations
1aeec43 CRITICAL FIX: profiled A/B harness let gp drift instead of holding it fixed
f483cae Parametrize gp-fraction A/B scripts; launch sweep at gp={0.995,0.99,0.988,0.985}
60042eb Add production's adaptive per-coordinate FD bandwidth; launch A/B at non-trivial gp start
1b4ec1b Fixed-iteration A/B (maxit=60, fast gradient): full still ~7.5x better, decisively [RETRACTED numbers, see 00_READ_FIRST_CORRECTION.md]
7560ca0 Build O(1)-incremental profiled gradient: 10-15x faster, validated to machine precision
fa61ee9 Run matched full-vs-profiled unrestricted outer A/B (upper, delta=1): full wins decisively [RETRACTED numbers]
5302d86 Wire profiled outer gradient (fixed-dual FD) + evaluator; D4/D20 gates all PASS
2abb119 Commit validated unrestricted profiled inner formulation and D20 omit-ROW gates
... (23 earlier commits back to merge-base cd17235)
```

## Corrected evidence baseline (from `profiled_scales_unrestricted_outer_ab_2026-08-01.zip`, fetched from Dropbox)

Fetched via:
```
rclone copy "dropbox:Gravity robustness/Analysis/Server Output/profiled_scales_unrestricted_outer_ab_2026-08-01.zip" <scratch> --progress
```

`00_READ_FIRST_CORRECTION.md` (read in full) supersedes the master report's sections 6/6b. Corrected, `gp`-genuinely-fixed A/B result (unrestricted family only, upper-bound direction only, one seed per point):

| Point | full Δ (n_grad) | profiled Δ (n_grad) | profiled better by |
|---|---|---|---|
| gp=0.99×calib | 0.15175 (61) | 0.14612 (61) | 3.85% |
| gp=0.985×calib | 0.36264 (60) | 0.33582 (57) | 7.99% |
| GT=5% continuation waypoint | 0.97312 (61) | 0.92158 (61) | 5.59% |

**The archive's own verdict, in its own words: `PORT_TO_RESTRICTED_FAMILIES = insufficient_evidence`** — "the corrected effect is real but from too small a sample (3 points, one A/B seed each, upper-bound direction only) to base a multi-family rewrite recommendation on."

### How this snapshot reconciles that verdict with this task

This task's own guardrails (no campaign launch, no production-default flip, explicit five-family equivalence/performance gates required before any merge candidacy) are structured as exactly the kind of additional evidence-gathering the correction doc says is missing. This port is being executed as validation/evidence-gathering infrastructure to convert `insufficient_evidence` into a real answer — not as an implementation already justified by the 4-8% unrestricted number alone. `PRODUCTION_DEFAULT_CHANGED = false` and `CAMPAIGN_LAUNCHED = false` hold throughout, per the mission brief.

## Active jobs at snapshot time — explicitly avoided

`ps aux` at snapshot time showed one live full-A-adjacent Julia process:

```
edav 3213513 ... julia -t 10 --project=. campaign_inputs/sigma3_W500k_2026-07-30/profile_full_table_2026-08-01.jl
```
running inside `/bbkinghome/edav/gravity_robustness/worktrees/campaign-sigma3-w500k-fullA-10x10-launch-2026-08-01` (started 10:10 today). Per this task's instruction not to interfere with any running full-A production campaign, that worktree and branch are left completely untouched by this port; the new port worktree/branch are fully independent (separate worktree, separate branch, no shared mutable state — KNITRO/Julia env vars are process-local).

## Julia toolchain

Per repo-wide guidance (memory `julia-toolchain-use-juliaup-not-shared-sw`), use `PATH="$HOME/.juliaup/bin:$PATH"`, not `/opt/shared_sw`. Confirmed `julia`/`juliaup` present under `~/.juliaup/bin`.
