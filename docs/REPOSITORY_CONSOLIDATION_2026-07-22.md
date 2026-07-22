# Repository consolidation — final report (2026-07-22)

Host: demand.mit.edu. Repo: single shared repo, git-common-dir
`/bbkinghome/edav/gravity_robustness/trade_robustness_modular/.git`. All
`gravity-fullA-*`/`trade_robustness_*`/`gravity-production-*` directories under
`/bbkinghome/edav/gravity_robustness/` are worktrees of this one repo.
`Trade-Model-Robustness/` is a **separate, unrelated repo** (own `.git`) — untouched.

Full census, classification, and preservation logs: see `phase2_classification.md` and
`phase3_preservation_log.md` in this directory; raw data in `for_each_ref.txt`,
`branch_summary.tsv`, `full_log_graph.txt`, `worktree_list.txt`.

## Canonical branches and hashes

| Branch | Hash | Model |
|---|---|---|
| `production/fullA-exact` | `04dc4a7d4b7bc2cb20df57026db554c3f1e58413` (= tip of `perf/fullA-factorized-price-production` @ `5bccc51a53ed0ba766b5262d40fb2a8a033ed7e4` + 1 docs commit) | All bilateral A_od outer variables, exact gravity-pivot map, C+ default backend |
| `production/sequential-linearized` | `716be804018ffcd74b253cef6ed0029035f92023` (= tip of `feature/sequential-inversion-perf` @ `a6b22648561819247fac4e5f3da2336765eddbc9` + 1 docs commit) | Only France/focal A_od outer; rest recovered via `recover_lfd` LFD/share-inversion |

Both proven tree-identical to their source branch before the docs commit
(`git diff --exit-code`, 0/0 ahead-behind); after adding `docs/REPOSITORY_BRANCH_POLICY.md`
the sole tree difference from the source branch is that one file (verified via
`git diff --stat`).

**Archive-hash discrepancy resolved**: the prior finalization archive cited an unrecorded
"8bc…" prefix hash for the full-A tip. No commit in the repo matches that prefix
(`git log --oneline --all | grep '^8bc'` empty) — apparent transcription error. The actual,
directly-verified hash is `5bccc51a53ed0ba766b5262d40fb2a8a033ed7e4`.

## Worktree paths and clean status (final)

- `/bbkinghome/edav/gravity_robustness/gravity-production-fullA-exact` — clean, tracks `origin/production/fullA-exact`
- `/bbkinghome/edav/gravity_robustness/gravity-production-sequential-linearized` — clean, tracks `origin/production/sequential-linearized`

Full final `git worktree list --porcelain` in `final_worktree_list.txt` (15 worktrees remain,
down from ~35 — see disposition table in `phase2_classification.md` for what each retained
worktree is and why).

## Proof the two lineages stay separate

- Full-A entry point `full_aod_diag/d4_exact/c10_d20_production_driver.jl` constructs a
  D²-1-dimensional free `zfree` (all bilateral A_od).
- Sequential entry point `sequential_gravity/run_profiled_production.jl` constructs
  `free_idx = vcat(3, collect(4:3+D), 1)` — D+1/D+2 free params (focal-only) — and recovers
  the rest via `recover_lfd`.
- `docs/REPOSITORY_BRANCH_POLICY.md`, committed separately on both trunks, states the
  non-merge rule explicitly and is now the durable record of this separation.

## Old-branch → disposition mapping

Full table in `phase2_classification.md`. Summary:

- **28 branches tagged then deleted** (all `REDUNDANT_ANCESTOR_FULLA`/`_SEQUENTIAL`, i.e.
  fully reachable from one of the two canonical tips, plus 5 trivial pre-fork placeholder
  branches identical to `main`). Includes the entire reported full-A chain
  (`integration/fullA-final-production-merge` → `perf/fullA-allocation-cache-cleanup` →
  `audit/fullA-postmerge-correctness` → `perf/fullA-factorized-price-production`) — redundancy
  re-verified live via `git merge-base --is-ancestor`, not assumed from the prior archive.
- **7 `archive/wip/*` safety branches** created to preserve uncommitted worktree content that
  didn't fit either "fully redundant" or "canonical" — see `phase3_preservation_log.md`. Two
  contain real, never-reviewed fixes flagged for human follow-up (a `screen_meta.worst_o`
  crash fix, and in-progress `SEQ_LAST_DIAG` diagnostic instrumentation on the sequential
  canonical worktree itself).
- **~15 branches retained untouched** — completed diagnostics (own final commit already is a
  comprehensive handoff/report; not deleted per "unknown means retain"), and ambiguous
  active-feature branches with real unmerged content (`diag/fullA-driver-delta5`,
  `integration/fullA-common-marginals`, `integration/fullA-d20-common-marginals`,
  `diag/sequential-inversion-perf`) that are not provably superseded and are flagged for
  human review rather than guessed at.
- `main`, `experiments-derivatives` (identical to main), and `sequential-profiled-gravity`
  (the repo's MAIN worktree, houses `.git`) left completely alone.

## Annotated tags created (28 `archive/precleanup/*` + 2 milestone tags)

All 30 tags pushed to `origin`. Full list in `final_branch_list.txt` / via
`git tag -l 'archive/precleanup/*'` and `git tag -l 'consolidation-2026-07-22/*'`.

## Branches deleted (local only; no remote branch was deleted, per instructions)

28 — see the "tagged then deleted" list above / `phase2_classification.md`.

## Worktrees removed vs retained

- **Removed** (21): all worktrees whose branch was deleted above, after dirty-content
  preservation (either committed to an `archive/wip/*` branch, or externally archived with
  checksums, or both). Full list in the Bash history of this session; each is named in
  `phase3_preservation_log.md`.
- **Retained** (15, listed in `final_worktree_list.txt`): the two new canonical worktrees, the
  repo's main worktree (`trade_robustness_modular`), the canonical sequential worktree
  (`trade_robustness_modular_perf`), and the worktrees for every branch classified
  `COMPLETED_DIAGNOSTIC`/`ACTIVE_FEATURE`/ambiguous above.

## External backups (generated-only artifacts, never committed)

`repo_cleanup_2026-07-22/generated_archives/` — 11 tarballs/snapshots, all entry-counts or
sizes verified against source before worktree removal, `SHA256SUMS.txt` covers all of them.

## Unresolved / ambiguous, retained for human review

1. `diag/fullA-driver-delta5` — real, unmerged direction-inversion + incumbent-seeding fixes
   (D=20-validated per prior session notes), not yet merged into `production/fullA-exact`.
2. `integration/fullA-common-marginals` and `integration/fullA-d20-common-marginals` — real,
   substantial diffs from a common ancestor with the merged `integration/fullA-cm-parallel-production`,
   but NOT provably subsumed by it (not an ancestor relationship). Own commit messages suggest
   the integration was abandoned in favor of the other branch, but this is not verified.
3. The `archive/wip/diag-fullA-d20-canonical-rerun-2026-07-22` safety branch contains what
   looks like the actual fix for the previously-known, unpatched
   `c10_d20_production_driver.jl` crash on `screen_meta.worst_o`. Recommend deliberate human
   review + cherry-pick onto `production/fullA-exact`.
4. The `archive/wip/feature-sequential-inversion-perf-2026-07-22` safety branch contains a
   real, in-progress `SEQ_LAST_DIAG` diagnostic addition to `run_profiled_production.jl`
   (never committed on the actual canonical worktree). Recommend human review before merging.
5. Remote anomaly (reported only, not touched): `refs/remotes/origin/diag/fullA-d4-exact` on
   GitHub points to a different commit (`08b07d0`, = local `integration/fullA-cm-parallel-production`)
   than what the local `diag/fullA-d4-exact` branch (deleted, now `08b07d0`≠`1279ed6`) tracked
   as its upstream — the remote branch name and content have drifted apart under the same name.

## Remote push/upstream status

- `origin` (`git@github.com:habibiscoding/Trade-Model-Robustness.git`) — **this is the actual
  repo all of this work lives in** (the user had believed it was `edwardwiles/cdw`; it is not).
  Both canonical branches pushed with upstream tracking set; all 28 `archive/precleanup/*` tags
  and both `consolidation-2026-07-22/*` milestone tags pushed. No remote branch deleted, no
  force push used anywhere in this task.
- `cdw` (`git@github.com:edwardwiles/cdw.git`) — added as a second remote per the user's
  request. Only `production/fullA-exact` and `production/sequential-linearized` (+ the 2
  milestone tags) pushed there, as new branches — cdw's pre-existing `main` and history
  untouched. See memory note `reference-cdw-repo-going-forward.md` (in the Claude memory
  store, not this repo) for the full explanation left for future sessions.

## Final verification command output

See `final_worktree_list.txt`, `final_branch_list.txt`, `final_fsck.txt` in this directory.
`git fsck --full` shows only pre-existing dangling objects dated 2026-07-20/21 (verified: none
match any hash touched by today's branch deletions — they are old superseded-cache/stash-style
commits from before this session, e.g. "pre-merge stash: uncommitted overnight Continuation-11
driver diff").

## How to enter either production lineage

```bash
# Full-A exact (all bilateral A_od outer variables)
cd /bbkinghome/edav/gravity_robustness/gravity-production-fullA-exact
git status   # should show: production/fullA-exact, clean

# Sequential-linearized (France/focal-only outer variables, LFD share-inversion)
cd /bbkinghome/edav/gravity_robustness/gravity-production-sequential-linearized
git status   # should show: production/sequential-linearized, clean
```

To recover any archived branch's content later:
```bash
git worktree add <path> archive/precleanup/<name>-2026-07-22   # read-only historical state
git worktree add <path> archive/wip/<name>-2026-07-22           # preserved uncommitted work
```
