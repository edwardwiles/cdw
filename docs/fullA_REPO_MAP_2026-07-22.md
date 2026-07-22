# Full-A_od repository map, 2026-07-22

Read-only census performed at session start from
`/bbkinghome/edav/gravity_robustness/gravity-fullA-postmerge-correctness` (host `demand.mit.edu`).
Nothing was deleted, stashed, reset, rebased, or merged during this census.

## 1. The production-line branch chain (fully linear, no divergence)

Verified via `git merge-base`/`git rev-list --left-right --count` (not assumed from any prior
handoff):

```
integration/fullA-final-production-merge  (2620097)
  │  +6 commits
  ▼
perf/fullA-allocation-cache-cleanup        (1279ed6)
  │  +21 commits
  ▼
audit/fullA-postmerge-correctness          (dc3196c)
  │  +5 commits
  ▼
perf/fullA-factorized-price-production     (204c266)   ← was HEAD at census; now 8bc… after this
                                                            session's own commits (see §5)
```

32 commits total from the base to the tip; **zero commits flow backward at any step** — confirms
none of the perf/audit work had been merged back into `integration/fullA-final-production-merge`
as of session start, and none of it diverges either (a straight-line history, not parallel forks
that need reconciling).

**Ahead/behind counts** (`git rev-list --left-right --count A...B`):

| Pair | A ahead | B ahead |
|---|---:|---:|
| final-production-merge ↔ alloc-cache-cleanup | 0 | 6 |
| alloc-cache-cleanup ↔ postmerge-correctness | 0 | 21 |
| postmerge-correctness ↔ factorized-price-production | 0 | 5 |
| final-production-merge ↔ factorized-price-production | 0 | 32 |

## 2. Worktree/branch state at census (before this session's edits)

| Branch | Worktree | HEAD @ census | Dirty? |
|---|---|---|---|
| `integration/fullA-final-production-merge` | `gravity-fullA-final-production-merge` | `2620097` | Yes — 3 modified result files (`cm_hessian_benchmark.csv`/`.jls`, `smoke_result.jls`) |
| `perf/fullA-allocation-cache-cleanup` | `gravity-fullA-alloc-cache-cleanup` | `1279ed6` | Yes — 1 untracked dir (`results/fullA_d4/cm_ckpt_smoke_test/`) |
| `audit/fullA-postmerge-correctness` | *(no dedicated worktree — see note)* | `dc3196c` | n/a |
| `perf/fullA-factorized-price-production` | `gravity-fullA-postmerge-correctness` | `204c266` | **Clean** |

**Note**: the worktree named `gravity-fullA-postmerge-correctness` — which the finalization
brief's handoff described as checked out on `audit/fullA-postmerge-correctness` — was actually
already on `perf/fullA-factorized-price-production` (the tip) at session start, clean. Per the
brief's own worktree-discipline rule ("prefer the existing clean worktree if available"), all of
this session's work was done directly in that worktree/branch — no new worktree or branch was
created.

`diag/fullA-d4-exact` shares `perf/fullA-allocation-cache-cleanup`'s exact HEAD commit but is a
**separate worktree** (`gravity-fullA-d4`) carrying a large pile of untracked continuation-11
scratch files (scripts, CSVs, docs) unrelated to this session's work — left untouched.

No stale hardcoded worktree paths were found in `test_*.jl` files that would crash if run (the
one cross-worktree reference found, in `c14_final_ab_benchmark.jl`, points at
`gravity-fullA-d20-canonical-rerun`, which exists and contains the referenced checkpoint file).

None of the 4 production-line branches have a configured upstream — nothing has been pushed to
`origin` for this line.

## 3. Other branches in the repository (25 `gravity-fullA-*` worktrees total)

Everything **merged into** the tip (`perf/fullA-factorized-price-production`) as of census:
`audit/fullA-postmerge-correctness`, `diag/fullA-d20-canonical-rerun`, `diag/fullA-d4-exact`,
`diag/fullA-d4-exact-jach-audit`, `diag/fullA-d4-exact-phase5-sequential`,
`diag/fullA-d4-exact-smoothed-consistent`, `integration/fullA-cm-parallel-production`,
`integration/fullA-d20-runtime-delta5`, `integration/fullA-fast-range-screen`,
`integration/fullA-final-production-merge`, `integration/fullA-negative-cache-audit`,
`perf/fullA-allocation-cache-cleanup`.

**Not merged** into the tip (separate, still-active experimental lines, out of scope for this
finalization task): `diag/fullA-d20-fast-infeasibility`, `diag/fullA-d20-inner-warmstarts`,
`diag/fullA-d20-qmc-delta1`, `diag/fullA-d20-range-screen-review`,
`diag/fullA-d20-warmstart-replay`, `diag/fullA-d4-exact-cm-conditioning`,
`diag/fullA-d4-exact-cm-hessian-arch`, `diag/fullA-d4-exact-cm-interval-hessian`,
`diag/fullA-d4-exact-common-marginals`, `diag/fullA-driver-delta5`,
`diag/fullA-inner-blas-threading`, `integration/fullA-common-marginals`,
`integration/fullA-d20-common-marginals`.

## 4. Which branch should become authoritative

**`perf/fullA-factorized-price-production`** (now carrying this session's finalization commits
on top of `204c266`) is the correct candidate: it is a strict superset of
`integration/fullA-final-production-merge`, `perf/fullA-allocation-cache-cleanup`, and
`audit/fullA-postmerge-correctness` (32 commits ahead, 0 behind, fully linear), it is the branch
this session did all its correctness/finalization work on, and it is already the most-advanced
tip in the repository's fullA production line.

## 5. This session's commits (finalization task, all on `perf/fullA-factorized-price-production`)

Itemized in `docs/fullA_ALLOCATION_CROSSDELTA_KB_GATE_2026-07-22.md` §Commits and in the final
`git log`/`git status` output at the end of this task. In brief: doc reconciliation (AUD-02/
benchmark-numbers supersession banners + `fullA_CURRENT_STATE_2026-07-22.md`), a real
production-blocking bug fix (`screened_eval`'s `exact_cache` keyword type), Phase 2B cache
counters + `maxit_override`, the `price_cache_backend` selector wiring (Phase 3), and the new
`:kbplus` backend (Phase 4/5) plus its correctness/benchmark scripts.

## 6. Stale branches/worktrees that may later be archived (not touched this session)

Candidates for a future consolidation pass, once someone re-confirms none of their own unmerged
work is still wanted:

- `diag/fullA-d20-*` diagnostic branches (fast-infeasibility, inner-warmstarts, qmc-delta1,
  range-screen-review, warmstart-replay) — each already has its own "final handoff doc" per
  their own commit messages, suggesting the investigations are complete and only the decision to
  merge-or-archive remains.
- `diag/fullA-d4-exact-cm-*` (conditioning, hessian-arch, interval-hessian) — common-marginals
  Hessian experiments, not part of this finalization task's scope.
- `integration/fullA-common-marginals` / `integration/fullA-d20-common-marginals` — a separate
  common-marginals integration line; whether/how it should merge with the factorized-price line
  is a real open question for the team, not something to guess at here.
- The large uncommitted continuation-11 scratch pile in `gravity-fullA-d4`
  (`diag/fullA-d4-exact` worktree) — should be triaged (commit what's worth keeping, discard the
  rest) by whoever owns that line, not silently deleted by a later automated pass.

## 7. Concrete, low-risk consolidation plan (proposal only — not executed this session)

1. Push `perf/fullA-factorized-price-production` (with this session's commits) to `origin` so it
   is not only-local.
2. Open a PR/fast-forward merge of `perf/fullA-factorized-price-production` into
   `integration/fullA-final-production-merge` — this is a clean fast-forward (no merge commit
   needed, no conflicts possible) given the fully linear history in §1, PROVIDED the team is
   comfortable treating `integration/fullA-final-production-merge` as "the" production branch
   going forward. If a different branch name is preferred as canonical, rename/re-point instead
   of creating a divergent merge.
3. After that fast-forward, `perf/fullA-allocation-cache-cleanup` and
   `audit/fullA-postmerge-correctness` become fully redundant (already ancestors) — safe to
   delete the branches (not the worktrees, until their own dirty state in §2 is triaged) once the
   team confirms no one has local work still pointed at them.
4. Separately triage the not-merged diagnostic/experimental branches in §3/§6: for each, either
   (a) merge if the work is production-ready and still wanted, (b) archive (tag + delete branch)
   if the investigation is complete and its conclusion is already captured in a doc, or (c) leave
   active if genuinely still in progress. This needs the branch owner's input, not a mechanical
   rule.
5. Only after 2-4: clean up the dirty worktrees noted in §2 (commit-or-discard the stray result
   files/untracked dirs), and prune worktrees for any branches deleted in step 3/4.
