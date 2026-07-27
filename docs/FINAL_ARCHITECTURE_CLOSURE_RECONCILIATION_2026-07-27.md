# Final Architecture Closure — Reconciliation — 2026-07-27

## Inherited state, verified against git (not assumed)

Task named source: `winner_aware_her_master_session_2026-07-27.zip`, claimed
`release/winner-aware-HER-and-no-dense-G-inner-solve-2026-07-27 @ 76d2f27`, 106 commits ahead of
`production/fullA-exact@f1fa8e7`.

Actual worktree state at session start (`worktrees/release-winner-aware-HER-and-no-dense-G-inner-solve-2026-07-27`):
HEAD = `a35022e` (one commit past the task's stated `76d2f27` — a docs-only commit, "Phase master
report, performance A/B synthesis, SHA256 manifest, and final reconciliation status"), 107 commits
ahead of base, working tree clean. Treated `a35022e` as the true inherited tip.

## Canonical production check

```
production/fullA-exact (local)  = f1fa8e770759c62b3f96c1024dd310f235ea463e
origin/production/fullA-exact   = f1fa8e770759c62b3f96c1024dd310f235ea463e
```

Both match the release branch's own recorded base exactly. **Canonical has not advanced.** No
rebase or selective-adopt step is needed before building on `a35022e` — it is still a direct,
unrebased descendant of current canonical `production/fullA-exact`.

Current production tags (most recent first): `common-frechet-cdf-cm-plus-level-production-ready-2026-07-26`,
`fixed-transformed-A-production-ready-2026-07-26`, `flexible-theta-transformed-A-production-ready-2026-07-26`,
... (full list in `git tag` output, unchanged from what the master report already recorded).

## New release branch

`release/final-architecture-closure-and-production-merge-2026-07-27`, branched directly from
`a35022e` (no cherry-picks, no rebase — see above). Working worktree:
`worktrees/release-final-architecture-closure-2026-07-27`.

## Sibling-branch check: `feature/shared-economic-moment-state-builder-2026-07-27`

A second unmerged branch was found during reconciliation: `feature/shared-economic-moment-state-builder-2026-07-27`
(tip `bfd8ab4`), forked from `e772e81` (itself an ancestor of `a35022e` — confirmed by
`git merge-base --is-ancestor e772e81 a35022e`). Its 3 unique commits (`343a975`, `3f8114b`,
`bfd8ab4`) add allocation-focused improvements to `build_economic_moment_state!` and the
unrestricted compressed-factual hot path.

**Not merged into this closure branch.** Two independent reasons:

1. `build_economic_moment_state!` (`compressed_factual_buffer_reuse.jl`) already exists on the
   `a35022e` lineage itself — both branches independently extended the same common ancestor
   (`e772e81`) — so the core capability this task's Goal 1/5 needs (shared in-place economic
   moment construction) is **already present**, not missing.
2. `git diff --name-only` shows real file overlap between the two lineages' own changes
   (`compressed_live.jl`, `compressed_live_v2.jl`, `c10_d20_production_driver.jl`,
   `compressed_moments.jl`, `compressed_inner_alt_solvers.jl`) — a cherry-pick here is a genuine
   merge-conflict risk, not the "zero file overlap, mechanical" case the prior three-branch
   reconciliation benefited from. Pulling in a small allocation optimization at real conflict-
   resolution risk fails this task's own priority order (§1: correctness/stability first; do not
   spend this session chasing small allocation deltas).

Recorded as a **known, deliberately deferred** future-cleanup item, not silently dropped: the
sibling branch's allocation gates and audit docs remain available at
`worktrees/shared-economic-moment-state-builder-2026-07-27` for a future session to reconcile
properly (via its own dedicated 3-way diff/merge pass) once this closure lands.

## Current defaults confirmed by direct code read (not grep-assumed)

- `CM_FRECHET_INNER_FG_BACKEND_DEFAULT` (`full_aod_diag/d4_exact/core_exact_hessian.jl:223`) =
  `Ref{Symbol}(:dense_reference)` — **not yet flipped**, confirms task's Goal 3 premise.
- `economic_A_gradient!` (`shared_a_gradient.jl`) is already wired as the production A-gradient
  call for CM+ZC (`cm_meanzc_production.jl`), origin-ZC (`cm_originzc_production.jl`), common-
  Fréchet (`cm_frechet_cplus.jl`), and flexible-CM (`cm_production_bundle.jl`). **Unrestricted is
  the only family not yet calling it** — confirms task's Goal 4 scope is exactly the unrestricted
  family, not a five-family rewrite.
- `build_economic_moment_state!` already exists and is already the production hot path for the
  compressed/operator families (`compressed_live.jl:342`).

## Plan for this closure phase

Two parallel implementation workstreams, dispatched as background agents into fresh worktrees
branched from this reconciled base, matching this project's own established pattern for
independent, large, well-scoped work (see `THREE_BRANCH_RECONCILIATION_PLAN_2026-07-27.md` and the
prior phase's 4-way agent dispatch):

1. `agent/default-flips-and-operator-setup-2026-07-27` — common-Fréchet FG default flip (Goal 3),
   unrestricted shared A-gradient default flip (Goal 4), `moment_representation` explicit dispatch
   + removal of generic composite-G setup (Goals 5-6).
2. `agent/cmzc-ecz-refactor-2026-07-27` — CM+ZC E/C/Z block partition + new `H_CZ` primitive, shared
   direct `H_ZZ` for CM+ZC and ZC-only (Goals 7-8).

Remaining goals (unrestricted dense-reporting removal, obsolete toggle removal, counter completion,
5x7 matrix, decisive release gates, canonical merge) are handled directly on the integration branch
after both agent branches merge back in, to keep the merge surface small and centrally verified.
