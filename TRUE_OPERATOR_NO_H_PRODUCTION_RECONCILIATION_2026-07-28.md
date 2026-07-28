# True no-H operator bundle: production reconciliation (2026-07-28)

## Remote clarification (important, corrects the task's own example commands)

This repo has **two** remotes:

```
origin  git@github.com:habibiscoding/Trade-Model-Robustness.git   (legacy, pre-migration)
cdw     git@github.com:edwardwiles/cdw.git                        (canonical, per explicit
                                                                     user direction 2026-07-22)
```

`origin/production/fullA-exact` is a **stale** ref frozen at the 2026-07-22
`safety/fullA-exact-pre-remediation` consolidation point (`670eac4`) — it does not reflect any of
the ~30 production merges landed since. The task's example reconciliation commands name `origin`;
those were run first for completeness, then re-run against `cdw`, which is the actual canonical
remote this project has used for every real production merge since 2026-07-22 (see memory
`reference-cdw-repo-going-forward`). All reconciliation below uses `cdw`.

## Canonical production state (`cdw` remote)

```
git fetch cdw --tags
git rev-parse cdw/production/fullA-exact
  -> 93f26df7dfba96dbc0be92223b5d829c357b8a61
```

Tip commit: "Session handoff doc + delta=1 upper/lower production smoke-test scripts (not yet
run)" — this is the exact commit the `campaign/five-family-bounds-2026-07-28` branch had reached
and merged to `cdw/production/fullA-exact` before that campaign was terminated (per this task's
own instructions: "The prior campaign has been terminated").

Local branch ref `production/fullA-exact` (this checkout) is stale at `b7435ee` (an ancestor of
`cdw/production/fullA-exact`) — a leftover from before the last `cdw` push landed. It is not used
for anything below; all ancestry checks are against the live `cdw/production/fullA-exact` ref.

## Ancestry: has canonical production advanced beyond the no-H branch's base?

```
git merge-base HEAD cdw/production/fullA-exact
  -> 93f26df7dfba96dbc0be92223b5d829c357b8a61   (== cdw/production/fullA-exact itself)

git log --oneline HEAD..cdw/production/fullA-exact   -> (empty)
git log --oneline cdw/production/fullA-exact..HEAD   -> 15 commits (this branch's own work)

git merge-base --is-ancestor cdw/production/fullA-exact HEAD  -> yes
git merge-base --is-ancestor HEAD cdw/production/fullA-exact  -> no
```

**Answer: no.** `cdw/production/fullA-exact` is exactly the commit this branch was built on top
of (`93f26df`); canonical production has not moved since. This is a **clean fast-forward**, not a
rebase or merge — no reconciliation of divergent history is required, and no risk of
mathematics-altering conflict resolution.

## No-H branch ancestry (`work/true-operator-bundle-no-H-2026-07-28`)

```
HEAD = 8b0c931940b33b7a2665659e16789aecd007cd2d
```

15 commits ahead of `93f26df` (cdw canonical tip), in order:

```
c350280  Hessian upper-only cleanup: remove no-op symmetrization for mirrored blocks
e292403  Legacy CC-H removal: audit, root-cause analysis, and target-architecture design (no code change)
2ffd4f2  Flexible-CM: eliminate priming-side dense economic-block fill (real fix, root-caused)
83f28c8  CM+ZC and origin-ZC: add and validate the same economic-block skip mechanism
eca6945  Add shared OperatorPsiBundle type (design + Hessian-side wiring, not yet integrated)
84a4198  Update session documentation to reflect real fixes (not just design)
6bd7196  Flexible-CM: wire the true no-H OperatorPsiBundle into production construction+priming
d7cedb3  Flexible-CM: operator-vs-dense-reference equivalence gate (real KNITRO D=4)
352d609  Gravity: trace the outer pivot algebra vs. the inner-solve gravity moment + real 100-point test
ff3c740  Gravity: resolve as same_equality_exactly_eliminated (scoped), retain the inner moment
23ea76e  Flexible-CM: structural + empirical field/allocation proof for OperatorPsiBundle
6a8fadc  All five families: wire and gate the true no-H OperatorPsiBundle (real KNITRO D=4)
ad1cd43  All five families: real D=20/W=100,000 equivalence gates -- ALL PASS
fe4775e  Final manifests, five-by-seven matrix, and master report -- all five families PASS
8b0c931  Add flexible-CM D=20 verification cross-check (Delta_dual/Delta_primal/KKT residual)
```

(Task's stated starting HEAD was `fe4775e`; the actual branch tip carries one additional
ad-hoc verification commit, `8b0c931`, added after that report was written — a pure addition, no
rewrite of prior commits. Reconciliation below covers the true tip.)

## Working tree status

```
git status --short   -> (empty; clean)
```

## Reconciliation action taken

None required beyond confirmation. Because canonical production has not advanced, no temporary
`release/five-family-true-operator-no-H-2026-07-28` integration branch, rebase, or merge commit is
needed — the existing `work/true-operator-bundle-no-H-2026-07-28` branch tip is already exactly
what a fast-forward of `cdw/production/fullA-exact` to it would produce. The mathematics is
untouched (no rebase means no re-application of commits onto a new base).

## Verdict

```
CANONICAL_PRODUCTION_REMOTE = cdw (git@github.com:edwardwiles/cdw.git)
CANONICAL_PRODUCTION_SHA    = 93f26df7dfba96dbc0be92223b5d829c357b8a61
CANONICAL_ADVANCED_BEYOND_BASE = no
RECONCILIATION_METHOD = clean_fast_forward_no_rebase_needed
NO_H_BRANCH_HEAD = 8b0c931940b33b7a2665659e16789aecd007cd2d
WORKING_TREE = clean
```
