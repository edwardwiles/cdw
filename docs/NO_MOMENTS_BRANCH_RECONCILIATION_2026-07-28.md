# Branch reconciliation — no-moments/no-composite-G task (2026-07-28)

Release state: **INHERITED_WORK_RECONCILED**

## Repo

Canonical repo: `/bbkinghome/edav/cdw` (remote `origin` = `git@github.com:edwardwiles/cdw.git`).
New worktree for this task: `/bbkinghome/edav/gravity_robustness/worktrees/release-no-moments-no-composite-g-2026-07-28`.

## Ancestry check (verbatim commands + output)

```
$ git fetch origin --quiet
$ git rev-parse production/fullA-exact origin/production/fullA-exact
f1fa8e770759c62b3f96c1024dd310f235ea463e
f1fa8e770759c62b3f96c1024dd310f235ea463e
```
Local and origin `production/fullA-exact` are identical — no drift to reconcile.

```
$ git merge-base --is-ancestor production/fullA-exact release/final-architecture-closure-and-production-merge-2026-07-27 && echo YES
YES
$ git rev-list --count production/fullA-exact..release/final-architecture-closure-and-production-merge-2026-07-27
118
```
The integration branch (tip `e6f9d29`) is a clean 118-commit fast-forward descendant of canonical
production (`f1fa8e7`). This is the base used for the new release branch.

```
$ git merge-base release/final-architecture-closure-and-production-merge-2026-07-27 agent/skip-cm-fill-ref-removal-2026-07-27
7d4b353dd8039fe9ddcf1895226d0f5e0828cf20
$ git merge-base --is-ancestor 7d4b353d... release/final-architecture-closure-and-production-merge-2026-07-27 && echo YES
YES
$ git rev-list --count 7d4b353d..release/final-architecture-closure-and-production-merge-2026-07-27
3
$ git log --oneline 7d4b353d..agent/skip-cm-fill-ref-removal-2026-07-27
bc6ffdb Goal 10: full root-cause report with real D=20 evidence and precise unrestricted-vs-restricted comparison
a7dc50e Goal 10: remove skip_cm_fill_ref mutable-Ref anti-pattern; skip mechanism itself found unsafe, reverted to always-fill for both flexible-CM and common-Frechet
```

The Goal-10 diagnostic branch (`agent/skip-cm-fill-ref-removal-2026-07-27` @ `bc6ffdb`) branches
from a point (`7d4b353`) that is 3 commits *behind* the integration tip. Those 3 integration-only
commits (`a506268`, `47fb98e`, `e6f9d29`) touch only `no_dense_g_counters.jl` (additive counter
fields) and 3 new doc files — **no overlap** with Goal-10's changed files (`cm_frechet_cplus.jl`,
`cm_frechet_level.jl`, `cm_frechet_lookup_production.jl`, `cm_hessian_architectures.jl`,
`cm_lookup_production.jl`, `cm_meanzc_production.jl`, `cm_production_bundle.jl`,
`diag_frechet_skip_cm_fill_test.jl`, `economic_operator.jl`, `operator_verification.jl`, 3 test
files, plus its own new doc). Confirmed by `git diff --stat` on both ranges before cherry-picking.

## Branch created

```
$ git worktree add -b release/no-moments-no-composite-G-production-2026-07-28 \
    /bbkinghome/edav/gravity_robustness/worktrees/release-no-moments-no-composite-g-2026-07-28 \
    release/final-architecture-closure-and-production-merge-2026-07-27
HEAD is now at e6f9d29
```

## Goal-10 commits selectively adopted

```
$ git cherry-pick a7dc50e972099cb8913aed48c33d7f163307a5c7 bc6ffdb7949ad862d52867685dcd768a2e47fddd
[release/no-moments-no-composite-G-production-2026-07-28 fb9e007] Goal 10: remove skip_cm_fill_ref mutable-Ref anti-pattern...
 14 files changed, 279 insertions(+), 159 deletions(-)
[release/no-moments-no-composite-G-production-2026-07-28 334e1b6] Goal 10: full root-cause report...
 1 file changed, 268 insertions(+)
```

Both cherry-picks applied cleanly, zero conflicts — consistent with the file-overlap check above.

Adopted:
- The **mechanical** part of `a7dc50e`: removal of the `skip_cm_fill_ref` mutable-`Ref{Bool}`
  anti-pattern, replaced by explicit `skip_fill::Bool` arguments threaded through call sites.
- The root-cause doc `bc6ffdb` (`docs/GOAL10_SKIP_CM_FILL_REF_REMOVAL_2026-07-27.md`) in full, as
  the authoritative diagnostic record this task's fix is based on.

**Explicitly NOT adopted as the intended endpoint**: `a7dc50e`'s substantive effect of reverting
flexible-CM and common-Fréchet to permanently "always fill dense columns" (skip disabled). That was
the correct, safe state *given* `_archC_prep_for_hessian!`'s dense-read dependency still existed.
This task's core fix removes that dependency, which reopens the question of whether skipping the
fill is safe — re-evaluated fresh in the follow-on phase (see master report), not assumed either
way.

## Resulting ancestry after cherry-pick

```
$ git rev-parse HEAD
334e1b6a9b2d59819034d00f549ac7335103eadb
$ git merge-base --is-ancestor production/fullA-exact HEAD && echo YES
YES
$ git rev-list --count production/fullA-exact..HEAD
120
```

`release/no-moments-no-composite-G-production-2026-07-28` is a clean 120-commit descendant of
canonical `production/fullA-exact`, containing all 118 integration commits plus the 2 selectively
adopted Goal-10 commits. No wholesale merge of either inherited branch was performed — this is a
targeted base + 2 cherry-picks, matching the task's "Do not merge either inherited branch wholesale
without checking current canonical ancestry" instruction.

**Release state reached: INHERITED_WORK_RECONCILED.**
