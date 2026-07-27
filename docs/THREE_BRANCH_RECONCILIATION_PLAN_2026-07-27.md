# Three-Branch Reconciliation Plan — 2026-07-27

## Source packages

The task names three "session zip" deliverables. All three exist locally as real git branches
(the zips on Dropbox are archived copies of the same commits, per each branch's own provenance
docs) — reconciliation was done directly against the git history, not by re-extracting the zips.

| Task's named package | Actual branch | Tip | Worktree |
|---|---|---|---|
| `finish_operator_stack_and_cm_basis_session_2026-07-27.zip` | `port/finish-operator-stack-no-dense-G-and-CM-basis-diagnosis-2026-07-26` | `65f34b0` | `shared-inner-fg-operator-2026-07-26` |
| `shared_outer_a_gradient_session_2026-07-27.zip` | `feature/shared-outer-a-gradient-2026-07-27` | `9af2af3` | `shared-a-gradient-2026-07-27` |
| `cm_basis_diagnosis_session_2026-07-27.zip` | `diag/cm-basis-interval-orthonormal-2026-07-27` | `92b9d7b` | `cm-basis-diagnosis-2026-07-27` |

## Ancestry (established by `git merge-base`, not assumed)

All three branches, and the base `production/fullA-exact`, are on ONE linear lineage up to a
fork point, not three independent trees:

```
production/fullA-exact (f1fa8e7)
  -> port/finish-five-family-optimization-stack-2026-07-26  (29b1c79, 25 commits)
     -> port/finish-operator-stack-...-2026-07-26            (a69b32d, +13 commits: Phase A items 1/3/5/6/8)
        |
        +--(continues)--> 65f34b0  (+10 more commits: Phase A item 4 D20, Phase B, master report,
        |                            fold-in of 3 dispatched background agents' results)
        |
        +--(forks at a69b32d)--> feature/shared-outer-a-gradient-2026-07-27 (9af2af3, +5 commits)
        |
        +--(forks at a69b32d)--> diag/cm-basis-interval-orthonormal-2026-07-27 (92b9d7b, +1 commit)
```

`shared-outer-a-gradient` and `cm-basis-diagnosis` were literally *dispatched by* the
`finish-operator-stack` session as background agents from that same commit (`a69b32d`) — this is
documented in `finish-operator-stack`'s own `65f34b0` commit ("Fold in final results from all
three dispatched background agents"), which is why the fork point is exact and why the three
branches' own final docs already cross-reference each other's headline findings.

**Zero file overlap** between the three branches' own unique work (verified by
`git diff --name-only a69b32d <tip>` for each, `comm -12` pairwise): `finish-operator-stack`'s
13 unique files, `shared-outer-a-gradient`'s 27, `cm-basis-diagnosis`'s 13 — no filename appears
in more than one list. This is the reason reconciliation is a mechanical cherry-pick, not a
merge-conflict-resolution exercise.

## Reconciliation mechanics performed

```
git worktree add -b release/shared-FG-verification-and-A-gradient-2026-07-27 <path> 65f34b0
git cherry-pick 281e830 c5cfddc 88aa0c4 2781583 9af2af3   # shared-outer-a-gradient's 5 unique commits
git cherry-pick 92b9d7b                                    # cm-basis-diagnosis's 1 unique commit
```

All 6 cherry-picks applied with **zero conflicts** (predicted by the file-overlap check above,
then confirmed mechanically). Post-merge sanity: `smoke_no_dense_g_five_families.jl` (real D=4)
runs clean on the merged HEAD — flexible_cm/cm_plus_zc/zc_only all show `operator_FG_calls=5`,
`dense_economic_G=0`, `full_G=0` at their production defaults, matching each source branch's own
individually-gated claims. This is the first time these three sessions' work has been run
*together* in one process.

Net: **59 commits** ahead of `production/fullA-exact` (53 from the finish-operator-stack lineage,
6 cherry-picked). No squashing — every inherited commit's own gating evidence (each commit message
carries its own D=4/D=20 test results) stays attached to that commit, per this project's existing
convention (`git log` messages ARE the gate record for this codebase, established well before this
task).

## Commit classification

Per the task's required taxonomy. Grouped by logical unit (individual commit-by-commit rationale
for all 59 would be redundant with the commit messages themselves, which already carry gating
evidence — this table adds the *reconciliation* judgment on top).

| Range | Commits | Classification | Rationale |
|---|---|---|---|
| `f1fa8e7..29b1c79` (finish-five-family-optimization-stack: Phases A-F, flexible-CM/common-Frechet lookup kernels, transformed-A default, unrestricted allocation-free FG) | 25 | **ADOPT_UNCHANGED** | Direct child of production/fullA-exact, no rebase needed. Each commit carries its own MATCHED_AB_PASSED / D=4+D=20 evidence. Re-ran the tip's own smoke test post-reconciliation (see above) as an integration check rather than re-verifying each of the 25 individually — consistent with "don't reopen already-gated work" guidance. |
| `29b1c79..a69b32d` (finish-operator-stack Phase A items 1/3/5/6/8: shared economic operator + origin-ZC/CM+ZC operator FG, first operator-verification proof-of-concept) | 13 | **ADOPT_UNCHANGED** | Same lineage, same reasoning. Includes `1900635 REWORK: revert restricted dual-bank default to opt-in` — a prior session's own self-correction, kept as-is (it IS the rework, not a candidate for further rework). |
| `a69b32d..65f34b0` (finish-operator-stack tail: flexible-CM/common-Frechet economic-block retrofit, CM+ZC/origin-ZC default flip, no-dense-G proof, 3 dispatched-agent fold-in) | 10 | **ADOPT_UNCHANGED** | This is the branch tip used as the release branch's own base commit — nothing to reclassify, it defines the starting point. |
| `281e830` (fix `dest_contrib_incremental_o1!` D-vs-Ddest stride bug) | 1 | **ADOPT_AFTER_REBASE** | Cherry-picked clean. This is a real, previously-undetected correctness bug (silently wrong only when `D==Ddest`) — see task §4's own "Fix the unresolved square-layout bug" ask, already fixed by the source branch; adopted verbatim. |
| `c5cfddc`,`88aa0c4`,`2781583` (`economic_A_gradient!`, verified-state diagnostic, ZC-only wiring) | 3 | **ADOPT_AFTER_REBASE** | Cherry-picked clean; each individually gated (D=4 bit-identical, D=20 bit-identical for the underlying function) in the source branch's own commits. |
| `9af2af3` (shared-a-gradient deliverable docs) | 1 | **ADOPT_AFTER_REBASE** | Docs-only; several of these documents are themselves inputs to this task's own required deliverables (superseded/extended, not duplicated — see below). |
| `92b9d7b` (CM basis/contrast Phase C diagnosis) | 1 | **ADOPT_AFTER_REBASE** | Cherry-picked clean. Resolves this task's own §7 requirement with real D=20 evidence; see `CM_BASIS_AND_CONTRAST_DEFAULT_RECONCILIATION_2026-07-27.md` for the one correction layered on top (see below). |

**REWORK applied on top of the reconciled base this session** (not a reclassification of an
inherited commit, but a new commit correcting one inherited branch's own open question): the
cm-basis-diagnosis branch left `ORIGIN_CONTRAST_DEFAULT=inconclusive` because it found a real
doc/code mismatch (three prior docs claimed orthonormal was already default; the actual wired
driver default is `:anchored`) without instructions on which to trust. This task's own prompt
resolves that ambiguity explicitly ("Actual code defaults to anchored; update stale documentation
that claimed orthonormal was already default") — so the correct action is not further
investigation but a documentation-correction commit. See
`CM_BASIS_AND_CONTRAST_DEFAULT_RECONCILIATION_2026-07-27.md`.

No commit in any of the three source branches was classified **REWORK**, **REFERENCE_ONLY**, or
**DROP** on its own merits — the zero-file-overlap property meant nothing needed to be redone or
discarded to reconcile the three branches with each other. (Two commits — `a34198b`, a real
mid-stream bugfix on the base lineage, and `1900635`, a real mid-stream default-revert — are
themselves examples of a REWORK pattern *already applied by the source sessions before this
task*, correctly left in place rather than re-reworked.)

## What this reconciliation does NOT yet address

The three source branches' own final-verdict docs are explicit about what remained undone when
each session ended (`port_ready_not_merged` / `partial_merge` in every case, not `merged_all`):

1. Common-Fréchet's D=20 performance-harness crash (`nStatus=-400` in `archC_frechet_base_state`)
   — reproduced but not root-caused by the source branch.
2. `composite_gradient_at_fast_buffered`/`_pooled`'s hardcoded square `reshape(...,D,D)` crash
   under the real D=20 production default (`:exclude_row`, D≠Ddest) — found, disclosed, not fixed.
3. Persistent `build_lfix_base_cache!` (task §5) — not attempted by any source branch (explicitly
   deprioritized, see `PERSISTENT_LFIX_BASE_WORKSPACE_2026-07-27.md`).
4. Shared A-gradient wiring for flexible_cm/common_frechet/cm_plus_zc (3 of 5 families) — not
   attempted.
5. Operator verification for flexible_cm/common_frechet/unrestricted (3 of 5 families) — not
   attempted.
6. Threading gates (serial/4/8/10/20 workers) for the shared A-gradient — not attempted (source
   branch tested serially only).
7. `skip_cm_fill_ref` removal — blocked on (5).

These are addressed as new work on top of the reconciled base, tracked in the other deliverable
docs listed in the release's final verdict.
