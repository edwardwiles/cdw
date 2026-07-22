# Repository state after CM closure — 2026-07-22

Git-common-dir: `trade_robustness_modular/.git`. Remotes: `cdw`
(`github.com/edwardwiles/cdw`, canonical) and `origin`
(`github.com/habibiscoding/Trade-Model-Robustness`, retired/stale — 20+ commits behind on
`production/fullA-exact`, not used by this campaign).

## Two canonical trunks

| Branch | Commit | Status |
|---|---|---|
| `production/fullA-exact` | `8e51d0a14b86cd77ba86e7f27c8beb42b62c5905` | canonical, pushed to `cdw` |
| `production/sequential-linearized` | `9ec3e46b5c37e8d5c0c1342883c8df7f82e8e7c6` | canonical, pushed to `cdw` (untouched this session, per task scope) |

`production/fullA-exact` is one documentation commit (`8e51d0a`, "Add CM production state doc")
ahead of the tagged baseline `cm-production-ready-2026-07-22` (`ac710bc`) — pure docs, no source
change.

## The one current performance branch/worktree

| Branch | Worktree | Based on |
|---|---|---|
| `perf/fullA-cm-postclosure-2026-07-22` | `gravity-perf-fullA-cm-postclosure` | `cm-production-ready-2026-07-22` tag (`ac710bc`) |

Pushed to `cdw`. This is the only branch/worktree created this session; no subagent/child
branches were created alongside it.

## Retained safety tags

Pre-existing (from prior sessions, left untouched):
`safety/fullA-exact-pre-remediation-2026-07-22`, `safety/sequential-linearized-pre-remediation-2026-07-22`,
`pre-consolidation-2026-07-21`, `pre-final-merge-2026-07-21`,
`post-remediation-2026-07-22/fullA-exact`, `post-remediation-2026-07-22/sequential-linearized`,
`consolidation-2026-07-22/fullA-exact`, `consolidation-2026-07-22/sequential-linearized`,
plus 21 `archive/precleanup/*` tags from an earlier branch-cleanup pass.

Created this session:
`cm-production-ready-2026-07-22` (annotated, at `ac710bc`, pushed to `cdw`) — see
`docs/CM_PRODUCTION_STATE_2026-07-22.md` for its full content.

No new safety tags were needed for deleted material this session, because **no branches were
deleted** — see classification below. Only two already-redundant *worktrees* were removed (their
branches remain, untouched).

## Branch/worktree inventory and classification

Classification key: **(1)** ancestor already incorporated into a canonical trunk — safe, nothing
unique; **(2)** active work not incorporated — real diagnostic/experimental commits, left alone;
**(3)** ambiguous; **(4)** disposable generated-output worktree.

| Ref | Worktree | Class | Note |
|---|---|---|---|
| `main` | (none) | 1 | Old pre-work snapshot; ancestor of both trunks. |
| `experiments-derivatives` | (none) | 1 | Ancestor of both trunks. |
| `sequential-profiled-gravity` | `trade_robustness_modular` (main worktree, houses `.git`) | 1 | Ancestor of `production/fullA-exact` — the long-running dev trunk that fed the merge. Cannot be removed (it *is* the git-common-dir worktree); has real untracked generated output (`batch_out_*/`, `results/`, phase5 logs) from real past runs, left untouched (not this session's to clean up, and not redundant — no other worktree holds it). |
| `feature/sequential-inversion-perf` | `trade_robustness_modular_perf` | 1 | Ancestor of `production/sequential-linearized`. Worktree has untracked generated CSVs (`c11_Aod_theta_full_delta*_upper*.csv`, `.throttled_pids`) — real run output, not touched (out of scope: sequential-linearized). |
| `remediation/fullA-exact-2026-07-22` | *(worktree removed this session)* | 1 | Fully incorporated (ancestor of `production/fullA-exact`); worktree `gravity-remediation-fullA-exact` was byte-identical-redundant with the production worktree (same lineage, clean, nothing uncommitted) — **removed** via `git worktree remove`. Branch ref itself left in place (cheap, and matches the prior session's own stated policy of not deleting remediation branches without separate remote-verification sign-off). |
| `remediation/sequential-linearized-2026-07-22` | *(worktree removed this session)* | 1 | Same as above, for the sequential trunk. Worktree `gravity-remediation-sequential-linearized` **removed**; branch left in place. |
| `experiment/fullA-cm-pairwise-zero-cov` | `gravity-experiment-fullA-cm-pairwise-zero-cov` | 1 (branch tip) **+ 2 (worktree contents)** | The branch's own commit (`82dd485`) is directly on the `production/fullA-exact` line (Closure Phase 3C). **But the worktree has real uncommitted new files** (`c40_meanzc_outer_driver.jl`, `c40_section7_d4_three_arm_trial.jl`, `c40_section8_d20_fixed_point_trial.jl`, `c40_test_meanzc_inner_solve_gates.jl`, `docs/experiment_cm_pairwise_zero_cov/`) — a follow-on experiment in progress, not this session's own doing. **Left completely untouched** (not committed, not stashed, not deleted) — no instruction to act on someone else's in-flight uncommitted experiment, and it is unrelated to the CM production path. |
| `diag/fullA-d20-fast-infeasibility` | `gravity-fullA-d20-fast-infeasibility` | 2 | Real diagnostic branch, not incorporated (screening-approach investigation; production uses a different/already-merged screen). Left alone. |
| `diag/fullA-d20-inner-warmstarts` | `gravity-fullA-d20-inner-warmstarts` | 2 | Warm-start Policy 2 investigation — tested and rejected net-negative per prior memory; historical record, not merged, correctly so. |
| `diag/fullA-d20-qmc-delta1` | `gravity-fullA-d20-qmc-delta1` | 2 | QMC calibration follow-up. Untracked `full_aod_diag/logs/` (generated) present but the branch itself holds real unmerged commits — not solely a disposable-output worktree, so not removed. |
| `diag/fullA-d20-range-screen-review` | `gravity-fullA-d20-range-screen-review` | 2 | Range-screen review/handoff; real content, not merged (screens actually shipped came in via a different, already-merged path). |
| `diag/fullA-d20-warmstart-replay` | `gravity-fullA-d20-warmstart-replay` | 2 | Continuation of the warm-start investigation; real content. |
| `diag/fullA-d4-exact-cm-conditioning` | (none currently checked out) | 2 | CM conditioning-basis comparison; real content, no live worktree. |
| `diag/fullA-d4-exact-cm-hessian-arch` | (none) | 2 | CM Hessian architecture comparison; real content, no live worktree. |
| `diag/fullA-d4-exact-cm-interval-hessian` | (none) | 2 | Cumulative↔interval transform report; real content, no live worktree. |
| `diag/fullA-d4-exact-common-marginals` | `gravity-fullA-d4-c12-common-marginals` | 2 | Continuation-12 CM D=20 production results writeup; real content, informational (production CM path itself came in via the remediation branch, not this one). |
| `diag/fullA-driver-delta5` | `gravity-fullA-driver-delta5` | 2 | Direction-inversion bug + gamma-bounds diagnostics; real content, not (yet) merged as its own commit (the *fix* it describes was independently re-derived/re-applied via the remediation branch's `a69fb21`/F1 work, per the closure report's Phase 3C). |
| `diag/fullA-final-gates-2026-07-22` | `gravity-diag-fullA-final-gates` | 2 | Independent final-gates/end-to-end benchmark report — a *reference* the closure report reads (`fullA_FINAL_RESIDUAL_GATES_AND_ENDTOEND_BENCHMARK_2026-07-22.md`), correctly kept separate from the production branch (it's an audit, not a fix). |
| `diag/fullA-inner-blas-threading` | `gravity-fullA-inner-blas-threading` | 2 | BLAS/threading report (1.04x, not 2.9x — corrected policy default already reflected in production's `.opt` files); real content. |
| `diag/sequential-inversion-perf` | `trade_robustness_modular_diag` | 2 | Out of scope (sequential-linearized); real content, untouched. |
| `integration/fullA-common-marginals` | `gravity-fullA-common-marginals-integration` | 2 | CM integration branch, superseded by the later, more complete remediation-branch CM unification (`de69a80`, "Unify CM driver onto the verified architecture") — real history, not deleted since it documents the integration path taken. |
| `integration/fullA-d20-common-marginals` | `gravity-fullA-d20-cm-integration` | 2 | CM performance-cost writeup at real D=20/delta=1; real content, informational. |
| `archive/wip/*` (7 branches: `d20-c10-benchmark`, `diag-fullA-d20-canonical-rerun`, `diag-fullA-d4-exact`, `feature-common-marginals`, `feature-sequential-inversion-perf`, `fullA-fast-range-screen-integration`, `worktree-agent-ad78f629`, all suffixed `-2026-07-22`) | (none) | 2, already safety-preserved | Created by a **prior** session's own repo-consolidation pass specifically to preserve uncommitted WIP before deleting the worktrees that held it (see each branch's own commit message: "WIP preservation: uncommitted ... from ... worktree (2026-07-22 repo consolidation)"). Already exactly the safety-tag pattern this task asks for, just as branches instead of tags — no further action needed or taken. |

No branch was classified **(3) ambiguous** — every non-canonical branch traced cleanly to either
"fully incorporated" or "real, identifiable, still-relevant diagnostic/experimental work." No
branch was deleted this session. No worktree was classified **(4) disposable-generated-output**
in the strict sense (a worktree holding *only* generated output with a fully-incorporated
branch) other than the two remediation worktrees already covered under (1).

## Net worktree change this session

- Removed: `gravity-remediation-fullA-exact`, `gravity-remediation-sequential-linearized` (clean,
  fully redundant with the production worktrees).
- Added: `gravity-perf-fullA-cm-postclosure` (the one authorized performance worktree).
- Everything else: unchanged.
