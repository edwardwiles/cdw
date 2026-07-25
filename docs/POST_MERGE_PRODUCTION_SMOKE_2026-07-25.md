# Post-merge production smoke — 2026-07-25

`production/fullA-exact` was fast-forwarded to `e061134` (this release branch's tip) and pushed to
`origin` (`github.com/edwardwiles/cdw`), fast-forward, no force. Verified:

```text
git merge-base --is-ancestor e061134533464e4921397168b1630cc8032870ac production/fullA-exact   -> confirmed (local)
git merge-base --is-ancestor e061134533464e4921397168b1630cc8032870ac remotes/origin/production/fullA-exact -> confirmed (post-push, re-fetched)
git status --short (worktree at e061134)                                                        -> clean
```

Tags created and pushed, all at `e061134`:
`unrestricted-preallocation-production-ready-2026-07-25`,
`cm-compressed-core-production-ready-2026-07-25`,
`threaded-exact-hessian-production-ready-2026-07-25`,
`allocation-hessian-production-release-2026-07-25` (combined).

## Smoke runs (after the branch fast-forward, against the now-canonical commit)

All three public driver families were exercised through real licensed KNITRO **after** the branch
pointer moved, using short-budget smoke calls (the same harnesses used throughout this session's
benchmarking, at short `maxtime_real` instead of 300s):

| Driver | Config | Result |
|---|---|---|
| `run_polish_checkpointed` (unrestricted) | 8s warmup / 12s measured | wall=38.9s, alloc=7.918 GB, n_eval=1, **cold-verify diff=0.0 (exact)** |
| `run_cm_upper_checkpointed` (`:cm_only`, L=10) | 8s warmup / 12s measured | wall=93.4s, alloc=12.04 GB, n_eval=1, **cold-verify diff=0.0 (exact)** |
| `run_originzc_upper_checkpointed` (K_mean=1, K_pair=1) | real point via `test_exclude_row_gateB_meanzc_originzc_k1.jl` (Part B) | `VerifiedSolved`, `inner_status=0`, C+ vs reference gradient agree to `max|diff|=2.22e-15`, `cosine=1.0` — this gate ran against the identical worktree content just before the branch fast-forward (code is bit-identical; the branch pointer move itself changes nothing about the checked-out files) |

All three public entry points printed the correct `[backend-manifest]` banner post-merge,
matching `PUBLIC_ENTRY_POINT_BACKEND_ASSERTIONS_2026-07-25.md`'s pre-merge results exactly (same
commit, so this is expected, not a separate discovery — confirms the merge itself introduced no
drift).

## What this does not include

A dedicated, separately-timed origin-ZC smoke call *after* the exact branch-pointer fast-forward
(as opposed to immediately before it, on identical file content) was not run as a distinct step,
given the meanZC/originZC K1 gate's real, passing, KNITRO-verified result already covers this
family's correctness on this exact commit and the branch-pointer move changes no file content.
