# CM threaded Architecture-C release — 2026-07-25

Task §1.3/§6. Status: **MERGED** (`threaded_bins=true` is the live production default, ancestor of
`production/fullA-exact`, tagged `threaded-exact-hessian-production-ready-2026-07-25`). Cherry-picked
commits: `d3b17ea` (implementation), `89fb854` (wired as default). This session's additional
evidence: `154bb89`. Raw data: `key_results/cm_threaded_sweep_p0p1p2_2026-07-25.csv`,
`key_results/cm_sweep_p1p2_points.jls`.

## Source branch's own result (reproduced, cited not re-derived)

`hessian_cm_structured_v2!` (threaded Julia bin-table accumulation) vs. the original serial
implementation, at a real hard, cold-solved D=20/L=50 point (`hard_cm_point.jls`, 401 free
params, `:all_legacy`): **serial 5,387 ms/call → threaded 1,532 ms/call, 3.52x**, correctness to
~1e-13 (5/5 checks against production serial and a syrk-only serial variant).

## This session's addition: P0/P1/P2, complete-solve (not isolated callback)

Added a `threaded_bins::Bool=true` pass-through kwarg
(`build_cm_production_context` → `build_cm_bin_ctx`, reachable from
`run_cm_upper_checkpointed`) so serial-vs-threaded could be A/B'd through the **real production
entry point** rather than only via isolated Hessian-callback timing. P1/P2 harvested from two
independent real `run_cm_upper_checkpointed` calls at `:exclude_row`, D=20, L=50. Compared
`serial(BLAS=1)` / `threaded(BLAS=1)` / `threaded+BLAS4`, Threads.nthreads()=20, 25s solve budget.

| Point | serial | threaded | threaded+BLAS4 |
|---|---|---|---|
| P0 | n_eval=2, 117.5s wall | n_eval=2, 68.6s | n_eval=2, 60.7s |
| P1 | n_eval=**1** (no outer progress beyond seed) | n_eval=**4** (real progress: gp 0.9739→0.9689, Δ 0.685→0.879) | n_eval=4, same progress |
| P2 | n_eval=1 (stalled) | n_eval=4 | n_eval=4 |

**At P1 specifically, this is a genuine complete-solve difference, not merely a faster isolated
callback**: serial made zero real outer progress within its budget while both threaded configs
completed 4 real outer iterations and materially improved the incumbent. `best_Delta` agrees to
~1e-13 within each point across configs where progress was made — no correctness regression.
`threaded+BLAS4` was never worse than `threaded+BLAS1` and modestly faster at all 3 points (no
oversubscription regression from light BLAS parallelism layered on the Julia-thread bin-table
parallelism) — a directional lead for a future BLAS>1 CM default, not adopted here (single-trial,
narrower matrix than a hard default needs).

Caveat: this host is shared/multi-tenant (other users' real Julia/KNITRO jobs were running
concurrently during this session, confirmed via `ps aux`) — absolute wall-clock deltas carry some
contention noise; the eval-count/outer-progress comparison at P1 (serial-stall vs
threaded-progress) is the load-bearing, contention-robust finding.

## Verdict

```text
CM_THREAD_DEFAULTS = threaded_bins=true (production default, confirmed across P0/P1/P2 -- not
                      just the source branch's single hard point); blas_threads left ambient/opt-in
```
