# Unrestricted BLAS-thread default selection — 2026-07-25

Task §5. Full detail/commit: `40f9c19`. Raw data: `key_results/blas_sweep_u_p0p1p2_2026-07-25.csv`,
`key_results/blas_sweep_u_p1p2_points.jls`.

## Method

Real D=20/W=80,000/`:exclude_row` context built once; P1 (early accepted point) and P2
(deepest/hardest accepted point) harvested from a real 150s trajectory from calibration
(`full_trace_ref`) — genuine production-visited points, not synthetic. BLAS in `[1,4,8,10,20]`,
30s solve budget per config, `pin_outer_algorithm=true` for a controlled comparison, at each of
P0/P1/P2. 15 real runs, real licensed KNITRO, 992s total.

## Result (n_eval = outer evals completed in the 30s budget, throughput proxy)

| Point | BLAS=1 | BLAS=4 | BLAS=8 | BLAS=10 | BLAS=20 |
|---|---:|---:|---:|---:|---:|
| P0 (calibration) | 5 | 8 | **10** | 8 | **10** |
| P1 (moderate) | 5 | 9 | **10** | 9 | **10** |
| P2 (hard) | 3 | **7** | 5 | 5 | 3 |

`best_gp`/`best_Delta` agree to 9-10 significant figures across all five BLAS settings within
each point, confirming BLAS thread count does not affect correctness.

## Decision

The task's own rule: set a centralized default only if one setting is "fastest or effectively
tied across P1 AND P2," doesn't regress P0 by more than 5%, and improves the complete inner solve.

**BLAS=8 wins decisively at P0/P1** (tied with BLAS=20 for best, ~2x over BLAS=1) — reconfirms the
source branch's own P0-only finding on current code. **At P2, BLAS=4 is best (n_eval=7) and
BLAS=20 actively regresses to tie BLAS=1** (both worst, n_eval=3) — a materially different,
non-monotone picture, consistent with oversubscription-like behavior once the workload shape
changes at a hard point. All P2 configs converge to essentially the same `best_Delta` (~0.98),
meaning `n_eval` at P2 is a fairly noisy single-run signal (3-7 evals in budget) — but that noise
is itself the reason not to force a default off P0-only evidence, exactly as the source branch's
own caution ("do not assume 8 remains best") anticipated.

**Verdict: BLAS=8 remains a validated, strong opt-in choice for calibration/near-optimal
workloads — NOT hard-defaulted.** The mechanism (`blas_threads` kwarg, wired via
`blas_thread_policy.jl`, default `nothing` = ambient/unset) is production-ready and available to
any caller (e.g. a continuation-schedule driver operating mostly near calibration) that wants to
opt in explicitly.

```text
UNRESTRICTED_BLAS_DEFAULT = unchanged (opt-in only, blas_threads=nothing default)
```
