# KNITRO Linear-Algebra Thread Benchmark — CDF-Only Fixed Fréchet — 2026-07-24

## Option names

Installed KNITRO: **13.0.1** (`/opt/shared_sw/knitro/13.0.1`, confirmed via `KNITRODIR` and
`.knitro_env.sh`; pinned per this repo's own standing note — 14.x lacks a valid site license on
this host). The live production option file (`full_aod_diag/ek_inner.opt`) uses the **deprecated**
KNITRO-12-era names (`par_numthreads`, `par_blasnumthreads`, `par_lsnumthreads` — confirmed still
defined, just marked "USE X" in `knitro.h`'s `KN_PARAM_PAR_*` block, not removed). Per the task's
instruction to use non-deprecated names, new experimental option files
(`full_aod_diag/d4_exact/frechet_bench_opts/ek_inner_nt{1,4,8,20}.opt`) were generated from the
production file with the deprecated thread lines stripped and replaced by the modern names:
`numthreads`, `blas_numthreads`, `linsolver_numthreads`. `ek_inner_nt1.opt` was diffed against the
unmodified production file and confirmed option-for-option equivalent at nt=1 (only the option
*names* differ, not effective values).

`par_concurrent_evals` was left at its production value (`yes`) — the file's own header comment
documents a prior live deadlock (`docs/fullA_nested_knitro_solve_hang_rootcause.md`) from setting
it to `no` with this codebase's nested-inner-solve architecture; not touched here.

## Activation confirmation (task requirement: do not infer from the option file alone)

A dedicated `outlev=1` rerun at nt=20 was captured. KNITRO's own printed option echo (not the
option *file*, the solver's own runtime confirmation):

```
blas_numthreads:         20
linsolver_numthreads:    20
numthreads:              1
```

`blas_numthreads` and `linsolver_numthreads` were genuinely accepted and active at 20. The general
`numthreads` knob echoed back as 1 despite the option file requesting 20 — plausibly because this
problem is unconstrained with no multistart/multi-algorithm parallelism in play
(`ms_enable=no`, `algorithm=auto` resolved to a single Interior/Direct run, `Number of constraints:
0`), so KNITRO's own presolve determined 1 is the effective value for that particular knob
regardless of the request. This does not affect the finding below, which is about
`blas_numthreads`/`linsolver_numthreads` specifically — the two options that actually govern the
dense KKT factorization's linear algebra, both confirmed active at the requested value.

## Benchmark: KNITRO thread count vs wall time, Julia-threaded(20) Hessian fixed, P1 cold each cell

(First attempt at this sweep was contaminated by the same `obj.use_cached_x` warm-start artifact
documented in `THREADED_CDF_ONLY_HESSIAN_BENCHMARK_2026-07-24.md` — every cell after the first
trivially reconverged in 0 iterations from the previous cell's already-converged solution at the
identical θ. Fixed by resetting `obj.x` before every cell; results below are the corrected rerun.)

| KNITRO threads (`blas_numthreads`=`linsolver_numthreads`) | wall | KNITRO-internal | iterations |
|---|---|---|---|
| 1 | 14.77s | 12.78s (86.5%) | 9 |
| 4 | 14.47s | 12.28s (84.9%) | 9 |
| 8 | 14.92s | 12.75s (85.5%) | 9 |
| 20 | 14.75s | 12.57s (85.2%) | 9 |
| 20 (outlev=1 activation-confirm rerun) | 14.65s | 12.36s (84.4%) | 9 |

## Finding

**KNITRO's own internal linear-algebra threading provides no measurable benefit at this problem
size** (n=1382; flat to within noise, 14.47-14.92s across the full 1→20 thread sweep, same
iteration count every time). This is consistent with the dense KKT factorization at this scale
(~1382³/3 ≈ 880M FLOPs) being small enough that thread-launch/synchronization overhead offsets any
parallel FLOP savings — a regime where single-threaded dense LAPACK/MKL routines are already
efficient. **Recommendation: leave KNITRO's own `blas_numthreads`/`linsolver_numthreads` at 1 in
production for the CDF-only inner solve** — the 20-thread setting is not harmful (no regression
observed) but adds no value and unnecessarily contends with the 20 Julia threads doing the actual
useful work (structured Hessian construction). The selected production backend is: **20-thread
Julia Hessian construction + KNITRO nt=1**.

This differs from what might be assumed for the larger `:cdf_power` problem (n=2382, ~5.1× more
factorization FLOPs) — that regime was not re-benchmarked in this session (deprioritized per the
addendum) and may cross the threshold where KNITRO threading helps; not claimed here either way.
