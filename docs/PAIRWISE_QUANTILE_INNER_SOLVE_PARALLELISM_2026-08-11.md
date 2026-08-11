# Where the inner-solve wall-clock actually goes, and the 1.52x that was sitting in `linsolver`

2026-08-11. Branch `feature/pq-free-mass-reparam-2026-08-10`, commit `8c1b1e1`.
Measured at real D=20 data, W=100,000, `cutoff_source=:frechet_theoretical`, 16 Julia threads.

## Summary

The claim "KNITRO's internal factorization is 80%+ of the inner-solve wall-clock" was **correct as
an accounting statement** -- at L=10 it is 96.6% -- but it was being used to imply something false,
namely that the time was therefore not ours to attack. It was. `linsolver auto` was selecting a
**serial** linear solver; switching to `ma97` is worth **1.52x at L=6 and 6.26x at L=10** -- an
L=10 inner solve goes from **1163 s to 186 s** -- with `Delta*` unchanged to ~7e-14 relative.

The L=6 figure badly understates the win, and the reason is Amdahl: at L=6 the factorization is
about two thirds of the solve, at L=10 it is 96.6%. **Tune this at the L you intend to run.**

The user pushed on the original claim on exactly the right grounds ("usually once I push Claudes on
that claim it eventually collapses and we learn it's something else"). What collapsed was not the
share but the fatalism attached to it.

## 1. The corrected accounting

L=10, W=100k, after the T3/T4 scatter rewrite and the packed-write fix
(`profile_pairwise_quantile_budget.jl`):

```
KN_solve wall                          1243.83 s   (nStatus=0)
  FG callbacks           9 calls          0.91 s   ( 0.1%)     0.102 s/call
  Hessian callbacks      8 calls         41.98 s   ( 3.4%)     5.247 s/call
  KNITRO itself (by difference)        1200.94 s   (96.6%)
prime_operator! (outside KN_solve)        0.29 s
```

Our own assembly work is now **3.5% of the solve**. Every remaining optimization of the Hessian
build -- sparsity exploitation, combo partitioning, HVP -- is bounded above by that 3.5%. This is
why the earlier round of assembly work (which was real: L=10 went 1084.6 s -> 922.5 s, and the
T3/T4 scatter alone was 12.03x at L=10) stopped paying: it had run out of its own budget.

The residual scales as a dense O(n^3) factorization. Controlled sweep
(`scaling_pairwise_quantile_knitro_residual.jl`) fits an exponent of **2.91** (local slopes
3.05-3.08), against n = 15,952 at L=10.

## 2. It is not the BLAS -- measured, not inferred

Three separate things had to be ruled out before the real cause was visible.

**MKL is present and live.** An earlier `ls` of the KNITRO lib directory suggested no MKL. That was
wrong: `strings libknitro1301.so` finds 65,260 MKL symbols -- it is *statically* linked.
`blasoption=intel` beats `blasoption=knitro` (netlib) by 3.4x, so the fast BLAS path was already
selected and already helping.

**No BLAS thread knob does anything.** Sweeping `OMP_NUM_THREADS` with everything else fixed
(L=6, `probe_pairwise_quantile_omp_threads.jl`, 2nd pass, JIT discarded):

| `OMP_NUM_THREADS` | wall | CPU | avg cores | `Delta*` |
|---|---|---|---|---|
| 1 | 31.23 s | 45.15 s | 1.45 | 0.00084757259947424639 |
| 10 | 29.12 s | 42.47 s | 1.46 | 0.00084757259947424639 |
| unset | 34.20 s | 48.67 s | 1.42 | 0.00084757259947424639 |

Bit-identical `Delta*`, identical iteration counts (`n_fg=5 n_hess=4`), wall-clock within noise.
`par_blasnumthreads` and `MKL_NUM_THREADS` behave the same way.

**The average-cores column is the diagnostic that cracked it.** ~1.45 average cores under *every*
configuration. Our Hessian callback is threaded across 16 Julia threads and accounts for roughly a
third of an L=6 solve, so it alone explains the 0.45 above one. KNITRO's own phase was therefore
running at **exactly one core**, no matter what was set. That is not a BLAS being given too few
threads; that is a code path that never calls a threaded BLAS at all.

## 3. It was the linear solver

KNITRO 13.0.1 ships MA27, MA57, MA86, MA97 and MKL PARDISO. `linsolver auto` was picking a serial
one. Same point, L=6, 2nd pass:

| `linsolver` | wall | avg cores | `Delta*` |
|---|---|---|---|
| `auto` (was) | 30.27 s | 1.48 | ...424639 |
| `mklpardiso` | 24.20 s | 3.22 | ...447494 |
| **`ma97`** | **19.00 s** | **3.67** | ...417949 |
| `ma86` | 21.30 s | 4.37 | ...410707 |

All four agree on `Delta*` to ~8e-15 relative -- this changes how the same system is factorized, not
what problem is solved.

**`ma86` is rejected despite being competitive.** KNITRO's own documentation calls it "parallel,
**non-deterministic**". `ma97` is the parallel *deterministic* solver, and it is also the fastest
here, so there is no trade-off to make.

### Thread count: the sweet spot is low, and 16 is worse than 4

With `par_lsnumthreads = par_blasnumthreads = N`:

| N | wall | avg cores |
|---|---|---|
| 1 | 25.84 s | 1.54 |
| **4** | **19.91 s** | **2.56** |
| 10 | 20.01 s | 3.77 |
| 16 | 26.62 s | 4.36 |

Two things worth keeping:

1. **`ma97` at one thread already beats `auto` (25.84 vs 30.27).** Part of the win is solver
   quality, not parallelism. Anyone reading this as a pure threading result will over-attribute it.
2. **N=16 is measurably *slower* than N=4** while burning 1.7x the cores. We thread our own
   callbacks across 16 Julia threads; handing KNITRO 16 more oversubscribes the box. Do not raise
   these to "use the whole machine".

Net at N=4: **1.52x for 1.7x the cores.**

## 4. `par_concurrent_evals`

The two kinds of parallelism have to be kept apart, and they were not:

- **KNITRO evaluating our callbacks concurrently -- we do NOT want this on the inner solve.** Our
  callbacks are themselves threaded (Hessian assembly, T1-T4 scatter). Two layers of parallelism
  multiply and oversubscribe.
- **KNITRO parallelising its own internal linear algebra -- we DO want this.** That is exactly what
  section 3 buys.

The shared `ek_inner.opt` had `par_concurrent_evals yes`. It was **inert** -- `par_numthreads 1`
gates it, and `n_fg`/`n_hess` are unchanged across every configuration measured above, so KNITRO
never did make a concurrent call. But it is no longer safe to leave implicit now that solver threads
are raised, so `ek_inner_pq.opt` sets it to `no` explicitly.

The **outer** opt file (`csw_outer_wallclock_sr1.opt`) is untouched: it already has
`par_concurrent_evals yes` with `par_numthreads -1`, which is the side where we want it on.

## 5. What changed

- **`full_aod_diag/ek_inner_pq.opt`** (new) -- `linsolver ma97`, `par_lsnumthreads 4`,
  `par_blasnumthreads 4`, `par_concurrent_evals no`. Differs from the shared file in exactly those
  four settings. Carries the full sweep in its own header.
- **`pairwise_quantile_checkpoint.jl`** -- new `inner_opt_override` kwarg, reusing the c10 driver's
  existing name and mechanism rather than inventing new plumbing. Resolves relative names against
  `full_aod_diag/`, hard-errors on a missing file, and logs the resolved path. `nothing` keeps the
  shared default.
- **`run_pairwise_quantile_production.jl`** -- `const INNER_OPT = "ek_inner_pq.opt"`, passed via the
  existing `common...` bundle.
- **`smoke_pairwise_quantile_outer_driver.jl`** -- now runs on `ek_inner_pq.opt`, so the gate covers
  the configuration production actually uses. **26 PASS / 0 FAIL.**

**This is deliberately NOT a change to the shared `ek_inner.opt`.** It was tuned on the
pairwise-quantile restriction's KKT structure (n ~ 16k, dense-ish restriction block) and has not
been measured on the other families. The finding is very likely to generalise -- `linsolver auto`
picking a serial solver is not PQ-specific -- but "likely" is not "measured", and the other families
have live campaigns.

## 5b. The L=10 result -- the one that matters

L=10, W=100k, `probe_pairwise_quantile_omp_threads.jl`, 2nd pass:

| `linsolver` | wall | CPU | avg cores | n_fg/n_hess | `Delta*` |
|---|---|---|---|---|---|
| `auto` | 1163.29 s | 1194.97 s | **1.03** | 9 / 8 | 0.003006857760821941 |
| `ma97` (10 threads) | **185.89 s** | 940.51 s | **5.06** | 8 / 7 | 0.0030068577608221483 |

**6.26x.** Both passes agree (auto 1197.85 / 1163.29; ma97 201.20 / 185.89), both `nStatus=0`,
both `VerifiedSolved`, `Delta*` agreeing to 7e-14.

Two features of this table are worth more than the headline ratio:

1. **`avg cores = 1.03` on the baseline is an independent confirmation of the 96.6% attribution.**
   Our callbacks are threaded across 16 Julia threads. If they were a meaningful share of the solve
   the average would sit well above one. At L=6 it is 1.45 (callbacks ~ a third of the solve); at
   L=10 it collapses to 1.03 (callbacks 3.5%). The by-difference profile and the core-occupancy
   measurement agree, by two completely different routes.
2. **CPU time FELL, 1195 s -> 940 s.** ma97 is not merely spreading the same work across more
   cores; it does ~1.27x less total work *and* parallelises ~5x. That is why the wall-clock ratio
   (6.26x) exceeds the core ratio (4.9x). It also converged in one fewer iteration.

At ~186 s per inner solve, L=10 outer search moves from "not viable" to "expensive but real"
(~100 outer evals in ~5 hours). Section 6 was written before this measurement and is superseded on
that point.

## 6. What this does and does not fix

At L=5 (production, `screen -S pq_prod_L5`) the outer loop runs ~17 s/eval and is healthy; 1.52x on
the inner solve is a straightforward speedup of an already-viable configuration.

At L=10 -- see section 5b, written after this paragraph and superseding it -- the measured gain is
**6.26x**, not 1.52x, and it does change the verdict: an L=10 inner solve is now ~186 s rather than
~19 minutes. Reducing `n` (the restriction row count, 15,570 of the 15,952 duals) remains the next
lever, but it is no longer the only thing standing between L=10 and a real outer search.

**Production selects the opt file by L** (`run_pairwise_quantile_production.jl`:
`PQ_L >= 8 ? t10 : t4`). Measured at W=100k:

| L | `auto` | best `ma97` | gain | avg cores (auto -> ma97) |
|---|---|---|---|---|
| 5 | 19.39 s | 15.07 s (t4) | 1.29x | 1.97 -> 2.64 |
| 6 | 30.27 s | 19.91 s (t4) | 1.52x | 1.48 -> 2.56 |
| 10 | 1163.29 s | 185.89 s (t10) | **6.26x** | 1.03 -> 5.06 |

The `auto` avg-cores column falling 1.97 -> 1.48 -> 1.03 as L grows IS the Amdahl story made
visible: the threaded part (our callbacks) shrinks from a half to a third to 3.5% of the solve.

t4-vs-t10 was not swept at L=10, so the L>=8 threshold is a reasonable interpolation, not a measured
optimum. Sweep it if a campaign runs at L in 7..9.

**The live L=5 campaign was deliberately NOT restarted onto this.** Its stages are wall-clock
budgeted (75/30/75 min), so 1.29x would buy more outer evaluations per stage rather than an earlier
finish; against that, delta=0.1 was already COMPLETE at `auto` quality, so a restart would leave the
campaign internally inconsistent across delta cells. There is also a footgun: `run_delta_cell` skips
any stage whose checkpoint file exists, and checkpoints are written every 120 s DURING a stage, so
killing mid-stage and re-running would silently treat a partial stage as finished.

## 7. Corrections to earlier claims in this workstream

Recorded because each was asserted before it was measured:

- **"86 s packed write = 46% of the L=10 solve."** A profiler artifact -- the profiler had
  hand-inlined the loop at top-level scope, where globals are non-concrete. Micro-benchmark:
  top-level 84.70 s, real callback serial 1.90 s, threaded column-walk 0.40 s. Right mechanism,
  wrong location. The threading/column-walk fix was still worth keeping.
- **"HVP is the L=10 lever."** Measured 4.4x *slower* (6084 CG calls vs 9).
- **"No MKL in KNITRO."** Wrong, from an `ls`; it is statically linked.
- **"No knob changes the single-threading."** Wrong -- no *BLAS* knob does. `linsolver` does.
- **"1.52x."** True at L=6, and quoted before L=10 had been measured. The L=10 figure is 6.26x.
  Tuning a solver at a cheap configuration and extrapolating to the expensive one understated the
  win by 4x here; it could as easily overstate it.
