# Threaded CDF-Only Structured Hessian Benchmark — 2026-07-24

Real D=20, D_dest=19, W=80,000, L=50, `destination_sample=:exclude_row`, `frechet_feature_set=:cdf_only`.
Machine: shared 208-core host, `uptime` load average ~140-150/208 throughout this session (heavy
but not saturated) — every wall-clock number below is an upper bound on intrinsic cost, flagged
explicitly per this project's standing convention, not presented as a clean isolated benchmark.
`OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1` exported for every run (project standing requirement).

n=1382 inner variables (ncore=382, ncm=D·L=1000), Hessian nonzeros=955,653 (complete upper
triangle), `KN_DENSE_ROWMAJOR` registration, `hessopt=exact`, algorithm=auto (resolved to
Interior/Direct at these points).

## Correctness (task brief §11 gate, applied to the threaded Hessian)

| Scale | max\|H_serial − H_threaded/syrk\| | Verdict |
|---|---|---|
| D=4 (synthetic), L=8 | 9.16e-16 (serial/gemm), 9.16e-16 (threaded/syrk) | PASS — machine precision |
| D=4 end-to-end Δ* (dense-solve vs threaded-solve) | diff=0.0 | PASS — bit-identical |
| D=20 real data, L=50 | 6.821e-13 (n=1382, 955,653 entries) | PASS — consistent with the D=4 result scaled by dynamic range |

## Hessian-callback-only microbenchmark (direct repeated calls, no KNITRO)

| Julia threads | serial (legacy) | v2 serial/syrk | v2 threaded/syrk | speedup vs serial |
|---|---|---|---|---|
| 1 | 3.69s median | 3.39s | 3.35s | 1.10× |
| 4 | 5.98s median¹ | 5.74s | 2.17s | 2.75× |
| 8 | 4.99s median¹ | 5.17s | 1.65s | 3.03× |
| 20 | 4.62s median¹ | 3.03s | 0.94s | 4.94× |

¹ Serial-baseline numbers vary run-to-run because they're not thread-count-dependent — the
variation reflects shared-machine contention noise across separate process launches, not a
methodological difference. The important column is the threaded-vs-that-run's-own-serial ratio.

Threading scales sub-linearly (4.94× at 20 threads, not ~20×) because the bin-table-building pass
this parallelizes is only part of the callback's cost — the O(NCORE²) `H_EE` syrk, the O(L·nO²)
congruence transforms, and the packing tail remain serial (small relative to the O(W·D·NCORE)
bin-table pass at these dimensions, but not zero). Consistent with `cm_hessian_threaded.jl`'s own
prior finding for the structurally-identical flexible-CM kernel.

Allocations: ~18.7MB per Hessian callback, flat across all thread counts (no allocation growth from
threading — thread-local scratch is preallocated once via `build_thread_local_scratch`, not
reallocated per call).

## Full inner-solve wall time (through KNITRO, `par_numthreads=1`/KNITRO-nt1 fixed on this axis)

P1 = `gp_target=(1-(kappa*+1e-4))^((sigma-1)/sigma)`, zfree=zfree* (the exact point the prior
port-prep session measured at ~660-1000s for `:cdf_power`). P0 = calibration. P2 =
`kappa*+2e-4` (nearby to P1).

**Methodological note found live in this session**: `cc_algo/inner_loop_functions.jl`'s
`inner_loop_initial_values` uses `obj.x` (whatever the *previous* solve on this `obj` instance
converged to, regardless of which θ produced it) as KNITRO's initial point whenever
`obj.use_cached_x && norm(obj.x)<1e6`. This means consecutive solves on the same `obj` are **never**
independent cold measurements unless `obj.x` is explicitly reset — and this auto-warm-start is
exactly what the real outer driver's reused `fpcx.ctx_cm.obj` benefits from in production. Both
numbers are reported below: a genuinely **cold** P1 (`obj.x` reset first, comparable to the
original ~660-1000s `:cdf_power` baseline, itself measured as a first-ever solve) and the natural
**warm-from-previous-point** number representative of real outer-search usage.

| Julia threads | P1 **COLD** (serial Hessian, first solve) | P0 (warm from P1) | P1 (threaded, warm from P0) | P2 (threaded, warm from P1) |
|---|---|---|---|---|
| 1 | 67.28s (9 iters) | 35.83s (5 iters, serial) | — | — |
| 4 | 67.28s² | 35.83s² | 16.26s (5 iters) | 17.75s (5 iters) |
| 8 | 62.38s (9 iters) | 34.07s (5 iters) | 13.58s (5 iters) | 15.34s (5 iters) |
| 20 | 41.53–51.52s (9 iters, run-to-run variance) | 20.58–24.48s (5 iters) | 8.62–10.99s (5 iters) | 8.62–11.43s (5 iters) |

² nt=4 process's cold-P1/warm-P0 numbers used the serial Hessian by design (only the Hessian
*construction* is threaded; these rows measure the "before threading" reference inside that same
process — consistent across nt=1/4/8 processes as expected since Julia-thread count doesn't affect
a serial code path).

**Headline finding**: `:cdf_only` at real D=20/W=80,000/L=50 is **already close to production-
feasible even before any threading** — a cold first-ever P1 solve is 41-67s (vs the documented
660-1000s for `:cdf_power`, a 10-20× reduction from the feature-set change alone, larger than the
5.1× cubic-scaling estimate implied, because `:cdf_only` also converges in fewer KNITRO iterations:
9 vs whatever `:cdf_power` needed at that same point). With the threaded Hessian, a warm (production-
realistic) new-point solve is **8.6-17.8s**, comfortably inside the §10 "preferred ≤30s" target and
well inside "required ≤60s."
