# H_ZZ realistic resource gate (2026-07-28) -- interrupted, backend flipped on explicit instruction

Scoped-down version of the task brief's §8 "realistic five-family process-parallel resource plan"
(explicitly framed there as an "if you have time" item; the prior session did not attempt it at
all). Given time constraints this release, scoped to the two families that actually touch H_ZZ
(origin_zc, cm_meanzc) rather than all five: isolated solo baselines at `BLAS_THREADS=8` for each,
then a concurrent 2-process pair, then compare against the BLAS=1 numbers already collected
incidentally by this release's other gates.

## What ran

`origin_zc` solo, `BLAS_THREADS=8`, `:blas_gemm`: **157.7s**, `nStatus=-401` (feasible), n_eval=1,
n_grad=1 -- completed cleanly, comparable in shape to this same driver's BLAS=1 runs earlier this
session (73-164s range depending on point/budget).

`cm_meanzc` solo, `BLAS_THREADS=8`, `:blas_gemm`: **killed after ~12 minutes wall time**, still
inside its real KNITRO solve, having not yet produced a result. Every other run of this exact
driver in this release (all at `BLAS_THREADS=1`) completed in 60-185s. This is a large,
unexplained slowdown -- plausible cause (not confirmed): thread oversubscription between Julia's
own `-t 4` threads, `OPENBLAS_NUM_THREADS=8`, and KNITRO's own internal parallelism (the run's log
showed KNITRO's deprecated-option warnings for `par_blasnumthreads`/`par_lsnumthreads`/etc.,
confirming KNITRO manages some of its own internal thread pools independent of the ambient BLAS
thread count). Not diagnosed further -- killed to unblock the rest of this release's work per an
explicit user instruction to move on.

The concurrent 2-process pair (origin_zc + cm_meanzc simultaneously) was never reached.

## Decision

`ZC_GRAM_BACKEND_DEFAULT[]` was briefly flipped to `:blas_gemm` on an initial explicit user
instruction, then reverted back to `:reference` (this release's final state) after the cm_meanzc
slowdown above was reported -- the user preferred not to ship an unresolved slowdown risk. The
prior session's ISOLATED real-D20 evidence for `:blas_gemm` (~2.77x faster than `:reference` at
production width nx=210, BLAS threads>=8) remains real and promising, and this session's own
origin_zc solo run confirms it's not obviously broken -- but it is **not** backed by a confirmed-
safe realistic-concurrency picture, and the cm_meanzc slowdown is a real, live, unexplained data
point against routinely giving 8-10 BLAS threads to a cm_meanzc production process. `:reference`
(today's unchanged production default) ships this release; `:blas_gemm` remains a well-evidenced
candidate for a future session that resolves the open question below.

## Top follow-up for a future session

1. Diagnose the cm_meanzc BLAS=8 slowdown: is it thread oversubscription (try `-t 1` or fewer
   Julia threads alongside high BLAS threads), a KNITRO-internal threading interaction, or
   something else? A `py-spy`/`perf`-style profile of the live process next time would answer this
   directly rather than needing to infer from elapsed time alone.
2. If it is oversubscription, cm_meanzc (the widest of the four families, n_E=382/n_Z=210/n_C=950)
   may need a DIFFERENT BLAS-thread recommendation than origin_zc, rather than sharing one number
   -- test them independently at several `(Julia threads, BLAS threads)` combinations before
   trusting `OPENBLAS_NUM_THREADS=10` unconditionally for cm_meanzc specifically.
3. Complete the originally-scoped 2-process concurrent test (and ideally the full 5-family
   realistic plan the task brief describes) once (1)/(2) are understood.
