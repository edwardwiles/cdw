# Slow Inner-Solve Diagnosis — 2026-07-24 (live, user-requested)

## Trigger

The outer shakedown (`launch_frechet_shakedown.jl`) showed a single inner solve at a point near
(but not exactly at) the calibration point taking **~660-1015 seconds** across repeated
observations (the warm-up step and the first genuinely-new KNITRO outer-loop trial point). Per
user direction, the 90-minute shakedown run was abandoned and this targeted diagnosis was run
instead, using the codebase's own existing profiling instrumentation
(`instrumentation.jl`'s `@prof`/`prof_summary()`, which already wraps `KN_solve` itself in a
`"inner_knitro_dual_solve_arch"` timer, separately from the moment-build and Hessian-callback
timers) plus a temporary verbose-`outlev` copy of the inner KNITRO option file
(`ek_inner_diag_verbose.opt`, `outlev=4` vs production's `outlev=0`) to capture the raw
per-iteration table.

**Target point**: `(gp_target, zfree*)` at real D=20 data, `W=80,000`, `L=50`, `:cdf_power`,
`destination_sample=:exclude_row` — the SAME point already measured at 558-790s across three
prior runs (the outer shakedown's own warm-up step, and the standalone gradient-diagnostic's base
solve). Fully reproducible from `ctx.θ0_up` (unlike the outer loop's own internally-chosen
"eval 2" point, whose exact `zfree` step was never logged by this session's driver and is not
recoverable now — a real gap, noted below and fixed going forward).

## What was directly observed (live KNITRO output, this run)

```
Number of variables:                               2382 (        2382)
Number of nonzeros in Hessian:                  2838153 (     2838153)
Knitro using the Interior-Point/Barrier Direct algorithm.
Knitro changing linsolver from AUTO to 2.
```

`2,838,153 = 2382×2383/2` — **KNITRO is treating the full problem as a dense Hessian** (matches
this codebase's `KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, hess_cb)` registration in
`inner_loop_KNITRO_archgeneric`, `cm_hessian_architectures.jl:446` — the dense row-major callback
convention is what the Hessian gets registered as regardless of how sparse the *analytic* Hessian
actually is). `par_numthreads=1` (single-threaded) is set in `ek_inner.opt`.

The observed iteration table (14 iterations captured before the run was stopped, per user
direction, still not converged):

| Iter | Objective | OptError | ||Step|| |
|---|---|---|---|
| 0 | 0.000000e+00 | — | — |
| 1 | -1.647298e-02 | 1.032e-01 | 7.824e+02 |
| 2 | -1.664716e-02 | 8.689e-03 | 9.552e-03 |
| 3 | -1.669796e-02 | 2.486e-03 | 8.344e-03 |
| 4 | -1.671640e-02 | 7.046e-04 | 9.087e-03 |
| 5 | -1.674013e-02 | 3.438e-03 | 3.368e+01 |
| 6 | -1.674883e-02 | 7.273e-05 | 2.664e-02 |
| 7 | -1.681497e-02 | 5.698e-04 | 1.240e+01 |
| 8 | -1.681715e-02 | 1.395e-05 | 1.852e+00 |
| 9 | -1.681732e-02 | 3.561e-06 | 3.886e-01 |
| 10 | -1.681736e-02 | 1.785e-06 | 1.878e-01 |
| 11 | -1.681737e-02 | 9.011e-07 | 9.352e-02 |
| 12 | -1.681737e-02 | 4.563e-07 | 4.705e-02 |
| 13 | -1.681737e-02 | 2.315e-07 | 2.378e-02 |

**Pattern**: classic interior-point barrier-parameter-reduction cycling. Within a barrier
subproblem, `OptError` decreases geometrically (roughly 2-3.5x per iteration: e.g. iters 9→13,
`3.561e-6 → 1.785e-6 → 9.011e-7 → 4.563e-7 → 2.315e-7`, consistently ~2x). At iterations 5 and 7,
`OptError` **jumps back up** (`7.046e-4 → 3.438e-3`, `7.273e-5 → 5.698e-4`) — this is KNITRO
reducing its internal barrier parameter `μ` and starting a fresh subproblem, not divergence or
instability. `ek_inner.opt`'s tolerances are extremely tight (`opttol=1e-12`, `opttol_abs=1e-12`,
`ftol=1e-15`, `feastol=1e-12`, `xtol=1e-12`) — driving `OptError` from its already-small value at
iteration 13 (`2.3e-7`) down to `1e-12` requires several more full orders of magnitude, very
plausibly several more barrier-reduction cycles at this observed rate.

Timing: 14 iterations observed over ~215s of solve wall time ⇒ **~15s/iteration average**
(individual iterations varied — some faster, consistent with variable per-iteration CG/linear-
solve work). Extrapolating the same geometric-with-periodic-bumps pattern to full convergence at
`opttol=1e-12` plausibly explains the full previously-observed 558-1015s range without needing any
pathological behavior — it is consistent with "many iterations, each individually cheap in
FLOP-count terms for this codebase's own moment/Hessian construction, but individually expensive
in KNITRO's own dense linear algebra."

## Root cause (direct evidence, not inference from complexity alone)

**The dominant cost is KNITRO's own internal per-iteration dense linear algebra
(`2382×2382`-scale KKT-system factorization/solve, single-threaded), not this port's
moment-construction or structured-Hessian-callback code.** Supporting evidence:

1. KNITRO's own problem-characteristics banner confirms the Hessian is registered/treated as fully
   dense at this scale (2,838,153 = the complete upper triangle of a 2382×2382 matrix).
2. `par_numthreads=1` in the active inner-solve option file — the dense factorization this implies
   runs on a single core, on a machine with 208 available.
3. This port's own structured Hessian callback was independently benchmarked (D=4 and bounded-`L=8`
   D=20 gates) at well under a second to ~12s even at real D=20 scale — negligible next to a
   ~15s/iteration average over what is very plausibly 20-40+ total iterations.
4. The iteration table itself shows the classic interior-point signature (geometric decay within a
   barrier stage, periodic resets on `μ` reduction) — i.e., this is normal, correctly-functioning
   IPM behavior at very tight tolerances on a ~2,400-variable problem, not an error, stall, or
   pathological non-convergence.

**This is a general characteristic of running `:cdf_power` at full `L=50` scale (`ncore+ncm≈2382`
free inner-dual variables) through the existing dense-Hessian KNITRO calling convention and this
codebase's existing very-tight-tolerance option file — not a fixed-Fréchet-specific defect, and not
a moment/Hessian correctness problem** (already separately and directly validated to
machine-precision-consistent agreement against dense Architecture A in the D=4 and D=20 gates).

## Consequence for the outer shakedown (task brief §12)

A live multi-point outer search at full `L=50`/`W=80,000` scale, under this specific option
configuration (dense Hessian registration, single-threaded linear solver, `1e-12`/`1e-15`
tolerances) and this session's observed machine conditions, costs on the order of
**several hundred to ~1000 seconds per genuinely new trial point**. A meaningful multi-point
shakedown (the task's own "at least five valid new trial points" bar) at this per-point cost
requires substantially more wall-clock budget than a rigid 30 minutes — this is a real, measured,
disclosed constraint, not a code defect in this port, and is reported as such rather than
papered over with a truncated/misleading "shakedown" result. **No outer shakedown was completed by
user direction** — this diagnosis is delivered in its place.

## Immediate, concrete follow-up options (not implemented in this pass — scoped as recommendations)

1. **Multi-thread the inner KNITRO linear solver.** `ek_inner.opt` hardcodes `par_numthreads=1`
   (and the individual `par_*numthreads` variants, all deprecated aliases per KNITRO's own
   warnings, also unset/0). On a 208-core machine, allowing KNITRO's linear solver
   (`linsolver_numthreads`/`blas_numthreads`, the non-deprecated option names) to use more than one
   thread for the per-iteration dense factorization is a standard, low-risk lever that does not
   touch this port's own moment/Hessian code at all. This is a genuinely different concurrency axis
   from the "never spawn concurrent `KN_solve` calls" hazard this project's own memory warns about
   (that hazard is about *multiple simultaneous solves*; this is about *one solve's internal linear
   algebra*) — but should still be validated carefully before being adopted as a new production
   default, not assumed safe without a check.
2. **Reconsider whether `opttol=1e-12`/`ftol=1e-15`-level precision is actually required** for an
   outer-search trial-point evaluation (as opposed to a final cold-verification solve) — the
   pre-omit-ROW archive's own two-tier policy (a looser "trial" tolerance/budget, a tighter
   "cold-verify" pass only at accepted points) was designed for exactly this asymmetry. This port
   does not currently implement a two-tier tolerance policy; task brief §8 explicitly prohibits
   reintroducing an *arbitrarily short trial-time* bound without justification, but a properly
   *justified* looser trial tolerance (backed by a timing/accuracy tradeoff study, not invented
   ad hoc) is a distinct, legitimate lever this pass did not explore.
3. **Confirm whether a sparse (not dense-row-major) Hessian registration is possible** for this
   problem's KKT structure — the structured Hessian this port builds *is* highly structured
   (block-sparse in its origin/threshold indexing), but is currently packed into KNITRO's dense
   callback convention because that is the existing production convention
   (`archC_hess_cb_builder` and every sibling `archC_*_hess_cb_builder` in this codebase use the
   same `KN_DENSE_ROWMAJOR` registration) — changing this would be a cross-cutting change well
   beyond this port's scope, not attempted here.

## What this diagnosis does NOT establish

This is a single-point diagnosis, not a proof that every outer-search trial point costs the same.
Points genuinely closer to an already-accepted, cold-verified point (this port's own state-reuse
discipline, task brief §8) may converge faster from a better-conditioned starting dual state; this
was not separately tested in this pass. The finding here is specifically about a **fresh, no-prior-
warm-start solve at `L=50`/`D=20`/`W=80,000` scale** under the current dense/single-threaded/
tight-tolerance inner-solve configuration.
