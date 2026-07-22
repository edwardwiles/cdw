# Root cause: outer-solve hang, fixed

Written 2026-07-22. This was reported as "the code doesn't even run" — a serious regression, not
an environment/load issue as first (wrongly) suspected. Full investigation below.

## Symptom

`test_driver_pooled_gradient_wiring.jl` and `test_cm_verified_success.jl` (both new end-to-end
wiring checks from this session's own earlier work) hung indefinitely — process alive, moderate
CPU%, zero new output — right after the KNITRO license banner printed and the outer solve's
first objective callback fired. Reproduced 4 times across a full session with escalating
diagnostics before the actual cause was found.

## What was ruled out (in order, each with real evidence, not assumption)

1. **Server load.** Initially misjudged: load average 42-58 looked alarming but this is a
   208-core host — that's ~20-28% utilization, not meaningful contention. Corrected after the
   user pointed this out.
2. **Outer solve's own thread count** (`par_numthreads=-1` → forced to `1`). No effect —
   identical hang, identical `SIGQUIT` stack trace.
3. **Inner solve's own BLAS thread count** (`ek_inner.opt`'s `par_blasnumthreads=0` → forced to
   `1`, alongside #2). No effect — identical hang, identical stack trace, third occurrence.
4. **Julia/BLAS-level threading env vars.** `docs/fullA_D20_checkpoint_resume_report.md`
   documents this exact function working on this exact machine (`demand.mit.edu`, confirmed via
   `hostname`) with `JULIA_NUM_THREADS=20`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.
   Reproduced that exact environment with the standard (unmodified) `.opt` files. No effect —
   identical hang, identical stack trace, fourth occurrence.
5. **Dimension.** A ~15-line, fully self-contained D=4 MWE (bare outer `KN_solve` whose callback
   calls `inner_loop_internal_profiled` once, no screening/checkpoint/driver machinery at all)
   reproduced the identical hang and stack trace in under 2 minutes instead of D=20's ~85s
   context build — confirming the deadlock is purely about the nested-`KN_solve` shape, not
   problem scale. (Per the user's own suggestion — this was the single most valuable diagnostic
   step, both for speed and for isolating the true minimal trigger.)
6. **Library corruption/recent patching.** `/opt/shared_sw/knitro/13.0.1/lib/{libknitro.so,
   libiomp5.so}` are unchanged since 2022/2018 respectively; `deps.jl`'s baked-in path correctly
   points at 13.0.1; `KNITRO.jl` package version (1.2.1) identical at both the historical and
   current commit.

## The actual cause

**Git bisection via direct A/B, not guessing**: the exact same D=4 MWE was copied into a
throwaway worktree checked out at commit `3855430` — the commit
`docs/fullA_final_production_merge_handoff.md` explicitly records as
"`PRODUCTION MERGE COMPLETE — READY FOR RUNS`". It ran the MWE cleanly: 54 callback calls,
real nested inner solves (mix of converged and `-300`/infeasible), outer solve converged in
1.49s. **No hang at the historical commit, same machine, same library.**

Diffing the ~23 commits between `3855430` and current HEAD for anything callback/threading-shaped
immediately surfaced `62dde8f` ("AUD-02: disable concurrent KNITRO evals; add cross-thread
callback guard"), which changed:

```diff
-par_concurrent_evals  yes
+par_concurrent_evals  no
```

in all three actively-loaded outer option files (`csw_outer_wallclock_{sr1,lbfgs,productfd}.opt`).
Reverting *only* this one setting on the current worktree (every other audit-remediation change
from this session left intact) reproduced the historical commit's clean run **bit-for-bit**: 54
callback calls, converged in 1.53s, identical objective value. Confirmed conclusively.

## Mechanism (best understanding, consistent with all evidence)

`par_concurrent_evals=no` tells KNITRO "only one thread may perform an evaluation at a time" —
AUD-02's intent was defense against a genuine hazard (two Julia threads racing on a shared
`PsiObjectiveBundle` instance). But this codebase's core architecture has always relied on a
different, legitimate pattern: the outer solve's own objective callback (`cb_F!`, running on a
single thread) synchronously nests a **second** `KN_solve` call for the inner dual problem,
before returning. There is no real concurrency here — it's one thread, one call stack, a solve
nested inside a solve.

KNITRO's own internal enforcement of "one thread at a time" appears to use a non-reentrant
OpenMP critical section (confirmed via `SIGQUIT`-captured stack traces: every hang was stuck in
`__kmp_acquire_queuing_lock`/`__kmpc_critical_with_hint` inside `KTR_lsq_set_jac_callback64`,
reached via the inner `KN_solve`, itself reached via `cb_F!`, itself reached via the outer
`KN_solve`). The same thread tries to enter that critical section a second time while it's still
"held" by the outer solve's own in-progress call — a classic non-reentrant-lock self-deadlock,
independent of actual thread count (which is why forcing every thread-count option to 1 never
helped: the lock isn't counting threads, it's a section the outer solve hasn't released yet).

AUD-02's *other* deliverable — a thread-aware guard in `cc_algo/PsiObjectiveBundle.jl` that
tracks the owning thread ID per bundle instance and explicitly allows legitimate same-thread
nesting while still catching genuine cross-thread reentry — is the correct fix for the hazard
AUD-02 was addressing. The `.opt` file change was redundant given that guard exists, and it's
the part that broke the architecture.

## Fix

Reverted `par_concurrent_evals` to `yes` in all three outer option files. The
`PsiObjectiveBundle.jl` thread-aware guard (added in the same commit, untouched by this fix)
remains as the actual defense-in-depth mechanism against genuine concurrent cross-thread access.

## Verification

- D=4 MWE: bit-identical to the historical commit's clean run (54 calls, 1.53s, same objective).
- Real D=20/W=80,000 `test_driver_pooled_gradient_wiring.jl`: both `use_pooled_gradient=false`
  and `=true` runs now complete (~90s total) with real KNITRO iterations and gradient
  evaluations, clean structured exit — vs. an indefinite silent hang before.
- One separate, pre-existing, unrelated issue surfaced now that the code reaches further than it
  ever did before: the test's own `knitro_status isa Int` assertion is false for KNITRO's native
  `Int32` status codes on this 64-bit system (`Int` means `Int64` in Julia) — a latent test-script
  bug, invisible until now because the process always hung before reaching that check. Not a
  production-code issue; a one-line test fix (`isa Int` → `isa Integer`) if the team wants that
  specific assertion to pass too.

## Lesson

A defense-in-depth fix that changes *both* a library-level option and adds application-level
protection should be validated against the exact pattern it might break (same-thread nested
calls), not just against the hazard it's designed to catch. The thread-aware guard here already
correctly distinguished the two cases in its own design — the redundant `.opt` change is what
slipped through without that same scrutiny.
