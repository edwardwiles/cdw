# CM interrupt→resume stall investigation — 2026-07-22

## Prior evidence (closure report, Phase 4 — read, not re-derived)

`docs/closure_2026-07-22/REMEDIATION_CLOSURE_AND_PROMOTION_REPORT_2026-07-22.md` Phase 4 already
ran a real D=20/W=80,000/L=50/delta=1 interrupt→resume shakedown and found a genuine hang: after a
210s external SIGTERM, the resume process's heartbeat log (`heartbeat_interval_s=10.0`) went
completely silent at t=104.7s (no further heartbeat line at all, even though the timer should have
fired every 10s), Julia's own SIGTERM stack dump printed (ending mid-`KN_solve`, `GC: 151`
recorded), and the process then sat alive and unresponsive for 100+ more seconds, requiring manual
`SIGKILL`.

That session explicitly checked and ruled out the known AUD-02 `par_concurrent_evals` regression
(confirmed `csw_outer_wallclock_sr1.opt` still has `par_concurrent_evals yes`; the captured stack
lacked AUD-02's specific `__kmp_acquire_queuing_lock`/`KTR_lsq_set_jac_callback64` signature).
**Classification reached: "KNITRO driver runs can hang past their own declared timeout"** (a
previously-documented, separate phenomenon — `gravity-robustness-knitro-hang-past-timeout` —
where an external `timeout`/`SIGTERM` does not reliably bound a real KNITRO process), with heavy
recorded GC activity (151 cycles) at the point of the dump consistent with, but not proven to be,
a severe GC-driven stall. **Not fully resolved to one definitive root cause** within that task's
bounded budget — disclosed as an open item, not asserted as "crash" or "deadlock."

## This session's own reproduction attempt

Constraints: real KNITRO capacity on this host is shared with an already-running, unrelated
production campaign (`production_runs/2026-07-22/fullA_exact_unrestricted_670eac4/`, chains A/B/C
of the *unrestricted* model at the pre-remediation commit — chain A completed during this
session, chain B was mid-flight) — see the launcher doc's smoke-test section. This session's own
CM reproduction attempt was therefore kept deliberately small (`JULIA_NUM_THREADS=6`, a single
short-budget stage) rather than repeating Phase 4's full-scale two-arm shakedown, to avoid adding
material load to a host that was already busy with a real, unrelated campaign.

Procedure actually executed (via the new `cm_production_stage_runner.jl`, `mode=calibration`,
`delta=1.0`, real D=20/W=80,000/L=50 data):
1. Launched the stage runner in the background.
2. A Monitor process polled for the first `stage_latest.jls` checkpoint to appear, then located the
   real PID via `pgrep`, confirmed its command line matched exactly (never killed a PID without
   re-confirming its cmdline against a stale listing, per this repo's own house rule), and sent a
   single `SIGTERM`.
3. Observed whether the process exited promptly (the normal case) or exhibited the same
   post-SIGTERM hang Phase 4 found.
4. If a checkpoint existed, ran `cm_cold_verify.jl` against it independently.
5. Attempted a `mode=resume` restart against the same checkpoint directory.

**Result: REPRODUCED, with much more specific evidence than Phase 4 had.**

Two prior attempts this session were methodologically invalid and are recorded for honesty, not
hidden: attempt 1 sent `SIGTERM` to the *shell wrapper* process (matched by an overly broad
`pgrep -f` pattern), not the real `julia` process, which then ran on to natural completion
undisturbed. Attempt 2 used a chain-perturbed start (`chain_perturb_seed=1`, perturbation scale
0.5) that KNITRO's own presolver could not evaluate at all (`knitro_status=-502`, `n_eval=0`) —
a real, separate bug, fixed in the same commit as this investigation (perturbation scale reduced
to 0.02, plus a new feasibility pre-check that fails fast with an actionable message instead of
silently burning wall budget on an unevaluable start — see `cm_production_stage_runner.jl`).

**Attempt 3 (real reproduction)**: real D=20/W=80,000/L=50/delta=1 run, unperturbed calibration
start, `JULIA_NUM_THREADS=6`. Confirmed the true `julia` PID via `ps -o comm=` (excluding the
shell wrapper this time), waited until eval 1's `cb_F!` had returned and its paired `cb_G!`
(gradient) call was genuinely in flight (per the heartbeat: `t=30.5s last_callback=cb_F! 9.8s
since last callback RETURNED n_eval=1 n_grad=0` — i.e. `cb_G!` for eval 1 had been running >=9.8s),
then sent a single `SIGTERM`.

- Julia printed its standard signal handler output immediately (`[92661] signal 15: Terminated`)
  with a full multi-threaded backtrace — **12 separate stack traces** (matching the 6 worker
  threads' `Threads.@threads` participation, doubled by GC/scheduler frames), all rooted in the
  SAME call path: `cb_G!` → the CM gradient's `composite_gradient_at_fast`-family →
  `select_bandwidth` (`composite_gradient.jl:214`) → `mass_at` → `count_winner_flips`
  (`composite_gradient.jl:83-90`) → `price_and_pTsigma_cell` (`lfix_incremental.jl:219`, an
  allocating `broadcast`/`copyto!`/`materialize` call), invoked from the per-coordinate threaded
  loop `do_coord!` (`composite_gradient_fast.jl:195,230`). One additional thread was idle,
  waiting in `uv_run`/`epoll_pwait` (the heartbeat timer's own event loop).
- The dump recorded **`Allocations: 173,266,596` and `GC: 140`** cycles at the moment of the
  signal — a very large allocation count for a single gradient call, consistent with real memory
  pressure driving frequent GC cycles.
- **Despite the signal-15 dump printing immediately, the process did NOT exit.** Polled every 2s:
  still alive (declining `%CPU`, process state `Sl`) at every check through t=30s post-`SIGTERM`,
  confirmed gone from `ps` only after an explicit `SIGKILL`.
- **Cold-verified the resulting checkpoint anyway** (`cm_cold_verify.jl`, fresh process): reported
  `Delta=0.00878804666675019`, cold `Delta_dual=0.00878804666675019`, `|diff|=0.0`,
  `verified_success=true`, `feasible=true` — the checkpoint/cold-verify correctness chain holds
  even across a genuine hang+forced-kill, consistent with Phase 4's own finding.

## Classification (per the brief's 8-way taxonomy)

**(2) GC/memory-pressure stall — best-supported classification, not proven as the singular root
cause.** Reasoning: the backtrace is rooted in an actively-executing, heavily-allocating,
multi-threaded numerical kernel (not a KNITRO/BLAS C call, unlike Phase 4's dump which ended
"mid-`KN_solve`" with no further detail) — Julia's stop-the-world GC requires every thread to
reach a safepoint before it can proceed, and signal delivery/handling itself is only fully
serviced once the runtime returns to a safe point; a large, sudden allocation burst (173M+
allocations, 140 GC cycles by the time of the dump) across 6 concurrently-running threads is a
textbook condition for that safepoint rendezvous to take an extended, sometimes very long, time —
which would explain both symptoms observed here and in Phase 4 (the signal handler prints
immediately, because printing a backtrace does not itself require full thread rendezvous, but
actual process exit does). This is NOT the same evidence pattern as (4) KNITRO internal wait (no
KNITRO frames appear anywhere in this dump) or (3) BLAS oversubscription (the hot code is Julia
broadcast/SIMD, not a BLAS call — consistent with `OPENBLAS_NUM_THREADS=1` already being set).
(6) deadlock in the stricter sense (a true circular lock wait) cannot be fully ruled out from a
single stack dump, but no lock-acquisition frames (e.g. `uv_mutex_lock`, `jl_mutex_lock`) appear
in the captured traces, which weighs against it. (1) ordinary long callback, (5) Julia task
exception, and (7) external-timeout artifact are all ruled out directly by the evidence (the
callback was genuinely still computing, no exception was thrown, and this session used a real
`SIGTERM`/`SIGKILL` sequence with direct `ps` confirmation, not a `timeout` wrapper). (8) not
reproduced does not apply to attempt 3 — it *was* reproduced, with two invalid prior attempts
disclosed above rather than omitted.

## Instrumentation added this session

None beyond what Phase 4 already added (`heartbeat_interval_s` kwarg, already in production code,
unchanged). Per the brief's own caution against speculative concurrency changes, no code fix was
attempted without a concretely reproduced-and-explained failure. The supervisor script
(`scripts/cm_production_supervisor.sh`) is this session's actual mitigation: it does not need the
hang's root cause to be known to detect and recover from it (poll-based staleness detection on log
growth + checkpoint mtime, graceful-then-forceful termination, restart from the last valid
schema-2 checkpoint).

**Important caveat on the 30s window**: this session's own escalation-to-`SIGKILL` threshold was
deliberately shortened (via the supervisor's env-overridable tunables, see
`docs/CM_PRODUCTION_LAUNCHER_2026-07-22.md`) to make a smoke test practical; it was *not* run
against the real 600s (10-minute) production `STALL_THRESHOLD_S`. It is therefore not established
here whether this specific stall would have self-resolved somewhere between 30s and 600s (a GC/
safepoint stall of that scale plausibly could resolve on its own once the allocating burst
finishes) or would have persisted indefinitely like Phase 4's. Either way, the supervisor's actual
production policy (graceful `SIGTERM`, wait, escalate to `SIGKILL` only after real evidence of no
progress) is unaffected — it does not need to know which of those two is true to do its job.

## Instrumentation added this session

None beyond what Phase 4 already added (`heartbeat_interval_s` kwarg, already in production code,
unchanged). Per the brief's own caution against speculative concurrency changes, no code fix was
attempted without a concretely reproduced-and-explained failure — and even with this session's
sharper localization (a specific threaded, allocating kernel), the underlying fix space (reduce
allocations in `price_and_pTsigma_cell`'s per-coordinate broadcast, or tune GC/thread behavior) is
exactly the kind of speculative concurrency/performance change the brief says not to make without
a fuller investigation than a pre-launch sprint allows. The supervisor script
(`scripts/cm_production_supervisor.sh`) is this session's actual mitigation: it does not need the
hang's root cause to be known to detect and recover from it (poll-based staleness detection on log
growth + checkpoint mtime, graceful-then-forceful termination, restart from the last valid
schema-2 checkpoint).

## Bottom line

The production launch does not depend on this stall being fully explained — the brief's own
closing instruction ("the production launch can proceed from the tagged baseline if the supervisor
works even when the underlying rare stall remains unexplained") is the operating basis here,
consistent with the closure report's own decision table ("CM internal production: READY, with one
flagged operational caveat"). This session adds a sharper, independently-reproduced localization
(a specific allocating threaded gradient kernel, not just "somewhere in KN_solve") but does not
change that operating basis.
