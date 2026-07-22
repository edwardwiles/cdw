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

**Result: [FILLED IN BELOW ONCE THE LIVE RUN COMPLETES]**

## Classification (per the brief's 8-way taxonomy)

Applied once the live evidence above is in — see the same 8 categories Phase 4 implicitly used:
(1) ordinary long callback, (2) GC/memory-pressure stall, (3) BLAS oversubscription, (4) KNITRO
internal wait, (5) Julia task exception, (6) deadlock, (7) external-timeout artifact, (8) not
reproduced.

## Instrumentation added this session

None beyond what Phase 4 already added (`heartbeat_interval_s` kwarg, already in production code,
unchanged). Per the brief's own caution against speculative concurrency changes, no code fix was
attempted without a concretely reproduced-and-explained failure. The supervisor script
(`scripts/cm_production_supervisor.sh`) is this session's actual mitigation: it does not need the
hang's root cause to be known to detect and recover from it (poll-based staleness detection on log
growth + checkpoint mtime, graceful-then-forceful termination, restart from the last valid
schema-2 checkpoint).

## Bottom line

The production launch does not depend on this stall being fully explained — the brief's own
closing instruction ("the production launch can proceed from the tagged baseline if the supervisor
works even when the underlying rare stall remains unexplained") is the operating basis here,
consistent with the closure report's own decision table ("CM internal production: READY, with one
flagged operational caveat").
