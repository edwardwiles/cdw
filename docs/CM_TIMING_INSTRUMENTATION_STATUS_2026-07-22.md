# Timing instrumentation status — 2026-07-22

## What already exists (verified by reading, not re-derived)

The closure report's own Phase 5 (`docs/closure_2026-07-22/REMEDIATION_CLOSURE_AND_PROMOTION_REPORT_2026-07-22.md`,
"Phase 5 — end-to-end performance measurement (scoped)") already added a small, additive, opt-in
`grad_trace_ref` kwarg on `run_polish_checkpointed` (unrestricted path) — zero behavior/allocation
change when `nothing` (the default) — and used it for a real same-trajectory backend replay
(`FULLA_SAME_TRAJECTORY_BACKEND_REPLAY_2026-07-22.csv`,
`FULLA_CALLBACK_WALL_DECOMPOSITION_2026-07-22.csv`) reconciled exactly against KNITRO's own native
`n_grad_calls`/`native_ga_evals` counters (0 discrepancy at both deltas tested).

That report is explicit that the **full** per-native-eval-category breakdown the original brief
asked for — screens counted separately from warm/cold/infeasible/exact-cache/
checkpoint-serialization/GC, at multiple deltas — was **not** built, calling it "the largest
remaining piece of the original brief... a genuinely large, multi-hour instrumentation-plus-
multiple-real-KNITRO-runs project on its own." This session did not build it either, for the same
reason, and because doing so touches the exact production callback code path on the eve of the
launch — not a place to add new untested instrumentation surface area under time pressure.

## What this session did instead

- Confirmed (by direct code read, `docs/CM_PRODUCTION_STATE_2026-07-22.md`) that the CM driver
  (`cm_checkpoint.jl`) has **no** existing timing/trace instrumentation of any kind beyond the
  `heartbeat_interval_s` watchdog — the `grad_trace_ref` mechanism above is unrestricted-path only
  and was never ported to the CM path.
- Did **not** add new timing instrumentation to the CM production callbacks
  (`run_cm_upper_checkpointed`'s `cb_F!`/`cb_G!`) this session — the campaign's own driver code is
  frozen at the tagged baseline; adding per-phase timers there would be exactly the kind of
  "speculative... change" the brief says not to make without a reproduced, concrete need, and the
  supervisor's log-growth/checkpoint-mtime watchdog (see `docs/CM_STALL_INVESTIGATION_2026-07-22.md`)
  does not require it to function.
- The smoke test run this session (see the launcher doc) produces ordinary per-eval log lines
  (`eval N t=... Delta=...`) which give coarse wall-clock-per-eval numbers for real reference, but
  this is not a new instrumentation mechanism — it is the pre-existing `cb_F!` logging.

## Bottom line

No CSV/Markdown per-callback-category timing report was produced this session for the CM path.
This mirrors the closure report's own honest scoping of Phase 5 and is consistent with the current
brief's own instruction not to claim end-to-end speedups from isolated kernels and not to let
optional instrumentation work delay the campaign. If this is wanted before or during the campaign,
budget it as a separate, dedicated multi-hour task, ported from the unrestricted path's
`grad_trace_ref` pattern to the CM path's `cb_F!`/`cb_G!`, with the same "additive kwarg, `nothing`
default, zero-overhead-when-off" discipline already established in this codebase.
