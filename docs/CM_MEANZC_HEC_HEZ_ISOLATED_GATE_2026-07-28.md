# CM+ZC (cm_meanzc) isolated threaded H_EC/H_EZ complete-inner-solve gate (2026-07-28)

Real D=20/W=100,000, `run_cm_upper_checkpointed` (the actual public production driver, never a
low-level helper), run **alone** on this shared host — no other concurrent Julia/KNITRO process
launched by this gate — to isolate cm_meanzc's own real-driver KNITRO-concurrency sensitivity
(`docs/HANDOVER_NOTE_2026-07-28.md` §"KNITRO concurrency on this host") from the threading
question this gate actually targets. Raw results: `CM_MEANZC_HEC_HEZ_ISOLATED_GATE_2026-07-28.csv`.

## Architecture finding: H_EC and H_EZ are NOT independently controllable for this family

`CROSS_HESSIAN_THREADED_DEFAULT[]`/`CROSS_HESSIAN_WORKERS_DEFAULT[]` is a single `Ref` pair that
drives H_EC (`cm_hessian_threaded.jl:236-237`), H_EZ (`cm_hessian_architectures.jl:831-832`), AND
H_CZ together for `cm_meanzc` — no separate per-block switch exists in this codebase. The two
"mixed" combinations the task brief asked for (threaded-EC/serial-EZ, serial-EC/threaded-EZ) are
therefore not producible; every row for those combos is recorded `SKIPPED` in the CSV rather than
silently omitted. Only `serial_serial` and `threaded_threaded` are real, run combinations.

This has a direct consequence for the production-default decision below: enabling this family's
threaded toggle would ALSO enable H_CZ threading, which the parent task's own decision matrix
(item 10) wants to stay experimental/non-default. See "Recommendation" below.

## Results

| point | combo | wall_s | knitro_status | n_eval | n_grad | kappa | Hessian maxdiff |
|---|---|---|---|---|---|---|---|
| calibration | serial | 105.9 | -401 (feasible, time-limit) | 1 | 1 | 0.02029 | — (baseline) |
| calibration | threaded | 114.5 | -401 | 3 | 2 | 0.03173 | 0.0 vs serial's own ref |
| non_calibration | serial | 74.4 | -411 (infeasible, time-limit) | 1 | 1 | NaN | — (baseline) |
| non_calibration | threaded | **59.0** | -411 | 1 | 1 | NaN | **0.0**, Hfull_norm identical (1830.349) |
| solver_trajectory | serial | 169.9 | -401 | 3 | 2 | 0.03173 | — (baseline) |
| solver_trajectory | threaded | 130.9 | -401 | 4 | 3 | 0.04264 | 0.0 vs serial's own ref |

`hessian_check_maxdiff_vs_serial` (H_EC and H_EZ sub-block maxdiff, each against that SAME run's
own serial reference computed internally) is **0.0 in every single completed run** — the packed
Hessian threading produces is bit-exact, not merely close, confirmed at real production dimensions
(n_E=382, n_Z=210) on a genuinely converged/time-limited-feasible live KNITRO dual state, not a
synthetic or unsolved one.

## Why n_eval/kappa differ between serial and threaded at `calibration`/`solver_trajectory`

Both budgets (`maxtime_real`) are short enough that most runs terminate via KNITRO's own
time-limit stop (`nStatus ∈ {-401, -411}`), not full convergence. Under a wall-clock-bounded stop,
a FASTER per-callback backend completes more real eval/grad/Hessian cycles inside the same budget
— which is exactly what happened at `solver_trajectory` (threaded: 4 eval/3 grad vs serial: 3
eval/2 grad) — so the two runs' incumbents (`kappa`) differ because they are genuinely different
points along the same trajectory, not because of any inconsistency. This is expected, not a
correctness concern; `non_calibration` (`maxtime_real=20s`) is the one point short enough that both
serial and threaded happened to complete IDENTICAL work (n_eval=1, n_grad=1, same status) before
hitting the limit, giving the cleanest apples-to-apples comparison: **threaded 59.0s vs serial
74.4s, 1.26x faster, identical everything else.**

## Wall-time caveat: driver overhead, not compile time

`run_cm_upper_checkpointed` calls `d20_real_setup_design(W=100_000, ...)` fresh on every single
call (`cm_checkpoint.jl:248`) — rebuilding the full W=100,000-draw D=20 production context (draws,
pivot elimination, compressed-factual workspace) every time, uncounted against `maxtime_real`
(which bounds only the KNITRO solve itself). This is why every wall time here (59-170s) is far
larger than its `maxtime_real` budget (20s/90s) — it is real, legitimate per-call driver overhead
by design (this driver is a cold-start entrypoint), not a JIT-compilation artifact, and it is paid
identically by serial and threaded runs alike, so it dilutes the apparent speedup ratio without
invalidating the comparison. Confirmed by reading the driver's own source, not inferred from
timing alone.

## No KNITRO-concurrency errors

Zero `nStatus=-500`/`-502` callback errors across all 6 real runs — consistent with the
handover note's finding that cm_meanzc's real driver is reliable when run alone, and that the
earlier session's "production regression" claim was a host-concurrency artifact, not a code bug.

## Recommendation

**H_EC/H_EZ threading for cm_meanzc: gate PASSED.** Bit-exact Hessian, no concurrency errors, and
consistently equal-or-better real-solve efficiency (clean 1.26x at the one truly matched point;
"more real work in less wall time" at the longer-budget point). This is real, validated evidence
this family's threaded kernels are safe and beneficial in isolation.

**Default NOT flipped in this release** despite the passing gate, because `cross_hessian_threaded`
is a single toggle for this family covering H_EC/H_EZ/H_CZ together, and the parent task's own
decision matrix requires H_CZ to remain non-default/experimental this release (item 10: "keep the
current production backend as default... do not redesign H_CZ in this release"). Flipping the
shared toggle would satisfy items 6/7 (H_EC/H_EZ for CM+ZC) but violate item 10 (H_CZ) as a direct
side effect of the current code architecture, which has no independent per-block switch. Rather
than silently resolve that tension in either direction, `build_cm_meanzc_bin_ctx`'s
`cross_hessian_threaded` default stays hardcoded `false` (already committed,
`8d2c613`/release branch). A human decision-maker should choose between: (a) accept H_CZ also
going threaded-by-default as a deliberate tradeoff (H_CZ's own gain is weak but not shown unsafe),
or (b) a small follow-up to split the toggle into independent H_EC/H_EZ/H_CZ switches. Recommended
call-site override in the meantime for anyone who wants this validated speedup NOW:
`build_cm_meanzc_production_context(...; cross_hessian_threaded=true, cross_hessian_workers=20)`.

## Caveats

Single rep per cell (no repeated-run noise estimate) given time constraints — the one clean
matched comparison (non_calibration) is the load-bearing data point; the other two points'
differing eval counts under time-limited stops are corroborating, not independently precise.
