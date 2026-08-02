# Profiled all-family outer A/B protocol (2026-08-01)

## Purpose

Defines how `full_aod_diag/d4_exact/PROFILED_ALL_FAMILY_OUTER_AB_HARNESS_2026-08-01.jl` is meant to
be run, once the inner branch's restricted-family evaluators are wired in, to compare the full
gamma-normalized formulation against the profiled destination-scale formulation for each family.
**No campaign has been launched under this protocol** — the only run performed this session is the
mechanical smoke test described below.

## Matched-comparison requirements (task §17, all enforced by the harness's design, not just
## documented)

| requirement | how the harness enforces it |
|---|---|
| same data and draws | both arms are called against the same `ctx` object, no re-sampling |
| same economic starting point | `w_start` is derived once (e.g. `reduce_calibration_to_w_profiled` or a shared continuation waypoint) and passed to both arms |
| same delta and direction | both arms read `ctx.δ`/`ctx.find_smallest` off the SAME `ctx` |
| same outer solver | both arms load the SAME `.opt` file (`csw_outer_wallclock_$(hessopt_tag).opt`), same `algorithm=3`, same `z_halfwidth=30` box |
| same wall-clock and gradient-call budgets | `maxtime_real`/`maxit_override` passed identically to both arms' invocation |
| same inner warm-start policy | the full arm uses the family's own existing production driver's warm-start policy unchanged (this harness does not touch it); the profiled arm's `evaluate_fn` re-solves from the profiled operator bundle's own default policy, also unchanged |
| same restriction backend | the profiled arm's `evaluate_fn`, once wired to a real restricted-family evaluator, must call the SAME restriction-moment/Hessian backend the family's production full-formulation path uses — this is an obligation on whoever wires `evaluate_fn`, not something the harness can verify structurally |
| screens either valid for both or disabled for both | inherited unchanged from the unrestricted-only predecessor harness's own documented asymmetry (profiled arm has no screens by construction; full arm explicitly disables `use_general_range_safety_net`) — re-audit per family once real evaluators are wired, since a restricted family's screen set may differ from the unrestricted family's |
| `gp` held fixed for a fixed-gp comparison | `run_profiled_family_outer_search` never adds `w[1]` (`gp`) to the KNITRO free-variable vector — `gp_fixed = w_start[1]` is captured once and never touched by any callback (this was the exact live bug found and fixed upstream on the unrestricted-only harness, commit `1aeec43`; the family-generic harness inherits the fix structurally, not by copying the fixed value around) |

## Required families (task §17)

```
unrestricted   -- READY (real evaluate_profiled_point via unrestricted_ab_arm)
flexible_CM    -- NOT READY (restricted_family_evaluator_not_ready)
ZC_only        -- NOT READY
CM_plus_ZC     -- NOT READY
common_Frechet -- optional, "may be added afterward" per task text; also NOT READY
```

## How to run once a family's real evaluator is wired

```julia
include("PROFILED_ALL_FAMILY_OUTER_AB_HARNESS_2026-08-01.jl")

# 1. Build ctx/spec/pe as the family's own inner branch code provides.
# 2. Build (evaluate_fn, family_ctx_builder) -- see unrestricted_ab_arm for the shape.
evaluate_fn, family_ctx_builder = flexible_cm_ab_arm_REAL(ctx, ...)   # replace the NOT-READY stub

# 3. Run the profiled arm.
result_profiled = run_profiled_family_outer_search("flexible_CM_profiled", w_start; ctx = ctx,
    evaluate_fn = evaluate_fn, family_ctx_builder = family_ctx_builder,
    maxtime_real = 1200.0, trace_csv = "flexible_CM_profiled_trace.csv")

# 4. Run the family's own existing full-formulation production driver, UNMODIFIED, with the
#    same maxtime_real/w_start/ctx, capturing the same fields (wall_ext, n_eval, n_grad_calls,
#    best.Delta_dual, best.t_elapsed).

# 5. Record: best verified objective by wall time, best verified objective by gradient-call
#    count, outer evaluations, gradient evaluations, actual inner solves, inner iterations and
#    Hessian callbacks, gradient wall time, failed/rejected moves -- into
#    PROFILED_RESTRICTED_OUTER_AB_PRELIMINARY_2026-08-01.csv (task §18, optional, only after
#    D4/D20/restriction-parameter gates all pass for that family).
```

## Smoke test performed this session (mechanical validation only, NOT a campaign)

`test_profiled_ab_harness_smoke_2026-08-01.jl`: D4, unrestricted arm, `maxit_override=8`,
`maxtime_real=60.0`, random small perturbation start. Confirmed:

- the harness correctly builds/validates the family context before spending KNITRO wall-clock;
- `cb_F!`/`cb_G!` wiring produces evaluations and gradient calls (`n_eval=22`, `n_grad=9` in the run
  performed);
- gp was never added to the KNITRO free-variable set (11 free vars for a 12-dimensional profiled
  outer vector, matching `n_free = length(r_free_start) = n_total - 1`);
- the "not ready" restricted arms throw the expected loud error rather than silently returning a
  fake evaluator.

Result: `HARNESS SMOKE TEST: PASS`, `restricted-arm-not-ready throws loudly: PASS`.

## Explicit non-actions this session

- No `PROFILED_RESTRICTED_OUTER_AB_PRELIMINARY_2026-08-01.csv` was produced (task §18 is gated on
  "if all inner and gradient gates pass early" for a REAL restricted family — none has a live inner
  context yet).
- No production default was changed.
- No W=80,000/100,000 run of any kind was launched.
