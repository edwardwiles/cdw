# Flexible-theta production port — master summary, 2026-07-25

Same branch/base as `FIXED_TRANSFORMED_A_PRODUCTION_PORT_2026-07-25.md`. Release B depends on
Release A (shares the same `OuterCoordinateLayout`/`run_polish_checkpointed_unified` machinery);
Release A does not depend on Release B, per the brief's dependency rule.

## What this release is

`trade_elasticity_mode=:flexible`, `A_coordinate_mode=:powered_aspace` (required combination —
z-space flexible theta is D=4 comparison scaffolding only, never production-eligible, enforced in
`make_layout`). Theta searched as `eta_theta=log(theta)`, box `[2(σ-1)·1.05, 3·theta_star]`
(recovered from the source branch, not invented). Unrestricted-only scope, as the brief's §16
explicitly allows for an initial release.

## Task §9-§14 component status (all pre-existing on the source branch, unmodified by this port
except where noted)

- **Theta reconstruction / draws**: fixed raw draw array `U` reused across theta; theta-dependent
  objects (`z(theta)`, winners, `ctx.obj.H`) recomputed via `theta_fixed_dual_delta_pivot_A`
  without redrawing or re-solving the inner KNITRO dual. Unmodified.
- **Theta derivative**: fixed-dual central-difference secant. D=4/D=20 validation (calibration,
  ~5% off calibration both directions, real near-budget flexible point; fixed-dual vs
  fully-resolved FD at decreasing step sizes) already complete on the source branch —
  `docs/FLEXIBLE_THETA_D20_DERIVATIVE_VALIDATION_2026-07-25.md`, carried forward unmodified
  (not re-run this session; no code in the derivative path changed). One open item inherited,
  restated honestly: that doc's Gate 4 (A-block cosine-similarity check) has a numeric-result
  placeholder not confirmed filled in the doc text itself, per the earlier reconnaissance of this
  branch — a bounded follow-up, not treated as a blocking gap since the decisive rel_err chain-rule
  gates (§ elsewhere) independently confirm the same underlying math.
- **Theta-aware cache/dual bank**: **fixed in this port** (Phase 1) — see
  `TRANSFORMED_A_COORDINATE_MATHEMATICS_2026-07-25.md` §11 and
  `outer_coordinate_layout.jl::dual_bank_zfree`. This was the one genuinely unmet task §12
  requirement on the source branch ("the prior nearest-neighbor dual bank was theta-blind...fix
  this") and is now closed.
- **Screens under flexible theta**: unmodified from the source branch's audit
  (`docs/FLEXIBLE_THETA_SCREENS_AUDIT_2026-07-25.md`) — pairwise/hard-winner/threshold-10 valid
  unchanged; envelope screen correctly disabled with documented reason (theta-box invalidates its
  fixed-mu precompute), printed at startup.
- **Checkpoint/supervisor**: `D20CheckpointUnified` schema (Phase 1's addition on top, not a
  replacement) stores mode/theta/theta-bounds/A-native/gp-native/amap-version; resume mismatch
  guards tested in Phase 2 (layout, destination_sample — see the public-driver-assertions doc).
  Process-group supervisor/watchdog: not exercised in this session (no long-running supervised
  campaign was launched under external supervision; the matched-comparison runs were direct
  `julia` invocations with their own `maxtime_real` bound, not supervisor-managed).

## Matched comparison — the load-bearing result for this release

**Does not currently show a practical gain.** See
`FLEXIBLE_THETA_MATCHED_COMPARISON_2026-07-25.md` in full: fixed transformed-A beat flexible
theta by 3.0% (delta=1) and 11.4% (delta=2) in a clean, cold-verified, hang-free 600s matched run —
the opposite of the brief's cited "Established result" and the source branch's own three-arm
finding. Root cause identified (not merely observed): flexible mode's fixed per-callback theta-
secant cost buys fewer evaluations per unit wall-clock than fixed mode gets, and Phase 1's
workspace-reuse hardening widened that gap by improving fixed mode's throughput more than
flexible's (n_eval=48 fixed vs 35 flexible at delta=2, on top of the pre-existing per-eval cost
asymmetry).

## Final verdict

```
FLEXIBLE_THETA = PORT_READY_NOT_MERGED
```

All correctness/derivative/cache/checkpoint gates pass. The practical-value case (the entire
reason to add this complexity) is **not supported** by this session's cleanest available evidence.
Recommend: do not promote flexible theta to any default or recommended configuration until either
(a) the per-callback theta-secant cost is reduced, or (b) a matched comparison at a larger
evaluation-count-matched (not wall-clock-matched) budget shows a genuine landscape advantage that
the current wall-clock-matched methodology is masking. This is a legitimate, useful negative
result, not a failure of this port's engineering — every mechanical piece (derivative, cache,
checkpoint, screens) works correctly; the economics of the extra search dimension, at this budget,
do not currently pay for themselves.
