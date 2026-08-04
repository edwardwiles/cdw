# Section 8 — bandwidth-cache material benefit measurement

The prior session's own bandwidth-cache gates (correctness: on/off equality, hit reuse, no-stale-
reuse) all passed at real D20/W=20,000, but no isolated performance benefit was ever measured —
task §8's own explicit ask. This session measured it directly.

## Protocol

`test_bandwidth_cache_benefit_2026-08-04.jl`, real D20 context, all three required families
(`unrestricted`, `flexible_cm`, `origin_zc`) at both required scales (W=20,000, W=100,000):

- Same decoded starting point (calibration), same solved inner dual (ONE `ProfiledLFixCache` per
  point in the sequence, built once — isolates bandwidth-selection cost from inner-solve time).
- Same sequence of 8 nearby outer points (small random walk from calibration, `Random.seed!(20260804)`,
  step size `0.005` per coordinate — small enough to stay within the cache's own `validity_radius=0.02`
  so hits are expected after the first visit).
- Same thread count (serial, `threaded=false` — isolates the cache's own contribution from
  threading, which was already gated separately in section 7).
- Compilation warmed via one untimed pass before any timed measurement.
- **Both execution orders run** (cache OFF-then-ON and ON-then-OFF) to rule out ordering artifacts.

## Results — real, material, consistent benefit across every family and scale

| Family | W | speedup order A (OFF→ON) | speedup order B (ON→OFF) |
|---|---|---|---|
| unrestricted | 20,000 | 1.38x | 1.33x |
| flexible_cm | 20,000 | 1.41x | 1.41x |
| origin_zc | 20,000 | 1.41x | 1.38x |
| unrestricted | 100,000 | 1.31x | 1.31x |
| flexible_cm | 100,000 | 1.35x | 1.25x |
| origin_zc | 100,000 | 1.33x | 1.34x |

All 6 runs: cache correctness held throughout (hit/miss/stale-eviction counters match the expected
pattern — misses on first visit to each coordinate, hits on repeat visits within
`validity_radius`), speedup is consistent (1.25x-1.41x) regardless of execution order or W scale,
and no failures or stale-reuse were observed in any run.

## Verdict

```
BANDWIDTH_CACHE = accepted_measured_gain_1.25x_to_1.41x_wall_time_speedup_all_3_required_families_both_scales_both_orders
```

Recommended: enabled in the production manifest (real, consistent, order-independent 25-41% wall-
time reduction on the gradient-engine coordinate loop, at zero observed correctness cost across
both this session's and the prior session's gates).

## Not done this session

- `common_frechet`/`cm_meanzc` were not included in this measurement (task's own minimum list was
  `unrestricted`/`flexible_CM`/`origin_ZC`) — not measured, not assumed to generalize.
- Only one starting-point/step-size configuration was tested; a systematically different outer
  algorithm step pattern was not swept.
