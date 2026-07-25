# Origin-ZC Hessian diagnosis — 2026-07-25

Task §7. Status: dense Architecture A **retained**, unchanged this release. Full detail/commit:
`6eed90a`. Raw data: `key_results/originzc_blas_sweep_2026-07-25.csv`,
`key_results/originzc_sweep_points.jls`.

Origin-ZC K=1 has a much smaller restriction block than CM and does not possess CM's
cumulative-bin structure — per the task's own explicit instruction, Architecture C was **not**
attempted here.

## Method

Real feasible P0 (calibration, `eta0 = log(mean(U^k))` per origin, K_mean=1) and P_near
(harvested via a real 60s `run_originzc_upper_checkpointed` call from P0). BLAS in
`[1,4,8,10,20]`, 30s solve budget, dense Architecture A throughout. 11 real runs, 716.6s total.

## Result

| Point | BLAS=1 | BLAS=4 | BLAS=8 | BLAS=10 | BLAS=20 |
|---|---:|---:|---:|---:|---:|
| P0 | 2 | 6 | 6 | 7 | 6 |
| P_near | 3 | 4 | 5 | 5 | 6 |

BLAS>1 clearly helps over BLAS=1 (roughly 2-3x more outer evals in the same budget at both
points), consistent with a dense Architecture A Hessian parallelizing as expected. **No single
value in {4,8,10,20} stands out as decisively fastest-or-tied-best at both points** — the P0
column peaks at BLAS=10 (7), P_near trends mildly upward through BLAS=20 (6) with no sharp peak.
Given this, no centralized BLAS default is set for this family.

A `blas_threads::Union{Nothing,Int}=nothing` kwarg was added to `run_originzc_upper_checkpointed`
this session (mechanism only, default `nothing` = zero behavior change) — needed for this
benchmark and now available as an opt-in for any future origin-ZC campaign.

## Verdict

```text
ORIGIN_ZC_HESSIAN = retained_architecture_a
```

Inner dimension at K_mean=1 remains small enough that dense Architecture A stays appropriate; no
evidence from this bounded benchmark suggests a structural Hessian bottleneck at this scale. If a
future K_mean/K_pair combination materially grows the restriction block, this diagnosis should be
revisited — not assumed to hold indefinitely.
