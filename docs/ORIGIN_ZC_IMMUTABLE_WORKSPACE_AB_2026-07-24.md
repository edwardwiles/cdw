# Origin-ZC immutable-workspace A/B — production port, 2026-07-24/25

## Change under test

`wrap_moments_with_originzc` (`full_aod_diag/d4_exact/cm_originzc_moments.jl`): the mean/pair
centered blocks (`Phi_R - t_R(eta)`) are now written directly into the destination moment matrix
`G`'s view via `mean_columns_direct!`/`pair_columns!` (in-place broadcast), and the per-call
`G_tmp` scratch buffer is now a closure-captured cache reused across calls, instead of a fresh
`similar(...)` allocation every outer evaluation. The pre-change allocating path is preserved
byte-for-byte as `wrap_moments_with_originzc_dense` for this A/B. No other file, moment
restriction, K_mean/K_pair semantics, screen, solver tolerance, or checkpoint semantic changed.

Commit: `6f98c69` on `port/restricted-immutable-workspaces-2026-07-24` (based on production tip
`d3342d4`).

## Correctness (D=4 + D=20)

- D=4 broad gates (`test_cm_originzc_pure_moments.jl`, `test_cm_meanzc_d4_gates.jl`-style coverage,
  `test_exclude_row_gateB_meanzc_originzc_k1.jl` at real D=20/W=80,000/:exclude_row): **ALL PASS**.
  `test_exclude_row_gateB_meanzc_originzc_k1.jl` in particular confirms, at the CURRENT production
  destination_sample default: VerifiedSolved, finite Delta_dual/gap/KKT, C+ vs reference gradient
  agreement (cosine 1.0, max|diff|=2.2e-15), gradient length D*Ddest+n_eta(layout)=400 correct.
- D=20/W=80,000 fixed-point benchmark (this task, N=20 reps, both P0 and P1 — see below):
  `Delta_dual` **bit-identical** between dense and cached at every point
  (P0: 0.003584286814559523; P1: 0.9842589797032454, both to all printed digits).
- Real 15-minute outer-loop A/B (below): both arms' final cold-verify matches their own
  checkpoint-recorded `Delta_dual` to `|diff| <= 3.8e-15` (machine precision).

## Fixed-point benchmark (N=20 reps, real D=20/W=80,000/L=50)

| point | variant | median (s) | min (s) | alloc (MB) | Delta_dual |
|---|---|---|---|---|---|
| P0 (calibration) | dense  | 3.399 | 2.721 | 22,870 | 0.00358429 |
| P0 (calibration) | cached | 2.900 | 2.373 | 15,517 | 0.00358429 |
| P1 (cold-verified, Delta=0.984, from this task's own outer A/B) | dense  | 2.797 | 2.471 | 22,787 | 0.98425898 |
| P1 (cold-verified, Delta=0.984) | cached | 2.663 | 2.376 | 15,453 | 0.98425898 |

- Median wall-time reduction: **14.7% at P0**, **4.8% at P1**.
- Min wall-time reduction: **12.8% at P0**, **3.9% at P1**.
- Allocation reduction: **32.1-32.2%** at both points.
- Peak RSS: unchanged between variants at each point (dominated by persistent U/Zraw_all buffers
  this change doesn't touch), consistent with the allocation win being GC-pressure-only, as
  expected.

Raw data: `restricted_workspace_benchmark_raw_2026-07-24.csv` (families `originZC`,
rows `P0_calibration`/`P1_origin_zc`).

## Real outer-loop A/B (task section 7)

Identical calibrated start, identical algorithm/option file/screens/cache policy, 900s (15 min)
wall-clock budget each, D=20/W=80,000/:exclude_row, K_mean=K_pair=1, delta=1.0,
cm_gradient_backend=:cplus. The ONLY difference between arms: a same-process redefinition of
`wrap_moments_with_originzc` to delegate to the dense reference path for the "dense" arm
(`restricted_workspace_outer_ab_originzc.jl`).

| variant | n_eval | n_grad | wall (s) | best kappa | best Delta_dual | eval/min | grad/min |
|---|---|---|---|---|---|---|---|
| dense  | 15 | 10 | 907.9 | 0.056538 | 0.98804 | 0.991 | 0.661 |
| cached | 17 | 11 | 924.5 | 0.056880 | 0.99750 | 1.103 | 0.714 |

Cached made **+11.3% more evaluations/minute** and **+8.0% more gradient calls/minute**, and
reached a **higher (better — this driver minimizes gp, kappa increases as gp decreases) kappa**
in essentially the same wall budget. Both arms' cold-verify (independent fresh inner re-solve)
matched their own checkpoint-recorded `Delta_dual` to machine precision (`3.77e-15` dense,
`2.66e-15` cached) — the outer trajectories differ (expected: KNITRO's wall-clock-driven maxtime
cutoff lands at a different iteration count depending on how fast each rep runs), but every
individual point either arm visited is independently verified consistent.

Logs: `production_runs/restricted_immutable_workspace_port_2026-07-24/logs/originzc_{dense,cached}_ab.log`.
Checkpoints/P1 seeds: `production_runs/restricted_immutable_workspace_port_2026-07-24/originzc_ab/{dense,cached}/`.

## Gate-by-gate (task section 7 merge criterion)

1. All correctness gates pass — **YES**.
2. Inner-solve median >=10% faster, OR allocation reduction >=20% with no wall-time regression
   >5% — **YES** (both hold: P0 median is 14.7% faster on its own; every point's allocation
   reduction is ~32% with wall time IMPROVED, never regressed, at every point).
3. Outer-loop verified progress no worse — **YES, better** (+11.3% eval/min, +8.0% grad/min,
   higher final kappa in the real 15-min A/B).
4. No new crash, timeout, or checkpoint issue — **YES** (both arms completed cleanly at "Time
   limit reached, current point is feasible"; no exceptions).

## Verdict

```
ORIGIN_ZC: MERGED_TO_PRODUCTION
```

Merged onto `production/fullA-exact` and tagged
`origin-zc-immutable-workspace-production-ready-2026-07-24` (see
`RESTRICTED_IMMUTABLE_WORKSPACE_PRODUCTION_PORT_2026-07-24.md` for the merge commit/ancestry
record).
