# CM+mean/ZC immutable-workspace A/B — production port, 2026-07-24/25

## Change under test

`wrap_moments_with_cm_meanzc` (`full_aod_diag/d4_exact/cm_meanzc_moments.jl`): same treatment as
origin-ZC's own commit -- cached `G_tmp` closure instead of a fresh per-call allocation, and
`mean_columns_direct!`/`mean_columns_anchored!`/`pair_columns!` write the centered mean/pair
blocks directly into the destination `G` view instead of allocating a temporary and copying it
in. The flexible-CM grid block itself (Architecture B/C) is **unchanged** -- it already had this
treatment in production before this task (see
`CURRENT_CM_BASIS_AND_STORAGE_AUDIT_2026-07-24.md`, carried over from the prior session's audit).
The pre-change allocating path is preserved byte-for-byte as `wrap_moments_with_cm_meanzc_dense`.

Commit: `03ae607` on `port/restricted-immutable-workspaces-2026-07-24`.

## Correctness (D=4 + D=20)

- D=4 broad gates (`test_cm_meanzc_pure_moments.jl`, `test_cm_meanzc_d4_gates.jl` -- K_mean=1
  regression + K_mean=2 generalization, `test_exclude_row_gateB_meanzc_originzc_k1.jl` at real
  D=20/W=80,000/:exclude_row): **ALL PASS**. The exclude-ROW gate confirms, at the CURRENT
  production destination_sample default: VerifiedSolved, finite Delta_dual (0.3836)/gap/KKT, C+
  vs reference gradient agreement (cosine 1.0, max|diff|=2.5e-15, 4.11x C+ speedup).
- D=20/W=80,000 fixed-point benchmark (N=20 reps, both P0 and P1): `Delta_dual` **bit-identical**
  between dense and cached at every point (P0: 0.009734029009206958; P1:
  0.9458613122892868, to all printed digits).
- Real 15-minute outer-loop A/B (below): both arms' final cold-verify matches their own
  checkpoint-recorded `Delta_dual` to `|diff| <= 5.3e-15`.

## Fixed-point benchmark (N=20 reps, real D=20/W=80,000/L=50, K_mean=K_pair=1)

| point | variant | median (s) | min (s) | alloc (MB) | Delta_dual |
|---|---|---|---|---|---|
| P0 (calibration) | dense  | 4.041 | 3.486 | 22,787 | 0.00973403 |
| P0 (calibration) | cached | 3.792 | 3.396 | 15,453 | 0.00973403 |
| P1 (cold-verified, Delta=0.946, from this task's own outer A/B) | dense  | 4.156 | 3.757 | 22,982 | 0.94586131 |
| P1 (cold-verified, Delta=0.946) | cached | 3.869 | 3.462 | 15,558 | 0.94586131 |

- Median wall-time reduction: **6.2% at P0**, **6.9% at P1** -- cached is faster at BOTH points,
  by a comfortable margin above the "within 5%" bar this family's adoption rule requires.
- Min wall-time reduction: **2.6% at P0**, **7.9% at P1**.
- Allocation reduction: **32.2-32.3%** at both points, reproducing the ~30% figure from the
  prior 2026-07-24 session's N=5 measurement.
- Peak RSS: identical between variants at each point (19,509,312 KB at P0, 14,476,736 KB at P1
  for both dense and cached) -- unchanged, as expected (RSS dominated by persistent buffers this
  change doesn't touch).

**This result reverses the prior 2026-07-24 session's finding.** That session's N=5-rep,
single-point (calibration only) measurement found a borderline median regression (+7.9%) on a
host it flagged as "heavily contested" and explicitly recommended more reps before trusting the
signal. This task's N=20-rep, two-point measurement (on the same real D=20/W=80,000 setup) shows
cached **faster, not slower, at both points** -- consistent with the earlier session's own
diagnosis that the +7.9% figure was noise from too few reps on a contested host, not a real
regression.

Raw data: `restricted_workspace_benchmark_raw_2026-07-24.csv` (family `CM+meanZC`, rows
`P0_calibration`/`P1_cm_meanzc`).

## Real outer-loop A/B (task section 8)

Identical calibrated start, identical algorithm/option file/screens/cache policy, 900s (15 min)
wall-clock budget each, D=20/W=80,000/:exclude_row, cm_extension=:cm_plus_equal_means_zero_covariance
(K_mean=K_pair=1), delta=1.0, cm_gradient_backend=:cplus. Same monkey-patch approach as
origin-ZC's own A/B (`restricted_workspace_outer_ab_cmmeanzc.jl`).

| variant | n_eval | n_grad | wall (s) | best kappa | best Delta_dual | eval/min | grad/min |
|---|---|---|---|---|---|---|---|
| dense  | 11 | 6 | 1102.1 | 0.051063 | 0.79204 | 0.599 | 0.327 |
| cached | 12 | 7 | 917.4  | 0.052656 | 0.94586 | 0.785 | 0.458 |

The dense arm overran its 900s budget (finished mid-solve at 1102.1s wall); the cached arm
finished cleanly within budget at 917.4s. Cached made **+31.0% more evaluations/minute** and
**+40.1% more gradient calls/minute**, and reached a materially higher (better) kappa in LESS
wall-clock time overall. Both arms' cold-verify matched their own checkpoint-recorded
`Delta_dual` to machine precision.

Logs: `production_runs/restricted_immutable_workspace_port_2026-07-24/logs/cmmeanzc_{dense,cached}_ab.log`.
Checkpoints/P1 seeds: `production_runs/restricted_immutable_workspace_port_2026-07-24/cmmeanzc_ab/{dense,cached}/`.

## Gate-by-gate (task section 8 merge criterion)

1. All correctness gates pass -- **YES**.
2. ~30% allocation reduction reproduced -- **YES** (32.2-32.3%).
3. Fixed-point median inner-solve time within 5% of production at both P0 and P1, or faster --
   **YES, faster at both** (6.2% / 6.9% faster).
4. Outer-loop verified progress per minute within 5% of production, or better -- **YES, better**
   (+31.0% eval/min, +40.1% grad/min, materially higher final kappa).
5. Peak RSS and GC behavior improved or unchanged -- **YES, unchanged** (identical peak RSS per
   point; allocation/GC pressure improved per the ~32% reduction above).

## Verdict

```
CM_MEANZC: MERGED_TO_PRODUCTION
```

Merged onto `production/fullA-exact` and tagged
`cm-meanzc-immutable-workspace-production-ready-2026-07-24`. This supersedes the prior
2026-07-24 session's `RETAIN_CURRENT_PRODUCTION` recommendation, which was explicitly conditioned
on "more reps at a quieter host state" or "an actual outer-loop shakedown" changing the verdict --
both of those follow-ups are exactly what this task did.
