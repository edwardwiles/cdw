# Brazil→Korea-excluded gravity regression, 2026-07-31

## Specification
- **Dependent variable:** `ln(pi[o,d])`, the OECD-ICIO-2018 bilateral trade share (row=exporter,
  col=importer), from `real_data/noah_D20/pi.csv`.
- **Regressor:** `ln(tau[o,d])`, the Teti/BACI goods-share-adjusted 2018 tariff-cost matrix,
  from `real_data/noah_D20/tau.csv`.
- **Fixed effects:** origin + destination two-way FE, via the exact (unbalanced-panel) within
  transform `within_transform_masked` (`misc/doubleDiff.jl`) — equivalent to OLS with a full set
  of origin/destination dummies (FWL), not an approximation.
- **Weights:** none (unweighted OLS, matching production's `master_prestep.jl` estimator).
- **Sample (production default spec):** `exclude_diagonal_gravity=true`,
  `destination_sample=:exclude_row` — drops the domestic/own-trade diagonal (`o==d`) and ROW as a
  destination, **plus, new in this task, the single cell origin=Brazil (`bra`, index 3),
  destination=Korea (`kor`, index 14)**.
- **Estimator:** `thetaHat = -sum(W_lnpi .* W_lntau) / sum(W_lntau .^ 2)`, matching
  `prestep/master_prestep.jl`'s production helper exactly (verified below). Reported
  `gravity_coefficient = -thetaHat` is the sign-mapped regression coefficient of `ln(pi)` on
  `ln(tau)` (task's `theta_star = -gravity_coefficient` convention).

## Sample counts
| | old (no Brazil→Korea exclusion) | new (Brazil→Korea excluded) |
|---|---|---|
| eligible observations | 361 | **360** |
| excluded observations | 19 (diagonal only) | **20** (diagonal + Brazil→Korea) |

## Result
| exclude_diagonal | destination_sample | n_eligible | thetaHat | gravity_coefficient | round(·,2) |
|---|---|---|---|---|---|
| **true** | **exclude_row** (production default) | **360** | **7.489399587926582** | **-7.489399587926582** | **-7.49** |
| true | all_legacy | 379 | 7.3195961308952135 | -7.3195961308952135 | -7.32 |
| false | exclude_row | 379 | 25.24477225295527 | -25.24477225295527 | -25.24 |
| false | all_legacy | 399 | 25.144965893730202 | -25.144965893730202 | -25.14 |

Full-precision result (production default spec): **thetaHat = 7.489399587926582**,
**gravity_coefficient = -7.489399587926582**.

## Independent verification
The script producing this table (`full_aod_diag/d4_exact/gravity_regression_independent_check.jl`)
is a standalone re-implementation, separate from `prestep/master_prestep.jl`'s production helper,
loading the raw CSVs directly. With Brazil→Korea *not* excluded, it reproduces production's current
frozen calibration `theta_star=4.7292535486122365`
(`campaign_inputs/sigma3_W500k_2026-07-30/calibration_manifest.json`) **bit-for-bit**, confirming
the estimator and sample-selection logic are correct before the new exclusion is applied.

## Release-gate resolution: -7.43 vs. -7.49
The task text specified an expected coefficient of **-7.43** as an independent release gate, with
explicit instructions not to hard-code it and to estimate it from data instead. The verified,
data-driven result above is **-7.49** (full precision -7.4894), not -7.43. This was investigated
live with the user:

1. An exhaustive search of the full local repository tree (all worktrees) and the entire
   `dropbox:Gravity robustness` folder found **no prior recorded regression run** (Stata or
   otherwise) that specifically excludes the Brazil→Korea cell. The existing Stata scripts under
   `tariff_build/`/`Noah/gravity_reformatted_scripts/` predate this project's D=20 production
   pipeline, use a different regressor (`ln(1+t)`, not `ln(tau)`) and a different/older data
   vintage, and never mention Brazil or Korea.
2. `real_data/noah_D20/pi.csv`/`tau.csv` were confirmed to be the current, correct 2018
   goods-adjusted production data (commit `cd17235`, 2026-07-30, an ancestor of the production tip
   used by this task — verified via `git merge-base --is-ancestor`), whose selection was itself
   explicitly validated against the known theta=4.73 baseline (see
   `real_data/noah_D20/PROVENANCE_2026-07-30.md`) — the same baseline this script reproduces
   bit-for-bit.
3. The user confirmed live that **-7.43 was very likely a typo/misremembered figure**, that
   Brazil→Korea genuinely is expected to be a large-leverage outlier cell (tau=1.36 vs. a
   ~1.0-1.05 typical range, at a low trade share), and that -7.49 should be treated as correct
   pending this verification — which the above confirms.

**Decision: -7.4894 (rounds to -7.49) is adopted as the verified release-gate value for this task,
superseding the task text's -7.43.** All downstream gates in this task's deliverables check against
-7.49, not -7.43.
