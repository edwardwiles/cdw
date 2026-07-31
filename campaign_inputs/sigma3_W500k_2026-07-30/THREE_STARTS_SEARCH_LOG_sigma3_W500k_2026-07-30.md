# Common five-family starting points -- search log (sigma3_W500k_2026-07-30)

Script: `full_aod_diag/d4_exact/common_five_starts_search.jl`  
Run: 2026-07-30T21:31:36.540  |  wall 17.4 min  |  peak RSS 67918.0 MB

## Configuration

| key | value |
|---|---|
| `W` | `500000` |
| `delta` | `0.01` |
| `find_smallest` | `true` |
| `draw_design` | `sobol_randomized` |
| `draw_seed` | `20260719` |
| `destination_sample` | `exclude_row` |
| `D` | `20` |
| `D_dest` | `19` |
| `A_coordinate_mode` | `powered_aspace` |
| `start_seed` | `20260730` |
| `radius0` | `0.01` |
| `max_per_radius` | `10` |
| `max_candidates` | `60` |
| `feasibility_tol` | `1.0e-6` |
| `cm_L` | `50` |
| `meanzc_K` | `2` |
| `originzc_K` | `2` |
| `draw_checksum_uniform` | `de0acb5535f75e1c62d6c37f47e53d3912b2d072f2b8ef949deacc03c4c58220` |
| `draw_checksum_transformed` | `ce75ae597246c544d55a72413b22bc8992e3263fde67e4cd3a24409e5341ca1e` |

## Outcome

**CORRECTED 2026-07-30** (the auto-generated text below originally read "3 of 6 common starts
accepted... INCOMPLETE", which was misleading -- it compared the FINAL selected count against the
POOL target using stale wording left over from before the SELECT_BY_DISTANCE step was added. The
underlying data (start_manifest.json) was always correct; only this prose was wrong. Corrected by
hand rather than by re-running the 17-minute search, since the source data is unaffected -- see
git history for the actual script fix applied for future runs.)

**Pool search: 6 of 6 candidates accepted** from 7 candidate evaluations (only 1 rejected, by
common_frechet, see the audit trail below) -- the pool search was fully successful, NOT
incomplete. **Final selection: 3 of 3 starts** -- calibration (`start1_calibration`) plus the 2
best-separated perturbations by max-min pairwise distance (`start2_perturbation`,
`start4_perturbation`, selected from the pool of 5 non-calibration candidates; min pairwise
distance = 0.1111). This is exactly the target outcome, not a shortfall.

| start | label | radius | candidate # | gp | max Delta* over families |
|---|---|---|---|---|---|
| 1 | `start1_calibration` | 0.0 | 1 | 0.9748242 | 9.7836e-05 |
| 2 | `start2_perturbation` | 0.01 | 2 | 0.97244003 | 1.5881e-01 |
| 3 | `start4_perturbation` | 0.01 | 4 | 0.97331013 | 6.8297e-02 |

## Radius schedule

| radius | candidates tried | accepted |
|---|---|---|
| 0.01 | 6 | 5 |

## Candidate-by-candidate audit trail

Rejection policy: a candidate failing ANY family is rejected for ALL five. Families are evaluated in the order unrestricted, flexible_cm, common_frechet, cm_meanzc, origin_zc and the check short-circuits at the first failure, so `first failing family` is the first in that order to fail, not necessarily the only one.

| candidate | kind | radius | accepted | first failing family | reason |
|---|---|---|---|---|---|
| 1 | calibration | 0.0 | true | - | `` |
| 2 | perturbation | 0.01 | true | - | `` |
| 3 | perturbation | 0.01 | true | - | `` |
| 4 | perturbation | 0.01 | true | - | `` |
| 5 | perturbation | 0.01 | true | - | `` |
| 6 | perturbation | 0.01 | false | common_frechet | `not_verified_success(ApproximateSolved)` |
| 7 | perturbation | 0.01 | true | - | `` |

## Per-family rejection counts

| family | times it was the first failure |
|---|---|
| unrestricted | 0 |
| flexible_cm | 0 |
| common_frechet | 1 |
| cm_meanzc | 0 |
| origin_zc | 0 |
| gp_bounds | 0 |

## Screen counters (cumulative over the whole search)

| family | calls | pairwise hits | hard-winner hits | passed |
|---|---|---|---|---|
| flexible_cm | 7 | 0 | 0 | 7 |
| common_frechet | 7 | 0 | 0 | 7 |
| cm_meanzc | 6 | 0 | 0 | 6 |
| origin_zc | 6 | 0 | 0 | 6 |

## Artifacts

- `THREE_STARTS_POOL_MANIFEST_sigma3_W500k_2026-07-30.json` -- full manifest (coordinates, A_od, per-family Delta*/residuals/checksums)
- `THREE_STARTS_POOL_MANIFEST_sigma3_W500k_2026-07-30.csv` -- one row per (start, family)
