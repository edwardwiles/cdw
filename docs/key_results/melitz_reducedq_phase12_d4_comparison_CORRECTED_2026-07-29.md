# Phase 12 D4 outer comparison (CORRECTED, rerun)

CORRECTED Phase 12 D4 outer comparison (2026-07-29 reduced-q validation session, Phase 0). This
table supersedes the original `docs/melitz_reduced_q_subspace_search_2026-07-29.md` Phase 12
table and its CSV: (1) the CSV is now properly quoted/standards-compliant; (2) `n_screened_above_cap`
is now its own explicit column, genuinely `0` throughout -- the Phase 7 cap screen did not fire
in this smoke test at any cell, and the ORIGINAL doc's prose attributing 91/157/111 "screened"
evaluations to the cap screen was a misread of the `n_numerical_failure` column; (3) EVERY row
here was produced by a FRESH rerun (not the original run's numbers) because the original
`sequential_reduced_q` rows' own `n_finite_solved+n_above_cap` totals failed a basic
trial-count cross-check that this script's `melitz_validate_typed_counters` now enforces before
any export -- see this script's own header comment for the full diagnosis. Every row below
therefore passed `melitz_validate_typed_counters(...; n_trials=<independently recorded trial
count>)` at write time.

| label | delta | direction | nStatus | wall_s | start_gt_pct | best_gt_pct | kappa_best | delta_best | within_budget | n_accepted_stages | n_accepted_steps | n_finite_solved | n_above_cap_evaluated | n_screened_above_cap | n_above_cap_total | n_infinite_delta | n_numerical_failure | n_affine_excluded | termination_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| production_(A,f) | 0.1 | upper | -410 | 17.66770100593567 | 6.523745799909331 | 10.158972621462247 | 0.8984102737853775 | 0.09239216502224216 | true | 1 | 82 | 12 | 15 | 0 | 15 | 55 | 0 | 0 | -410 |
| full_experimental_(A,q) | 0.1 | upper | -410 | 3.858374834060669 | 6.523745799909331 | 9.280834344707333 | 0.9071916565529267 | 0.05113469427754156 | true | 1 | 85 | 14 | 17 | 0 | 17 | 54 | 0 | 0 | -410 |
| sequential_reduced_q | 0.1 | upper | -1 | 28.425582885742188 | 6.523745799909331 | 12.313418444675605 | 0.8768658155532439 | 0.09998889033643815 | true | 5 | 494 | 369 | 34 | 0 | 34 | 0 | 91 | 0 | no_improvement_streak |
| production_(A,f) | 0.1 | lower | -410 | 2.875948190689087 | 6.523745799909331 | 3.5342692584883206 | 0.9646573074151168 | 0.09940544818169578 | true | 1 | 107 | 38 | 59 | 0 | 59 | 10 | 0 | 0 | -410 |
| full_experimental_(A,q) | 0.1 | lower | -410 | 2.9908430576324463 | 6.523745799909331 | 4.447560584108534 | 0.9555243941589147 | 0.06676548807543914 | true | 1 | 77 | 25 | 33 | 0 | 33 | 19 | 0 | 0 | -410 |
| sequential_reduced_q | 0.1 | lower | -1 | 18.915787935256958 | 6.523745799909331 | 3.077842181952084 | 0.9692215781804792 | 0.0998507691328749 | true | 3 | 317 | 259 | 12 | 0 | 12 | 0 | 46 | 0 | no_improvement_streak |
| production_(A,f) | 0.5 | upper | -410 | 3.131188154220581 | 6.523745799909331 | 14.867338835156986 | 0.8513266116484302 | 0.4928896976552007 | true | 1 | 97 | 29 | 66 | 0 | 66 | 2 | 0 | 0 | -410 |
| full_experimental_(A,q) | 0.5 | upper | -410 | 1.9265689849853516 | 6.523745799909331 | 13.049876788761095 | 0.869501232112389 | 0.39127213338966443 | true | 1 | 78 | 10 | 13 | 0 | 13 | 55 | 0 | 0 | -410 |
| sequential_reduced_q | 0.5 | upper | -1 | 19.27854299545288 | 6.523745799909331 | 17.39380804190056 | 0.8260619195809944 | 0.49932065421858923 | true | 5 | 550 | 117 | 236 | 0 | 236 | 40 | 157 | 0 | max_stages |
| production_(A,f) | 0.5 | lower | -410 | 2.282409191131592 | 6.523745799909331 | 2.9455898062457964 | 0.970544101937542 | 0.4614820178558934 | true | 1 | 96 | 15 | 58 | 0 | 58 | 23 | 0 | 0 | -410 |
| full_experimental_(A,q) | 0.5 | lower | -410 | 3.5596630573272705 | 6.523745799909331 | 3.0321561228311866 | 0.9696784387716881 | 0.49941207027022594 | true | 1 | 103 | 33 | 24 | 0 | 24 | 46 | 0 | 0 | -410 |
| sequential_reduced_q | 0.5 | lower | -1 | 28.455984115600586 | 6.523745799909331 | 0.056170094332419485 | 0.9994382990566758 | 0.4766272477161889 | true | 5 | 491 | 241 | 139 | 0 | 139 | 0 | 111 | 0 | max_stages |
