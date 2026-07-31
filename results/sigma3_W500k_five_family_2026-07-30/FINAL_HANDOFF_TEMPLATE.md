# sigma3/W500k five-family production campaign — final handoff (TEMPLATE, fill in with real values before use)

**Do not treat this file as a completion signal.** It is a template only until every placeholder
below is replaced with a real, verified value and the file is renamed/finalized. The watcher
Claude should key off `READY_TO_LAUNCH` + `PREFLIGHT_SUMMARY.json`, not this file's mere existence.

## Prepared campaign root
`results/sigma3_W500k_five_family_2026-07-30/`

## Key checksums / SHAs
- production_sha: <from campaign_config.json>
- data_manifest_sha256: <...>
- calibration_manifest_sha256: <...>
- start_manifest_sha256: <...>
- campaign_config sha256: <cat campaign_config.sha256>

## Manifests
- campaign_config.json
- data_manifest.json
- calibration_manifest.json
- start_manifest.json
- resource_plan.json

## Preflight reports
- PREFLIGHT_SUMMARY.json (authoritative)
- preflight_manifests/ (item 5, per-family backend manifests)
- upper_smoke_report.json / lower_smoke_report.json (items 6/7)
- perturbation_smoke_report.json (item 8)
- strategy_handoff_smoke_report.json (item 9)
- resource_smoke_report.json (item 10)

## Command scripts
- LAUNCH_COMMAND.sh (default: direct_sr1, BFGS polish NOT enabled)
- STATUS_COMMAND.sh
- RESUME_COMMAND.sh
- STOP_COMMAND.sh

## Final verdict

```
HARDENED_OPERATOR_ARCHITECTURE = <pass_all_five_real_entrypoints | fail_...>
NEW_DATA = <snapshotted_validated | fail_...>
SIGMA = <3.0_everywhere | fail_...>
SOBOL_W500K = <frozen_shared_all_families | fail_...>
CALIBRATION = <new_data_sigma3_pass_all_five | fail_...>
ZC_K2 = <exact_first_and_second_moment_model_pass | fail_...>
THREE_COMMON_STARTS = <calibration_plus_two_perturbations_pass_all_five | fail_...>
OUTER_STRATEGIES_READY =
    direct_sr1:<pass/fail>
    optional_bfgs_polish:<pass/fail>
PREFLIGHT =
    inner:<pass/fail>
    upper_smoke:<pass/fail>
    lower_smoke:<pass/fail>
    perturbation_callbacks:<pass/fail>
    resource:<pass/fail>
CROSS_CELL_OUTER_CONTINUATION_CALLS = 0
DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS = 0
CAMPAIGN = <READY_TO_LAUNCH | NOT_READY_<single_blocker>>
LAUNCHED = false
```

## Notes for whoever launches this
- Rebase this branch onto cdw/production/fullA-exact's current HEAD before launching (the unified
  random-draw pipeline work landed there mid-session; confirmed zero file overlap with this
  campaign's own commits at rebase time -- but re-verify, don't assume it's still true if time has
  passed).
- Known, documented, accepted-risk gaps (not blockers, but worth knowing): checkpoint resume-
  mismatch guards don't yet compare exclude_diagonal_gravity/sigmaHat (mitigated by a fresh
  checkpoint namespace, not by the guard itself); cm_checkpoint.jl/cm_originzc_checkpoint.jl's
  outer_direct_hessopt wiring is proven-by-pattern but only unrestricted was the FIRST family
  live-tested (see commit history -- origin_zc was subsequently tested too, in the strategy
  handoff smoke).
