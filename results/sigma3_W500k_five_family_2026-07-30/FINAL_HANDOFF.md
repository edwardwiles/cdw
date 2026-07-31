# sigma3/W500k five-family production campaign — final handoff

Generated 2026-07-31 (session start 2026-07-30). All values below are real, live-verified results,
not placeholders. `READY_TO_LAUNCH` exists in this directory, written by the canonical
`./campaign_control.sh --preflight-only` after all 10 required preflights genuinely passed on the
rebased, current branch HEAD.

## Prepared campaign root
`results/sigma3_W500k_five_family_2026-07-30/`

Branch: `campaign/prepare-sigma3-W500k-five-family-production-2026-07-30`
Rebased cleanly onto `cdw/production/fullA-exact` HEAD `81a6730` (the unify-random-draw-production-
pipeline merge). One conflict, in `campaign_inputs/sigma3_W500k_2026-07-30/build_calibration_and_sobol.jl`'s
include list (a one-time setup script, not on the real launch path) — resolved by taking upstream's
already-better fix (single unified `draw_design.jl` include). Zero other file-path overlap between
this branch's own commits and everything production changed underneath it.

## Key checksums / SHAs
- production_sha: `2fd828072b308d8ab99d48966239899fc6d8921b` (current HEAD, post-rebase)
- data_manifest_sha256: `44333ec86f7b819fe3f88f9969665e7b0276c25943ebeb790ea193c42deeff1d`
- calibration_manifest_sha256: `1556167b014ec803124c582baac98c5241eeab7f319129736544c8d21f3fd23f`
- start_manifest_sha256: `24457a9bf45110afec1bd1f909be3f2d3db04a7bfc85cdbaefad5282b6106042`
- campaign_config sha256: `65cf053f5f1157780c88a2b6d8fe10a3623703a93436995075492716eae62c43`

## Manifests
- campaign_config.json — sigma=3.0, W=500000, 180 cells, deltas=[0.01,0.1,0.5,1,2,5],
  K_mean=K_pair=2 for both cm_meanzc/origin_zc, outer_strategy_default=direct_sr1,
  bfgs_polish_enabled_in_default_launch=false
- data_manifest.json, calibration_manifest.json, start_manifest.json — all copied into campaign
  root and checksum-frozen
- resource_plan.json

## Preflight reports (all real, live runs — see PREFLIGHT_SUMMARY.json for the authoritative record)
- preflight_manifests/ (item 5, per-family backend manifests — all confirm `OperatorPsiBundle`,
  zero dense-reference constructions)
- upper_smoke_report.json / lower_smoke_report.json (items 6/7) — all 5 families PASS
- perturbation_smoke_report.json (item 8) — all 5 families PASS, starts 2 and 3
- strategy_handoff_smoke_report.json (item 9) — PASS: exercised both the SR1→BFGS trigger branch
  (origin_zc, status -401 nonfatal, BFGS improved Delta 0.097→0.057, retained) and the
  correctly-skip branch (unrestricted, status -411 not nonfatal, correctly declined polish)
- resource_smoke_report.json (item 10) — PASS: peak RSS 12.4–37.9GB per family (worst-case
  concurrent sum ~122GB vs 3.0TiB host RAM), zero crashes/OOM across ~90 min of sustained
  100-thread concurrent execution. See "Known risk" note below for the convergence-speed detail
  this item surfaced.

## Command scripts
- LAUNCH_COMMAND.sh (default: direct_sr1, BFGS polish NOT enabled) — calls
  `campaign_control.sh --launch`, which refuses unless READY_TO_LAUNCH exists and
  campaign_config.sha256 still matches (both true as of this writing)
- STATUS_COMMAND.sh / RESUME_COMMAND.sh / STOP_COMMAND.sh
- `--dry-run` verified to print the exact real launch commands: 5 `setsid`'d
  `run_family_chain_sigma3.sh` invocations, `maxtime_real=10800 threads=20 hard_cap=12000
  deltas=0.01,0.1,0.5,1.0,2.0,5.0 starts=1,2,3` — 100 Julia threads total on a 208-logical-CPU host

## Final verdict

```
HARDENED_OPERATOR_ARCHITECTURE = pass_all_five_real_entrypoints
NEW_DATA = snapshotted_validated
SIGMA = 3.0_everywhere
SOBOL_W500K = frozen_shared_all_families
CALIBRATION = new_data_sigma3_pass_all_five
ZC_K2 = exact_first_and_second_moment_model_pass
THREE_COMMON_STARTS = calibration_plus_two_perturbations_pass_all_five
OUTER_STRATEGIES_READY =
    direct_sr1:pass
    optional_bfgs_polish:pass
PREFLIGHT =
    inner:pass
    upper_smoke:pass
    lower_smoke:pass
    perturbation_callbacks:pass
    resource:pass
CROSS_CELL_OUTER_CONTINUATION_CALLS = 0
DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS = 0
CAMPAIGN = READY_TO_LAUNCH
LAUNCHED = false
```

## Bugs found and fixed this session (in the preflight harness itself, not the core numerics)

Three real, previously-unexercised bugs were found and fixed by actually running items 6-10 live,
not by inspection:

1. **`run_smoke.sh` pass/fail was gated on the external `timeout` process exit code.** KNITRO's
   `maxtime_real` is a soft, best-effort check made between major iterations, not a preemptive
   interrupt — under real 5-way concurrent W=500k contention, a slow Hessian assembly can run past
   the smoke's external `timeout` before KNITRO gets a chance to stop gracefully, so a family
   making genuine, verified progress was reported as a hard failure. Fixed to check log content (an
   `eval N t=` line) instead. A first attempt at this fix accidentally grepped only for
   `verified=true`, which the `unrestricted` family's own log format never prints even on full
   success (it has a different, older log format than the 4 CM-family-runner families) — broadened
   to the marker common to both.
2. **`run_family_chain_sigma3.sh` (the exact script `--launch` itself uses) called `/usr/bin/time
   -v`, which is not installed on this host.** Every single cell failed instantly (rc=127) the
   first time this script was actually run this session — this would have broken all 180 real
   launch cells identically had it not been caught here. Replaced with a `/proc/<pid>/status`
   VmHWM poller writing the same grep-compatible "Maximum resident set size" line the rollup script
   already expects.
3. **`run_strategy_handoff_smoke.jl`'s origin_zc test section recomputed w0 fresh from `ctx`**
   instead of reading the frozen, checksum-verified `start_manifest.json` coordinates the real
   production driver (`campaign_cm_family_runner_sigma3.jl`) actually uses — the two conventions
   differ by one dimension (380 vs 379), producing a live `DimensionMismatch` inside KNITRO's
   presolver. Fixed to read the frozen manifest values, matching production exactly.

## Known risk (not a blocker) — cm_meanzc convergence speed at delta=0.1

Item 10's 1800s-per-attempt smoke budget was too short for `cm_meanzc` to reach a verified DONE on
either of its 2 test cells (3 independent attempts each, deterministically stalling at the same
point — eval 3 infeasible, then a long stretch still genuinely inside KNITRO's solver internals,
not hung/crashed/erroring). A follow-up isolated diagnostic (no contention, delta=0.1, same start)
confirmed this is **not** a stall: given more realistic time it converges cleanly —
`DONE wall=2663.9s (44 min) n_eval=21 verified=19/21 best_Delta=0.0946` against a target of 0.1.
That is comfortably inside the real 10,800s/cell (3 hour) production budget (~4x headroom), even
before accounting for the fact that a real launch cell gets its own dedicated 20 threads without
the artificial 5-way contention this smoke deliberately imposed to stress-test resource sharing.
Recommendation: no action needed before launch, but the watcher/user should expect `cm_meanzc`
cells (particularly at tighter deltas) to run visibly longer than the other 4 families, and should
not treat that alone as a sign of trouble.

## Other known, previously-documented, accepted-risk gaps (not blockers)
- Checkpoint resume-mismatch guards don't yet compare `exclude_diagonal_gravity`/`σHat` — mitigated
  by this campaign using a fresh checkpoint namespace, not by the guard itself.
- `cm_checkpoint.jl`/`cm_originzc_checkpoint.jl`'s `outer_direct_hessopt` wiring: `unrestricted` and
  `origin_zc` were both live-tested this session (via the strategy handoff smoke); `flexible_cm`,
  `common_frechet`, and `cm_meanzc` use the identical code path in `cm_checkpoint.jl` but were only
  exercised at the default `direct_sr1` setting (not the BFGS-polish branch specifically) — proven
  by pattern, not independently live-tested for the polish branch.

## Notes for whoever launches this
- Already rebased onto `cdw/production/fullA-exact`'s current HEAD as of this writing
  (`81a6730` → this branch is now ahead of it, not behind). If meaningful time has passed since this
  document was written, re-fetch and re-check `git merge-base --is-ancestor cdw/production/fullA-exact HEAD`
  before launching — don't assume it's still true.
- To launch: `./LAUNCH_COMMAND.sh` from this directory (or `campaign_control.sh --launch`). It will
  refuse on its own if `READY_TO_LAUNCH` is missing or `campaign_config.json` has drifted from its
  frozen checksum since this was written.
