# QMC/pseudorandom duplication: classified reachability inventory (2026-07-30)

Every repository occurrence of `_qmc|QMC|sobol|halton|pseudorandom|draw_design` was enumerated
(180 files matched the raw grep) and triaged. Below, occurrences are grouped by classification;
individual test/bench/diagnostic files that merely *call* the stable `d20_real_setup_design`
resolver (its signature does not change in this refactor) are summarized by count rather than
listed file-by-file, since none of them require code changes.

## DRAW_GENERATION_REQUIRED (design-specific code, correctly stays design-specific)

- `full_aod_diag/d4_exact/qmc_draws.jl` -- `pseudorandom_U`, `halton_U`, `sobol_U` generators.
- `prepare_cc/drawU.jl`, `prepare_cc/genRands.jl` -- production pseudorandom generation
  (`genExpRands!`, `genExpRandsStratified!`, `genExpRandsImportanceSampling!`) and the Exp(1)
  transform used when no `U` is injected.
- `cc_algo/rhalton.jl` -- scrambled-Halton implementation (Owen-style digit scrambling).
- `cc_algo/boot.jl` -- unrelated `sobol`-adjacent match, confirmed false-positive on manual read
  (matched on an unrelated bootstrap-sampling comment, not draw-design code).

## DRAW_METADATA_REQUIRED (provenance fields/functions, no numerical branch -- correct as-is)

- `draw_design::Symbol` struct fields: `cm_checkpoint.jl` (6 occurrences), `cm_originzc_checkpoint.jl`
  (3), `cm_checkpoint_fingerprint.jl` (1), `c10_d20_production_driver.jl` (2),
  `c10_d20_production_driver_unified.jl` (1), `c10_d20_production_driver_flexible_theta_A.jl` (1).
- `draw_design.jl::draw_design_meta` and `VALID_DRAW_DESIGNS`/`DRAW_DESIGN_DESCRIPTIONS` --
  provenance-logging only, no dispatch on numerical behavior.
- `guard_checkpoint_path`/`reuse_matches`/`CMCheckpointV*` mismatch checks compare `draw_design`
  for equality against a requested value; never branch code paths on it.

## DUPLICATED_CONTEXT_OR_SOLVER_PIPELINE (the real defect -- see call-graph doc for line ranges)

1. `master_prepare_cc` (`prepare_cc/master_prepare_cc.jl`) / `master_prepare_cc_qmc`
   (`full_aod_diag/d4_exact/qmc_context_real_d20.jl:62-354`).
2. `build_ad_context_real_d20` (`context_real_d20.jl:45-54`) / `build_ad_context_real_d20_qmc`
   (`qmc_context_real_d20.jl:357-366`).
3. `d20_real_setup` (`context_real_d20.jl:66-188`) / `d20_real_setup_qmc`
   (`qmc_context_real_d20.jl:380-450`) -- drifted (missing `threshold_state`, missing screen block).
4. The screen-construction block itself: 2 live copies (`context_real_d20.jl` inline,
   `draw_design.jl`'s QMC-branch patch) compensating for the 3rd copy's absence in
   `d20_real_setup_qmc`.
5. `exp_from_uniform01` (`qmc_context_real_d20.jl:46-49`) -- not itself duplicated, but homed
   inside the file being deleted; needs relocation to shared infrastructure it can share with
   `genExpRands!`'s equivalent inline transform (task step 3: one shared implementation).

**Total distinct duplicated/drifted function bodies: 3 (+1 triplicated helper block, screens).**
**Files containing duplicate pipeline code: 1 (`qmc_context_real_d20.jl`, 450 lines) + drift
compensation in `draw_design.jl` (~15 lines).**

## DUPLICATED_CAMPAIGN_PIPELINE

**None found.** Exhaustive search (`find -iname "*campaign*"`, `find -iname "*QMC*"`,
`rg "production_campaign"`) across the full repository turned up exactly 3 campaign scripts
(`campaign_cm_family_runner.jl`, `campaign_unrestricted_runner.jl`, `campaign_cell_io.jl`), all
fully generic over `draw_design`, and zero QMC-specific campaign scripts, supervisors, or family
runners at any point in git history (`git log --all --grep='QMC'` on the campaign-script paths
returns nothing beyond the exploratory `c10_phase7_*`/`c10_stratmarg_*` diagnostics classified
below). `DUPLICATED_QMC_CAMPAIGN_BODIES_REMAINING = 0` already, before any deletion.

## DIAGNOSTIC_OR_TEST_ONLY

- **~100 production-facing call sites of `d20_real_setup_design`** across production drivers
  (`c10_d20_production_driver.jl` and its `_unified`/`_flexible_theta_A` variants),
  checkpoint/stage-runner infrastructure (`cm_checkpoint.jl`, `cm_originzc_checkpoint.jl`,
  `cm_production_stage_runner.jl`, `originzc_production_stage_runner.jl`, `reusable_context.jl`,
  `reconcile_checkpoint*.jl`, `checkpoint_resume_exclude_row_cplus.jl`, `cm_cold_verify.jl`,
  `originzc_cold_verify.jl`), and ~80 `test_*.jl`/`bench_*.jl`/`diag_*.jl`/numbered-continuation
  (`c9_`-`c34_`) scripts. **None require changes** -- `d20_real_setup_design`'s signature and
  return shape are unchanged by this refactor; only its internals stop branching post-draw.
- `test_draw_design.jl` -- the existing draw-design regression test; extended, not replaced (§14
  of the implementation).
- `full_aod_diag/d4_exact/qmc_fixed_points/{lower,upper}_candidate_w.csv` -- cached numeric data
  from a prior investigation, not code; unaffected.
- `docs/fullA_D20_qmc_investigation_report.md` -- historical investigation report; left as
  historical record (not updated retroactively, matching this repo's convention of not rewriting
  past dated reports).

### The 5 scripts that call the duplicated pipeline directly (need updating in Phase 9, not deletable as pure dead code)

`c10_stratmarg_screen_sweep.jl`, `c10_stratmarg_followup_w20000.jl`,
`c10_phase7_short_continuation.jl`, `c10_phase7_qmc_wiring_smoketest.jl`,
`c10_phase7_qmc_precision_comparison.jl`, and `c10_stratmarg_draws.jl` (calls
`exp_from_uniform01` directly, doesn't call `d20_real_setup_qmc` itself but shares the same
include chain). These are one-off numbered-continuation diagnostic scripts (not part of any
standing test/gate suite -- confirmed by checking they are not referenced from any test-runner
script or CI list), but since deleting `d20_real_setup_qmc`/`exp_from_uniform01`'s current location
would break them outright, they are repointed at the new `PrecomputedDrawDesign` resolver path
(§9) rather than left broken or silently deleted.

## DEAD_CODE

**None identified.** Every file matching the grep pattern is either genuinely draw-related,
metadata-related, or a live (if historical/diagnostic) caller. No orphaned QMC-era code with zero
callers was found.

## Summary table

| Classification | Count | Action |
|---|---|---|
| DRAW_GENERATION_REQUIRED | 4 files | Keep unchanged |
| DRAW_METADATA_REQUIRED | ~14 occurrences across 6 files | Keep unchanged |
| DUPLICATED_CONTEXT_OR_SOLVER_PIPELINE | 3 function pairs + 1 triplicated block, 2 files | Unify + delete |
| DUPLICATED_CAMPAIGN_PIPELINE | 0 | None to do |
| DIAGNOSTIC_OR_TEST_ONLY | ~100 stable call sites (no change) + 6 files needing a repoint | Repoint 6 files only |
| DEAD_CODE | 0 | None to do |
