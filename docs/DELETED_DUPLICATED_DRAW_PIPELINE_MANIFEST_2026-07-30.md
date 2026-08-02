# Deleted duplicated draw pipeline manifest (task §19)

## Deleted

- **`full_aod_diag/d4_exact/qmc_context_real_d20.jl`** (450 lines) -- entire file removed, commit
  `1ae1f54`. Contained the three duplicated functions:
  - `master_prepare_cc_qmc` (near-identical duplicate of `prepare_cc/master_prepare_cc.jl::master_prepare_cc`)
  - `build_ad_context_real_d20_qmc` (routing duplicate of `context_real_d20.jl::build_ad_context_real_d20`)
  - `d20_real_setup_qmc` (drifted duplicate of `context_real_d20.jl::d20_real_setup` -- missing
    `threshold_state`, missing screen construction)

  Also contained `exp_from_uniform01`, which was NOT dead code (used by `qmc_draws.jl` and
  `c10_stratmarg_draws.jl`) -- relocated to `prepare_cc/genRands.jl` as a thin wrapper around the
  new shared `transform_unit01_to_exp1!`, not deleted.

- **Compensating screen-construction patch** in `draw_design.jl`'s `:sobol_randomized`/
  `:halton_scrambled` branch (~15 lines: the `screen_pairwise = nothing; ...; if build_screen ...`
  block that rebuilt `ctx.pairwise`/`ctx.witness` after the fact because `d20_real_setup_qmc`
  didn't build them). Removed because `d20_real_setup` now builds screens for every design, so
  there is nothing left to compensate for.

## Not deleted (found to not exist)

- No `production_campaign_QMC` script, QMC-specific family runner, or QMC-specific supervisor was
  found anywhere in the repository (`DUPLICATED_CAMPAIGN_PIPELINE` in the reachability audit) --
  there was nothing of this kind to delete.

## Repointed, not deleted

Five diagnostic scripts that called `d20_real_setup_qmc` directly were repointed onto
`d20_real_setup(...; U=...)` (the same unified function every other caller now uses), not deleted --
they retain independent historical/diagnostic value (a stratified-marginal tail-coverage
investigation, a QMC-precision comparison, a draw-type continuation comparison) unrelated to the
duplication bug itself:

- `c10_stratmarg_screen_sweep.jl`
- `c10_stratmarg_followup_w20000.jl`
- `c10_phase7_short_continuation.jl`
- `c10_phase7_qmc_wiring_smoketest.jl`
- `c10_phase7_qmc_precision_comparison.jl`

Plus one comment-only fix (no code change) in `c10_stratmarg_draws.jl`, whose `exp_from_uniform01`
dependency moved location.

One stray include of the now-deleted file was found and fixed during the merge with concurrently-
landed `production/fullA-exact` work: `campaign_inputs/sigma3_W500k_2026-07-30/build_calibration_and_sobol.jl`
(a campaign-prep script that landed on `production/fullA-exact` itself, not the live campaign's own
untouched worktree).

## Final verification

```
$ rg -n "master_prepare_cc_qmc"          # 0 live hits (grep-confirmed, see reachability audit)
$ rg -n "build_ad_context_real_d20_qmc"  # 0 live hits
$ rg -n "d20_real_setup_qmc"             # 0 live hits
$ rg -n "production_campaign_QMC"        # 0 hits (never existed)
$ rg -n "screen parity"                  # 0 hits (the compensating-patch comment is gone)
$ bash scripts/static_draw_design_duplication_guard_2026-07-30.sh   # PASS, 0 violations
```

```
DUPLICATED_QMC_CONTEXT_FUNCTIONS_REMAINING = 0
DUPLICATED_QMC_CAMPAIGN_BODIES_REMAINING = 0
```

Git history (not a preserved fallback body) is the record of the deleted code, per task
instruction -- see commit `1ae1f54` and the reachability/call-graph docs for exact prior content
and line ranges.
