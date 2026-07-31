# Campaign driver consolidation (task §17)

## Finding: there was nothing to consolidate

The task instructs deleting/retiring separate `production_campaign_QMC`, `production_campaign_real`,
QMC-specific family runners, and QMC-specific supervisors, and keeping one canonical campaign
driver whose draw-related arguments are limited to `--draw-design`/`--draw-seed`/`--draw-artifact`.

The reachability audit (`QMC_PSEUDORANDOM_DUPLICATION_REACHABILITY_2026-07-30.md`,
`DUPLICATED_CAMPAIGN_PIPELINE`) already established this exhaustively: there is exactly one family
of campaign scripts in this repository (`campaign_cm_family_runner.jl`,
`campaign_unrestricted_runner.jl`, `campaign_cell_io.jl`, plus the checkpoint/stage-runner
infrastructure `cm_checkpoint.jl`/`cm_originzc_checkpoint.jl`/`cm_production_stage_runner.jl`/
`originzc_production_stage_runner.jl`), and none of them are QMC-specific. `git log --all --grep`
and `find -iname "*campaign*QMC*"`/`-iname "*QMC*campaign*"` across the whole repository (not just
this branch) return nothing. `DUPLICATED_QMC_CAMPAIGN_BODIES_REMAINING = 0`, and it was already 0
before this task started -- the campaign layer was already correctly built on top of the
`draw_design::Symbol` provenance-field convention, never on a duplicated QMC-specific driver.

## What the campaign layer already does correctly (verified, not assumed)

Every real campaign entry point (`run_cm_upper_checkpointed`/`run_originzc_upper_checkpointed`/
the third real driver in `c10_d20_production_driver_unified.jl`, and their checkpoint structs)
already:

- accepts `draw_design::Symbol` and `draw_seed::Int` as ordinary keyword arguments;
- calls the single `d20_real_setup_design` resolver (this task's work made that resolver's
  *internals* branch-free post-draw; the campaign layer's own call sites needed no changes at
  all, since `d20_real_setup_design`'s signature is unchanged);
- persists `draw_design`/`draw_seed` as checkpoint provenance fields, compared for equality on
  resume (`guard_checkpoint_path`, `reuse_matches`) -- never as a numerical dispatch.

Confirmed live (2026-07-30, this task): after merging in the concurrently-landed
`exclude_diagonal_gravity`/`σHat` production-driver wiring (commit `8c1832e` on
`production/fullA-exact`), all 3 real drivers still call `d20_real_setup_design` with the same
kwarg shape; no campaign-layer code needed to change to accommodate either that merge or this
task's internal unification.

## What this task did add: `--draw-design precomputed`

The one gap found was not at the campaign-driver level but at the diagnostic level: five
historical scripts (`c10_stratmarg_screen_sweep.jl` and friends) called the now-deleted
`d20_real_setup_qmc` directly with a hand-built matrix, because `d20_real_setup_design` had no
"inject a precomputed matrix" option. `draw_design.jl` now accepts `draw_design=:precomputed` with
a `U_precomputed` kwarg (backed by `PrecomputedDrawDesign`, `draw_design_types.jl`), and those five
scripts were repointed onto it (see `POST_DRAW_METHOD_IDENTITY_PROOF_2026-07-30.md` and the commit
"Repoint the 5 diagnostic scripts..."). This closes the one path that previously required a
QMC-specific entry point for anything other than generated Sobol/Halton draws.

## Verdict

```
PRODUCTION_CAMPAIGN_DRIVER = single_canonical
DUPLICATED_QMC_CAMPAIGN_BODIES_REMAINING = 0
```
