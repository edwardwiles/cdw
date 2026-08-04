# Scientific manifest validation — continuation/polish campaign, 2026-08-03

Gate required by the task spec (Section 3) before importing any seed: `SCIENTIFIC_MANIFEST_MATCH = true`.

## Method

Not taken on the handoff doc's word. Directly traced the full call chain in this worktree
(`campaign/fullA-continuation-polish-2026-08-03`, checked out from `origin/production/fullA-exact`
@ `4c3dad5c4ff208b6abd560176ba3d0007ee4888b`) for every entry point the old campaign actually used,
per the CLAUDE.md rule that an intermediate layer's own default — not the deepest function's, not
memory of what "should" be true — governs behavior unless a caller overrides it.

**Zero drift check**: `git diff --stat 146b10e82b39585cba311510ee78122f1c45baed 4c3dad5 -- full_aod_diag/d4_exact/`
returns empty — this worktree's driver code is byte-identical to the campaign's own commit. No
version-skew risk for any of the checks below.

## Field-by-field comparison

| Field | Required (task Section 3) | Campaign manifest / runner | Verified how |
|---|---|---|---|
| Formulation | FULL gamma-normalized | `run_polish_checkpointed_unified` / `run_cm_upper_checkpointed` / `run_originzc_upper_checkpointed` (FULL family, not REDUCED) | `campaign_unrestricted_runner.jl:171`, `campaign_cm_family_runner.jl:189-200` call these, not any `profiled_*`/`reduced_*` REDUCED entry point |
| D / focal | D=20, France focal | `D=20, D_dest=19` in manifest `config` | `COMMON_FIVE_STARTS_MANIFEST_2026-07-28.json` `config.D=20, config.D_dest=19` |
| sigma | 3.0 | `σHat::Union{Nothing,Float64} = 3.0` (own default, not overridden by any campaign call site) | `c10_d20_production_driver_unified.jl:168`, `cm_checkpoint.jl:103`, `cm_originzc_checkpoint.jl:33`; confirmed no campaign runner passes `σHat=` at any call site (`grep σHat campaign_*.jl` → 0 hits) |
| gravity theta/mask, own-trade excluded | current mask, diagonal excluded | `exclude_diagonal_gravity::Bool = true` (own default, not overridden) | same 3 function signatures as sigma row |
| Brazil-Korea excluded | excluded | `gravity_exclude_cells::AbstractVector = default_gravity_exclude_cells_brazil_korea()` (own default) | same 3 function signatures |
| destination_sample | `:exclude_row` | `:exclude_row` | Explicit at `campaign_unrestricted_runner.jl:172`; own default `= :exclude_row` at `cm_checkpoint.jl:696` / `cm_originzc_checkpoint.jl:506`, not overridden by CM-family runner |
| draw_design | randomized Sobol | `:sobol_randomized` | `campaign_unrestricted_runner.jl:41`, `campaign_cm_family_runner.jl:67` (`const DRAW_DESIGN = :sobol_randomized`) |
| draw_seed | (manifest-pinned) | `20260719` | manifest `config.draw_seed`, cross-checked against per-start `checksum_w_hash` re-verification both runners perform at startup (`campaign_*_runner.jl` refuses to run on checksum mismatch) |
| W | 100,000 | `100000` | manifest `config.W`; CLI arg confirmed live via `ps` during monitoring (`... campaign_cm_family_runner.jl flexible_cm lower ... 7200`, W baked into manifest not CLI) |
| Family-specific L/K | production values | `CM_L=50, cm_contrasts=:orthonormal` (Frechet/flexcm/cm_meanzc); `MEANZC_K_mean=MEANZC_K_pair=1`; `ORIGINZC_K_mean=ORIGINZC_K_pair=1` | manifest `config` block + `campaign_cm_family_runner.jl:69-71` constants (`CM_L=50, MEANZC_K=1, ORIGINZC_K=1`) — both sources agree |
| nu policy | same as original FULL campaign | `cm_meanzc_nu=[1.0]`, `origin_zc_nu` = 19-vector all ≈1.0 (max abs deviation from 1.0 ≈ 5e-5) | manifest `shared_extra_coordinates`, shared identically across all 5 families' starts (same JSON object) |
| A_coordinate_mode | (not explicitly required by task, but load-bearing for continuity) | `:powered_aspace` (current production default) | manifest `config.A_coordinate_mode="powered_aspace"`; confirmed **not** overridden by `campaign_cm_family_runner.jl` (no `A_coordinate_mode=` at either `fn(...)` call site) — this campaign is **not** subject to the `:legacy_z` staleness issue found in the separate, unrelated FULL-vs-REDUCED diagnostic session (`unrestricted_flexcm_stopped_settings_finding_2026-08-03`); that issue was specific to ad-hoc scripts outside this campaign |

## Verdict

```
SCIENTIFIC_MANIFEST_MATCH = true
```

Every field the task requires matches, traced through the actual call chain (not the deepest
function alone, not an intermediate layer alone — every layer between the campaign runner and the
context builder). Zero code drift between the campaign's commit and this worktree's HEAD for the
relevant files. Seeds from `CONTINUATION_SEED_MANIFEST_2026-08-03.csv` are cleared for import.

One caveat carried forward from the handoff (not a manifest mismatch, a separate open item):
whether the `.jls` checkpoint files actually bundle a separable, reusable inner dual alongside the
outer vector has not yet been confirmed by deserializing one — next step.
