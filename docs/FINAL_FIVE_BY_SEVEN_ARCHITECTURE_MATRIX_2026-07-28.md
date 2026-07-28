# Final 5×7 architecture matrix — 2026-07-28

Columns: (1) economic moment construction, (2) outer A-gradient, (3) economic FG, (4) restriction
FG, (5) H_EE, (6) H_ER (cross-block), (7) H_RR (restriction self-block). Outer A-gradient is
explicitly out of this task's scope (not touched) — reported as unchanged/pre-existing.

## 1. Unrestricted

| Col | Production backend | Shared fn | Default | Reference backend | D=20 gate | calls moments!? | reads/materializes G? | allocates G-sized storage? |
|---|---|---|---|---|---|---|---|---|
| Economic moment construction | `build_economic_moment_state!`/`cf_build` direct | shared | yes | `:dense` mode | pre-existing | no | no | no |
| Outer A-gradient | unchanged (out of scope) | — | — | — | — | — | — | — |
| Economic FG | `compressed_dual_index!`/`compressed_cc_value_grad!` | `dual_index!(st::CompressedCBState,...)` (this task) | yes | `:dense` | test_shared_core_hessian_d4_gates.jl | no | no | no |
| Restriction FG | n/a (no restriction block) | — | — | — | — | — | — | — |
| H_EE | `winner_pair_hessian!`/`hessian_core_winner_pair!` via `operator_prep_for_hessian!` (this task) | shared | yes | `:dense_reference` | D=4+D=20 PASS | no | no | no |
| H_ER | n/a | — | — | — | — | — | — | — |
| H_RR | n/a | — | — | — | — | — | — | — |

## 2. Flexible CM

| Col | Production backend | Shared fn | Default | Reference backend | D=20 gate | calls moments!? | reads/materializes G? | allocates G-sized storage? |
|---|---|---|---|---|---|---|---|---|
| Economic moment construction | `wrap_moments_with_cm_archB` (priming, once/solve) | family-specific | yes | `:dense_reference` | D=4+D=20 PASS | **yes** (1/solve) | economic: yes (unconditional); CM-grid: **no** (skip_fill_safe re-enabled) | yes (obj.H full width) |
| Outer A-gradient | unchanged (out of scope) | — | — | — | — | — | — | — |
| Economic FG | `CMLookupState`'s `dual_index!` | `economic_forward!` (shared) | yes | dense gemv fallback | D=4+D=20 PASS | no | no | no |
| Restriction FG | `CMLookupState`'s CM-bin lookup | shared bin-lookup kernels | yes | dense gemv fallback | D=4+D=20 PASS | no | no | no |
| H_EE | `_fill_cm_HEE!` winner-pair branch | shared (`fill_core_hessian_upper!`) | yes | dense else-branch (lazy `E`, this task) | D=4+D=20 PASS | no | no (E lazy, this task) | no (view only, not stored) |
| H_ER (H_EC) | `winner_pair_cross_hessian_fill!`/`_block!` | shared | yes | dense else-branch (lazy `E`) | D=4+D=20 PASS | no | no | no |
| H_RR (H_CC) | bin-contingency-table (`build_bin_tables!`, `fill_S=false`) | shared | yes | n/a (always structured) | D=4+D=20 PASS | no | no | no |

## 3. Common Fréchet

| Col | Production backend | Shared fn | Default | Reference backend | D=20 gate | calls moments!? | reads/materializes G? | allocates G-sized storage? |
|---|---|---|---|---|---|---|---|---|
| Economic moment construction | `wrap_moments_with_cm_frechet_archB` (priming) | family-specific | yes | `:dense_reference` | D=4+D=20 PASS | **yes** (1/solve) | economic+CM-grid+level: yes (unconditional, `skip_fill_safe` NOT re-enabled, out of scope) | yes |
| Outer A-gradient | unchanged (out of scope) | — | — | — | — | — | — | — |
| Economic FG | `CMFrechetLookupState`'s `dual_index!` | `economic_forward!` (shared) | yes | dense gemv fallback | D=4+D=20 PASS | no | no | no |
| Restriction FG | CM-bin lookup + level-anchor forward | shared + `frechet_level_forward_sum!` | yes | dense gemv fallback | D=4+D=20 PASS | no | no | no |
| H_EE | `_fill_cm_HEE!` winner-pair branch | shared | yes | dense else-branch (lazy `E`) | D=4+D=20 PASS | no | no | no |
| H_ER (H_EC + H_E,level) | `winner_pair_cross_hessian_fill!` + `winner_pair_cross_hessian_esum!`/`colsum!` | shared | yes | dense else-branch (lazy `E`, **fixed this session**: `threaded_bins=true` now actually wired, was silently hardcoded false) | D=4 20/20 + D=20 ALL PASS | no | no | no |
| H_RR (H_CC + H_level,level) | bin-contingency-table | shared | yes | n/a | D=4+D=20 PASS | no | no | no |

## 4. CM+ZC

| Col | Production backend | Shared fn | Default | Reference backend | D=20 gate | calls moments!? | reads/materializes G? | allocates G-sized storage? |
|---|---|---|---|---|---|---|---|---|
| Economic moment construction | `wrap_moments_with_cm_meanzc` (priming) | family-specific | yes | `:dense_reference` | D=4+D=20 PASS | **yes** (1/solve) | economic+CM-grid+mean/pair: yes (unconditional, no skip mechanism exists) | yes |
| Outer A-gradient | unchanged (out of scope) | — | — | — | — | — | — | — |
| Economic FG | `CMMeanZCOperatorState`'s `dual_index!` | `economic_forward!` (shared) | yes | dense gemv fallback | D=4+D=20 PASS | no | no | no |
| Restriction FG | `restriction_forward!` (ZC) + CM-bin lookup | shared | yes | dense gemv fallback | D=4+D=20 PASS | no | no | no |
| H_EE | `_fill_cm_HEE!` winner-pair branch (widened `ncore_core<NCORE`) | shared | yes | dense else-branch (lazy `E`) | D=4+D=20 PASS | no | no | no |
| H_EM (H_ER) | `winner_pair_cross_hessian_zc_block!` sourced from `hzz_centered.Zc` (**fixed this session**: was reading dense `H`) | shared with origin-ZC | yes | dense else-branch | D=4+D=20 PASS (K_pair=1 case exercises this exactly) | no | **no (fixed this session)** | no |
| H_MM (H_RR/H_ZZ) | `zc_restriction_gram!` | shared with origin-ZC | yes | dense `EM'*EM` else-branch | D=4+D=20 PASS | no | no | no |

## 5. ZC-only (origin-ZC)

| Col | Production backend | Shared fn | Default | Reference backend | D=20 gate | calls moments!? | reads/materializes G? | allocates G-sized storage? |
|---|---|---|---|---|---|---|---|---|
| Economic moment construction | `wrap_moments_with_originzc` (priming) | family-specific | yes | `:dense_reference` | D=4+D=20 PASS | **yes** (1/solve) | economic+mean/pair: yes (unconditional, no skip mechanism exists) | yes |
| Outer A-gradient | unchanged (out of scope) | — | — | — | — | — | — | — |
| Economic FG | `OriginZCOperatorState`'s `dual_index!` | `economic_forward!` (shared) | yes | dense gemv fallback | D=4+D=20 PASS | no | no | no |
| Restriction FG | `restriction_forward!` (ZC) | shared | yes | dense gemv fallback | D=4+D=20 PASS | no | no | no |
| H_EE | `fill_core_hessian_upper!` (via `archA_partitioned_hess_cb_builder`) | shared | yes | dense else-branch | D=4+D=20 PASS | no | no | no |
| H_ER | `winner_pair_cross_hessian_zc_block!` sourced from `hzz_centered.Zc` (**fixed this session**) | shared with CM+ZC | yes | dense else-branch | D=4+D=20 PASS | no | **no (fixed this session)** | no |
| H_RR | `zc_restriction_gram!` | shared with CM+ZC | yes | dense `HC_eta'*HC_eta` else-branch | D=4+D=20 PASS | no | no | no |

## Summary

Every production cell in columns 3–7 (economic FG, restriction FG, H_EE, H_ER, H_RR) is now
operator/native and dense-G-free for all 5 families, confirmed by direct trace and gated at D=4 and
real D=20/W=80,000. Column 1 (economic moment construction) is the one column NOT fully closed this
session: all 4 restricted families still call `moments!`/`select_G_from_H` once per inner solve to
materialize the dense economic block (and, for 3 of 4, the restriction blocks too) — an attempt to
remove this was made and reverted after a real regression (see master report §5). The matrix is
therefore **complete for its own stated columns 3-7** but **incomplete for column 1** across all 4
restricted families — `incomplete_economic_moment_construction_flexcm_frechet_cmzc_originzc`.
