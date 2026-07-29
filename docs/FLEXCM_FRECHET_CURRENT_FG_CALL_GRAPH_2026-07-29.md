# Flexible-CM / Common-Fréchet inner-FG call graph (as of production/fullA-exact @ 8aae735)

Audited 2026-07-29, before any harmonization edits in this task. Traced from the real production
entry point (`campaign_cm_family_runner.jl`) through context construction, dual layout, forward
operator, FG evaluation, transpose/gradient, and verification, for both `flexible_cm` and
`common_frechet`.

## Traced path

`campaign_cm_family_runner.jl` → `cm_checkpoint.jl::run_cm_upper_checkpointed` (family dispatch via
`is_frechet`) → `build_cm_production_context` / `build_cm_frechet_production_context` →
`archC_base_state` / `archC_frechet_base_state` → `inner_loop_internal_cmlookup_production` /
`inner_loop_internal_cmfrechetlookup_production` → `CMLookupState` / `CMFrechetLookupState` →
`economic_forward!` / `economic_transpose!` → `hessian_cm_structured!` →
`_verify_inner_solution_operator_cm_core` → `economic_A_gradient!`.

## Function table

| function | file:line | flexcm? | frechet? | block | classification |
|---|---|---|---|---|---|
| `run_cm_upper_checkpointed` (dispatch on `is_frechet`) | `cm_checkpoint.jl:591`, `900-909`, `1136-1149` | yes | yes | scalar/dispatch | shared-wrapper |
| `build_cm_production_context` | `cm_production_bundle.jl:65` | yes | no | context build | specialized (flexcm entry) |
| `build_cm_frechet_production_context` | `cm_frechet_level.jl:306` | no | yes | context build | specialized (frechet entry, near-parallel structure) |
| `build_cm_bin_ctx` / `CMBinHessCtx` | `cm_hessian_architectures.jl:623,426` | yes | yes | packing/layout | **shared** |
| `compute_bin_indices`, `fill_cm_columns_from_bins!` | `cm_hessian_architectures.jl:104,140` | yes | yes | C (bins) | shared |
| `economic_forward!` | `economic_operator.jl:62` | yes | yes | E | **shared** (literal) |
| `economic_transpose!` | `economic_operator.jl:81` | yes | yes | E | **shared** (literal) |
| `economic_operator_workspace` | `economic_operator.jl:49` | yes | yes | E | shared |
| `apply_contrast!` | `cm_lookup_kernels.jl:203` | yes | yes | C | **shared** |
| `suffix_sums!` | `cm_lookup_kernels.jl:213` | yes | yes | C | **shared** |
| `cumulative_forward_contribution!` | `cm_lookup_kernels.jl:164` | yes | yes | C | **shared** |
| `build_weighted_histogram!` | `cm_lookup_kernels.jl:235` | yes | yes | C | **shared** |
| `prefix_sums!` | `cm_lookup_kernels.jl:261` | yes | yes | C | **shared** |
| `cumulative_backward_gradient_from_prefix!` | `cm_lookup_kernels.jl:300` | yes | yes | C | **shared** (moved here 2026-07-28 to de-duplicate) |
| `CMLookupState` (struct + ctor) | `cm_lookup_kernels.jl:323,374` (pre-refactor) | yes | no | packing (E+C state) | pre-refactor: near-duplicate of `CMFrechetLookupState` |
| `CMFrechetLookupState{O}` (struct + ctor) | `cm_frechet_lookup_kernels.jl:107,157` | no | yes | packing (E+C+F state) | genuinely-extra level fields on top of the identical E+C field set |
| `dual_index!(::CMLookupState, x)` | `cm_lookup_kernels.jl:401` (pre-refactor) | yes | no | E+C forward | pre-refactor: near-duplicate of Fréchet's own |
| `dual_index!(::CMFrechetLookupState, x)` | `cm_frechet_lookup_kernels.jl:187` (pre-refactor) | no | yes | E+C+F forward | pre-refactor: near-duplicate of flexCM's own |
| `(st::CMLookupState)(x,g)` FG functor | `cm_lookup_kernels.jl:445` (pre-refactor) | yes | no | objective+gradient | pre-refactor: near-duplicate pair |
| `(st::CMFrechetLookupState)(x,g)` FG functor | `cm_frechet_lookup_kernels.jl:235` (pre-refactor) | no | yes | objective+gradient | pre-refactor: near-duplicate pair |
| `inner_loop_KNITRO_cmlookup_production` | `cm_lookup_production.jl:55` | yes | no | KNITRO wiring | duplicate structure (KNITRO plumbing, not hot-path math) |
| `inner_loop_KNITRO_cmfrechetlookup_production` | `cm_frechet_lookup_production.jl:12` | no | yes | KNITRO wiring | duplicate of above, own docstring calls it "analogue" |
| `inner_loop_internal_cmlookup_production` | `cm_lookup_production.jl:111` | yes | no | priming + solve orchestration | near-duplicate of below (KNITRO plumbing) |
| `inner_loop_internal_cmfrechetlookup_production` | `cm_frechet_lookup_production.jl:92` | no | yes | priming + solve orchestration | near-duplicate, differs only in `level_targets` arg + state type |
| `_adapt_hess_cb_for_lookup` | `cm_lookup_production.jl:43` | yes | yes | Hessian/FG adapter | **shared** (literal) |
| `OperatorPsiBundle` / `prime_operator!` | `operator_psi_bundle.jl:63,144` | yes (default) | opt-in only | E priming, no-H bundle | shared type/function, asymmetric DEFAULT usage (see gap below) |
| `archC_base_state` / `archC_frechet_base_state` | `cm_production_bundle.jl:219` / `cm_frechet_cplus.jl:111` | yes / no | no / yes | orchestration | specialized, near-parallel |
| `archC_verified_state` / `archC_frechet_verified_state` | `cm_production_bundle.jl:305` / `cm_frechet_cplus.jl:233` | yes / no | no / yes | verification orchestration | specialized, near-parallel |
| `verify_inner_solution_operator_cm!` / `_cm_frechet!` | `operator_verification.jl:160,178` | yes / no | no / yes | verification | shared-wrapper → shared core |
| `_verify_inner_solution_operator_cm_core` | `operator_verification.jl:195` | yes | yes | verification | **shared** (one impl, `level_targets::Union{Nothing,Vector}` selector) |
| `hessian_cm_structured!` / `_v2!` | `cm_hessian_architectures.jl:1063` / `cm_hessian_threaded.jl:175` | yes | yes | Hessian | **shared** (`extension::Any=nothing` selector) |
| `hessian_cm_frechet_structured!` / `_v2!` | `cm_frechet_hessian.jl:278` / `cm_frechet_hessian_threaded.jl:38` | no | yes | Hessian | thin wrapper → shared |
| `fill_cm_HCC!`, `pack_upper_cm_hessian!` | `cm_hessian_architectures.jl:1030,1011` | yes | yes | H_CC / packing | **shared** |
| `CMFrechetExtension`, `_fill_frechet_level_blocks!` | `cm_frechet_hessian.jl:48,72` | no | yes | Hessian level blocks | genuinely Fréchet-specific |
| `economic_A_gradient!` | `shared_a_gradient.jl:445` | yes | yes | outer (g,A_od) gradient | **shared** |
| `composite_gradient_at_Cplus_from_cache` | `lfix_cm_cplus.jl:77` | yes | yes | outer gradient coordinate loop | **shared** |
| `cm_fixed_contribution` | `lfix_cm_aware.jl:83` (pre-refactor) | yes | no | E-block fixed-λ value | pre-refactor: math duplicated inline by Fréchet's own |
| `frechet_cm_level_fixed_contribution` | `cm_frechet_cplus.jl:64` (pre-refactor) | no | yes | E-block fixed-λ + level | pre-refactor: inlined a copy of the CM part (own docstring admits this) |
| `frechet_level_forward_sum!`, `frechet_level_suffix_sums!`, `frechet_level_backward_gradient!` | `cm_frechet_cplus.jl:36`, `cm_frechet_lookup_kernels.jl:48,71` | no | yes | F forward/backward | genuinely Fréchet-specific |

## Harmonization status found (git log, pre-existing)

```
4fbd720 Harmonization step 1: extract shared fill_cm_HCC! (H_CC block)
a2a867a Harmonization step 2: common Fréchet uses the shared pack_upper_cm_hessian!
19adac1 Harmonization step 3: CMFrechetExtension with persistent level-block scratch
d9db756 Harmonization steps 4-5: common Frechet's Hessian fill is now the shared hessian_cm_structured!
680b0f9 Harmonization step 6: shared cumulative_backward_gradient_from_prefix! (CM transpose)
13d3aca Harmonization step 7: shared _verify_inner_solution_operator_cm_core (CM verification)
```

Steps 1-5 (Hessian) and step 7 (verification) were complete. Step 6 only moved one small backward
kernel; the FG-callback layer itself (state structs, `dual_index!`, the FG functor, KNITRO wiring)
had NOT been harmonized to the same standard — exactly the gap this task closes for the FG core
(state-struct/`dual_index!`/FG-functor level; see the duplication audit and target-block-contract
docs for what was harmonized in this task vs. deliberately left as parallel structure).

## Known pre-existing asymmetry, NOT fixed in this task (flagged as remaining gap)

`build_cm_production_context`'s `moment_representation` kwarg defaults to the mutable global
`MOMENT_REPRESENTATION[]` (currently `:operator`, the true no-H `OperatorPsiBundle` path).
`build_cm_frechet_production_context`'s same-named kwarg is hardcoded to `:dense_reference` and does
not consult that global — common_frechet's `archC_frechet_base_state` unconditionally materializes
the dense CM/level `G` columns once per outer point (`skip_fill_safe_frechet = false # ALWAYS
false`, `cm_frechet_cplus.jl:142`), per that file's own comment because two real, reproduced
`nStatus=-400` failures were found when Fréchet skipped the dense fill and the root cause was never
identified. This is a genuine, currently-necessary asymmetry (not lazy duplication) that means
common_frechet has not adopted the true no-H operator architecture flexible_cm uses by default. It
is out of scope for this FG-callback-duplication task (root-causing the `-400` failures is separate,
unsolved work) and is called out as the highest-priority remaining gap in the master report.
