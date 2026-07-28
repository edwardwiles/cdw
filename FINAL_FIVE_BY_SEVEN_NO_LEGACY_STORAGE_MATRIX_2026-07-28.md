# Final 5×7 no-legacy-storage matrix — 2026-07-28

**PARTIALLY SUPERSEDED — see `SESSION_MASTER_VERDICT_2026-07-28.md`.** Column 1 (economic moment
construction) below was written before this session's real fixes landed: flexible-CM/CM+ZC/
origin-ZC's priming-side economic-block dense fill is now genuinely *skipped* (not merely
"unconditional" as this doc still says for those three), root-caused and D=4-validated. Common-
Fréchet's own row is accurate as-is (investigated separately this session, left unresolved and
unchanged). The struct-level claims (every family still constructs `PsiObjectiveBundleImplicit`,
still has a legacy `H` field) remain accurate — only the *priming behavior* changed, not the type.

Columns: (1) economic moment construction, (2) outer A-gradient (out of scope, unchanged), (3)
economic FG, (4) restriction FG, (5) H_EE, (6) H_ER/H_EC (cross-block), (7) H_RR/H_CC (restriction
self-block). This session independently re-verified every claim against current source (not copied
from the prior session's matrix) via a dedicated read-only audit agent — see
`LEGACY_CC_H_G_CONSTRUCTOR_AND_CALLSITE_AUDIT_2026-07-28.md` for the full site list.

Columns 3-7 (Hessian-callback-side operator machinery) were already closed by the prior merged
session and remain closed — re-confirmed here by re-running the exact D=4 gate that exercises them
(`test_shared_core_hessian_d4_gates.jl`, 40/40 PASS) both before and after this session's
independent Hessian-upper-only cleanup (a different, additive change — see
`PRODUCTION_HESSIAN_UPPER_ONLY_RELEASE_2026-07-28.md`). Column 1 remains the sole open gap, in all
4 restricted families, unchanged this session (attempted fix reverted by the prior session; this
session designed but did not implement the correct architecture — see
`OPERATOR_DENSE_BUNDLE_TYPE_SEPARATION_2026-07-28.md`).

## 1. Unrestricted

| Col | Backend | Legacy H owned/read? | Composite G owned/read? |
|---|---|---|---|
| Economic moment construction | `build_economic_moment_state!`/`cf_build` direct, no `moments!` call at all | `obj.H` allocated (struct-mandated) but only 2-3 columns ever written under the default backend | no |
| Economic FG | `compressed_dual_index!`/`compressed_cc_value_grad!` (operator) | no | no |
| H_EE | `winner_pair_hessian!` via `operator_prep_for_hessian!` | no | no |
| H_ER / H_RR | n/a (no restriction block) | n/a | n/a |

## 2. Flexible CM

| Col | Backend | Legacy H owned/read? | Composite G owned/read? |
|---|---|---|---|
| Economic moment construction | `wrap_moments_with_cm_archB`, unconditional economic-block dense fill, once/inner solve | **yes** — full `W×(2+ncore+ncm)` `obj.H`, + 1 throwaway `W×(2+ncore+ncm)` per context build (`build_cm_augmented_obj`, new finding) | **yes** (priming side only) |
| Economic FG | `CMLookupState`'s `dual_index!`/`economic_forward!` (operator) | no | no |
| H_EE | `fill_core_hessian_upper!` via `operator_prep_for_hessian!` | no | no |
| H_EC | `winner_pair_cross_hessian_fill!` (winner-bin) | no | no |

## 3. Common Fréchet

| Col | Backend | Legacy H owned/read? | Composite G owned/read? |
|---|---|---|---|
| Economic moment construction | `wrap_moments_with_cm_frechet_archB`, unconditional economic+CM-grid+level fill, once/inner solve (no skip mechanism engaged in production) | **yes** — full `W×(2+ncore+ncm)` `obj.H`, + 1 throwaway per context build (`build_cm_frechet_level_augmented_obj`) | **yes** |
| Economic FG | `CMFrechetLookupState`'s `dual_index!` (operator) | no | no |
| H_EE | `fill_core_hessian_upper!` via `operator_prep_for_hessian!` | no | no |
| H_EC / H_E,level | winner-pair cross-Hessian (winner-bin) | no | no |

## 4. CM+ZC

| Col | Backend | Legacy H owned/read? | Composite G owned/read? |
|---|---|---|---|
| Economic moment construction | `wrap_moments_with_cm_meanzc`, unconditional full fill, once/inner solve, **no skip mechanism exists** | **yes** — full `W×(2+ncore_econ+n_mean+n_pair+ncm)` `obj.H` (no throwaway duplicate for this family) | **yes** |
| Economic FG | `CMMeanZCOperatorState`'s `dual_index!` (operator) | no | no |
| H_EE (widened) | `_fill_cm_HEE!`'s winner-pair branch | no | no |
| H_EM / H_MM | `winner_pair_cross_hessian_zc_block!` sourced from `hzz_centered.Zc` | no (fixed by prior session) | no |

## 5. ZC-only (origin-ZC)

| Col | Backend | Legacy H owned/read? | Composite G owned/read? |
|---|---|---|---|
| Economic moment construction | `wrap_moments_with_originzc`, unconditional full fill, once/inner solve, **no skip mechanism exists** | **yes** — full `W×(2+ncore_econ+n_mean+n_pair)` `obj.H` (no throwaway duplicate) | **yes** |
| Economic FG | `OriginZCOperatorState`'s `dual_index!` (operator) | no | no |
| H_EE | `fill_core_hessian_upper!` via `operator_prep_for_hessian!` | no | no |
| H_ER / H_RR | `winner_pair_cross_hessian_zc_block!` / `zc_restriction_gram!`, **this session's Hessian-upper-only cleanup also removed a provably dead mirror write here** | no | no |

## Bundle-wide properties

```
has_legacy_H_field       = unrestricted:yes, flexible_cm:yes, common_frechet:yes, cm_plus_zc:yes, zc_only:yes
has_K_field               = same (yes for all 5 -- shared struct, H's column 1)
has_moments_field         = unrestricted:yes-but-unused, flexible_cm:yes, common_frechet:yes, cm_plus_zc:yes, zc_only:yes
has_select_G_interface    = yes for all 5 (inherited from PsiObjectiveBundleImplicit's type, method exists generically)
G_sized_storage           = unrestricted:yes(mostly-dead), flexible_cm:yes(+throwaway), common_frechet:yes(+throwaway),
                             cm_plus_zc:yes, zc_only:yes
```

## Summary

Columns 3-7 are complete and dense-G-free for all 5 families, both before and after this session
(the prior merged session's work, re-verified). Column 1 is incomplete for all 4 restricted
families — unchanged status from the prior session, `PRODUCTION_MOMENTS_CALLS = 4`,
`PRODUCTION_SELECT_G_FROM_H_CALLS = 4` — plus a newly-identified (not previously named) additional
waste source: throwaway full-width `obj_cm` allocations in flexible-CM's and common-Fréchet's
context-build helpers (`LEGACY_CC_H_G_CONSTRUCTOR_AND_CALLSITE_AUDIT_2026-07-28.md` §4).

```
FIVE_FAMILY_ARCHITECTURE = incomplete_economic_moment_construction_flexcm_frechet_cmzc_originzc
```
