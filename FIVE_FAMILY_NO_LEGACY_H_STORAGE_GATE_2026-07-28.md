# Five-family no-legacy-H-storage gate — 2026-07-28

**PARTIALLY SUPERSEDED — see `SESSION_MASTER_VERDICT_2026-07-28.md` for the accurate final
state.** This document was written before this session's real, root-caused, D=4-validated fixes
(flexible-CM/CM+ZC/origin-ZC's priming-side economic-block skip, and the `OperatorPsiBundle` type)
landed. The specific numeric gate below (`LEGACY_CC_H_MATRIX_ALLOCATIONS = 0`, an actual field/
allocation removal) is still **not met** — that part of this document remains accurate — but the
framing that nothing changed this session is now wrong; real, validated progress was made toward
it (see the master verdict for what).

This is the gate the task requires to reach `LEGACY_CC_H_MATRIX_ALLOCATIONS = 0` /
`G_SIZED_BACKING_STORAGE_ALLOCATIONS = 0`. **Not met this session** — reported honestly below
rather than claimed passing.

## Gate requirement

```
LEGACY_CC_H_MATRIX_ALLOCATIONS = 0
G_SIZED_BACKING_STORAGE_ALLOCATIONS = 0
CC_PAYOFF_K_VECTOR_ALLOCATIONS = 0
ONES_VECTOR_ALLOCATIONS = 0
PRODUCTION_MOMENTS_CALLS = 0
PRODUCTION_SELECT_G_FROM_H_CALLS = 0
COMPOSITE_G_MATERIALIZATIONS = 0
OPERATOR_BUNDLES_WITH_H_FIELD = none
OPERATOR_BUNDLES_WITH_K_FIELD = none
OPERATOR_BUNDLES_WITH_MOMENTS_FIELD = none
```

## Actual state this session (unchanged from the inherited branch state — see
`LEGACY_CC_H_G_CONSTRUCTOR_AND_CALLSITE_AUDIT_2026-07-28.md` for full detail)

```
LEGACY_CC_H_MATRIX_ALLOCATIONS       = 5 (1 per family, all still PsiObjectiveBundleImplicit)
                                        + 2 throwaway (flexible-CM/common-Fréchet, new finding)
G_SIZED_BACKING_STORAGE_ALLOCATIONS  = same as above
CC_PAYOFF_K_VECTOR_ALLOCATIONS       = 0 (K is a required O(W) scalar fill, not a removable
                                        artifact -- classified COUNTERFACTUAL_EQUILIBRIUM_MOMENT,
                                        see audit §3; this specific sub-count IS already at target)
ONES_VECTOR_ALLOCATIONS              = 1 per PsiObjectiveBundleImplicit construction (struct
                                        default `ones(M)`), not yet 0
PRODUCTION_MOMENTS_CALLS             = 4 (flexible-CM, common-Fréchet, CM+ZC, origin-ZC; each
                                        once per inner solve; unrestricted needs none)
PRODUCTION_SELECT_G_FROM_H_CALLS     = 4, paired 1:1
COMPOSITE_G_MATERIALIZATIONS         = 0 on the Hessian-callback side (already true, inherited
                                        from the prior session's merged work); >0 on the priming
                                        side (the blocker)
OPERATOR_BUNDLES_WITH_H_FIELD        = all 5 (no genuinely separate OperatorPsiObjectiveBundle
                                        type exists yet -- see design doc)
OPERATOR_BUNDLES_WITH_K_FIELD        = all 5 (same struct, same reason)
OPERATOR_BUNDLES_WITH_MOMENTS_FIELD  = all 5 (same struct, same reason)
```

## Why this session did not close the gap

The single blocking piece — removing the priming-side dense economic-block fill for the 4
restricted families — was attempted by the prior session and reverted after a real, reproduced
numerical regression that neither session fully root-caused (see
`OPERATOR_DENSE_BUNDLE_TYPE_SEPARATION_2026-07-28.md` §1 for this session's sharper, but still
unconfirmed, lead on the mechanism). Given a full prior session's effort on this exact question
went unresolved, this session judged a second unvalidated live attempt too risky for a scientific
production codebase and instead: (a) re-verified the audit numbers independently against current
source rather than trusting the prior report, (b) found one new genuinely-safe-but-unimplemented
waste site (§4 of the audit doc, the throwaway `obj_cm` allocations), and (c) designed the correct
target architecture (genuinely separate operator/dense-reference types) with an explanation of
*why* the simpler "runtime flag on a shared struct" approach both sessions tried is structurally
unsound, not just unlucky.

## HIGHEST_PRIORITY_REMAINING_GAP

Root-cause the H_EE priming-fill regression empirically (instrumented Julia run comparing
`cctx.core_hessian_backend`, `skip_fill_safe`'s actual boolean, and `core_ws`/`cf` identity for
`pcx_d` vs `pcx_p` at the exact point of divergence — see design doc §1 for the precise starting
hypothesis), OR bypass it entirely by implementing genuinely separate priming functions per
`moment_representation` (no shared closure, no shared flag) rather than attempting another runtime
toggle on the existing shared closure.
