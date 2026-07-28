# Operator / dense-reference bundle type separation — design + root-cause analysis — 2026-07-28

## 1. Why a runtime flag on one shared struct is the wrong shape (evidence, not assumption)

The prior session's only live attempt at removing the priming-side economic-block dense fill
extended the existing `skip_fill::Bool` kwarg (already used for the CM-grid columns) to also gate
`materialize_dense_factual_structured!`. It was reverted after
`test_shared_core_hessian_d4_gates.jl` caught a real (`max|Δ|=0.0336`, explicitly noted as not
floating-point noise) H_EE mismatch between two contexts built for comparison
(`pcx_d` forced to `cctx.core_hessian_backend=:dense_reference`, `pcx_p` left at
`:exact_winner_pair_parallel`). A follow-up attempt to guard the skip decision on
`cctx.core_hessian_backend !== :dense_reference` **also** reproduced the same mismatch, even with
the economic-skip extension itself fully reverted — i.e., merely adding a new read of
`cctx.core_hessian_backend` to `archC_base_state`'s `skip_fill_safe` boolean (which, with the
extension gone, only ever changes whether the **unrelated** CM-grid columns get filled) was enough
to reproduce a divergence in **H_EE**, a block that both this codebase's own extensive Hessian-path
comments and this session's independent trace of `_fill_cm_HEE!` agree does not depend on the
CM-grid columns at all under the winner-pair backend.

This session traced the mechanism as far as: `_fill_cm_HEE!`
(`cm_hessian_architectures.jl:690-768`) dispatches on `cctx.core_hessian_backend`. When it is
`:dense_reference` (as `pcx_d` explicitly sets, by design, since it exists to be the dense ground
truth for the comparison), `_fill_cm_HEE!` takes its **else**-branch unconditionally, which lazily
constructs `E = @view H[:, 2:1+NCORE]` and reads it directly — i.e. `pcx_d`'s own H_EE computation
is, by construction, **required** to have a correctly-filled dense economic block; the winner-pair
backend (`pcx_p`) does not need it. `archC_base_state`'s `skip_fill_safe` gate
(`cm_production_bundle.jl:235`) is `use_lookup && MOMENT_REPRESENTATION[]==:operator &&
cctx.cm_cross_hessian_backend==:winner_bin` — note this reads `cm_cross_hessian_backend` (the
**cross**-block backend selector), not `core_hessian_backend` (the **self**-block/H_EE selector)
that the test actually varies between `pcx_d`/`pcx_p`. This session's working hypothesis, reached
by direct trace but **not yet confirmed by an instrumented empirical run** (time-boxed out of this
session, see below): the priming-side skip decision is keyed off the wrong backend flag entirely —
whether the economic block gets filled for a given context should be a function of *that context's
own* dense-vs-operator commitment (`core_hessian_backend`, or an explicit
`moment_representation` field on the context itself), never a global (`MOMENT_REPRESENTATION[]`) or
cross-block (`cm_cross_hessian_backend`) flag unrelated to what the *H_EE* computation is about to
need. A single shared closure toggled by an incomplete/mismatched flag is exactly the kind of bug
class a genuinely-separate-types architecture (this section's actual deliverable) makes structurally
impossible: if `:dense_reference` mode is a **different concrete type**, built by a **different
priming function** that unconditionally fills (byte-identical to today's already-validated
behavior), there is no shared closure, no shared flag, and no possibility of one context's
backend-selection accidentally leaking into another's priming decision.

**This hypothesis was not empirically verified this session** — doing so requires an instrumented
Julia run (print/trace `cctx.core_hessian_backend`, `skip_fill_safe`'s actual boolean, and
`core_ws`/`cf` object identity at the exact point of divergence for both `pcx_d` and `pcx_p`), which
this session judged not worth the risk of burning the remaining time budget on a second unresolved
attempt, given a full prior session already spent significant effort on this exact question without
closing it. It is recorded here as the single most concrete, actionable lead for whoever picks this
up next — a sharper starting point than the prior session's "suspected core_ws/cf-identity
interaction, not yet confirmed."

## 2. Target architecture

```julia
# Shared by every family's operator-mode inner solve. No H, no G, no K, no ones, no moments!,
# no select_G_from_H possible by construction (the type has no such field).
struct OperatorPsiObjectiveBundle{EconState, RestrictState, DualWS, HessWS, VerifyWS}
    economic_state       ::EconState        # e.g. CompressedFactual ref / core_cf_ref-equivalent
    restriction_state    ::RestrictState    # CM bins/ZC raw state, family-specific, immutable per solve
    dual_index_workspace ::DualWS           # r = -ζ - E'λ_E - R'λ_R scratch (operator_dual_index!)
    hessian_workspace     ::HessWS          # HessianWeightCache + per-family Hessian scratch
    verification_workspace ::VerifyWS       # operator-only verification scratch (§9 of the task)
    counters              ::OperatorBundleCounters
end

# Explicit opt-in only (moment_representation = :dense_reference). Retains the legacy container
# and legacy interface completely unchanged, for correctness comparison / non-production use only.
struct DenseReferencePsiObjectiveBundle
    legacy_cc_H ::Matrix{Float64}          # [K | ones | G], the ORIGINAL layout, byte-identical
    moments!    ::Function
    # ... every other PsiObjectiveBundleImplicit field the dense path still needs
end

select_G_from_H(::OperatorPsiObjectiveBundle, args...) =
    error("Dense G access is forbidden in operator mode")
# no method defined for (::OperatorPsiObjectiveBundle).moments! -- there is no such field, so
# `obj.moments!` is a MethodError/field-access error at the language level, not a runtime check.
```

Per family (task §5), `RestrictState`/`HessWS` specialize:

| Family | `restriction_state` | extra `hessian_workspace` beyond shared `HessianWeightCache` |
|---|---|---|
| Unrestricted | `nothing` (no restriction block) | none |
| Flexible-CM | CM bins + contrast (`R`) transforms | bin-contingency-table scratch (`Ttab`/`Stab`/`CT`/`CScum`) |
| Common-Fréchet | CM bins/contrasts + level-anchor targets | same + level-anchor scratch |
| CM+ZC | CM native state + ZC native/raw structured state | `H_CC`/`H_CZ`/`H_ZZ` scratch, shared `hzz_centered` |
| ZC-only | ZC native/raw structured state | direct `H_ZZ` scratch |

None of these own a concatenated `[E|C]`/`[E|Z]`/`[E|C|Z]` matrix — the task's own text already
matches what this session's audit confirmed is *already true on the Hessian-callback side* (the
prior session's Sections 1-4 work); the only genuinely outstanding piece is the **priming-side**
economic block, which under this design simply never gets built at all for
`OperatorPsiObjectiveBundle` — the operator priming function computes `cf` (or the family-native
equivalent: `CMMeanZCOperatorState`'s/`OriginZCOperatorState`'s own already-operator-based
`dual_index!` inputs) and publishes it, full stop, with no `Gtmp`/`materialize_dense_factual_structured!`
call in that code path at all — not a skipped branch, an absent one.

## 3. Why this was not implemented live this session

Implementing this for real requires either (a) a genuine new concrete type per family, threaded
through every KNITRO callback registration site (`KN_set_cb_*`) and every place `obj::Any`/
`obj::PsiObjectiveBundleImplicit` is currently type-annotated across
`cm_lookup_production.jl`/`cm_meanzc_lookup_production.jl`/`cm_frechet_lookup_production.jl`/
`cm_originzc_lookup_production.jl`/`cm_hessian_architectures.jl` (a wide, multi-file surface), or
(b) a narrower first cut that keeps `PsiObjectiveBundleImplicit` as the concrete type registered
with KNITRO (required for the `@with_kw`-generated field access and the existing generic functor
fallback to keep working for the `:dense_reference` opt-out) but genuinely never allocates/reads its
`H` field in operator mode — which is exactly the point the priming-side regression above blocks.
Given (a) a prior full session already spent real effort on the narrower (b) path and left it
unresolved with a real, reproduced numerical regression, and (b) this session's own time is split
with an independent addendum task (Hessian upper-only cleanup, see
`PRODUCTION_HESSIAN_UPPER_ONLY_AUDIT_2026-07-28.md`), attempting a live, unvalidated structural
change to 5 families' production KNITRO wiring was judged too high-risk to ship without the
empirical instrumentation pass described in §1 above — shipping a "looks right, gates weren't run
long enough to prove it" change to a scientific estimation codebase is exactly the failure mode
this project's own `CLAUDE.md`/memory record warns against repeatedly (see
`feedback-verify-before-causal-claims`, `feedback-gravity-elimination-zero-is-not-calibration`).

**What IS safe and was left as designed-but-not-yet-wired**: the throwaway `obj_cm` allocation in
`build_cm_augmented_obj`/`build_cm_frechet_level_augmented_obj`
(`LEGACY_CC_H_G_CONSTRUCTOR_AND_CALLSITE_AUDIT_2026-07-28.md` §4) is a genuinely dead allocation
with zero read-path risk, confirmed by exhaustive field trace — the correct, low-risk fix is to
split each function into a cheap bookkeeping half and an opt-in full-bundle half, with the
production call sites using only the cheap half. This is a real, bounded refactor (touches 2
production files plus ~30 non-production caller sites that would keep using the existing
"opt-in full bundle" code path unchanged) and is the top recommended next step.

## 4. Honest verdict for this section

```
OPERATOR_BUNDLE_TYPE =
    unrestricted:PsiObjectiveBundleImplicit (unchanged; H allocated but production-default backend
                 never writes its economic columns -- see callsite audit §1)
    flexible_cm:PsiObjectiveBundleImplicit (unchanged; economic block still unconditionally filled)
    common_frechet:PsiObjectiveBundleImplicit (unchanged; economic+CM-grid+level all unconditional)
    cm_plus_zc:PsiObjectiveBundleImplicit (unchanged; all blocks unconditional, no skip mechanism)
    zc_only:PsiObjectiveBundleImplicit (unchanged; all blocks unconditional, no skip mechanism)

FIVE_FAMILY_ARCHITECTURE = incomplete_economic_moment_construction_all_four_restricted_families
```

No family's production bundle was changed to a genuinely separate operator type this session. The
design above, and the sharper (but not empirically confirmed) root-cause lead in §1, are this
session's real contribution toward closing this gap; they are handed off, not shipped as fixed.
