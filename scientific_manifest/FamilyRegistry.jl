module FamilyRegistryMod
# FamilyRegistry -- task profiled-outer-production-readiness-2026-08-03 section 4: "one family
# registry keyed by (family, formulation) ... each entry must identify TRUTHFUL capabilities."
#
# Pure data + a canonicalization function. Deliberately has ZERO `include`/`using` of anything
# under full_aod_diag/d4_exact -- every fact recorded below was verified by reading the real
# source there (file/line references in comments, not asserted), but this module itself stays
# dependency-free so it can be `include`d by RunManifest-aware code without pulling in KNITRO,
# the D20 dataset, or any of the ~90-file include chain the real runner scripts need.
#
# Deliberately separate from Melitz (same discipline as ScientificManifest/RunManifest).
#
# LOAD-BEARING FINDING this file exists to fix: FULL's family-name convention
# (production_backend_manifest.jl's `resolve_*_manifest` functions, matches RunManifest's
# VALID_FAMILIES) and REDUCED's family-name convention (`family_kind(fctx)`,
# profiled_family_adapters_2026-08-01.jl / profiled_originzc_family_adapter_2026-08-02.jl /
# profiled_cmzc_family_adapter_2026-08-02.jl / profiled_restricted_family_adapters_2026-08-02.jl)
# are TWO DIFFERENT SYMBOL SETS for the same five families:
#
#   canonical (FULL)    REDUCED family_kind(fctx)
#   :unrestricted    <-> :unrestricted     (same)
#   :flexible_cm     <-> :flexible_CM      (case differs)
#   :common_frechet  <-> :common_Frechet   (case differs)
#   :origin_zc       <-> :ZC_only          (different name entirely)
#   :cm_meanzc       <-> :CM_plus_ZC       (different name entirely)
#
# Confirmed live 2026-08-03 by grepping every `family_kind(::SomeFamilyCtx) = :...` method
# definition and cross-checking against production_backend_manifest.jl's own family symbols and
# a real D4 gate's own assertions (test_phase13_production_runner_d4_gate_2026-08-02.jl:118,
# `manifest_oz.family == :ZC_only`). Nothing here renames the REDUCED code's own symbols --
# `family_kind(fctx)` keeps returning exactly what it always has. This module only adds the
# translation a caller needs before writing a canonical RunManifest/checkpoint `family` field
# from a REDUCED `fctx`, so the two conventions never get silently conflated (e.g. a checkpoint
# namespace built from one convention being compared against a manifest built from the other).

export CANONICAL_FAMILIES, REDUCED_FAMILY_KIND_TO_CANONICAL, CANONICAL_TO_REDUCED_FAMILY_KIND,
    canonical_family_of_reduced_kind, reduced_kind_of_canonical_family, FamilyCapability,
    FAMILY_CAPABILITY_REGISTRY, capability, production_ready_families

"The five real families, canonical (FULL-convention) spelling -- matches RunManifest.VALID_FAMILIES."
const CANONICAL_FAMILIES = (:unrestricted, :flexible_cm, :common_frechet, :origin_zc, :cm_meanzc)

"""
REDUCED's own `family_kind(fctx)` symbol -> canonical symbol. Source (file:line, this branch,
2026-08-03): `profiled_family_adapters_2026-08-01.jl:65` (`:unrestricted`),
`profiled_restricted_family_adapters_2026-08-02.jl:64` (`:flexible_CM`),
`profiled_restricted_family_adapters_2026-08-02.jl:156` (`:common_Frechet`),
`profiled_originzc_family_adapter_2026-08-02.jl:66` (`:ZC_only`),
`profiled_cmzc_family_adapter_2026-08-02.jl:78` (`:CM_plus_ZC`).
"""
const REDUCED_FAMILY_KIND_TO_CANONICAL = Dict(
    :unrestricted => :unrestricted,
    :flexible_CM => :flexible_cm,
    :common_Frechet => :common_frechet,
    :ZC_only => :origin_zc,
    :CM_plus_ZC => :cm_meanzc,
)

const CANONICAL_TO_REDUCED_FAMILY_KIND = Dict(v => k for (k, v) in REDUCED_FAMILY_KIND_TO_CANONICAL)

"""
    canonical_family_of_reduced_kind(kind::Symbol) -> Symbol

Throws (does not silently pass through) if `kind` is not one of REDUCED's five known
`family_kind(fctx)` values -- an unrecognized family_kind is a real bug to surface, not a value
to guess a translation for.
"""
function canonical_family_of_reduced_kind(kind::Symbol)
    haskey(REDUCED_FAMILY_KIND_TO_CANONICAL, kind) ||
        error("canonical_family_of_reduced_kind: unrecognized REDUCED family_kind=:$kind, expected one of $(Tuple(keys(REDUCED_FAMILY_KIND_TO_CANONICAL)))")
    return REDUCED_FAMILY_KIND_TO_CANONICAL[kind]
end

"""
    reduced_kind_of_canonical_family(family::Symbol) -> Symbol

Inverse of `canonical_family_of_reduced_kind`. Throws on an unrecognized canonical family.
"""
function reduced_kind_of_canonical_family(family::Symbol)
    haskey(CANONICAL_TO_REDUCED_FAMILY_KIND, family) ||
        error("reduced_kind_of_canonical_family: unrecognized canonical family=:$family, expected one of $CANONICAL_FAMILIES")
    return CANONICAL_TO_REDUCED_FAMILY_KIND[family]
end

"""
    FamilyCapability

One (family, formulation) cell of the registry. Every field is a real, verified fact about
today's codebase (2026-08-03, this branch) -- a `nothing`/`false`/`:not_wired` means "verified
absent", not "unknown"; see each field's inline doc for the verification source. Do not add a
field value without a corresponding read of the actual source file.
"""
struct FamilyCapability
    family::Symbol
    formulation::Symbol                 # :full_gamma_normalized | :profiled_destination_scales
    context_constructor::String         # function that builds ctx (both formulations share this)
    outer_evaluator::String             # the real production/prototype driver function name
    inner_verifier::Union{Nothing,String}
    outer_gradient::String
    checkpoint_resume::Bool
    checkpoint_function::Union{Nothing,String}
    free_nu_supported::Bool
    coordinate_modes::Vector{Symbol}
    threaded_outer_gradient::Bool
    bandwidth_cache::Bool
    production_ready::Bool
    notes::String
end

_key(family::Symbol, formulation::Symbol) = (family, formulation)

const _FULL = :full_gamma_normalized
const _RED = :profiled_destination_scales

"""
FAMILY_CAPABILITY_REGISTRY -- keyed by `(family::Symbol, formulation::Symbol)`.

FULL rows verified against `full_aod_diag/d4_exact/production_backend_manifest.jl`,
`c10_d20_production_driver_unified.jl`, `cm_checkpoint.jl` (`run_cm_upper_checkpointed`),
`cm_originzc_checkpoint.jl` (`run_originzc_upper_checkpointed`) -- all three real production
entry points confirmed to already implement `resume_from` (grep for the literal kwarg in each
file, 2026-08-03). `production_ready=true` for all five FULL rows reflects the live W=100k 10x10
production campaign already running under these exact drivers as of this session (per
[[full-vs-reduced-forensic-audit-2026-08-03]]), not a fresh judgment made here.

REDUCED rows verified against `profiled_production_outer_constrained_2026-08-02.jl`
(`run_profiled_upper_constrained`, confirmed by direct read to have NO checkpoint_path/
resume_from parameter as of this branch's HEAD before this session's own commit adding it -- see
that commit's message for exactly what changed), `profiled_zc_lane_point_evaluators_2026-08-02.jl`
(evaluators), `reduced_operator_verification_2026-08-01.jl` (verifier). `free_nu_supported=false`
for every REDUCED row: as of 2026-08-03, confirmed that no `CMZCFreeNuAdapter` or equivalent
existed anywhere in the tree. UPDATE 2026-08-04: `profiled_zc_free_eta_2026-08-04.jl` now provides
a genuine free-eta_nu evaluator + analytic gradient for origin_zc/cm_meanzc (D4-verified, ~1e-10 vs
fixed-dual FD) -- `free_nu_supported` is DELIBERATELY still `false` in both rows, because this field
describes what the production driver (`run_profiled_upper_constrained`) supports, and that
evaluator is not yet wired into it. See each row's own notes and
docs/audits/profiled-functional-readiness-closeout-2026-08-03/CONTINUATION_2026-08-04.md for the
real implementation/verification detail.
`coordinate_modes=[:profiled_pivot_anchor_relative]` for REDUCED rows: the mode REDUCED's
`gravity_pivot_on_retained.jl`/`relative_a_coordinate_2026-07-31.jl` machinery actually
implements has no formal Symbol anywhere in the codebase (zero `A_coordinate_mode=` grep hits in
any REDUCED runner script) -- `:profiled_pivot_anchor_relative` is a NEW name introduced by this
registry to give that already-real coordinate system a citable symbol, not a code change; no
runner sets `A_coordinate_mode` to this value yet (task section 7 work, not done in this
commit). `:profiled_powered_relative_A` is NOT listed for any REDUCED row -- not implemented
(task section 7, not started).
"""
const FAMILY_CAPABILITY_REGISTRY = Dict(
    _key(:unrestricted, _FULL) => FamilyCapability(:unrestricted, _FULL,
        "d20_real_setup_design", "run_polish_checkpointed_unified", "run_polish_checkpointed_unified's own cb_F! (c10_d20_production_driver_unified.jl)",
        "analytic (unrestricted core gradient engine)", true, "load_checkpoint_unified/save_checkpoint_unified",
        false, [:legacy_z, :powered_aspace, :z_space, :theta_decoupled_aspace], true, true, true,
        "Real production entry point; live W=100k 10x10 campaign uses this driver."),
    _key(:flexible_cm, _FULL) => FamilyCapability(:flexible_cm, _FULL,
        "d20_real_setup_design", "run_cm_upper_checkpointed", "cm_production_value_verified",
        "analytic (shared CM outer gradient engine)", true, "load_cm_checkpoint/save_cm_checkpoint (schema 9)",
        false, [:legacy_z, :powered_aspace], true, true, true,
        "Real production entry point."),
    _key(:common_frechet, _FULL) => FamilyCapability(:common_frechet, _FULL,
        "d20_real_setup_design", "run_cm_upper_checkpointed", "cm_production_value_verified",
        "analytic (shared CM outer gradient engine + Frechet level block)", true, "load_cm_checkpoint/save_cm_checkpoint (schema 9)",
        false, [:legacy_z, :powered_aspace], true, true, true,
        "Same driver/checkpoint infra as flexible_cm, marginal_restriction=:common_flexible."),
    _key(:origin_zc, _FULL) => FamilyCapability(:origin_zc, _FULL,
        "d20_real_setup_design", "run_originzc_upper_checkpointed", "cm_production_value_verified",
        "analytic (shared CM outer gradient engine)", true, "load_cm_checkpoint_v10/save (schema 10, cm_originzc_checkpoint.jl)",
        true, [:legacy_z, :powered_aspace], true, true, true,
        "Real production entry point; only FULL family with confirmed free-nu today (eta_nu searched by KNITRO)."),
    _key(:cm_meanzc, _FULL) => FamilyCapability(:cm_meanzc, _FULL,
        "d20_real_setup_design", "run_cm_upper_checkpointed", "cm_production_value_verified",
        "analytic (shared CM outer gradient engine)", true, "load_cm_checkpoint/save_cm_checkpoint (schema 9)",
        true, [:legacy_z, :powered_aspace], true, true, true,
        "cm_extension=:cm_plus_moments variant of run_cm_upper_checkpointed; free-nu confirmed via same mechanism as origin_zc."),

    _key(:unrestricted, _RED) => FamilyCapability(:unrestricted, _RED,
        "d20_real_setup_design", "run_profiled_upper_constrained", "reduced verify_fn (inner_status-only classification unless caller supplies verify_fn)",
        "shared_family_outer_gradient", true, "_write_profiled_constrained_checkpoint/load_cm_checkpoint_v11 (via run_profiled_upper_constrained)",
        false, [:profiled_pivot_anchor_relative], false, false, false,
        "family_kind(fctx)=:unrestricted (same spelling as canonical); UnrestrictedFamilyCtx exists (profiled_family_adapters_2026-08-01.jl). checkpoint_resume verified 2026-08-03 (test_all_family_checkpoint_resume_2026-08-03.jl, 52/52 checks PASS across all 5 REDUCED families incl. this one) AND via the canonical CLI runner's own real D20/W=20,000 smoke (bin/run_profiled_model.jl, prior session): real checkpoint.jls + run_manifest.json written, feasible incumbent found. W=100k warm-start/fast-reject, free-nu (n/a for this family), and D20+ outer-gradient gates still missing -- production_ready stays false."),
    _key(:flexible_cm, _RED) => FamilyCapability(:flexible_cm, _RED,
        "d20_real_setup_design", "run_profiled_upper_constrained", "reduced verify_fn",
        "shared_family_outer_gradient", true, "_write_profiled_constrained_checkpoint/load_cm_checkpoint_v11 (via run_profiled_upper_constrained)",
        false, [:profiled_pivot_anchor_relative], false, false, false,
        "REDUCED family_kind(fctx)=:flexible_CM (case differs from canonical :flexible_cm -- see REDUCED_FAMILY_KIND_TO_CANONICAL). Exercised by run_outer_flexcm_reduced_constrained_2026-08-02.jl. checkpoint_resume verified 2026-08-03 (test_all_family_checkpoint_resume_2026-08-03.jl, 52/52 checks) AND via a real D20/W=20,000 canonical CLI smoke (2026-08-04): status=-401 (time-limit, expected), best gp=0.9767958864585219 Delta=0.05497330, checkpoint.jls+run_manifest.json written. W=100,000 warm-start CONFIRMED (2026-08-04): exact status+Delta-star match cold vs warm. Fast-rejection: eval18 captured point is fast (nStatus=-300, 12.72s) at W=20,000 but needs a wider maxit at W=100,000 (a real W-dependence finding, not a gap -- see docs/audits/profiled-functional-readiness-closeout-2026-08-03/CONTINUATION_2026-08-04.md). D20+ outer-gradient gate still missing -- production_ready stays false."),
    _key(:common_frechet, _RED) => FamilyCapability(:common_frechet, _RED,
        "d20_real_setup_design", "run_profiled_upper_constrained (generic over fctx/evaluate_fn; no dedicated common_frechet reduced_constrained script found this session)",
        "reduced verify_fn", "shared_family_outer_gradient", true, "_write_profiled_constrained_checkpoint/load_cm_checkpoint_v11 (via run_profiled_upper_constrained)",
        false, [:profiled_pivot_anchor_relative], false, false, false,
        "FrechetFamilyCtx exists (profiled_restricted_family_adapters_2026-08-02.jl); no dedicated run_outer_*_reduced_constrained script for this family found -- run_frechet_outer_control_frechet.jl exists but was not read this session, may be a different (older/unconstrained) driver. Verify before use. checkpoint_resume verified 2026-08-03 (test_all_family_checkpoint_resume_2026-08-03.jl, 52/52 checks) AND via a real D20/W=20,000 canonical CLI smoke (2026-08-04): status=-401, best gp=0.9746430740140182 Delta=0.10990155. Threaded HESSIAN parity (NOT the same field as threaded_outer_gradient -- left false, no direct evidence for that specific claim) confirmed at real D20/W=20,000 (2026-08-04): threaded==serial to ~1e-14, 3/3 fixed states. W=100k warm-start/fast-reject + D20+ outer-gradient gate still missing -- production_ready stays false."),
    _key(:origin_zc, _RED) => FamilyCapability(:origin_zc, _RED,
        "d20_real_setup_design", "run_profiled_upper_constrained", "reduced verify_fn",
        "shared_family_outer_gradient", true, "_write_profiled_constrained_checkpoint/load_cm_checkpoint_v11 (via run_profiled_upper_constrained, added task2-section5-2026-08-03)",
        false, [:profiled_pivot_anchor_relative], false, false, false,
        "REDUCED family_kind(fctx)=:ZC_only. Exercised by run_outer_originzc_reduced_constrained_2026-08-02.jl. checkpoint_resume verified live 2026-08-03 with a real D4 KNITRO round trip (18/18 checks) AND via a real D20/W=20,000 canonical CLI smoke (2026-08-04): status=-401, best gp=0.9652292135031784 Delta=0.73818994. W=100,000 warm-start CONFIRMED (2026-08-04): exact status+Delta-star match cold vs warm. W=100,000 fast-rejection CONFIRMED (2026-08-04): a systematically-constructed infeasible point (additive log-A shift from calibration) rejects in ~14s (nStatus=-300) at real W=100,000, after being classified at cheaper W=20,000 first (0.40s there). Free eta_nu: `evaluate_profiled_originzc_point(w_econ, eta_nu, fctx, pes)` (4-arg, explicit eta_nu, profiled_zc_free_eta_2026-08-04.jl) + `reduced_originzc_outer_gradient_with_eta` IMPLEMENTED and D4-VERIFIED (~1e-10 vs fixed-dual FD, calibration+perturbed points) 2026-08-04 -- `free_nu_supported` left false here because this evaluator is NOT yet wired into `run_profiled_upper_constrained`/the canonical CLI runner (the field describes what the PRODUCTION driver supports, not what an evaluator function alone supports). D20/W=20k+ eta gates and the eta-generation cache/checkpoint wiring (task §8.4) remain open -- production_ready stays false."),
    _key(:cm_meanzc, _RED) => FamilyCapability(:cm_meanzc, _RED,
        "d20_real_setup_design", "run_profiled_upper_constrained (generic; no dedicated cm_meanzc reduced_constrained script found this session)",
        "reduced verify_fn", "shared_family_outer_gradient", true, "_write_profiled_constrained_checkpoint/load_cm_checkpoint_v11 (via run_profiled_upper_constrained)",
        false, [:profiled_pivot_anchor_relative], false, false, false,
        "REDUCED family_kind(fctx)=:CM_plus_ZC. CMZCFamilyCtx/evaluate_profiled_cmzc_point exist. checkpoint_resume verified 2026-08-03 (test_all_family_checkpoint_resume_2026-08-03.jl, 52/52 checks) AND via a real D20/W=20,000 canonical CLI smoke (2026-08-04): status=-401, best gp=0.9781193894488888 Delta=0.03772924. W=100,000 warm-start CONFIRMED (2026-08-04): exact status+Delta-star match cold vs warm. W=100,000 fast-rejection CONFIRMED (2026-08-04): same construction/methodology as origin_zc's row, ~14s to nStatus=-300 at real W=100,000. Free eta_nu: `evaluate_profiled_cmzc_point(w_econ, eta_nu, fctx, pes)` (4-arg, profiled_zc_free_eta_2026-08-04.jl) + `reduced_cmzc_outer_gradient_with_eta` IMPLEMENTED and D4-VERIFIED (~1e-10 vs fixed-dual FD) 2026-08-04 -- `free_nu_supported` left false, same reasoning as origin_zc's row (not yet wired into the production driver). D20/W=20k+ eta gates and eta-generation cache/checkpoint wiring remain open -- production_ready stays false."),
)

"""
    capability(family::Symbol, formulation::Symbol) -> FamilyCapability

Throws if `(family, formulation)` is not a registered cell.
"""
function capability(family::Symbol, formulation::Symbol)
    haskey(FAMILY_CAPABILITY_REGISTRY, (family, formulation)) ||
        error("capability: no registry entry for (family=:$family, formulation=:$formulation)")
    return FAMILY_CAPABILITY_REGISTRY[(family, formulation)]
end

"""
    production_ready_families(formulation::Symbol) -> Vector{Symbol}

Families marked `production_ready=true` for the given formulation, per the registry above.
"""
function production_ready_families(formulation::Symbol)
    return sort([f for f in CANONICAL_FAMILIES if capability(f, formulation).production_ready])
end

end # module FamilyRegistryMod
