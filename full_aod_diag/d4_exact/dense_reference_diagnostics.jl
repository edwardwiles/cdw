# ================================================================================================
# architecture/production-operator-bundle-hardening-2026-07-30, task §2/§6/§14.
#
# The ONLY sanctioned way to construct a dense-capable (PsiObjectiveBundleImplicit /
# DenseReferencePsiObjectiveBundle) bundle for anything downstream of this task's own production
# API. Diagnostic/test-only by construction: `prepare_context` requires a `DenseReferencePermit`,
# prints the mandatory banner, records the construction (stack trace + counter), and -- the one
# hard rule this whole task exists to enforce -- refuses outright if the caller's declared purpose
# is production.
#
# This module does NOT reimplement dense bundle construction. It calls the SAME family builder
# functions production code calls (build_cm_production_context, etc.), passing
# moment_representation=:dense_reference explicitly -- the builders themselves are unchanged and
# still support both values (equivalence tests still call them directly with either value, byte-
# identically to before this task; see task §11/§13 -- direct builder calls remain an allowlisted
# diagnostic/test pattern, this module is the NEW preferred path but does not obsolete the old one).
# ================================================================================================

isdefined(Main, :ProductionContext) || include(joinpath(@__DIR__, "production_bundle_api.jl"))

module DenseReferenceDiagnostics

# `Main` is always visible as a global binding from any module -- no import needed. This module is
# `include`d at Main's own top level, so `Main.DenseReferencePermit` etc. (production_bundle_api.jl,
# included just above) resolve correctly.

"""
    prepare_context(family::Symbol, runner::String, build_inner::Function;
                     permit::Main.DenseReferencePermit,
                     purpose::Main.RunPurpose = Main.DenseReferencePurpose(permit)) -> Main.DenseReferenceContext

`build_inner` is a zero-argument closure, supplied by the caller, that constructs the family's
dense-capable context -- e.g. `() -> build_cm_production_context(ctx, CS; L=10, contrasts=:anchored,
probs=probs, moment_representation=:dense_reference)`. This function does not itself pick which
family builder to call (family construction is genuinely family-specific, task §2's own
"family-specific code may construct economic/CM/Frechet/ZC state" carve-out) -- it validates the
result, prints the mandatory banner, and enforces the permit/purpose rule.

Throws immediately, before calling `build_inner`, if `purpose isa Main.ProductionPurpose` --
production runners must not (and, per production_bundle_api.jl, cannot -- prepare_production_run
never routes here) reach this function with a production purpose. A caller passing
`purpose=Main.ProductionPurpose()` here is themselves the bug being guarded against, not a
legitimate "opt out of the banner" path.
"""
function prepare_context(family::Symbol, runner::String, build_inner::Function;
                          permit::Main.DenseReferencePermit,
                          purpose::Main.RunPurpose = Main.DenseReferencePurpose(permit))
    purpose isa Main.ProductionPurpose &&
        error("DenseReferenceDiagnostics.prepare_context($family, $runner): called with " *
              "purpose=ProductionPurpose() -- dense reference construction under a production " *
              "purpose is always fatal, by design (task §6). This function is diagnostic/test-only.")

    inner = build_inner()
    obj = Main._resolve_bundle(inner)

    _announce_dense_construction!(family, runner, permit, obj)

    return Main.DenseReferenceContext(permit, family, runner, inner, obj)
end

"""
    construct_dense_reference_bundle_guard(bundle_type, permit, family, runner; purpose)

Lower-level hook usable directly from inside a family builder (e.g. immediately before it
constructs `PsiObjectiveBundleImplicit`) when a caller wants the banner/counter/permit-check
without going through the full `prepare_context` wrapper. `prepare_context` above calls
`_announce_dense_construction!` internally; this is exposed separately for callers that need the
banner emitted before construction rather than after (e.g. a builder that wants to fail fast on a
missing permit before doing any work).
"""
function require_permit(permit::Union{Nothing,Main.DenseReferencePermit}, caller::String)
    permit === nothing &&
        error("require_permit($caller): dense-reference construction requires a DenseReferencePermit " *
              "(reason=..., caller=...) -- construction without a permit is refused, by design (task §14).")
    return permit
end

function _announce_dense_construction!(family::Symbol, runner::String, permit::Main.DenseReferencePermit, obj)
    Main.DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS[] += 1
    bt = sprint(Base.show_backtrace, backtrace())
    push!(Main.DENSE_REFERENCE_CONSTRUCTION_LOG,
          (reason = permit.reason, caller = permit.caller, family = family, runner = runner,
           bundle_type = string(typeof(obj)), at = string(Main.now()), backtrace = bt))
    banner = string(
        "\n", "="^80, "\n",
        "ATTENTION: DENSE REFERENCE BUNDLE CONSTRUCTED\n",
        "This representation is diagnostic/test-only.\n",
        "It must not be used for a production outer solve or campaign.\n",
        "Family:      ", family, "\n",
        "Runner:      ", runner, "\n",
        "Reason:      ", permit.reason, "\n",
        "Caller:      ", permit.caller, "\n",
        "Bundle type: ", typeof(obj), "\n",
        "="^80, "\n")
    print(stderr, banner)
    flush(stderr)
    return nothing
end

end # module DenseReferenceDiagnostics
