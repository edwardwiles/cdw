# ============================================================================
# Production outer bridge task (2026-08-01), §14 (A/B comparability
# hardening) + §15 (production plumbing / checkpoint schema).
# ADDITIVE ONLY.
# ============================================================================

isdefined(Main, :OuterRunManifest) ||
    error("profiled_ab_comparability_and_plumbing_2026-08-01.jl requires profiled_production_outer_runner_2026-08-01.jl to be included first.")

# ----------------------------------------------------------------------------
# §14: A/B comparability hardening. The existing protocol doc
# (PROFILED_ALL_FAMILY_OUTER_AB_PROTOCOL_2026-08-01.md) SAYS warm-start and
# screen policies should match between arms; nothing in code PROVED that
# before this. `assert_ab_comparable` makes that a real, throwing check.
# ----------------------------------------------------------------------------

"""
    assert_ab_comparable(manifest_full::OuterRunManifest, manifest_profiled::OuterRunManifest;
        allow_gp_free_diff::Bool = false) -> Nothing

Throws unless the two arms' manifests agree on every field task §14 lists as
required to match: `production_subsystems` (dual_bank_policy,
obj_x_reuse_policy, exact_cache_policy, screen_set, restriction_backend,
solver_options_file), `hessopt_tag`, `knitro_opt_file`, `family`,
`economic_parameterization`. `gp_free` (fixed-gp vs. free-gp) is EXPECTED to
differ between a `:fixed_gp_parameterization_ab` arm and a
`:production_bound_search` arm by design -- only asserted equal when
`allow_gp_free_diff=false` (the default: use `true` explicitly when
comparing across those two modes on purpose, e.g. to show the profiled arm's
own two modes against each other; leave `false`, the default, when comparing
two arms that are BOTH supposed to be the same mode, since a silent gp-free
mismatch there is exactly the kind of "compared a full arm with dual-bank
warm starts to a profiled arm that cold-starts... without labeling the
difference" bug task §14 exists to catch).

Any field EITHER side has left `:not_yet_wired`/`:unknown_pending_inner`
also throws (rather than silently comparing two placeholders as "equal") --
an A/B run cannot claim comparability on a dimension neither arm has
actually recorded a real value for.
"""
function assert_ab_comparable(manifest_full::OuterRunManifest, manifest_profiled::OuterRunManifest;
        allow_gp_free_diff::Bool = false)
    placeholder(x) = x isa Symbol && x in (:not_yet_wired, :unknown_pending_inner)

    for f in fieldnames(typeof(default_production_subsystems_manifest()))
        va = getfield(manifest_full.production_subsystems, f)
        vb = getfield(manifest_profiled.production_subsystems, f)
        (placeholder(va) || placeholder(vb)) &&
            error("assert_ab_comparable: production_subsystems.$f is a placeholder on at least one arm (full=$va, profiled=$vb) -- cannot assert comparability on an unrecorded dimension")
        va == vb ||
            error("assert_ab_comparable: production_subsystems.$f differs (full=$va, profiled=$vb) -- refusing to run/compare a mismatched A/B without this being an explicit, labeled experiment")
    end

    manifest_full.hessopt_tag == manifest_profiled.hessopt_tag ||
        error("assert_ab_comparable: hessopt_tag differs (full=$(manifest_full.hessopt_tag), profiled=$(manifest_profiled.hessopt_tag))")
    manifest_full.knitro_opt_file == manifest_profiled.knitro_opt_file ||
        error("assert_ab_comparable: knitro_opt_file differs (full=$(manifest_full.knitro_opt_file), profiled=$(manifest_profiled.knitro_opt_file))")
    manifest_full.family == manifest_profiled.family ||
        error("assert_ab_comparable: family differs (full=$(manifest_full.family), profiled=$(manifest_profiled.family))")
    manifest_full.economic_parameterization == manifest_profiled.economic_parameterization ||
        error("assert_ab_comparable: economic_parameterization differs -- this is a genuine full-vs-profiled A/B, that field is EXPECTED and REQUIRED to differ; if you see this error, one arm was built wrong (both claim the same parameterization)")

    if !allow_gp_free_diff
        manifest_full.gp_free == manifest_profiled.gp_free ||
            error("assert_ab_comparable: gp_free differs (full=$(manifest_full.gp_free), profiled=$(manifest_profiled.gp_free)) and allow_gp_free_diff=false -- pass allow_gp_free_diff=true only if this is an intentional cross-mode comparison")
    end

    return nothing
end

# ----------------------------------------------------------------------------
# §15: production plumbing -- immutable config + output metadata, distinct
# checkpoint/result namespaces so a full-formulation checkpoint cannot be
# loaded into a profiled context (and vice versa).
# ----------------------------------------------------------------------------

const VALID_ECONOMIC_PARAMETERIZATIONS = (:full_gamma_normalized, :profiled_destination_scales)

"""
    ProfiledProductionConfig

Immutable configuration + output metadata (task §15). One instance per run,
written alongside every checkpoint/result file. `checkpoint_namespace`
(computed, not caller-supplied) guarantees a `:full_gamma_normalized`
checkpoint and a `:profiled_destination_scales` checkpoint can never collide
on disk or be silently cross-loaded.
"""
struct ProfiledProductionConfig
    economic_parameterization::Symbol
    stable_layout_digest::String
    combined_outer_coordinate_names::Vector{String}
    inner_dual_layout_digest::String   # placeholder until the inner branch exposes its own digest -- see below
    hessian_backends::NamedTuple
    dual_bank_policy::Symbol
    checkpoint_schema_version::String
    full_a_recovery_convention::Symbol
    checkpoint_namespace::String
end

"""
    build_profiled_production_config(fctx; economic_parameterization=:profiled_destination_scales,
        inner_dual_layout_digest=:unknown_pending_inner, dual_bank_policy=:not_yet_wired) -> ProfiledProductionConfig

Throws if `economic_parameterization` is not one of `VALID_ECONOMIC_PARAMETERIZATIONS` (task §15's
own two named conventions). `hessian_backends` defaults to this branch's own record of the
validated-but-not-yet-merged ZC production choice (task §16: H_ZZ=blas_syrk, H_CZ=draw_chunk_
reordered, H_EZ=drawmajor_v2) -- NOT because this branch uses them (it touches no Hessian code),
but so a downstream checkpoint reader can see what backend selection this run's manifest EXPECTS
once integrated, without this bridge branch importing or depending on any Hessian kernel file.
`checkpoint_schema_version` is bumped whenever `ProfiledProductionConfig`'s own field set changes
(currently "1.0.0" -- this file's first version).
"""
function build_profiled_production_config(fctx;
        economic_parameterization::Symbol = :profiled_destination_scales,
        inner_dual_layout_digest = :unknown_pending_inner,
        dual_bank_policy::Symbol = :not_yet_wired,
        restriction_outer_param_names::Vector{Symbol} = Symbol[])
    economic_parameterization in VALID_ECONOMIC_PARAMETERIZATIONS ||
        error("build_profiled_production_config: economic_parameterization=:$economic_parameterization not in $VALID_ECONOMIC_PARAMETERIZATIONS")

    digest = stable_layout_digest(fctx; restriction_outer_param_names = restriction_outer_param_names)
    names = _combined_names_or_fallback(fctx)
    idld = inner_dual_layout_digest isa AbstractString ? inner_dual_layout_digest : string(inner_dual_layout_digest)
    hess = (H_ZZ = :blas_syrk, H_CZ = :draw_chunk_reordered, H_EZ = :drawmajor_v2, source = "zc-hessian-backend-closeout-2026-08-01, committed@1807ef5, not merged/tagged")

    namespace = "$(economic_parameterization)__$(family_kind(fctx))__$(digest[1:16])"

    return ProfiledProductionConfig(economic_parameterization, digest, names, idld, hess,
        dual_bank_policy, "1.0.0", :profiled_pivot_recovery, namespace)
end

"""
    assert_checkpoint_compatible(cfg::ProfiledProductionConfig, loaded_namespace::AbstractString) -> Nothing

Throws unless `loaded_namespace == cfg.checkpoint_namespace` -- the concrete mechanism preventing
a full-formulation checkpoint from being loaded into a profiled context (task §15) or vice versa,
and preventing cross-family or cross-layout loads within the profiled family too (namespace
includes `family_kind` and the first 16 hex chars of `stable_layout_digest`).
"""
function assert_checkpoint_compatible(cfg::ProfiledProductionConfig, loaded_namespace::AbstractString)
    loaded_namespace == cfg.checkpoint_namespace ||
        error("assert_checkpoint_compatible: checkpoint namespace mismatch -- loaded=\"$loaded_namespace\" != current=\"$(cfg.checkpoint_namespace)\" (different economic_parameterization, family, or layout digest; refusing to load)")
    return nothing
end
