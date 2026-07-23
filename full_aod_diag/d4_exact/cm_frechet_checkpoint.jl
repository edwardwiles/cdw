# ============================================================================
# Fixed-Fréchet-marginals checkpoint schema (2026-07-23), schema 5. Follows
# the EXACT same discipline `CMCheckpointV4` (cm_checkpoint.jl) established
# for the CM+moments(+ZC) integration: `CMCheckpointV4` is retained
# PERMANENTLY UNCHANGED (every existing schema-4 file must keep loading), a
# NEW type `CMCheckpointV5` is introduced for schema>=5, and an
# `upgrade_schema4` function fills the new fields with the ONE value
# consistent with every schema-4 file's own provenance
# (`marginal_mode=:common_flexible` -- the fixed-Fréchet restriction did not
# exist anywhere in the codebase when any schema-4 file was written).
# Purely additive: does not modify cm_checkpoint.jl.
# ============================================================================

using Serialization: serialize, deserialize

const CM_FRECHET_CHECKPOINT_SCHEMA = 5

"""
    CMCheckpointV5

Identical to `CMCheckpointV4` except eight new fields, appended at the end:
`marginal_mode`, `frechet_theta_star`, `frechet_scale`, `frechet_sigma`,
`frechet_probs`, `frechet_thresholds_checksum`, `frechet_target_checksum`,
`frechet_feature_layout_version`. `CMCheckpointV4` is retained permanently,
read-only, for every schema-4 file already written by the CM+moments(+ZC)
production campaign.
"""
struct CMCheckpointV5
    schema::Int
    run_id::String
    label::String
    branch::Symbol
    find_smallest::Bool
    delta::Float64
    W::Int
    draw_seed::Int
    draw_design::Symbol
    draw_checksum_uniform::String
    draw_checksum_transformed::String
    cm_L::Int
    cm_probs::Vector{Float64}
    cm_contrasts::Symbol
    cm_grid_rule::Symbol
    cm_basis::Symbol
    cm_hessian_backend::Symbol
    cm_gradient_backend::Symbol
    cm_extension::Symbol
    meanzc_K_mean::Int
    meanzc_K_pair::Int
    meanzc_basis::Symbol
    moment_layout_version::Int
    g::Float64
    zfree::Vector{Float64}
    eta_nu::Vector{Float64}
    logA_full::Matrix{Float64}
    dual_warm_start::Vector{Float64}
    bandwidth_cache::Dict{Int,Float64}
    best_feasible::Any
    n_eval::Int
    n_grad::Int
    wall_elapsed::Float64
    wall_budget_remaining::Float64
    checkpoint_reason::Symbol
    knitro_version::String
    marginal_mode::Symbol                    # :common_flexible | :frechet_reference
    frechet_theta_star::Float64              # NaN when marginal_mode=:common_flexible
    frechet_scale::Float64                   # NaN when marginal_mode=:common_flexible
    frechet_sigma::Float64                   # NaN when marginal_mode=:common_flexible
    frechet_probs::Vector{Float64}           # == cm_probs when active; Float64[] when :common_flexible
    frechet_thresholds_checksum::String      # sha256 of targets.thresholds; "" when :common_flexible
    frechet_target_checksum::String          # targets.target_sha256; "" when :common_flexible
    frechet_feature_layout_version::Int      # FRECHET_FEATURE_LAYOUT_VERSION; 0 when :common_flexible
end

"Atomic-ish checkpoint write, same discipline as `save_cm_checkpoint`."
function save_cm_frechet_checkpoint(path::AbstractString, ckpt::CMCheckpointV5)
    tmp = path * ".tmp"
    serialize(tmp, ckpt)
    mv(tmp, path; force = true)
    return path
end

"""
    upgrade_schema4(old::CMCheckpointV4) -> CMCheckpointV5

CORRECT (not a guess) for every schema-4 file that exists: fixed-Fréchet
marginals did not exist anywhere in the codebase when any schema-4 file was
written, so `marginal_mode=:common_flexible` with sentinel (`NaN`/empty)
frechet-specific fields is the only value consistent with those files' own
provenance -- mirrors `CMCheckpointV4`'s own `upgrade_schema3`.
"""
function upgrade_schema4(old::CMCheckpointV4)
    return CMCheckpointV5(CM_FRECHET_CHECKPOINT_SCHEMA, old.run_id, old.label, old.branch, old.find_smallest,
        old.delta, old.W, old.draw_seed, old.draw_design, old.draw_checksum_uniform, old.draw_checksum_transformed,
        old.cm_L, old.cm_probs, old.cm_contrasts, old.cm_grid_rule, old.cm_basis, old.cm_hessian_backend,
        old.cm_gradient_backend, old.cm_extension, old.meanzc_K_mean, old.meanzc_K_pair, old.meanzc_basis,
        old.moment_layout_version, old.g, old.zfree, old.eta_nu, old.logA_full, old.dual_warm_start,
        old.bandwidth_cache, old.best_feasible, old.n_eval, old.n_grad, old.wall_elapsed, old.wall_budget_remaining,
        old.checkpoint_reason, old.knitro_version,
        :common_flexible, NaN, NaN, NaN, Float64[], "", "", 0)
end

"""
    load_cm_frechet_checkpoint(path) -> CMCheckpointV5

Tries the CURRENT (schema>=5) shape first; falls back to schema-4
(`load_cm_checkpoint`'s own fallback chain, reused unchanged) then
`upgrade_schema4`.
"""
function load_cm_frechet_checkpoint(path::AbstractString)
    try
        return deserialize(path)::CMCheckpointV5
    catch e
        (e isa TypeError || e isa EOFError || e isa MethodError) || rethrow()
        return upgrade_schema4(load_cm_checkpoint(path))
    end
end

"""
    cm_frechet_checkpoint_context(cfg::CMFrechetConfig, targets::Union{Nothing,FrechetReferenceTargets}) -> NamedTuple

The fingerprint fields task brief §9 requires be persisted AND checked on
resume: `marginal_mode`, reference family/theta_star/scale/sigma, grid
probabilities, grid-thresholds checksum, target-vector checksum, feature-
layout version, `L`, contrasts, gradient backend (the last two threaded
through by the caller, not stored on `targets`).
"""
function cm_frechet_checkpoint_context(cfg::CMFrechetConfig, targets::Union{Nothing,FrechetReferenceTargets})
    if cfg.marginal_mode === :common_flexible || targets === nothing
        return (marginal_mode = :common_flexible, theta_star = NaN, scale = NaN, sigma = NaN,
                probs = Float64[], thresholds_checksum = "", target_checksum = "", feature_layout_version = 0)
    end
    return (marginal_mode = :frechet_reference, theta_star = targets.theta_star, scale = targets.scale,
            sigma = targets.sigma, probs = targets.probs, thresholds_checksum = _float_vector_sha256(targets.thresholds),
            target_checksum = targets.target_sha256, feature_layout_version = targets.feature_layout_version)
end

"""
    cm_frechet_checkpoint_refusal_reason(ckpt::CMCheckpointV5, requested) -> Union{Nothing,String}

`requested` is a `cm_frechet_checkpoint_context(...)`-shaped NamedTuple for
the CURRENT call's config. Returns `nothing` if compatible (safe to resume),
else a human-readable reason string. Refuses (task brief §9) under: changed
`marginal_mode`; changed `theta_star`; changed `sigma`; changed grid
probabilities OR thresholds checksum; changed target checksum; changed `L`;
changed feature layout; changed draw checksum (checked by the CALLER via the
existing `draw_checksum_uniform`/`draw_checksum_transformed` fields, already
present on `CMCheckpointV5` -- not duplicated here).
"""
function cm_frechet_checkpoint_refusal_reason(ckpt::CMCheckpointV5, requested)
    ckpt.marginal_mode == requested.marginal_mode ||
        return "marginal_mode mismatch: checkpoint=:$(ckpt.marginal_mode), requested=:$(requested.marginal_mode)"
    if requested.marginal_mode === :frechet_reference
        ckpt.frechet_theta_star == requested.theta_star ||
            return "frechet_theta_star mismatch: checkpoint=$(ckpt.frechet_theta_star), requested=$(requested.theta_star)"
        ckpt.frechet_sigma == requested.sigma ||
            return "frechet_sigma mismatch: checkpoint=$(ckpt.frechet_sigma), requested=$(requested.sigma)"
        ckpt.frechet_probs == requested.probs ||
            return "frechet_probs (grid probabilities) mismatch"
        ckpt.frechet_thresholds_checksum == requested.thresholds_checksum ||
            return "frechet_thresholds_checksum mismatch: checkpoint=$(ckpt.frechet_thresholds_checksum), requested=$(requested.thresholds_checksum)"
        ckpt.frechet_target_checksum == requested.target_checksum ||
            return "frechet_target_checksum mismatch: checkpoint=$(ckpt.frechet_target_checksum), requested=$(requested.target_checksum)"
        ckpt.cm_L == length(requested.probs) ||
            return "cm_L mismatch: checkpoint=$(ckpt.cm_L), requested=$(length(requested.probs))"
        ckpt.frechet_feature_layout_version == requested.feature_layout_version ||
            return "frechet_feature_layout_version mismatch: checkpoint=$(ckpt.frechet_feature_layout_version), requested=$(requested.feature_layout_version)"
    end
    return nothing
end
