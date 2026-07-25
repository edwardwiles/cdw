# ============================================================================
# Fixed-Fréchet-marginals checkpoint (port-prep 2026-07-24). NEW type name,
# NOT `CMCheckpointV8` or any variant of the shared `CMCheckpointVN` numbering
# sequence -- `git grep -n "CMCheckpointV"` at this branch's base commit shows
# V3/V4/V5/V6/V7 all already taken (V5/V7 by the UNRELATED origin-specific-ZC
# family in cm_originzc_checkpoint.jl), exactly the collision this repo has
# hit before (memory: checkpoint-schema-bump-collision-check). Using a
# genuinely distinct base name (`CMFrechetCheckpointV1`) sidesteps that
# numbering sequence entirely rather than trying to claim the next free
# integer in a sequence two OTHER files also mutate.
#
# Field-list convention follows `CMCheckpointV6` (cm_checkpoint.jl) exactly
# where the concept is shared (draw/data checksums, state/progress fields,
# active-sample/layout fields), per docs/CM_PRODUCTION_HOOK_INTERFACE_SPEC_2026-07-24.md
# §5's explicit naming-convention guidance -- field NAMES are reused verbatim
# (`destination_sample`, `row_idx`, `D_dest`, `draw_seed`, `draw_checksum_*`),
# not re-invented as synonyms.
# ============================================================================

using Serialization: serialize, deserialize
using SHA: sha256, bytes2hex

const CM_FRECHET_CHECKPOINT_SCHEMA = 1

"""
    CMFrechetCheckpointV1

Checkpoint for the `marginal_mode=:frechet_reference` restricted model.
"""
struct CMFrechetCheckpointV1
    schema::Int
    run_id::String
    label::String
    find_smallest::Bool
    delta::Float64
    W::Int
    draw_seed::Int
    draw_design::Symbol
    draw_checksum_uniform::String
    draw_checksum_transformed::String
    # ---- active destination-sample/layout (task brief §10) ----
    destination_sample::Symbol         # :all_legacy | :exclude_row
    row_idx::Union{Nothing,Int}
    D::Int                             # origin count
    D_dest::Int                        # destination count
    # ---- live calibration provenance (task brief §3/§10) ----
    theta_star::Float64                # 1/muHat, LIVE
    muHat::Float64
    sigma::Float64
    gravity_sample_version::Any
    theta_calibration_version::Any
    # ---- fixed-Frechet feature configuration (task brief §10) ----
    frechet_L::Int
    frechet_probs::Vector{Float64}
    frechet_feature_set::Symbol        # :cdf_only | :cdf_power
    frechet_basis::Symbol              # :cumulative | :interval
    cm_contrasts::Symbol
    cm_grid_rule::Symbol
    target_sha256_cdf::String
    target_sha256_power::String
    feature_layout_version::Int
    core_winner_engine_version::Int    # bump if the core-moment/winner interface this branch
                                        # consumes (docs/CM_PRODUCTION_HOOK_INTERFACE_SPEC) changes
    # ---- state/progress (mirrors CMCheckpointV6's own field set) ----
    g::Float64
    zfree::Vector{Float64}
    logA_full::Matrix{Float64}
    dual_warm_start::Vector{Float64}
    best_feasible::Any
    n_eval::Int
    n_grad::Int
    wall_elapsed::Float64
    wall_budget_remaining::Float64
    checkpoint_reason::Symbol
    knitro_version::String
end

const CORE_WINNER_ENGINE_VERSION = 1   # bump when the canonical winner-engine integration
                                        # (FIXED_FRECHET_CANONICAL_WINNER_INTEGRATION_MANIFEST_2026-07-24.md)
                                        # lands and this branch's core-moment provider changes shape.

"sha256 hex digest of raw draw bytes -- same checksum discipline as `CMCheckpointV6`'s `draw_checksum_uniform`/`draw_checksum_transformed`."
function _frechet_checksum(v::AbstractArray{Float64})
    return bytes2hex(sha256(reinterpret(UInt8, vec(collect(v)))))
end

"""
    build_cm_frechet_checkpoint(ctx, cfg::CMFrechetConfig, targets::FrechetReferenceTargets,
                                 fpcx; run_id, label, g, zfree, logA_full, dual_warm_start,
                                 best_feasible, n_eval, n_grad, wall_elapsed, wall_budget_remaining,
                                 checkpoint_reason, knitro_version) -> CMFrechetCheckpointV1

Assembles a fingerprint that REJECTS a mismatched resume loudly (task brief
§10: "Reject incompatible checkpoints loudly") -- see `check_cm_frechet_checkpoint_compatible`.
"""
function build_cm_frechet_checkpoint(ctx, cfg::CMFrechetConfig, targets::FrechetReferenceTargets, fpcx;
        run_id::AbstractString, label::AbstractString,
        g::Float64, zfree::Vector{Float64}, logA_full::Matrix{Float64}, dual_warm_start::Vector{Float64},
        best_feasible, n_eval::Int, n_grad::Int, wall_elapsed::Float64, wall_budget_remaining::Float64,
        checkpoint_reason::Symbol, knitro_version::AbstractString,
        draw_seed::Int = -1, draw_design::Symbol = :unknown)
    ds = hasproperty(ctx, :destination_sample) ? ctx.destination_sample : :unknown
    row_idx = hasproperty(ctx, :row_idx) ? ctx.row_idx : nothing
    D_dest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    gsv = hasproperty(ctx, :gravity_sample_version) ? ctx.gravity_sample_version : nothing
    tcv = hasproperty(ctx, :theta_calibration_version) ? ctx.theta_calibration_version : nothing
    return CMFrechetCheckpointV1(CM_FRECHET_CHECKPOINT_SCHEMA, String(run_id), String(label),
        ctx.find_smallest, ctx.δ, size(ctx.U, 1), draw_seed, draw_design,
        _frechet_checksum(ctx.U), _frechet_checksum(ctx.U),   # transformed==uniform here (U IS the native representation this codebase uses; no separate transform stage)
        ds, row_idx, ctx.D, D_dest,
        targets.theta_star, ctx.μHat, targets.sigma, gsv, tcv,
        length(targets.probs), targets.probs, cfg.frechet_feature_set, cfg.frechet_basis,
        cfg.cm.contrasts, cfg.cm.cm_grid_rule,
        targets.target_sha256, targets.power_target_sha256, targets.feature_layout_version,
        CORE_WINNER_ENGINE_VERSION,
        g, zfree, logA_full, dual_warm_start, best_feasible, n_eval, n_grad,
        wall_elapsed, wall_budget_remaining, checkpoint_reason, String(knitro_version))
end

"Atomic-ish checkpoint write, same discipline as `save_cm_checkpoint` (cm_checkpoint.jl): write to a temp path then `mv`."
function save_cm_frechet_checkpoint(path::AbstractString, ckpt::CMFrechetCheckpointV1)
    tmp = path * ".tmp." * string(getpid())
    open(tmp, "w") do io
        serialize(io, ckpt)
    end
    mv(tmp, path; force = true)
    return nothing
end

"Loads a `CMFrechetCheckpointV1`. No legacy-schema fallback chain exists yet (schema 1 is the first)."
function load_cm_frechet_checkpoint(path::AbstractString)::CMFrechetCheckpointV1
    return open(path, "r") do io
        deserialize(io)::CMFrechetCheckpointV1
    end
end

"""
    check_cm_frechet_checkpoint_compatible(ckpt::CMFrechetCheckpointV1, ctx, cfg::CMFrechetConfig,
                                            targets::FrechetReferenceTargets) -> Nothing

Rejects (throws) a resume whose fingerprint does not match the CURRENT run's
active configuration -- task brief §10's "Reject incompatible checkpoints
loudly." Checked: destination_sample/row_idx/D/D_dest, feature_set, basis,
contrasts, L/probs, target hashes, draw seed+checksum (if the checkpoint
recorded one), core_winner_engine_version.
"""
function check_cm_frechet_checkpoint_compatible(ckpt::CMFrechetCheckpointV1, ctx, cfg::CMFrechetConfig,
        targets::FrechetReferenceTargets)
    ds = hasproperty(ctx, :destination_sample) ? ctx.destination_sample : :unknown
    row_idx = hasproperty(ctx, :row_idx) ? ctx.row_idx : nothing
    D_dest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    mismatches = String[]
    ckpt.destination_sample === ds || push!(mismatches, "destination_sample: ckpt=$(ckpt.destination_sample) vs current=$ds")
    ckpt.row_idx == row_idx || push!(mismatches, "row_idx: ckpt=$(ckpt.row_idx) vs current=$row_idx")
    ckpt.D == ctx.D || push!(mismatches, "D: ckpt=$(ckpt.D) vs current=$(ctx.D)")
    ckpt.D_dest == D_dest || push!(mismatches, "D_dest: ckpt=$(ckpt.D_dest) vs current=$D_dest")
    ckpt.frechet_feature_set === cfg.frechet_feature_set || push!(mismatches, "frechet_feature_set: ckpt=$(ckpt.frechet_feature_set) vs current=$(cfg.frechet_feature_set)")
    ckpt.frechet_basis === cfg.frechet_basis || push!(mismatches, "frechet_basis: ckpt=$(ckpt.frechet_basis) vs current=$(cfg.frechet_basis)")
    ckpt.cm_contrasts === cfg.cm.contrasts || push!(mismatches, "cm_contrasts: ckpt=$(ckpt.cm_contrasts) vs current=$(cfg.cm.contrasts)")
    ckpt.frechet_L == length(targets.probs) || push!(mismatches, "frechet_L: ckpt=$(ckpt.frechet_L) vs current=$(length(targets.probs))")
    ckpt.target_sha256_cdf == targets.target_sha256 || push!(mismatches, "target_sha256_cdf mismatch (grid/probability construction changed)")
    ckpt.target_sha256_power == targets.power_target_sha256 || push!(mismatches, "target_sha256_power mismatch (theta*/sigma/grid changed)")
    ckpt.feature_layout_version == targets.feature_layout_version || push!(mismatches, "feature_layout_version: ckpt=$(ckpt.feature_layout_version) vs current=$(targets.feature_layout_version)")
    ckpt.core_winner_engine_version == CORE_WINNER_ENGINE_VERSION || push!(mismatches, "core_winner_engine_version: ckpt=$(ckpt.core_winner_engine_version) vs current=$CORE_WINNER_ENGINE_VERSION")
    isempty(mismatches) || error("check_cm_frechet_checkpoint_compatible: REJECTED, $(length(mismatches)) mismatch(es):\n  " *
        join(mismatches, "\n  "))
    return nothing
end
