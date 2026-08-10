# ============================================================================
# CM+ZC-CROSS production context construction, fixed-contribution fold, and outer gradient.
# Mirrors `build_cm_meanzc_production_context`/`meanzc_fixed_contribution`/
# `build_lfix_base_cache_cm_meanzc`/`cm_meanzc_production_gradient` (cm_meanzc_production.jl)
# EXACTLY, minus the `:dense_reference` branch -- this family is `:operator`-only by construction
# (see cm_meanzc_cross_moments.jl's file header, and CLAUDE.md's no-dense-fallback rule).
#
# Everything downstream of `aug` is REUSED COMPLETELY UNCHANGED: `build_cm_meanzc_bin_ctx`
# (which now reads `aug.layout` when present -- the single edit this family needed in shared code),
# `archC_meanzc_base_state`/`archC_meanzc_verified_state`, `CMMeanZCOperatorState`'s FG callback,
# `zc_restriction_gram!`/`bin_zc_cross_hessian_fill!`/`winner_pair_cross_hessian_zc_block!`'s
# Hessian blocks, `cm_fixed_contribution_meanzc_layout` (which reads `aug.n_pair`, so the CM-grid
# block auto-shifts), and `operator_verification.jl`. Confirmed by direct reading (2026-08-09) that
# none of that code reads `K_pair`/`Zpairraw_all` as anything other than "however many
# pair-restriction columns/blocks exist".
# ============================================================================

isdefined(Main, :build_cm_meanzc_cross_augmented_obj) || include(joinpath(@__DIR__, "cm_meanzc_cross_moments.jl"))
isdefined(Main, :build_cm_meanzc_bin_ctx) || include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
isdefined(Main, :build_focal_kstar_derivative_info) || include(joinpath(@__DIR__, "autarky_cf.jl"))   # Variant D: apply_focal_kstar_chain_rule!/nu_star_value_and_dgrad

using LinearAlgebra: dot

"""
    build_cm_meanzc_cross_production_context(ctx, CS; L, K_mean, K_pair, include_truncated_moment,
        contrasts=:orthonormal, meanzc_basis=:direct, probs=nothing,
        inner_fg_backend=CM_MEANZC_INNER_FG_BACKEND_DEFAULT[], aml=nothing) -> (ctx_cm, aug, cctx, bins)

CM+ZC-CROSS analog of `build_cm_meanzc_production_context`. Builds `aug` via
`build_cm_meanzc_cross_augmented_obj`, then hands it to the (now `aug.layout`-aware) shared
`build_cm_meanzc_bin_ctx` exactly as the base family does. `inner_fg_backend` must be `:operator`
(this family has no dense-reference moments! closure at all -- see the moments file's header); a
`:dense_reference` request is a hard error, not a silent substitution.
"""
function build_cm_meanzc_cross_production_context(ctx, CS; L::Int, K_mean::Int, K_pair::Int,
                                                   include_truncated_moment::Bool,
                                                   contrasts::Symbol = :orthonormal,
                                                   meanzc_basis::Symbol = :direct,
                                                   probs::Union{Nothing,AbstractVector{Float64}} = nothing,
                                                   inner_fg_backend::Symbol = CM_MEANZC_INNER_FG_BACKEND_DEFAULT[],
                                                   aml::Union{Nothing,ActiveMeanLayout} = nothing)
    inner_fg_backend === :operator ||
        error("build_cm_meanzc_cross_production_context: CM+ZC-CROSS is :operator-only, got inner_fg_backend=:$inner_fg_backend " *
              "(this family builds an OperatorPsiBundle and no moments! closure at all -- there is nothing for :dense_reference to call)")
    println(stdout, "cm_restriction_basis [CM+ZC-CROSS] = cumulative_cdf_contrasts (CM grid) + shared mean + K_pair^2 cross-power pairwise-ZC targets")
    println(stdout, "cm_internal_feature_storage [CM+ZC-CROSS] = bin_indices")
    flush(stdout)
    isdefined(Main, :record_cm_feature_context_build!) && record_cm_feature_context_build!()
    aug = build_cm_meanzc_cross_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair,
        include_truncated_moment = include_truncated_moment, contrasts = contrasts,
        meanzc_basis = meanzc_basis, probs = probs, aml = aml)
    ctx_cm = merge(ctx, (obj = aug.obj_cm,))
    cctx = build_cm_meanzc_bin_ctx(ctx, aug; inner_fg_backend = inner_fg_backend)
    bins = cm_bin_indices_for(ctx, aug)
    pcx_result = (ctx_cm = ctx_cm, aug = aug, cctx = cctx, bins = bins)
    STASH_LIVE_PCX_ENABLED[] && (CMZC_LIVE_PCX_STASH[] = pcx_result)
    return pcx_result
end

"""
    meanzc_cross_fixed_contribution(base::BaseDualState, aug, νvec) -> Vector{Float64}

CM+ZC-CROSS analog of `meanzc_fixed_contribution` (cm_meanzc_production.jl): folds the fixed
lambda*-weighted mean+cross-pair restriction contribution into `q0` for every draw `s`, extended
over the `K_pair^2` grid.

The mean-block loop is byte-identical to the base function, INCLUDING its `aml`-aware ragged-column
subsetting. The pair-block loop is the SAME accumulation pattern with the ONE substitution the
handover identifies: `νvec[k]^2 * sum(λ_pair_k)` becomes `νvec[k1]*νvec[k2] * sum(λ_pair_klin)`
(equivalently `dot(pair_targets(layout,νvec,klin,D), λ_pair_klin)` -- written out as the scalar
product because the shared-nu target is constant across pairs, so materializing the length-`npair`
target vector every call would be pure waste; this is the same simplification the base function
already makes, and reduces to it exactly when `k1==k2`).
"""
function meanzc_cross_fixed_contribution(base::BaseDualState, aug, νvec::AbstractVector{Float64})
    layout = aug.layout
    layout isa SharedByPowerCrossLayout || error("meanzc_cross_fixed_contribution: aug.layout must be a SharedByPowerCrossLayout, got $(typeof(layout))")
    K_mean = aug.K_mean; K_pair = aug.K_pair
    length(νvec) == K_mean || error("meanzc_cross_fixed_contribution: length(νvec)=$(length(νvec)) != aug.K_mean=$K_mean")
    ncore_econ = aug.ncore_econ
    D = size(aug.Zraw_all[1], 2)
    npair = D * (D - 1) ÷ 2
    @assert length(base.λstar) >= ncore_econ - 1 + aug.n_mean + aug.n_pair "base.λstar too short for aug's (ncore_econ,n_mean,n_pair) -- was base solved against aug.obj_cm?"
    W = size(aug.Zraw_all[1], 1)
    out = zeros(W)
    mean_start = ncore_econ
    aml_local = hasproperty(aug, :aml) ? aug.aml : nothing
    mean_offset = aml_local !== nothing && aml_local.active ? mean_offset_from_aml(aml_local) : collect(0:D:K_mean*D)
    n_mean_active = mean_offset[end]
    for k in 1:K_mean
        active = aml_local !== nothing && aml_local.active ? aml_local.mean_active_origins[k] : collect(1:D)
        λ_mean_k = @view base.λstar[mean_start+mean_offset[k] : mean_start+mean_offset[k+1]-1]
        Zk_active = length(active) == D ? aug.Zraw_all[k] : @view aug.Zraw_all[k][:, active]
        out .+= Zk_active * λ_mean_k
        out .-= νvec[k] * sum(λ_mean_k)
    end
    pair_start0 = ncore_econ + n_mean_active
    levels = cross_pair_level_index(K_pair)
    for (klin, (k1, k2)) in enumerate(levels)
        λ_pair_klin = @view base.λstar[pair_start0+(klin-1)*npair : pair_start0+klin*npair-1]
        out .+= aug.Zpairraw_all[klin] * λ_pair_klin
        out .-= (νvec[k1] * νvec[k2]) * sum(λ_pair_klin)
    end
    return out
end

"""
    build_lfix_base_cache_cm_meanzc_cross(x_free0, ctx_cm, base, ctx, aug, bins, νvec; validate_dense=false) -> LFixBaseCache

CM+ZC-CROSS analog of `build_lfix_base_cache_cm_meanzc` (cm_meanzc_production.jl): folds in BOTH
`cm_fixed_contribution_meanzc_layout` (the CM-grid block, REUSED UNCHANGED -- it re-slices at
`ncore_econ + n_mean + n_pair`, which is already the correct cross-grid offset since `aug.n_pair`
is `K_pair^2*npair`) AND `meanzc_cross_fixed_contribution` (not the base family's diagonal-only
`meanzc_fixed_contribution`).
"""
function build_lfix_base_cache_cm_meanzc_cross(x_free0::AbstractVector, ctx_cm, base::BaseDualState,
                                                ctx, aug, bins::AbstractMatrix{<:Unsigned}, νvec::AbstractVector{Float64};
                                                validate_dense::Bool = false)
    cache0 = build_lfix_base_cache(x_free0, ctx_cm, base; validate_dense = validate_dense)
    cm_contrib0 = cm_fixed_contribution_meanzc_layout(base, ctx, aug, bins)
    meanzc_contrib0 = meanzc_cross_fixed_contribution(base, aug, νvec)
    return with_q0(cache0, cache0.q0 .- cm_contrib0 .- meanzc_contrib0)
end

"""
    cm_meanzc_cross_production_gradient(x_free0, νvec, pcx, ctx, pe; base=nothing, verify=nothing,
        gradient_backend=:shared_inplace_pooled, econ_ws=nothing, kwargs...) -> (g_ext, meta)

CM+ZC-CROSS analog of `cm_meanzc_production_gradient` (cm_meanzc_production.jl): the (g,A_od) block
via the SAME shared `economic_A_gradient!` backend (unaffected by the pair-block extension -- it
only ever reads `cache.q0`, already correctly folded above), PLUS the analytic
`∂Delta_dual/∂η_{ν,k}` vector appended as the last `n_eta(layout) = K_mean` components (or
`aml.n_eta_active` under Variant D), computed by `d_delta_dual_d_eta_nu_cross_vec` /
`d_delta_dual_d_eta_active_and_nustar_shared_cross` rather than the base family's diagonal-only
functions.
"""
function cm_meanzc_cross_production_gradient(x_free0::AbstractVector, νvec::AbstractVector{Float64}, pcx, ctx, pe;
        base::Union{Nothing,BaseDualState} = nothing, verify = nothing,
        gradient_backend::Symbol = :shared_inplace_pooled,
        econ_ws::Union{Nothing,EconomicAGradientWorkspace} = nothing, kwargs...)
    if base === nothing || verify === nothing
        base, verify = archC_meanzc_verified_state(x_free0, νvec, pcx.ctx_cm, pcx.cctx)
    end
    cache = build_lfix_base_cache_cm_meanzc_cross(x_free0, pcx.ctx_cm, base, ctx, pcx.aug, pcx.bins, νvec)
    if gradient_backend === :shared_inplace_pooled
        D = pcx.ctx_cm.D; Ddest = hasproperty(pcx.ctx_cm, :D_dest) ? pcx.ctx_cm.D_dest : pcx.ctx_cm.D
        ws = econ_ws === nothing ? get_or_build_econ_a_grad_ws(cache.W) : econ_ws
        g_econ = zeros(D * Ddest)
        meta = economic_A_gradient!(g_econ, base, pcx.ctx_cm, pe, ws; cache = cache, kwargs...)
    elseif gradient_backend === :legacy_unbuffered
        g_econ, meta = composite_gradient_at_fast(x_free0, pcx.ctx_cm, pe; base = base, cache = cache, kwargs...)
    else
        error("cm_meanzc_cross_production_gradient: gradient_backend must be :shared_inplace_pooled|:legacy_unbuffered, got $gradient_backend")
    end
    aml = hasproperty(pcx.aug, :aml) ? pcx.aug.aml : nothing
    if aml !== nothing && aml.active
        eta_grad_active, d_delta_d_nu_star = d_delta_dual_d_eta_active_and_nustar_shared_cross(base.λstar, pcx.aug, aml, νvec; mean_m = verify.m_mean)
        θ_full = CS.reconstruct_full(x_free0, ctx.m)
        info = build_focal_kstar_derivative_info(ctx, pe)
        D2_econ = length(g_econ)
        apply_focal_kstar_chain_rule!(g_econ, θ_full, ctx, info, D2_econ, d_delta_d_nu_star)
        return vcat(g_econ, eta_grad_active), meta
    end
    d_eta = d_delta_dual_d_eta_nu_cross_vec(base.λstar, pcx.aug, νvec; mean_m = verify.m_mean)
    return vcat(g_econ, d_eta), meta
end
