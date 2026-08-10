# ============================================================================
# OZC-CROSS production context construction. Mirrors `build_originzc_augmented_obj`/
# `build_originzc_production_context` (cm_originzc_moments.jl/cm_originzc_production.jl) EXACTLY,
# minus the `dense_reference` branch -- this family is `:operator`-only by construction (production
# never uses `:dense_reference` for origin-ZC either, per `build_originzc_augmented_obj`'s own
# restriction of Variant D to `:operator`; OZC-CROSS makes the SAME restriction from the start rather
# than adding a dense-reference `moments!` closure nobody would use -- see CLAUDE.md's
# no-dense-fallback rule). `aml` (Variant D focal-row omission) IS supported, as of 2026-08-09 --
# see `build_originzc_cross_augmented_obj`'s own docstring and
# `d_delta_dual_d_eta_active_and_nustar_cross`/`cm_originzc_cross_production_gradient`'s aml branch
# below.
#
# Everything downstream of `aug` (`build_originzc_core_hess_ctx`, `archOZ_base_state`/
# `archOZ_verified_state`, `OriginZCOperatorState`'s FG callback, `zc_restriction_gram!`/
# `winner_pair_cross_hessian_zc_block!`'s Hessian blocks, `operator_verification.jl`) is REUSED
# COMPLETELY UNCHANGED from cm_hessian_architectures.jl/cm_originzc_production.jl/
# cm_originzc_lookup_kernels.jl -- confirmed by direct reading (2026-08-09) that none of that code
# reads `K_pair`/`Zpairraw_all` as anything other than "however many pair-restriction columns/blocks
# exist"; this file's only job is to hand it the RIGHT `Zpairraw_all`/`layout` (K_pair^2 cross grid
# instead of K_pair diagonal levels).
# ============================================================================

isdefined(Main, :OperatorPsiBundle) || include(joinpath(@__DIR__, "operator_psi_bundle.jl"))
isdefined(Main, :cross_pair_level_index) || include(joinpath(@__DIR__, "cm_originzc_cross_moments.jl"))
isdefined(Main, :OriginByPowerCrossLayout) || include(joinpath(@__DIR__, "cm_originzc_cross_target_layout.jl"))
isdefined(Main, :build_originzc_core_hess_ctx) || include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
isdefined(Main, :build_focal_kstar_derivative_info) || include(joinpath(@__DIR__, "autarky_cf.jl"))   # 2026-08-09 Variant D: apply_focal_kstar_chain_rule!/nu_star_value_and_dgrad

using LinearAlgebra: dot

"""
    build_originzc_cross_augmented_obj(ctx, CS, layout::OriginByPowerCrossLayout;
                                        aml::Union{Nothing,ActiveMeanLayout}=nothing) -> NamedTuple

OZC-CROSS analog of `build_originzc_augmented_obj`'s `:operator` branch: constructs a NEW
`OperatorPsiBundle` restricted by the normal production economic/gravity moments plus origin-
specific mean moments (UNCHANGED) and the K_pair^2 cross-power pairwise-ZC moments (NEW, via
`build_raw_cross_pair_matrix_levels`). `ctx.obj` is left untouched. Returns the same NamedTuple
shape `build_originzc_core_hess_ctx` expects (`obj_cm`, `ncore_econ`, `Zraw_all`, `Zpairraw_all`,
`layout`, `aml`).

`aml` (2026-08-09, Variant D support): optional focal `k*=sigma-1` mean-row omission, IDENTICAL
mechanism to the base family's own (`ActiveMeanLayout`, `cm_originzc_target_layout.jl`) -- it
touches ONLY the mean block (`n_mean` becomes `mean_offset_from_aml(aml)[end]` instead of the dense
`K_mean*D`, and `build_originzc_core_hess_ctx`'s own aml-aware branch picks up the ragged
`ZCRestrictionOperator` constructor automatically). The PAIR block (Zpairraw_all/n_pair) is
COMPLETELY UNCHANGED by aml -- exactly as documented for the base family (`mean_offset_from_aml`'s
own docstring / `originzc_fixed_contribution`'s comment): the pair loop still runs over every
unordered pair at every level, including every pair involving the focal origin. `nothing` (default):
zero behavior change, byte-identical to every existing caller.
"""
function build_originzc_cross_augmented_obj(ctx, CS, layout::OriginByPowerCrossLayout;
        aml::Union{Nothing,ActiveMeanLayout} = nothing)
    aml === nothing || aml.base === layout ||
        error("build_originzc_cross_augmented_obj: aml.base must be === layout (got a different layout object)")
    obj0 = ctx.obj
    ncore_econ = obj0.d
    D = ctx.D
    K_mean = layout.K_mean; K_pair = layout.K_pair

    # K_pair=0 in this call: only Zraw_all (the per-level Frechet-power features) is needed here --
    # the diagonal-only Zpairraw_all this function would otherwise also build is unused (this
    # family builds its OWN Zpairraw_all, from Zraw_all, via build_raw_cross_pair_matrix_levels).
    Zraw_all, _ = build_raw_mean_pair_matrix_levels(ctx.U, K_mean, 0; μ = ctx.μHat)
    Zpairraw_all = build_raw_cross_pair_matrix_levels(Zraw_all, K_pair)

    npair = div(D * (D - 1), 2)
    n_mean_dense = K_mean * D
    n_mean = (aml !== nothing && aml.active) ? mean_offset_from_aml(aml)[end] : n_mean_dense
    n_pair = K_pair^2 * npair
    @assert n_mean_dense + n_pair == n_originzc_cross_moments(D, K_mean, K_pair)

    d_new = ncore_econ + n_mean + n_pair
    outer_constr_index_new = obj0.outer_constr_index + n_mean + n_pair

    obj_oz = OperatorPsiBundle(δ = obj0.δ, find_smallest = obj0.find_smallest,
        γ = obj0.γ, l = obj0.l, outer_constr_index = outer_constr_index_new,
        inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
        U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
        use_cached_x = obj0.use_cached_x, threshold_state = obj0.threshold_state,
        inner_loop_opt = obj0.inner_loop_opt)
    @assert obj_oz.outer_constr_index == d_new

    return (obj_cm = obj_oz, ncore = ncore_econ,
            Zraw_all = Zraw_all, Zpairraw_all = Zpairraw_all, layout = layout,
            K_mean = K_mean, K_pair = K_pair, n_mean = n_mean, n_pair = n_pair,
            ncore_econ = ncore_econ, core_cf_ref = Ref{Any}(nothing), moments_skip! = nothing,
            aml = aml)
end

"""
    build_originzc_cross_production_context(ctx, CS, layout::OriginByPowerCrossLayout;
                                             aml=nothing, zc_cross_hessian_backend=...) -> (ctx_cm, aug, octx)

OZC-CROSS analog of `build_originzc_production_context`: builds `aug` via
`build_originzc_cross_augmented_obj` above, then hands it to the UNCHANGED shared
`build_originzc_core_hess_ctx` (cm_hessian_architectures.jl) exactly as the base family does --
that function is column-count-agnostic (reads `Zraw_all`/`Zpairraw_all` off `aug` and builds a
`ZCRestrictionOperator` from them, never assuming anything about K_pair's meaning) AND already
aml-aware (dispatches to the ragged `ZCRestrictionOperator(...,aml)` constructor when
`aug.aml !== nothing && aug.aml.active`, unchanged, no edit needed here).
"""
function build_originzc_cross_production_context(ctx, CS, layout::OriginByPowerCrossLayout;
        aml::Union{Nothing,ActiveMeanLayout} = nothing,
        zc_cross_hessian_backend::Symbol = ORIGINZC_ZC_CROSS_HESSIAN_BACKEND_DEFAULT[])
    println(stdout, "cm_restriction_basis [OZC-CROSS] = none (no CM-grid block; origin-specific mean + K_pair^2 cross-power pairwise-ZC targets only)")
    flush(stdout)
    aug = build_originzc_cross_augmented_obj(ctx, CS, layout; aml = aml)
    octx = build_originzc_core_hess_ctx(aug, ctx; fg_backend = :operator, zc_cross_hessian_backend = zc_cross_hessian_backend)
    ctx_cm = merge(ctx, (obj = aug.obj_cm, octx = octx))
    return (ctx_cm = ctx_cm, aug = aug, octx = octx)
end

# ----------------------------------------------------------------------------
# Outer-gradient envelope formula + fixed-contribution fold, extended to the K_pair^2 grid.
# 2026-08-09 task, phase 2: no NEW outer parameters (nu_{o,k} is unchanged, n_eta(layout) is
# unchanged) -- but the accumulation into d(Delta_dual)/d(nu) and the mean/pair fixed-contribution
# fold DO need extending, since more inner restrictions now touch each nu_{o,k}, and (for k1!=k2)
# power k1 attaches to origin o while power k2 attaches to the OTHER origin p in the same pair --
# not the same k for both, unlike the base family's diagonal-only formula.
# ----------------------------------------------------------------------------

"""
    d_delta_dual_d_eta_origin_cross_vec(λstar, aug, νfull; mean_m) -> Vector{Float64}

OZC-CROSS analog of `d_delta_dual_d_eta_origin_vec` (cm_originzc_moments.jl): the IDENTICAL
envelope-derivative formula (math note Section 4's `d(Delta_dual)/d(nu) = -mean_m*(...)`,
`d(Delta_dual)/d(eta) = nu*d(Delta_dual)/d(nu)` chain rule), extended to the `K_pair^2` ordered
cross-power grid (`aug.layout::OriginByPowerCrossLayout`). The mean-block loop is byte-identical to
the base formula (mean targets/derivatives are completely unaffected by the pair-block extension).
The pair-block loop is the one genuine extension: for target `t_{op,klin}(nu) = nu_{o,k1}*nu_{p,k2}`
(`(k1,k2) = cross_pair_level_index(layout.K_pair)[klin]`, `o<p` canonical), the envelope derivative
contributes `-mean_m*lambda_pair,op,klin* * nu_{p,k2}` to `d(Delta_dual)/d(nu_{o,k1})` and
symmetrically `-mean_m*lambda_pair,op,klin* * nu_{o,k1}` to `d(Delta_dual)/d(nu_{p,k2})` -- `k1`
attaches to `o`, `k2` to `p`, NOT the same level for both (unlike the base family, where the pair
loop only ever touches `d_nu` at index `k` for both origins). Reduces exactly to
`d_delta_dual_d_eta_origin_vec`'s formula when `K_pair=1` (single diagonal combo, `k1=k2=1`).
"""
function d_delta_dual_d_eta_origin_cross_vec(λstar::AbstractVector{Float64}, aug, νfull::AbstractVector{Float64}; mean_m::Float64)
    layout = aug.layout
    layout isa OriginByPowerCrossLayout || error("d_delta_dual_d_eta_origin_cross_vec: aug.layout must be an OriginByPowerCrossLayout, got $(typeof(layout))")
    K_mean = layout.K_mean; K_pair = layout.K_pair
    D = size(aug.Zraw_all[1], 2)
    npair = D * (D - 1) ÷ 2
    length(νfull) == n_eta(layout) || error("d_delta_dual_d_eta_origin_cross_vec: length(νfull)=$(length(νfull)) != n_eta(layout)=$(n_eta(layout))")
    ncore_econ = aug.ncore_econ
    mean_start = ncore_econ
    pair_start0 = ncore_econ + K_mean * D
    d_nu = zeros(n_eta(layout))
    pairs = packed_pair_index(D)
    for k in 1:K_mean
        λ_mean_k = @view λstar[mean_start+(k-1)*D : mean_start+k*D-1]
        for o in 1:D
            idx = target_index(layout, o, k)
            d_nu[idx] -= λ_mean_k[o]
        end
    end
    levels = cross_pair_level_index(K_pair)
    for (klin, (k1, k2)) in enumerate(levels)
        λ_pair_klin = @view λstar[pair_start0+(klin-1)*npair : pair_start0+klin*npair-1]
        for (j, (o, p)) in enumerate(pairs)
            idx_o = target_index(layout, o, k1)
            idx_p = target_index(layout, p, k2)
            nu_o = νfull[idx_o]; nu_p = νfull[idx_p]
            d_nu[idx_o] -= nu_p * λ_pair_klin[j]
            d_nu[idx_p] -= nu_o * λ_pair_klin[j]
        end
    end
    d_nu .*= mean_m
    return νfull .* d_nu   # chain rule, nu = exp(eta)
end

"""
    d_delta_dual_d_eta_active_and_nustar_cross(λstar, aug, aml::ActiveMeanLayout, nu_eff::Vector{Float64};
                                                mean_m::Float64) -> (eta_grad_active::Vector{Float64}, d_delta_d_nu_star::Float64)

OZC-CROSS analog of `d_delta_dual_d_eta_active_and_nustar` (cm_originzc_moments.jl) -- Variant D
(focal `k*` mean-row omission) applied to the `K_pair^2` cross grid. The mean-block loop is copied
verbatim from the base function (ragged-aware via `aml.mean_active_origins`/`mean_offset_from_aml`
-- byte-identical, since the mean block never changes for OZC-CROSS). The pair-block loop is
`d_delta_dual_d_eta_origin_cross_vec`'s cross-grid accumulation (power `k1` on the lower-indexed
origin `o`, `k2` on the higher `p`), with the ragged-aware `pair_start0 = ncore_econ + n_mean_active`
offset substituted for the dense `ncore_econ + K_mean*D` that function uses -- the pair block itself
is otherwise completely unaffected by `aml` (Variant D only removes a MEAN row; every pair
restriction involving the focal origin at every level is retained, exactly as documented in
`mean_offset_from_aml`'s own docstring).

`λstar` must come from an inner solve that used the `aml`-constructed ragged `ZCRestrictionOperator`
(`build_originzc_core_hess_ctx`'s aml branch, already generic). `nu_eff` is the FULL dense nu vector
(length `n_eta(aml.base)`, via `scatter_nu_eff`).
"""
function d_delta_dual_d_eta_active_and_nustar_cross(λstar::AbstractVector{Float64}, aug, aml::ActiveMeanLayout,
                                                      nu_eff::AbstractVector{Float64}; mean_m::Float64)
    layout = aml.base
    layout isa OriginByPowerCrossLayout || error("d_delta_dual_d_eta_active_and_nustar_cross: aml.base must be an OriginByPowerCrossLayout, got $(typeof(layout))")
    K_mean = layout.K_mean; K_pair = layout.K_pair
    D = layout.D
    npair = D * (D - 1) ÷ 2
    length(nu_eff) == n_eta(layout) || error("d_delta_dual_d_eta_active_and_nustar_cross: length(nu_eff)=$(length(nu_eff)) != n_eta(layout)=$(n_eta(layout))")
    ncore_econ = aug.ncore_econ
    mean_start = ncore_econ
    mean_offset = mean_offset_from_aml(aml)
    n_mean_active = mean_offset[end]
    pair_start0 = ncore_econ + n_mean_active   # ragged-aware: active count, not K_mean*D
    expected_len = (ncore_econ - 1) + n_mean_active + K_pair^2 * npair
    length(λstar) >= expected_len ||
        error("d_delta_dual_d_eta_active_and_nustar_cross: length(λstar)=$(length(λstar)) < expected_len=$expected_len " *
              "(ncore_econ=$ncore_econ, n_mean_active=$n_mean_active, K_pair=$K_pair, npair=$npair) -- " *
              "aug/λstar dimension mismatch, refusing to silently misindex.")
    d_nu = zeros(n_eta(layout))
    pairs = packed_pair_index(D)
    # Mean-block loop: byte-identical to d_delta_dual_d_eta_active_and_nustar's own (mean block is
    # unaffected by the cross-pair extension).
    for k in 1:K_mean
        active_k = aml.mean_active_origins[k]
        λ_mean_k = @view λstar[mean_start+mean_offset[k] : mean_start+mean_offset[k+1]-1]
        for (jo, o) in enumerate(active_k)
            idx = target_index(layout, o, k)
            d_nu[idx] -= λ_mean_k[jo]
        end
    end
    # Pair-block loop: d_delta_dual_d_eta_origin_cross_vec's cross-grid accumulation, ragged offset.
    levels = cross_pair_level_index(K_pair)
    for (klin, (k1, k2)) in enumerate(levels)
        λ_pair_klin = @view λstar[pair_start0+(klin-1)*npair : pair_start0+klin*npair-1]
        for (j, (o, p)) in enumerate(pairs)
            idx_o = target_index(layout, o, k1)
            idx_p = target_index(layout, p, k2)
            nu_o = nu_eff[idx_o]; nu_p = nu_eff[idx_p]
            d_nu[idx_o] -= nu_p * λ_pair_klin[j]
            d_nu[idx_p] -= nu_o * λ_pair_klin[j]
        end
    end
    d_nu .*= mean_m
    eta_grad_dense = nu_eff .* d_nu   # chain rule, nu = exp(eta), well-defined at every dense index
    eta_grad_active, _ = gather_active_grad(aml, eta_grad_dense)
    # d_delta_d_nu_star is the RAW (un-nu-multiplied) d(Delta)/d(nu_star) -- gathered from d_nu
    # directly, NOT from eta_grad_dense (matching the base function's own identical reasoning).
    _, d_delta_d_nu_star = gather_active_grad(aml, d_nu)
    return eta_grad_active, d_delta_d_nu_star
end

"""
    originzc_cross_fixed_contribution(base, aug, νfull) -> Vector{Float64}

OZC-CROSS analog of `originzc_fixed_contribution` (cm_originzc_moments.jl): folds the fixed
lambda*-weighted mean+cross-pair restriction contribution into `q0` for every draw `s`, extended
over the `K_pair^2` grid. The mean-block loop is byte-identical to the base function, INCLUDING its
`aml`-aware ragged-column subsetting (Variant D, 2026-08-09: previously hardcoded dense
`ncore_econ+K_mean*D`/`(k-1)*D` -- now reads `aug.aml` the same way `originzc_fixed_contribution`
does). The pair-block loop is the SAME accumulation pattern (`out .+= Zpair_k*lambda_k; out .-=
dot(target_k,lambda_k)`), just iterating over `cross_pair_level_index`'s `klin` blocks instead of
`1:K_pair` diagonal levels -- `pair_targets`'s `OriginByPowerCrossLayout` method already handles the
`(k1,k2)` decode, so the loop body itself needs no other change beyond the ragged `pair_start0`
offset (the pair block is otherwise completely unaffected by `aml`, same reasoning as
`mean_offset_from_aml`'s own docstring).
"""
function originzc_cross_fixed_contribution(base, aug, νfull::AbstractVector{Float64})
    layout = aug.layout
    layout isa OriginByPowerCrossLayout || error("originzc_cross_fixed_contribution: aug.layout must be an OriginByPowerCrossLayout, got $(typeof(layout))")
    K_mean = layout.K_mean; K_pair = layout.K_pair
    length(νfull) == n_eta(layout) || error("originzc_cross_fixed_contribution: length(νfull)=$(length(νfull)) != n_eta(layout)=$(n_eta(layout))")
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
        dense_targets_k = mean_targets(layout, νfull, k, D)
        targets_active = length(active) == D ? dense_targets_k : dense_targets_k[active]
        out .+= Zk_active * λ_mean_k
        out .-= dot(targets_active, λ_mean_k)
    end
    pair_start0 = ncore_econ + n_mean_active
    levels = cross_pair_level_index(K_pair)
    for klin in 1:length(levels)
        λ_pair_klin = @view base.λstar[pair_start0+(klin-1)*npair : pair_start0+klin*npair-1]
        out .+= aug.Zpairraw_all[klin] * λ_pair_klin
        out .-= dot(pair_targets(layout, νfull, klin, D), λ_pair_klin)
    end
    return out
end

"""
    build_lfix_base_cache_originzc_cross(x_free0, ctx_cm, base, aug, νfull; validate_dense=false) -> LFixBaseCache

OZC-CROSS analog of `build_lfix_base_cache_originzc` (cm_originzc_production.jl): folds
`originzc_cross_fixed_contribution` (not the base family's `originzc_fixed_contribution`) into `q0`.
"""
function build_lfix_base_cache_originzc_cross(x_free0::AbstractVector, ctx_cm, base::BaseDualState,
                                               aug, νfull::AbstractVector{Float64}; validate_dense::Bool = false)
    cache0 = build_lfix_base_cache(x_free0, ctx_cm, base; validate_dense = validate_dense)
    contrib0 = originzc_cross_fixed_contribution(base, aug, νfull)
    return with_q0(cache0, cache0.q0 .- contrib0)
end

"""
    cm_originzc_cross_production_gradient(x_free0, νfull, pcx, ctx, pe; base=nothing, verify=nothing,
                                           gradient_backend=:shared_inplace_pooled, econ_ws=nothing, kwargs...) -> (g_ext, meta)

OZC-CROSS analog of `cm_originzc_production_gradient` (cm_originzc_production.jl): the (g,A_od)
block via the SAME shared `economic_A_gradient!` backend (unaffected by the pair-block extension --
it only ever reads `cache.q0`, already correctly folded above), PLUS
`d_delta_dual_d_eta_origin_cross_vec` appended as the last `n_eta(layout)` components. `aml` branch
(2026-08-09, Variant D): mirrors `cm_originzc_production_gradient`'s own aml branch EXACTLY, using
`d_delta_dual_d_eta_active_and_nustar_cross` in place of the base family's
`d_delta_dual_d_eta_active_and_nustar`.
"""
function cm_originzc_cross_production_gradient(x_free0::AbstractVector, νfull::AbstractVector{Float64}, pcx, ctx, pe;
        base::Union{Nothing,BaseDualState} = nothing, verify = nothing,
        gradient_backend::Symbol = :shared_inplace_pooled,
        econ_ws::Union{Nothing,EconomicAGradientWorkspace} = nothing, kwargs...)
    if base === nothing || verify === nothing
        base, verify = archOZ_verified_state(x_free0, νfull, pcx.ctx_cm)
    end
    cache = build_lfix_base_cache_originzc_cross(x_free0, pcx.ctx_cm, base, pcx.aug, νfull)
    if gradient_backend === :shared_inplace_pooled
        D = pcx.ctx_cm.D; Ddest = hasproperty(pcx.ctx_cm, :D_dest) ? pcx.ctx_cm.D_dest : pcx.ctx_cm.D
        ws = econ_ws === nothing ? get_or_build_econ_a_grad_ws(cache.W) : econ_ws
        g_econ = zeros(D * Ddest)
        meta = economic_A_gradient!(g_econ, base, pcx.ctx_cm, pe, ws; cache = cache, kwargs...)
    elseif gradient_backend === :legacy_unbuffered
        g_econ, meta = composite_gradient_at_fast(x_free0, pcx.ctx_cm, pe; base = base, cache = cache, kwargs...)
    else
        error("cm_originzc_cross_production_gradient: gradient_backend must be :shared_inplace_pooled|:legacy_unbuffered, got $gradient_backend")
    end
    aml = hasproperty(pcx.aug, :aml) ? pcx.aug.aml : nothing   # 2026-08-09 Variant D
    if aml !== nothing && aml.active
        eta_grad_active, d_delta_d_nu_star = d_delta_dual_d_eta_active_and_nustar_cross(base.λstar, pcx.aug, aml, νfull; mean_m = verify.m_mean)
        θ_full = CS.reconstruct_full(x_free0, ctx.m)
        info = build_focal_kstar_derivative_info(ctx, pe)
        D2_econ = length(g_econ)
        apply_focal_kstar_chain_rule!(g_econ, θ_full, ctx, info, D2_econ, d_delta_d_nu_star)
        return vcat(g_econ, eta_grad_active), meta
    end
    d_eta = d_delta_dual_d_eta_origin_cross_vec(base.λstar, pcx.aug, νfull; mean_m = verify.m_mean)
    return vcat(g_econ, d_eta), meta
end
