# ============================================================================
# Origin-specific-ZC (no CM) production integration onto CM-C+, 2026-07-23.
#
# Same decomposition argument lfix_cm_cplus.jl's/cm_meanzc_cplus.jl's own
# headers prove applies here unchanged: the augmented base-point dual scalar
#     q_s(theta) = -zeta* - lambda_G*'G_s(theta) - lambda_originzc*'Z_s
# has a CONSTANT `lambda_originzc*'Z_s` term across every outer (g,A_od)
# coordinate probe (originzc_fixed_contribution, cm_originzc_production.jl,
# already-proven result reused verbatim). Backend C+'s
# build_lfix_base_cache_C!/composite_gradient_at_Cplus_from_cache satisfy the
# same two conditions regardless of what sits after the economic block, so
# composite_gradient_at_Cplus_from_cache (lfix_cm_cplus.jl) is reused here
# COMPLETELY UNMODIFIED.
#
# PURELY ADDITIVE: does not modify lfix_cm_cplus.jl, cm_originzc_production.jl,
# cm_originzc_moments.jl, lfix_cm_aware.jl, lfix_factorized.jl,
# lfix_factorized_workspace.jl, or cm_production_bundle.jl. Every function
# here is new.
# ============================================================================

"""
    build_lfix_base_cache_originzc_C!(ws, x_free0, ctx_cm, base, aug, νfull; validate_dense=false) -> LFixBaseCacheC

No-CM analog of `build_lfix_base_cache_cm_meanzc_C!`, for Backend C+. Same
contract: `ctx_cm.obj === aug.obj_cm`. Calls `build_lfix_base_cache_C!`
UNCHANGED, then folds in ONLY `originzc_fixed_contribution` (no CM-grid
block for this arm).
"""
function build_lfix_base_cache_originzc_C!(ws::LFixFactorizedWorkspace, x_free0::AbstractVector, ctx_cm,
                                            base::BaseDualState, aug, νfull::AbstractVector{Float64};
                                            validate_dense::Bool = false)
    cache0 = build_lfix_base_cache_C!(ws, x_free0, ctx_cm, base; validate_dense = validate_dense)
    contrib0 = originzc_fixed_contribution(base, aug, νfull)
    return with_q0_C(cache0, cache0.q0 .- contrib0)
end

"""
    composite_gradient_at_Cplus_originzc(x_free0, νfull, ctx_cm, pe, aug, pool, ws; base=nothing, cache=nothing, kwargs...) -> (g, meta)

Backend C+'s (g,A_od) block only, structural twin of
`composite_gradient_at_Cplus_cm_meanzc`. Does NOT append the eta coordinates
-- see `cm_originzc_production_gradient_cplus` below for the full outer
vector.
"""
function composite_gradient_at_Cplus_originzc(x_free0::AbstractVector, νfull::AbstractVector{Float64}, ctx_cm, pe,
        aug, pool::GradWorkspacePool, ws::LFixFactorizedWorkspace;
        base::Union{Nothing,BaseDualState} = nothing, cache::Union{Nothing,LFixBaseCacheC} = nothing,
        validate_dense::Bool = false, kwargs...)
    base = base === nothing ? archOZ_base_state(x_free0, νfull, ctx_cm) : base
    if cache === nothing
        cache = build_lfix_base_cache_originzc_C!(ws, x_free0, ctx_cm, base, aug, νfull; validate_dense = validate_dense)
    end
    return composite_gradient_at_Cplus_from_cache(x_free0, ctx_cm, pe, pool, cache; base = base, kwargs...)
end

"""
    cm_originzc_production_gradient_cplus(x_free0, νfull, pcx, ctx, pe, pool, ws; base=nothing, verify=nothing, kwargs...) -> (g_ext, meta)

`:cplus`-backend analog of `cm_originzc_production_gradient`. `g_ext` has
length `D^2 + n_eta(layout)`.
"""
function cm_originzc_production_gradient_cplus(x_free0::AbstractVector, νfull::AbstractVector{Float64}, pcx, ctx, pe,
        pool::GradWorkspacePool, ws::LFixFactorizedWorkspace; base::Union{Nothing,BaseDualState} = nothing,
        verify = nothing, kwargs...)
    if base === nothing || verify === nothing
        base, verify = archOZ_verified_state(x_free0, νfull, pcx.ctx_cm)
    end
    cache = build_lfix_base_cache_originzc_C!(ws, x_free0, pcx.ctx_cm, base, pcx.aug, νfull)
    g_econ, meta = composite_gradient_at_Cplus_from_cache(x_free0, pcx.ctx_cm, pe, pool, cache; base = base, kwargs...)
    aml = hasproperty(pcx.aug, :aml) ? pcx.aug.aml : nothing   # fix/zc-profile-focal-sigmaminus1-mean-2026-08-07
    if aml !== nothing && aml.active
        eta_grad_active, d_delta_d_nu_star = d_delta_dual_d_eta_active_and_nustar(base.λstar, pcx.aug, aml, νfull; mean_m = verify.m_mean)
        θ_full = CS.reconstruct_full(x_free0, ctx.m)
        info = build_focal_kstar_derivative_info(ctx, pe)
        D2_econ = length(g_econ)
        apply_focal_kstar_chain_rule!(g_econ, θ_full, ctx, info, D2_econ, d_delta_d_nu_star)
        return vcat(g_econ, eta_grad_active), meta
    end
    d_eta = d_delta_dual_d_eta_origin_vec(base.λstar, pcx.aug, νfull; mean_m = verify.m_mean)
    return vcat(g_econ, d_eta), meta
end
