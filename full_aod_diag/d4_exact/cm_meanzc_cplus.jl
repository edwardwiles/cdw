# ============================================================================
# CM+moments(+ZC) production integration onto CM-C+, 2026-07-23.
#
# CM-aware C+ analog of lfix_cm_cplus.jl, generalized to the meanzc extension
# family. Exactly the same decomposition argument lfix_cm_cplus.jl's own
# header proves for plain CM applies here unchanged: the CM-augmented
# base-point dual scalar
#     q_s(theta) = -zeta* - lambda_G*'G_s(theta) - lambda_C*'C_s - lambda_meanzc*'Z_s
# has a CONSTANT `lambda_C*'C_s + lambda_meanzc*'Z_s` term across every outer
# (g,A_od) coordinate probe (cm_meanzc_production.jl's own already-proven
# result -- meanzc_fixed_contribution/cm_fixed_contribution_meanzc_layout are
# reused verbatim, not re-derived). Backend C+'s
# build_lfix_base_cache_C!/composite_gradient_at_Cplus_from_cache satisfy the
# same two conditions lfix_cm_cplus.jl's header states (only ever index
# base.lambda*[1:D^2] for the economic core columns, identical counterfactual-
# column tail check) regardless of what sits after the economic block, so
# composite_gradient_at_Cplus_from_cache (lfix_cm_cplus.jl) is reused here
# COMPLETELY UNMODIFIED -- it never needs to know a mean/ZC/CM block exists at
# all, only that q0 was folded correctly before it was called.
#
# PURELY ADDITIVE: does not modify lfix_cm_cplus.jl, cm_meanzc_production.jl,
# lfix_cm_aware.jl, lfix_factorized.jl, lfix_factorized_workspace.jl, or
# cm_production_bundle.jl. Every function here is new.
# `cm_gradient_backend=:reference` (unchanged default meaning for this
# extension too) never reaches any function defined in this file.
#
# Expected already included by caller: gradient_workspace.jl,
# lfix_factorized_workspace.jl, lfix_cm_cplus.jl (with_q0_C,
# composite_gradient_at_Cplus_from_cache), cm_meanzc_moments.jl,
# cm_meanzc_production.jl (archC_meanzc_base_state/verified_state,
# meanzc_fixed_contribution, cm_fixed_contribution_meanzc_layout,
# d_delta_dual_d_eta_nu_vec).
# ============================================================================

"""
    build_lfix_base_cache_cm_meanzc_C!(ws, x_free0, ctx_cm, base, ctx, aug, bins, νvec; validate_dense=false) -> LFixBaseCacheC

CM+moments(+ZC)-aware analog of `build_lfix_base_cache_cm_meanzc`
(cm_meanzc_production.jl), for Backend C+. Same contract:
`ctx_cm.obj === aug.obj_cm` (the meanzc-augmented objective `base` was
actually solved against). Calls `build_lfix_base_cache_C!` UNCHANGED against
`ctx_cm`, then folds in BOTH `cm_fixed_contribution_meanzc_layout` (the
CM-grid block, re-sliced at its position under the meanzc column layout) AND
`meanzc_fixed_contribution` (the mean/pair blocks at every level) -- the SAME
two fixed-contribution functions the `:reference` meanzc path already uses,
one shared CM+meanzc-algebra implementation for both backends.
"""
function build_lfix_base_cache_cm_meanzc_C!(ws::LFixFactorizedWorkspace, x_free0::AbstractVector, ctx_cm,
                                             base::BaseDualState, ctx, aug, bins::AbstractMatrix{<:Unsigned},
                                             νvec::AbstractVector{Float64}; validate_dense::Bool = false)
    cache0 = build_lfix_base_cache_C!(ws, x_free0, ctx_cm, base; validate_dense = validate_dense)
    cm_contrib0 = cm_fixed_contribution_meanzc_layout(base, ctx, aug, bins)
    meanzc_contrib0 = meanzc_fixed_contribution(base, aug, νvec)
    return with_q0_C(cache0, cache0.q0 .- cm_contrib0 .- meanzc_contrib0)
end

"""
    composite_gradient_at_Cplus_cm_meanzc(x_free0, νvec, ctx_cm, pe, ctx, aug, bins, pool, ws, cctx; base=nothing, cache=nothing, kwargs...) -> (g, meta)

CM+moments(+ZC)-aware entry point for Backend C+'s (g,A_od) block only,
structural twin of `composite_gradient_at_Cplus_cm` (lfix_cm_cplus.jl) but
threading `νvec` through to `archC_meanzc_base_state`/
`build_lfix_base_cache_cm_meanzc_C!`. Does NOT append the eta_nu coordinates
-- see `cm_meanzc_production_gradient_cplus` below for the full outer vector.
"""
function composite_gradient_at_Cplus_cm_meanzc(x_free0::AbstractVector, νvec::AbstractVector{Float64}, ctx_cm, pe,
        ctx, aug, bins::AbstractMatrix{<:Unsigned}, pool::GradWorkspacePool, ws::LFixFactorizedWorkspace, cctx;
        base::Union{Nothing,BaseDualState} = nothing, cache::Union{Nothing,LFixBaseCacheC} = nothing,
        validate_dense::Bool = false, kwargs...)
    base = base === nothing ? archC_meanzc_base_state(x_free0, νvec, ctx_cm, cctx) : base
    if cache === nothing
        cache = build_lfix_base_cache_cm_meanzc_C!(ws, x_free0, ctx_cm, base, ctx, aug, bins, νvec; validate_dense = validate_dense)
    end
    return composite_gradient_at_Cplus_from_cache(x_free0, ctx_cm, pe, pool, cache; base = base, kwargs...)
end

"""
    cm_meanzc_production_gradient_cplus(x_free0, νvec, pcx, ctx, pe, pool, ws; base=nothing, verify=nothing, kwargs...) -> (g_ext, meta)

`:cplus`-backend analog of `cm_meanzc_production_gradient`
(cm_meanzc_production.jl), exact structural twin of
`cm_production_gradient_cplus` (lfix_cm_cplus.jl): the (g,A_od) block via
`composite_gradient_at_Cplus_from_cache` on a cache built from the
meanzc-augmented base state, PLUS the analytic `∂Delta_dual/∂η_{ν,k}` vector
(`d_delta_dual_d_eta_nu_vec`, cm_meanzc_moments.jl -- backend-independent by
construction, since it depends only on `base.λstar`/`verify.m_mean`, not on
which outer-gradient backend produced the (g,A_od) block) appended as the
last `K_mean` components, identical in formula to the `:reference` path's own
`cm_meanzc_production_gradient`. `g_ext` has length `D^2 + K_mean`. If `verify`
is not supplied, one extra verified inner solve is performed to get `m_mean`
(matching `cm_meanzc_production_gradient`'s own contract).
"""
function cm_meanzc_production_gradient_cplus(x_free0::AbstractVector, νvec::AbstractVector{Float64}, pcx, ctx, pe,
        pool::GradWorkspacePool, ws::LFixFactorizedWorkspace; base::Union{Nothing,BaseDualState} = nothing,
        verify = nothing, kwargs...)
    if base === nothing || verify === nothing
        base, verify = archC_meanzc_verified_state(x_free0, νvec, pcx.ctx_cm, pcx.cctx)
    end
    cache = build_lfix_base_cache_cm_meanzc_C!(ws, x_free0, pcx.ctx_cm, base, ctx, pcx.aug, pcx.bins, νvec)
    g_econ, meta = composite_gradient_at_Cplus_from_cache(x_free0, pcx.ctx_cm, pe, pool, cache; base = base, kwargs...)
    d_eta = d_delta_dual_d_eta_nu_vec(base.λstar, pcx.aug, νvec; mean_m = verify.m_mean)
    return vcat(g_econ, d_eta), meta
end
