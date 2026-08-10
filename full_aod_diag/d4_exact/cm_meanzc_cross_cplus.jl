# ============================================================================
# CM+ZC-CROSS integration onto Backend C+ (2026-08-09). Structural twin of cm_meanzc_cplus.jl (the
# base diagonal family's own C+ integration), which this mirrors function-for-function; read that
# file's header first -- its decomposition argument applies here COMPLETELY UNCHANGED, and is the
# reason this file can exist at all:
#
#   the CM-augmented base-point dual scalar
#       q_s(theta) = -zeta* - lambda_G*'G_s(theta) - lambda_C*'C_s - lambda_ZC*'Z_s
#   has a CONSTANT `lambda_C*'C_s + lambda_ZC*'Z_s` term across every outer (g,A_od) coordinate
#   probe. That is a property of the CM-grid and RESTRICTION blocks BEING FIXED during the economic-
#   coordinate probe, NOT of the restriction block's internal shape -- so it holds identically
#   whether the pair block is the base family's K_pair diagonal levels or CM+ZC-CROSS's K_pair^2
#   ordered cross grid. Consequently `composite_gradient_at_Cplus_from_cache` (lfix_cm_cplus.jl) and
#   `build_lfix_base_cache_C!` are reused here COMPLETELY UNMODIFIED, exactly as the base family
#   reuses them.
#
# WHY THIS FILE EXISTS: `cm_gradient_backend = :cplus` is the DEFAULT of
# `run_cm_upper_checkpointed` (cm_checkpoint.jl). Adding a family without a C+ gradient would leave
# it unusable in its own production driver's default configuration -- the exact mistake made and
# corrected for OZC-CROSS (see cm_originzc_cross_cplus.jl's header, and trap 4 of the CM+ZC-CROSS
# handover). This file supplies the missing piece up front rather than erroring.
#
# PURELY ADDITIVE: modifies nothing. Every function here is new. The only two things that differ
# from the base family's C+ path are (1) the fixed-contribution fold uses
# `meanzc_cross_fixed_contribution`, and (2) the appended eta block uses
# `d_delta_dual_d_eta_nu_cross_vec` / `d_delta_dual_d_eta_active_and_nustar_shared_cross` -- i.e.
# exactly the same two substitutions that distinguish `cm_meanzc_cross_production_gradient` from
# `cm_meanzc_production_gradient` on the non-C+ path.
# ============================================================================

isdefined(Main, :build_lfix_base_cache_cm_meanzc_C!) || include(joinpath(@__DIR__, "cm_meanzc_cplus.jl"))
isdefined(Main, :meanzc_cross_fixed_contribution) || include(joinpath(@__DIR__, "cm_meanzc_cross_production.jl"))

"""
    build_lfix_base_cache_cm_meanzc_cross_C!(ws, x_free0, ctx_cm, base, ctx, aug, bins, νvec; validate_dense=false) -> LFixBaseCacheC

CM+ZC-CROSS analog of `build_lfix_base_cache_cm_meanzc_C!` (cm_meanzc_cplus.jl). Same contract
(`ctx_cm.obj === aug.obj_cm`), calls the SAME UNMODIFIED `build_lfix_base_cache_C!`, folds in the
UNCHANGED `cm_fixed_contribution_meanzc_layout` (CM-grid block) and `meanzc_cross_fixed_contribution`
(the K_pair^2 cross-grid mean/pair fold) instead of the base family's diagonal-only
`meanzc_fixed_contribution`. That fold is itself already aml-aware, so Variant D needs no special
handling here.
"""
function build_lfix_base_cache_cm_meanzc_cross_C!(ws::LFixFactorizedWorkspace, x_free0::AbstractVector, ctx_cm,
                                                   base::BaseDualState, ctx, aug, bins::AbstractMatrix{<:Unsigned},
                                                   νvec::AbstractVector{Float64}; validate_dense::Bool = false)
    cache0 = build_lfix_base_cache_C!(ws, x_free0, ctx_cm, base; validate_dense = validate_dense)
    cm_contrib0 = cm_fixed_contribution_meanzc_layout(base, ctx, aug, bins)
    meanzc_contrib0 = meanzc_cross_fixed_contribution(base, aug, νvec)
    return with_q0_C(cache0, cache0.q0 .- cm_contrib0 .- meanzc_contrib0)
end

"""
    composite_gradient_at_Cplus_cm_meanzc_cross(x_free0, νvec, ctx_cm, pe, ctx, aug, bins, pool, ws, cctx; base=nothing, cache=nothing, kwargs...) -> (g, meta)

CM+ZC-CROSS analog of `composite_gradient_at_Cplus_cm_meanzc` (cm_meanzc_cplus.jl): Backend C+'s
(g,A_od) block only. Does NOT append the eta_nu coordinates -- see
`cm_meanzc_cross_production_gradient_cplus` below for the full outer vector.
"""
function composite_gradient_at_Cplus_cm_meanzc_cross(x_free0::AbstractVector, νvec::AbstractVector{Float64}, ctx_cm, pe,
        ctx, aug, bins::AbstractMatrix{<:Unsigned}, pool::GradWorkspacePool, ws::LFixFactorizedWorkspace, cctx;
        base::Union{Nothing,BaseDualState} = nothing, cache::Union{Nothing,LFixBaseCacheC} = nothing,
        validate_dense::Bool = false, kwargs...)
    base = base === nothing ? archC_meanzc_base_state(x_free0, νvec, ctx_cm, cctx) : base
    if cache === nothing
        cache = build_lfix_base_cache_cm_meanzc_cross_C!(ws, x_free0, ctx_cm, base, ctx, aug, bins, νvec; validate_dense = validate_dense)
    end
    return composite_gradient_at_Cplus_from_cache(x_free0, ctx_cm, pe, pool, cache; base = base, kwargs...)
end

"""
    cm_meanzc_cross_production_gradient_cplus(x_free0, νvec, pcx, ctx, pe, pool, ws; base=nothing, verify=nothing, kwargs...) -> (g_ext, meta)

CM+ZC-CROSS analog of `cm_meanzc_production_gradient_cplus` (cm_meanzc_cplus.jl), identical in
structure with exactly the two substitutions named in this file's header. Returns
`vcat(g_econ, eta_block)`, where the eta block is `aml.n_eta_active` long when Variant D is active
and `n_eta(layout) = K_mean` otherwise -- the SAME contract `cm_meanzc_cross_production_gradient`
already satisfies, so the driver's `cb_G!` needs no shape change when switching backends.
"""
function cm_meanzc_cross_production_gradient_cplus(x_free0::AbstractVector, νvec::AbstractVector{Float64}, pcx, ctx, pe,
        pool::GradWorkspacePool, ws::LFixFactorizedWorkspace; base::Union{Nothing,BaseDualState} = nothing,
        verify = nothing, kwargs...)
    if base === nothing || verify === nothing
        base, verify = archC_meanzc_verified_state(x_free0, νvec, pcx.ctx_cm, pcx.cctx)
    end
    cache = build_lfix_base_cache_cm_meanzc_cross_C!(ws, x_free0, pcx.ctx_cm, base, ctx, pcx.aug, pcx.bins, νvec)
    g_econ, meta = composite_gradient_at_Cplus_from_cache(x_free0, pcx.ctx_cm, pe, pool, cache; base = base, kwargs...)
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
