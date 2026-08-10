# ============================================================================
# OZC-CROSS integration onto Backend C+ (2026-08-09). Structural twin of
# cm_originzc_cplus.jl (the base diagonal family's own C+ integration), which this mirrors
# function-for-function; read that file's header first -- its decomposition argument applies here
# COMPLETELY UNCHANGED, and is the reason this file can exist at all:
#
#   the augmented base-point dual scalar q_s(theta) = -zeta* - lambda_G*'G_s(theta) - lambda_ZC*'Z_s
#   has a CONSTANT `lambda_ZC*'Z_s` term across every outer (g,A_od) coordinate probe. That is a
#   property of the RESTRICTION BLOCK BEING FIXED during the economic-coordinate probe, NOT of the
#   restriction block's internal shape -- so it holds identically whether the pair block is the base
#   family's K_pair diagonal levels or OZC-CROSS's K_pair^2 ordered cross grid. Consequently
#   `composite_gradient_at_Cplus_from_cache` (lfix_cm_cplus.jl) and `build_lfix_base_cache_C!` are
#   reused here COMPLETELY UNMODIFIED, exactly as the base family reuses them.
#
# WHY THIS FILE EXISTS: `cm_gradient_backend = :cplus` is the DEFAULT of
# `run_originzc_upper_checkpointed` (cm_originzc_checkpoint.jl). An earlier draft of the OZC-CROSS
# driver wiring made that default HARD-ERROR for OriginByPowerCrossLayout, on the (correct) grounds
# that `cm_originzc_production_gradient_cplus` calls the base family's diagonal-only
# `d_delta_dual_d_eta_origin_vec`/`originzc_fixed_contribution` internally and would therefore have
# silently computed the WRONG gradient for the cross grid. Erroring was the right call versus
# silently-wrong, but it left the family unusable through its own production driver's default
# configuration. This file supplies the missing piece instead of refusing.
#
# PURELY ADDITIVE: modifies nothing. Every function here is new. The only two things that differ
# from the base family's C+ path are (1) the fixed-contribution fold uses
# `originzc_cross_fixed_contribution`, and (2) the appended eta block uses
# `d_delta_dual_d_eta_origin_cross_vec` / `d_delta_dual_d_eta_active_and_nustar_cross` -- i.e.
# exactly the same two substitutions that distinguish `cm_originzc_cross_production_gradient` from
# `cm_originzc_production_gradient` on the non-C+ path (already verified this session: D4 residuals
# ~1e-12..1e-15, D20 W=100k fixed-contribution fold vs independent operator recompute at machine
# precision, analytic-vs-FD eta gradient matching the base family's own FD agreement).
# ============================================================================

isdefined(Main, :build_lfix_base_cache_originzc_C!) || include(joinpath(@__DIR__, "cm_originzc_cplus.jl"))
isdefined(Main, :originzc_cross_fixed_contribution) || include(joinpath(@__DIR__, "cm_originzc_cross_production.jl"))

"""
    build_lfix_base_cache_originzc_cross_C!(ws, x_free0, ctx_cm, base, aug, νfull; validate_dense=false) -> LFixBaseCacheC

OZC-CROSS analog of `build_lfix_base_cache_originzc_C!` (cm_originzc_cplus.jl). Same contract
(`ctx_cm.obj === aug.obj_cm`), calls the SAME UNMODIFIED `build_lfix_base_cache_C!`, and folds in
`originzc_cross_fixed_contribution` (the K_pair^2 cross-grid fold) instead of the base family's
diagonal-only `originzc_fixed_contribution`. That fold is itself already aml-aware, so Variant D
needs no special handling here.
"""
function build_lfix_base_cache_originzc_cross_C!(ws::LFixFactorizedWorkspace, x_free0::AbstractVector, ctx_cm,
                                                  base::BaseDualState, aug, νfull::AbstractVector{Float64};
                                                  validate_dense::Bool = false)
    cache0 = build_lfix_base_cache_C!(ws, x_free0, ctx_cm, base; validate_dense = validate_dense)
    contrib0 = originzc_cross_fixed_contribution(base, aug, νfull)
    return with_q0_C(cache0, cache0.q0 .- contrib0)
end

"""
    cm_originzc_cross_production_gradient_cplus(x_free0, νfull, pcx, ctx, pe, pool, ws;
                                                 base=nothing, verify=nothing, kwargs...) -> (g_ext, meta)

OZC-CROSS analog of `cm_originzc_production_gradient_cplus` (cm_originzc_cplus.jl), byte-for-byte
identical in structure with exactly two substitutions (the cross cache builder above, and the cross
eta-gradient functions). Returns `vcat(g_econ, eta_block)`, where the eta block is
`aml.n_eta_active` long when Variant D is active and `n_eta(layout)` long otherwise -- the SAME
contract `cm_originzc_cross_production_gradient` already satisfies, so the driver's `cb_G!` needs no
shape change when switching backends.
"""
function cm_originzc_cross_production_gradient_cplus(x_free0::AbstractVector, νfull::AbstractVector{Float64}, pcx, ctx, pe,
        pool::GradWorkspacePool, ws::LFixFactorizedWorkspace; base::Union{Nothing,BaseDualState} = nothing,
        verify = nothing, kwargs...)
    if base === nothing || verify === nothing
        base, verify = archOZ_verified_state(x_free0, νfull, pcx.ctx_cm)
    end
    cache = build_lfix_base_cache_originzc_cross_C!(ws, x_free0, pcx.ctx_cm, base, pcx.aug, νfull)
    g_econ, meta = composite_gradient_at_Cplus_from_cache(x_free0, pcx.ctx_cm, pe, pool, cache; base = base, kwargs...)
    aml = hasproperty(pcx.aug, :aml) ? pcx.aug.aml : nothing
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
