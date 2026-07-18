# ============================================================================
# Phase 1C: D/W-parametrized context builder for the baseline scaling
# benchmark. Additive only -- does NOT modify setup_context.jl's AD_PARAMS
# constant or context.jl's d4_exact_setup (both remain the single source of
# truth for the actual D=4/W=8000 investigation). Reuses the exact same
# underlying machinery (master_setup/master_prestep/master_prepare_cc,
# build_theta_gammanorm, theoretical_gammaprime_bounds, FreeParamMap,
# PsiObjectiveBundleImplicit) with a LOCAL, per-call override of DFake/W/
# Jac_W merged into a copy of AD_PARAMS -- confirmed working at D=6 by direct
# probe before writing this file (nTotalMoments 18->38 D=4->D=6, no errors).
#
# COMPUTATIONAL BENCHMARK ONLY, per the task's explicit instruction: "Never
# compare the economic bound across different D as though it were the same
# economy" -- these are structurally-different synthetic economies (fresh
# random draws/parameters at each D), used here ONLY to measure cost scaling,
# never to compare kappa/Delta across D.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))   # -> AD_PARAMS, build_ad_context, master_setup etc., CS, d4_exact_setup

"Build (so, pp, params_used) at an OVERRIDDEN (D, W), independent of the module-level AD_PARAMS constant."
function build_ad_context_scaled(; D::Int, W::Int)
    params = merge(AD_PARAMS, (DFake = D, W = W, Jac_W = W))
    so = master_setup(params)
    up = (; params..., D = so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
    checkParams(up)
    ps = master_prestep(so.data, so.counters, up)
    pp = master_prepare_cc(so.data, so.counters, ps, up)
    return so, pp, params
end

"""
    d_exact_setup_scaled(; D, W, δ=1.0, find_smallest=true) -> NamedTuple

Mirrors `context.jl::d4_exact_setup` exactly (same free-parameter layout,
bounds construction, PsiObjectiveBundleImplicit wiring) but at an arbitrary
(D, W) instead of the fixed D=4/W=8000. Returns the same field set so every
existing D=4 diagnostic function (`evaluate_fullA`, `compute_winners`,
`build_pivot_elimination`, etc.) works unchanged on the returned `ctx`.
"""
function d_exact_setup_scaled(; D::Int, W::Int, δ::Float64 = 1.0, find_smallest::Bool = true,
        outer_loop_opt::AbstractString = joinpath(D4X_ROOT, "full_aod_diag", "csw_outer_25.opt"),
        inner_loop_opt::AbstractString = joinpath(D4X_ROOT, "full_aod_diag", "ek_inner.opt"),
        needs_outer_moment_jacobian::Bool = true)
    so, pp, params_used = build_ad_context_scaled(D = D, W = W)
    Dact = so.D; bi = params_used.baseIndex; σ = params_used.σHat; μHat = pp.γ.μHat
    @unpack θ_initial, θ_initial_up, U, γ, outer_constr_index, nTotalMoments, complement_index, inequality_index = pp
    Aod_offset = 3 + Dact

    θ0_up = build_theta_gammanorm(θ_initial_up, Dact, bi, μHat, σ)
    bounds = theoretical_gammaprime_bounds(γ, σ)
    θ0_up[3+Dact] = clamp(θ0_up[3+Dact], bounds.γp_lo, bounds.γp_hi)

    θ_lo = (θ0_up .* 0.0001)[:]; θ_hi = (θ0_up .* 10000)[:]
    θ_lo[2] = θ0_up[2]; θ_hi[2] = θ0_up[2]
    θ_lo[1] = θ0_up[1]; θ_hi[1] = θ0_up[1]
    for d in 1:Dact
        θ_lo[2+d] = θ0_up[2+d]; θ_hi[2+d] = θ0_up[2+d]
    end
    θ_lo[3+Dact] = bounds.γp_lo; θ_hi[3+Dact] = bounds.γp_hi

    l_full = length(θ0_up)
    free_idx = vcat(3 + Dact, collect(Aod_offset+1:Aod_offset+Dact^2))
    fixed_idx = vcat(1, 2, collect(3:2+Dact))
    fixed_vals = θ0_up[fixed_idx]
    m = CS.FreeParamMap(l_full, free_idx, fixed_idx, fixed_vals)
    @assert CS.n_free(m) == 1 + Dact^2

    Aod_free_pos = [1 + (d - 1) * Dact + o for o in 1:Dact, d in 1:Dact]
    τ = γ.τ
    q_tilde, N_obs = precompute_q_tilde(τ)

    obj = CS.PsiObjectiveBundleImplicit(δ = δ, find_smallest = find_smallest, γ = γ,
        (moments!) = EK_moments_gammanorm_directgp!, moments_jacobian! = error, d = nTotalMoments,
        outer_constr_index = outer_constr_index, inequality_index = inequality_index,
        complement_index = complement_index, l = l_full, U = U, N = params_used.Jac_W,
        lower_limit = -50, use_cached_x = true,
        outer_loop_opt = outer_loop_opt, inner_loop_opt = inner_loop_opt,
        needs_outer_moment_jacobian = needs_outer_moment_jacobian)
    @assert obj.outer_constr_index == obj.d

    return (so = so, pp = pp, D = Dact, W = W, bi = bi, σ = σ, μHat = μHat, γ = γ, U = U,
            θ0_up = θ0_up, θ_lo = θ_lo, θ_hi = θ_hi, l_full = l_full,
            free_idx = free_idx, fixed_idx = fixed_idx, fixed_vals = fixed_vals, m = m,
            Aod_offset = Aod_offset, Aod_free_pos = Aod_free_pos,
            τ = τ, q_tilde = q_tilde, N_obs = N_obs, obj = obj,
            nTotalMoments = nTotalMoments, outer_constr_index = outer_constr_index,
            bounds = bounds, δ = δ, find_smallest = find_smallest)
end
