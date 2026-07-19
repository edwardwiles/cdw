# ============================================================================
# Continuation 9, Phase 1 integration: real-data D=20 context builder for the
# full-A_od diagnostic machinery. Additive only -- does NOT modify
# context.jl's d4_exact_setup or context_scaled.jl's d_exact_setup_scaled
# (both remain synthetic-only, unchanged). Mirrors context_scaled.jl's
# build_ad_context_scaled / d_exact_setup_scaled pattern exactly, but forces
# fakeData=3 (real data, see setup/importData.jl) and DFake=20 instead of
# accepting an arbitrary synthetic D.
#
# Real-data support (setup/importData.jl's fakeData==3 branch,
# setup/defineCounter.jl's autarky tau^Inf fix) was ported into this worktree
# from trade_robustness_modular_perf @ 778c362 (feature/sequential-inversion-perf)
# -- see docs/fullA_D20_production_path_audit.md for the audit that found this
# worktree previously had NO real-data loading code at all. Data files:
# real_data/noah_D20/{countries,L,pi,tau}.csv, copied verbatim (md5-verified)
# from that worktree.
#
# Focal country: France (baseIndex=2), matches AD_PARAMS's existing default --
# no override needed. sigma=2.5 (assumed), theta estimated via gravity
# (thetaHat=0) -- both already AD_PARAMS defaults, unchanged.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))   # -> AD_PARAMS, build_ad_context, master_setup etc., CS, d4_exact_setup

const D20_REAL = 20

"Build (so, pp, params_used) for the REAL D=20 economy at a given W, independent of AD_PARAMS."
function build_ad_context_real_d20(; W::Int)
    params = merge(AD_PARAMS, (fakeData = 3, DFake = D20_REAL, W = W, Jac_W = W))
    so = master_setup(params)
    @assert so.D == D20_REAL "master_setup returned D=$(so.D), expected $(D20_REAL) -- real_data/noah_D20 CSVs may be malformed"
    up = (; params..., D = so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
    checkParams(up)
    ps = master_prestep(so.data, so.counters, up)
    pp = master_prepare_cc(so.data, so.counters, ps, up)
    return so, pp, params
end

"""
    d20_real_setup(; W, δ=1.0, find_smallest=true) -> NamedTuple

Mirrors `context_scaled.jl::d_exact_setup_scaled` exactly (same free-parameter
layout, bounds construction, PsiObjectiveBundleImplicit wiring) but against the
REAL D=20 dataset (France focal) instead of a synthetic economy at an arbitrary
D. Returns the same field set so every existing D=4 diagnostic function
(`evaluate_fullA`, `compute_winners`, `build_pivot_elimination`, etc.) works
unchanged on the returned `ctx`.
"""
function d20_real_setup(; W::Int, δ::Float64 = 1.0, find_smallest::Bool = true,
        outer_loop_opt::AbstractString = joinpath(D4X_ROOT, "full_aod_diag", "csw_outer_25.opt"),
        inner_loop_opt::AbstractString = joinpath(D4X_ROOT, "full_aod_diag", "ek_inner.opt"),
        needs_outer_moment_jacobian::Bool = true)
    so, pp, params_used = build_ad_context_real_d20(W = W)
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
