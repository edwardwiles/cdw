# Investigate (2026-08-09, user question): does the K=3 (A) analytic-vs-FD gap shrink as W grows?
# d4_exact_setup() hardcodes W=8000 (AD_PARAMS.W/.Jac_W, full_aod_diag/ad_benchmark/setup_context.jl)
# with no W kwarg -- this script mirrors build_ad_context()/d4_exact_setup()'s own body exactly,
# substituting a larger W into AD_PARAMS via NamedTuple override, calling the SAME
# master_setup/master_prestep/master_prepare_cc pipeline unchanged. This is a pure re-draw-at-
# larger-W test (same D=4 economy, same mu/sigma/calibration point), isolating whether the K=3 (A)
# gap is a finite-W/FD-conditioning artifact (should shrink) or a real formula bug (would not).
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl"]
    include(joinpath(D4X, f))
end
using SpecialFunctions: gamma

"""
    d4_exact_setup_bigW(W; kwargs...) -> NamedTuple

Mirrors d4_exact_setup()'s body EXACTLY (full_aod_diag/d4_exact/context.jl), except AD_PARAMS'
W/Jac_W are overridden to `W` before calling master_setup/master_prestep/master_prepare_cc (the
SAME three functions build_ad_context() calls, unchanged) -- everything downstream (theta layout,
bounds, free_idx, obj construction) is byte-identical to d4_exact_setup, just at a different draw
count.
"""
function d4_exact_setup_bigW(W::Int; δ::Float64 = 1.0, find_smallest::Bool = true,
                              outer_loop_opt::AbstractString = joinpath(D4X_ROOT, "full_aod_diag", "csw_outer_25.opt"),
                              inner_loop_opt::AbstractString = joinpath(D4X_ROOT, "full_aod_diag", "ek_inner.opt"),
                              needs_outer_moment_jacobian::Bool = true)
    AD_PARAMS_W = (; AD_PARAMS..., W = W, Jac_W = W)
    so = master_setup(AD_PARAMS_W)
    up = (; AD_PARAMS_W..., D = so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
    checkParams(up)
    ps = master_prestep(so.data, so.counters, up)
    pp = master_prepare_cc(so.data, so.counters, ps, up)

    D = so.D; bi = AD_PARAMS_W.baseIndex; σ = AD_PARAMS_W.σHat; μHat = pp.γ.μHat
    @unpack θ_initial, θ_initial_up, U, γ, outer_constr_index, nTotalMoments, complement_index, inequality_index = pp
    Aod_offset = 3 + D
    θ0_up = build_theta_gammanorm(θ_initial_up, D, bi, μHat, σ)
    bounds = theoretical_gammaprime_bounds(γ, σ)
    θ0_up[3+D] = clamp(θ0_up[3+D], bounds.γp_lo, bounds.γp_hi)
    θ_lo = (θ0_up .* 0.0001)[:]; θ_hi = (θ0_up .* 10000)[:]
    θ_lo[2] = θ0_up[2]; θ_hi[2] = θ0_up[2]
    θ_lo[1] = θ0_up[1]; θ_hi[1] = θ0_up[1]
    for d in 1:D
        θ_lo[2+d] = θ0_up[2+d]; θ_hi[2+d] = θ0_up[2+d]
    end
    θ_lo[3+D] = bounds.γp_lo; θ_hi[3+D] = bounds.γp_hi
    l_full = length(θ0_up)
    free_idx = vcat(3 + D, collect(Aod_offset+1:Aod_offset+D^2))
    fixed_idx = vcat(1, 2, collect(3:2+D))
    fixed_vals = θ0_up[fixed_idx]
    m = CS.FreeParamMap(l_full, free_idx, fixed_idx, fixed_vals)
    Aod_free_pos = [1 + (d - 1) * D + o for o in 1:D, d in 1:D]
    τ = γ.τ
    q_tilde, N_obs = precompute_q_tilde(τ)
    obj = CS.PsiObjectiveBundleImplicit(δ = δ, find_smallest = find_smallest, γ = γ,
        (moments!) = EK_moments_gammanorm_directgp!, moments_jacobian! = error, d = nTotalMoments,
        outer_constr_index = outer_constr_index, inequality_index = inequality_index,
        complement_index = complement_index, l = l_full, U = U, N = AD_PARAMS_W.Jac_W,
        lower_limit = -50, use_cached_x = true,
        outer_loop_opt = outer_loop_opt, inner_loop_opt = inner_loop_opt,
        needs_outer_moment_jacobian = needs_outer_moment_jacobian)
    return (so = so, pp = pp, D = D, bi = bi, σ = σ, μHat = μHat, γ = γ, U = U,
            θ0_up = θ0_up, θ_lo = θ_lo, θ_hi = θ_hi, l_full = l_full,
            free_idx = free_idx, fixed_idx = fixed_idx, fixed_vals = fixed_vals, m = m,
            Aod_offset = Aod_offset, Aod_free_pos = Aod_free_pos, obj = obj)
end

K_mean = 3; K_pair = 3
for W in [8000, 40000, 160000]
    println("\n==== D4 OZC-CROSS K=$K_mean/$K_pair at W=$W ====")
    ctx = d4_exact_setup_bigW(W; find_smallest = true, needs_outer_moment_jacobian = false)
    D = ctx.D
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    nu0_origin(K::Int, D::Int) = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K]...)
    νfull0 = nu0_origin(K_mean, D)
    layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
    pcx = build_originzc_cross_production_context(ctx, CS, layout)
    base, verify = archOZ_verified_state(x_free_calib, νfull0, pcx.ctx_cm)
    println("W=$W: inner_status=$(verify.inner_status) Delta_dual=$(verify.Delta_dual) max|lambda|=$(maximum(abs.(base.λstar)))")

    g_analytic = d_delta_dual_d_eta_origin_cross_vec(base.λstar, pcx.aug, νfull0; mean_m = verify.m_mean)
    g_fd = d_delta_dual_d_eta_origin_fd(x_free_calib, νfull0, pcx.ctx_cm; h = 1e-4)
    d_ag = maximum(abs.(g_analytic .- g_fd))
    rel = d_ag / max(1e-8, maximum(abs.(g_fd)))
    println("W=$W: (A) max abs diff=$d_ag  max rel diff=$rel")
end
