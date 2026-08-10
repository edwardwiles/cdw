# Control check (2026-08-09): does the SAME q0-fold-vs-operator-recompute comparison methodology
# show a similar-sized gap for the UNMODIFIED base origin-ZC family (originzc_fixed_contribution,
# byte-unchanged production code)? If yes, the ~3e-5 gap seen for OZC-CROSS is a property of
# comparing build_lfix_base_cache's closed-form economic block against the operator economic block
# in general (pre-existing, unrelated to the new cross-power code) -- not a bug in
# originzc_cross_fixed_contribution.
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl"]
    include(joinpath(D4X, f))
end
using SpecialFunctions: gamma

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
x_free_calib = ctx.θ0_up[ctx.free_idx]
nu0_origin(K::Int, D::Int) = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K]...)

println("ctx.μHat = ", ctx.μHat)
for (K_mean, K_pair) in [(1, 1), (2, 2), (3, 3)]
    νfull0 = nu0_origin(K_mean, D)
    println("K=$K_mean nu0 = $νfull0")
    layout = OriginByPowerLayout(D, K_mean, K_pair)
    pcx = build_originzc_production_context(ctx, CS, layout; fg_backend = :operator)
    base, verify = archOZ_verified_state(x_free_calib, νfull0, pcx.ctx_cm)
    println("K=$K_mean/$K_pair: inner_status=$(verify.inner_status) Delta_dual=$(verify.Delta_dual) max|lambda|=$(maximum(abs.(base.λstar)))")

    cache0 = build_lfix_base_cache(x_free_calib, pcx.ctx_cm, base)
    contrib0 = originzc_fixed_contribution(base, pcx.aug, νfull0)
    q0_via_fold = cache0.q0 .- contrib0

    st = pcx.octx.fg_lookup_st
    q0_via_operator = copy(dual_index!(st, vcat(base.ζstar, base.λstar)))
    d_q0 = maximum(abs.(q0_via_fold .- q0_via_operator))
    println("BASE FAMILY K=$K_mean/$K_pair: (B)-style q0-fold vs operator-recompute max abs diff = $d_q0")

    g_analytic = d_delta_dual_d_eta_origin_vec(base.λstar, pcx.aug, νfull0; mean_m = verify.m_mean)
    g_fd = d_delta_dual_d_eta_origin_fd(x_free_calib, νfull0, pcx.ctx_cm; h = 1e-4)
    d_ag = maximum(abs.(g_analytic .- g_fd))
    rel = d_ag / max(1e-8, maximum(abs.(g_fd)))
    println("BASE FAMILY K=$K_mean/$K_pair: (A)-style analytic vs FD max abs diff=$d_ag max rel diff=$rel")
end
