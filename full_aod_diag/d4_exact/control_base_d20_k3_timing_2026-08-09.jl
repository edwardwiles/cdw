# Control (2026-08-09, user request): the UNMODIFIED base origin-ZC family (diagonal-only pair
# restriction, OriginByPowerLayout) at the SAME D20/:exclude_row/W=100,000/K_mean=3/K_pair=3
# config as the OZC-CROSS run that took ~297s and gave Delta_dual=0.067 -- isolates whether that
# is specific to OZC-CROSS's much bigger restriction block (n_pair=1710 vs base's n_pair=570) or a
# property of this launch config (threading/BLAS) shared by both.
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "country_resolve.jl"]
    include(joinpath(D4X, f))
end
using SpecialFunctions: gamma

W = 100_000
println("Building D20 real-data context (W=$W)...")
flush(stdout)
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
D = ctx.D
println("ctx built. D=$D μHat=$(ctx.μHat)")
flush(stdout)
x_free_calib = ctx.θ0_up[ctx.free_idx]
nu0_origin(K::Int, D::Int) = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K]...)

K_mean = 3; K_pair = 3
layout = OriginByPowerLayout(D, K_mean, K_pair)
νfull0 = nu0_origin(K_mean, D)
pcx = build_originzc_production_context(ctx, CS, layout; fg_backend = :operator)
println("n_pair(base, diagonal-only)=$(pcx.aug.n_pair)  n_mean=$(pcx.aug.n_mean)")
flush(stdout)
t = @elapsed (base, verify) = archOZ_verified_state(x_free_calib, νfull0, pcx.ctx_cm)
println("BASE FAMILY (diagonal-only) K=$K_mean/$K_pair: solve=$(t)s inner_status=$(verify.inner_status) Delta_dual=$(verify.Delta_dual) max_abs_moment_kkt_resid=$(verify.max_abs_moment_kkt_resid)")
