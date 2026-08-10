# D20 real-data dense-vs-operator comparison for OZC-CROSS (2026-08-09, user request): does the
# SAME 7x-jump-showing K=3/3 calibration point give the same high Delta* through a COMPLETELY
# different computational path (dense G matrix, Hessian via literal matrix multiplication,
# Architecture A's dense-BLAS inner solve) as through the operator/ZCRestrictionOperator FG path?
# If yes, the jump is not an artifact of the operator-path machinery specifically.
#
# W kept moderate (see ARGS) because the dense Hessian costs O(W*d^2) per KNITRO inner iteration
# (d = ncore_econ + n_mean + n_pair ~ 2100+ at K=3/3 cross) -- confirmed correct first at D4 scale
# (dense_vs_operator_d4_2026-08-09.jl, ALL PASS, agrees to ~10 sig figs) before running this.
#
# Usage: julia --project=. -t 8 full_aod_diag/d4_exact/dense_vs_operator_d20_2026-08-09.jl <W>
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "country_resolve.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl",
          "dense_reference_ozc_cross_2026-08-09.jl"]
    include(joinpath(D4X, f))
end
using Statistics, LinearAlgebra
using SpecialFunctions: gamma

W = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8000
println("Building D20 real-data context (W=$W)...")
flush(stdout)
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
D = ctx.D
println("ctx built. D=$D sigma=$(ctx.σ) muHat=$(ctx.μHat) bi=$(ctx.bi) W=$W")
flush(stdout)
x_free_calib = ctx.θ0_up[ctx.free_idx]

K_mean = 3; K_pair = 3
layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
νfull0 = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K_mean]...)

println("\n---- operator path ----")
flush(stdout)
pcx_op = build_originzc_cross_production_context(ctx, CS, layout)
t_op = @elapsed (base_op, verify_op) = archOZ_verified_state(x_free_calib, νfull0, pcx_op.ctx_cm; verification_backend = :operator)
println("  operator:        $(round(t_op,digits=1))s  inner_status=$(verify_op.inner_status)  Delta_dual=$(verify_op.Delta_dual)  kkt_resid=$(verify_op.max_abs_moment_kkt_resid)")
flush(stdout)

println("\n---- dense_reference path ----")
flush(stdout)
pcx_dn = build_originzc_cross_production_context_dense(ctx, CS, layout)
t_dn = @elapsed (base_dn, verify_dn) = archOZ_verified_state(x_free_calib, νfull0, pcx_dn.ctx_cm; verification_backend = :dense_reference)
println("  dense_reference: $(round(t_dn,digits=1))s  inner_status=$(verify_dn.inner_status)  Delta_dual=$(verify_dn.Delta_dual)  kkt_resid=$(verify_dn.max_abs_moment_kkt_resid)")
flush(stdout)

println("\n==== SUMMARY (W=$W, K_mean=K_pair=3) ====")
println("  operator Delta*        = $(verify_op.Delta_dual)")
println("  dense_reference Delta* = $(verify_dn.Delta_dual)")
println("  abs diff = $(abs(verify_op.Delta_dual - verify_dn.Delta_dual))")
