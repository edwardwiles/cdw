# W-sensitivity check on the OZC-CROSS 7x Delta* jump (2026-08-09 follow-up, user question): if the
# gap is a finite-sample/Monte-Carlo artifact of level-2 z-power features being noisier at moderate
# W, it should shrink as W grows; if it's a genuine population-level effect it should persist. Runs
# ONLY the Variant D (kstar=2) point for both families at a given W -- verify_ozc_cross_variant_d_d20
# already confirmed Variant D's Delta* is bit-identical to the no-aml baseline at W=100k for both
# families, so the no-aml points don't need re-running at every W to track the ratio.
#
# Usage: julia --project=. -t 8 full_aod_diag/d4_exact/w_sensitivity_variant_d_2026-08-09.jl <W>
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
          "compressed_live.jl", "autarky_cf.jl", "cm_checkpoint.jl", "cm_originzc_checkpoint.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl"]
    include(joinpath(D4X, f))
end
using Statistics, LinearAlgebra
using SpecialFunctions: gamma

W = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
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

const K_mean = 3
const K_pair = 3
const KSTAR = 2
νfull0_dense_theory = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K_mean]...)

function run_aml_point(label, layout, aml)
    println("\n---- $label (W=$W) ----")
    flush(stdout)
    pcx = layout isa OriginByPowerCrossLayout ?
        build_originzc_cross_production_context(ctx, CS, layout; aml = aml) :
        build_originzc_production_context(ctx, CS, layout; fg_backend = :operator, aml = aml)
    nu_star = originzc_profiled_nu_value(x_free_calib, ctx)
    νfull_active, _ = gather_active_grad(aml, νfull0_dense_theory)
    νfull = scatter_nu_eff(aml, νfull_active, nu_star)
    t = @elapsed begin
        base, verify = archOZ_verified_state(x_free_calib, νfull, pcx.ctx_cm)
    end
    println("  W=$W  solve: $(round(t, digits=1))s  inner_status=$(verify.inner_status)  Delta_dual=$(verify.Delta_dual)  max_abs_moment_kkt_resid=$(verify.max_abs_moment_kkt_resid)")
    flush(stdout)
    return verify.Delta_dual
end

base_layout = OriginByPowerLayout(D, K_mean, K_pair)
cross_layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
base_aml = ActiveMeanLayout(base_layout, ctx.bi, KSTAR, D)
cross_aml = ActiveMeanLayout(cross_layout, ctx.bi, KSTAR, D)

d_base = run_aml_point("base family, Variant D (kstar=2)", base_layout, base_aml)
d_cross = run_aml_point("OZC-CROSS, Variant D (kstar=2)", cross_layout, cross_aml)

println("\n==== SUMMARY (W=$W) ====")
println("  base Delta*  = $d_base")
println("  cross Delta* = $d_cross")
println("  ratio cross/base = $(d_cross / d_base)")
