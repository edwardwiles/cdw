# Real Delta* solve under a chosen draw design (2026-08-09, user question about Sobol QMC).
# Companion to qmc_vs_pseudorandom_discrepancy_2026-08-09.jl, which showed (no solve needed) that
# scrambled Sobol shrinks the raw moment discrepancies by ~5-7x (mean block) / ~4x (pair block) and
# the chi2-like statistic by ~30-100x at matched W. Since Delta* is essentially quadratic in those
# discrepancies, that predicts a ~30-40x smaller Delta*. This script CONFIRMS it with the actual
# inner KNITRO solve, for both families, at K_mean=K_pair=3, Variant D (kstar=2) active.
#
# CAVEAT, stated explicitly because it affects interpretation: the calibration point itself
# (ctx.theta0_up) is produced by the context build, which consumes the draws -- so switching draw
# design changes BOTH the draws and (potentially) the exact calibration point. This is therefore
# "Delta* at each design's own calibration point", the natural production-relevant comparison, not a
# controlled experiment holding theta fixed across designs. The script prints gp/kappa from each
# context so any shift in the point itself is visible rather than hidden.
#
# Usage: julia --project=. -t 8 .../delta_star_by_draw_design_2026-08-09.jl <W> <draw_design>
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
          "compressed_live.jl", "autarky_cf.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl"]
    include(joinpath(D4X, f))
end
using Statistics, LinearAlgebra, Printf
using SpecialFunctions: gamma

function local_profiled_nu_value(xf::AbstractVector{Float64}, ctx)
    θ_full = CS.reconstruct_full(xf, ctx.m)
    AodPow = aod_pow_matrix(θ_full, ctx)
    γ_prime_bi = θ_full[3+ctx.D]
    cf_num, cf_denom, _ = autarky_cf_scalars(ctx.obj, AodPow, ctx.σ, γ_prime_bi)
    return cf_denom / cf_num
end

W = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
DESIGN = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :pseudorandom
println("Building D20 real-data context (W=$W, draw_design=$DESIGN)...")
flush(stdout)
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = DESIGN, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
D = ctx.D
x_free_calib = ctx.θ0_up[ctx.free_idx]
println("ctx built. D=$D sigma=$(ctx.σ) muHat=$(ctx.μHat) bi=$(ctx.bi) W=$W design=$DESIGN")
@printf("calibration gp = %.12f\n", x_free_calib[1])
flush(stdout)

const K_mean = 3; const K_pair = 3; const KSTAR = 2
νfull0_dense_theory = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K_mean]...)

function run_point(label, layout, aml)
    println("\n---- $label (W=$W, $DESIGN) ----")
    flush(stdout)
    pcx = layout isa OriginByPowerCrossLayout ?
        build_originzc_cross_production_context(ctx, CS, layout; aml = aml) :
        build_originzc_production_context(ctx, CS, layout; fg_backend = :operator, aml = aml)
    nu_star = local_profiled_nu_value(x_free_calib, ctx)
    νfull_active, _ = gather_active_grad(aml, νfull0_dense_theory)
    νfull = scatter_nu_eff(aml, νfull_active, nu_star)
    t = @elapsed begin
        base, verify = archOZ_verified_state(x_free_calib, νfull, pcx.ctx_cm)
    end
    @printf("  solve %.1fs  inner_status=%d  Delta_dual=%.10f  kkt_resid=%.3e\n",
            t, verify.inner_status, verify.Delta_dual, verify.max_abs_moment_kkt_resid)
    flush(stdout)
    return verify.Delta_dual
end

base_layout  = OriginByPowerLayout(D, K_mean, K_pair)
cross_layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
db = run_point("base family, Variant D",  base_layout,  ActiveMeanLayout(base_layout, ctx.bi, KSTAR, D))
dc = run_point("OZC-CROSS, Variant D",    cross_layout, ActiveMeanLayout(cross_layout, ctx.bi, KSTAR, D))

println("\n==== SUMMARY (W=$W, draw_design=$DESIGN, K=3/3, Variant D) ====")
@printf("  base  Delta* = %.10f\n", db)
@printf("  cross Delta* = %.10f\n", dc)
@printf("  ratio cross/base = %.3f\n", dc/db)
println("\n  reference (pseudorandom, W=100k): base 0.0094898881, cross 0.0667306865, ratio 7.032")
