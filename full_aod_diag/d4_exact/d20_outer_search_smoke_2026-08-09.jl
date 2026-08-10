# D20 real-data outer-search smoke test for OZC-CROSS (2026-08-09 task, phase 3): confirms a real
# outer KNITRO loop, using the shared inner-solve/gradient machinery already verified this session,
# genuinely explores multiple (gp, zfree, eta) points and can solve the inner-loop program at each
# -- NOT a full production run (short maxtime_real, K_mean=1/K_pair=1, the cheapest already-
# converging config, chosen so several outer evaluations fit in a short wall-clock budget rather
# than getting only 1-2 evaluations at K=3/3's ~5min/solve cost).
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
          "originzc_cross_outer_driver_2026-08-09.jl"]
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
D = ctx.D; Ddest = ctx.D_dest
println("ctx built. D=$D D_dest=$Ddest μHat=$(ctx.μHat)")
flush(stdout)

pe_g = build_pivot_elimination(ctx)
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest]
zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, Ddest), pe_g)
gp0 = ctx.θ0_up[3+D]

K_mean = 1; K_pair = 1
layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
nu0 = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K_mean]...)
eta0 = log.(nu0)
w0 = vcat(gp0, zfree0, eta0)
println("w0 length=$(length(w0)) (gp=1, zfree=$(length(zfree0)), eta=$(length(eta0)))")
flush(stdout)

pcx = build_originzc_cross_production_context(ctx, CS, layout)

println("Starting outer KNITRO search (delta=1.0, maxtime_real=120s)...")
flush(stdout)
result = run_originzc_cross_upper(pcx, ctx, pe_g, w0; delta = 1.0, maxtime_real = 120.0, verbose = true)

println("\n=== FINAL RESULT ===")
println("knitro_status=", result.knitro_status, " wall=", result.wall, " n_eval=", result.n_eval, " n_grad=", result.n_grad)
println("kappa=", result.kappa)
println("best=", result.best)
println("n_distinct_gp_tried=", length(unique(round.([t.gp for t in result.trace]; digits=6))))
