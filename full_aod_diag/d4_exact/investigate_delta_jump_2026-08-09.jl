# Investigate the 7x Delta* jump (K=3/3 diagonal-only base: 0.00949 vs OZC-CROSS: 0.067) at D20
# real data (2026-08-09, user question): "I would have thought the tightest ones are the 3
# diagonal, and that's present in the old version too." Two diagnostics:
#
# (1) FAST, no re-solve: for each of the 9 (k1,k2) blocks, the RAW (untilted, uniform-weight)
#     empirical discrepancy from the theoretical nu0 target -- mean_s(Zpair[s,pair]) - target[pair],
#     averaged/maxed over the 190 origin pairs. If the 6 off-diagonal blocks have systematically
#     LARGER raw discrepancies than the 3 diagonal ones (which the base family ALSO has to satisfy),
#     that's a real, non-bug explanation: the off-diagonal restrictions are simply harder to satisfy
#     at this nu0/data, not something introduced by a coding error (the K=1/K=2 residual/(B) checks
#     already confirm the restrictions are being imposed with the intended, correct targets).
#
# (2) At the ALREADY-SOLVED K=3/3 point (re-solved once here, ~5min), per-block max|lambda*| -- are
#     the 6 off-diagonal dual multipliers comparable in magnitude to the 3 diagonal ones, or do a
#     few specific ones dominate (pointing at a small number of origin-pair/level combos driving
#     most of the Delta* increase, rather than a uniform effect across all new restrictions)?
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
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl"]
    include(joinpath(D4X, f))
end
using Statistics, LinearAlgebra
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

K_mean = 3; K_pair = 3
layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
νfull0 = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K_mean]...)
pcx = build_originzc_cross_production_context(ctx, CS, layout)
aug = pcx.aug

println("\n==== Diagnostic (1): raw (untilted, uniform-weight) discrepancy per (k1,k2) block ====")
levels = cross_pair_level_index(K_pair)
for (klin, (k1, k2)) in enumerate(levels)
    target = pair_targets(layout, νfull0, klin, D)
    raw_mean = vec(mean(aug.Zpairraw_all[klin], dims = 1))
    disc = raw_mean .- target
    tag = k1 == k2 ? "DIAGONAL" : "off-diag"
    println("  (k1=$k1,k2=$k2) [$tag]: mean|disc|=$(mean(abs.(disc)))  max|disc|=$(maximum(abs.(disc)))  mean|target|=$(mean(abs.(target)))  mean_rel=$(mean(abs.(disc)) / mean(abs.(target)))")
end
flush(stdout)

println("\n==== Diagnostic (2): solving K=3/3 once for per-block max|lambda*| ====")
flush(stdout)
t = @elapsed (base, verify) = archOZ_verified_state(x_free_calib, νfull0, pcx.ctx_cm)
println("solve: $(t)s inner_status=$(verify.inner_status) Delta_dual=$(verify.Delta_dual)")
flush(stdout)

ncore_econ = aug.ncore_econ
npair = D * (D - 1) ÷ 2
pair_start0 = ncore_econ + K_mean * D
for (klin, (k1, k2)) in enumerate(levels)
    λ_pair_klin = @view base.λstar[pair_start0+(klin-1)*npair : pair_start0+klin*npair-1]
    tag = k1 == k2 ? "DIAGONAL" : "off-diag"
    println("  (k1=$k1,k2=$k2) [$tag]: max|lambda*|=$(maximum(abs.(λ_pair_klin)))  mean|lambda*|=$(mean(abs.(λ_pair_klin)))")
end
