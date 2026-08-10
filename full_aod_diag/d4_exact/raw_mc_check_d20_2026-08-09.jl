# Raw (UNTILTED, uniform-weight) Monte Carlo check at the calibration point (2026-08-09, user
# request): "sum up the rows in G and divide by W" -- i.e. mean(Z[:,col]) - target, BEFORE any
# tilting/solving. G's mean/pair columns are built as `dest = Z .- targets'` (mean_columns_direct!/
# pair_columns!, cm_originzc_moments.jl lines 61-67 -- confirmed by reading, not assumed), so the
# raw column mean under uniform weights is directly mean(Z[:,col]) - target, no solve needed at all.
# This is the check my earlier "MC average under F*" script did NOT do -- that one used the SOLVED
# m_star weights, which trivially satisfy the moments to solver tolerance by construction (equality
# constraints); THIS checks whether the THEORETICAL nu0 already matches the RAW empirical data, i.e.
# how much tilting the solver actually has to do, and flags which specific moments are largest.
#
# Uses the SAME point as investigate_delta_jump_2026-08-09.jl (last session, established the 7x
# jump): x_free_calib = ctx.θ0_up[ctx.free_idx], plain theoretical nu0 (Gamma(1-mu*k)), NO Variant D.
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
using Statistics, LinearAlgebra, Printf
using SpecialFunctions: gamma

W = 100_000
println("Building D20 real-data context (W=$W)...")
flush(stdout)
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
D = ctx.D
println("ctx built. D=$D sigma=$(ctx.σ) muHat=$(ctx.μHat) bi=$(ctx.bi) W=$W")
flush(stdout)

const K_mean = 3
const K_pair = 3
νfull0 = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K_mean]...)
bi = ctx.bi
pairs = packed_pair_index(D)

println("\n================ base family (diagonal-only) ================")
base_layout = OriginByPowerLayout(D, K_mean, K_pair)
Zraw_all_b, Zpairraw_all_b = build_raw_mean_pair_matrix_levels(ctx.U, K_mean, K_pair; μ = ctx.μHat)

println("\n-- mean block: mean(z_o^k) - target_o --")
for k in 1:K_mean
    tgt = mean_targets(base_layout, νfull0, k, D)
    raw = vec(mean(Zraw_all_b[k], dims = 1))
    disc = raw .- tgt
    @printf("  level k=%d: max|disc|=%.6e  mean|disc|=%.6e  mean|target|=%.4f  worst_origin=%d (disc=%.6e)\n",
            k, maximum(abs.(disc)), mean(abs.(disc)), mean(abs.(tgt)), argmax(abs.(disc)), disc[argmax(abs.(disc))])
end

println("\n-- pair block (diagonal only): mean(z_o^k * z_p^k) - target --")
for k in 1:K_pair
    tgt = pair_targets(base_layout, νfull0, k, D)
    raw = vec(mean(Zpairraw_all_b[k], dims = 1))
    disc = raw .- tgt
    j = argmax(abs.(disc)); (o, p) = pairs[j]
    @printf("  level k=%d: max|disc|=%.6e  mean|disc|=%.6e  mean|target|=%.4f  worst_pair=(%d,%d) disc=%.6e\n",
            k, maximum(abs.(disc)), mean(abs.(disc)), mean(abs.(tgt)), o, p, disc[j])
end

println("\n================ OZC-CROSS (K_pair^2 grid) ================")
cross_layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
Zpairraw_all_c = build_raw_cross_pair_matrix_levels(Zraw_all_b, K_pair)   # SAME Zraw_all as base (identical mean-level features)
levels = cross_pair_level_index(K_pair)

println("\n-- pair block (full K_pair^2 grid): mean(z_o^k1 * z_p^k2) - target --")
all_worst = NamedTuple[]
for (klin, (k1, k2)) in enumerate(levels)
    tgt = pair_targets(cross_layout, νfull0, klin, D)
    raw = vec(mean(Zpairraw_all_c[klin], dims = 1))
    disc = raw .- tgt
    j = argmax(abs.(disc)); (o, p) = pairs[j]
    disc_bi = [abs(disc[jj]) for (jj, (oo, pp)) in enumerate(pairs) if oo == bi || pp == bi]
    disc_nonbi = [abs(disc[jj]) for (jj, (oo, pp)) in enumerate(pairs) if oo != bi && pp != bi]
    tag = k1 == k2 ? "DIAGONAL" : "off-diag"
    @printf("  (k1=%d,k2=%d) [%s]: max|disc|=%.6e  mean|disc|=%.6e  mean|target|=%.4f  worst_pair=(%d,%d) disc=%.6e | max_bi=%.3e max_nonbi=%.3e\n",
            k1, k2, tag, maximum(abs.(disc)), mean(abs.(disc)), mean(abs.(tgt)), o, p, disc[j],
            maximum(disc_bi), maximum(disc_nonbi))
    push!(all_worst, (k1 = k1, k2 = k2, max_abs_disc = maximum(abs.(disc)), mean_abs_disc = mean(abs.(disc))))
end

println("\n==== Ranked by mean|disc| (which (k1,k2) blocks are furthest from target on average) ====")
for r in sort(all_worst, by = x -> -x.mean_abs_disc)
    @printf("  (k1=%d,k2=%d): mean|disc|=%.6e  max|disc|=%.6e\n", r.k1, r.k2, r.mean_abs_disc, r.max_abs_disc)
end
