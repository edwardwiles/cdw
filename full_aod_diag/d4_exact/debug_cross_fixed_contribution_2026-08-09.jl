# Debug (2026-08-09): originzc_cross_fixed_contribution shows a ~3e-5 mismatch even at K=1
# (mathematically identical to the base family, which matches at machine precision). Cross-check
# my hand-rolled dot/matrix-vector formula against the ALREADY-VALIDATED restriction_forward!
# primitive (zc_restriction_operator.jl) computed on the SAME op/ws/lambda -- if these two
# INDEPENDENT computations of the same quantity disagree, the bug is in my hand-rolled formula.
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

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
x_free_calib = ctx.θ0_up[ctx.free_idx]
nu0_origin(K::Int, D::Int) = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K]...)

K_mean = 1; K_pair = 1
layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
νfull0 = nu0_origin(K_mean, D)
pcx = build_originzc_cross_production_context(ctx, CS, layout)
base, verify = archOZ_verified_state(x_free_calib, νfull0, pcx.ctx_cm)
aug = pcx.aug
ncore_econ = aug.ncore_econ
npair = D * (D - 1) ÷ 2

# mine
contrib0_mine = originzc_cross_fixed_contribution(base, aug, νfull0)

# ground truth via the validated shared operator primitive
op = pcx.octx.hzz_zc_op
ws = pcx.octx.hzz_zc_ws
refresh_zc_targets!(ws, op, aug.layout, νfull0)
mean_start = ncore_econ
pair_start0 = ncore_econ + K_mean * D
λ_mean = collect(base.λstar[mean_start : mean_start + K_mean * D - 1])
λ_pair = collect(base.λstar[pair_start0 : pair_start0 + K_pair^2 * npair - 1])
W = size(aug.Zraw_all[1], 1)
neg_out = zeros(W)
restriction_forward!(neg_out, λ_mean, λ_pair, op, ws)   # neg_out = -(R*lambda) = -Phi*lambda + t'lambda
contrib0_via_op = -neg_out   # = Phi*lambda - t'lambda, matching contrib0's definition

d = maximum(abs.(contrib0_mine .- contrib0_via_op))
println("max abs diff (mine vs validated-operator): $d")
println("contrib0_mine[1:5]     = ", contrib0_mine[1:5])
println("contrib0_via_op[1:5]   = ", contrib0_via_op[1:5])
println("K_mean=$K_mean layout.K_mean=$(aug.layout.K_mean) layout.K_pair=$(aug.layout.K_pair)")
println("n_eta(layout)=$(n_eta(aug.layout)) length(νfull0)=$(length(νfull0))")
println("aug.n_mean=$(aug.n_mean) aug.n_pair=$(aug.n_pair) op K_mean=$(op.K_mean) op K_pair=$(op.K_pair) op mean_offset=$(op.mean_offset)")

# also isolate: mean-only contribution (zero out pair lambda) via both paths
out_meanonly = zeros(W)
out_meanonly .+= aug.Zraw_all[1] * λ_mean
out_meanonly .-= dot(mean_targets(aug.layout, νfull0, 1, D), λ_mean)
neg_meanonly = zeros(W)
restriction_forward!(neg_meanonly, λ_mean, zeros(K_pair^2 * npair), op, ws)
println("mean-only max abs diff: ", maximum(abs.(out_meanonly .- (-neg_meanonly))))

out_paironly = zeros(W)
out_paironly .+= aug.Zpairraw_all[1] * λ_pair
out_paironly .-= dot(pair_targets(aug.layout, νfull0, 1, D), λ_pair)
neg_p800 = zeros(W)
restriction_forward!(neg_p800, zeros(K_mean * D), λ_pair, op, ws)
println("pair-only max abs diff: ", maximum(abs.(out_paironly .- (-neg_p800))))
