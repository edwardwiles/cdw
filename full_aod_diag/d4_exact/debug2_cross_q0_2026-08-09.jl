# Debug 2 (2026-08-09): originzc_cross_fixed_contribution itself was just proven correct
# (matches restriction_forward! via octx.hzz_zc_op to 2.2e-16). So the (B) check's ~3e-5 mismatch
# must come from cache0.q0 vs the fg_lookup_st-based dual_index! recompute specifically. Compare
# BOTH operator instances (fg_lookup_st.op/zc_ws vs hzz_zc_op/hzz_zc_ws -- two SEPARATE
# ZCRestrictionOperator/workspace objects per build_originzc_core_hess_ctx) against cache0.q0.
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

cache0 = build_lfix_base_cache(x_free_calib, pcx.ctx_cm, base)
contrib0 = originzc_cross_fixed_contribution(base, aug, νfull0)
q0_via_fold = cache0.q0 .- contrib0

# (1) via fg_lookup_st (what my original verify script used)
st = pcx.octx.fg_lookup_st
q0_via_fg = copy(dual_index!(st, vcat(base.ζstar, base.λstar)))
println("q0_via_fold vs q0_via_fg (fg_lookup_st):     max abs diff = ", maximum(abs.(q0_via_fold .- q0_via_fg)))
println("cache0.q0 vs (q0_via_fg + contrib0):          max abs diff = ", maximum(abs.(cache0.q0 .- (q0_via_fg .+ contrib0))))

# (2) reconstruct arg0 manually via hzz_zc_op + economic_forward! (independent of fg_lookup_st entirely)
op2 = pcx.octx.hzz_zc_op
ws2 = pcx.octx.hzz_zc_ws
refresh_zc_targets!(ws2, op2, aug.layout, νfull0)
ncore1 = aug.ncore_econ - 1
λ_E = collect(base.λstar[1:ncore1])
λ_mean = collect(base.λstar[aug.ncore_econ : aug.ncore_econ + aug.n_mean - 1])
λ_pair = collect(base.λstar[aug.ncore_econ + aug.n_mean : aug.ncore_econ + aug.n_mean + aug.n_pair - 1])
W = size(aug.Zraw_all[1], 1)
arg0_manual = fill(-base.ζstar, W)
cf = pcx.octx.core_cf_ref[]
econ_ws = economic_operator_workspace(cf)
econ_buf = zeros(W)
economic_forward!(econ_buf, λ_E, cf, econ_ws)
arg0_manual .-= econ_buf
restriction_forward!(arg0_manual, λ_mean, λ_pair, op2, ws2)
println("q0_via_fold vs arg0_manual (hzz_zc_op path):  max abs diff = ", maximum(abs.(q0_via_fold .- arg0_manual)))
println("q0_via_fg   vs arg0_manual (hzz_zc_op path):  max abs diff = ", maximum(abs.(q0_via_fg .- arg0_manual)))

# also print raw values at first index
println("q0_via_fold[1]=", q0_via_fold[1], " q0_via_fg[1]=", q0_via_fg[1], " arg0_manual[1]=", arg0_manual[1], " cache0.q0[1]=", cache0.q0[1])
println("st.op === op2 ? ", st.op === op2)
println("st.op.Zraw_all[1] === op2.Zraw_all[1] ? ", st.op.Zraw_all[1] === op2.Zraw_all[1])
println("st.zc_ws.targets_mean = ", st.zc_ws.targets_mean, "  ws2.targets_mean = ", ws2.targets_mean)
