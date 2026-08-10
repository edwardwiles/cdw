# D4 correctness check for the new dense-reference OZC-CROSS path (2026-08-09): confirms
# wrap_moments_with_originzc_cross/build_originzc_cross_augmented_obj_dense (dense_reference_ozc_
# cross_2026-08-09.jl) give the SAME Delta_dual as the already-verified operator path, at identical
# calibration points, K=1/1,2/2,3/3. Must pass before trusting the D20-scale dense-vs-operator
# comparison (dense_vs_operator_d20_2026-08-09.jl).
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
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl",
          "dense_reference_ozc_cross_2026-08-09.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Statistics
using SpecialFunctions: gamma

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
x_free_calib = ctx.θ0_up[ctx.free_idx]
nu0_origin(K::Int, D::Int) = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K]...)

for (K_mean, K_pair) in [(1, 1), (2, 2), (3, 3)]
    println("\n==== D4 dense-vs-operator OZC-CROSS K_mean=$K_mean K_pair=$K_pair ====")
    layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
    νfull0 = nu0_origin(K_mean, D)

    pcx_op = build_originzc_cross_production_context(ctx, CS, layout)
    base_op, verify_op = archOZ_verified_state(x_free_calib, νfull0, pcx_op.ctx_cm; verification_backend = :operator)
    println("  operator:       inner_status=$(verify_op.inner_status)  Delta_dual=$(verify_op.Delta_dual)  kkt_resid=$(verify_op.max_abs_moment_kkt_resid)")

    pcx_dn = build_originzc_cross_production_context_dense(ctx, CS, layout)
    base_dn, verify_dn = archOZ_verified_state(x_free_calib, νfull0, pcx_dn.ctx_cm; verification_backend = :dense_reference)
    println("  dense_reference: inner_status=$(verify_dn.inner_status)  Delta_dual=$(verify_dn.Delta_dual)  kkt_resid=$(verify_dn.max_abs_moment_kkt_resid)")

    d_abs = abs(verify_op.Delta_dual - verify_dn.Delta_dual)
    check("K=$K_mean/$K_pair: operator vs dense_reference Delta_dual agree", d_abs < 1e-6)
end

println("\n", ALL_PASS[] ? "ALL PASS" : "SOME FAILED")
exit(ALL_PASS[] ? 0 : 1)
