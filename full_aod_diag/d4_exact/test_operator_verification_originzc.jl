# port/shared-inner-fg-operator-and-verification-2026-07-26, Job 2 gate: validates
# verify_inner_solution_operator_originzc! against the pre-existing dense verification path
# (archOZ_verified_state's own obj(inner_x,constr=...)/select_G_from_H recompute) at D=4.
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "operator_verification.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
D = ctx.D
nu0_origin(K::Int, D::Int) = vcat([fill(Float64(factorial(k)), D) for k in 1:K]...)

for (K_mean, K_pair) in [(1, 0), (1, 1), (2, 2)]
    layout = OriginByPowerLayout(D, K_mean, K_pair)
    νfull0 = nu0_origin(K_mean, D)
    pcx = build_originzc_production_context(ctx, CS, layout; fg_backend = :operator)
    _, base, verify = cm_originzc_production_value_verified(x_free_calib, νfull0, pcx)
    check("K=$K_mean/$K_pair: dense verify succeeded (status=$(verify.inner_status))", verify.inner_status == 0)

    octx = pcx.ctx_cm.octx
    cf = octx.core_cf_ref[]
    check("K=$K_mean/$K_pair: cf is a real CompressedFactual", cf isa CompressedFactual)
    ov = verify_inner_solution_operator_originzc!(base.ζstar, base.λstar, cf, octx.fg_zc_op, octx.fg_layout, νfull0, pcx.ctx_cm.obj, size(ctx.U, 1))

    # ov.f is the inner CC-dual objective (mean(Psi(r))+zeta) at the converged (zeta*,lambda*) --
    # NOT obj.H_save (a different, outer-gravity-model quantity from fill_K_directgp!, unrelated to
    # the inner dual solve at all -- comparing the two would be a category error). The meaningful
    # verification check is stationarity: g_lambda (the operator-recomputed full dual gradient at
    # the converged point) should be ~0, and should match the dense verifier's own KKT residual.
    check("K=$K_mean/$K_pair: operator-recomputed inner objective is finite ($(ov.f))", isfinite(ov.f))
    check("K=$K_mean/$K_pair: operator KKT residual is small (stationarity, resid=$(ov.kkt_resid))", ov.kkt_resid < 1e-6)
    check("K=$K_mean/$K_pair: operator KKT residual agrees with dense max_abs_moment_kkt_resid (dense=$(verify.max_abs_moment_kkt_resid), operator=$(ov.kkt_resid))",
          abs(ov.kkt_resid - verify.max_abs_moment_kkt_resid) < 1e-6)
    @printf "  K=%d/%d  dense_kkt=%.3e  operator_kkt=%.3e\n" K_mean K_pair verify.max_abs_moment_kkt_resid ov.kkt_resid
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
