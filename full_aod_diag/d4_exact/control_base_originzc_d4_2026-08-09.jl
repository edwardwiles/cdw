# Control check for the OZC-CROSS smoke test (2026-08-09): does the UNMODIFIED base origin-ZC
# family (OriginByPowerLayout, build_originzc_production_context) also fail to converge at K_mean=1
# K_pair=1 with the SAME nu0/x_free_calib the OZC-CROSS smoke test used? If yes, the -300 is a
# property of this naive nu0 at this point, not a bug in the new cross-power code.
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl"]
    include(joinpath(D4X, f))
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
x_free_calib = ctx.θ0_up[ctx.free_idx]
nu0_origin(K::Int, D::Int) = vcat([fill(Float64(factorial(k)), D) for k in 1:K]...)

for (K_mean, K_pair) in [(1, 0), (1, 1), (2, 2)]
    layout = OriginByPowerLayout(D, K_mean, K_pair)
    νfull0 = nu0_origin(K_mean, D)
    pcx = build_originzc_production_context(ctx, CS, layout; fg_backend = :operator)
    try
        base, verify = archOZ_verified_state(x_free_calib, νfull0, pcx.ctx_cm)
        println("BASE FAMILY K=$K_mean/$K_pair: CONVERGED inner_status=$(verify.inner_status) Delta_dual=$(verify.Delta_dual) max_abs_moment_kkt_resid=$(verify.max_abs_moment_kkt_resid)")
    catch e
        println("BASE FAMILY K=$K_mean/$K_pair: FAILED -- ", sprint(showerror, e))
    end
end
