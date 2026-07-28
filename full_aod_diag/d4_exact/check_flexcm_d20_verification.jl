# Ad hoc check: what does nStatus=-103 mean, and how well does the flexible-CM D=20/W=100,000
# solved point pass the model's own verification measures (KKT residual, Delta_dual/Delta_primal
# gap, moment residuals)? Run for BOTH the dense-reference and operator bundles at the same point.
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl", "context_real_d20.jl"]
    include(joinpath(_D4E, f))
end
lp(xs...) = (println(xs...); flush(stdout))

lp("Building real D=20/W=100,000 context...")
ctx = d20_real_setup(W = 100_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]
lp("Context built.")

pcx_d = build_cm_production_context(ctx, CS; L = 50, contrasts = :orthonormal, use_compressed_core = true,
                                     threaded_bins = true, moment_representation = :dense_reference)
pcx_o = build_cm_production_context(ctx, CS; L = 50, contrasts = :orthonormal, use_compressed_core = true,
                                     threaded_bins = true, moment_representation = :operator)
pcx_d.cctx.core_hessian_backend = :exact_winner_pair_parallel
pcx_o.cctx.core_hessian_backend = :exact_winner_pair_parallel
pcx_d.cctx.cm_cross_hessian_backend = :winner_bin
pcx_o.cctx.cm_cross_hessian_backend = :winner_bin

lp("="^90); lp("Dense-reference bundle: archC_verified_state (operator verification backend, production default)")
t0 = time()
base_d, verify_d = archC_verified_state(x_free_calib, pcx_d.ctx_cm, pcx_d.cctx; verification_backend = :operator)
lp("  solved in ", time() - t0, "s")
for k in propertynames(verify_d)
    lp("  ", k, " = ", getproperty(verify_d, k))
end

lp("="^90); lp("Operator bundle: archC_verified_state (operator verification backend, production default)")
t0 = time()
base_o, verify_o = archC_verified_state(x_free_calib, pcx_o.ctx_cm, pcx_o.cctx; verification_backend = :operator)
lp("  solved in ", time() - t0, "s")
for k in propertynames(verify_o)
    lp("  ", k, " = ", getproperty(verify_o, k))
end

lp("="^90); lp("Cross-check: dense vs operator verification agreement")
for k in propertynames(verify_d)
    vd = getproperty(verify_d, k); vo = getproperty(verify_o, k)
    if vd isa Number && vo isa Number
        lp("  ", k, ": dense=", vd, " operator=", vo, " |Δ|=", abs(vd - vo))
    end
end
lp("DONE")
