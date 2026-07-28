# Final-architecture-closure task, Goal 7/13: isolated check that CM+ZC's PRODUCTION-DEFAULT
# configuration alone (no dense-reference comparison arm in the same run) shows
# dense_cross_hessian_calls == 0. The gate script this was extracted from
# (test_cm_meanzc_hcz_hzz_direct_d20.jl) deliberately runs BOTH :dense_reference and :winner_bin
# backends in the same process for A/B comparison, so its own counter dump is not a clean read of
# "production alone" -- this script isolates that by never touching :dense_reference at all.
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl", "instrumentation.jl",
          "oracle_fast.jl", "gravity_elimination.jl", "structured_moment_build.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl",
          "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "cm_outer_driver.jl",
          "cm_meanzc_moments.jl", "cm_originzc_target_layout.jl", "cm_meanzc_production.jl", "nested_quantile_grids.jl",
          "winner_pair_cross_hessian.jl", "zc_restriction_operator.jl", "operator_verification.jl"]
    include(joinpath(D4X, f))
end

println("Building real D=20 context (W=80000, delta=1.0, destination_sample=:exclude_row)..."); flush(stdout)
ctx = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]

println("cm_cross_hessian_backend default = $(CM_MEANZC_CM_CROSS_HESSIAN_BACKEND_DEFAULT[])")
println("zc_cross_hessian_backend default = $(ORIGINZC_ZC_CROSS_HESSIAN_BACKEND_DEFAULT[])")
@assert CM_MEANZC_CM_CROSS_HESSIAN_BACKEND_DEFAULT[] == :winner_bin
@assert ORIGINZC_ZC_CROSS_HESSIAN_BACKEND_DEFAULT[] == :winner_bin

reset_no_dense_g_counters!()
pcx_z = build_cm_meanzc_production_context(ctx, CS; L = 50, K_mean = 1, K_pair = 1, contrasts = :anchored)
νvec0 = [1.0]
fval, base, verify = cm_meanzc_production_value_verified(x_free_calib, νvec0, pcx_z)
r = no_dense_g_report()
println("\n=== PRODUCTION-ONLY counter snapshot (K_mean=1,K_pair=1, default backends, no dense-reference call anywhere) ===")
for k in fieldnames(typeof(r))
    println("    $k = $(getfield(r, k))")
end
nStatus = verify.inner_status
@assert nStatus in (0, -100, -101, -103) "inner solve failed, nStatus=$nStatus"
@assert r.dense_cross_hessian_calls == 0 "PRODUCTION INVARIANT VIOLATED: dense_cross_hessian_calls=$(r.dense_cross_hessian_calls) != 0"
@assert r.winner_cross_hessian_calls > 0 "no winner-aware cross-Hessian calls recorded at all -- suspicious, the Hessian callback may not have fired"
println("\n>>> PASS: production-default CM+ZC (K_mean=1,K_pair=1) shows dense_cross_hessian_calls=0, winner_cross_hessian_calls=$(r.winner_cross_hessian_calls)")
flush(stdout)
