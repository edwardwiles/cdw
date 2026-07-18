include(joinpath(@__DIR__, "c8_gammabranch_core.jl"))
using Printf

println("\n" , "="^78)
println("SMOKE TEST 1: raw Delta at the lower incumbent's own (g,A) point (no reoptimization)")
xf_lower = x_free_from_w(W_LOWER_INCUMBENT)
r_lower = evaluate_fullA(xf_lower, ctx; cache = nothing, warm = false)
@printf("  g=%.10f  Delta_dual=%.10f  Delta-delta=%.4e  inner_status=%d  gravity=%.3e\n",
    r_lower.gamma_focal_prime, r_lower.Delta_dual, r_lower.Delta_dual - ctx.δ, r_lower.inner_status, r_lower.gravity_value)

println("\n", "="^78)
println("SMOKE TEST 2: is the lower incumbent's own A already the constrained min_A Delta(g,A) for its g?")
res = profile_delta_at_gamma_c8(G_LOWER_INCUMBENT, ZFREE_LOWER_INCUMBENT, ctx, pe; moment_repr = :dense, maxtime_real = 30.0, hessopt_tag = "sr1")
@printf("  reoptimized: knitro_status=%d n_eval=%d wall=%.1fs best_Delta=%.10f (Delta-delta=%.4e)\n",
    res.knitro_status, res.n_eval, res.wall, res.best_Delta, res.best_Delta - ctx.δ)
reldiff_A = res.best_zfree === nothing ? NaN : norm(Aod_vec_c8(res.best_zfree) .- Aod_vec_c8(ZFREE_LOWER_INCUMBENT)) / norm(Aod_vec_c8(ZFREE_LOWER_INCUMBENT))
@printf("  relL2(A_reopt - A_lower_incumbent) = %.4e\n", reldiff_A)

println("\n", "="^78)
println("SMOKE TEST 3: dense vs compressed timing at g=G_LOWER_INCUMBENT, single F-eval, warm=false then warm=true")
t_dense_cold = @elapsed evaluate_fullA(xf_lower, ctx; cache = nothing, warm = false)
t_dense_warm = @elapsed evaluate_fullA(xf_lower, ctx; cache = nothing, warm = true)
t_comp_cold = @elapsed evaluate_fullA_fast(xf_lower, ctx; cache = nothing, warm = false, moment_representation = :compressed)
t_comp_warm = @elapsed evaluate_fullA_fast(xf_lower, ctx; cache = nothing, warm = true, moment_representation = :compressed)
@printf("  dense:      cold=%.4fs warm=%.4fs\n", t_dense_cold, t_dense_warm)
@printf("  compressed: cold=%.4fs warm=%.4fs\n", t_comp_cold, t_comp_warm)

println("\nSMOKE TESTS DONE")
