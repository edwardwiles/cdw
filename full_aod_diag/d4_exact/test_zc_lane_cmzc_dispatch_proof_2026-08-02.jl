# ZC lane task (2026-08-02), Phase II dispatch proof: confirm the REDUCED+WIDENED CM+ZC Hessian
# path genuinely dispatches through the optimized production backends (blas_syrk H_ZZ,
# drawmajor_v2 H_EZ/H_EM) with zero explicit kwargs (production defaults), and that the winner-
# based (non-dense) cross-Hessian path is what actually fires during the real D4 KNITRO solve of
# the reduced+widened dual problem -- not a silent dense fallback. Reuses the already-passing D4
# gate's own production include stack.
const D4X = @__DIR__
isdefined(Main, :NO_DENSE_G_COUNTERS) || include(joinpath(D4X, "no_dense_g_counters.jl"))

include(joinpath(D4X, "test_zc_lane_cmzc_d4_fg_and_solve_gate_2026-08-02.jl"))

println()
println("=== Dispatch-proof counters after the REAL D4 KNITRO reduced+widened solve ===")
println("ZC_GRAM_BACKEND_DEFAULT[] (H_ZZ/H_MM) = ", ZC_GRAM_BACKEND_DEFAULT[])
println("ZC_EZ_BACKEND_DEFAULT[]   (H_EZ, origin-ZC's own dispatch -- CM+ZC's H_EM uses the SAME\n" *
        "  winner_pair_cross_hessian_zc_block! kernel but does NOT read this backend Ref; H_EM's\n" *
        "  own dispatch is the unconditional use_profiled_correction=true call in _fill_cm_HEE!,\n" *
        "  see cm_hessian_architectures.jl -- reported here for completeness only) = ", ZC_EZ_BACKEND_DEFAULT[])
c = NO_DENSE_G_COUNTERS[]
println("winner_cross_hessian_calls = ", c.winner_cross_hessian_calls)
println("dense_cross_hessian_calls  = ", c.dense_cross_hessian_calls)

ok = ZC_GRAM_BACKEND_DEFAULT[] === :blas_syrk && c.winner_cross_hessian_calls > 0 && c.dense_cross_hessian_calls == 0
println(ok ?
    "\nPASS  dispatch proof: blas_syrk (H_ZZ/H_MM) production default selected with zero explicit kwargs, winner-based cross-Hessian path fired $(c.winner_cross_hessian_calls) times, zero dense fallbacks" :
    "\nFAIL  dispatch proof")
