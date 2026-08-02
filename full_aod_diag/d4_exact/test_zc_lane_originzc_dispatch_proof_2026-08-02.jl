# ZC lane task (2026-08-02), Phase I dispatch proof: confirm the REDUCED origin-ZC Hessian path
# genuinely dispatches through the optimized production backends (blas_syrk H_ZZ, drawmajor_v2
# H_EZ) with zero explicit kwargs (production defaults), and that the winner-based (non-dense)
# cross-Hessian path is what actually fires during a REAL D4 KNITRO solve of the reduced origin-ZC
# dual problem -- not a silent dense fallback. Reuses the canonical production include stack (via
# the already-independently-passing D4 gate script itself, not a hand-picked subset) so this is a
# genuine production-stack dispatch proof, not a synthetic isolated call.
const D4X = @__DIR__
reset_counters_after = () -> begin
    c = NO_DENSE_G_COUNTERS[]
    println()
    println("=== Dispatch-proof counters after the REAL D4 KNITRO reduced solve ===")
    println("ZC_EZ_BACKEND_DEFAULT[]   = ", ZC_EZ_BACKEND_DEFAULT[])
    println("ZC_GRAM_BACKEND_DEFAULT[] = ", ZC_GRAM_BACKEND_DEFAULT[])
    println("winner_cross_hessian_calls = ", c.winner_cross_hessian_calls)
    println("dense_cross_hessian_calls  = ", c.dense_cross_hessian_calls)
    ok = ZC_EZ_BACKEND_DEFAULT[] === :drawmajor_v2 && ZC_GRAM_BACKEND_DEFAULT[] === :blas_syrk &&
         c.winner_cross_hessian_calls > 0 && c.dense_cross_hessian_calls == 0
    println(ok ?
        "\nPASS  dispatch proof: drawmajor_v2/blas_syrk production defaults selected with zero explicit kwargs, winner-based cross-Hessian path fired $(c.winner_cross_hessian_calls) times, zero dense fallbacks" :
        "\nFAIL  dispatch proof")
    ok
end

include(joinpath(D4X, "test_profiled_originzc_d4_fg_and_solve_gate_2026-08-01.jl"))
# NO_DENSE_G_COUNTERS is reset to defaults at process start (0s) and this gate script is the ONLY
# thing that ran a Hessian callback in this process, so its own counters already reflect exactly
# the reduced-origin-ZC KNITRO solve above -- no explicit reset_no_dense_g_counters!() needed
# (would just be resetting to what it already is at this point in a fresh process).
@assert reset_counters_after()
