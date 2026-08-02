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

# ================================================================================================
# Phase 7 (dispatch-counters, 2026-08-02) extension: BACKEND-SPECIFIC runtime dispatch counters,
# isolated per scenario (REDUCED vs FULL) via an explicit counter reset between runs -- mirrors
# the CM+ZC dispatch-proof test's identical extension (test_zc_lane_cmzc_dispatch_proof_
# 2026-08-02.jl). Origin-ZC has no CM-grid block, hence no H_CZ/draw_chunk_reordered concern here
# (confirmed: OriginZCCoreHessCtx has no hcz_prep_backend field at all).
#
# Performance closeout task (2026-08-02), Section 6: this test used to document a confirmed gap
# (the REDUCED/PROFILED branch in archA_partitioned_hess_cb_builder called
# winner_pair_cross_hessian_zc_block! unconditionally for H_EZ, never consulting
# octx.zc_ez_backend). That gather is now wired through the same octx.zc_ez_backend dispatch the
# non-profiled branch already used, mirroring CM+ZC's own H_EM fix -- drawmajor_v2's W-scale
# scatter loop itself was not touched. This test now asserts the REDUCED path genuinely
# dispatches, matching FULL.
println()
println("=== Phase 7: backend-specific dispatch counters (isolated per scenario) ===")

reset_no_dense_g_counters!()
# Force a genuine cold restart (obj.x .= NaN) before this isolated re-solve -- reusing the SAME
# `obj` (already at its converged x from the include'd gate script's own Part2 run) with a warm
# start would let KNITRO re-confirm optimality in ~0 Newton iterations, calling the Hessian
# callback zero times and making this isolated check vacuous (per this repo's own CLAUDE.md note:
# warm/cold affects speed only, never whether/where it converges -- forcing cold is safe here,
# purely to guarantee genuine fresh Hessian evaluations get counted).
ctx_cm_reduced.obj.use_cached_x = false
ctx_cm_reduced.obj.x .= NaN
base_reduced2 = archOZ_base_state(x_free_calib, νfull0, ctx_cm_reduced)
cr = NO_DENSE_G_COUNTERS[]
@printf("REDUCED (profiled_layout set -- this branch's own production path):\n")
@printf("  winner_cross_hessian_calls=%d  dense_cross_hessian_calls=%d\n", cr.winner_cross_hessian_calls, cr.dense_cross_hessian_calls)
@printf("  blas_syrk:    dispatch=%d  fallback=%d\n", cr.blas_syrk_dispatch_count, cr.blas_syrk_fallback_count)
@printf("  drawmajor_v2: dispatch=%d  fallback=%d\n", cr.drawmajor_v2_dispatch_count, cr.drawmajor_v2_fallback_count)
reduced_blas_syrk_ok = cr.blas_syrk_dispatch_count > 0 && cr.blas_syrk_fallback_count == 0
reduced_drawmajor_ok = cr.drawmajor_v2_dispatch_count > 0 && cr.drawmajor_v2_fallback_count == 0
check("REDUCED path: blas_syrk genuinely dispatches (positive>0, negative==0)", reduced_blas_syrk_ok)
check("REDUCED path: drawmajor_v2 now genuinely dispatches (positive>0, negative==0) -- gap closed", reduced_drawmajor_ok)

reset_no_dense_g_counters!()
ctx_cm_full.obj.use_cached_x = false
ctx_cm_full.obj.x .= NaN
base_full2 = archOZ_base_state(x_free_calib, νfull0, ctx_cm_full)
cf_ = NO_DENSE_G_COUNTERS[]
@printf("\nFULL (profiled_layout=nothing -- non-reduced legacy family, zero explicit backend kwargs):\n")
@printf("  winner_cross_hessian_calls=%d  dense_cross_hessian_calls=%d\n", cf_.winner_cross_hessian_calls, cf_.dense_cross_hessian_calls)
@printf("  blas_syrk:    dispatch=%d  fallback=%d\n", cf_.blas_syrk_dispatch_count, cf_.blas_syrk_fallback_count)
@printf("  drawmajor_v2: dispatch=%d  fallback=%d\n", cf_.drawmajor_v2_dispatch_count, cf_.drawmajor_v2_fallback_count)
full_blas_syrk_ok = cf_.blas_syrk_dispatch_count > 0 && cf_.blas_syrk_fallback_count == 0
full_drawmajor_ok = cf_.drawmajor_v2_dispatch_count > 0 && cf_.drawmajor_v2_fallback_count == 0
check("FULL (non-reduced) path: blas_syrk genuinely dispatches (positive>0, negative==0)", full_blas_syrk_ok)
check("FULL (non-reduced) path: drawmajor_v2 genuinely dispatches (positive>0, negative==0) -- H_EZ's if/elseif chain IS wired to zc_ez_backend here", full_drawmajor_ok)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
