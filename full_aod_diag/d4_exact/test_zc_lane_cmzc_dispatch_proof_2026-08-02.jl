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

# ================================================================================================
# Phase 7 (dispatch-counters, 2026-08-02) extension: BACKEND-SPECIFIC runtime dispatch counters.
# The checks above only prove "the winner-aware (non-dense-G) cross-Hessian path fired" -- NOT
# which specific optimized kernel backend ran within that path. This section adds a genuine
# runtime proof, per backend, using the NEW no_dense_g_counters.jl fields
# (blas_syrk/drawmajor_v2/draw_chunk_reordered _dispatch_count/_fallback_count), isolating each
# scenario with an explicit counter reset so the attribution is exact (not mixed across runs).
#
# Performance closeout task (2026-08-02), Section 6/7: this test used to document a confirmed gap
# (drawmajor_v2/draw_chunk_reordered architecturally unreachable on the reduced/profiled path,
# because the profiled H_EM gather in _fill_cm_HEE! called winner_pair_cross_hessian_zc_block!
# unconditionally, and the profiled H_CZ gather in hessian_cm_structured! called
# bin_zc_cross_hessian_fill! directly, bypassing hcz_prep_dispatch! entirely). Both gathers are now
# wired through the SAME dispatch pattern the non-profiled/threaded paths already used
# (cctx.zc_ez_backend for H_EM; cctx.hcz_prep_backend via hcz_prep_dispatch! for H_CZ) -- see
# cm_hessian_architectures.jl's profiled branches. drawmajor_v2's own W-scale scatter loop
# (hez_drawmajor_v2_candidate_2026-08-01.jl) was not rewritten; it gained the same
# use_profiled_correction branch the serial/threaded kernels already had, gated exactly the way
# those kernels gate it (validated bit-identical to the serial kernel at D4,
# test_profiled_hez_drawmajor_v2_d4_2026-08-02.jl). hcz_prep_dispatch! itself was not touched at
# all -- the profiled H_CZ gather now simply calls it instead of the bare fallback kernel. This
# test now asserts the REDUCED path genuinely dispatches through both backends, matching FULL.
println()
println("=== Phase 7: backend-specific dispatch counters (isolated per scenario) ===")

reset_no_dense_g_counters!()
# Force a genuine cold restart before each isolated re-solve below -- reusing an `obj` already at
# its converged x (from the include'd gate script's own Part2 run, or a prior scenario here) with a
# warm start would let KNITRO re-confirm optimality in ~0 Newton iterations, calling the Hessian
# callback zero times and making the check vacuous (per CLAUDE.md: warm/cold affects speed only,
# never whether/where it converges -- forcing cold is safe, purely to guarantee fresh Hessian calls).
ctx_cm_reduced.obj.use_cached_x = false
ctx_cm_reduced.obj.x .= NaN
base_reduced2 = archC_meanzc_base_state(x_free_calib, νvec0, ctx_cm_reduced, cctx_reduced)
cr = NO_DENSE_G_COUNTERS[]
@printf("REDUCED (profiled_layout set, threaded_bins=false -- this branch's own production path):\n")
@printf("  winner_cross_hessian_calls=%d  dense_cross_hessian_calls=%d\n", cr.winner_cross_hessian_calls, cr.dense_cross_hessian_calls)
@printf("  blas_syrk:            dispatch=%d  fallback=%d\n", cr.blas_syrk_dispatch_count, cr.blas_syrk_fallback_count)
@printf("  drawmajor_v2:         dispatch=%d  fallback=%d\n", cr.drawmajor_v2_dispatch_count, cr.drawmajor_v2_fallback_count)
@printf("  draw_chunk_reordered: dispatch=%d  fallback=%d\n", cr.draw_chunk_reordered_dispatch_count, cr.draw_chunk_reordered_fallback_count)
reduced_blas_syrk_ok = cr.blas_syrk_dispatch_count > 0 && cr.blas_syrk_fallback_count == 0
reduced_drawmajor_ok = cr.drawmajor_v2_dispatch_count > 0 && cr.drawmajor_v2_fallback_count == 0
reduced_hcz_ok = cr.draw_chunk_reordered_dispatch_count > 0 && cr.draw_chunk_reordered_fallback_count == 0
check("REDUCED path: blas_syrk genuinely dispatches (positive>0, negative==0)", reduced_blas_syrk_ok)
check("REDUCED path: drawmajor_v2 now genuinely dispatches (positive>0, negative==0) -- gap closed", reduced_drawmajor_ok)
check("REDUCED path: draw_chunk_reordered now genuinely dispatches (positive>0, negative==0) -- gap closed", reduced_hcz_ok)

reset_no_dense_g_counters!()
ctx_cm_full.obj.use_cached_x = false
ctx_cm_full.obj.x .= NaN
base_full2 = archC_meanzc_base_state(x_free_calib, νvec0, ctx_cm_full, cctx_full)
cf_ = NO_DENSE_G_COUNTERS[]
@printf("\nFULL (profiled_layout=nothing, threaded_bins=false -- non-reduced legacy family):\n")
@printf("  winner_cross_hessian_calls=%d  dense_cross_hessian_calls=%d\n", cf_.winner_cross_hessian_calls, cf_.dense_cross_hessian_calls)
@printf("  blas_syrk:            dispatch=%d  fallback=%d\n", cf_.blas_syrk_dispatch_count, cf_.blas_syrk_fallback_count)
@printf("  drawmajor_v2:         dispatch=%d  fallback=%d\n", cf_.drawmajor_v2_dispatch_count, cf_.drawmajor_v2_fallback_count)
@printf("  draw_chunk_reordered: dispatch=%d  fallback=%d\n", cf_.draw_chunk_reordered_dispatch_count, cf_.draw_chunk_reordered_fallback_count)
full_blas_syrk_ok = cf_.blas_syrk_dispatch_count > 0 && cf_.blas_syrk_fallback_count == 0
full_drawmajor_ok = cf_.drawmajor_v2_dispatch_count > 0 && cf_.drawmajor_v2_fallback_count == 0
check("FULL (non-reduced) path: blas_syrk genuinely dispatches (positive>0, negative==0)", full_blas_syrk_ok)
check("FULL (non-reduced) path: drawmajor_v2 genuinely dispatches (positive>0, negative==0) -- H_EM's if/elseif chain IS wired to zc_ez_backend here", full_drawmajor_ok)

# Third scenario: a FULL (non-reduced) cctx built with threaded_bins LEFT AT ITS PRODUCTION
# DEFAULT (true, not overridden) -- the ONLY way to reach hessian_cm_structured_v2!/
# hcz_prep_dispatch!, hence the only way :draw_chunk_reordered can genuinely fire. Zero explicit
# backend/threading kwargs on this cctx build (core_hessian_backend/zc_cross_hessian_backend are
# orthogonal, already-required overrides for THIS family shared with every other gate in this repo).
reset_no_dense_g_counters!()
cctx_full_threaded = build_cm_meanzc_bin_ctx(ctx, aug_full; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin)
@printf("\ncctx_full_threaded: use_threaded_bins=%s  hcz_prep_backend=%s  zc_gram_backend=%s  zc_ez_backend=%s\n",
    cctx_full_threaded.use_threaded_bins, cctx_full_threaded.hcz_prep_backend, cctx_full_threaded.zc_gram_backend, cctx_full_threaded.zc_ez_backend)
threaded_ok = false
try
    ctx_cm_full.obj.use_cached_x = false
    ctx_cm_full.obj.x .= NaN
    base_full_threaded = archC_meanzc_base_state(x_free_calib, νvec0, ctx_cm_full, cctx_full_threaded)
    ct = NO_DENSE_G_COUNTERS[]
    @printf("FULL, threaded_bins=true (genuine zero-kwarg production default -- proves draw_chunk_reordered):\n")
    @printf("  inner_status=%d\n", base_full_threaded.inner_status)
    @printf("  blas_syrk:            dispatch=%d  fallback=%d\n", ct.blas_syrk_dispatch_count, ct.blas_syrk_fallback_count)
    @printf("  drawmajor_v2:         dispatch=%d  fallback=%d\n", ct.drawmajor_v2_dispatch_count, ct.drawmajor_v2_fallback_count)
    @printf("  draw_chunk_reordered: dispatch=%d  fallback=%d\n", ct.draw_chunk_reordered_dispatch_count, ct.draw_chunk_reordered_fallback_count)
    # NOTE: try/catch introduces its own scope in Julia -- without `global` here, this assignment
    # would silently shadow the OUTER `threaded_ok` (declared before the try) with a new local,
    # leaving the outer one at its initial `false` forever regardless of what happens in here.
    # Caught live: an earlier version of this exact block printed all-correct dispatch counts but
    # still reported FAIL, root-caused to precisely this scoping gap.
    global threaded_ok = base_full_threaded.inner_status in (0, -100, -101, -103) &&
        ct.draw_chunk_reordered_dispatch_count > 0 && ct.draw_chunk_reordered_fallback_count == 0 &&
        ct.blas_syrk_dispatch_count > 0 && ct.blas_syrk_fallback_count == 0 &&
        ct.drawmajor_v2_dispatch_count > 0 && ct.drawmajor_v2_fallback_count == 0
catch e
    println("FULL threaded_bins=true solve threw: ", sprint(showerror, e))
    showerror(stdout, e, catch_backtrace())
    println()
    global threaded_ok = false
end
check("FULL threaded_bins=true (true zero-kwarg production default): ALL THREE optimized backends genuinely dispatch (positive>0, negative==0 each)", threaded_ok)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
