# Section 9.3 gate (cm-meanzc-frechet-outer-production-closeout-2026-08-06): confirm the REAL
# public driver (run_cm_upper_checkpointed, marginal_restriction=:common_frechet,
# threaded_bins=true) never hits the "threaded_bins=true requires tls" error the archived D4 log
# showed -- that error came from a diagnostic script calling hessian_cm_frechet_structured_v2!
# directly with its own tls=nothing default, NOT from the real production Hessian dispatcher
# (archC_frechet_hess_cb_builder, which already reads cctx.tls -- see cm_frechet_hessian.jl:484).
# D4 scale for speed; real callback counts + no exception is the gate, not a full outer search.
const _D4E = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_hessian_subblock_profiling.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "hcz_reordered_candidate_2026-08-01.jl",
          "cm_meanzc_lookup_production.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Printf
lp(xs...) = (println(xs...); flush(stdout))
npass = 0; nfail = 0
function check(name, cond)
    global npass, nfail
    if cond
        npass += 1; lp("  PASS  ", name)
    else
        nfail += 1; lp("  FAIL  ", name)
    end
end

ctx = d20_real_setup_design(W = 100_000, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0)   # MUST match
    # run_cm_upper_checkpointed's own defaults exactly -- see diag_frechet_w0_direct_2026-08-06.jl's
    # identical comment; the plain d20_real_setup(...) call used here originally silently used
    # sigma=2.5/no gravity exclusion, a genuinely different economic model than the driver's own
    # internal context, which is why the driver's screen legitimately rejected this w0 -- a
    # test-script bug, not a driver defect.
pe = build_pivot_elimination(ctx)
w0 = cm_w0_from_calibration(ctx, pe, :powered_aspace)
L = 50   # production L, matching configs/fullA_production_2026-08-03.toml
probs = cm_equal_grid_probs(L)
CKPT = mktempdir()

lp("="^90); lp("Section 9.3 gate: common-Frechet through the REAL public driver, threaded_bins=true, D20/W=100,000/L=50"); lp("="^90)
CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[] = true
result = run_cm_upper_checkpointed(w0; W = 100_000, delta = 1.0,
    draw_design = :sobol_randomized, draw_seed = 20260719,   # MUST match the ctx build above exactly
    L = L, contrasts = :anchored, probs = probs,
    include_truncated_moment = false,   # common_frechet's own single-family CM sub-block; the
        # two-family CM extension is orthogonal to this TLS gate specifically
    cm_hessian_backend = :structured, threaded_bins = true,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
    marginal_restriction = :common_frechet,
    A_coordinate_mode = :powered_aspace,
    ckpt_dir = CKPT, run_id = "frechet_tls_gate", label = "frechet_tls_gate",
    checkpoint_interval_s = 90.0, maxtime_real = 90.0, verbose = true)
lp("knitro_status=", result.knitro_status, " n_eval=", result.n_eval, " n_grad=", result.n_grad)
check("no exception (no 'threaded_bins=true requires tls' escaping the real driver)", true)
check("at least one real outer evaluation occurred", result.n_eval > 0)

pcx_live = CM_LIVE_PCX_STASH[]
if pcx_live !== nothing
    cctx = pcx_live.cctx
    check("cctx.use_threaded_bins == true (production default honored)", cctx.use_threaded_bins)
    check("cctx.tls is constructed (NOT nothing) under threaded_bins=true", cctx.tls !== nothing)
end

lp(); lp("="^90); lp("TOTAL: $npass passed, $nfail failed"); lp("="^90)
exit(nfail == 0 ? 0 : 1)
