# Section 11 gate (cm-meanzc-frechet-outer-production-closeout-2026-08-06): W=5,000 and
# W=20,000 tiers, both families, through the REAL public driver -- short outer runs, threaded
# backends, checkpoint-write + resume round-trip. Real production settings (sigma=3, exclude_
# diagonal_gravity=true, Brazil-Korea exclusion, sobol_randomized draws) throughout.
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

L = 50
probs = cm_equal_grid_probs(L)
GRAV = default_gravity_exclude_cells_brazil_korea()

function ctx_for(W)
    d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
        draw_design = :sobol_randomized, draw_seed = 20260719,
        destination_sample = :exclude_row, exclude_diagonal_gravity = true,
        gravity_exclude_cells = GRAV, σHat = 3.0)
end

for W in (5_000, 20_000)
    lp("="^90); lp("=== CM+ZC tier, W=$W ==="); lp("="^90)
    ctx = ctx_for(W)
    pe = build_pivot_elimination(ctx)
    K_mean = 1; K_pair = 1
    nu_bounds = meanzc_default_nu_bounds(ctx, K_mean)
    pcx_nu0 = build_cm_meanzc_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = :anchored,
        include_truncated_moment = true, meanzc_basis = :direct, probs = probs, moment_representation = :operator)
    nu1_guess = sum(pcx_nu0.aug.Zraw_all[1]) / length(pcx_nu0.aug.Zraw_all[1])
    eta0 = [log(nu1_guess)]
    w0 = vcat(cm_w0_from_calibration(ctx, pe, :powered_aspace), eta0)
    CKPT = mktempdir()
    result_a = run_cm_upper_checkpointed(w0; W = W, delta = 1.0, draw_design = :sobol_randomized, draw_seed = 20260719,
        L = L, contrasts = :anchored, probs = probs, include_truncated_moment = true,
        cm_hessian_backend = :structured, threaded_bins = true,
        cm_extension = :cm_plus_moments, meanzc_K_mean = K_mean, meanzc_K_pair = K_pair,
        meanzc_nu_bounds = nu_bounds, meanzc_basis = :direct,
        exclude_diagonal_gravity = true, gravity_exclude_cells = GRAV, σHat = 3.0, destination_sample = :exclude_row,
        A_coordinate_mode = :powered_aspace,
        ckpt_dir = CKPT, run_id = "cmzc_w$(W)", label = "cmzc_w$(W)",
        checkpoint_interval_s = 5.0, maxtime_real = 60.0, verbose = false)
    @printf "  [cmzc W=%d] n_eval=%d n_grad=%d knitro_status=%d best=%s\n" W result_a.n_eval result_a.n_grad result_a.knitro_status string(result_a.best !== nothing)
    check("cmzc W=$W: at least one real gradient evaluation", result_a.n_grad > 0)
    check("cmzc W=$W: checkpoint file written", isfile(result_a.ckpt_path))

    result_b = run_cm_upper_checkpointed(nothing; W = W, delta = 1.0, draw_design = :sobol_randomized, draw_seed = 20260719,
        L = L, contrasts = :anchored, probs = probs, include_truncated_moment = true,
        cm_hessian_backend = :structured, threaded_bins = true,
        cm_extension = :cm_plus_moments, meanzc_K_mean = K_mean, meanzc_K_pair = K_pair,
        meanzc_nu_bounds = nu_bounds, meanzc_basis = :direct,
        exclude_diagonal_gravity = true, gravity_exclude_cells = GRAV, σHat = 3.0, destination_sample = :exclude_row,
        A_coordinate_mode = :powered_aspace,
        ckpt_dir = CKPT, run_id = "cmzc_w$(W)", label = "cmzc_w$(W)_resume",
        checkpoint_interval_s = 30.0, maxtime_real = 20.0, verbose = false, resume_from = result_a.ckpt_path)
    @printf "  [cmzc W=%d resume] n_eval=%d n_grad=%d knitro_status=%d\n" W result_b.n_eval result_b.n_grad result_b.knitro_status
    check("cmzc W=$W: resume carries n_eval forward (>= pre-resume count)", result_b.n_eval >= result_a.n_eval)
end

for W in (5_000, 20_000)
    lp("="^90); lp("=== common-Frechet tier, W=$W ==="); lp("="^90)
    ctx = ctx_for(W)
    pe = build_pivot_elimination(ctx)
    w0 = cm_w0_from_calibration(ctx, pe, :powered_aspace)
    CKPT = mktempdir()
    result_a = run_cm_upper_checkpointed(w0; W = W, delta = 1.0, draw_design = :sobol_randomized, draw_seed = 20260719,
        L = L, contrasts = :anchored, probs = probs, include_truncated_moment = false,
        cm_hessian_backend = :structured, threaded_bins = true,
        exclude_diagonal_gravity = true, gravity_exclude_cells = GRAV, σHat = 3.0, destination_sample = :exclude_row,
        marginal_restriction = :common_frechet, A_coordinate_mode = :powered_aspace,
        ckpt_dir = CKPT, run_id = "frechet_w$(W)", label = "frechet_w$(W)",
        checkpoint_interval_s = 5.0, maxtime_real = 60.0, verbose = false)
    @printf "  [frechet W=%d] n_eval=%d n_grad=%d knitro_status=%d\n" W result_a.n_eval result_a.n_grad result_a.knitro_status
    check("frechet W=$W: at least one real gradient evaluation", result_a.n_grad > 0)
    check("frechet W=$W: checkpoint file written", isfile(result_a.ckpt_path))

    result_b = run_cm_upper_checkpointed(nothing; W = W, delta = 1.0, draw_design = :sobol_randomized, draw_seed = 20260719,
        L = L, contrasts = :anchored, probs = probs, include_truncated_moment = false,
        cm_hessian_backend = :structured, threaded_bins = true,
        exclude_diagonal_gravity = true, gravity_exclude_cells = GRAV, σHat = 3.0, destination_sample = :exclude_row,
        marginal_restriction = :common_frechet, A_coordinate_mode = :powered_aspace,
        ckpt_dir = CKPT, run_id = "frechet_w$(W)", label = "frechet_w$(W)_resume",
        checkpoint_interval_s = 30.0, maxtime_real = 20.0, verbose = false, resume_from = result_a.ckpt_path)
    @printf "  [frechet W=%d resume] n_eval=%d n_grad=%d knitro_status=%d\n" W result_b.n_eval result_b.n_grad result_b.knitro_status
    check("frechet W=$W: resume carries n_eval forward (>= pre-resume count)", result_b.n_eval >= result_a.n_eval)
end

lp(); lp("="^90); lp("TOTAL: $npass passed, $nfail failed"); lp("="^90)
exit(nfail == 0 ? 0 : 1)
