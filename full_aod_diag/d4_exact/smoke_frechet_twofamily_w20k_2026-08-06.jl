# Task section 6: short real public-driver outer smoke for common-Frechet TWO-FAMILY
# (include_truncated_moment=true, cdf_plus_power), D20/W=20,000, real production manifest.
const _D4E = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_hessian_subblock_profiling.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "shared_a_gradient.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
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
using Printf, Dates
lp(xs...) = (println(xs...); flush(stdout))

W = 20_000; L = 50
probs = cm_equal_grid_probs(L)
GRAV = default_gravity_exclude_cells_brazil_korea()
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :sobol_randomized, draw_seed = 20260719,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true, gravity_exclude_cells = GRAV, σHat = 3.0)
pe = build_pivot_elimination(ctx)
w0 = cm_w0_from_calibration(ctx, pe, :powered_aspace)

OUT = joinpath(_D4E, "..", "..", "..", "..", "repo_scratch", "cm-extensions-gradient-and-production-final-2026-08-06", "frechet_2fam_smoke_w20k")
isdir(OUT) || mkpath(OUT)

t0 = time()
result = run_cm_upper_checkpointed(w0;
    W = W, delta = 1.0, draw_design = :sobol_randomized, draw_seed = 20260719, L = L, contrasts = :anchored, probs = probs,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true, gravity_exclude_cells = GRAV, σHat = 3.0,
    marginal_restriction = :common_frechet, include_truncated_moment = true,
    ckpt_dir = OUT, run_id = "frechet_2fam_smoke", label = "frechet_2fam_smoke",
    checkpoint_interval_s = 60.0, maxtime_real = 900.0, verbose = true)
wall = time() - t0

lp("="^100)
@printf("RESULT frechet_2fam_smoke: wall=%.1fs knitro_status=%s n_eval=%d n_grad=%d kappa=%s\n",
    wall, string(result.knitro_status), result.n_eval, result.n_grad, string(result.kappa))
lp("="^100)
