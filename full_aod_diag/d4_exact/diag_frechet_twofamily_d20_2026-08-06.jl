# Task section 6: two-family common-Fréchet (cdf_plus_power) at real D20 data, calibration point
# through the real production path (archC_frechet_verified_state), D20/W=20,000 then W=100,000.
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
using Printf
lp(xs...) = (println(xs...); flush(stdout))

for W in (20_000, 100_000)
    L = 50
    probs = cm_equal_grid_probs(L)
    GRAV = default_gravity_exclude_cells_brazil_korea()
    ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :sobol_randomized, draw_seed = 20260719,
        destination_sample = :exclude_row, exclude_diagonal_gravity = true, gravity_exclude_cells = GRAV, σHat = 3.0)
    pe = build_pivot_elimination(ctx)
    theta_cm = cm_fixed_theta(ctx); xy_cm = precompute_cm_aspace_xy(ctx)
    x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
    xf_from_w_econ(w_econ) = x_free_from_w(vcat(w_econ[1], cm_z_from_a(w_econ[2:end], theta_cm, xy_cm, pe)))
    w0 = cm_w0_from_calibration(ctx, pe, :powered_aspace)
    xf0 = xf_from_w_econ(w0)

    lp("="^100); lp("=== common-Frechet TWO-FAMILY (cdf_plus_power), D20/W=", W, "/L=", L, " ==="); lp("="^100)
    pcx = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = :anchored, probs = probs,
        cm_hessian_backend = :structured, threaded_bins = true, include_truncated_moment = true,
        moment_representation = :operator)
    lp("[frechet-2fam] context built: n_families=", pcx.cctx.n_families, " Pow!==nothing=", pcx.cctx.Pow !== nothing,
       " inner_fg_backend=", pcx.cctx.inner_fg_backend, " total_marginal_moments=", pcx.aug.ncm)

    t0 = time()
    base, verify = archC_frechet_verified_state(xf0, pcx.ctx_cm, pcx.cctx, pcx.aug.level_targets)
    t1 = time()
    lp("[frechet-2fam] CALIBRATION: t=", round(t1 - t0, digits = 1), "s inner_status=", verify.inner_status,
       " Delta_dual=", verify.Delta_dual, " feasible=", verify.Delta_dual < 1.0,
       " primal_dual_gap=", verify.primal_dual_gap, " max_abs_moment_kkt_resid=", verify.max_abs_moment_kkt_resid)

    w1 = copy(w0); w1[1] -= 2e-4
    xf1 = xf_from_w_econ(w1)
    t2 = time()
    base1, verify1 = archC_frechet_verified_state(xf1, pcx.ctx_cm, pcx.cctx, pcx.aug.level_targets)
    t3 = time()
    lp("[frechet-2fam] NEARBY: t=", round(t3 - t2, digits = 1), "s inner_status=", verify1.inner_status,
       " Delta_dual=", verify1.Delta_dual, " feasible=", verify1.Delta_dual < 1.0,
       " primal_dual_gap=", verify1.primal_dual_gap, " max_abs_moment_kkt_resid=", verify1.max_abs_moment_kkt_resid)
end
lp("=== DONE ===")
