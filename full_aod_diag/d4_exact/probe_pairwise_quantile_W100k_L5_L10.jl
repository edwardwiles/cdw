_D4E = joinpath(@__DIR__)
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "compressed_factual_buffer_reuse.jl", "cm_screen_bridge.jl", "gradient_workspace.jl",
          "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl",
          "cm_outer_driver.jl", "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl",
          "cm_meanzc_production.jl", "cm_meanzc_cplus.jl", "incumbent_logic.jl", "cm_checkpoint.jl",
          "cm_originzc_target_layout.jl", "zc_restriction_operator_ragged.jl", "cm_aspace_coordinate.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_config.jl",
          "cm_originzc_checkpoint.jl", "direction_bounds.jl", "cm_frechet_hessian.jl", "cm_frechet_level.jl",
          "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl", "country_resolve.jl",
          "cross_delta_cache.jl", "compressed_moments.jl", "canonical_price_precompute_workspace.jl",
          "hard_score_b_cache.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "lfix_buffer_reuse.jl", "lfix_base_workspace_pooled.jl", "lfix_kbplus_workspace.jl",
          "bandwidth_cache_policy.jl", "fast_range_screen.jl", "dual_bank.jl", "negative_cache.jl",
          "dual_bank_ab_harness.jl", "reusable_context.jl", "organic_failure_capture.jl",
          "multistart_seed_generator.jl",
          # ---- this family ----
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "operator_psi_bundle.jl",
          "cm_callback_health.jl", "shared_a_gradient.jl", "operator_verification.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_verification.jl",
          "pairwise_quantile_production.jl",
          "pairwise_quantile_outer_production.jl", "pairwise_quantile_checkpoint.jl"]
    include(joinpath(_D4E, f))
end

using LinearAlgebra, Printf
lp(xs...) = (println(xs...); flush(stdout))

# ================================================================================================
# Feasibility probe: is L=10 attainable at the production W=100,000, D=20 (user request 2026-08-10)?
# One VALUE-ONLY inner solve per (W,L) at the real calibration point, delta=50 so the early-abort
# threshold is Inf and cannot be mistaken for a failure. L=5 runs first as the known-good control.
# ================================================================================================
GRAV = default_gravity_exclude_cells_brazil_korea()
CASES = [(100_000, 5), (100_000, 10)]
for (W, L) in CASES
    lp("="^90)
    lp("CASE W=", W, " L=", L, "  n_total_rows=", n_total_rows(20, L), "  n_raw=", (L-1)*20)
    flush(stdout)
    t_ctx = time()
    ctx_raw = d20_real_setup_design(W=W, δ=50.0, find_smallest=true, draw_design=:sobol_randomized,
        draw_seed=20260719, destination_sample=:exclude_row, exclude_diagonal_gravity=true,
        gravity_exclude_cells=GRAV, σHat=3.0, inner_lower_limit=-10.0)
    ctx = attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)
    lp("  ctx built in ", round(time()-t_ctx, digits=1), "s")
    flush(stdout)
    layout = PairwiseQuantileMassLayout(ctx.D, L)
    t_pcx = time()
    pcx = build_pairwise_quantile_production_context(ctx, layout;
        cutoff_source = :empirical_quantile, min_bin_count = max(10, W ÷ (2 * L^2)))
    lp("  production context built in ", round(time()-t_pcx, digits=1), "s")
    flush(stdout)
    geo = build_aspace_geometry(ctx)
    w_cal = cm_w0_from_calibration(ctx, geo.pe, :powered_aspace)
    xf = x_free_from_w(vcat(w_cal[1], cm_z_from_a(w_cal[2:end], cm_fixed_theta(ctx),
        precompute_cm_aspace_xy(ctx), geo.pe)), geo.pe)
    mass0 = uniform_mass_raw(layout)   # mu = 1/L, version A's own fixed target
    lp("  starting REAL KNITRO inner solve at ", Base.Libc.strftime(time()), " ...")
    flush(stdout)
    t0 = time()
    try
        base, v = archPQ_verified_state(xf, mass0, pcx.ctx_cm)
        lp("RESULT W=", W, " L=", L, " -> Delta_dual=", v.Delta_dual, " status=", v.inner_status,
           " class=", classify_inner_result(v), " n_fg=", v.n_fg, " n_hess=", v.n_hess,
           " secs=", round(time()-t0, digits=1))
    catch e
        lp("RESULT W=", W, " L=", L, " -> FAILED secs=", round(time()-t0, digits=1), " : ",
           first(sprint(showerror, e), 250))
    end
    flush(stdout)
end
lp("PROBE COMPLETE")
