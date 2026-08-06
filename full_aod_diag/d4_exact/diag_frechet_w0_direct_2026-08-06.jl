# Direct (non-KNITRO) diagnostic: call the same function cb_F! calls
# (cm_frechet_production_value_verified_screened) directly on the generic calibration w0, to see
# the REAL Julia exception behind the outer driver's opaque KNITRO -500/-502 status (KNITRO.jl's
# own callback wrapper would otherwise swallow it exactly like the pre-895b99b Pow= bug did).
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

ctx = d20_real_setup_design(W = 100_000, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0)   # MUST match
    # run_cm_upper_checkpointed's own defaults exactly (sigma=3.0, exclude_diagonal_gravity=true,
    # Brazil-Korea exclusion) -- the bare d20_real_setup(...) convenience call used here originally
    # silently defaulted to sigma=2.5/no exclusion, a DIFFERENT economic model than what the real
    # driver's own internal context construction uses. A w0 encoded under one model and decoded/
    # screened under another is not a genuine calibration point in either -- this was a test-script
    # bug (mismatched context construction), not a driver defect. See CLAUDE.md's own top warning
    # ("never let a function default a scientific parameter") -- this is exactly that trap.
pe = build_pivot_elimination(ctx)
w0 = cm_w0_from_calibration(ctx, pe, :powered_aspace)
theta_cm = cm_fixed_theta(ctx); xy_cm = precompute_cm_aspace_xy(ctx)
xf = x_free_from_w(vcat(w0[1], cm_z_from_a(w0[2:end], theta_cm, xy_cm, pe)), pe)
x_free_calib = ctx.θ0_up[ctx.free_idx]
lp("max|xf - x_free_calib| = ", maximum(abs.(xf .- x_free_calib)))

L = 50
probs = cm_equal_grid_probs(L)
lp("=== build_cm_frechet_production_context ===")
pcx = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = :anchored, probs = probs,
    cm_hessian_backend = :structured, threaded_bins = true, include_truncated_moment = false,
    moment_representation = :operator)
pcx = with_screen_counters(pcx)

lp("=== direct call: cm_frechet_production_value_verified_screened(xf, pcx) ===")
try
    K, base, verify = cm_frechet_production_value_verified_screened(xf, pcx; counters = pcx.screen_counters)
    lp("SUCCESS: K=", K, " Delta_dual=", verify.Delta_dual, " verified=", is_verified_success(verify))
catch e
    lp("THREW: typeof=", typeof(e))
    showerror(stdout, e, catch_backtrace())
    println()
end
lp("=== DONE ===")
