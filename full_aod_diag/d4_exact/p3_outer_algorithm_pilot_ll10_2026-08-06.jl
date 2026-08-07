# fix/fullA-lower-limit-and-hotpath-2026-08-06, task P3: short controlled pilot comparing
# algorithm=auto (-> Direct) vs pin_outer_algorithm=true (CG+L-BFGS) for common-Frechet two-family,
# BOTH now under the corrected inner_lower_limit=-10 (the prior session's own A/B, docs/audits/
# cm-extensions-hotpath-2026-08-06/MASTER.md §10.3, used the OLD lower_limit=-50 -- stale for this
# specific question since that fix changes attempt cost/count materially, §11.3/11.4 of that same
# report). Real D20/W=20,000, run A then B SEQUENTIALLY on the SAME pinned core set (not
# concurrently) for a clean, uncontended, apples-to-apples wall-clock comparison -- matching the
# task's own "same starting point, manifest, budget, cores, warm-start policy" requirement.
#
# maxtime_real=300s per run (a genuinely SHORT pilot per the task's own "lower priority" framing
# and this session's remaining time budget -- NOT a full matched-budget replication of the prior
# session's own 900-1074s runs; fewer major iterations will complete, but the per-attempt cost and
# per-major-iteration wall-clock ratios this pilot targets are still meaningful at this budget).
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
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl",
          "cm_frechet_cplus.jl", "cm_meanzc_lookup_production.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Printf

lp(xs...) = (println(xs...); flush(stdout))
lp("nthreads()=", Threads.nthreads())

const W = 20_000
const DELTA = 1.0
const L = 50
const BUDGET = 300.0

ctx = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
    inner_lower_limit = -10.0)
pe = build_pivot_elimination(ctx)
w0 = cm_w0_from_calibration(ctx, pe, :powered_aspace)
probs = cm_equal_grid_probs(L)

function run_one(label, pin)
    CKPT = mktempdir()
    lp("="^90); lp("P3 run: ", label, " (pin_outer_algorithm=", pin, ")"); lp("="^90)
    t0 = time()
    result = run_cm_upper_checkpointed(w0; W = W, delta = DELTA,
        draw_design = :sobol_randomized, draw_seed = 20260719,
        L = L, contrasts = :anchored, probs = probs,
        include_truncated_moment = true,
        cm_hessian_backend = :structured, threaded_bins = true,
        exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
        marginal_restriction = :common_frechet,
        A_coordinate_mode = :powered_aspace,
        inner_lower_limit = -10.0,
        pin_outer_algorithm = pin,
        ckpt_dir = CKPT, run_id = "p3_$label", label = "p3_$label",
        checkpoint_interval_s = 90.0, maxtime_real = BUDGET, verbose = true)
    wall = time() - t0
    @printf("RESULT[%s]: wall=%.1fs knitro_status=%s n_eval=%d n_grad=%d\n",
        label, wall, string(result.knitro_status), result.n_eval, result.n_grad)
    return (label = label, wall = wall, result = result)
end

resA = run_one("auto_direct", false)
resB = run_one("cg_lbfgs", true)

lp("="^90); lp("P3 SUMMARY (common-Frechet two-family, D20/W=$W, budget=$(BUDGET)s, inner_lower_limit=-10.0)"); lp("="^90)
for r in (resA, resB)
    @printf("%-15s wall=%7.1fs  n_eval=%4d  n_grad=%4d  knitro_status=%s\n",
        r.label, r.wall, r.result.n_eval, r.result.n_grad, string(r.result.knitro_status))
end
lp("="^90); lp("DONE"); lp("="^90)
