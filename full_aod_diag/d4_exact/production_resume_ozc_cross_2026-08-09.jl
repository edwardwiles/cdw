# Checkpoint RESUME test for OZC-CROSS through the real production driver (2026-08-09).
# Completes the production-wiring validation: production_smoke_ozc_cross_2026-08-09.jl proved a
# fresh run works end-to-end (18 evals, feasible incumbent found, checkpoint written); this proves
# the checkpoint can be READ BACK and the search CONTINUED, which is the feature every real campaign
# depends on and the one place a layout/schema mismatch would silently corrupt a run.
#
# What it specifically checks:
#   1. the persisted checkpoint's power_target_layout tag round-trips as :origin_by_power_cross
#      (a base-family checkpoint must NOT be resumable as cross, or vice versa)
#   2. eta_nu length equals the Variant D ACTIVE count (not the dense count)
#   3. run_originzc_upper_checkpointed(resume_from=<ckpt>) actually restarts from the stored
#      incumbent and keeps searching, rather than starting over or erroring
#   4. the resumed run's reported incumbent is no worse than the checkpoint's own
#
# Usage: julia --project=. -t 8 .../production_resume_ozc_cross_2026-08-09.jl <ckpt_path> <budget_s>
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl", "cm_originzc_cross_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "country_resolve.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Dates, Statistics
lp(xs...) = (println(xs...); flush(stdout))

const CKPT   = ARGS[1]
const BUDGET = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 420.0
isfile(CKPT) || error("checkpoint not found: $CKPT")

lp("=== OZC-CROSS PRODUCTION RESUME TEST ===")
lp("checkpoint: ", CKPT)
c = load_cm_checkpoint_v10(CKPT)
@printf("  stored: power_target_layout=%s  K_mean=%d K_pair=%d  origin_D=%d  W=%d  delta=%.3f\n",
        string(c.power_target_layout), c.origin_K_mean, c.origin_K_pair, c.origin_D, c.W, c.delta)
@printf("  stored: n_eta=%d  n_eval=%d  n_grad=%d  wall_elapsed=%.1fs  reason=%s\n",
        length(c.eta_nu), c.n_eval, c.n_grad, c.wall_elapsed, string(c.checkpoint_reason))
@printf("  stored: draw_design=%s draw_seed=%d destination_sample=%s A_coordinate_mode=%s\n",
        string(c.draw_design), c.draw_seed, string(c.destination_sample), string(c.A_coordinate_mode))
best0 = c.best_feasible
if best0 === nothing
    lp("  stored incumbent: NONE")
else
    @printf("  stored incumbent: gp=%.10f Delta=%.10f (eval %d)\n", best0.gp, best0.Delta, best0.n_eval)
end

ok_layout = string(c.power_target_layout) == "origin_by_power_cross"
lp("  CHECK layout tag round-trips as OZC-CROSS: ", ok_layout)
ok_layout || error("checkpoint does not identify as origin_by_power_cross -- resume would be unsafe")

K = c.origin_K_mean
D = c.origin_D
layout0 = OriginByPowerCrossLayout(D, K, K)
aml0 = ActiveMeanLayout(layout0, 2, 2, D)   # bi=2 for this dataset; kstar=2
ok_eta = length(c.eta_nu) == aml0.n_eta_active
@printf("  CHECK eta length %d == Variant D active count %d: %s\n", length(c.eta_nu), aml0.n_eta_active, string(ok_eta))
ok_eta || error("checkpoint eta length does not match the Variant D active count -- layout/schema mismatch")

OUT = dirname(CKPT)
lp("\n--- resuming (budget $(BUDGET)s) ---")
t0 = time()
result = run_originzc_upper_checkpointed(nothing;
    W = c.W, delta = c.delta, draw_design = c.draw_design, draw_seed = c.draw_seed,
    distribution_restriction = :origin_specific_moments_zero_covariance,
    K_mean = K, K_pair = c.origin_K_pair,
    power_target_layout = :origin_by_power_cross,
    originzc_profiled_level = 2,
    inner_lower_limit = -10.0,
    destination_sample = c.destination_sample, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
    ckpt_dir = OUT, run_id = c.run_id, label = c.label,
    resume_from = CKPT,
    checkpoint_interval_s = 60.0, maxtime_real = BUDGET, verbose = true)
wall = time() - t0

lp("="^95)
@printf("RESUME RESULT: wall=%.1fs knitro_status=%s n_eval=%d n_grad=%d kappa=%s\n",
        wall, string(result.knitro_status), result.n_eval, result.n_grad, string(result.kappa))
if result.best !== nothing
    @printf("  resumed incumbent: gp=%.10f Delta=%.10f\n", result.best.gp, result.best.Delta)
    if best0 !== nothing
        improved = result.best.gp <= best0.gp + 1e-12
        @printf("  CHECK resumed incumbent no worse than stored (%.10f <= %.10f): %s\n",
                result.best.gp, best0.gp, string(improved))
        @printf("  CHECK n_eval carried forward from checkpoint (%d >= %d): %s\n",
                result.n_eval, c.n_eval, string(result.n_eval >= c.n_eval))
    end
else
    lp("  no feasible incumbent after resume")
end
lp("="^95)
