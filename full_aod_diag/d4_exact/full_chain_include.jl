# full_chain_include.jl -- the exact include chain continuation_campaign_cell_driver.jl uses,
# extracted so other reusable scripts (delta_star_path_bisection.jl, seed_registry.jl, the K=3
# preflight) can load the same real production driver chain without duplicating this list and
# risking drift between copies. Include order matters (each file depends on ones before it) --
# preserved byte-identical to continuation_campaign_cell_driver.jl's own list.

const D4E_INCLUDE_DIR = @__DIR__

for f in ["c10_d20_production_driver.jl", "flexible_theta.jl", "flexible_theta_aspace_production.jl",
          "outer_coordinate_layout.jl", "c10_d20_production_driver_unified.jl"]
    include(joinpath(D4E_INCLUDE_DIR, f))
end
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
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl", "postmerge_smoke_diagnostics.jl", "cross_hessian_live_stash_2026-07-28.jl"]
    include(joinpath(D4E_INCLUDE_DIR, f))
end
include(joinpath(D4E_INCLUDE_DIR, "continuation_polish_orchestrator.jl"))
include(joinpath(D4E_INCLUDE_DIR, "continuation_polish_run_fn.jl"))
