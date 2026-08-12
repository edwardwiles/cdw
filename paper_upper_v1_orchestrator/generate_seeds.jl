# paper_upper_v1: real multistart seed generation, driven entirely by the frozen protocol
# manifest's [seeds] block (A_scale/gp_scale must already be frozen -- see scan_seed_scales.jl --
# not "SCAN_REQUIRED" placeholders).
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=10 julia --project=. \
#       paper_upper_v1_orchestrator/generate_seeds.jl protocols/paper_upper_v1.toml <campaign_root>

using TOML, Serialization, Random

const D4E = joinpath(@__DIR__, "..", "full_aod_diag", "d4_exact")
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl", "lfix_cm_aware.jl",
          "cm_hessian_architectures.jl", "cm_production_bundle.jl", "compressed_factual_buffer_reuse.jl", "cm_screen_bridge.jl", "gradient_workspace.jl",
          "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl",
          "cm_outer_driver.jl", "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl",
          "cm_meanzc_cplus.jl", "cm_checkpoint.jl",
          "cm_originzc_target_layout.jl", "zc_restriction_operator_ragged.jl", "cm_aspace_coordinate.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_config.jl",
          "cm_originzc_checkpoint.jl", "direction_bounds.jl", "cm_frechet_hessian.jl", "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl", "country_resolve.jl",
          "cross_delta_cache.jl", "compressed_moments.jl", "canonical_price_precompute_workspace.jl",
          "hard_score_b_cache.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "lfix_buffer_reuse.jl", "lfix_base_workspace_pooled.jl", "lfix_kbplus_workspace.jl",
          "bandwidth_cache_policy.jl", "fast_range_screen.jl", "dual_bank.jl", "negative_cache.jl",
          "dual_bank_ab_harness.jl", "incumbent_logic.jl", "reusable_context.jl", "organic_failure_capture.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl", "cm_originzc_cross_cplus.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl", "cm_meanzc_cross_production.jl", "cm_meanzc_cross_cplus.jl",
          "multistart_seed_generator.jl"]
    include(joinpath(D4E, f))
end

lp(xs...) = (println(xs...); flush(stdout))

const PROTOCOL_TOML = ARGS[1]
const CAMPAIGN_ROOT = ARGS[2]
const MANIFEST = TOML.parsefile(PROTOCOL_TOML)
const SCI = MANIFEST["scientific"]
const SEEDS_CFG = MANIFEST["seeds"]

SEEDS_CFG["A_scale"] isa Number || error("generate_seeds: [seeds].A_scale is not frozen yet (got '$(SEEDS_CFG["A_scale"])') -- run scan_seed_scales.jl first and freeze a real value into the protocol manifest.")
SEEDS_CFG["gp_scale"] isa Number || error("generate_seeds: [seeds].gp_scale is not frozen yet (got '$(SEEDS_CFG["gp_scale"])') -- run scan_seed_scales.jl first.")

grav = default_gravity_exclude_cells_brazil_korea()
lp("Building ctx: W=", SCI["W"], " draw_seed=", SCI["draw_seed"], " draw_design=", SCI["draw_design"])
ctx_raw = d20_real_setup_design(W = SCI["W"], δ = SEEDS_CFG["seed_delta_max"], find_smallest = SCI["find_smallest"],
    draw_design = Symbol(SCI["draw_design"]), draw_seed = SCI["draw_seed"], destination_sample = Symbol(SCI["destination_sample"]),
    exclude_diagonal_gravity = SCI["exclude_diagonal_gravity"], gravity_exclude_cells = grav,
    σHat = SCI["sigma"], inner_lower_limit = SCI["inner_lower_limit"])
ctx = attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)
lp("ctx built: D=", ctx.D, " W=", ctx.W)

family_specs = SEEDS_CFG["family_specs"] == "paper_five_family_seed_specs" ? paper_five_family_seed_specs(ctx) :
    error("generate_seeds: unknown family_specs preset '$(SEEDS_CFG["family_specs"])'")

delta_max = SEEDS_CFG["seed_delta_max"] - SEEDS_CFG["seed_delta_qualification_tolerance"]
out_dir = joinpath(CAMPAIGN_ROOT, SEEDS_CFG["output_dir"])

lp("Generating ", SEEDS_CFG["number_of_starts"], " starts (", SEEDS_CFG["randomized_noncalibration_starts"],
   " randomized + calibration=", SEEDS_CFG["include_calibration"], "), delta_max=", delta_max,
   " A_scale=", SEEDS_CFG["A_scale"], " gp_scale=", SEEDS_CFG["gp_scale"], " -> ", out_dir)

result = generate_multistart_seeds(ctx;
    M = SEEDS_CFG["number_of_starts"], direction = Symbol(SCI["direction"]), delta_max = delta_max,
    family_specs = family_specs, W = SCI["W"], rng_seed = UInt64(SEEDS_CFG["master_rng_seed"]),
    A_scale = Float64(SEEDS_CFG["A_scale"]), gp_scale = Float64(SEEDS_CFG["gp_scale"]),
    max_attempts = SEEDS_CFG["max_attempts"], include_calibration = SEEDS_CFG["include_calibration"],
    min_seed_distance = SEEDS_CFG["min_seed_distance"], radius_mode = Symbol(SEEDS_CFG["radius_mode"]),
    output_dir = out_dir)

lp("="^100)
lp("SEED GENERATION DONE: n_attempted=", result.n_attempted, " n_accepted=", result.n_accepted,
   " stop_reason=", result.stop_reason)
lp("accepted seed ids: ", [s.seed_id for s in result.accepted])
if result.n_accepted < SEEDS_CFG["number_of_starts"]
    lp("!!! WARNING: only ", result.n_accepted, "/", SEEDS_CFG["number_of_starts"],
       " requested starts were accepted within max_attempts=", SEEDS_CFG["max_attempts"],
       " -- per protocol, DO NOT hand-select substitutes. Either raise max_attempts and re-run, ",
       "or report this as a genuine stop per the task brief.")
end
lp("="^100)
