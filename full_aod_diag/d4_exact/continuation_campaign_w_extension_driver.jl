# continuation_campaign_w_extension_driver.jl -- tests whether a larger W (more Monte Carlo draws)
# lets a delta=2 upper-bound point already found at a smaller W push further, by seeding directly
# from that point (no envelope-CSV baseline machinery -- there is no W=250k "original campaign" to
# inherit from; the seed IS the W=100k campaign's own delta=2 result, by direct user request).
#
# Usage:
#   CAMPAIGN_W=<new_W> julia --project=. -t <threads> continuation_campaign_w_extension_driver.jl \
#       <family> <direction> <target_delta> <seed_report_path> <seed_W_label> <output_dir> [explore_budget_s] [polish_budget_s]
#
# <seed_report_path> is a canonical report.jls from a DIFFERENT (smaller-W) run of this same
# campaign (e.g. campaign_output/<family>/upper/delta_2.0/report.jls, the W=100k result).
# <seed_W_label> is just a string for logging/output naming (e.g. "100k") -- CAMPAIGN_W (env) is
# the actual new W this cell runs under.

const D4E = "/bbkinghome/edav/cdw_worktrees/fullA-continuation-polish-2026-08-03/full_aod_diag/d4_exact"
haskey(ENV, "CAMPAIGN_W") || error("set CAMPAIGN_W (e.g. `export CAMPAIGN_W=250000`) before running this script.")

for f in ["c10_d20_production_driver.jl", "flexible_theta.jl", "flexible_theta_aspace_production.jl",
          "outer_coordinate_layout.jl", "c10_d20_production_driver_unified.jl"]
    include(joinpath(D4E, f))
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
    include(joinpath(D4E, f))
end
include(joinpath(D4E, "continuation_polish_orchestrator.jl"))
include(joinpath(D4E, "continuation_polish_run_fn.jl"))

lp(xs...) = (println(xs...); flush(stdout))

const FAMILY = ARGS[1]
const DIRECTION = Symbol(ARGS[2])
const TARGET_DELTA = parse(Float64, ARGS[3])
const SEED_REPORT_PATH = ARGS[4]
const SEED_W_LABEL = ARGS[5]
const OUTPUT_DIR = ARGS[6]
const EXPLORE_BUDGET_S = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 1800.0
const POLISH_BUDGET_S = length(ARGS) >= 8 ? parse(Float64, ARGS[8]) : 1800.0

isdir(OUTPUT_DIR) || mkpath(OUTPUT_DIR)
polish_stage_for(family::String) = family == "unrestricted" ? POLISH_DIRECT_BFGS : POLISH_SQP

lp("=== W-EXTENSION CELL ", FAMILY, " / ", DIRECTION, " / delta=", TARGET_DELTA, " ===")
lp("new_W=", CONTPOLISH_W, " seeded from W=", SEED_W_LABEL, " report at ", SEED_REPORT_PATH)
lp("explore_budget_s=", EXPLORE_BUDGET_S, " polish_budget_s=", POLISH_BUDGET_S)

isfile(SEED_REPORT_PATH) || error("seed report not found: $SEED_REPORT_PATH")
seed_state = load_run_state(SEED_REPORT_PATH)
seed_report = seed_state.report
lp("Seed (W=", SEED_W_LABEL, "): GT=", seed_report.final_GT, " Delta*=", seed_report.final_Delta_star,
   " w length=", length(seed_report.final_w))

seed = Seed(FAMILY, DIRECTION, "H_w$(SEED_W_LABEL)_delta$(TARGET_DELTA)_point", TARGET_DELTA, 0,
            seed_report.final_GT, seed_report.final_Delta_star, seed_report.final_w, "", SEED_REPORT_PATH)

# Standalone envelope containing ONLY the W-label-annotated seed's own value, as a same-run
# reference floor (not a claim that it's a proven feasible/optimal bound at the NEW W -- Delta* is
# re-verified fresh at the new W by the real KNITRO solve below, since Delta* is itself a
# W-dependent Monte-Carlo-moment quantity, not something carried over from the old W run).
env = MonotoneEnvelope()
register!(env, FAMILY, DIRECTION, TARGET_DELTA, seed_report.final_GT, seed_report.final_w,
          seed_report.final_Delta_star,
          (source_delta = TARGET_DELTA, source_start = 0, outer_vector_path = SEED_REPORT_PATH, outer_vector_sha256 = ""))

ACTIVE_TARGET_DELTA[] = TARGET_DELTA
t0 = time()
report = run_target_cell!(env, FAMILY, DIRECTION, TARGET_DELTA, [seed], EXPLORE_DIRECT_SR1,
                           polish_stage_for(FAMILY), production_run_fn, OUTPUT_DIR;
                           explore_budget_s = EXPLORE_BUDGET_S, polish_budget_s = POLISH_BUDGET_S)
wall = time() - t0

lp("=== W-EXTENSION RESULT ", FAMILY, " / ", DIRECTION, " / delta=", TARGET_DELTA, " (W=", CONTPOLISH_W, ") ===")
lp("  W=", SEED_W_LABEL, " seed GT=", seed_report.final_GT, " -> new W=", CONTPOLISH_W, " GT=", report.final_GT)
lp("  delta_GT=", report.final_GT - seed_report.final_GT, " result_source=", report.result_source,
   " algorithm=", report.algorithm)
lp("  final_Delta_star=", report.final_Delta_star, " knitro_status=", report.knitro_status)
lp("  n_eval=", report.n_eval, " wall_s=", report.wall_s, " total_driver_wall_s=", round(wall, digits = 1))

save_run_state(joinpath(OUTPUT_DIR, "report.jls"), OrchestratorRunState(FAMILY, DIRECTION, TARGET_DELTA,
    EXPLORE_DIRECT_SR1, report.seed_role, string(Dates.now()), report))
lp("Saved report to ", joinpath(OUTPUT_DIR, "report.jls"))

open(joinpath(dirname(OUTPUT_DIR), "..", "w_extension_summary.csv"), "a") do io
    println(io, join([FAMILY, string(DIRECTION), TARGET_DELTA, "W$(SEED_W_LABEL)_to_W$(CONTPOLISH_W)",
                       seed_report.final_GT, report.final_GT, report.final_GT - seed_report.final_GT,
                       report.result_source, report.algorithm, report.knitro_status, report.n_eval,
                       round(report.wall_s, digits = 1)], ","))
end
lp("=== DONE ", FAMILY, " / ", DIRECTION, " / delta=", TARGET_DELTA, " (W=", CONTPOLISH_W, ") ===")
