# continuation_campaign_cell_driver.jl -- reusable per-target-cell driver for the Section 9-15
# staged continuation/polish campaign. One process = one (family, direction, target_delta) cell.
#
# Usage:
#   julia --project=. -t <threads> continuation_campaign_cell_driver.jl <family> <direction> <target_delta> <output_dir> [explore_budget_s=1500] [polish_budget_s=1500]
#
# direction in {upper, lower}. Loads the ORIGINAL MONOTONE_INCUMBENT_ENVELOPE_2026-08-03.csv as the
# baseline incumbent set, plus any already-completed campaign cell reports for the SAME
# family/direction at a smaller delta (so the lower delta=0.5->1->2 continuation chain correctly
# inherits an improved delta=1 result, not just the original delta=0.5 baseline) -- see
# `campaign_report_dir_for` below for the fixed naming convention this relies on.

const D4E = "/bbkinghome/edav/cdw_worktrees/fullA-continuation-polish-2026-08-03/full_aod_diag/d4_exact"
const CAMPAIGN_OUT = "/bbkinghome/edav/gravity_robustness/worktrees/campaign-fullA-W100k-10x10-2026-08-02/results/campaign_output_2026-08-02"
const ORIGINAL_ENVELOPE_CSV = "/bbkinghome/edav/gravity_robustness/worktrees/campaign-fullA-W100k-10x10-2026-08-02/MONOTONE_INCUMBENT_ENVELOPE_2026-08-03.csv"
const ORIGINAL_SEED_MANIFEST_CSV = "/bbkinghome/edav/gravity_robustness/worktrees/campaign-fullA-W100k-10x10-2026-08-02/CONTINUATION_SEED_MANIFEST_2026-08-03.csv"
const CAMPAIGN_RESULTS_ROOT = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/campaign_output"

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
const OUTPUT_DIR = ARGS[4]
const EXPLORE_BUDGET_S = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1500.0
const POLISH_BUDGET_S = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 1500.0
# Cross-family relaxation seed (optional): unrestricted is a strict relaxation of every restricted
# family (same (gp,A) -> Delta* map, just fewer/no extra moment restrictions), so a restricted
# family's own best point at the SAME delta is, by construction, already Delta*-feasible for
# unrestricted -- confirmed by direct code read that every restricted family's w0 is
# [w_a; optional_extra_tail] with w_a the IDENTICAL 380-length [gp;A_nonpivot] vector unrestricted
# uses directly (campaign_cm_family_runner.jl's own w0 construction). Passing
# <cross_seed_family> <cross_seed_delta> loads that family's report at that delta and offers its
# w[1:380] as an extra seed candidate -- used to fix cross-family monotonicity violations
# (unrestricted must weakly dominate every restricted family at fixed delta/direction; found and
# confirmed live 2026-08-04: origin_zc's own upper bound exceeded unrestricted's own at
# delta=0.01/0.1/0.5/1.0, a logical impossibility this seed resolves by construction).
const CROSS_SEED_FAMILY = length(ARGS) >= 7 ? ARGS[7] : nothing
const CROSS_SEED_DELTA = length(ARGS) >= 8 ? parse(Float64, ARGS[8]) : nothing

isdir(OUTPUT_DIR) || mkpath(OUTPUT_DIR)
isdir(CAMPAIGN_RESULTS_ROOT) || mkpath(CAMPAIGN_RESULTS_ROOT)

# Per-family polish-arm choice, per PILOT_TOURNAMENT_VERDICT_2026-08-03.md.
polish_stage_for(family::String) = family == "unrestricted" ? POLISH_DIRECT_BFGS : POLISH_SQP

campaign_report_path_for(family::String, direction::Symbol, delta::Real) =
    joinpath(CAMPAIGN_RESULTS_ROOT, family, string(direction), "delta_$(delta)", "report.jls")

lp("=== CAMPAIGN CELL ", FAMILY, " / ", DIRECTION, " / delta=", TARGET_DELTA, " ===")
lp("explore_budget_s=", EXPLORE_BUDGET_S, " polish_budget_s=", POLISH_BUDGET_S, " output_dir=", OUTPUT_DIR)

# 1. Baseline envelope from the ORIGINAL (pre-continuation) campaign.
env = load_envelope_csv(ORIGINAL_ENVELOPE_CSV)
lp("Loaded baseline envelope: ", length(env.rows), " rows.")

# 2. Layer in any already-completed CAMPAIGN cell reports for this exact family/direction at a
#    smaller delta (supports the lower delta=0.5->1->2 continuation chain: delta=2's env must see
#    delta=1's NEW result if it improved on delta=0.5's original one) -- AND at the SAME target
#    delta, from a prior refinement round on this exact cell (a "round 2" push: re-running the same
#    (family,direction,delta) seeds from its own best-so-far result to squeeze further, since many
#    cells hit their time budget while still improving). The never-regress rule makes repeated
#    rounds on the same cell always safe -- a round that finds nothing new just re-exports the
#    current incumbent.
for prior_delta in (0.01, 0.1, 0.5, 1.0, 1.5, TARGET_DELTA)
    prior_delta > TARGET_DELTA && continue
    prior_path = campaign_report_path_for(FAMILY, DIRECTION, prior_delta)
    isfile(prior_path) || continue
    prior_report = load_run_state(prior_path).report
    register!(env, FAMILY, DIRECTION, prior_delta, prior_report.final_GT, prior_report.final_w,
              prior_report.final_Delta_star,
              (source_delta = prior_delta, source_start = 0, outer_vector_path = prior_path,
               outer_vector_sha256 = ""))
    lp("Layered in campaign-produced result at delta=", prior_delta, ": GT=", prior_report.final_GT,
       " (source=", prior_report.result_source, ")")
end

inherited_before = envelope_at(env, FAMILY, DIRECTION, TARGET_DELTA)
lp("Inherited incumbent for this target: ", inherited_before === nothing ? "none" : inherited_before.GT)

# 3. Seed candidates: every "available" row in the original seed manifest for this family/direction
#    (up to 4 roles: A/B/C/D per task Section 6), deduplicated.
all_seed_rows = csv_rows_as_namedtuples(ORIGINAL_SEED_MANIFEST_CSV)
seeds = Seed[]
for r in all_seed_rows
    String(r.family) == FAMILY && Symbol(r.direction) == DIRECTION || continue
    lowercase(String(r.available)) == "available" || continue
    w = load_checkpoint_w(String(r.outer_vector_path), FAMILY)
    push!(seeds, Seed(FAMILY, DIRECTION, String(r.seed_role), parse(Float64, r.delta), parse(Int, r.start),
                       parse(Float64, r.GT), parse(Float64, r.Delta_star), w, String(r.outer_vector_sha256),
                       String(r.outer_vector_path)))
end
lp("Loaded ", length(seeds), " raw seed candidates from the original manifest.")

# Also offer the just-layered-in campaign incumbent (if any, and if better than every manifest seed)
# as an explicit additional seed candidate -- dedup_seeds will drop it if it's a near-duplicate of
# one already in the list.
if inherited_before !== nothing
    push!(seeds, Seed(FAMILY, DIRECTION, "F_current_envelope_incumbent", inherited_before.source_delta,
                       0, inherited_before.GT, inherited_before.Delta_star, inherited_before.w, "", ""))
end

if CROSS_SEED_FAMILY !== nothing
    FAMILY == "unrestricted" ||
        error("cross-family seeding is only meaningful feeding INTO unrestricted (the strict relaxation of every other family) -- got FAMILY=$FAMILY")
    cross_path = campaign_report_path_for(CROSS_SEED_FAMILY, DIRECTION, CROSS_SEED_DELTA)
    isfile(cross_path) || error("cross-family seed report not found: $cross_path")
    cross_report = load_run_state(cross_path).report
    length(cross_report.final_w) >= 380 ||
        error("cross-family seed w too short ($(length(cross_report.final_w))) -- expected >=380 ([gp;A_nonpivot] block)")
    w_cross = cross_report.final_w[1:380]
    push!(seeds, Seed(FAMILY, DIRECTION, "G_cross_family_$(CROSS_SEED_FAMILY)_delta_$(CROSS_SEED_DELTA)",
                       CROSS_SEED_DELTA, 0, cross_report.final_GT, cross_report.final_Delta_star, w_cross, "", cross_path))
    lp("Cross-family seed loaded: ", CROSS_SEED_FAMILY, "@delta=", CROSS_SEED_DELTA, " GT=", cross_report.final_GT,
       " Delta*=", cross_report.final_Delta_star, " (sliced to first 380 components)")
end

isempty(seeds) && inherited_before === nothing &&
    error("No seeds available and no inherited incumbent for $FAMILY/$DIRECTION/delta=$TARGET_DELTA -- cannot proceed.")

ACTIVE_TARGET_DELTA[] = TARGET_DELTA
t0 = time()
report = run_target_cell!(env, FAMILY, DIRECTION, TARGET_DELTA, seeds, EXPLORE_DIRECT_SR1,
                           polish_stage_for(FAMILY), production_run_fn, OUTPUT_DIR;
                           explore_budget_s = EXPLORE_BUDGET_S, polish_budget_s = POLISH_BUDGET_S)
wall = time() - t0

lp("=== CAMPAIGN CELL RESULT ", FAMILY, " / ", DIRECTION, " / delta=", TARGET_DELTA, " ===")
lp("  result_source=", report.result_source, " algorithm=", report.algorithm, " seed_role=", report.seed_role)
lp("  inherited_GT=", report.inherited_GT, " new_GT=", report.new_GT, " final_GT=", report.final_GT)
lp("  final_Delta_star=", report.final_Delta_star, " knitro_status=", report.knitro_status)
lp("  n_eval=", report.n_eval, " wall_s=", report.wall_s, " total_driver_wall_s=", round(wall, digits = 1))

# Persist: canonical location (for the next delta in this same family/direction's chain to find)
# AND the caller-specified OUTPUT_DIR (for this invocation's own record-keeping).
canonical_path = campaign_report_path_for(FAMILY, DIRECTION, TARGET_DELTA)
isdir(dirname(canonical_path)) || mkpath(dirname(canonical_path))
state = OrchestratorRunState(FAMILY, DIRECTION, TARGET_DELTA, EXPLORE_DIRECT_SR1, report.seed_role,
                              string(Dates.now()), report)
save_run_state(canonical_path, state)
save_run_state(joinpath(OUTPUT_DIR, "report.jls"), state)
lp("Saved report to ", canonical_path, " and ", joinpath(OUTPUT_DIR, "report.jls"))

open(joinpath(CAMPAIGN_RESULTS_ROOT, "campaign_summary.csv"), "a") do io
    println(io, join([FAMILY, string(DIRECTION), TARGET_DELTA, report.result_source, report.algorithm,
                       report.seed_role, something(report.inherited_GT, "NaN"), something(report.new_GT, "NaN"),
                       report.final_GT, report.final_Delta_star, report.knitro_status, report.n_eval,
                       round(report.wall_s, digits = 1)], ","))
end
lp("=== DONE ", FAMILY, " / ", DIRECTION, " / delta=", TARGET_DELTA, " ===")
