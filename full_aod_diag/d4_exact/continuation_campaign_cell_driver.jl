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
# post-verifier-fix W=100k rerun + K=3 campaign task (2026-08-04): redirected from the original
# campaign_output/ to a namespaced POST_VERIFY_FIX/ directory so the fresh, fixed-gate rerun never
# overwrites the archived PRE_FIX_VERIFIER results (task's own explicit "do not overwrite" rule) --
# ORIGINAL_ENVELOPE_CSV/ORIGINAL_SEED_MANIFEST_CSV above still correctly point at the old,
# read-only K1 baseline (reused as S1 seed input, never written to).
const CAMPAIGN_RESULTS_ROOT = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/POST_VERIFY_FIX/campaign_output"

include(joinpath(D4E, "full_chain_include.jl"))
isdefined(Main, :MANIFEST_K3_HASH) || include(joinpath(D4E, "w100k_manifest.jl"))

lp(xs...) = (println(xs...); flush(stdout))

# `extra_seed:...` tokens (see below) are filtered out BEFORE positional parsing so they can be
# placed anywhere on the command line without shifting FAMILY/DIRECTION/.../CROSS_SEED_DELTA --
# confirmed live 2026-08-04 that leaving them in ARGS for the positional reads below silently
# misassigns an extra_seed token to CROSS_SEED_FAMILY (7th positional slot), which then throws
# ("cross-family seeding is only meaningful feeding INTO unrestricted") for any other family.
const POSITIONAL_ARGS = [a for a in ARGS if !startswith(a, "extra_seed:")]

const FAMILY = POSITIONAL_ARGS[1]
const DIRECTION = Symbol(POSITIONAL_ARGS[2])

# Stamp every quarantine record this process writes (quarantine.jl, task §2.5) with the manifest
# hash that actually governs this cell -- MANIFEST_K3_HASH for the two ZC families (K_mean=K_pair=3
# this campaign), MANIFEST_K1_HASH otherwise (the K fields are inert/unused for the three non-ZC
# families, but every other scientific field is identical between the two manifests, so K1's hash
# is the correct shared identity for them, not a mismatch).
ACTIVE_CAMPAIGN_MANIFEST_HASH[] = FAMILY in ("origin_zc", "cm_meanzc") ? MANIFEST_K3_HASH : MANIFEST_K1_HASH
const TARGET_DELTA = parse(Float64, POSITIONAL_ARGS[3])
const OUTPUT_DIR = POSITIONAL_ARGS[4]
const EXPLORE_BUDGET_S = length(POSITIONAL_ARGS) >= 5 ? parse(Float64, POSITIONAL_ARGS[5]) : 1500.0
const POLISH_BUDGET_S = length(POSITIONAL_ARGS) >= 6 ? parse(Float64, POSITIONAL_ARGS[6]) : 1500.0
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
const CROSS_SEED_FAMILY = length(POSITIONAL_ARGS) >= 7 ? POSITIONAL_ARGS[7] : nothing
const CROSS_SEED_DELTA = length(POSITIONAL_ARGS) >= 8 ? parse(Float64, POSITIONAL_ARGS[8]) : nothing

# Generic extra-seed mechanism (TARGETED_K3_EXTENSIONS_2026-08-04, Sections 5/6): any number of
# `extra_seed:<report.jls path>[:role_label]` tokens anywhere in ARGS (order-independent, does not
# disturb the fixed positional args above). Unlike CROSS_SEED_FAMILY (cross-FAMILY, same delta,
# sliced to [1:380]), this loads the SAME family's own w[full length] from an arbitrary report --
# e.g. a same-family boundary point at a DIFFERENT delta (section 6's S3: a backtracked delta=2.0
# point offered as a candidate seed for the delta=1.0 cell), or an immutable frozen-registry path
# (section 5.1's reverified primary seed). Caller is responsible for economic sanity of what it
# points at; this mechanism only loads+wraps, it does not interpret the source cell's own delta.
const EXTRA_SEED_TOKENS = [a for a in ARGS if startswith(a, "extra_seed:")]

function load_extra_seed(token::String, family::String, direction::Symbol)
    parts = split(token, ':'; limit = 3)
    path = parts[2]
    role = length(parts) >= 3 ? parts[3] : "extra_seed_$(basename(dirname(path)))"
    isfile(path) || error("load_extra_seed: file not found: $path")
    state = load_run_state(path)
    r = state.report
    r === nothing && error("load_extra_seed: $path has no report (nothing) -- not usable as a seed")
    return Seed(family, direction, role, r.target_delta, 0, r.final_GT, r.final_Delta_star, r.final_w, "", path)
end

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
for prior_delta in (0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, TARGET_DELTA)
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

for token in EXTRA_SEED_TOKENS
    s = load_extra_seed(token, FAMILY, DIRECTION)
    push!(seeds, s)
    lp("Extra seed loaded: role=", s.role, " source_delta=", s.source_delta, " GT=", s.GT,
       " Delta*=", s.Delta_star, " path=", s.source_path)
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
