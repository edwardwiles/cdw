# paper_upper_v1: empirical (A_scale, gp_scale) scan, REQUIRED before the real 10-seed generation
# run (multistart_seed_generator.jl's own header: "A_scale/gp_scale are NOT tuned defaults -- an
# aggressive scale can make every random draw genuinely infeasible, confirmed live 0/10 vs 10/10
# acceptance between two scale choices at the same W").
#
# Scans the manifest's [seeds.scale_scan] grid at the REAL protocol W/delta_max/family_specs,
# using evaluate_attempt(...) directly (a handful of trial attempts per cell, not a full M-seed
# generation run), and reports acceptance-rate-proxy stats per cell so a real (A_scale, gp_scale)
# can be chosen and frozen into [seeds] before generate_seeds.jl runs for real.
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=10 julia --project=. \
#       paper_upper_v1_orchestrator/scan_seed_scales.jl protocols/paper_upper_v1.toml <campaign_root>

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
const SCAN_CFG = SEEDS_CFG["scale_scan"]

grav = default_gravity_exclude_cells_brazil_korea()
lp("Building ctx at real protocol scale: W=", SCI["W"], " draw_seed=", SCI["draw_seed"])
ctx_raw = d20_real_setup_design(W = SCI["W"], δ = SEEDS_CFG["seed_delta_max"], find_smallest = SCI["find_smallest"],
    draw_design = Symbol(SCI["draw_design"]), draw_seed = SCI["draw_seed"], destination_sample = Symbol(SCI["destination_sample"]),
    exclude_diagonal_gravity = SCI["exclude_diagonal_gravity"], gravity_exclude_cells = grav,
    σHat = SCI["sigma"], inner_lower_limit = SCI["inner_lower_limit"])
ctx = attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)
lp("ctx built.")

family_specs = paper_five_family_seed_specs(ctx)
manifest_digest = compute_manifest_digest(ctx, family_specs, SCI["W"])
geo = build_aspace_geometry(ctx)
w_cal = cm_w0_from_calibration(ctx, geo.pe, :powered_aspace)
gp_cal, gp_target = gp_calibration_and_target(ctx, Symbol(SCI["direction"]))

delta_max = SEEDS_CFG["seed_delta_max"] - SEEDS_CFG["seed_delta_qualification_tolerance"]
rng_seed = UInt64(SEEDS_CFG["master_rng_seed"])
n_trials = SCAN_CFG["trial_attempts_per_cell"]

report = NamedTuple[]
attempt_id = 0
for A_scale in Float64.(SCAN_CFG["candidate_A_scales"]), gp_scale in Float64.(SCAN_CFG["candidate_gp_scales"])
    n_accept = 0
    first_reject_family = Dict{Symbol,Int}()
    for _ in 1:n_trials
        global attempt_id += 1
        res = evaluate_attempt(ctx, geo, w_cal, gp_cal, gp_target, family_specs, attempt_id, 1,
            rng_seed, manifest_digest, A_scale, gp_scale, delta_max)
        if res.rejection_reason === nothing
            n_accept += 1
        else
            fam = Symbol(split(String(res.rejection_reason), "_")[end])
            first_reject_family[res.rejection_reason] = get(first_reject_family, res.rejection_reason, 0) + 1
        end
    end
    lp("A_scale=", A_scale, " gp_scale=", gp_scale, " accept=", n_accept, "/", n_trials,
       " rejections=", first_reject_family)
    push!(report, (A_scale = A_scale, gp_scale = gp_scale, accept = n_accept, n_trials = n_trials,
                    rejections = first_reject_family))
end

mkpath(joinpath(CAMPAIGN_ROOT, "seeds"))
open(joinpath(CAMPAIGN_ROOT, "seeds", "scale_scan_report.txt"), "w") do io
    for r in report
        println(io, r)
    end
end
lp("="^100)
lp("SCAN DONE. Report written to ", joinpath(CAMPAIGN_ROOT, "seeds", "scale_scan_report.txt"))
lp("Pick the (A_scale, gp_scale) with acceptance in a healthy middle range (neither ~0/", n_trials,
   " nor trivially ", n_trials, "/", n_trials, " at the SMALLEST radius, which would just mean 'too timid to learn anything') and freeze it into protocols/paper_upper_v1.toml's [seeds] block before running generate_seeds.jl for real.")
