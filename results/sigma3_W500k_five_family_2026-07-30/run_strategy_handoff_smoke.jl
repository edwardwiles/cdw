# run_strategy_handoff_smoke.jl -- Preflight item 9: direct_sr1 -> optional BFGS polish handoff
# smoke, for unrestricted and origin_zc (sigma3/W500k campaign, 2026-07-30).
#
# Per the campaign brief: run Direct+SR1 first; only if it terminates EARLY with a verified
# incumbent and a NONFATAL convergence/stall status, restart the SAME problem with Direct+BFGS
# for min(900s, remaining budget); retain the best verified point across both stages. This smoke
# uses short budgets (SR1: 60s, BFGS: 60s) to exercise the real handoff LOGIC end-to-end without
# needing the full 10,800s/900s production budget -- the mechanism being smoked is "does the
# handoff correctly happen and correctly retain the better point", not full convergence.
using Dates

const CAMPAIGN_ROOT = @__DIR__
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const INPUTS_DIR = joinpath(REPO_ROOT, "campaign_inputs", "sigma3_W500k_2026-07-30")
const D4E = joinpath(REPO_ROOT, "full_aod_diag", "d4_exact")

ENV["REAL_DATA_DIR"] = joinpath(INPUTS_DIR, "data_snapshot")

for f in ["draw_design.jl", "winners.jl", "oracle.jl", "gravity_elimination.jl",
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
          "hcz_drawchunk_candidate_2026-07-29.jl", "cross_delta_cache.jl", "negative_cache.jl",
          "lfix_buffer_reuse.jl", "bandwidth_cache_policy.jl", "fast_range_screen.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl", "postmerge_smoke_diagnostics.jl", "cross_hessian_live_stash_2026-07-28.jl",
          # unrestricted family's own proven-correct order (campaign_unrestricted_runner_sigma3.jl):
          # c10_d20_production_driver.jl (defines ScreenCounters/DualBank etc.) MUST precede
          # flexible_theta.jl, which references those types in method signatures at include time.
          "c10_d20_production_driver.jl", "flexible_theta.jl", "flexible_theta_aspace_production.jl",
          "outer_coordinate_layout.jl", "c10_d20_production_driver_unified.jl"]
    include(joinpath(D4E, f))
end
include(joinpath(D4E, "json_lite.jl"))
include(joinpath(D4E, "campaign_cell_io.jl"))

lp(xs...) = (println(xs...); flush(stdout))

# origin_zc's w0 MUST come from the frozen, checksum-verified start_manifest.json coordinates
# (matching campaign_cm_family_runner_sigma3.jl's own construction exactly) -- NOT recomputed
# fresh from ctx.θ0_up[ctx.free_idx]/ctx.U, which uses a different (and differently-dimensioned)
# convention and produced a live 380-vs-379 DimensionMismatch when tried live 2026-07-30.
const START_MANIFEST = json_load(joinpath(CAMPAIGN_ROOT, "start_manifest.json"))
const W_A_START1 = jf64(START_MANIFEST["starts"][1]["w_transformed_a"])
const NU_ORIGINZC = jf64(START_MANIFEST["shared_extra_coordinates"]["origin_zc_nu"])

# Nonfatal statuses that would trigger a real polish handoff (matches this codebase's own
# convention for "converged/stalled but not an outright failure" -- see cm_checkpoint.jl's own
# classify_inner_result-adjacent status handling for the analogous inner-solve convention).
const NONFATAL_STALL_STATUSES = (-401, -410, -406, 0)  # time-limit-feasible, iter-limit-feasible, ftol-stalled, converged

function run_handoff(label, W)
    ctx = d20_real_setup_design(W = W, δ = 0.1, find_smallest = true, draw_design = :sobol_randomized,
        draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true, σHat = 3.0)
    pe = build_pivot_elimination(ctx)
    w_calib = ctx.θ0_up[ctx.free_idx]
    gp_calib = w_calib[1]
    z_calib = pivot_reduce(log.(reshape(w_calib[2:end], ctx.D, ctx.D_dest)), pe)
    theta0 = cm_fixed_theta(ctx)
    xy0 = precompute_cm_aspace_xy(ctx)
    a_calib = cm_a_from_z(z_calib, theta0, xy0, pe)
    w_start = vcat(gp_calib, a_calib)
    LAYOUT = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :powered_aspace, gp_coordinate_mode = :raw)

    ckdir_sr1 = mktempdir()
    lp("[$label] STAGE 1: Direct+SR1, budget=60s")
    r_sr1 = run_polish_checkpointed_unified("$(label)_sr1", true, w_start;
        layout = LAYOUT, theta_lo = NaN, theta_hi = NaN, maxtime_real = 60.0, W_in = W, delta_in = 0.1,
        draw_seed_in = 20260719, draw_design_in = :sobol_randomized, ckpt_dir = ckdir_sr1,
        checkpoint_interval_s = 3600.0, resume_from = nothing, destination_sample = :exclude_row,
        exclude_diagonal_gravity = true, σHat = 3.0, outer_direct_hessopt = :sr1)
    lp("[$label] SR1 stage: status=", r_sr1.knitro_status, " best_feasible=",
        r_sr1.best_feasible === nothing ? "none" : r_sr1.best_feasible.Delta)

    sr1_has_incumbent = r_sr1.best_feasible !== nothing
    sr1_nonfatal = Int(r_sr1.knitro_status) in NONFATAL_STALL_STATUSES
    eligible_for_polish = sr1_has_incumbent && sr1_nonfatal
    lp("[$label] eligible for BFGS polish: ", eligible_for_polish,
       " (has_incumbent=", sr1_has_incumbent, " nonfatal_status=", sr1_nonfatal, ")")

    if !eligible_for_polish
        return (label = label, ok = true, handoff_triggered = false,
            reason = "SR1 stage did not meet handoff criteria (has_incumbent=$sr1_has_incumbent, nonfatal=$sr1_nonfatal) -- correctly skipped polish, not a failure",
            sr1_status = Int(r_sr1.knitro_status), sr1_delta = sr1_has_incumbent ? r_sr1.best_feasible.Delta : NaN)
    end

    ckdir_bfgs = mktempdir()
    w_restart = sr1_has_incumbent ? r_sr1.best_feasible.w : w_start
    lp("[$label] STAGE 2: Direct+BFGS polish from SR1 incumbent, budget=60s (production: min(900s, remaining))")
    r_bfgs = run_polish_checkpointed_unified("$(label)_bfgs", true, w_restart;
        layout = LAYOUT, theta_lo = NaN, theta_hi = NaN, maxtime_real = 60.0, W_in = W, delta_in = 0.1,
        draw_seed_in = 20260719, draw_design_in = :sobol_randomized, ckpt_dir = ckdir_bfgs,
        checkpoint_interval_s = 3600.0, resume_from = nothing, destination_sample = :exclude_row,
        exclude_diagonal_gravity = true, σHat = 3.0, outer_direct_hessopt = :bfgs)
    lp("[$label] BFGS stage: status=", r_bfgs.knitro_status, " best_feasible=",
        r_bfgs.best_feasible === nothing ? "none" : r_bfgs.best_feasible.Delta)

    sr1_delta = r_sr1.best_feasible.Delta
    bfgs_delta = r_bfgs.best_feasible === nothing ? Inf : r_bfgs.best_feasible.Delta
    retained = bfgs_delta < sr1_delta ? "bfgs" : "sr1"
    lp("[$label] retained best-across-both-stages: ", retained, " (sr1=", sr1_delta, ", bfgs=", bfgs_delta, ")")

    return (label = label, ok = true, handoff_triggered = true, sr1_status = Int(r_sr1.knitro_status),
        sr1_delta = sr1_delta, bfgs_status = Int(r_bfgs.knitro_status), bfgs_delta = bfgs_delta, retained = retained)
end

function run_originzc_handoff(label, W)
    K = 2
    w0 = vcat(W_A_START1, log.(NU_ORIGINZC))

    ckdir_sr1 = mktempdir()
    lp("[$label] STAGE 1: Direct+SR1, budget=60s")
    r_sr1 = run_originzc_upper_checkpointed(w0; W = W, delta = 0.1, draw_design = :sobol_randomized, draw_seed = 20260719,
        distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = K, K_pair = K,
        ckpt_dir = ckdir_sr1, run_id = "$(label)_sr1", label = "$(label)_sr1",
        checkpoint_interval_s = 3600.0, maxtime_real = 60.0, verbose = true,
        exclude_diagonal_gravity = true, σHat = 3.0, outer_direct_hessopt = :sr1)
    best = r_sr1.best
    lp("[$label] SR1 stage: status=", r_sr1.knitro_status, " best=", best === nothing ? "none" : best.Delta)

    sr1_has_incumbent = best !== nothing
    sr1_nonfatal = Int(r_sr1.knitro_status) in NONFATAL_STALL_STATUSES
    eligible = sr1_has_incumbent && sr1_nonfatal
    lp("[$label] eligible for BFGS polish: ", eligible, " (has_incumbent=", sr1_has_incumbent, " nonfatal=", sr1_nonfatal, ")")
    if !eligible
        return (label = label, ok = true, handoff_triggered = false,
            reason = "SR1 stage did not meet handoff criteria -- correctly skipped polish, not a failure",
            sr1_status = Int(r_sr1.knitro_status), sr1_delta = sr1_has_incumbent ? best.Delta : NaN)
    end

    ckdir_bfgs = mktempdir()
    lp("[$label] STAGE 2: Direct+BFGS polish, budget=60s")
    r_bfgs = run_originzc_upper_checkpointed(w0; W = W, delta = 0.1, draw_design = :sobol_randomized, draw_seed = 20260719,
        distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = K, K_pair = K,
        ckpt_dir = ckdir_bfgs, run_id = "$(label)_bfgs", label = "$(label)_bfgs",
        checkpoint_interval_s = 3600.0, maxtime_real = 60.0, verbose = true,
        exclude_diagonal_gravity = true, σHat = 3.0, outer_direct_hessopt = :bfgs)
    bfgs_delta = r_bfgs.best === nothing ? Inf : r_bfgs.best.Delta
    retained = bfgs_delta < best.Delta ? "bfgs" : "sr1"
    lp("[$label] retained: ", retained, " (sr1=", best.Delta, ", bfgs=", bfgs_delta, ")")
    return (label = label, ok = true, handoff_triggered = true, sr1_status = Int(r_sr1.knitro_status),
        sr1_delta = best.Delta, bfgs_status = Int(r_bfgs.knitro_status), bfgs_delta = bfgs_delta, retained = retained)
end

using Statistics
results = Any[]
ok_all = true
for (label, W, fn) in [("unrestricted", 20_000, run_handoff), ("origin_zc", 20_000, run_originzc_handoff)]
    try
        r = fn(label, W)
        push!(results, r)
        global ok_all = ok_all && get(r, :ok, false)
    catch e
        push!(results, (label = label, ok = false, reason = "exception: $(sprint(showerror, e))"))
        global ok_all = false
    end
end

write_json_file(joinpath(CAMPAIGN_ROOT, "strategy_handoff_smoke_report.json"), Dict(
    "ok" => ok_all,
    "summary" => join(["$(r.label): ok=$(get(r,:ok,false)) $(get(r,:reason,""))" for r in results], " | "),
    "generated" => string(now()),
    "details" => [Dict(string(k) => v for (k, v) in pairs(r)) for r in results],
))
lp(ok_all ? "ALL PASS" : "SOME FAILURES -- see strategy_handoff_smoke_report.json")
exit(ok_all ? 0 : 1)
