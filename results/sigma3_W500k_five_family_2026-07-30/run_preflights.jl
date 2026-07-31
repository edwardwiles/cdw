# run_preflights.jl -- sigma3/W500k five-family production campaign (2026-07-30)
#
# Executes the campaign brief's 10 required preflights and writes one JSON report per item plus
# a consolidated PREFLIGHT_SUMMARY.json. Returns nonzero (and refuses READY_TO_LAUNCH, via the
# caller campaign_control.sh) if any item fails. Per the brief's own instruction: "Use 60-second
# outer smokes, extending an individual W500k family up to 180 seconds only when needed to
# complete one valid callback."
#
# Usage: julia --project=. -t 8 results/sigma3_W500k_five_family_2026-07-30/run_preflights.jl
using Dates

const CAMPAIGN_ROOT = @__DIR__
const REPO_ROOT = normpath(joinpath(CAMPAIGN_ROOT, "..", ".."))
const INPUTS_DIR = joinpath(REPO_ROOT, "campaign_inputs", "sigma3_W500k_2026-07-30")
const DRIVERS_DIR = joinpath(INPUTS_DIR, "drivers")
const D4E = joinpath(REPO_ROOT, "full_aod_diag", "d4_exact")

ENV["REAL_DATA_DIR"] = joinpath(INPUTS_DIR, "data_snapshot")

lp(xs...) = (println(xs...); flush(stdout))
include(joinpath(D4E, "json_lite.jl"))  # json_load
include(joinpath(D4E, "campaign_cell_io.jl"))  # write_json_file

results = Dict{String,Any}()
ok_all = true
record!(name, ok, detail) = (results[name] = Dict("ok" => ok, "detail" => detail); global ok_all = ok_all && ok;
    lp(ok ? "[PASS] " : "[FAIL] ", name, ": ", detail))

lp("="^100); lp("REQUIRED PREFLIGHTS -- sigma3/W500k five-family campaign -- ", now()); lp("="^100)

# ---- 1. New-data snapshot and validation ----
try
    manifest = json_load(joinpath(INPUTS_DIR, "data_manifest.json"))
    ok = manifest["validation_all_pass"] == true
    record!("1_data_snapshot_validation", ok, "data_manifest.json validation_all_pass=$(manifest["validation_all_pass"])")
catch e
    record!("1_data_snapshot_validation", false, "exception: $(sprint(showerror, e))")
end

# ---- 2. Sigma-3 calibration ----
try
    manifest = json_load(joinpath(INPUTS_DIR, "calibration_manifest.json"))
    theta_ok = abs(Float64(manifest["theta_star"]) - 4.7292535486122365) < 1e-9
    grav_ok = abs(Float64(manifest["gravity_residual_at_calibration"])) < 1e-9
    sigma_ok = Float64(manifest["sigma"]) == 3.0
    ok = theta_ok && grav_ok && sigma_ok
    record!("2_sigma3_calibration", ok, "theta*=$(manifest["theta_star"]) grav_resid=$(manifest["gravity_residual_at_calibration"]) sigma=$(manifest["sigma"])")
catch e
    record!("2_sigma3_calibration", false, "exception: $(sprint(showerror, e))")
end

# ---- 3. W500k Sobol generation and validation ----
try
    manifest = json_load(joinpath(INPUTS_DIR, "calibration_manifest.json"))
    ok = Int(manifest["W"]) == 500_000 && manifest["draw_design"] == "sobol_randomized" &&
         Int(manifest["draw_seed"]) == 20260719 && !isempty(manifest["draw_checksum_uniform"]) &&
         !isempty(manifest["draw_checksum_transformed"])
    record!("3_sobol_w500k_validation", ok, "W=$(manifest["W"]) design=$(manifest["draw_design"]) seed=$(manifest["draw_seed"])")
catch e
    record!("3_sobol_w500k_validation", false, "exception: $(sprint(showerror, e))")
end

# ---- 4. Direct inner solves for all 15 family-start combinations ----
try
    start_manifest_path = joinpath(CAMPAIGN_ROOT, "start_manifest.json")
    if !isfile(start_manifest_path)
        record!("4_direct_inner_solves_15_combos", false, "start_manifest.json not yet present -- Phase C (three_starts_search.jl) has not completed")
    else
        sm = json_load(start_manifest_path)
        starts = sm["starts"]
        n_starts = length(starts)
        all_ok = true
        detail_parts = String[]
        for s in starts, fam in ("unrestricted", "flexible_cm", "common_frechet", "cm_meanzc", "origin_zc")
            r = get(get(s, "families", Dict()), fam, nothing)
            this_ok = r !== nothing && get(r, "ok", false) == true
            all_ok &= this_ok
            push!(detail_parts, "$(fam)/start$(s["index"])=$(this_ok ? "ok" : "FAIL")")
        end
        ok = all_ok && n_starts == 3
        record!("4_direct_inner_solves_15_combos", ok, "$(n_starts) starts x 5 families: " * join(detail_parts, " "))
    end
catch e
    record!("4_direct_inner_solves_15_combos", false, "exception: $(sprint(showerror, e))")
end

# ---- 5. Exact campaign-entry-point operator-only gate for all five families ----
# Matches full_aod_diag/d4_exact/test_all_family_real_production_entrypoints_operator_bundle.jl's
# own pattern: ONE shared base ctx, unrestricted uses it directly, the other 4 families layer
# their real build_*_production_context wrapper on top -- NOT 5 independent base contexts (an
# earlier version of this script incorrectly built a bare d20_real_setup_design ctx per family,
# which would only actually exercise the unrestricted family's real bundle path).
try
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
              "hcz_drawchunk_candidate_2026-07-29.jl", "cm_lookup_production.jl", "cm_frechet_lookup_kernels.jl",
              "cm_meanzc_lookup_production.jl", "cm_originzc_lookup_production.jl",
              "production_bundle_api.jl", "production_bundle_preflight.jl"]
        include(joinpath(D4E, f))
    end
    gate_ctx = d20_real_setup_design(W = 2000, δ = 0.1, find_smallest = true, draw_design = :sobol_randomized,
        draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true, σHat = 3.0)
    gate_probs = nested_grid_sequence([10, 20, 50])[50]
    gate_layout = OriginByPowerLayout(gate_ctx.D, 2, 2)
    build_inner_for(family) =
        family === :unrestricted   ? (() -> build_unrestricted_operator_ctx(gate_ctx; moment_representation = :operator)) :
        family === :flexible_cm    ? (() -> build_cm_production_context(gate_ctx, CS; L = 50, contrasts = :orthonormal, probs = gate_probs)) :
        family === :common_frechet ? (() -> build_cm_frechet_production_context(gate_ctx, CS; L = 50, contrasts = :orthonormal, probs = gate_probs, cm_hessian_backend = :structured)) :
        family === :cm_meanzc      ? (() -> build_cm_meanzc_production_context(gate_ctx, CS; L = 50, K_mean = 2, K_pair = 2, contrasts = :orthonormal, probs = gate_probs)) :
        family === :origin_zc      ? (() -> build_originzc_production_context(gate_ctx, CS, gate_layout)) :
        error("build_inner_for: unknown family $family")
    gate_dir = joinpath(CAMPAIGN_ROOT, "preflight_manifests")
    ok = campaign_preflight([:unrestricted, :flexible_cm, :common_frechet, :cm_meanzc, :origin_zc], build_inner_for; manifest_dir = gate_dir)
    record!("5_operator_only_gate_all_families", ok, "campaign_preflight() over all 5 families (family-specific real build_*_production_context wrappers), manifests in $gate_dir")
catch e
    record!("5_operator_only_gate_all_families", false, "exception: $(sprint(showerror, e))")
end

# ---- 6-10: real 5-family parallel/strategy/resource smokes ----
# These require actually launching 5 concurrent Julia processes (matching --launch's own
# process-group design) at short (60-180s) budgets, and are executed via the SAME
# run_family_chain_sigma3.sh / campaign_control.sh machinery --launch itself uses, not a
# separate ad hoc path -- so a preflight pass is evidence about the real launch command, not a
# proxy for it. Implemented as a shell-level smoke runner: see run_parallel_smoke.sh in this
# directory (upper/lower calibration smokes, item 6-7), run_perturbation_smoke.sh (item 8),
# run_strategy_handoff_smoke.sh (item 9), and run_resource_smoke.sh (item 10). Each writes its
# own <item>_smoke_report.json here; this script reads them back rather than re-running the
# smoke itself (they are real multi-minute external processes, not something to launch inline
# from inside this same Julia process).
for (item, script, desc) in [
        ("6_upper_smoke_calibration_delta0.1", "upper_smoke_report.json", "five-family parallel upper smoke at calibration, delta=0.1"),
        ("7_lower_smoke_calibration_delta0.1", "lower_smoke_report.json", "five-family parallel lower smoke at calibration, delta=0.1"),
        ("8_perturbation_callback_starts23", "perturbation_smoke_report.json", "one direct outer callback for starts 2 and 3, every family"),
        ("9_strategy_handoff_smoke", "strategy_handoff_smoke_report.json", "direct_sr1 -> optional BFGS polish handoff, unrestricted + origin_zc"),
        ("10_five_family_resource_smoke", "resource_smoke_report.json", "five-family combined parallel resource smoke (100 threads)"),
    ]
    report_path = joinpath(CAMPAIGN_ROOT, script)
    if isfile(report_path)
        try
            r = json_load(report_path)
            record!(item, get(r, "ok", false) == true, desc * " -- " * get(r, "summary", "(no summary)"))
        catch e
            record!(item, false, "$desc -- report unreadable: $(sprint(showerror, e))")
        end
    else
        record!(item, false, "$desc -- NOT YET RUN (report $script absent; see the corresponding run_*_smoke.sh)")
    end
end

lp("="^100)
lp(ok_all ? "ALL PREFLIGHTS PASS" : "PREFLIGHT FAILURE -- see items above")
lp("="^100)

write_json_file(joinpath(CAMPAIGN_ROOT, "PREFLIGHT_SUMMARY.json"), Dict(
    "generated_at" => string(now()), "ok_all" => ok_all, "items" => results))

exit(ok_all ? 0 : 1)
