# Operator-only structural preflight, all 5 families, WITH the Brazil-Korea gravity exclusion
# (2026-07-31). Adapted from the campaign branch's fixed run_preflights.jl item 5
# (campaign/timeout-fix-and-launch-sigma3-W500k-2026-07-31, commits fbcf579/d9ca91b -- "use real
# per-family build_*_production_context wrappers"), adding gravity_exclude_cells to gate_ctx's
# construction. Cheap (W=2000, real D=20 shape, structural check only).
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
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "cm_lookup_production.jl", "cm_frechet_lookup_kernels.jl",
          "cm_meanzc_lookup_production.jl", "cm_originzc_lookup_production.jl",
          "production_bundle_api.jl", "production_bundle_preflight.jl", "country_resolve.jl"]
    include(joinpath(_D4E, f))
end
using DelimitedFiles

countries = vec(readdlm(joinpath(_D4E, "..", "..", "real_data", "noah_D20", "countries.csv"), ',', String))
bra_idx = resolve_country_index(countries, "Brazil")
kor_idx = resolve_country_index(countries, "Korea")
row_idx_g = findfirst(==("row"), countries)
named_dest = filter(!=(row_idx_g), 1:length(countries))
kor_slot = global_to_dest_slot(kor_idx, named_dest)
gravity_exclude_cells = [(bra_idx, kor_slot)]
println("gravity_exclude_cells = ", gravity_exclude_cells)

gate_ctx = d20_real_setup_design(W = 2000, δ = 0.1, find_smallest = true, draw_design = :sobol_randomized,
    draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = gravity_exclude_cells, σHat = 3.0)
gate_probs = nested_grid_sequence([10, 20, 50])[50]
gate_layout = OriginByPowerLayout(gate_ctx.D, 2, 2)
build_inner_for(family) =
    family === :unrestricted   ? (() -> build_unrestricted_operator_ctx(gate_ctx; moment_representation = :operator)) :
    family === :flexible_cm    ? (() -> build_cm_production_context(gate_ctx, CS; L = 50, contrasts = :orthonormal, probs = gate_probs)) :
    family === :common_frechet ? (() -> build_cm_frechet_production_context(gate_ctx, CS; L = 50, contrasts = :orthonormal, probs = gate_probs, cm_hessian_backend = :structured)) :
    family === :cm_meanzc      ? (() -> build_cm_meanzc_production_context(gate_ctx, CS; L = 50, K_mean = 2, K_pair = 2, contrasts = :orthonormal, probs = gate_probs)) :
    family === :origin_zc      ? (() -> build_originzc_production_context(gate_ctx, CS, gate_layout)) :
    error("build_inner_for: unknown family $family")
gate_dir = joinpath(_D4E, "..", "..", "brazil_korea_preflight_manifests_2026-07-31")
ok = campaign_preflight([:unrestricted, :flexible_cm, :common_frechet, :cm_meanzc, :origin_zc], build_inner_for; manifest_dir = gate_dir)
println(ok ? "ALL FAMILIES OPERATOR-ONLY PREFLIGHT: PASS" : "PREFLIGHT FAILED")

# also check the gravity residual at this ctx is machine-zero (calibration point, with BK excluded)
pe = build_pivot_elimination(gate_ctx)
z_calib = log.(reshape(gate_ctx.θ0_up[gate_ctx.Aod_offset+1:gate_ctx.Aod_offset+gate_ctx.D*gate_ctx.D_dest], gate_ctx.D, gate_ctx.D_dest))
resid = gravity_from_logz(pivot_expand(pivot_reduce(z_calib, pe), pe), gate_ctx)
println("gravity residual at calibration (gate_ctx, BK excluded) = ", resid)
ok2 = abs(resid) < 1e-9
println(ok2 ? "GRAVITY RESIDUAL: PASS" : "GRAVITY RESIDUAL: FAIL")

exit((ok && ok2) ? 0 : 1)
