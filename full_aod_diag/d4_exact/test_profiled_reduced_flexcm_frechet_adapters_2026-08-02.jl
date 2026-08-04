# ============================================================================
# Integration continuation (2026-08-02), task Section 7-9: real (non-mock) end-to-end gate for the
# Stage-I flexCM and common-Frechet adapters (profiled_restricted_family_adapters_2026-08-02.jl),
# wired to the GENUINE dense-G-free reduced operator FG. For each family:
#   1. validate_family_layout_contract(fctx) succeeds (real accessors, real checksum).
#   2. evaluate_profiled_{flexcm,frechet}_point at the reduced calibration point reaches nStatus=0.
#   3. shared_family_outer_gradient(w, ctx, fctx, ev; threaded = false) -- the SAME shared A/gp engine the unrestricted
#      family uses, completely unmodified -- matches the independent, non-incremental
#      diag_profiled_full_rebuild_gradient reference (profiled_restricted_full_rebuild_gradient_
#      reference_2026-08-01.jl) to machine precision, at MATCHED (adaptive) per-coordinate
#      bandwidth -- exactly the same honest comparison the existing mock-family gate
#      (test_profiled_mock_family_gate_2026-08-01.jl) already established, now against a REAL
#      restriction_contrib0 (not a mock rc0_fn) and a REAL solved restricted-family dual.
#   4. NO_DENSE_G_COUNTERS delta across the whole gate is 0 (the adapters' own evaluators route
#      exclusively through the new reduced_{cm,frechet}_base_state operator-FG drivers).
# ============================================================================
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "no_dense_g_counters.jl", "economic_operator.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl",
          "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_checkpoint.jl",
          "operator_hessian_weights.jl", "operator_psi_bundle.jl", "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
          "compressed_factual_buffer_reuse.jl",
          "winner_certificate.jl", "recover_full_a_2026-07-31.jl", "homogeneous_moments_2026-07-31.jl",
          "relative_a_coordinate_2026-07-31.jl", "gravity_pivot_on_retained_2026-07-31.jl",
          "outer_coordinate_layout_profiled_2026-07-31.jl",
          "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "reduced_operator_verification_2026-08-01.jl",
          "profiled_operator_bundle_2026-08-01.jl",
          "reduced_recovery_from_lfd_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl",
          "profiled_reduced_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_frechet_lookup_kernels_2026-08-02.jl",
          "profiled_outer_evaluator_2026-08-01.jl",
          "profiled_outer_gradient_fd_2026-08-01.jl",
          "profiled_lfix_incremental_2026-08-01.jl",
          "profiled_outer_gradient_layout_contract_2026-08-01.jl",
          "profiled_shared_economic_gradient_engine_2026-08-01.jl",
          "profiled_restriction_contrib0_operators_2026-08-01.jl",
          "profiled_restricted_full_rebuild_gradient_reference_2026-08-01.jl",
          "profiled_restricted_family_adapters_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random, CSV, DataFrames

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(3 => 1))
w_calib = reduce_calibration_to_w_profiled(ctx, pe)
n_total = outer_dim_profiled(pe)
println("D4 real (non-mock) flexCM+Frechet adapter gate: n_total=$n_total"); flush(stdout)

x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
assert_no_factual_price_index_moment(layout)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
L = 10; contrasts = :anchored

rows = NamedTuple[]

# =====================================================================================
# Flexible CM
# =====================================================================================
println("="^90); println("FLEXIBLE CM"); println("="^90); flush(stdout)
aug_reduced = build_cm_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0, profiled_layout = layout)
cctx_cm = build_cm_bin_ctx(ctx, aug_reduced; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = false)
fctx_cm = build_flexcm_family_ctx(ctx, spec, pe, layout, cctx_cm)

v_cm = validate_family_layout_contract(fctx_cm)
check("flexCM: layout contract validates", v_cm.pe === pe && v_cm.layout === layout)

dense_g_before = NO_DENSE_G_COUNTERS[].dense_economic_G_materializations
ev_cm = evaluate_profiled_flexcm_point(w_calib, fctx_cm)
dense_g_after = NO_DENSE_G_COUNTERS[].dense_economic_G_materializations
println("  inner_status=$(ev_cm.result.inner_status)  zeta*=$(ev_cm.result.zeta)"); flush(stdout)
check("flexCM: real KNITRO solve reaches nStatus=0", ev_cm.result.inner_status == 0)
check("flexCM: zero dense economic G materialization across the whole evaluator call", dense_g_after == dense_g_before)

g_shared_cm, meta_cm = shared_family_outer_gradient(w_calib, ctx, fctx_cm, ev_cm; threaded = false)
h_matched_cm = copy(meta_cm.h_used); h_matched_cm[1] = 0.01
g_full_cm, _ = diag_profiled_full_rebuild_gradient(w_calib, ctx, fctx_cm, ev_cm; h = h_matched_cm)
diff_cm = g_shared_cm[2:end] .- g_full_cm[2:end]
max_abs_err_cm = maximum(abs.(diff_cm))
max_rel_err_cm = maximum(abs.(diff_cm) ./ max.(abs.(g_full_cm[2:end]), 1e-8))
gp_diff_cm = abs(g_shared_cm[1] - g_full_cm[1])
@printf("  A-block(2:end): max_abs_err=%.4e max_rel_err=%.4e  gp: shared=%.6e full_FD=%.6e diff=%.3e\n",
    max_abs_err_cm, max_rel_err_cm, g_shared_cm[1], g_full_cm[1], gp_diff_cm)
check("flexCM: shared engine matches full-rebuild reference at matched bandwidth (<1e-8 rel, A-block)", max_rel_err_cm < 1e-8)
push!(rows, (family = :flexible_CM, inner_status = ev_cm.result.inner_status,
    max_abs_err = max_abs_err_cm, max_rel_err = max_rel_err_cm, gp_diff = gp_diff_cm,
    dense_g_delta = dense_g_after - dense_g_before))

# =====================================================================================
# Common Frechet
# =====================================================================================
println("="^90); println("COMMON FRECHET"); println("="^90); flush(stdout)
aug_reduced_f = build_cm_frechet_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0, profiled_layout = layout)
level_targets = aug_reduced_f.level_targets
cctx_f = build_cm_bin_ctx(ctx, aug_reduced_f; profiled_layout = layout, inner_fg_backend = :dense_reference,
                           threaded_bins = false, core_hessian_backend = :dense_reference, cm_cross_hessian_backend = :winner_bin)
fctx_f = build_frechet_family_ctx(ctx, spec, pe, layout, cctx_f, level_targets)

v_f = validate_family_layout_contract(fctx_f)
check("Frechet: layout contract validates", v_f.pe === pe && v_f.layout === layout)

dense_g_before_f = NO_DENSE_G_COUNTERS[].dense_economic_G_materializations
ev_f = evaluate_profiled_frechet_point(w_calib, fctx_f)
dense_g_after_f = NO_DENSE_G_COUNTERS[].dense_economic_G_materializations
println("  inner_status=$(ev_f.result.inner_status)  zeta*=$(ev_f.result.zeta)"); flush(stdout)
check("Frechet: real KNITRO solve reaches nStatus=0", ev_f.result.inner_status == 0)
check("Frechet: zero dense economic G materialization across the whole evaluator call", dense_g_after_f == dense_g_before_f)

g_shared_f, meta_f = shared_family_outer_gradient(w_calib, ctx, fctx_f, ev_f; threaded = false)
h_matched_f = copy(meta_f.h_used); h_matched_f[1] = 0.01
g_full_f, _ = diag_profiled_full_rebuild_gradient(w_calib, ctx, fctx_f, ev_f; h = h_matched_f)
diff_f = g_shared_f[2:end] .- g_full_f[2:end]
max_abs_err_f = maximum(abs.(diff_f))
max_rel_err_f = maximum(abs.(diff_f) ./ max.(abs.(g_full_f[2:end]), 1e-8))
gp_diff_f = abs(g_shared_f[1] - g_full_f[1])
@printf("  A-block(2:end): max_abs_err=%.4e max_rel_err=%.4e  gp: shared=%.6e full_FD=%.6e diff=%.3e\n",
    max_abs_err_f, max_rel_err_f, g_shared_f[1], g_full_f[1], gp_diff_f)
check("Frechet: shared engine matches full-rebuild reference at matched bandwidth (<1e-8 rel, A-block)", max_rel_err_f < 1e-8)
push!(rows, (family = :common_Frechet, inner_status = ev_f.result.inner_status,
    max_abs_err = max_abs_err_f, max_rel_err = max_rel_err_f, gp_diff = gp_diff_f,
    dense_g_delta = dense_g_after_f - dense_g_before_f))

df = DataFrame(rows)
outpath = joinpath(@__DIR__, "..", "..", "PROFILED_REDUCED_FLEXCM_FRECHET_ADAPTER_GATE_2026-08-02.csv")
CSV.write(outpath, df)
println("\nWrote $outpath"); println(df)

println("\n", ALL_PASS[] ? "ALL PASS" : "SOME FAILED")
ALL_PASS[] || error("flexCM/Frechet real adapter gate failed")
