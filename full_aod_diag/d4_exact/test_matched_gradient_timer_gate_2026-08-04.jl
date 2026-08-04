# task §5 (profiled-outer-ab-readiness-2026-08-04): matched FULL/REDUCED outer-gradient timer
# instrumentation gate. Real D20, real calibration point, both formulations, warmup-then-measure
# protocol (task's own "exclude compilation through explicit warmup" instruction).
D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "no_dense_g_counters.jl", "economic_operator.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl",
          "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl",
          "blas_thread_policy.jl", "knitro_outer_algorithm.jl", "production_backend_manifest.jl",
          "incumbent_logic.jl", "cm_hessian_subblock_profiling.jl", "production_bundle_api.jl", "country_resolve.jl",
          "cm_exact_cache_production.jl", "cm_checkpoint.jl", "draw_design.jl",
          "operator_hessian_weights.jl", "operator_psi_bundle.jl", "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl",
          "gravity_pivot_on_retained_2026-07-31.jl", "outer_coordinate_layout_profiled_2026-07-31.jl",
          "reduced_operator_verification_2026-08-01.jl",
          "recover_full_a_2026-07-31.jl",
          "profiled_reduced_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_frechet_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_originzc_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl",
          "profiled_outer_gradient_layout_contract_2026-08-01.jl",
          "profiled_stable_layout_digest_2026-08-01.jl",
          "profiled_operator_bundle_2026-08-01.jl",
          "profiled_outer_evaluator_2026-08-01.jl",
          "profiled_lfix_incremental_2026-08-01.jl",
          "profiled_shared_economic_gradient_engine_2026-08-01.jl",
          "profiled_family_adapters_2026-08-01.jl",
          "profiled_restriction_contrib0_operators_2026-08-01.jl",
          "profiled_restricted_family_adapters_2026-08-02.jl",
          "profiled_matched_gradient_instrumentation_2026-08-04.jl"]
    include(joinpath(D4X, f))
end

const W_VAL = parse(Int, get(ENV, "GATE_W", "20000"))
lp(xs...) = (println(xs...); flush(stdout))
lp("Threads.nthreads() = ", Threads.nthreads())

ctx = d20_real_setup_design(; W = W_VAL, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = [(3, 14)], σHat = 3.0)
D = ctx.D
korea_idx, brazil_idx = 14, 3

# ---- REDUCED side (flexible_CM) ----
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
θ0 = ctx.θ0_up
z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*ctx.D_dest], D, ctx.D_dest))
gauge = build_anchor_gauge(z_calib, spec)
pe_r = build_pivot_elimination_on_retained(ctx, spec, gauge)
gp0 = θ0[3+D]
w0 = reduce_to_w_profiled(gp0, z_calib, pe_r)
θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
aug = build_cm_augmented_obj_archB(ctx, CS; L = 50, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
cctx = build_cm_bin_ctx(ctx, aug; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = true)
fctx = build_flexcm_family_ctx(ctx, spec, pe_r, layout, cctx)
ev = evaluate_profiled_flexcm_point(w0, fctx)

lp("=== REDUCED (flexible_CM) instrumented gradient ===")
# warmup (untimed, excludes JIT compilation from the measured report)
_ = instrumented_shared_family_outer_gradient(w0, ctx, fctx, ev; threaded = false)
_ = instrumented_shared_family_outer_gradient(w0, ctx, fctx, ev; threaded = true)

g_ref, _ = shared_family_outer_gradient(w0, ctx, fctx, ev; threaded = false)
g_ser, meta_ser, rep_ser = instrumented_shared_family_outer_gradient(w0, ctx, fctx, ev; threaded = false)
g_thr, meta_thr, rep_thr = instrumented_shared_family_outer_gradient(w0, ctx, fctx, ev; threaded = true)

err_ser_vs_ref = maximum(abs.(g_ref .- g_ser))
err_thr_vs_ref = maximum(abs.(g_ref .- g_thr))
lp("  instrumented(serial) vs production shared_family_outer_gradient max abs diff = ", err_ser_vs_ref)
lp("  instrumented(threaded) vs production shared_family_outer_gradient max abs diff = ", err_thr_vs_ref)
lp("  REDUCED serial   report: ", rep_ser)
lp("  REDUCED threaded report: ", rep_thr)
lp("  REDUCED serial   accounting_ratio = ", rep_ser.accounting_ratio, "  PASS(>=0.98)=", accounting_check(rep_ser))
# task §7 fix (2026-08-04): the threaded report's own accounting_ratio (sum-of-coordinate-times /
# wall) is legitimately >1 under real concurrency -- printed for visibility, NOT gated on
# accounting_check any more (that would prove nothing about wall-time closure). The genuine
# wall-time gate for a threaded report is wall_accounting_ratio (critical-path-based).
lp("  REDUCED threaded accounting_ratio (aggregate-worker/wall, NOT a wall-time gate, informational) = ", rep_thr.accounting_ratio)
lp("  REDUCED threaded wall_accounting_ratio = ", rep_thr.wall_accounting_ratio, "  PASS(0.90-1.15)=", wall_accounting_check(rep_thr))
lp("  REDUCED threaded critical_path_coord_s = ", rep_thr.critical_path_coord_s, "  threads_used = ", rep_thr.threads_used, " (>1 required)")

# ---- FULL side (unrestricted) ----
lp("")
lp("=== FULL (unrestricted) instrumented gradient ===")
pe_f = build_pivot_elimination(ctx)
x_free0 = ctx.θ0_up[ctx.free_idx]

_ = instrumented_composite_gradient_at_fast(x_free0, ctx, pe_f; threaded = false)
_ = instrumented_composite_gradient_at_fast(x_free0, ctx, pe_f; threaded = true)

gF_ref, _ = composite_gradient_at_fast(x_free0, ctx, pe_f; threaded = false, h_mode = :adaptive, validate_frac = 0.0)
gF_ser, metaF_ser, repF_ser = instrumented_composite_gradient_at_fast(x_free0, ctx, pe_f; threaded = false)
gF_thr, metaF_thr, repF_thr = instrumented_composite_gradient_at_fast(x_free0, ctx, pe_f; threaded = true)

errF_ser_vs_ref = maximum(abs.(gF_ref .- gF_ser))
errF_thr_vs_ref = maximum(abs.(gF_ref .- gF_thr))
lp("  instrumented(serial) vs production composite_gradient_at_fast max abs diff = ", errF_ser_vs_ref)
lp("  instrumented(threaded) vs production composite_gradient_at_fast max abs diff = ", errF_thr_vs_ref)
lp("  FULL serial   report: ", repF_ser)
lp("  FULL threaded report: ", repF_thr)
lp("  FULL serial   accounting_ratio = ", repF_ser.accounting_ratio, "  PASS(>=0.98)=", accounting_check(repF_ser))
lp("  FULL threaded accounting_ratio (aggregate-worker/wall, NOT a wall-time gate, informational) = ", repF_thr.accounting_ratio)
lp("  FULL threaded wall_accounting_ratio = ", repF_thr.wall_accounting_ratio, "  PASS(0.90-1.15)=", wall_accounting_check(repF_thr))
lp("  FULL threaded critical_path_coord_s = ", repF_thr.critical_path_coord_s, "  threads_used = ", repF_thr.threads_used, " (>1 required)")

lp("")
gate = (err_ser_vs_ref == 0.0) && (err_thr_vs_ref == 0.0) && accounting_check(rep_ser) && wall_accounting_check(rep_thr) &&
       (errF_ser_vs_ref == 0.0) && (errF_thr_vs_ref == 0.0) && accounting_check(repF_ser) && wall_accounting_check(repF_thr) &&
       (rep_thr.threads_used > 1) && (repF_thr.threads_used > 1)
lp("MATCHED_TIMER_GATE_W", W_VAL, ": ", gate ? "PASS" : "FAIL")
