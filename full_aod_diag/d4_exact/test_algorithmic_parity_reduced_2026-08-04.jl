# Task §11 (profiled-outer-ab-completion-2026-08-04): REDUCED-side algorithmic-parity confirmation.
# Real D20/W=20,000, JULIA_NUM_THREADS=1 (launch with -t 1), OPENBLAS_NUM_THREADS=1,
# threaded_gradient=false (coordinate threading disabled), cache=nothing (bandwidth cache
# disabled, the driver's own default), same outer algorithm/options both arms, powered A-coordinate
# mode (since it passed its own gates). Confirms the REDUCED side of an algorithmic-parity harness
# genuinely runs under these settings -- see SECTION11_ALGORITHMIC_PARITY.md for why the FULL side
# was not built this session (a precise, honest blocker, not silently skipped).
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
          "profiled_originzc_family_adapter_2026-08-02.jl",
          "profiled_cmzc_family_adapter_2026-08-02.jl",
          "profiled_zc_lane_point_evaluators_2026-08-02.jl",
          "profiled_zc_free_eta_2026-08-04.jl",
          "profiled_reduced_basis_cache_bank_2026-08-02.jl",
          "profiled_screen_bridge_2026-08-02.jl",
          "profiled_production_outer_runner_2026-08-01.jl",
          "profiled_ab_comparability_and_plumbing_2026-08-01.jl",
          "cm_aspace_coordinate.jl", "profiled_powered_relative_a_2026-08-04.jl",
          "profiled_coordinate_mode_dispatch_2026-08-04.jl",
          "profiled_production_outer_constrained_2026-08-02.jl",
          "profiled_zc_free_nu_production_driver_2026-08-04.jl"]
    include(joinpath(D4X, f))
end

lp(xs...) = (println(xs...); flush(stdout))
const W_VAL = 20000
const MAXTIME = 60.0
lp("Threads.nthreads() = ", Threads.nthreads(), " (algorithmic-parity requires 1)")
Threads.nthreads() == 1 || error("test_algorithmic_parity_reduced_2026-08-04: must be launched with -t 1 for a genuine algorithmic-parity confirmation, got Threads.nthreads()=", Threads.nthreads())

ctx = d20_real_setup_design(; W = W_VAL, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = [(3, 14)], σHat = 3.0)
D = ctx.D
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(14 => 3))
θ0 = ctx.θ0_up
z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*ctx.D_dest], D, ctx.D_dest))
gauge = build_anchor_gauge(z_calib, spec)
pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
gp0 = θ0[3+D]
w0 = reduce_to_w_profiled(gp0, z_calib, pe)

θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)

results = Dict{Symbol,Bool}()

lp("\n=== flexible_cm, algorithmic parity (1 thread, no cache, threaded_gradient=false, powered mode) ===")
aug = build_cm_augmented_obj_archB(ctx, CS; L = 50, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
cctx = build_cm_bin_ctx(ctx, aug; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = false)
fctx = build_flexcm_family_ctx(ctx, spec, pe, layout, cctx)
ckpt = tempname() * ".jls"
res_cm = run_profiled_upper_constrained("algparity_flexcm", w0; fctx = fctx,
    evaluate_fn = (w, ff) -> evaluate_profiled_flexcm_point(w, ff),
    ctx = ctx, pe = pe, delta = 1.0, maxtime_real = MAXTIME, hessopt_tag = "sr1",
    a_coordinate_mode = :profiled_powered_relative_A,
    checkpoint_path = ckpt, checkpoint_interval_s = 1000.0, verbose = false,
    threaded_gradient = false, cache = nothing)
lp("  n_eval=", res_cm.n_eval, " n_grad=", res_cm.n_grad, " wall=", res_cm.wall,
   " best=", res_cm.best === nothing ? "none" : "gp=$(res_cm.best.gp) Delta=$(res_cm.best.Delta)", " status=", res_cm.knitro_status)
results[:flexible_cm] = res_cm.n_eval > 0 && res_cm.n_grad > 0

lp("\n=== origin_ZC, algorithmic parity (1 thread, no cache, threaded=false, powered mode) ===")
layout_o = OriginByPowerLayout(D, 1, 0)
νvec0 = fill(1.0, D)
aug_o = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
octx_o = build_originzc_core_hess_ctx(aug_o, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
fctx_o = build_originzc_family_ctx(ctx, spec, pe, layout, aug_o)
pes_o = OriginZCPointEvalState(octx_o, νvec0)
eta_nu0 = zeros(n_eta(fctx_o.zc_layout))
eta_bounds = originzc_default_nu_bounds(ctx, fctx_o.zc_layout)
ckpt_o = tempname() * ".jls"
res_oz = run_profiled_upper_constrained_free_nu("algparity_originzc", w0, eta_nu0;
    fctx = fctx_o, evaluate_fn_free_nu = evaluate_profiled_originzc_point,
    gradient_fn_free_nu = (w, eta, c, ff, ev) -> reduced_originzc_outer_gradient_with_eta(w, eta, c, ff, ev; threaded = false),
    pes = pes_o, ctx = ctx, pe = pe, eta_bounds = eta_bounds, delta = 1.0, maxtime_real = MAXTIME,
    hessopt_tag = "sr1", a_coordinate_mode = :profiled_powered_relative_A,
    checkpoint_path = ckpt_o, checkpoint_interval_s = 1000.0, verbose = false, cache = nothing)
lp("  n_eval=", res_oz.n_eval, " n_grad=", res_oz.n_grad, " wall=", res_oz.wall,
   " best=", res_oz.best === nothing ? "none" : "gp=$(res_oz.best.gp) Delta=$(res_oz.best.Delta)", " status=", res_oz.knitro_status)
results[:origin_zc] = res_oz.n_eval > 0 && res_oz.n_grad > 0

lp("\nALL_RESULTS: ", results)
all_pass = all(values(results))
lp("ALGORITHMIC_PARITY_REDUCED_SIDE_CONFIRMATION: ", all_pass ? "PASS" : "FAIL")
all_pass || error("test_algorithmic_parity_reduced_2026-08-04: one or more checks failed -- see ALL_RESULTS above")
