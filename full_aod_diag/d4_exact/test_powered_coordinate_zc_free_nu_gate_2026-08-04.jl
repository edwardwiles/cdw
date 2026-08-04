# Task §6.1/6.2 continuation (2026-08-04): powered A-coordinate mode wired through
# run_profiled_upper_constrained_free_nu (origin_ZC/CM_plus_ZC), the SEPARATE driver from
# run_profiled_upper_constrained (already gated for flexible_cm/common_frechet in
# test_powered_coordinate_production_gate_2026-08-04.jl). Real D4 KNITRO context, origin_ZC.
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
          "cm_exact_cache_production.jl", "cm_checkpoint.jl",
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
          "profiled_originzc_family_adapter_2026-08-02.jl",
          "profiled_cmzc_family_adapter_2026-08-02.jl",
          "profiled_zc_lane_point_evaluators_2026-08-02.jl",
          "profiled_zc_free_eta_2026-08-04.jl",
          "profiled_production_outer_runner_2026-08-01.jl",
          "profiled_ab_comparability_and_plumbing_2026-08-01.jl",
          "cm_aspace_coordinate.jl", "profiled_powered_relative_a_2026-08-04.jl",
          "profiled_coordinate_mode_dispatch_2026-08-04.jl",
          "profiled_reduced_basis_cache_bank_2026-08-02.jl",
          "profiled_screen_bridge_2026-08-02.jl",
          "profiled_production_outer_constrained_2026-08-02.jl",
          "profiled_zc_free_nu_production_driver_2026-08-04.jl"]
    include(joinpath(D4X, f))
end

lp(xs...) = (println(xs...); flush(stdout))
results = Dict{Symbol,Bool}()

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(3 => 1))
θ0 = ctx.θ0_up
z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))
gauge = build_anchor_gauge(z_calib, spec)
pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
gp0 = θ0[3+D]
w0 = reduce_to_w_profiled(gp0, z_calib, pe)

θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)

layout_o = OriginByPowerLayout(D, 1, 0)
νvec0 = fill(1.0, D)
aug = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
octx = build_originzc_core_hess_ctx(aug, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
fctx = build_originzc_family_ctx(ctx, spec, pe, layout, aug)
pes = OriginZCPointEvalState(octx, νvec0)
eta_nu0 = zeros(n_eta(fctx.zc_layout))
eta_bounds = originzc_default_nu_bounds(ctx, fctx.zc_layout)
lp("origin_ZC D4: n_free=", length(w0) - 1, " n_eta=", length(eta_nu0))

ckpt_dir = mktempdir()

lp("\n=== 1. Native mode short run (regression) ===")
res_native = run_profiled_upper_constrained_free_nu("gate_oz_native", w0, eta_nu0;
    fctx = fctx, evaluate_fn_free_nu = evaluate_profiled_originzc_point,
    gradient_fn_free_nu = (w, eta, c, ff, ev) -> reduced_originzc_outer_gradient_with_eta(w, eta, c, ff, ev; threaded = false), pes = pes,
    ctx = ctx, pe = pe, eta_bounds = eta_bounds, delta = 1.0, maxtime_real = 8.0, hessopt_tag = "sr1",
    a_coordinate_mode = :profiled_pivot_anchor_relative,
    checkpoint_path = joinpath(ckpt_dir, "native.jls"), checkpoint_interval_s = 1000.0, verbose = false)
lp("  n_eval=", res_native.n_eval, " n_grad=", res_native.n_grad, " status=", res_native.knitro_status,
   " eta_nu_final=", res_native.best === nothing ? "none" : res_native.best.eta_nu)
results[:native_completes] = res_native.n_eval > 0 && res_native.n_grad > 0
results[:native_eta_moves] = res_native.best !== nothing && maximum(abs.(res_native.best.eta_nu .- eta_nu0)) > 1e-8

lp("\n=== 2. Powered mode short run ===")
res_powered = run_profiled_upper_constrained_free_nu("gate_oz_powered", w0, eta_nu0;
    fctx = fctx, evaluate_fn_free_nu = evaluate_profiled_originzc_point,
    gradient_fn_free_nu = (w, eta, c, ff, ev) -> reduced_originzc_outer_gradient_with_eta(w, eta, c, ff, ev; threaded = false), pes = pes,
    ctx = ctx, pe = pe, eta_bounds = eta_bounds, delta = 1.0, maxtime_real = 8.0, hessopt_tag = "sr1",
    a_coordinate_mode = :profiled_powered_relative_A,
    checkpoint_path = joinpath(ckpt_dir, "powered.jls"), checkpoint_interval_s = 1000.0, verbose = false)
lp("  n_eval=", res_powered.n_eval, " n_grad=", res_powered.n_grad, " status=", res_powered.knitro_status,
   " eta_nu_final=", res_powered.best === nothing ? "none" : res_powered.best.eta_nu)
results[:powered_completes] = res_powered.n_eval > 0 && res_powered.n_grad > 0
results[:powered_eta_moves] = res_powered.best !== nothing && maximum(abs.(res_powered.best.eta_nu .- eta_nu0)) > 1e-8

lp("\n=== 3. Checkpoint mode-mismatch refusal ===")
local mismatch_refused = false
try
    global mismatch_refused
    run_profiled_upper_constrained_free_nu("gate_oz_mismatch", w0, eta_nu0;
        fctx = fctx, evaluate_fn_free_nu = evaluate_profiled_originzc_point,
        gradient_fn_free_nu = (w, eta, c, ff, ev) -> reduced_originzc_outer_gradient_with_eta(w, eta, c, ff, ev; threaded = false), pes = pes,
        ctx = ctx, pe = pe, eta_bounds = eta_bounds, delta = 1.0, maxtime_real = 2.0, hessopt_tag = "sr1",
        a_coordinate_mode = :profiled_powered_relative_A,
        checkpoint_path = joinpath(ckpt_dir, "native.jls"), checkpoint_interval_s = 1000.0,
        resume_from = joinpath(ckpt_dir, "native.jls"), verbose = false)
catch e
    global mismatch_refused = occursin("namespace mismatch", string(e))
    lp("  resume under different mode correctly refused: ", string(e)[1:min(150, end)])
end
results[:checkpoint_mode_mismatch_refused] = mismatch_refused

lp("\nALL_RESULTS: ", results)
all_pass = all(values(results))
lp("\nPOWERED_COORDINATE_ZC_FREE_NU_GATE (D4, origin_ZC): ", all_pass ? "PASS" : "FAIL")
all_pass || error("test_powered_coordinate_zc_free_nu_gate_2026-08-04: one or more checks failed -- see ALL_RESULTS above")
