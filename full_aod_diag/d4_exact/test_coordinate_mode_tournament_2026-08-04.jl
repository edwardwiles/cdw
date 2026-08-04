# Task §10 (profiled-outer-ab-completion-2026-08-04): coordinate-mode tournament, native vs
# powered REDUCED A-coordinate mode. Real D20/W=20,000 KNITRO context, run via the SAME production
# driver (run_profiled_upper_constrained / run_profiled_upper_constrained_free_nu) both modes use,
# same decoded initial state, same outer algorithm/options, same threading/cache policy, same
# time/evaluation caps, both execution orders. GATE_FAMILY selects flexible_cm (representative of
# the run_profiled_upper_constrained code path, shared with common_frechet) or origin_zc
# (representative of run_profiled_upper_constrained_free_nu, shared with cm_meanzc) --
# common_frechet/cm_meanzc were not independently run (identical wiring, not independently
# executed this session, same honesty discipline as sections 6/9).
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
const W_VAL = parse(Int, get(ENV, "GATE_W", "20000"))
const FAMILY = Symbol(get(ENV, "GATE_FAMILY", "flexible_cm"))
const MAXTIME = parse(Float64, get(ENV, "GATE_MAXTIME", "150.0"))
FAMILY in (:flexible_cm, :origin_zc) ||
    error("test_coordinate_mode_tournament_2026-08-04: GATE_FAMILY must be flexible_cm|origin_zc, got $FAMILY")
lp("Threads.nthreads() = ", Threads.nthreads(), "  W=", W_VAL, "  FAMILY=", FAMILY, "  maxtime=", MAXTIME)

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

function run_arm(mode::Symbol, label::String)
    ckpt = tempname() * ".jls"
    if FAMILY == :flexible_cm
        aug = build_cm_augmented_obj_archB(ctx, CS; L = 50, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
        cctx = build_cm_bin_ctx(ctx, aug; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = true)
        fctx = build_flexcm_family_ctx(ctx, spec, pe, layout, cctx)
        evaluate_fn = (w, ff) -> evaluate_profiled_flexcm_point(w, ff)
        res = run_profiled_upper_constrained(label, w0; fctx = fctx, evaluate_fn = evaluate_fn,
            ctx = ctx, pe = pe, delta = 1.0, maxtime_real = MAXTIME, hessopt_tag = "sr1",
            a_coordinate_mode = mode, checkpoint_path = ckpt, checkpoint_interval_s = 1000.0,
            verbose = false, threaded_gradient = true)
        return (n_eval = res.n_eval, n_grad = res.n_grad, wall = res.wall,
            best_gp = res.best === nothing ? NaN : res.best.gp,
            best_Delta = res.best === nothing ? NaN : res.best.Delta,
            status = res.knitro_status)
    else # :origin_zc
        layout_o = OriginByPowerLayout(D, 1, 0)
        νvec0 = fill(1.0, D)
        aug = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
        octx = build_originzc_core_hess_ctx(aug, ctx; core_hessian_backend = :exact_winner_pair_parallel,
            zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
        fctx = build_originzc_family_ctx(ctx, spec, pe, layout, aug)
        pes = OriginZCPointEvalState(octx, νvec0)
        eta_nu0 = zeros(n_eta(fctx.zc_layout))
        eta_bounds = originzc_default_nu_bounds(ctx, fctx.zc_layout)
        res = run_profiled_upper_constrained_free_nu(label, w0, eta_nu0;
            fctx = fctx, evaluate_fn_free_nu = evaluate_profiled_originzc_point,
            gradient_fn_free_nu = (w, eta, c, ff, ev) -> reduced_originzc_outer_gradient_with_eta(w, eta, c, ff, ev; threaded = true),
            pes = pes, ctx = ctx, pe = pe, eta_bounds = eta_bounds, delta = 1.0, maxtime_real = MAXTIME,
            hessopt_tag = "sr1", a_coordinate_mode = mode, checkpoint_path = ckpt, checkpoint_interval_s = 1000.0,
            verbose = false)
        return (n_eval = res.n_eval, n_grad = res.n_grad, wall = res.wall,
            best_gp = res.best === nothing ? NaN : res.best.gp,
            best_Delta = res.best === nothing ? NaN : res.best.Delta,
            status = res.knitro_status)
    end
end

lp("\n=== Order A: native then powered ===")
resA_native = run_arm(:profiled_pivot_anchor_relative, "tourn_$(FAMILY)_A_native")
lp("  native: ", resA_native)
resA_powered = run_arm(:profiled_powered_relative_A, "tourn_$(FAMILY)_A_powered")
lp("  powered: ", resA_powered)

lp("\n=== Order B: powered then native ===")
resB_powered = run_arm(:profiled_powered_relative_A, "tourn_$(FAMILY)_B_powered")
lp("  powered: ", resB_powered)
resB_native = run_arm(:profiled_pivot_anchor_relative, "tourn_$(FAMILY)_B_native")
lp("  native: ", resB_native)

lp("\n=== Summary ===")
lp("native  A: n_eval=", resA_native.n_eval, " n_grad=", resA_native.n_grad, " wall=", resA_native.wall, " best_gp=", resA_native.best_gp, " Delta=", resA_native.best_Delta, " status=", resA_native.status)
lp("powered A: n_eval=", resA_powered.n_eval, " n_grad=", resA_powered.n_grad, " wall=", resA_powered.wall, " best_gp=", resA_powered.best_gp, " Delta=", resA_powered.best_Delta, " status=", resA_powered.status)
lp("powered B: n_eval=", resB_powered.n_eval, " n_grad=", resB_powered.n_grad, " wall=", resB_powered.wall, " best_gp=", resB_powered.best_gp, " Delta=", resB_powered.best_Delta, " status=", resB_powered.status)
lp("native  B: n_eval=", resB_native.n_eval, " n_grad=", resB_native.n_grad, " wall=", resB_native.wall, " best_gp=", resB_native.best_gp, " Delta=", resB_native.best_Delta, " status=", resB_native.status)

grad_per_s_native = (resA_native.n_grad + resB_native.n_grad) / (resA_native.wall + resB_native.wall)
grad_per_s_powered = (resA_powered.n_grad + resB_powered.n_grad) / (resA_powered.wall + resB_powered.wall)
lp("\nTOURNAMENT_", FAMILY, "_W", W_VAL, ": native_grad_per_s=", round(grad_per_s_native, digits=3),
   "  powered_grad_per_s=", round(grad_per_s_powered, digits=3),
   "  native_best_gp_avg=", (resA_native.best_gp + resB_native.best_gp)/2,
   "  powered_best_gp_avg=", (resA_powered.best_gp + resB_powered.best_gp)/2)
