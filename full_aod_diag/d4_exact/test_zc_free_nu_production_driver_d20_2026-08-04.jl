# fix/profiled-functional-readiness-closeout-2026-08-03, task §8.4: D20/W=20,000 scale extension
# of test_zc_free_nu_production_driver_d4_2026-08-04.jl -- SAME real-KNITRO-outer-solve gate
# (joint [gp;A_free;eta_nu] search, checkpoint/resume round-trip of eta_nu), at real D20 data.
# W_VAL controls scale (set below; pass as ARGS[1] to override).
const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
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
          "profiled_reduced_basis_cache_bank_2026-08-02.jl",
          "profiled_screen_bridge_2026-08-02.jl",
          "cm_aspace_coordinate.jl", "profiled_powered_relative_a_2026-08-04.jl",
          "profiled_coordinate_mode_dispatch_2026-08-04.jl",
          "profiled_production_outer_constrained_2026-08-02.jl",
          "profiled_zc_free_nu_production_driver_2026-08-04.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, Random, LinearAlgebra

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

W_VAL = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20_000
println("W_VAL = $W_VAL"); flush(stdout)
t0 = time()
println("PID=", getpid()); flush(stdout)

t_ctx = @elapsed ctx = d20_real_setup_design(W = W_VAL, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0)
@printf("context build: %.2fs\n", t_ctx); flush(stdout)
D = ctx.D
korea_idx, brazil_idx = 14, 3
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
θ0 = ctx.θ0_up
z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*ctx.D_dest], D, ctx.D_dest))
gauge = build_anchor_gauge(z_calib, spec)
pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
w_profiled_calib = reduce_to_w_profiled(θ0[3+D], z_calib, pe)

println("\n" * "#"^90); println("FAMILY: origin_ZC (D20/W=$W_VAL)"); println("#"^90); flush(stdout)
layout_o = OriginByPowerLayout(D, 1, 0)
νvec0 = fill(1.0, D)
aug_reduced_oz = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
octx_reduced_oz = build_originzc_core_hess_ctx(aug_reduced_oz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
fctx_oz = build_originzc_family_ctx(ctx, spec, pe, layout, aug_reduced_oz)
pes_oz = OriginZCPointEvalState(octx_reduced_oz, νvec0)
eta_nu0_oz = zeros(n_eta(fctx_oz.zc_layout))
eta_bounds_oz = originzc_default_nu_bounds(ctx, fctx_oz.zc_layout)
@printf("origin_ZC: n_eta=%d, eta_bounds[1]=%s\n", length(eta_nu0_oz), eta_bounds_oz[1]); flush(stdout)

ckpt_dir = mktempdir()
ckpt_path_oz = joinpath(ckpt_dir, "originzc_freenu_d20.jls")

println("\n--- Part 1: real KNITRO solve, joint [gp;A_free;eta_nu], real D20/W=$W_VAL ---"); flush(stdout)
result1_oz = run_profiled_upper_constrained_free_nu("d20_freenu_oz_part1", w_profiled_calib, eta_nu0_oz;
    fctx = fctx_oz, evaluate_fn_free_nu = evaluate_profiled_originzc_point,
    gradient_fn_free_nu = reduced_originzc_outer_gradient_with_eta, pes = pes_oz,
    ctx = ctx, pe = pe, eta_bounds = eta_bounds_oz, delta = 1.0, maxtime_real = 45.0, hessopt_tag = "sr1",
    checkpoint_path = ckpt_path_oz, checkpoint_interval_s = 1000.0, verbose = false)

check("origin_ZC D20: at least one eval ran", result1_oz.n_eval > 0)
check("origin_ZC D20: at least one gradient call ran", result1_oz.n_grad > 0)
check("origin_ZC D20: checkpoint file written", isfile(ckpt_path_oz))
loaded1_oz = load_cm_checkpoint_v11(ckpt_path_oz)
check("origin_ZC D20: checkpoint eta_nu length matches n_eta", length(loaded1_oz.eta_nu) == length(eta_nu0_oz))
check("origin_ZC D20: checkpoint eta_nu is NOT the hardcoded-empty base-driver value", !isempty(loaded1_oz.eta_nu))
@printf("origin_ZC D20 part1: n_eval=%d n_grad=%d wall=%.2fs xsol_eta=%s\n", result1_oz.n_eval, result1_oz.n_grad, result1_oz.wall, result1_oz.xsol_eta)
flush(stdout)
eta_moved_oz = norm(result1_oz.xsol_eta .- eta_nu0_oz) > 1e-6
check("origin_ZC D20: eta_nu genuinely moved from its start value", eta_moved_oz)

println("\n--- Part 2: resume_from restores BOTH economic vars AND eta_nu (D20) ---"); flush(stdout)
result2_oz = run_profiled_upper_constrained_free_nu("d20_freenu_oz_part2", w_profiled_calib, eta_nu0_oz;
    fctx = fctx_oz, evaluate_fn_free_nu = evaluate_profiled_originzc_point,
    gradient_fn_free_nu = reduced_originzc_outer_gradient_with_eta, pes = pes_oz,
    ctx = ctx, pe = pe, eta_bounds = eta_bounds_oz, delta = 1.0, maxtime_real = 15.0, hessopt_tag = "sr1",
    checkpoint_path = ckpt_path_oz, checkpoint_interval_s = 1000.0, verbose = false, resume_from = ckpt_path_oz)
check("origin_ZC D20: resumed n_eval >= part1 (cumulative)", result2_oz.n_eval >= result1_oz.n_eval)
check("origin_ZC D20: resumed n_grad >= part1 (cumulative)", result2_oz.n_grad >= result1_oz.n_grad)
@printf("origin_ZC D20 part2 (resumed): n_eval=%d n_grad=%d wall=%.2fs\n", result2_oz.n_eval, result2_oz.n_grad, result2_oz.wall)
flush(stdout)

println("\n" * "#"^90); println("FAMILY: CM_plus_ZC (D20/W=$W_VAL)"); println("#"^90); flush(stdout)
K_MEAN, K_PAIR, L_GRID = 1, 0, 3
νvec0_cm = fill(1.0, K_MEAN)
aug_reduced_cz = build_cm_meanzc_augmented_obj(ctx, CS; L = L_GRID, K_mean = K_MEAN, K_pair = K_PAIR,
    base_obj = reduced_obj0, profiled_layout = layout)
cctx_reduced_cz = build_cm_meanzc_bin_ctx(ctx, aug_reduced_cz; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = true, inner_fg_backend = :dense_reference,
    profiled_layout = layout)
bins_u32 = cctx_reduced_cz.Bidx isa Matrix{UInt32} ? cctx_reduced_cz.Bidx : Matrix{UInt32}(cctx_reduced_cz.Bidx)
fctx_cz = build_cmzc_family_ctx(ctx, spec, pe, layout, aug_reduced_cz, cctx_reduced_cz, bins_u32)
pes_cz = CMZCPointEvalState(cctx_reduced_cz, νvec0_cm)
eta_nu0_cz = zeros(n_eta(fctx_cz.zc_layout))
eta_bounds_cz = originzc_default_nu_bounds(ctx, fctx_cz.zc_layout)
@printf("CM_plus_ZC: n_eta=%d, eta_bounds[1]=%s\n", length(eta_nu0_cz), eta_bounds_cz[1]); flush(stdout)

ckpt_path_cz = joinpath(ckpt_dir, "cmzc_freenu_d20.jls")
result1_cz = run_profiled_upper_constrained_free_nu("d20_freenu_cz_part1", w_profiled_calib, eta_nu0_cz;
    fctx = fctx_cz, evaluate_fn_free_nu = evaluate_profiled_cmzc_point,
    gradient_fn_free_nu = reduced_cmzc_outer_gradient_with_eta, pes = pes_cz,
    ctx = ctx, pe = pe, eta_bounds = eta_bounds_cz, delta = 1.0, maxtime_real = 45.0, hessopt_tag = "sr1",
    checkpoint_path = ckpt_path_cz, checkpoint_interval_s = 1000.0, verbose = false)

check("CM_plus_ZC D20: at least one eval ran", result1_cz.n_eval > 0)
check("CM_plus_ZC D20: at least one gradient call ran", result1_cz.n_grad > 0)
check("CM_plus_ZC D20: checkpoint file written", isfile(ckpt_path_cz))
loaded1_cz = load_cm_checkpoint_v11(ckpt_path_cz)
check("CM_plus_ZC D20: checkpoint eta_nu length matches n_eta", length(loaded1_cz.eta_nu) == length(eta_nu0_cz))
check("CM_plus_ZC D20: checkpoint eta_nu is NOT the hardcoded-empty base-driver value", !isempty(loaded1_cz.eta_nu))
@printf("CM_plus_ZC D20 part1: n_eval=%d n_grad=%d wall=%.2fs xsol_eta=%s\n", result1_cz.n_eval, result1_cz.n_grad, result1_cz.wall, result1_cz.xsol_eta)
flush(stdout)
eta_moved_cz = norm(result1_cz.xsol_eta .- eta_nu0_cz) > 1e-6
check("CM_plus_ZC D20: eta_nu genuinely moved from its start value", eta_moved_cz)

println("\n--- CM_plus_ZC D20: resume round-trip ---"); flush(stdout)
result2_cz = run_profiled_upper_constrained_free_nu("d20_freenu_cz_part2", w_profiled_calib, eta_nu0_cz;
    fctx = fctx_cz, evaluate_fn_free_nu = evaluate_profiled_cmzc_point,
    gradient_fn_free_nu = reduced_cmzc_outer_gradient_with_eta, pes = pes_cz,
    ctx = ctx, pe = pe, eta_bounds = eta_bounds_cz, delta = 1.0, maxtime_real = 15.0, hessopt_tag = "sr1",
    checkpoint_path = ckpt_path_cz, checkpoint_interval_s = 1000.0, verbose = false, resume_from = ckpt_path_cz)
check("CM_plus_ZC D20: resumed n_eval >= part1 (cumulative)", result2_cz.n_eval >= result1_cz.n_eval)
check("CM_plus_ZC D20: resumed n_grad >= part1 (cumulative)", result2_cz.n_grad >= result1_cz.n_grad)
@printf("CM_plus_ZC D20 part2 (resumed): n_eval=%d n_grad=%d wall=%.2fs\n", result2_cz.n_eval, result2_cz.n_grad, result2_cz.wall)
flush(stdout)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
@printf("TOTAL WALL: %.1fs\n", time() - t0)
exit(ALL_PASS[] ? 0 : 1)
