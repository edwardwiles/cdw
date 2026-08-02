# Phase 13 (integration/phase12-13-runner-checkpoints-2026-08-02) D20/W20000 gate: real
# production-scale run of the wired outer runner for origin-ZC and CM+ZC (the two families with
# a real restriction outer parameter, per the task's own instruction), :fixed_gp_parameterization_ab
# mode, screens ON (real D20 ctx build_screen=true default), exact cache + checkpoint ON.
# Confirms the wired runner produces a REAL end-to-end result at production scale, not just that
# it doesn't crash. W is read from ENV["PHASE13_D20_W"] (defaults to 20000, per task's own ask).
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
          "profiled_reduced_basis_cache_bank_2026-08-02.jl",
          "profiled_screen_bridge_2026-08-02.jl",
          "dual_bank.jl", "cm_dual_bank_production.jl",
          "profiled_production_outer_runner_2026-08-01.jl",
          "profiled_ab_comparability_and_plumbing_2026-08-01.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, Random
flush(stdout)

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
    flush(stdout)
end

function build_profiled_ab_spec_pe(ctx; global_overrides::Dict{Int,Int} = Dict{Int,Int}())
    D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    spec = build_anchor_spec_from_ctx(ctx; global_overrides = global_overrides)
    θ0 = ctx.θ0_up
    z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))
    gauge = build_anchor_gauge(z_calib, spec)
    pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
    return spec, gauge, pe
end
function reduce_calibration_to_w_profiled(ctx, pe::PivotGravityElimOnRetained)
    D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    θ0 = ctx.θ0_up
    z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))
    gp0 = θ0[3+D]
    return reduce_to_w_profiled(gp0, z_calib, pe)
end

const W_VAL = parse(Int, get(ENV, "PHASE13_D20_W", "20000"))
println("="^90); println("Building real D=20 :exclude_row context at W=$W_VAL (build_screen=true default) ..."); flush(stdout)
ctx = d20_real_setup(W = W_VAL, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
D = ctx.D; Ddest = ctx.D_dest
@assert D == 20 && Ddest == 19 "expected live D=20, Ddest=19 -- got D=$D, Ddest=$Ddest"
check("ctx.pairwise precomputed by default (build_screen=true default)", ctx.pairwise !== nothing)

korea_idx = 14; brazil_idx = 3
spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(korea_idx => brazil_idx))
θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
w_profiled_calib = reduce_calibration_to_w_profiled(ctx, pe)
println("D20 outer_dim_profiled = $(outer_dim_profiled(pe))  total_reduced_economic_moments=$(layout.total_reduced_economic_moments)"); flush(stdout)

ckpt_dir = mktempdir()

println("\n" * "="^90); println("FAMILY 1: origin-ZC, real D20/W=$W_VAL"); println("="^90); flush(stdout)
layout_o = OriginByPowerLayout(D, 1, 0)
νvec0 = fill(1.0, D)
t_build_oz = @elapsed begin
    aug_reduced_oz = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
    octx_reduced_oz = build_originzc_core_hess_ctx(aug_reduced_oz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
        zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
end
println("built origin-ZC reduced octx in $(round(t_build_oz,digits=1))s"); flush(stdout)
fctx_oz = build_originzc_family_ctx(ctx, spec, pe, layout, aug_reduced_oz)
pes_oz = OriginZCPointEvalState(octx_reduced_oz, νvec0)
inner_digest_oz = profiled_inner_layout_digest(:ZC_only; K_mean = 1, K_pair = 0,
    core_hessian_backend = :exact_winner_pair_parallel, zc_cross_hessian_backend = :winner_bin)
cache_oz = profiled_cm_production_exact_cache()
bank_oz = ProfiledRestrictedDualBank(8)
ckpt_path_oz = joinpath(ckpt_dir, "originzc_d20.jls")

t_run_oz = @elapsed result_oz, manifest_oz, config_oz = run_profiled_production_outer(
    :fixed_gp_parameterization_ab, "originzc_d20_w$(W_VAL)", w_profiled_calib;
    fctx = fctx_oz, evaluate_fn = (w, f) -> evaluate_profiled_originzc_point(w, f, pes_oz),
    ctx = ctx, pe = pe, maxtime_real = 120.0, hessopt_tag = "sr1", maxit_override = 15,
    use_screen = true, cache = cache_oz, bank = bank_oz,
    checkpoint_path = ckpt_path_oz, checkpoint_interval_s = 10.0,
    inner_layout_digest = inner_digest_oz)
@printf("origin-ZC D20/W=%d: wall=%.1fs n_eval=%d n_grad=%d knitro_status=%d\n",
    W_VAL, t_run_oz, result_oz.n_eval, result_oz.n_grad_calls, result_oz.knitro_status)
check("origin-ZC D20: at least one eval ran", result_oz.n_eval > 0)
check("origin-ZC D20: found a verified-feasible best incumbent", result_oz.best !== nothing)
result_oz.best !== nothing && @printf("  best Delta=%.10f at eval %d, t=%.1fs\n", result_oz.best.Delta, result_oz.best.n_eval, result_oz.best.t_elapsed)
check("origin-ZC D20: checkpoint written and round-trips as CMCheckpointV11", isfile(ckpt_path_oz) && load_cm_checkpoint_v11(ckpt_path_oz) isa CMCheckpointV11)
check("origin-ZC D20: exact cache recorded lookups", PROFILED_CM_EXACT_CACHE_COUNTERS[].lookups > 0)
flush(stdout)

println("\n" * "="^90); println("FAMILY 2: CM+ZC, real D20/W=$W_VAL"); println("="^90); flush(stdout)
const K_MEAN, K_PAIR, L_GRID = 1, 0, 10
νvec0_cm = fill(1.0, K_MEAN)
t_build_cz = @elapsed begin
    aug_reduced_cz = build_cm_meanzc_augmented_obj(ctx, CS; L = L_GRID, K_mean = K_MEAN, K_pair = K_PAIR,
        base_obj = reduced_obj0, profiled_layout = layout)
    cctx_reduced_cz = build_cm_meanzc_bin_ctx(ctx, aug_reduced_cz; core_hessian_backend = :exact_winner_pair_parallel,
        zc_cross_hessian_backend = :winner_bin, threaded_bins = false, inner_fg_backend = :dense_reference,
        profiled_layout = layout)
end
println("built CM+ZC reduced cctx in $(round(t_build_cz,digits=1))s"); flush(stdout)
bins_u32 = cctx_reduced_cz.Bidx isa Matrix{UInt32} ? cctx_reduced_cz.Bidx : Matrix{UInt32}(cctx_reduced_cz.Bidx)
fctx_cz = build_cmzc_family_ctx(ctx, spec, pe, layout, aug_reduced_cz, cctx_reduced_cz, bins_u32)
pes_cz = CMZCPointEvalState(cctx_reduced_cz, νvec0_cm)
inner_digest_cz = profiled_inner_layout_digest(:CM_plus_ZC; K_mean = K_MEAN, K_pair = K_PAIR, L = L_GRID,
    core_hessian_backend = :exact_winner_pair_parallel, zc_cross_hessian_backend = :winner_bin)
cache_cz = profiled_cm_production_exact_cache()
bank_cz = ProfiledRestrictedDualBank(8)
ckpt_path_cz = joinpath(ckpt_dir, "cmzc_d20.jls")

t_run_cz = @elapsed result_cz, manifest_cz, config_cz = run_profiled_production_outer(
    :fixed_gp_parameterization_ab, "cmzc_d20_w$(W_VAL)", w_profiled_calib;
    fctx = fctx_cz, evaluate_fn = (w, f) -> evaluate_profiled_cmzc_point(w, f, pes_cz),
    ctx = ctx, pe = pe, maxtime_real = 120.0, hessopt_tag = "sr1", maxit_override = 15,
    use_screen = true, cache = cache_cz, bank = bank_cz,
    checkpoint_path = ckpt_path_cz, checkpoint_interval_s = 10.0,
    inner_layout_digest = inner_digest_cz)
@printf("CM+ZC D20/W=%d: wall=%.1fs n_eval=%d n_grad=%d knitro_status=%d\n",
    W_VAL, t_run_cz, result_cz.n_eval, result_cz.n_grad_calls, result_cz.knitro_status)
check("CM+ZC D20: at least one eval ran", result_cz.n_eval > 0)
check("CM+ZC D20: found a verified-feasible best incumbent", result_cz.best !== nothing)
result_cz.best !== nothing && @printf("  best Delta=%.10f at eval %d, t=%.1fs\n", result_cz.best.Delta, result_cz.best.n_eval, result_cz.best.t_elapsed)
check("CM+ZC D20: checkpoint written and round-trips as CMCheckpointV11", isfile(ckpt_path_cz) && load_cm_checkpoint_v11(ckpt_path_cz) isa CMCheckpointV11)
check("CM+ZC D20: checkpoint namespace differs from origin-ZC's", config_cz.checkpoint_namespace != config_oz.checkpoint_namespace)

println()
println("W=$W_VAL  ", ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
