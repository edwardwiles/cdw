# Phase 13 (integration/phase12-13-runner-checkpoints-2026-08-02) D4 gate: wires the
# recovered+adapted production outer runner (profiled_production_outer_runner_2026-08-01.jl) to
# actually drive a short real KNITRO outer search, in :fixed_gp_parameterization_ab mode, for
# origin-ZC and CM+ZC, at D4 (fast). Confirms Direct+SR1 wiring, cache/checkpoint plumbing, and
# that the runner produces a REAL result (KNITRO status + a verified-feasible best incumbent),
# not merely that it doesn't crash.
const D4X = @__DIR__
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
          "profiled_reduced_basis_cache_bank_2026-08-02.jl",
          "profiled_screen_bridge_2026-08-02.jl",
          "dual_bank.jl", "cm_dual_bank_production.jl",
          "profiled_production_outer_runner_2026-08-01.jl",
          "profiled_ab_comparability_and_plumbing_2026-08-01.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

"Inlined copy, same rationale/cross-talk-avoidance note as the ZC-lane sibling gates."
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

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
spec, gauge, pe = build_profiled_ab_spec_pe(ctx)
θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
w_profiled_calib = reduce_calibration_to_w_profiled(ctx, pe)
println("D4 outer_dim_profiled = $(outer_dim_profiled(pe))")

println("\n" * "="^90); println("FAMILY 1: origin-ZC through the wired production runner"); println("="^90)
layout_o = OriginByPowerLayout(D, 1, 0)
νvec0 = fill(1.0, D)
aug_reduced_oz = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
octx_reduced_oz = build_originzc_core_hess_ctx(aug_reduced_oz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
fctx_oz = build_originzc_family_ctx(ctx, spec, pe, layout, aug_reduced_oz)
pes_oz = OriginZCPointEvalState(octx_reduced_oz, νvec0)
inner_digest_oz = profiled_inner_layout_digest(:ZC_only; K_mean = 1, K_pair = 0,
    core_hessian_backend = :exact_winner_pair_parallel, zc_cross_hessian_backend = :winner_bin)

cache_oz = profiled_cm_production_exact_cache()
bank_oz = ProfiledRestrictedDualBank(8)
ckpt_dir = mktempdir()
ckpt_path_oz = joinpath(ckpt_dir, "originzc_d4.jls")

result_oz, manifest_oz, config_oz = run_profiled_production_outer(:fixed_gp_parameterization_ab, "originzc_d4", w_profiled_calib;
    fctx = fctx_oz, evaluate_fn = (w, f) -> evaluate_profiled_originzc_point(w, f, pes_oz),
    ctx = ctx, pe = pe, maxtime_real = 30.0, hessopt_tag = "sr1", maxit_override = 15,
    use_screen = false, cache = cache_oz, bank = bank_oz,
    checkpoint_path = ckpt_path_oz, checkpoint_interval_s = 5.0,
    inner_layout_digest = inner_digest_oz)

check("origin-ZC runner: manifest mode recorded correctly", manifest_oz.mode == :fixed_gp_parameterization_ab)
check("origin-ZC runner: manifest gp_free == false (fixed-gp mode)", manifest_oz.gp_free == false)
check("origin-ZC runner: manifest family == :ZC_only", manifest_oz.family == :ZC_only)
check("origin-ZC runner: KNITRO terminated with a recognized status", result_oz.knitro_status in (0, -100, -101, -103, -200, -400, -401)) # -400/-401 = maxit/maxtime hit WITH a feasible point (knitro_status.jl), a real terminal status given this gate's own maxit_override=15
check("origin-ZC runner: at least one eval ran", result_oz.n_eval > 0)
check("origin-ZC runner: found a verified-feasible best incumbent", result_oz.best !== nothing)
@printf("origin-ZC: n_eval=%d n_grad=%d best_Delta=%s wall=%.2fs\n", result_oz.n_eval, result_oz.n_grad_calls,
    result_oz.best === nothing ? "none" : string(result_oz.best.Delta), result_oz.wall_ext)

check("origin-ZC runner: config.economic_parameterization == :profiled_destination_scales", config_oz.economic_parameterization == :profiled_destination_scales)
check("origin-ZC runner: config.inner_dual_layout_digest is the REAL digest, not the placeholder", config_oz.inner_dual_layout_digest == inner_digest_oz && config_oz.inner_dual_layout_digest != "unknown_pending_inner")
check("origin-ZC runner: checkpoint file written", isfile(ckpt_path_oz))
loaded_ckpt_oz = load_cm_checkpoint_v11(ckpt_path_oz)
check("origin-ZC runner: checkpoint round-trips as CMCheckpointV11", loaded_ckpt_oz isa CMCheckpointV11)
check("origin-ZC runner: checkpoint namespace matches config (assert_checkpoint_compatible passes)",
    assert_checkpoint_compatible(config_oz, loaded_ckpt_oz.checkpoint_namespace) === nothing)
check("origin-ZC runner: exact cache recorded real lookups", PROFILED_CM_EXACT_CACHE_COUNTERS[].lookups > 0)
check("origin-ZC runner: dual bank recorded at least one success", length(bank_oz.bank.bank.history) > 0)

println("\n" * "="^90); println("FAMILY 2: CM+ZC through the wired production runner"); println("="^90)
const K_MEAN, K_PAIR, L_GRID = 1, 0, 3
νvec0_cm = fill(1.0, K_MEAN)
aug_reduced_cz = build_cm_meanzc_augmented_obj(ctx, CS; L = L_GRID, K_mean = K_MEAN, K_pair = K_PAIR,
    base_obj = reduced_obj0, profiled_layout = layout)
cctx_reduced_cz = build_cm_meanzc_bin_ctx(ctx, aug_reduced_cz; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = false, inner_fg_backend = :dense_reference,
    profiled_layout = layout)
bins_u32 = cctx_reduced_cz.Bidx isa Matrix{UInt32} ? cctx_reduced_cz.Bidx : Matrix{UInt32}(cctx_reduced_cz.Bidx)
fctx_cz = build_cmzc_family_ctx(ctx, spec, pe, layout, aug_reduced_cz, cctx_reduced_cz, bins_u32)
pes_cz = CMZCPointEvalState(cctx_reduced_cz, νvec0_cm)
inner_digest_cz = profiled_inner_layout_digest(:CM_plus_ZC; K_mean = K_MEAN, K_pair = K_PAIR, L = L_GRID,
    core_hessian_backend = :exact_winner_pair_parallel, zc_cross_hessian_backend = :winner_bin)

cache_cz = profiled_cm_production_exact_cache()
bank_cz = ProfiledRestrictedDualBank(8)
ckpt_path_cz = joinpath(ckpt_dir, "cmzc_d4.jls")

result_cz, manifest_cz, config_cz = run_profiled_production_outer(:fixed_gp_parameterization_ab, "cmzc_d4", w_profiled_calib;
    fctx = fctx_cz, evaluate_fn = (w, f) -> evaluate_profiled_cmzc_point(w, f, pes_cz),
    ctx = ctx, pe = pe, maxtime_real = 30.0, hessopt_tag = "sr1", maxit_override = 15,
    use_screen = false, cache = cache_cz, bank = bank_cz,
    checkpoint_path = ckpt_path_cz, checkpoint_interval_s = 5.0,
    inner_layout_digest = inner_digest_cz)

check("CM+ZC runner: manifest family == :CM_plus_ZC", manifest_cz.family == :CM_plus_ZC)
check("CM+ZC runner: KNITRO terminated with a recognized status", result_cz.knitro_status in (0, -100, -101, -103, -200, -400, -401))
check("CM+ZC runner: at least one eval ran", result_cz.n_eval > 0)
check("CM+ZC runner: found a verified-feasible best incumbent", result_cz.best !== nothing)
@printf("CM+ZC: n_eval=%d n_grad=%d best_Delta=%s wall=%.2fs\n", result_cz.n_eval, result_cz.n_grad_calls,
    result_cz.best === nothing ? "none" : string(result_cz.best.Delta), result_cz.wall_ext)

check("CM+ZC runner: checkpoint namespace DIFFERS from origin-ZC's (different family, no collision)",
    config_cz.checkpoint_namespace != config_oz.checkpoint_namespace)
mismatch_cross_family = false
try
    assert_checkpoint_compatible(config_cz, loaded_ckpt_oz.checkpoint_namespace)
catch e
    global mismatch_cross_family = e isa ErrorException
end
check("CM+ZC config refuses origin-ZC's checkpoint namespace (assert_checkpoint_compatible throws)", mismatch_cross_family)

println("\n" * "="^90); println("§14 A/B comparability check (both arms same mode/backend/family-independent fields)"); println("="^90)
mismatch_ab = false
try
    assert_ab_comparable(manifest_oz, manifest_cz)
catch e
    global mismatch_ab = e isa ErrorException
end
check("assert_ab_comparable correctly refuses to compare two DIFFERENT families (family field must match)", mismatch_ab)

# manifest_oz/manifest_cz both carry REAL :not_yet_wired placeholders at D4 (no screens available
# at this context), and assert_ab_comparable's own contract (its docstring, ADAPTED file above) is
# that it throws on ANY placeholder field on EITHER side -- even comparing a manifest against
# itself -- since "an A/B run cannot claim comparability on a dimension neither arm has actually
# recorded a real value for". So the correct positive-path check needs manifests with NO
# placeholders; build two artificial ones (same family, fully wired subsystems, same everything)
# to prove the pass-through path, rather than reusing manifest_oz (which correctly throws even
# against itself, and SHOULD -- that is not a bug in the function).
real_subsystems = merge(default_production_subsystems_manifest(; use_screen = true, use_cache = true, use_bank = true),
    (restriction_backend = :zc_restriction_operator,))   # restriction_backend has no real wiring anywhere yet (honest
    # :not_yet_wired always) -- overridden to a synthetic non-placeholder value ONLY for this unit-level
    # demonstration of assert_ab_comparable's own pass-through path, not a claim that it is production-wired.
manifest_a = OuterRunManifest(:fixed_gp_parameterization_ab, :ZC_only, "a", ["r1"], 1, false, "sr1", "opt", "digestX", :profiled_destination_scales, real_subsystems, "t")
manifest_b = OuterRunManifest(:fixed_gp_parameterization_ab, :ZC_only, "b", ["r1"], 1, false, "sr1", "opt", "digestX", :profiled_destination_scales, real_subsystems, "t2")
ok_self_compare = (assert_ab_comparable(manifest_a, manifest_b) === nothing)
check("assert_ab_comparable passes for two fully-wired, matching manifests (no placeholders)", ok_self_compare)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
