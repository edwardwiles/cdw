# fix/profiled-functional-readiness-closeout-2026-08-03, section 5: extends
# test_profiled_upper_constrained_checkpoint_resume_2026-08-03.jl's origin_ZC-only gate to the
# remaining four families (unrestricted, flexible_CM, common_frechet, CM_plus_ZC), proving the
# SAME generic run_profiled_upper_constrained checkpoint/resume mechanism (CMCheckpointV11 +
# assert_checkpoint_compatible) round-trips for each family's own real fctx/evaluate_fn, not just
# origin_ZC's. Per family: Part 1 (short run writes a real checkpoint with real W/delta), Part 2
# (independent resumed call continues n_eval/n_grad/wall/best_feasible cumulatively), Part 3 (two
# representative mismatch axes -- cross-family and cross-W -- hard-refuse; the other three axes
# were already proven generic/family-agnostic by origin_ZC's own 5-axis gate, so are not repeated
# per family here). D4 (not D20) so the whole file runs in well under a minute of real KNITRO time.
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
          "profiled_reduced_basis_cache_bank_2026-08-02.jl",
          "profiled_screen_bridge_2026-08-02.jl",
          "dual_bank.jl", "cm_dual_bank_production.jl",
          "profiled_production_outer_runner_2026-08-01.jl",
          "profiled_ab_comparability_and_plumbing_2026-08-01.jl",
          "profiled_production_outer_constrained_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
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

refuses(f) = try
    f()
    false
catch e
    e isa ErrorException
end

"""
Runs the common Part1(write)/Part2(resume, cumulative)/Part3(family + W mismatch refuse) gate for
one family. `build_alt_fctx_evaluate` builds a DIFFERENT family's (fctx, evaluate_fn) pair, reusing
`w_profiled_calib` unchanged, purely to probe the cross-family mismatch axis (never solved to
completion, only used to trigger the refusal).
"""
function run_family_checkpoint_gate(fam_name::String, fctx, evaluate_fn::Function,
        ctx, pe, w_profiled_calib::Vector{Float64}, build_alt_fctx_evaluate::Function)
    println("\n" * "="^90); println("Family: $fam_name"); println("="^90); flush(stdout)
    ckpt_dir = mktempdir()
    ckpt_path = joinpath(ckpt_dir, "$(fam_name)_constrained_d4.jls")

    result1 = run_profiled_upper_constrained("d4_ckpt_$(fam_name)_p1", w_profiled_calib; fctx, evaluate_fn,
        ctx = ctx, pe = pe, delta = 1.0, maxtime_real = 2.0, hessopt_tag = "sr1",
        checkpoint_path = ckpt_path, checkpoint_interval_s = 1000.0, verbose = false, threaded_gradient = false)
    check("$fam_name: part1 at least one eval ran", result1.n_eval > 0)
    check("$fam_name: part1 checkpoint file was written", isfile(ckpt_path))
    loaded1 = load_cm_checkpoint_v11(ckpt_path)
    check("$fam_name: part1 checkpoint round-trips as CMCheckpointV11", loaded1 isa CMCheckpointV11)
    check("$fam_name: part1 checkpoint recorded the REAL W", loaded1.W == (hasproperty(ctx, :W) ? ctx.W : 0))
    check("$fam_name: part1 checkpoint recorded the REAL delta", loaded1.delta == 1.0)
    check("$fam_name: part1 checkpoint n_eval matches result1.n_eval", loaded1.n_eval == result1.n_eval)
    @printf("%s part1: n_eval=%d n_grad=%d wall=%.2fs best=%s\n", fam_name, result1.n_eval, result1.n_grad, result1.wall,
        result1.best === nothing ? "none" : @sprintf("gp=%.6f Delta=%.6f", result1.best.gp, result1.best.Delta))

    result2 = run_profiled_upper_constrained("d4_ckpt_$(fam_name)_p2", w_profiled_calib; fctx, evaluate_fn,
        ctx = ctx, pe = pe, delta = 1.0, maxtime_real = 2.0, hessopt_tag = "sr1",
        checkpoint_path = ckpt_path, checkpoint_interval_s = 1000.0, verbose = false,
        resume_from = ckpt_path, threaded_gradient = false)
    check("$fam_name: part2 resumed n_eval >= part1's (cumulative)", result2.n_eval >= result1.n_eval)
    check("$fam_name: part2 resumed n_grad >= part1's (cumulative)", result2.n_grad >= result1.n_grad)
    check("$fam_name: part2 resumed wall >= part1's (cumulative)", result2.wall >= result1.wall)
    if result1.best !== nothing
        check("$fam_name: part2 resumed best_feasible.gp never regresses",
            result2.best !== nothing && result2.best.gp <= result1.best.gp + 1e-9)
        check("$fam_name: part2 resumed best_feasible remains feasible",
            result2.best !== nothing && result2.best.Delta <= 1.0 + 1e-6)
    end
    @printf("%s part2: n_eval=%d n_grad=%d wall=%.2fs\n", fam_name, result2.n_eval, result2.n_grad, result2.wall)

    fctx_alt, evaluate_fn_alt = build_alt_fctx_evaluate()
    check("$fam_name: refuses resume from a different family's checkpoint", refuses() do
        run_profiled_upper_constrained("d4_ckpt_$(fam_name)_wrongfamily", w_profiled_calib; fctx = fctx_alt,
            evaluate_fn = evaluate_fn_alt, ctx = ctx, pe = pe, delta = 1.0, maxtime_real = 1.0,
            resume_from = ckpt_path, verbose = false, threaded_gradient = false)
    end)
    check("$fam_name: refuses resume with a different W", refuses() do
        ctx_wrongW = merge(ctx, (W = 999999,))
        run_profiled_upper_constrained("d4_ckpt_$(fam_name)_wrongW", w_profiled_calib; fctx, evaluate_fn,
            ctx = ctx_wrongW, pe = pe, delta = 1.0, maxtime_real = 1.0, resume_from = ckpt_path, verbose = false, threaded_gradient = false)
    end)
end

println("="^90); println("Building D4 shared context for all-family checkpoint/resume gate"); println("="^90)
ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
spec, gauge, pe = build_profiled_ab_spec_pe(ctx)
θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
w_profiled_calib = reduce_calibration_to_w_profiled(ctx, pe)
L = 10; contrasts = :anchored

# Build ONE fctx/evaluate_fn per family, reusing the SAME real adapters other gates in this
# directory already use (profiled_restricted_family_adapters_2026-08-02.jl,
# profiled_originzc_family_adapter_2026-08-02.jl / profiled_cmzc_family_adapter_2026-08-02.jl,
# profiled_family_adapters_2026-08-01.jl for unrestricted).

# --- unrestricted ---
ev0_u = evaluate_profiled_point(w_profiled_calib, ctx, spec, pe)
fctx_u = build_unrestricted_family_ctx(ctx, spec, pe, ev0_u)
evaluate_fn_u = (w, f) -> evaluate_profiled_point(w, f.ctx, f.spec, f.pe)

# --- flexible_CM ---
aug_reduced_cm = build_cm_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0, profiled_layout = layout)
cctx_cm = build_cm_bin_ctx(ctx, aug_reduced_cm; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = false)
fctx_cm = build_flexcm_family_ctx(ctx, spec, pe, layout, cctx_cm)
evaluate_fn_cm = (w, f) -> evaluate_profiled_flexcm_point(w, f)

# --- common_frechet ---
aug_reduced_f = build_cm_frechet_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0, profiled_layout = layout)
level_targets = aug_reduced_f.level_targets
cctx_f = build_cm_bin_ctx(ctx, aug_reduced_f; profiled_layout = layout, inner_fg_backend = :dense_reference,
                           threaded_bins = false, core_hessian_backend = :dense_reference, cm_cross_hessian_backend = :winner_bin)
fctx_f = build_frechet_family_ctx(ctx, spec, pe, layout, cctx_f, level_targets)
evaluate_fn_f = (w, f) -> evaluate_profiled_frechet_point(w, f)

# --- origin_ZC (built for use as the cross-family mismatch probe below; already gated on its own
# by test_profiled_upper_constrained_checkpoint_resume_2026-08-03.jl) ---
layout_o = OriginByPowerLayout(D, 1, 0)
νvec0 = fill(1.0, D)
aug_reduced_oz = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
octx_reduced_oz = build_originzc_core_hess_ctx(aug_reduced_oz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
fctx_oz = build_originzc_family_ctx(ctx, spec, pe, layout, aug_reduced_oz)
pes_oz = OriginZCPointEvalState(octx_reduced_oz, νvec0)
evaluate_fn_oz = (w, f) -> evaluate_profiled_originzc_point(w, f, pes_oz)

# --- CM_plus_ZC (cm_meanzc) ---
K_MEAN, K_PAIR, L_GRID = 1, 0, 3
νvec0_cm = fill(1.0, K_MEAN)
aug_reduced_cz = build_cm_meanzc_augmented_obj(ctx, CS; L = L_GRID, K_mean = K_MEAN, K_pair = K_PAIR,
    base_obj = reduced_obj0, profiled_layout = layout)
cctx_reduced_cz = build_cm_meanzc_bin_ctx(ctx, aug_reduced_cz; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = false, inner_fg_backend = :dense_reference,
    profiled_layout = layout)
bins_u32 = cctx_reduced_cz.Bidx isa Matrix{UInt32} ? cctx_reduced_cz.Bidx : Matrix{UInt32}(cctx_reduced_cz.Bidx)
fctx_cz = build_cmzc_family_ctx(ctx, spec, pe, layout, aug_reduced_cz, cctx_reduced_cz, bins_u32)
pes_cz = CMZCPointEvalState(cctx_reduced_cz, νvec0_cm)
evaluate_fn_cz = (w, f) -> evaluate_profiled_cmzc_point(w, f, pes_cz)

run_family_checkpoint_gate("unrestricted", fctx_u, evaluate_fn_u, ctx, pe, w_profiled_calib,
    () -> (fctx_cm, evaluate_fn_cm))
run_family_checkpoint_gate("flexible_CM", fctx_cm, evaluate_fn_cm, ctx, pe, w_profiled_calib,
    () -> (fctx_f, evaluate_fn_f))
run_family_checkpoint_gate("common_frechet", fctx_f, evaluate_fn_f, ctx, pe, w_profiled_calib,
    () -> (fctx_cm, evaluate_fn_cm))
run_family_checkpoint_gate("CM_plus_ZC", fctx_cz, evaluate_fn_cz, ctx, pe, w_profiled_calib,
    () -> (fctx_oz, evaluate_fn_oz))

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
exit(ALL_PASS[] ? 0 : 1)
