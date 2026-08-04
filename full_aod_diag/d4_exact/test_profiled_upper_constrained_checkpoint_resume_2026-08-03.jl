# profiled-outer-production-readiness-2026-08-03 (task 2), section 5 gate: real D4 KNITRO
# checkpoint/resume round trip for run_profiled_upper_constrained (the genuine constrained
# REDUCED runner the task names explicitly). Confirms: a short run writes a real CMCheckpointV11;
# a second call with resume_from picks up n_eval/n_grad/best_feasible cumulatively rather than
# restarting; and five independent mismatch axes (namespace/family, W, delta, draw config,
# outer-vector length) each cause a hard refusal, not a silent resume. D4 (not D20) so this runs
# in seconds, matching this directory's own "_d4_gate" convention for fast real KNITRO gates.
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

"Inlined copy, same as the phase13 D4 gate's own helper (test_phase13_production_runner_d4_gate_2026-08-02.jl)."
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

println("="^90); println("Building D4 origin-ZC fctx for run_profiled_upper_constrained checkpoint/resume gate"); println("="^90)
ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
spec, gauge, pe = build_profiled_ab_spec_pe(ctx)
θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
w_profiled_calib = reduce_calibration_to_w_profiled(ctx, pe)

layout_o = OriginByPowerLayout(D, 1, 0)
νvec0 = fill(1.0, D)
aug_reduced_oz = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
octx_reduced_oz = build_originzc_core_hess_ctx(aug_reduced_oz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
fctx_oz = build_originzc_family_ctx(ctx, spec, pe, layout, aug_reduced_oz)
pes_oz = OriginZCPointEvalState(octx_reduced_oz, νvec0)
evaluate_fn = (w, f) -> evaluate_profiled_originzc_point(w, f, pes_oz)

ckpt_dir = mktempdir()
ckpt_path = joinpath(ckpt_dir, "originzc_constrained_d4.jls")

println("\n" * "="^90); println("Part 1: short run writes a real CMCheckpointV11"); println("="^90)
result1 = run_profiled_upper_constrained("d4_ckpt_part1", w_profiled_calib; fctx = fctx_oz, evaluate_fn,
    ctx = ctx, pe = pe, delta = 1.0, maxtime_real = 2.0, hessopt_tag = "sr1",
    checkpoint_path = ckpt_path, checkpoint_interval_s = 1000.0, verbose = false)

check("part1: at least one eval ran", result1.n_eval > 0)
check("part1: checkpoint file was written", isfile(ckpt_path))
loaded1 = load_cm_checkpoint_v11(ckpt_path)
check("part1: checkpoint round-trips as CMCheckpointV11", loaded1 isa CMCheckpointV11)
check("part1: checkpoint recorded the REAL W (not the old scaffold's hardcoded 0)", loaded1.W == (hasproperty(ctx, :W) ? ctx.W : 0))
check("part1: checkpoint recorded the REAL delta (not the old scaffold's hardcoded 1.0-always)", loaded1.delta == 1.0)
check("part1: checkpoint n_eval matches result1.n_eval", loaded1.n_eval == result1.n_eval)
check("part1: checkpoint n_grad matches result1.n_grad", loaded1.n_grad == result1.n_grad)
@printf("part1: n_eval=%d n_grad=%d wall=%.2fs best=%s\n", result1.n_eval, result1.n_grad, result1.wall,
    result1.best === nothing ? "none" : @sprintf("gp=%.6f Delta=%.6f", result1.best.gp, result1.best.Delta))

println("\n" * "="^90); println("Part 2: resume_from a FRESH call, cumulative n_eval/n_grad/wall"); println("="^90)
result2 = run_profiled_upper_constrained("d4_ckpt_part2", w_profiled_calib; fctx = fctx_oz, evaluate_fn,
    ctx = ctx, pe = pe, delta = 1.0, maxtime_real = 2.0, hessopt_tag = "sr1",
    checkpoint_path = ckpt_path, checkpoint_interval_s = 1000.0, verbose = false,
    resume_from = ckpt_path)

check("part2: resumed n_eval >= part1's n_eval (cumulative, not reset)", result2.n_eval >= result1.n_eval)
check("part2: resumed n_grad >= part1's n_grad (cumulative, not reset)", result2.n_grad >= result1.n_grad)
check("part2: resumed wall >= part1's wall (cumulative, not reset)", result2.wall >= result1.wall)
if result1.best !== nothing
    # The driver MINIMIZES gp subject to Delta<=delta -- gp (not Delta) is the objective, so gp is
    # the quantity that must never regress across a resume. Delta is only the constraint value and
    # is expected to move toward the delta boundary (get numerically "worse", while staying
    # feasible) as gp improves -- confirmed live: part1 found gp=0.9610/Delta=0.0012 (lots of
    # slack), part2 (resumed) found gp=0.9006/Delta=0.9979 (near the boundary) -- gp improved as
    # expected; asserting Delta improves too was this test's own bug on the first run, not a
    # defect in the driver -- fixed here.
    check("part2: resumed best_feasible.gp is at least as good (<=) as part1's (never regresses -- gp is the actual objective)",
        result2.best !== nothing && result2.best.gp <= result1.best.gp + 1e-9)
    check("part2: resumed best_feasible remains feasible (Delta <= delta + 1e-6)",
        result2.best !== nothing && result2.best.Delta <= 1.0 + 1e-6)
end
@printf("part2: n_eval=%d n_grad=%d wall=%.2fs best=%s\n", result2.n_eval, result2.n_grad, result2.wall,
    result2.best === nothing ? "none" : @sprintf("gp=%.6f Delta=%.6f", result2.best.gp, result2.best.Delta))

println("\n" * "="^90); println("Part 3: five independent mismatch axes each refuse resume (hard error, not silent)"); println("="^90)

refuses(f) = try
    f()
    false
catch e
    e isa ErrorException
end

check("refuses resume: different family (cross-namespace via cm_meanzc fctx)", refuses() do
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
    run_profiled_upper_constrained("d4_ckpt_wrongfamily", vcat(w_profiled_calib, zeros(0)); fctx = fctx_cz,
        evaluate_fn = (w, f) -> evaluate_profiled_cmzc_point(w, f, pes_cz),
        ctx = ctx, pe = pe, delta = 1.0, maxtime_real = 1.0, resume_from = ckpt_path, verbose = false)
end)

check("refuses resume: different W", refuses() do
    ctx_wrongW = merge(ctx, (W = 999999,))
    run_profiled_upper_constrained("d4_ckpt_wrongW", w_profiled_calib; fctx = fctx_oz, evaluate_fn,
        ctx = ctx_wrongW, pe = pe, delta = 1.0, maxtime_real = 1.0, resume_from = ckpt_path, verbose = false)
end)

check("refuses resume: different delta", refuses() do
    run_profiled_upper_constrained("d4_ckpt_wrongdelta", w_profiled_calib; fctx = fctx_oz, evaluate_fn,
        ctx = ctx, pe = pe, delta = 2.5, maxtime_real = 1.0, resume_from = ckpt_path, verbose = false)
end)

check("refuses resume: different draw_design (D20-shaped ctx claiming draws where checkpoint has none)", refuses() do
    ctx_fake_draws = merge(ctx, (draw_meta = (checksum_uniform = "fake", checksum_transformed = "fake"),
        draw_design = :sobol_randomized, draw_seed = 1, W = hasproperty(ctx, :W) ? ctx.W : 0))
    run_profiled_upper_constrained("d4_ckpt_wrongdraws", w_profiled_calib; fctx = fctx_oz, evaluate_fn,
        ctx = ctx_fake_draws, pe = pe, delta = 1.0, maxtime_real = 1.0, resume_from = ckpt_path, verbose = false)
end)

check("refuses resume: different outer-vector length", refuses() do
    run_profiled_upper_constrained("d4_ckpt_wronglen", vcat(w_profiled_calib, 0.0); fctx = fctx_oz, evaluate_fn,
        ctx = ctx, pe = pe, delta = 1.0, maxtime_real = 1.0, resume_from = ckpt_path, verbose = false)
end)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
exit(ALL_PASS[] ? 0 : 1)
