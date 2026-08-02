# Performance closeout task (2026-08-02), Section 13, REDUCED arm: real, genuinely matched outer
# search for origin-ZC using the NEW run_profiled_upper_constrained driver
# (profiled_production_outer_constrained_2026-08-02.jl), which solves the SAME constrained problem
# (minimize gp s.t. Delta<=delta) the FULL-arm production driver solves -- see that file's header
# for the full rationale.
#
# ctx is built via d20_real_setup_design with EXPLICIT kwargs matching what
# run_outer_originzc_full_2026-08-02.jl passes to run_originzc_upper_checkpointed (same W, delta,
# find_smallest, draw_design, draw_seed, destination_sample, exclude_diagonal_gravity,
# gravity_exclude_cells, sigmaHat) -- d20_real_setup_design is a deterministic function of these
# arguments (fixed draw_seed), so two independent calls with identical arguments produce identical
# ctx/data. Prints kappa/gp0 diagnostics for an empirical cross-check against the FULL run's own
# printed values (belt-and-suspenders on top of the determinism argument).
#
# nu (eta) is NOT a free variable here, matching the REDUCED path's own documented scope boundary
# (profiled_production_outer_runner_2026-08-01.jl's own header: "nu_full is fixed per evaluate_fn
# closure ... not a regression, just an unclaimed extension") -- fixed at nu=1.0 for every origin,
# matching the FULL run's own nu_bounds pinning to log(1.0)=0 for the same origins.
#
# Usage: julia run_outer_originzc_reduced_constrained_2026-08-02.jl <W> <maxtime_real_s>
const D4X = @__DIR__
t0_total = time()
# Verbatim include list from test_phase13_production_runner_d20_w20000_gate_2026-08-02.jl (the
# known-working origin-ZC/CM+ZC reduced-outer gate), plus this task's own new constrained-driver
# file appended at the end.
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
          "profiled_ab_comparability_and_plumbing_2026-08-01.jl",
          "profiled_production_outer_constrained_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Printf, Random
lp(xs...) = (println(xs...); flush(stdout))
lp("PID=", getpid(), "  include done at t=", round(time() - t0_total, digits = 1), "s")

const W_VAL = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
const MAXT = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 600.0
const DELTA = 1.0

t_ctx = @elapsed ctx = d20_real_setup_design(W = W_VAL, δ = DELTA, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0)
D = ctx.D; Ddest = ctx.D_dest
@printf("MATCHED-CTX DIAGNOSTIC: context build=%.2fs  D=%d Ddest=%d  kappa_upper=%.10f  gp0=%.10f  gp_bounds=(%.6f,%.6f)\n",
    t_ctx, D, Ddest, ctx.bounds.κ_max, ctx.θ0_up[3+D], ctx.bounds.γp_lo, ctx.bounds.γp_hi)
flush(stdout)

x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)

korea_idx = 14; brazil_idx = 3
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
z_calib = log.(reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))
gauge = build_anchor_gauge(z_calib, spec)
pe = build_pivot_elimination_on_retained(ctx, spec, gauge)

cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)

gp0 = ctx.θ0_up[3+D]
w_profiled_calib = reduce_to_w_profiled(gp0, z_calib, pe)
@printf("outer_dim_profiled=%d  total_reduced_econ=%d  w_profiled_calib[1]=%.10f\n",
    outer_dim_profiled(pe), layout.total_reduced_economic_moments, w_profiled_calib[1])
flush(stdout)

layout_o = OriginByPowerLayout(D, 1, 0)
νvec0 = fill(1.0, D)
aug_reduced_oz = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
octx_reduced_oz = build_originzc_core_hess_ctx(aug_reduced_oz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
fctx_oz = build_originzc_family_ctx(ctx, spec, pe, layout, aug_reduced_oz)
pes_oz = OriginZCPointEvalState(octx_reduced_oz, νvec0)

cache_oz = profiled_cm_production_exact_cache()

outdir = joinpath(D4X, "..", "..", "docs")
mkpath(outdir)
trace_csv = joinpath(outdir, "OUTER_ORIGINZC_REDUCED_CONSTRAINED_W$(W_VAL)_2026-08-02.csv")

lp("="^100); lp("REDUCED constrained outer search: origin-ZC  W=$(W_VAL)  maxtime_real=$(MAXT)s  delta=$(DELTA)")
lp("="^100)
result = run_profiled_upper_constrained("originzc_reduced_w$(W_VAL)", w_profiled_calib;
    fctx = fctx_oz, evaluate_fn = (w, f) -> evaluate_profiled_originzc_point(w, f, pes_oz),
    ctx = ctx, pe = pe, delta = DELTA, maxtime_real = MAXT, hessopt_tag = "sr1",
    use_screen = true, cache = cache_oz, trace_csv = trace_csv, verbose = true)

lp("="^100)
@printf("REDUCED RESULT: knitro_status=%d wall=%.1fs n_eval=%d n_grad=%d\n",
    result.knitro_status, result.wall, result.n_eval, result.n_grad)
if result.best !== nothing
    @printf("  best feasible: gp=%.10f Delta=%.10f found_at_eval=%d t=%.1fs\n",
        result.best.gp, result.best.Delta, result.best.n_eval, result.best.t_elapsed)
else
    lp("  NO feasible incumbent found")
end
lp("Wrote ", trace_csv)
@printf("TOTAL WALL: %.2fs\n", time() - t0_total)
lp("DONE")
