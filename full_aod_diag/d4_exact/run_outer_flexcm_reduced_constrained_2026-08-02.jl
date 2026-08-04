# Performance closeout task (2026-08-02), Section 13, REDUCED arm, flexible_CM: real, genuinely
# matched outer search using run_profiled_upper_constrained (family-agnostic, already validated for
# origin-ZC) with FlexCMFamilyCtx/evaluate_profiled_flexcm_point.
#
# Switched from origin-ZC to flexible_CM per explicit user correction: origin-ZC/CM+ZC have a real
# `nu` restriction parameter that matters economically -- pinning it to compare FULL vs REDUCED (as
# the origin-ZC attempt did) tests a crippled, economically-neutered version of exactly the family
# where the restriction is the point. Flexible_CM has NO nu/eta axis at all in either formulation
# (no CM+ZC extension, no common-Frechet level block) -- both arms search the identical (gp, A)
# space with nothing held back on either side, so no pinning of any kind is needed here.
#
# ctx built via d20_real_setup_design with explicit kwargs matching
# run_outer_flexcm_full_2026-08-02.jl's own call to run_cm_upper_checkpointed.
#
# Usage: julia run_outer_flexcm_reduced_constrained_2026-08-02.jl <W> <maxtime_real_s>
const D4X = @__DIR__
t0_total = time()
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
          "profiled_reduced_frechet_lookup_kernels_2026-08-02.jl",
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
          "cm_aspace_coordinate.jl", "profiled_powered_relative_a_2026-08-04.jl",
          "profiled_coordinate_mode_dispatch_2026-08-04.jl",
          "profiled_production_outer_constrained_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Printf, Random
lp(xs...) = (println(xs...); flush(stdout))
lp("PID=", getpid(), "  include done at t=", round(time() - t0_total, digits = 1), "s")

const W_VAL = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
const MAXT = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 600.0
const DELTA = 1.0
const L_VAL = 50

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

aug_reduced_flexcm = build_cm_augmented_obj_archB(ctx, CS; L = L_VAL, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
# threaded_bins=true (profiled-inner-readiness-2026-08-03): see run_coldsolve_flexcm_w100k_2026-08-02.jl's
# own comment -- gate-proven at D4 to match serial to ~1e-14 at production's 10-thread policy.
cctx_reduced_flexcm = build_cm_bin_ctx(ctx, aug_reduced_flexcm; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = true)
fctx_flexcm = build_flexcm_family_ctx(ctx, spec, pe, layout, cctx_reduced_flexcm)

cache_flexcm = profiled_cm_production_exact_cache()

outdir = joinpath(D4X, "..", "..", "docs")
mkpath(outdir)
trace_csv = joinpath(outdir, "OUTER_FLEXCM_REDUCED_CONSTRAINED_W$(W_VAL)_2026-08-02.csv")

lp("="^100); lp("REDUCED constrained outer search: flexible_CM  W=$(W_VAL)  maxtime_real=$(MAXT)s  delta=$(DELTA)")
lp("="^100)
result = run_profiled_upper_constrained("flexcm_reduced_w$(W_VAL)", w_profiled_calib;
    fctx = fctx_flexcm, evaluate_fn = (w, f) -> evaluate_profiled_flexcm_point(w, f),
    ctx = ctx, pe = pe, delta = DELTA, maxtime_real = MAXT, hessopt_tag = "sr1",
    use_screen = true, cache = cache_flexcm, trace_csv = trace_csv, verbose = true,
    threaded_gradient = false)

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
