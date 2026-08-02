# Phase 9/production-dims gate (integration/profiled-all-five-production-closeout, 2026-08-02):
# GENUINE-COLD origin-ZC solve at PRODUCTION dimensions (D=20, Ddest=19, K_mean=3, K_pair=3,
# W=100,000 -- NOT the smaller K_mean=1/K_pair=0 config prior sessions used for tractability), via
# the zero-dense reduced/operator inner-solve path (`reduced_originzc_base_state`). Must be launched
# as a FRESH `julia` process -- see run_coldsolve_flexcm_w100k_2026-08-02.jl's own header for why.
# Origin-ZC has NO CM-grid/L component at all (pure mean/pair-Z restriction, no bilateral bins), so
# there is no `L` parameter here -- confirmed by direct read of build_originzc_augmented_obj.
const D4X = @__DIR__
t0_total = time()
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl",
          "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_checkpoint.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl",
          "operator_verification.jl",
          "profiled_reduced_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_originzc_lookup_kernels_2026-08-02.jl",
          "reduced_originzc_verification_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random
flush(stdout)
println("PID=", getpid(), "  include done at t=", round(time() - t0_total, digits = 1), "s"); flush(stdout)

const W_VAL = 100_000
const D_VAL, DDEST_VAL, K_MEAN, K_PAIR = 20, 19, 3, 3

println("="^90); println("GENUINE-COLD origin-ZC solve, PRODUCTION DIMS: D=$D_VAL Ddest=$DDEST_VAL K_mean=$K_MEAN K_pair=$K_PAIR W=$W_VAL")
println("="^90); flush(stdout)

t_ctx = @elapsed ctx = d20_real_setup(W = W_VAL, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
@printf("context build: %.2fs\n", t_ctx); flush(stdout)
D = ctx.D; Ddest = ctx.D_dest
@assert D == D_VAL && Ddest == DDEST_VAL "expected D=$D_VAL/Ddest=$DDEST_VAL, got D=$D/Ddest=$Ddest"
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
W = size(ctx.U, 1)

korea_idx = 14; brazil_idx = 3
t_layout = @elapsed begin
    spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
    cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
    has_france = cf_probe.cf_col > 0
    layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
    reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
end
@printf("layout build: %.2fs  total_reduced_econ=%d  france=%s\n", t_layout, layout.total_reduced_economic_moments, has_france)
flush(stdout)

layout_o = OriginByPowerLayout(D, K_MEAN, K_PAIR)

# BUG FIX (found live this session): OriginByPowerLayout's target_index(o,k) = (k-1)*D + o requires
# νfull to have length K_MEAN*D (one value per origin PER level k=1:K_MEAN), not just D -- confirmed
# by test_cm_originzc_checkpoint.jl's own `eta0 = randn(K_mean * D)` construction. The K_mean=1
# D20 gates (this repo's existing test_zc_lane_originzc_recover_resolve_d20_2026-08-02.jl) never
# exposed this because at K_mean=1, K_MEAN*D == D coincidentally. At this task's mandated K_mean=3,
# passing length-D caused a BoundsError (mean_targets indexing νfull[(k-1)*D+o] for k=2,3 reads past
# the end of a length-D vector) inside refresh_zc_targets! -> reduced_originzc_base_state.
νfull0 = fill(1.0, K_MEAN * D)

t_build = @elapsed begin
    aug_reduced = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
    octx_reduced = build_originzc_core_hess_ctx(aug_reduced, ctx; core_hessian_backend = :exact_winner_pair_parallel,
                                                 zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
end
n_mean_o = n_mean(octx_reduced.hzz_zc_op); n_pair_o = n_pair(octx_reduced.hzz_zc_op)
@printf("reduced object/octx build: %.2fs  n_mean=%d  n_pair=%d\n", t_build, n_mean_o, n_pair_o)
flush(stdout)

reset_no_dense_g_counters!()
println("Starting COLD KNITRO inner solve ..."); flush(stdout)
t_solve = @elapsed base = reduced_originzc_base_state(x_free_calib, ctx, layout, octx_reduced, νfull0)
@printf("SOLVE: wall=%.2fs  nStatus=%d  zeta*=%.10f  n_fg=%d  n_hess=%d\n",
    t_solve, base.inner_status, base.ζstar, base.n_fg, base.n_hess)
flush(stdout)

c_disp = NO_DENSE_G_COUNTERS[]
@printf("dense_economic_G_materializations=%d (0 expected)\n", c_disp.dense_economic_G_materializations)
flush(stdout)

ncolI_reduced = layout.total_reduced_economic_moments
cf_reduced = octx_reduced.core_cf_ref[]
beta_r = base.λstar[1:ncolI_reduced]
lmean_r = base.λstar[ncolI_reduced+1:ncolI_reduced+n_mean_o]
lpair_r = base.λstar[ncolI_reduced+n_mean_o+1:ncolI_reduced+n_mean_o+n_pair_o]
t_verify = @elapsed begin
    ov = verify_inner_solution_reduced_originzc!(base.ζstar, beta_r, lmean_r, lpair_r,
        cf_reduced, ctx, θ_full_calib, layout, octx_reduced.hzz_zc_op, octx_reduced.hzz_zc_layout, νfull0,
        aug_reduced.obj_cm, W)
    mw, verify = verify_namedtuple_from_operator(ov, aug_reduced.obj_cm, W, base.inner_status)
end
@printf("VERIFY: wall=%.2fs  %s\n", t_verify, verify)
flush(stdout)

ok = base.inner_status == 0 && verify.max_abs_moment_kkt_resid < 1e-4 &&
     c_disp.dense_economic_G_materializations == 0
@printf("\nTOTAL WALL: %.2fs\n", time() - t0_total)
println("ORIGINZC_W100K_COLD_SOLVE_RESULT: ", ok ? "PASS" : "FAIL")
ok || exit(1)
