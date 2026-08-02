# ZC lane task (2026-08-02), Phase I: recover-then-resolve equivalence for origin-ZC. Direct
# analogue of test_unrestricted_knitro_d4_symmetric_2026-08-01.jl's STEP 5-7 (theory doc
# PROFILED_DESTINATION_SCALE_THEORY_2026-07-31.md section 2.3's actual comparison theorem), applied
# to origin-ZC's own restricted (economic + Z-restriction) reduced solve instead of the unrestricted
# family. Uses the SAME two contexts test_profiled_originzc_d4_fg_and_solve_gate_2026-08-01.jl
# already built and validated (octx_full/octx_reduced, K_mean=1/K_pair=0), plus the new
# verify_inner_solution_reduced_originzc! (reduced_originzc_verification_2026-08-02.jl) needed
# because neither pre-existing verifier covers "reduced economic width + a real ZC restriction
# block" together.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
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
          "operator_verification.jl", "recover_full_a_2026-07-31.jl", "reduced_recovery_from_lfd_2026-08-01.jl",
          "reduced_originzc_verification_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
Random.seed!(2026)
D = ctx.D
W = size(ctx.U, 1)

spec = build_anchor_spec_from_ctx(ctx)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)

layout_o = OriginByPowerLayout(ctx.D, 1, 0)   # K_mean=1, K_pair=0 -- the one safe config
νfull0 = fill(1.0, ctx.D)

aug_full = build_originzc_augmented_obj(ctx, CS, layout_o; moment_representation = :dense_reference)
octx_full = build_originzc_core_hess_ctx(aug_full, ctx; core_hessian_backend = :exact_winner_pair_parallel,
                                          zc_cross_hessian_backend = :winner_bin)
ctx_cm_full = (obj = aug_full.obj_cm, m = ctx.m, octx = octx_full)

aug_reduced = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
octx_reduced = build_originzc_core_hess_ctx(aug_reduced, ctx; core_hessian_backend = :exact_winner_pair_parallel,
                                             zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
ctx_cm_reduced = (obj = aug_reduced.obj_cm, m = ctx.m, octx = octx_reduced)

n_mean_o = n_mean(octx_reduced.hzz_zc_op); n_pair_o = n_pair(octx_reduced.hzz_zc_op)
ncolI_reduced = layout.total_reduced_economic_moments

println("="^78); println("STEP 1: REDUCED origin-ZC inner solve at calibration"); println("="^78)
base_reduced = archOZ_base_state(x_free_calib, νfull0, ctx_cm_reduced)
check("REDUCED inner solve converges to OPTIMALITY", base_reduced.inner_status == 0)
cf_reduced = octx_reduced.core_cf_ref[]
cf_reduced isa CompressedFactual || error("octx_reduced.core_cf_ref[] is not a CompressedFactual after solve")
beta_r = base_reduced.λstar[1:ncolI_reduced]
lmean_r = base_reduced.λstar[ncolI_reduced+1:ncolI_reduced+n_mean_o]
lpair_r = base_reduced.λstar[ncolI_reduced+n_mean_o+1:ncolI_reduced+n_mean_o+n_pair_o]
ov_r = verify_inner_solution_reduced_originzc!(base_reduced.ζstar, beta_r, lmean_r, lpair_r,
    cf_reduced, ctx, θ_full_calib, layout, octx_reduced.hzz_zc_op, octx_reduced.hzz_zc_layout, νfull0,
    aug_reduced.obj_cm, W)
mw_r, verify_r = verify_namedtuple_from_operator(ov_r, aug_reduced.obj_cm, W, base_reduced.inner_status)
println("verify_r = ", verify_r)
check("REDUCED verify: kkt_resid tight", verify_r.max_abs_moment_kkt_resid < 1e-6)

println("\n" * "="^78); println("STEP 2: naive same-theta comparison to the FULL/legacy solve (informational)"); println("="^78)
base_full = archOZ_base_state(x_free_calib, νfull0, ctx_cm_full)
check("FULL inner solve converges", base_full.inner_status in (0, -100, -101, -103))
cf_full = octx_full.core_cf_ref[]
ov_full = verify_inner_solution_operator_originzc!(base_full.ζstar, base_full.λstar, cf_full,
    octx_full.hzz_zc_op, octx_full.hzz_zc_layout, νfull0, aug_full.obj_cm, W)
mw_full, verify_full = verify_namedtuple_from_operator(ov_full, aug_full.obj_cm, W, base_full.inner_status)
println("verify_full (naive, same theta) = ", verify_full)
mw_diff_naive = maximum(abs.(mw_full .- mw_r))
@printf("naive max|m_weights_full - m_weights_reduced| = %.4g (NOT expected to be ~0, same reasoning as the unrestricted family's own STEP 3-4)\n", mw_diff_naive)

println("\n" * "="^78); println("STEP 3: RECOVER full-A using the REDUCED solve's own verified LFD"); println("="^78)
z_recovered, c_recover, gamma_tilde = recover_gamma_normalized_full_A_from_lfd(θ_full_calib, ctx, cf_reduced, mw_r)
println("gamma_tilde (LFD-weighted E[M_d]/denom[d]) = ", gamma_tilde)
θ_full_recovered = copy(collect(θ_full_calib))
θ_full_recovered[ctx.Aod_offset+1:ctx.Aod_offset+D^2] .= vec(exp.(z_recovered))
recovery_change = maximum(abs.(θ_full_recovered[ctx.Aod_offset+1:ctx.Aod_offset+D^2] .-
                                θ_full_calib[ctx.Aod_offset+1:ctx.Aod_offset+D^2]))
@printf("max|A_recovered - A_calib| = %.4g (0 would mean recovery was a no-op)\n", recovery_change)
x_free_recovered = θ_full_recovered[ctx.free_idx]

println("\n" * "="^78); println("STEP 4: re-solve the LEGACY FULL origin-ZC problem AT THE RECOVERED A"); println("="^78)
base_full2 = archOZ_base_state(x_free_recovered, νfull0, ctx_cm_full)
check("FULL@recovered inner solve converges", base_full2.inner_status in (0, -100, -101, -103))
cf_full2 = octx_full.core_cf_ref[]
ov_full2 = verify_inner_solution_operator_originzc!(base_full2.ζstar, base_full2.λstar, cf_full2,
    octx_full.hzz_zc_op, octx_full.hzz_zc_layout, νfull0, aug_full.obj_cm, W)
mw_full2, verify_full2 = verify_namedtuple_from_operator(ov_full2, aug_full.obj_cm, W, base_full2.inner_status)
println("verify_full2 (FULL @ recovered A) = ", verify_full2)

println("\n" * "="^78); println("STEP 5: THE DECISIVE COMPARISON (reduced's own solve vs. FULL re-solved at the recovered A)"); println("="^78)
mw_diff2 = maximum(abs.(mw_full2 .- mw_r))
delta_primal_diff = abs(verify_full2.Delta_primal - verify_r.Delta_primal)
@printf("%-32s %18.10g %18.10g %14.4g\n", "Delta_primal", verify_full2.Delta_primal, verify_r.Delta_primal, delta_primal_diff)
@printf("%-32s %18.10g %18.10g %14.4g\n", "Delta_dual", verify_full2.Delta_dual, verify_r.Delta_dual, abs(verify_full2.Delta_dual - verify_r.Delta_dual))
@printf("%-32s %18.4g\n", "max|m_weights diff|", mw_diff2)

tol_LFD = 1e-4; tol_div = 1e-4
lfd_ok = mw_diff2 < tol_LFD * max(1.0, maximum(abs.(mw_r)))
div_ok = delta_primal_diff < tol_div * max(1.0, abs(verify_r.Delta_primal))
check("recover-then-resolve: LFD match (tol $tol_LFD rel)", lfd_ok)
check("recover-then-resolve: Delta_primal match (tol $tol_div rel)", div_ok)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
