# ZC lane task (2026-08-02), Phase I: real D20 omit-ROW recover-then-resolve equivalence for
# origin-ZC. Direct D20 analogue of test_zc_lane_originzc_recover_resolve_2026-08-02.jl (which
# validated this decisively at D4), run at real production scale (destination_sample=:exclude_row,
# D=20/Ddest=19). W is read from ENV["ZC_LANE_D20_W"] (defaults to 20000) so the same script drives
# both required points (W=10-20k and W=80k) without duplication.
const D4X = @__DIR__
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
          "operator_verification.jl", "recover_full_a_2026-07-31.jl", "reduced_recovery_from_lfd_2026-08-01.jl",
          "reduced_originzc_verification_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random
flush(stdout)

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
    flush(stdout)
end

const W_VAL = parse(Int, get(ENV, "ZC_LANE_D20_W", "20000"))
println("="^90); println("Building real D=20 :exclude_row context at W=$W_VAL ..."); flush(stdout)
ctx = d20_real_setup(W = W_VAL, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
D = ctx.D; Ddest = ctx.D_dest
println("D=$D  Ddest=$Ddest  bi=$(ctx.bi)")
@assert D == 20 && Ddest == 19 "expected live D=20, Ddest=19 -- got D=$D, Ddest=$Ddest"
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
Random.seed!(2026)
W = size(ctx.U, 1)

korea_idx = 14; brazil_idx = 3
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
println("LIVE DIMENSIONS: active_A=$(D*Ddest)  retained_factual=$(length(layout.retained_full_factual_j))  france_ratio_present=$has_france  total_reduced=$(layout.total_reduced_economic_moments)")
flush(stdout)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)

layout_o = OriginByPowerLayout(ctx.D, 1, 0)   # K_mean=1, K_pair=0 -- the one config confirmed safe against archOZ_base_state's K>1 crash
νfull0 = fill(1.0, ctx.D)

t_build_full = @elapsed begin
    aug_full = build_originzc_augmented_obj(ctx, CS, layout_o; moment_representation = :dense_reference)
    octx_full = build_originzc_core_hess_ctx(aug_full, ctx; core_hessian_backend = :exact_winner_pair_parallel,
                                              zc_cross_hessian_backend = :winner_bin)
end
ctx_cm_full = (obj = aug_full.obj_cm, m = ctx.m, octx = octx_full)
println("built FULL octx in $(t_build_full)s"); flush(stdout)

t_build_reduced = @elapsed begin
    aug_reduced = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
    octx_reduced = build_originzc_core_hess_ctx(aug_reduced, ctx; core_hessian_backend = :exact_winner_pair_parallel,
                                                 zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
end
ctx_cm_reduced = (obj = aug_reduced.obj_cm, m = ctx.m, octx = octx_reduced)
println("built REDUCED octx in $(t_build_reduced)s"); flush(stdout)

println("ZC_EZ_BACKEND_DEFAULT[]=", ZC_EZ_BACKEND_DEFAULT[], "  ZC_GRAM_BACKEND_DEFAULT[]=", ZC_GRAM_BACKEND_DEFAULT[])
n_mean_o = n_mean(octx_reduced.hzz_zc_op); n_pair_o = n_pair(octx_reduced.hzz_zc_op)
ncolI_reduced = layout.total_reduced_economic_moments

println("\n" * "="^90); println("STEP 1: REDUCED origin-ZC inner solve at calibration (D20, W=$W_VAL)"); println("="^90); flush(stdout)
reset_no_dense_g_counters!()
t_reduced = @elapsed base_reduced = archOZ_base_state(x_free_calib, νfull0, ctx_cm_reduced)
check("REDUCED inner solve converges to OPTIMALITY", base_reduced.inner_status == 0)
@printf("REDUCED: time=%.2fs  nStatus=%d  zeta*=%.10f\n", t_reduced, base_reduced.inner_status, base_reduced.ζstar)
flush(stdout)
c_disp = NO_DENSE_G_COUNTERS[]
check("dispatch proof: winner-based cross-Hessian path fired, zero dense fallback",
    c_disp.winner_cross_hessian_calls > 0 && c_disp.dense_cross_hessian_calls == 0)
@printf("winner_cross_hessian_calls=%d  dense_cross_hessian_calls=%d\n", c_disp.winner_cross_hessian_calls, c_disp.dense_cross_hessian_calls)

cf_reduced = octx_reduced.core_cf_ref[]
beta_r = base_reduced.λstar[1:ncolI_reduced]
lmean_r = base_reduced.λstar[ncolI_reduced+1:ncolI_reduced+n_mean_o]
lpair_r = base_reduced.λstar[ncolI_reduced+n_mean_o+1:ncolI_reduced+n_mean_o+n_pair_o]
ov_r = verify_inner_solution_reduced_originzc!(base_reduced.ζstar, beta_r, lmean_r, lpair_r,
    cf_reduced, ctx, θ_full_calib, layout, octx_reduced.hzz_zc_op, octx_reduced.hzz_zc_layout, νfull0,
    aug_reduced.obj_cm, W)
mw_r, verify_r = verify_namedtuple_from_operator(ov_r, aug_reduced.obj_cm, W, base_reduced.inner_status)
println("verify_r = ", verify_r)
check("REDUCED verify: kkt_resid tight", verify_r.max_abs_moment_kkt_resid < 1e-4)
flush(stdout)

println("\n" * "="^90); println("STEP 2: RECOVER full-A using the REDUCED solve's own verified LFD"); println("="^90); flush(stdout)
z_recovered, c_recover, gamma_tilde = recover_gamma_normalized_full_A_from_lfd(θ_full_calib, ctx, cf_reduced, mw_r)
θ_full_recovered = copy(collect(θ_full_calib))
θ_full_recovered[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest] .= vec(exp.(z_recovered))
recovery_change_log = maximum(abs.(z_recovered .- log.(reshape(θ_full_calib[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))))
@printf("max|log A_recovered - log A_calib| = %.4g (0 would mean recovery was a no-op; LOG-space per CLAUDE.md's own real-D20-scale warning)\n", recovery_change_log)
x_free_recovered = θ_full_recovered[ctx.free_idx]
flush(stdout)

println("\n" * "="^90); println("STEP 3: re-solve the LEGACY FULL origin-ZC problem AT THE RECOVERED A"); println("="^90); flush(stdout)
t_full2 = @elapsed base_full2 = archOZ_base_state(x_free_recovered, νfull0, ctx_cm_full)
check("FULL@recovered inner solve converges", base_full2.inner_status in (0, -100, -101, -103))
@printf("FULL@recovered: time=%.2fs  nStatus=%d\n", t_full2, base_full2.inner_status)
flush(stdout)
cf_full2 = octx_full.core_cf_ref[]
ov_full2 = verify_inner_solution_operator_originzc!(base_full2.ζstar, base_full2.λstar, cf_full2,
    octx_full.hzz_zc_op, octx_full.hzz_zc_layout, νfull0, aug_full.obj_cm, W)
mw_full2, verify_full2 = verify_namedtuple_from_operator(ov_full2, aug_full.obj_cm, W, base_full2.inner_status)
println("verify_full2 (FULL @ recovered A) = ", verify_full2)
flush(stdout)

println("\n" * "="^90); println("STEP 4: THE DECISIVE COMPARISON (D=20, W=$W_VAL)"); println("="^90)
mw_diff2 = maximum(abs.(mw_full2 .- mw_r))
delta_primal_diff = abs(verify_full2.Delta_primal - verify_r.Delta_primal)
@printf("%-32s %18.10g %18.10g %14.4g\n", "Delta_primal", verify_full2.Delta_primal, verify_r.Delta_primal, delta_primal_diff)
@printf("%-32s %18.10g %18.10g %14.4g\n", "Delta_dual", verify_full2.Delta_dual, verify_r.Delta_dual, abs(verify_full2.Delta_dual - verify_r.Delta_dual))
@printf("%-32s %18.4g\n", "max|m_weights diff|", mw_diff2)
flush(stdout)

tol_LFD = 1e-3; tol_div = 1e-3   # slightly looser than D4's 1e-4 -- larger W/scale, real-data noise floor
lfd_ok = mw_diff2 < tol_LFD * max(1.0, maximum(abs.(mw_r)))
div_ok = delta_primal_diff < tol_div * max(1.0, abs(verify_r.Delta_primal))
check("recover-then-resolve: LFD match (tol $tol_LFD rel)", lfd_ok)
check("recover-then-resolve: Delta_primal match (tol $tol_div rel)", div_ok)

println()
println("W=$W_VAL  ", ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
