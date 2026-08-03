# profiled-inner-readiness-2026-08-03, task §7: negative gate -- proves the newly-wired independent
# verification actually DISTINGUISHES "KNITRO solver status accepted" from "independently verified
# optimal" for all 4 restricted families. For each family: solve a genuine point (residual should be
# tiny), then corrupt one dual coordinate by a large amount and re-verify WITHOUT re-solving (exactly
# what a forged/corrupted "accepted" status would look like) -- the KKT residual must blow up by
# orders of magnitude, proving a caller gating on `max_abs_moment_kkt_resid`/`mean_m_resid` would
# correctly reject it rather than trusting `inner_status` alone.
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
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_checkpoint.jl",
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
          "profiled_restriction_contrib0_operators_2026-08-01.jl",
          "profiled_restricted_family_adapters_2026-08-02.jl",
          "profiled_originzc_family_adapter_2026-08-02.jl",
          "profiled_cmzc_family_adapter_2026-08-02.jl",
          "profiled_zc_lane_point_evaluators_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

"Inlined copy, same rationale as the sibling gates."
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

"Corrupts one entry of ev.result.beta by a large additive amount, re-runs the SAME family's
independent verifier on the corrupted (zeta*, beta_bad) without re-solving, and checks the
resulting max_abs_moment_kkt_resid is orders of magnitude larger than the genuine solve's own
residual -- i.e. the verifier would catch a forged/corrupted 'accepted' dual point."
function check_negative(family_name::AbstractString, genuine_resid::Float64, corrupted_resid::Float64)
    check("$family_name: genuine solve has small KKT residual (<1e-4)", genuine_resid < 1e-4)
    check("$family_name: corrupted beta produces a MUCH larger KKT residual (>=100x genuine, and >1e-2)",
        corrupted_resid > max(100 * genuine_resid, 1e-2))
    @printf("  %-14s genuine_resid=%.3e  corrupted_resid=%.3e  ratio=%.1fx\n",
        family_name, genuine_resid, corrupted_resid, corrupted_resid / max(genuine_resid, 1e-300))
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

println("="^90); println("Negative gate 1/4: flexible_CM"); println("="^90)
L = 10; contrasts = :anchored
aug_reduced = build_cm_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0, profiled_layout = layout)
cctx_cm = build_cm_bin_ctx(ctx, aug_reduced; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = false)
fctx_cm = build_flexcm_family_ctx(ctx, spec, pe, layout, cctx_cm)
ev_cm = evaluate_profiled_flexcm_point(w_profiled_calib, fctx_cm)
genuine_resid_cm = ev_cm.result.max_abs_moment_kkt_resid
bins_u_cm = cctx_cm.Bidx isa Matrix{UInt32} ? cctx_cm.Bidx : Matrix{UInt32}(cctx_cm.Bidx)
beta_bad_cm = copy(ev_cm.result.beta); beta_bad_cm[1] += 50.0
ov_bad_cm = verify_inner_solution_reduced_cm!(ev_cm.result.zeta, beta_bad_cm, ev_cm.st.cf, ctx, ev_cm.theta_full,
    layout, cctx_cm.L, length(cctx_cm.origins), cctx_cm.origins, cctx_cm.refIndex1, bins_u_cm, cctx_cm.R,
    ev_cm.obj, ev_cm.st.cf.W)
check_negative("flexible_CM", genuine_resid_cm, ov_bad_cm.kkt_resid)

println("\n" * "="^90); println("Negative gate 2/4: common_frechet"); println("="^90)
aug_reduced_f = build_cm_frechet_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0, profiled_layout = layout)
level_targets = aug_reduced_f.level_targets
cctx_f = build_cm_bin_ctx(ctx, aug_reduced_f; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = false)
fctx_f = build_frechet_family_ctx(ctx, spec, pe, layout, cctx_f, level_targets)
ev_f = evaluate_profiled_frechet_point(w_profiled_calib, fctx_f)
genuine_resid_f = ev_f.result.max_abs_moment_kkt_resid
bins_u_f = cctx_f.Bidx isa Matrix{UInt32} ? cctx_f.Bidx : Matrix{UInt32}(cctx_f.Bidx)
beta_bad_f = copy(ev_f.result.beta); beta_bad_f[1] += 50.0
ov_bad_f = verify_inner_solution_reduced_cm_frechet!(ev_f.result.zeta, beta_bad_f, ev_f.st.cf, ctx, ev_f.theta_full,
    layout, cctx_f.L, length(cctx_f.origins), cctx_f.origins, cctx_f.refIndex1, bins_u_f, cctx_f.R, level_targets,
    ev_f.obj, ev_f.st.cf.W)
check_negative("common_frechet", genuine_resid_f, ov_bad_f.kkt_resid)

println("\n" * "="^90); println("Negative gate 3/4: origin_ZC"); println("="^90)
layout_o = OriginByPowerLayout(D, 1, 0)
νvec0 = fill(1.0, D)
aug_reduced_oz = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
octx_reduced_oz = build_originzc_core_hess_ctx(aug_reduced_oz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
fctx_oz = build_originzc_family_ctx(ctx, spec, pe, layout, aug_reduced_oz)
pes_oz = OriginZCPointEvalState(octx_reduced_oz, νvec0)
ev_oz = evaluate_profiled_originzc_point(w_profiled_calib, fctx_oz, pes_oz)
genuine_resid_oz = ev_oz.result.max_abs_moment_kkt_resid
n_econ_oz = layout.total_reduced_economic_moments
cf_solved_oz = ev_oz.st.cf
op_oz = octx_reduced_oz.hzz_zc_op !== nothing ? octx_reduced_oz.hzz_zc_op : error("origin_ZC negative gate: octx.hzz_zc_op is nothing")
beta_bad_oz = copy(ev_oz.result.beta); beta_bad_oz[1] += 50.0
β_econ_bad = @view beta_bad_oz[1:n_econ_oz]
λ_mean_bad = @view beta_bad_oz[n_econ_oz+1:n_econ_oz+n_mean(op_oz)]
λ_pair_bad = @view beta_bad_oz[n_econ_oz+n_mean(op_oz)+1:n_econ_oz+n_mean(op_oz)+n_pair(op_oz)]
ov_bad_oz = verify_inner_solution_reduced_originzc!(ev_oz.result.zeta, β_econ_bad, λ_mean_bad, λ_pair_bad,
    cf_solved_oz, ctx, ev_oz.theta_full, layout, op_oz, octx_reduced_oz.hzz_zc_layout, νvec0, ev_oz.obj, cf_solved_oz.W)
check_negative("origin_ZC", genuine_resid_oz, ov_bad_oz.kkt_resid)

println("\n" * "="^90); println("Negative gate 4/4: CM+ZC (cm_meanzc)"); println("="^90)
K_MEAN, K_PAIR, L_GRID = 1, 0, 3
νvec0_cm = fill(1.0, K_MEAN)
aug_reduced_cz = build_cm_meanzc_augmented_obj(ctx, CS; L = L_GRID, K_mean = K_MEAN, K_pair = K_PAIR,
    base_obj = reduced_obj0, profiled_layout = layout)
cctx_reduced_cz = build_cm_meanzc_bin_ctx(ctx, aug_reduced_cz; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = false, inner_fg_backend = :dense_reference,
    profiled_layout = layout)
bins_u32_cz = cctx_reduced_cz.Bidx isa Matrix{UInt32} ? cctx_reduced_cz.Bidx : Matrix{UInt32}(cctx_reduced_cz.Bidx)
fctx_cz = build_cmzc_family_ctx(ctx, spec, pe, layout, aug_reduced_cz, cctx_reduced_cz, bins_u32_cz)
pes_cz = CMZCPointEvalState(cctx_reduced_cz, νvec0_cm)
ev_cz = evaluate_profiled_cmzc_point(w_profiled_calib, fctx_cz, pes_cz)
genuine_resid_cz = ev_cz.result.max_abs_moment_kkt_resid
beta_bad_cz = copy(ev_cz.result.beta); beta_bad_cz[1] += 50.0
ov_bad_cz = verify_inner_solution_reduced_cmzc!(ev_cz.result.zeta, beta_bad_cz, ev_cz.st.cf, ctx, ev_cz.theta_full,
    layout, ev_cz.st.zc_op, ev_cz.st.zc_layout, νvec0_cm, cctx_reduced_cz.L, length(cctx_reduced_cz.origins),
    cctx_reduced_cz.origins, cctx_reduced_cz.refIndex1, bins_u32_cz, cctx_reduced_cz.R, ev_cz.obj, ev_cz.st.cf.W)
check_negative("CM_plus_ZC", genuine_resid_cz, ov_bad_cz.kkt_resid)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
