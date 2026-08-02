# Integration task 2026-08-02, Stage 6: the FULLY-FAITHFUL restriction_contrib0! gate the outer
# bridge's own profiled_restricted_q_decomposition_gate_2026-08-01.jl explicitly could not build
# (its own header: "a fully faithful gate would compare against
#   -zeta - reduced_homogeneous_dual_contraction(beta_reduced, ...) - restriction_contrib0
# using the family's genuinely REDUCED (anchor-excluded) economic dual. That reduced economic dual
# does not exist for any restricted family in this worktree" -- true when that branch was written,
# false now that the inner branch's real reduced contexts are merged in.
#
# This gate solves each Stage-I family's REAL REDUCED D4 dual problem (same construction as each
# family's own test_profiled_*_d4_fg_and_solve_gate_2026-08-01.jl Part 2), then at the solved point
# compares TWO independently-built restriction contributions:
#   (a) q_decomposition's own q_restriction (obtained by subtraction from the real solved obj.arg0,
#       now that q_economic's sign bug is fixed -- see profiled_restricted_accessors_2026-08-01.jl)
#   (b) restriction_contrib0_{flexcm,frechet,originzc}! -- the outer bridge's own operator-only
#       forward kernel, called against the SAME solved restriction dual (extracted via
#       restriction_dual_ranges), reconstructed via -(SW .* rc0) per that operator's own documented
#       arg0 = -zeta - econ_buf - restriction_raw convention.
# These are TWO GENUINELY DIFFERENT CODE PATHS (subtraction-from-truth vs. an independent forward
# operator) -- agreement is a real, decisive cross-check, not a tautology.
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
          "gravity_pivot_on_retained_2026-07-31.jl",
          "outer_coordinate_layout_profiled_2026-07-31.jl", "recover_full_a_2026-07-31.jl",
          "homogeneous_moments_2026-07-31.jl", "profiled_operator_bundle_2026-08-01.jl",
          "reduced_operator_verification_2026-08-01.jl", "operator_psi_bundle.jl",
          "compressed_factual_buffer_reuse.jl",
          "profiled_outer_evaluator_2026-08-01.jl",
          "profiled_restricted_accessors_2026-08-01.jl",
          "profiled_restriction_contrib0_operators_2026-08-01.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

ALL_PASS = Ref(true)
rows = NamedTuple[]
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end
function record!(family, max_abs_err, max_rel_err, pass)
    push!(rows, (family = family, max_abs_err = max_abs_err, max_rel_err = max_rel_err, pass = pass))
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
Random.seed!(2026)

spec = build_anchor_spec_from_ctx(ctx)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
assert_no_factual_price_index_moment(layout)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
L = 10; contrasts = :anchored

# =====================================================================================
# flexible CM
# =====================================================================================
println("="^90); println("flexible CM -- restriction_contrib0_flexcm! vs q_decomposition.q_restriction, REAL REDUCED D4 solve"); println("="^90)
aug_reduced = build_cm_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0, profiled_layout = layout)
obj_reduced = aug_reduced.obj_cm
cctx_reduced = build_cm_bin_ctx(ctx, aug_reduced; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = false)
ctx_cm_reduced = (obj = obj_reduced, m = ctx.m)
base_reduced = archC_base_state(x_free_calib, ctx_cm_reduced, cctx_reduced)
@printf("  inner_status=%d  zeta*=%.10f\n", base_reduced.inner_status, base_reduced.ζstar)
check("flexible CM: reduced solve reaches optimality (nStatus==0)", base_reduced.inner_status == 0)

cf = cctx_reduced.core_cf_ref[]::CompressedFactual
qd = q_decomposition(cctx_reduced, obj_reduced, cf, θ_full_calib, base_reduced.ζstar, base_reduced.λstar)
rr = restriction_dual_ranges(cctx_reduced)
# BUNDLE-TYPE FINDING (2026-08-02): restriction_dual_ranges's `rr.C` upper bound (built from
# cctx.NCORE+cctx.ncm, a bundle-type-agnostic structural quantity) is exactly 1 column too wide
# for PsiObjectiveBundleImplicit (the bundle type d4_exact_setup/build_reduced_base_obj_for_family
# produces -- confirmed: context.jl's d4_exact_setup builds CS.PsiObjectiveBundleImplicit).
# cc_algo/inner_loop_functions.jl:140 -- inner_loop_number_variables(obj::PsiObjectiveBundleImplicit)
# = obj.outer_constr_index, NOT 1+obj.outer_constr_index (the PsiObjectiveBundleExplicit convention
# restriction_dual_ranges implicitly assumes) -- so length(λstar) = outer_constr_index - 1, one
# short of what rr.C's upper bound (dual index NCORE+ncm+1, i.e. λstar index NCORE+ncm) needs. No
# existing gate exercised this: q_decomposition itself never indexes past the economic/gravity
# block (it gets q_restriction by subtraction), and no prior test indexed λstar via rr.C/rr.F. This
# is a REAL, previously-undetected off-by-one in the accessor, not a bug in q_decomposition, the FG/
# Hessian machinery, or this gate's own construction -- see the integration master report. Verified
# live: cctx_reduced.NCORE=14, ncm=30, len(λstar)=43 (not the 44 the NCORE+ncm formula implies).
# WORKAROUND for this gate only (does not fix the accessor): the restriction block is documented as
# the LAST contiguous block in dual-index order with nothing after it, so a TRAILING slice of
# λstar of width cctx.ncm is robust to exactly where the block's start boundary sits.
λ_cm = base_reduced.λstar[end-cctx_reduced.ncm+1:end]
@printf("  BUNDLE-TYPE-OFFSET CHECK: cctx.NCORE+ncm=%d  len(λstar)=%d  (off-by-%d, using trailing-%d-slice workaround)\n",
    cctx_reduced.NCORE + cctx_reduced.ncm, length(base_reduced.λstar),
    (cctx_reduced.NCORE + cctx_reduced.ncm) - length(base_reduced.λstar), cctx_reduced.ncm)
bins_u = cctx_reduced.Bidx isa Matrix{UInt32} ? cctx_reduced.Bidx : Matrix{UInt32}(cctx_reduced.Bidx)
ws_flex = FlexCMRestrictionWorkspace(length(cctx_reduced.origins), cctx_reduced.L, cf.W)
rc0 = zeros(cf.W)
restriction_contrib0_flexcm!(rc0, λ_cm, bins_u, cctx_reduced.refIndex1, cctx_reduced.origins, cctx_reduced.R, cf.SW, ws_flex)
q_restriction_recon = -(cf.SW .* rc0)
err = maximum(abs.(qd.q_restriction .- q_restriction_recon))
relerr = err / max(maximum(abs.(qd.q_restriction)), eps())
@printf("  max_abs_err=%.3e  max_rel_err=%.3e\n", err, relerr)
check("flexible CM: restriction_contrib0_flexcm! matches q_decomposition.q_restriction at REAL REDUCED context (<1e-8)", err < 1e-8)
record!("flexible_CM", err, relerr, err < 1e-8)

# =====================================================================================
# common Fréchet
# =====================================================================================
println("="^90); println("common Fréchet -- restriction_contrib0_frechet! vs q_decomposition.q_restriction, REAL REDUCED D4 solve"); println("="^90)
aug_reduced_f = build_cm_frechet_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0, profiled_layout = layout)
obj_reduced_f = aug_reduced_f.obj_cm
cctx_reduced_f = build_cm_bin_ctx(ctx, aug_reduced_f; profiled_layout = layout, inner_fg_backend = :dense_reference,
                                   threaded_bins = false, core_hessian_backend = :dense_reference,
                                   cm_cross_hessian_backend = :winner_bin)
ctx_cm_reduced_f = (obj = obj_reduced_f, m = ctx.m)
base_reduced_f = archC_frechet_base_state(x_free_calib, ctx_cm_reduced_f, cctx_reduced_f, aug_reduced_f.level_targets)
@printf("  inner_status=%d  zeta*=%.10f\n", base_reduced_f.inner_status, base_reduced_f.ζstar)
check("common Fréchet: reduced solve reaches optimality (nStatus==0)", base_reduced_f.inner_status == 0)

cf_f = cctx_reduced_f.core_cf_ref[]::CompressedFactual
qd_f = q_decomposition(cctx_reduced_f, obj_reduced_f, cf_f, θ_full_calib, base_reduced_f.ζstar, base_reduced_f.λstar)
rr_f = restriction_dual_ranges(cctx_reduced_f)
# Same trailing-slice workaround as flexible CM above (same PsiObjectiveBundleImplicit off-by-one).
# Column order [C | F] (CM-grid then level, per restriction_dual_ranges's own docstring) -> F is
# the trailing ncm_level entries, C is the ncm_cm entries immediately before that.
ncm_level_f = cctx_reduced_f.L
ncm_cm_f = cctx_reduced_f.ncm - ncm_level_f
λ_level_f = base_reduced_f.λstar[end-ncm_level_f+1:end]
λ_cm_f = base_reduced_f.λstar[end-ncm_level_f-ncm_cm_f+1:end-ncm_level_f]
bins_uf = cctx_reduced_f.Bidx isa Matrix{UInt32} ? cctx_reduced_f.Bidx : Matrix{UInt32}(cctx_reduced_f.Bidx)
ws_frechet = FrechetRestrictionWorkspace(length(cctx_reduced_f.origins), cctx_reduced_f.L, cf_f.W)
rc0_f = zeros(cf_f.W)
restriction_contrib0_frechet!(rc0_f, λ_cm_f, λ_level_f, bins_uf, cctx_reduced_f.refIndex1, cctx_reduced_f.origins,
                               cctx_reduced_f.R, cctx_reduced_f.D, aug_reduced_f.level_targets, cf_f.SW, ws_frechet)
q_restriction_recon_f = -(cf_f.SW .* rc0_f)
err_f = maximum(abs.(qd_f.q_restriction .- q_restriction_recon_f))
relerr_f = err_f / max(maximum(abs.(qd_f.q_restriction)), eps())
@printf("  max_abs_err=%.3e  max_rel_err=%.3e\n", err_f, relerr_f)
check("common Fréchet: restriction_contrib0_frechet! matches q_decomposition.q_restriction at REAL REDUCED context (<1e-8)", err_f < 1e-8)
record!("common_Frechet", err_f, relerr_f, err_f < 1e-8)

# =====================================================================================
# ZC-only (origin-ZC)
# =====================================================================================
println("="^90); println("ZC-only (origin-ZC) -- restriction_contrib0_originzc! vs q_decomposition.q_restriction, REAL REDUCED D4 solve"); println("="^90)
layout_o = OriginByPowerLayout(ctx.D, 1, 0)
νfull0 = fill(1.0, ctx.D)
aug_reduced_oz = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
obj_reduced_oz = aug_reduced_oz.obj_cm
octx_reduced = build_originzc_core_hess_ctx(aug_reduced_oz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
                                             zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
ctx_cm_reduced_oz = (obj = obj_reduced_oz, m = ctx.m, octx = octx_reduced)
base_reduced_oz = archOZ_base_state(x_free_calib, νfull0, ctx_cm_reduced_oz)
@printf("  inner_status=%d  zeta*=%.10f\n", base_reduced_oz.inner_status, base_reduced_oz.ζstar)
check("ZC-only: reduced solve reaches optimality (nStatus==0)", base_reduced_oz.inner_status == 0)

cf_oz = octx_reduced.core_cf_ref[]::CompressedFactual
qd_oz = q_decomposition(octx_reduced, obj_reduced_oz, cf_oz, θ_full_calib, base_reduced_oz.ζstar, base_reduced_oz.λstar)
rr_oz = restriction_dual_ranges(octx_reduced)
# fg_zc_op/fg_layout are only populated for fg_backend=:operator; this build uses :dense_reference,
# so use hzz_zc_op/hzz_zc_layout instead -- "Built ALWAYS (unlike fg_zc_op) whenever octx.n_eta > 0"
# per cm_hessian_architectures.jl's own comment, and the same field restriction_outer_parameter_layout
# (profiled_restricted_accessors_2026-08-01.jl) already uses for octx.
zc_op = octx_reduced.hzz_zc_op
zc_layout = octx_reduced.hzz_zc_layout
n_mean_oz = n_mean(zc_op); n_pair_oz = n_pair(zc_op)
# Same trailing-slice workaround: Z is the sole, final restriction block, width n_mean_oz+n_pair_oz.
λ_Z = base_reduced_oz.λstar[end-(n_mean_oz+n_pair_oz)+1:end]
λ_mean_oz = λ_Z[1:n_mean_oz]
λ_pair_oz = λ_Z[n_mean_oz+1:n_mean_oz+n_pair_oz]
zc_ws_oz = ZCRestrictionWorkspace(zc_op)
refresh_zc_targets!(zc_ws_oz, zc_op, zc_layout, νfull0)
rc0_oz = zeros(cf_oz.W)
restriction_contrib0_originzc!(rc0_oz, λ_mean_oz, λ_pair_oz, zc_op, zc_ws_oz, cf_oz.SW)
q_restriction_recon_oz = -(cf_oz.SW .* rc0_oz)
err_oz = maximum(abs.(qd_oz.q_restriction .- q_restriction_recon_oz))
relerr_oz = err_oz / max(maximum(abs.(qd_oz.q_restriction)), eps())
@printf("  max_abs_err=%.3e  max_rel_err=%.3e\n", err_oz, relerr_oz)
check("ZC-only: restriction_contrib0_originzc! matches q_decomposition.q_restriction at REAL REDUCED context (<1e-8)", err_oz < 1e-8)
record!("ZC_only", err_oz, relerr_oz, err_oz < 1e-8)

println()
using CSV, DataFrames
df = DataFrame(rows)
out_csv = joinpath(dirname(dirname(@__DIR__)), "PROFILED_REDUCED_RESTRICTION_CONTRIB0_VS_QDECOMPOSITION_2026-08-02.csv")
CSV.write(out_csv, df)
println(df)
println("\nWrote $out_csv")
println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
