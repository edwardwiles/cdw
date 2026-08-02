# Restricted-inner-endtoend task (2026-08-01), addendum: D4 verification gate for the outer-gradient
# workstream's requested accessor surface (profiled_restricted_accessors_2026-08-01.jl). Checks:
#   1. profiled_economic_layout/profiled_anchor_spec/profiled_outer_coordinate_layout run and are
#      internally consistent (same anchor map underlying both layers).
#   2. economic_dual_range + restriction_dual_ranges partition the FULL reduced dual vector exactly
#      (verify_dual_ranges_partition), for flexible CM, common Frechet, and ZC-only -- the three
#      families this session actually built reduced contexts for.
#   3. CM+ZC is confirmed to error loudly (not silently mis-partition), matching this session's own
#      documented deferral (CMZC_WIDENED_CORE_FINDING_2026-08-01.md).
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
          "profiled_restricted_accessors_2026-08-01.jl"]
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

# ---- 1. Shared layout objects run and are consistent ----
spec_a = profiled_anchor_spec(ctx)
layout = profiled_economic_layout(ctx)
spec_b, gauge, pe = profiled_outer_coordinate_layout(ctx)
check("profiled_economic_layout runs", layout isa ProfiledEconomicMomentLayout)
check("profiled_anchor_spec / profiled_outer_coordinate_layout share the same anchor map",
    spec_a.anchor_origin == spec_b.anchor_origin)
check("profiled_economic_layout's own anchor_origin_by_slot matches the shared spec",
    layout.anchor_origin_by_slot == spec_a.anchor_origin)

L = 10; contrasts = :anchored
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)

# ---- 2a. Flexible CM ----
aug_cm = build_cm_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0, profiled_layout = layout)
cctx_cm = build_cm_bin_ctx(ctx, aug_cm; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = false)
er_cm = economic_dual_range(cctx_cm)
rr_cm = restriction_dual_ranges(cctx_cm)
check("flexible CM: economic_dual_range == 2:(2+n_econ)", er_cm == 2:(2 + layout.total_reduced_economic_moments))
check("flexible CM: restriction_dual_ranges has just :C", keys(rr_cm) == (:C,))
check("flexible CM: dual ranges partition the full reduced dual vector",
    verify_dual_ranges_partition(cctx_cm, 1 + aug_cm.obj_cm.outer_constr_index))
@printf("  flexible CM: economic=%s  restriction=%s  total_dual_dim=%d\n", er_cm, rr_cm, 1 + aug_cm.obj_cm.outer_constr_index)

# ---- 2b. Common Frechet ----
aug_fr = build_cm_frechet_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0, profiled_layout = layout)
cctx_fr = build_cm_bin_ctx(ctx, aug_fr; profiled_layout = layout, inner_fg_backend = :dense_reference,
                            threaded_bins = false, core_hessian_backend = :dense_reference, cm_cross_hessian_backend = :winner_bin)
_resolve_frechet_ext!(cctx_fr, aug_fr.level_targets)   # populate frechet_ext_cache so restriction_dual_ranges can dispatch on it
er_fr = economic_dual_range(cctx_fr)
rr_fr = restriction_dual_ranges(cctx_fr)
check("common Frechet: economic_dual_range == 2:(2+n_econ)", er_fr == 2:(2 + layout.total_reduced_economic_moments))
check("common Frechet: restriction_dual_ranges has :C then :F", keys(rr_fr) == (:C, :F))
check("common Frechet: :F range has width L", length(rr_fr.F) == L)
check("common Frechet: dual ranges partition the full reduced dual vector",
    verify_dual_ranges_partition(cctx_fr, 1 + aug_fr.obj_cm.outer_constr_index))
@printf("  common Frechet: economic=%s  restriction=%s  total_dual_dim=%d\n", er_fr, rr_fr, 1 + aug_fr.obj_cm.outer_constr_index)

# ---- 2c. ZC-only (origin-ZC) ----
layout_o = OriginByPowerLayout(ctx.D, 1, 0)
aug_oz = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
octx_oz = build_originzc_core_hess_ctx(aug_oz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
                                        zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
er_oz = economic_dual_range(octx_oz)
rr_oz = restriction_dual_ranges(octx_oz)
check("ZC-only: economic_dual_range == 2:(2+n_econ)", er_oz == 2:(2 + layout.total_reduced_economic_moments))
check("ZC-only: restriction_dual_ranges has just :Z", keys(rr_oz) == (:Z,))
check("ZC-only: dual ranges partition the full reduced dual vector",
    verify_dual_ranges_partition(octx_oz, 1 + aug_oz.obj_cm.outer_constr_index))
@printf("  ZC-only: economic=%s  restriction=%s  total_dual_dim=%d\n", er_oz, rr_oz, 1 + aug_oz.obj_cm.outer_constr_index)

# ---- 3. CM+ZC errors loudly, does not silently mis-partition ----
layout_meanzc = OriginByPowerLayout(ctx.D, 1, 0)
aug_zc = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = 1, K_pair = 0, contrasts = contrasts)
cctx_zc = build_cm_meanzc_bin_ctx(ctx, aug_zc)
threw = false
try
    economic_dual_range(cctx_zc)
catch e
    global threw = e isa ErrorException
end
check("CM+ZC (widened-core): economic_dual_range errors loudly rather than silently mis-partitioning", threw)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
