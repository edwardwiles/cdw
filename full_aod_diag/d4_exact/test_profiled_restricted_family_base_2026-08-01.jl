# Claude Code task 2026-08-01 (all-families completion): construction-only D4 dimension gate for
# `build_reduced_base_obj_for_family` + the additive `base_obj` keyword on
# `build_cm_augmented_obj_archB`. Verifies the reduced economic-block width propagates correctly
# through to `obj_cm.d`/`outer_constr_index`, `cctx.NCORE`, and the CM-grid column offset
# (`pregrav`/`cm_cols` start) -- does NOT run a KNITRO solve or check any Hessian numerics (that is
# explicitly out of scope for this session, see PROFILED_ALL_FAMILY_COMPLETION_MASTER_2026-08-01.md).
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "threaded_cross_hessian.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)

spec = build_anchor_spec_from_ctx(ctx)
cf_full = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_full.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
assert_no_factual_price_index_moment(layout)

check("layout: total_reduced_economic_moments = D*Ddest - Ddest + (has_france?1:0)",
    layout.total_reduced_economic_moments == (cf_full.D * cf_full.D_dest - cf_full.D_dest) + (has_france ? 1 : 0))

reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
n_expected = 1 + layout.total_reduced_economic_moments
check("reduced_obj0.d == 1+total_reduced_economic_moments", reduced_obj0.d == n_expected)
check("reduced_obj0.outer_constr_index == reduced_obj0.d", reduced_obj0.outer_constr_index == reduced_obj0.d)
check("reduced_obj0.d < ctx.obj.d (genuine reduction)", reduced_obj0.d < ctx.obj.d)

for contrasts in (:anchored, :orthonormal), L in (10, 20)
    aug_full = build_cm_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts)
    aug_reduced = build_cm_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0)

    check("contrasts=$contrasts L=$L: aug_full.ncore == ctx.obj.d (baseline unchanged)", aug_full.ncore == ctx.obj.d)
    check("contrasts=$contrasts L=$L: aug_reduced.ncore == n_expected", aug_reduced.ncore == n_expected)
    check("contrasts=$contrasts L=$L: aug_reduced.ncm == aug_full.ncm (restriction block untouched)",
        aug_reduced.ncm == aug_full.ncm)
    check("contrasts=$contrasts L=$L: aug_reduced.obj_cm.d == n_expected+ncm",
        aug_reduced.obj_cm.d == n_expected + aug_reduced.ncm)
    check("contrasts=$contrasts L=$L: aug_reduced.obj_cm.d < aug_full.obj_cm.d (genuine dual-dim reduction)",
        aug_reduced.obj_cm.d < aug_full.obj_cm.d)
    check("contrasts=$contrasts L=$L: aug_reduced.obj_cm.outer_constr_index == aug_reduced.obj_cm.d",
        aug_reduced.obj_cm.outer_constr_index == aug_reduced.obj_cm.d)

    cctx_full = build_cm_bin_ctx(ctx, aug_full)
    cctx_reduced = build_cm_bin_ctx(ctx, aug_reduced)
    check("contrasts=$contrasts L=$L: cctx_reduced.NCORE == n_expected", cctx_reduced.NCORE == n_expected)
    check("contrasts=$contrasts L=$L: cctx_reduced.Hfull size shrinks vs full",
        size(cctx_reduced.Hfull, 1) < size(cctx_full.Hfull, 1))
    check("contrasts=$contrasts L=$L: cctx_reduced.ncm == cctx_full.ncm (CM-grid block itself untouched)",
        cctx_reduced.ncm == cctx_full.ncm)
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
