# fix/profiled-functional-readiness-closeout-2026-08-03, section 3.2: the D4 REDUCED/profiled-
# layout threaded_bins gate common_frechet was genuinely missing. `test_cm_frechet_threaded_
# hessian_gates.jl` (pre-existing) only exercises the FULL/dense formulation (zero
# `profiled_layout` references, confirmed by grep this session) -- NOT the same thing the task's
# own baseline doc conflated it with. `hessian_cm_frechet_structured_v2!` (cm_frechet_hessian_
# threaded.jl) itself is a thin wrapper around the SAME `hessian_cm_structured_v2!`
# (cm_hessian_threaded.jl) flexible_CM's own gate (test_flexcm_threaded_profiled_d4_2026-08-02.jl)
# already proved correct under `profiled_layout` -- this test proves that porting holds once the
# Frechet level-block extension (`_resolve_frechet_ext!`) is layered on top, which is new,
# untested surface area (the level block's own three cross-Hessian sub-blocks).
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl"]
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

spec = build_anchor_spec_from_ctx(ctx)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
assert_no_factual_price_index_moment(layout)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
L = 10; contrasts = :anchored

aug_reduced_f = build_cm_frechet_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0, profiled_layout = layout)
obj_reduced = aug_reduced_f.obj_cm
level_targets = aug_reduced_f.level_targets

W = size(ctx.U, 1)
K_probe = Vector{Float64}(undef, W)
G_probe = Matrix{Float64}(undef, W, obj_reduced.d)
obj_reduced.moments!(K_probe, G_probe, collect(θ_full_calib), ctx.U, obj_reduced)

cctx_serial = build_cm_bin_ctx(ctx, aug_reduced_f; profiled_layout = layout, inner_fg_backend = :dense_reference,
    threaded_bins = false, core_hessian_backend = :dense_reference, cm_cross_hessian_backend = :winner_bin)
cctx_threaded = build_cm_bin_ctx(ctx, aug_reduced_f; profiled_layout = layout, inner_fg_backend = :dense_reference,
    threaded_bins = true, core_hessian_backend = :dense_reference, cm_cross_hessian_backend = :winner_bin)
@printf("cctx_serial: ncore_core=%d NCORE=%d   cctx_threaded: use_threaded_bins=%s tls===nothing? %s\n",
    cctx_serial.ncore_core, cctx_serial.NCORE, cctx_threaded.use_threaded_bins, cctx_threaded.tls === nothing)

n = obj_reduced.outer_constr_index
Random.seed!(9301)
xs = [vcat(0.0, zeros(n - 1)), 0.01 .* randn(n), 0.05 .* randn(n), -0.03 .* randn(n)]
h_len = (cctx_serial.NCORE + cctx_serial.ncm) * (cctx_serial.NCORE + cctx_serial.ncm + 1) ÷ 2
maxdiff_overall = 0.0
n_exceptions = 0
for (pi_, x) in enumerate(xs)
    _prep_dual_index_for_archC!(cctx_serial, obj_reduced, x)
    h_serial = zeros(h_len)
    try
        hessian_cm_frechet_structured!(h_serial, obj_reduced, cctx_serial, level_targets)
    catch e
        global n_exceptions += 1
        println("EXCEPTION (serial, pt$pi_): ", e)
        continue
    end

    _prep_dual_index_for_archC!(cctx_threaded, obj_reduced, x)
    h_threaded = zeros(h_len)
    try
        hessian_cm_frechet_structured_v2!(h_threaded, obj_reduced, cctx_threaded, level_targets;
            threaded_bins = true, tls = cctx_threaded.tls)
    catch e
        global n_exceptions += 1
        println("EXCEPTION (threaded, pt$pi_): ", e)
        continue
    end

    maxdiff = maximum(abs.(h_serial .- h_threaded))
    global maxdiff_overall = max(maxdiff_overall, maxdiff)
    check("pt$pi_: common_frechet threaded_bins=true matches false (max|Δ|=$(maxdiff))", maxdiff < 1e-10)
end
check("zero exceptions across all points/threading modes", n_exceptions == 0)
@printf("maxdiff_overall=%.4e  n_exceptions=%d\n", maxdiff_overall, n_exceptions)

# Also confirm the ForwardDiff-exact reduced Hessian regression (if wired for this family) still
# agrees at the SAME points under threaded_bins=true -- reuses whatever independent cross-check
# machinery this directory already has (reduced_operator_verification_2026-08-01.jl's own
# verify_inner_solution_reduced_cm_frechet! computes an independent KKT residual, not a Hessian
# FD cross-check, so it is not re-derived here -- the threaded/serial agreement above is the
# correctness gate this section needs).

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
exit(ALL_PASS[] ? 0 : 1)
