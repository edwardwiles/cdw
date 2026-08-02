# Performance closeout task (2026-08-02), Section 8 follow-up: confirm the newly-ported profiled
# branch in hessian_cm_structured_v2! (cm_hessian_threaded.jl) also works for the simpler,
# non-widened flexible-CM case (cctx.ncore_core == cctx.NCORE, no CM+ZC mean/pair block) -- this is
# the ORIGINAL profiled_layout use case (predates CM+ZC's widening), sharing the exact same code
# path this task ported. threaded_bins=true and threaded_bins=false must agree to machine precision.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl",
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

aug_reduced = build_cm_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0, profiled_layout = layout)
obj_reduced = aug_reduced.obj_cm

# core_cf_ref[] is populated as a side effect of the moments! callback (cm_meanzc_moments.jl:343's
# analogue for flexible CM), NOT by _prep_dual_index_for_archC! -- a real KNITRO solve always calls
# moments! before the first Hessian callback, but this direct-call test must do the same once,
# up front, since it never runs a solve. Shared by every cctx built from this SAME aug_reduced.
W = size(ctx.U, 1)
K_probe = Vector{Float64}(undef, W)
G_probe = Matrix{Float64}(undef, W, obj_reduced.d)
obj_reduced.moments!(K_probe, G_probe, collect(θ_full_calib), ctx.U, obj_reduced)

cctx_serial = build_cm_bin_ctx(ctx, aug_reduced; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = false)
cctx_threaded = build_cm_bin_ctx(ctx, aug_reduced; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = true)
@printf("cctx_serial: ncore_core=%d NCORE=%d (no widening expected)   cctx_threaded: use_threaded_bins=%s tls===nothing? %s\n",
    cctx_serial.ncore_core, cctx_serial.NCORE, cctx_threaded.use_threaded_bins, cctx_threaded.tls === nothing)
check("flexible-CM: ncore_core == NCORE (no CM+ZC widening -- the ORIGINAL profiled use case)", cctx_serial.ncore_core == cctx_serial.NCORE)

n = obj_reduced.outer_constr_index
Random.seed!(9300)
xs = [vcat(0.0, zeros(n - 1)), 0.01 .* randn(n), 0.05 .* randn(n)]
h_len = (cctx_serial.NCORE + cctx_serial.ncm) * (cctx_serial.NCORE + cctx_serial.ncm + 1) ÷ 2
maxdiff_overall = 0.0
for (pi_, x) in enumerate(xs)
    _prep_dual_index_for_archC!(cctx_serial, obj_reduced, x)
    h_serial = zeros(h_len)
    hessian_cm_structured!(h_serial, obj_reduced, cctx_serial)

    _prep_dual_index_for_archC!(cctx_threaded, obj_reduced, x)
    h_threaded = zeros(h_len)
    hessian_cm_structured_v2!(h_threaded, obj_reduced, cctx_threaded; threaded_bins = true, tls = cctx_threaded.tls)

    maxdiff = maximum(abs.(h_serial .- h_threaded))
    global maxdiff_overall = max(maxdiff_overall, maxdiff)
    check("pt$pi_: flexible-CM threaded_bins=true matches false (max|Δ|=$(maxdiff))", maxdiff < 1e-10)
end

println("Overall max|Δ|: ", maxdiff_overall)
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
