# 2026-08-05 truncated-power task: correctness gate for the two-family (eq.35+eq.36) extension to
# CMLookupState's operator FG (cm_lookup_kernels.jl) -- the matrix-free forward/backward kernels
# real production's inner KNITRO solve actually calls every iteration (not just the Hessian).
# Compares CMLookupState's (f,g) output directly against the dense reference `obj(x,g)` at several
# random points, for BOTH the :suffix (real production default) and :interval bases.
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "oracle_fast.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl"]
    include(joinpath(_D4E, f))
end
using Printf, LinearAlgebra, Random

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
    return cond
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full = CS.reconstruct_full(x_free_calib, ctx.m)
L = 4

aug = build_cm_augmented_obj(ctx, CS; L = L, include_truncated_moment = true)
@assert aug.n_families == 2
objA = aug.obj_cm
n = objA.outer_constr_index
K = zeros(size(ctx.U, 1))
objA.moments!(K, CS.select_G_from_H(objA, objA.H), θ_full, objA.U, objA)
objA.H[:, 1] .= K
objA.H[:, 2] .= 1.0

cctx = build_cm_bin_ctx(ctx, aug)
bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)

Random.seed!(3033)
xs = [vcat(0.0, zeros(n - 1)), 0.01 .* randn(n), 0.05 .* randn(n)]

for method in (:suffix, :interval)
    println("="^100)
    println("method = $method")
    println("="^100)
    st = CMLookupState(objA, cctx.NCORE, cctx.ncm, cctx.L, cctx.origins, cctx.refIndex1, bins_u, cctx.R;
                        method = method, Pow = cctx.Pow)
    for (pi_, x) in enumerate(xs)
        gA = Vector{Float64}(undef, n)
        fA = objA(x, gA)

        gC = Vector{Float64}(undef, n)
        fC = st(x, gC)

        ferr = abs(fA - fC)
        gerr = maximum(abs.(gA .- gC))
        check("method=$method point $pi_: f matches (|Δf|=$ferr)", ferr < 1e-9)
        check("method=$method point $pi_: g matches (max|Δg|=$gerr)", gerr < 1e-9)
        # Split check: the eq.35 vs eq.36 halves of g's CM slice, separately, to localize a
        # potential failure to one family specifically rather than "CM block, somewhere".
        ncore1 = cctx.NCORE - 1
        ncm_cdf = cctx.nO * cctx.L
        g_cdf_err = maximum(abs.(gA[2+ncore1:1+ncore1+ncm_cdf] .- gC[2+ncore1:1+ncore1+ncm_cdf]))
        g_pow_err = maximum(abs.(gA[2+ncore1+ncm_cdf:1+ncore1+2*ncm_cdf] .- gC[2+ncore1+ncm_cdf:1+ncore1+2*ncm_cdf]))
        check("method=$method point $pi_: g CDF-block matches (max|Δg_cdf|=$g_cdf_err)", g_cdf_err < 1e-9)
        check("method=$method point $pi_: g POW-block matches (max|Δg_pow|=$g_pow_err)", g_pow_err < 1e-9)
    end
end

println("="^80)
if isempty(FAILURES)
    println("ALL OPERATOR-FG TWO-FAMILY GATES PASS")
else
    println("FAILURES: ", FAILURES)
    exit(1)
end
