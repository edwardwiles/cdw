# Same direct test as hzz_originzc_direct_blas_vs_ref_2026-07-29.jl, but for cm_meanzc (CM+ZC),
# the family where the real bake-off failure was observed. Raw numbers and raw exceptions only.
const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

function unpack_packed(h::AbstractVector, n::Int)
    Hd = zeros(n, n)
    k = 1
    for i in 1:n, j in i:n
        Hd[i, j] = h[k]; Hd[j, i] = h[k]
        k += 1
    end
    return Hd
end

println("Building real D=20 context (W=100000, delta=1.0)..."); flush(stdout)
ctx = d20_real_setup(W = 100_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]
nu0vec(K::Int) = [Float64(factorial(k)) for k in 1:K]
Random.seed!(2027)

K_mean, K_pair, L = 1, 1, 50
aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = :orthonormal, meanzc_basis = :direct)
cctx = build_cm_meanzc_bin_ctx(ctx, aug; cm_cross_hessian_backend = :winner_bin, zc_cross_hessian_backend = :winner_bin)
ctx_cm = merge(ctx, (obj = aug.obj_cm,))
ν0 = nu0vec(K_mean)
NCORE = cctx.NCORE; ncm = cctx.ncm; n = NCORE + ncm
nx = n_restriction(cctx.hzz_zc_op)
println("NCORE=$NCORE ncm=$ncm n=$n nx (n_restriction)=$nx"); flush(stdout)

Random.seed!(20260719)
param_vectors = [
    ("calibration point (θ0_up)", x_free_calib),
    ("perturbed +1% on econ params", x_free_calib .* (1.0 .+ 0.01 .* randn(length(x_free_calib)))),
    ("perturbed +1% (different draw)", x_free_calib .* (1.0 .+ 0.01 .* randn(length(x_free_calib)))),
]

for (label, xf0) in param_vectors
    println("\n" * "="^80)
    println("PARAMETER VECTOR: $label")
    println("="^80); flush(stdout)

    local base
    cctx.zc_gram_backend = :reference
    try
        base = archC_meanzc_base_state(xf0, ν0, ctx_cm, cctx)
    catch e
        println("INNER SOLVE (base state, :reference backend for context build) THREW:")
        println(sprint(showerror, e))
        continue
    end
    println("inner solve status = ", base.inner_status, "  (feasible set is (0,-100,-101,-103))")
    x0v = vcat(base.ζstar, base.λstar)
    println("x0v[1:5] = ", x0v[1:5])

    local h_ref, HZZ_ref
    cctx.zc_gram_backend = :reference
    println("\n--- H_ZZ WITHOUT BLAS (:reference) ---")
    try
        _archC_prep_for_hessian!(ctx_cm.obj, x0v)
        h_ref = Vector{Float64}(undef, n * (n + 1) ÷ 2)
        hessian_cm_structured!(h_ref, ctx_cm.obj, cctx)
        Hfull_ref = unpack_packed(h_ref, n)
        HZZ_ref = Hfull_ref[cctx.ncore_core+1:NCORE, cctx.ncore_core+1:NCORE]
        println("SUCCESS. H_ZZ is $(size(HZZ_ref)). All finite: ", all(isfinite, HZZ_ref))
        println("HZZ_ref diagonal[1:5] = ", diag(HZZ_ref)[1:min(5,nx)])
    catch e
        println("THREW: ", sprint(showerror, e))
        HZZ_ref = nothing
    end

    local h_blas, HZZ_blas
    cctx.zc_gram_backend = :blas_gemm
    println("\n--- H_ZZ WITH BLAS (:blas_gemm) ---")
    try
        _archC_prep_for_hessian!(ctx_cm.obj, x0v)
        h_blas = Vector{Float64}(undef, n * (n + 1) ÷ 2)
        hessian_cm_structured!(h_blas, ctx_cm.obj, cctx)
        Hfull_blas = unpack_packed(h_blas, n)
        HZZ_blas = Hfull_blas[NCORE-nx+1:NCORE, NCORE-nx+1:NCORE]
        println("SUCCESS. H_ZZ is $(size(HZZ_blas)). All finite: ", all(isfinite, HZZ_blas))
        println("HZZ_blas diagonal[1:5] = ", diag(HZZ_blas)[1:min(5,nx)])
    catch e
        println("THREW: ", sprint(showerror, e))
        HZZ_blas = nothing
    end
    cctx.zc_gram_backend = :reference

    if HZZ_ref !== nothing && HZZ_blas !== nothing
        maxdiff = maximum(abs.(HZZ_ref .- HZZ_blas))
        relscale = max(1.0, maximum(abs.(HZZ_ref)))
        println("\nCOMPARISON: max|HZZ_ref - HZZ_blas| = $maxdiff   (scale=$relscale, relative=$(maxdiff/relscale))")
    else
        println("\nCOMPARISON: cannot compare -- at least one backend did not produce a value.")
    end
    flush(stdout)
end
println("\nDONE.")
