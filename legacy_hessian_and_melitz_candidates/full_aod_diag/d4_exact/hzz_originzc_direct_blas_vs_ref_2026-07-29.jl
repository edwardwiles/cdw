# Direct answer to: run the origin-ZC (ZC-only) inner loop at a few real parameter vectors, with
# and without BLAS for the H_ZZ (HRR) block, and show exactly what happens -- values, errors,
# stack traces. No summary statistics, no inference -- raw numbers and raw exceptions only.
const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_originzc_moments.jl",
          "cm_originzc_lookup_production.jl", "cm_originzc_production.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

"Invoke archA_partitioned_hess_cb_builder(octx)'s closure directly with mock KNITRO evalRequest/evalResult."
function packed_hess_via_octx(octx, obj, x::AbstractVector{Float64}, n::Int)
    cb = archA_partitioned_hess_cb_builder(octx)
    h = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    evalRequest = (x = x,)
    evalResult = (hess = h,)
    cb(nothing, nothing, evalRequest, evalResult, obj)
    return h
end

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

K_mean, K_pair = 1, 1
ν0 = nu0vec(K_mean)
layout = SharedByPowerLayout(K_mean, K_pair)
aug = build_originzc_augmented_obj(ctx, CS, layout)
ctx_cm = merge(ctx, (obj = aug.obj_cm,))
octx = build_originzc_core_hess_ctx(aug; zc_cross_hessian_backend = :winner_bin)
ctx_cm = merge(ctx_cm, (octx = octx,))
println("hzz_zc_op built: ", octx.hzz_zc_op !== nothing); flush(stdout)
NCORE = octx.NCORE; n_eta_dim = octx.n_eta; n = NCORE + n_eta_dim
nx = n_restriction(octx.hzz_zc_op)
println("NCORE=$NCORE n_eta_dim=$n_eta_dim n=$n nx (n_restriction)=$nx"); flush(stdout)

# --- a few parameter vectors: the real calibration point, plus two small perturbations of it ---
Random.seed!(20260719)
param_vectors = [
    ("calibration point (θ0_up)", x_free_calib, ν0),
    ("perturbed +1% on econ params", x_free_calib .* (1.0 .+ 0.01 .* randn(length(x_free_calib))), ν0),
    ("perturbed +1% (different draw)", x_free_calib .* (1.0 .+ 0.01 .* randn(length(x_free_calib))), ν0),
]

for (label, xf0, nu0) in param_vectors
    println("\n" * "="^80)
    println("PARAMETER VECTOR: $label")
    println("="^80); flush(stdout)

    local base
    try
        base = archOZ_base_state(xf0, nu0, ctx_cm)
    catch e
        println("INNER SOLVE (base state) THREW: ", sprint(showerror, e))
        continue
    end
    println("inner solve status = ", base.inner_status, "  (feasible set is (0,-100,-101,-103))")
    x0v = vcat(base.ζstar, base.λstar)
    println("x0v[1:5] = ", x0v[1:5])

    # ---- WITHOUT BLAS: octx.zc_gram_backend = :reference ----
    octx.zc_gram_backend = :reference
    println("\n--- H_ZZ (HRR) WITHOUT BLAS (:reference) ---")
    local h_ref, HRR_ref
    try
        h_ref = packed_hess_via_octx(octx, ctx_cm.obj, x0v, n)
        Hfull_ref = unpack_packed(h_ref, n)
        HRR_ref = Hfull_ref[NCORE+1:NCORE+nx, NCORE+1:NCORE+nx]
        println("SUCCESS. H_ZZ is $(size(HRR_ref)). All finite: ", all(isfinite, HRR_ref))
        println("HRR_ref[1:3,1:3] = ")
        display(HRR_ref[1:min(3,nx), 1:min(3,nx)])
        println()
        println("HRR_ref diagonal[1:5] = ", diag(HRR_ref)[1:min(5,nx)])
    catch e
        println("THREW: ", sprint(showerror, e))
        for (i, frame) in enumerate(stacktrace(catch_backtrace()))
            i > 15 && break
            println("  [$i] ", frame)
        end
        HRR_ref = nothing
    end

    # ---- WITH BLAS: octx.zc_gram_backend = :blas_gemm ----
    octx.zc_gram_backend = :blas_gemm
    println("\n--- H_ZZ (HRR) WITH BLAS (:blas_gemm) ---")
    local h_blas, HRR_blas
    try
        h_blas = packed_hess_via_octx(octx, ctx_cm.obj, x0v, n)
        Hfull_blas = unpack_packed(h_blas, n)
        HRR_blas = Hfull_blas[NCORE+1:NCORE+nx, NCORE+1:NCORE+nx]
        println("SUCCESS. H_ZZ is $(size(HRR_blas)). All finite: ", all(isfinite, HRR_blas))
        println("HRR_blas[1:3,1:3] = ")
        display(HRR_blas[1:min(3,nx), 1:min(3,nx)])
        println()
        println("HRR_blas diagonal[1:5] = ", diag(HRR_blas)[1:min(5,nx)])
    catch e
        println("THREW: ", sprint(showerror, e))
        for (i, frame) in enumerate(stacktrace(catch_backtrace()))
            i > 15 && break
            println("  [$i] ", frame)
        end
        HRR_blas = nothing
    end
    octx.zc_gram_backend = :reference   # restore default before next parameter vector

    # ---- direct comparison ----
    if HRR_ref !== nothing && HRR_blas !== nothing
        maxdiff = maximum(abs.(HRR_ref .- HRR_blas))
        relscale = max(1.0, maximum(abs.(HRR_ref)))
        println("\nCOMPARISON: max|HRR_ref - HRR_blas| = $maxdiff   (scale=$relscale, relative=$(maxdiff/relscale))")
    else
        println("\nCOMPARISON: cannot compare -- at least one backend did not produce a value.")
    end
    flush(stdout)
end
println("\nDONE.")
