# Continuation (branch diag/fullA-d4-exact-cm-hessian-arch), step 1: pure
# numerical validation of Architectures B/C/D against Architecture A (the
# trusted baseline dense-BLAS `hessian!`, unchanged), at a FIXED point --
# no KNITRO solve yet (that's c13_bench_hessian_archs.jl). L in {10,20,50},
# both contrasts. Any disagreement beyond float tolerance is treated as a bug
# to find and fix, per task brief.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
using Printf, LinearAlgebra, Random

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full = CS.reconstruct_full(x_free_calib, ctx.m)

"Unpack a packed upper-triangular Hessian vector (as produced by hessian!/hessian_cm_structured!) into a dense symmetric n x n matrix."
function unpack_packed(h::AbstractVector, n::Int)
    M = Matrix{Float64}(undef, n, n)
    k = 1
    @inbounds for i in 1:n, j in i:n
        M[i, j] = h[k]; M[j, i] = h[k]
        k += 1
    end
    return M
end

function validate_at(L::Int, contrasts::Symbol; seed::Int = 1000 + L)
    println("\n=== L=$L contrasts=$contrasts ===")

    # ---- Architecture A: reference ----
    aug = build_cm_augmented_obj(ctx, CS; L = L, contrasts = contrasts)
    objA = aug.obj_cm
    n = objA.outer_constr_index
    K = zeros(size(ctx.U,1));
    objA.moments!(K, CS.select_G_from_H(objA, objA.H), θ_full, objA.U, objA)
    objA.H[:,1] .= K
    objA.H[:,2] .= 1.0

    Random.seed!(seed)
    npts = 3
    xs = [vcat(0.0, zeros(n-1)), 0.01 .* randn(n), 0.05 .* randn(n)]

    results = NamedTuple[]

    for (pi_, x) in enumerate(xs)
        hA = Vector{Float64}(undef, n*(n+1)÷2)
        objA(x, h = hA)   # unconditional gemv!+Psi! prep happens inside, then hessian!
        HA = unpack_packed(hA, n)

        # ---- Architecture C: structured, same H (objA.H), swap Hessian fn ----
        cctx = build_cm_bin_ctx(ctx, aug)
        hC = Vector{Float64}(undef, n*(n+1)÷2)
        _archC_prep_for_hessian!(objA, x)
        hessian_cm_structured!(hC, objA, cctx)
        HC = unpack_packed(hC, n)
        errC = maximum(abs.(HC .- HA))
        relC = errC / max(1.0, maximum(abs.(HA)))

        # ---- Architecture D: HVP, random directions ----
        Random.seed!(seed*7 + pi_)
        nv = 5
        maxerrD = 0.0
        for _ in 1:nv
            v = randn(n)
            _archC_prep_for_hessian!(objA, x)
            zbuf = Vector{Float64}(undef, size(objA.H,1))
            Hv = Vector{Float64}(undef, n)
            hvp_dense!(Hv, objA, v, zbuf)
            HvA = HA * v
            maxerrD = max(maxerrD, maximum(abs.(Hv .- HvA)) / max(1.0, maximum(abs.(HvA))))
        end

        push!(results, (pt = pi_, errC_abs = errC, errC_rel = relC, errD_rel = maxerrD))
        @printf("  x[%d]: |H_C - H_A|_inf = %.3e (rel %.3e)   HVP_D rel err (max over %d dirs) = %.3e\n",
                pi_, errC, relC, nv, maxerrD)
    end

    # ---- Architecture B: independent moment construction, compare G matrix + Hessian ----
    augB = build_cm_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts)
    objB = augB.obj_cm
    KB = zeros(size(ctx.U,1))
    objB.moments!(KB, CS.select_G_from_H(objB, objB.H), θ_full, objB.U, objB)
    objB.H[:,1] .= KB
    objB.H[:,2] .= 1.0
    GA = CS.select_G_from_H(objA, objA.H)
    GB = CS.select_G_from_H(objB, objB.H)
    gerr = maximum(abs.(GA .- GB))
    @printf("  Architecture B: max|G_A - G_B| = %.3e  (K match: %s)\n", gerr, K == KB)

    x = xs[2]
    hB = Vector{Float64}(undef, n*(n+1)÷2)
    objB(x, h = hB)
    HB = unpack_packed(hB, n)
    hA2 = Vector{Float64}(undef, n*(n+1)÷2)
    objA(x, h = hA2)
    HA2 = unpack_packed(hA2, n)
    errB = maximum(abs.(HB .- HA2))
    @printf("  Architecture B Hessian vs A at x[2]: |H_B - H_A|_inf = %.3e\n", errB)

    return (L = L, contrasts = contrasts, results = results, gerr = gerr, errB = errB)
end

all_results = NamedTuple[]
for L in (10, 20, 50)
    for contrasts in (:anchored, :orthonormal)
        push!(all_results, validate_at(L, contrasts))
    end
end

println("\n=== SUMMARY ===")
for r in all_results
    maxC = maximum(x.errC_abs for x in r.results)
    maxD = maximum(x.errD_rel for x in r.results)
    @printf("L=%2d %-10s : max|H_C-H_A|=%.3e  max HVP_D rel=%.3e  G_A-G_B=%.3e  H_B-H_A=%.3e\n",
            r.L, string(r.contrasts), maxC, maxD, r.gerr, r.errB)
end
