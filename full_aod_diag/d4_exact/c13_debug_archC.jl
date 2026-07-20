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

L = 10
aug = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
objA = aug.obj_cm
n = objA.outer_constr_index
NCORE = aug.ncore
ncm = aug.ncm
println("NCORE=$NCORE ncm=$ncm n=$n nO=", length(aug.origins), " origins=", aug.origins, " refIndex1=", aug.refIndex1)

K = zeros(size(ctx.U,1))
objA.moments!(K, CS.select_G_from_H(objA, objA.H), θ_full, objA.U, objA)
objA.H[:,1] .= K
objA.H[:,2] .= 1.0

Random.seed!(2)
x = 0.01 .* randn(n)

hA = Vector{Float64}(undef, n*(n+1)÷2)
objA(x, h = hA)
function unpack_packed(h::AbstractVector, n::Int)
    Mm = Matrix{Float64}(undef, n, n)
    k = 1
    @inbounds for i in 1:n, j in i:n
        Mm[i, j] = h[k]; Mm[j, i] = h[k]
        k += 1
    end
    return Mm
end
HA = unpack_packed(hA, n)

cctx = build_cm_bin_ctx(ctx, aug)
hC = Vector{Float64}(undef, n*(n+1)÷2)
_archC_prep_for_hessian!(objA, x)
hessian_cm_structured!(hC, objA, cctx)
HC = unpack_packed(hC, n)

HEE_A = HA[1:NCORE,1:NCORE]; HEE_C = HC[1:NCORE,1:NCORE]
HEC_A = HA[1:NCORE,NCORE+1:end]; HEC_C = HC[1:NCORE,NCORE+1:end]
HCC_A = HA[NCORE+1:end,NCORE+1:end]; HCC_C = HC[NCORE+1:end,NCORE+1:end]

@printf("H_EE err = %.3e\n", maximum(abs.(HEE_A .- HEE_C)))
@printf("H_EC err = %.3e\n", maximum(abs.(HEC_A .- HEC_C)))
@printf("H_CC err = %.3e\n", maximum(abs.(HCC_A .- HCC_C)))

# Check CM columns directly against G matrix in objA.H, and check bin-based recompute matches CM values directly.
G = CS.select_G_from_H(objA, objA.H)
CMcols = G[:, NCORE:NCORE+ncm-1]
z = aug.z
Bidx = compute_bin_indices(ctx.U, z)
nO = length(aug.origins)
# recompute raw CM (anchored -> R=nothing) column (l=1,oi=1) directly from bins and compare to aug.CM / G
o1 = aug.origins[1]
l = 1
colidx = (l-1)*nO + 1
manual = Float64.(Bidx[:,o1] .<= l) .- Float64.(Bidx[:,aug.refIndex1] .<= l)
@printf("manual CM col1 vs G col: max diff = %.3e\n", maximum(abs.(manual .- CMcols[:,colidx])))
@printf("manual CM col1 vs aug.CM: max diff = %.3e\n", maximum(abs.(manual .- aug.CM[:,colidx])))

# Now check H_EC[j=1 (ones), (l=1,oi=1)] by hand: (1/M) sum_s w_s * 1 * CM[s,colidx]
ddc = zeros(length(K))
objA.ddPsi!(ddc, objA.arg0)
Mval = objA.M
manualHEC_11 = sum(ddc .* CMcols[:,colidx]) / Mval
@printf("H_EC[1,1] (ones vs CM col1): A=%.6e  manual-direct=%.6e  C=%.6e\n", HEC_A[1,1], manualHEC_11, HEC_C[1,1])

# check H_CC[1,1] (col1,col1) by hand
manualHCC_11 = sum(ddc .* CMcols[:,colidx] .* CMcols[:,colidx]) / Mval
@printf("H_CC[1,1]: A=%.6e manual-direct=%.6e C=%.6e\n", HCC_A[1,1], manualHCC_11, HCC_C[1,1])

println("\n--- elementwise H_EC ratio (A/C), first 6x6 block ---")
for j in 1:6
    println([round(HEC_A[j,k]/HEC_C[j,k], digits=4) for k in 1:6])
end
println("\nrow-1 (ones) HEC_A: ", HEC_A[1,1:6])
println("row-1 (ones) HEC_C: ", HEC_C[1,1:6])
println("row-2 HEC_A: ", HEC_A[2,1:6])
println("row-2 HEC_C: ", HEC_C[2,1:6])
