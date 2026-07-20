# Quick diagnostic: where does the rank-47-of-48 deficiency (anchored, L=10) come from --
# core moments alone, CM block alone, or a cross-dependency? Also checks baseline (no-CM) rank.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "c12b_interval_common_marginals_moments.jl"))
using LinearAlgebra: svdvals
using Printf

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]

rankinfo(M; label = "") = begin
    sv = svdvals(M)
    tol = maximum(size(M)) * eps(maximum(sv))
    rnk = count(>(tol), sv)
    @printf("%-30s size=%-14s rank=%-4d smallest_sv=%.3e largest_sv=%.3e\n", label, string(size(M)), rnk, minimum(sv), maximum(sv))
    return sv
end

# baseline (no CM) core moments at calibration
θ_full0 = CS.reconstruct_full(x_free_calib, ctx.m)
K0, inner_x0, nStatus0 = CS.inner_loop_internal(ctx.obj, θ_full0)
@printf("baseline nStatus=%d\n", nStatus0)
W = size(ctx.U, 1); d0 = ctx.obj.d
G0 = zeros(W, d0)
ctx.obj.moments!(zeros(W), G0, θ_full0, ctx.U, ctx.obj)
rankinfo(G0; label = "core-only (no CM)")

# CM block alone (anchored, L=10)
CM, z, origins = precalc_common_marginals_cdf(ctx.U, ctx.γ.refIndex1, 10; contrasts = :anchored)
rankinfo(CM; label = "CM block alone (anchored,L=10)")

obj_cm = build_cm_augmented_obj_from_CM(ctx, CS, CM)
ctx_cm = merge(ctx, (obj = obj_cm,))
r = evaluate_fullA(x_free_calib, ctx_cm; use_cache = false, warm = false)
Gfull = zeros(W, obj_cm.d)
obj_cm.moments!(zeros(W), Gfull, r.θ_full, ctx.U, obj_cm)
rankinfo(Gfull; label = "core+CM (anchored,L=10)")

# is the CM block itself rank-deficient? (nO*L = 30 cols; check its own rank precisely)
sv_cm = svdvals(CM)
@printf("CM block singular values (last 5): %s\n", sv_cm[end-4:end])

# core-only singular values (last 5) to see if the deficiency is inherited from core
sv_core = svdvals(G0)
@printf("core-only singular values (last 5): %s\n", sv_core[end-4:end])
