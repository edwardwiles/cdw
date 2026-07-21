# Smoke test for the conditioning-battery machinery (Hessian extraction, rank, timing) before
# scaling to the full 4-basis x 3-L x >=2-point x 4-refIndex battery.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "c12b_interval_common_marginals_moments.jl"))
using LinearAlgebra: cond, svdvals
using Printf

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]

function hessian_at(obj_cm)
    n = obj_cm.outer_constr_index   # = length(x) = length(obj_cm.x)
    hbuf = zeros(div(n * (n + 1), 2))
    obj_cm(obj_cm.x, h = hbuf)
    Hfull = Matrix{Float64}(undef, n, n)
    k = 1
    for i in 1:n, j in i:n
        Hfull[i, j] = hbuf[k]; Hfull[j, i] = hbuf[k]
        k += 1
    end
    return Hfull
end

t0 = time()
CM, z, origins = precalc_common_marginals_cdf(ctx.U, ctx.γ.refIndex1, 10; contrasts = :anchored)
obj_cm = build_cm_augmented_obj_from_CM(ctx, CS, CM)
ctx_cm = merge(ctx, (obj = obj_cm,))
r = evaluate_fullA(x_free_calib, ctx_cm; use_cache = false, warm = false)
t1 = time()
@printf("anchored L=10: nStatus=%d Delta_dual=%.6f t=%.2fs\n", r.inner_status, r.Delta_dual, t1 - t0)

Hfull = hessian_at(obj_cm)
@printf("Hessian size=%s  cond=%.4e\n", size(Hfull), cond(Hfull))

W = size(ctx.U, 1); d = obj_cm.d
K = zeros(W); G = zeros(W, d)
obj_cm.moments!(K, G, r.θ_full, ctx.U, obj_cm)
sv = svdvals(G)
tol = maximum(size(G)) * eps(maximum(sv))
rnk = count(>(tol), sv)
@printf("moment matrix G: size=%s rank=%d (tol=%.2e) smallest/largest sv=%.2e / %.2e\n",
        size(G), rnk, tol, minimum(sv), maximum(sv))

@printf("max |lambda| overall = %.4e\n", maximum(abs.(r.lambda)))
