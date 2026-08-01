# ============================================================================
# Task §9 gate: homogeneous forward/transpose contraction vs. direct
# computation from the already-verified per-cell homogeneous moments.
# ADDITIVE ONLY -- see homogeneous_contraction_2026-07-31.jl header.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(@__DIR__, "recover_full_a_2026-07-31.jl"))
include(joinpath(@__DIR__, "homogeneous_moments_2026-07-31.jl"))
include(joinpath(@__DIR__, "homogeneous_contraction_2026-07-31.jl"))
using Random, LinearAlgebra

ctx = d4_exact_setup()
D = ctx.D; Ddest = D
θ0 = copy(ctx.θ0_up)
bi = ctx.bi
rng = MersenneTwister(1082026)

println("="^78); println("SETUP: build the FULL per-draw homogeneous moment matrix by hand (already-verified path)"); println("="^78)
cf = build_compressed_factual(θ0, ctx; check_ties = false)
W = cf.W; ncol = cf.oci - 1
println("D=$D Ddest=$Ddest W=$W ncol=$ncol cf_col=$(cf.cf_col)")

Gnew = zeros(W, ncol)
Hmoments = homogeneous_factual_moment(θ0, ctx)   # Dict d -> W x D matrix
for d in 1:D
    s = dest_slot(ctx, d)
    for o in 1:D
        j = s + (o - 1) * Ddest
        Gnew[:, j] .= Hmoments[d][:, o] .* cf.nrm[j] .* cf.gdiv[j]   # apply the SAME nrm/gdiv post-scale the real kernels apply
    end
end
if cf.cf_col > 0
    Hfrance = homogeneous_france_moment(θ0, ctx)
    j = cf.cf_col
    Gnew[:, j] .= Hfrance .* cf.nrm[j] .* cf.gdiv[j]
end
# usePMM==0 in this context (confirmed elsewhere in this repo); SW applied at the very end below,
# matching compressed_dual_contraction's own `t[w] = SW[w]*(acc-pmmterm)` convention.
@assert cf.usePMM == 0 "this test assumes usePMM==0, matching every existing config in this repo"

println("\n" * "="^78); println("TEST 1: forward contraction matches direct G_new*β to near machine precision"); println("="^78)
for trial in 1:3
    β = randn(rng, ncol) .* 0.3
    t_kernel = homogeneous_dual_contraction(β, cf, ctx, θ0)
    t_direct = cf.SW[1:W] .* (Gnew * β)
    maxdiff = maximum(abs.(t_kernel .- t_direct))
    relscale = maximum(abs.(t_direct))
    println("trial $trial: max|t_kernel - t_direct| = $maxdiff  (scale ~$relscale)")
    @assert maxdiff < 1e-8 * max(1.0, relscale) "TEST 1 FAILED: forward contraction mismatch"
end
println("PASS")

println("\n" * "="^78); println("TEST 2: transpose contraction matches direct G_new'*weights to near machine precision"); println("="^78)
Bscratch = zeros(D, Ddest)
Tscratch = zeros(Ddest)
for trial in 1:3
    weights = randn(rng, W) .* 0.3
    v_kernel = zeros(ncol)
    homogeneous_transpose_contraction!(v_kernel, weights, cf, ctx, θ0, Bscratch, Tscratch)
    v_direct = Gnew' * (cf.SW[1:W] .* weights)
    maxdiff = maximum(abs.(v_kernel .- v_direct))
    relscale = maximum(abs.(v_direct))
    println("trial $trial: max|v_kernel - v_direct| = $maxdiff  (scale ~$relscale)")
    @assert maxdiff < 1e-8 * max(1.0, relscale) "TEST 2 FAILED: transpose contraction mismatch"
end
println("PASS")

println("\n" * "="^78); println("TEST 3: forward/transpose are exact adjoints of each other (sanity: β'*(G'w) == (Gβ)'w)"); println("="^78)
β = randn(rng, ncol) .* 0.2
weights = randn(rng, W) .* 0.2
t_kernel = homogeneous_dual_contraction(β, cf, ctx, θ0)
v_kernel = zeros(ncol)
homogeneous_transpose_contraction!(v_kernel, weights, cf, ctx, θ0, Bscratch, Tscratch)
lhs = dot(weights, t_kernel)
rhs = dot(β, v_kernel)
println("dot(weights, t) = $lhs   dot(β, v) = $rhs   diff = $(abs(lhs-rhs))")
@assert abs(lhs - rhs) < 1e-6 * max(1.0, abs(lhs))
println("PASS -- forward and transpose kernels are consistent adjoints")

println("\n" * "="^78); println("ALL TESTS PASSED"); println("="^78)
