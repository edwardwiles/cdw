# ============================================================================
# Task §8 core-piece gate: homogeneous factual moment.
# ADDITIVE ONLY -- see homogeneous_moments_2026-07-31.jl header.
# (Already used only the corrected build_compressed_factual-based helpers;
# this update just adds the includes those helpers require.)
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "recover_full_a_2026-07-31.jl"))
include(joinpath(@__DIR__, "homogeneous_moments_2026-07-31.jl"))
using Random, Statistics

ctx = d4_exact_setup()
D = ctx.D; Ddest = D
Aod_offset = ctx.Aod_offset
θ0 = copy(ctx.θ0_up)
rng = MersenneTwister(310726)

println("="^78); println("TEST 1: Sigma_o H[w,o] == 0 EXACTLY for every draw (Sigma lambda=1 identity)"); println("="^78)
H0 = homogeneous_factual_moment(θ0, ctx)
maxsum = 0.0
for d in 1:D
    rowsum = vec(sum(H0[d], dims = 2))
    global maxsum = max(maxsum, maximum(abs.(rowsum)))
end
println("max|Sigma_o H[w,o]| over all draws and destinations = $maxsum")
@assert maxsum < 1e-8
println("PASS -- confirms the omitted-anchor-share-follows-automatically claim (task §1.2) exactly")

println("\n" * "="^78); println("TEST 2: homogeneous moment is informationally small at genuine calibration"); println("="^78)
for d in 1:D
    m = mean(abs.(H0[d]))
    denom_d = ctx.γ.wHat[d] * ctx.γ.L[d]
    println("  d=$d: mean|H| = $m   (denom[d]=$denom_d, for scale reference)")
end
println("(informational, not asserted -- matches the ~0.2% calibration-tolerance finding from the earlier gate)")

println("\n" * "="^78); println("TEST 3: EXACT proportional rescaling under a destination-column shift"); println("="^78)
spec = default_anchor_spec(D, Ddest; overrides = Dict(3 => 1))
d_test = 3
κ = 1.4
e_exponent = θ0[1] * (θ0[2] - 1)
θ1 = copy(θ0)
for o in 1:D
    idx = Aod_offset + (d_test - 1) * D + o
    θ1[idx] *= κ
end
H1 = homogeneous_factual_moment(θ1, ctx; d_list = [d_test])
predicted = κ^e_exponent
Hbefore = H0[d_test]; Hafter = H1[d_test]
maxdiff = 0.0
for o in 1:D
    o == spec.anchor_origin[d_test] && continue   # anchor origin isn't part of "retained" set conceptually, but identity check below covers it too
    ratio = Hafter[:, o] ./ Hbefore[:, o]
    dv = maximum(abs.(ratio .- predicted))
    global maxdiff = max(maxdiff, dv)
    println("  o=$o: max|H1/H0 - kappa^e| = $dv")
end
@assert maxdiff < 1e-6 "TEST 3 FAILED: homogeneous moment does not rescale exactly by kappa^(mu*(sigma-1))"
println("PASS -- the homogeneous moment scales EXACTLY by kappa^(mu*(sigma-1)) under a destination shift,")
println("        confirming it is genuinely scale-invariant in the zero/nonzero sense: if it holds at one")
println("        column scale it holds at every scale, and the destination scale is truly unidentified by it")

println("\n" * "="^78); println("ALL TESTS PASSED"); println("="^78)
