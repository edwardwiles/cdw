# ============================================================================
# Claude Code task 2026-08-01, §6/§7 gate: REDUCED forward/transpose vs. (1) a
# direct dense reduced-G built from the already-verified homogeneous moments,
# (2) the OLD full homogeneous kernel evaluated at the zero-anchor-embedded
# point (exact algebraic equivalence, not just close), (3) mutual adjointness.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(@__DIR__, "recover_full_a_2026-07-31.jl"))
include(joinpath(@__DIR__, "homogeneous_moments_2026-07-31.jl"))
include(joinpath(@__DIR__, "homogeneous_contraction_2026-07-31.jl"))
include(joinpath(@__DIR__, "profiled_economic_moment_layout_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_homogeneous_contraction_2026-08-01.jl"))
using Random, LinearAlgebra

ctx = d4_exact_setup()
D = ctx.D; Ddest = D
θ0 = copy(ctx.θ0_up)
rng = MersenneTwister(20260801)

cf = build_compressed_factual(θ0, ctx; check_ties = false)
W = cf.W; ncolI_full = cf.oci - 1
has_france = cf.cf_col > 0
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(3 => 1))
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
n_red = layout.total_reduced_economic_moments
println("D=$D Ddest=$Ddest W=$W ncolI_full=$ncolI_full n_reduced=$n_red")
assert_no_factual_price_index_moment(layout)

println("\n" * "="^78); println("SETUP: direct dense REDUCED G (restrict already-verified full homogeneous moments to retained columns)"); println("="^78)
Gred = zeros(W, n_red)
Hmoments = homogeneous_factual_moment(θ0, ctx)
for k in eachindex(layout.retained_full_factual_j)
    o = layout.retained_origin[k]; s = layout.retained_slot[k]
    j_full = layout.retained_full_factual_j[k]
    Gred[:, k] .= Hmoments[s][:, o] .* cf.nrm[j_full] .* cf.gdiv[j_full]
end
if has_france
    Hfrance = homogeneous_france_moment(θ0, ctx)
    j_full = cf.cf_col
    Gred[:, layout.france_ratio_reduced_j] .= Hfrance .* cf.nrm[j_full] .* cf.gdiv[j_full]
end
@assert cf.usePMM == 0 "this test assumes usePMM==0, matching every existing config in this repo"

println("\n" * "="^78); println("TEST 1: reduced forward matches direct dense Gred*β"); println("="^78)
for trial in 1:3
    β = randn(rng, n_red) .* 0.3
    t_kernel = reduced_homogeneous_dual_contraction(β, cf, ctx, θ0, layout)
    t_direct = cf.SW[1:W] .* (Gred * β)
    maxdiff = maximum(abs.(t_kernel .- t_direct))
    relscale = maximum(abs.(t_direct))
    println("trial $trial: max|t_kernel - t_direct| = $maxdiff (scale ~$relscale)")
    @assert maxdiff < 1e-8 * max(1.0, relscale) "TEST 1 FAILED"
end
println("PASS")

println("\n" * "="^78); println("TEST 2: reduced transpose matches direct dense Gred'*weights"); println("="^78)
Bscratch = zeros(D, Ddest); Tscratch = zeros(Ddest)
for trial in 1:3
    weights = randn(rng, W) .* 0.3
    v_kernel = zeros(n_red)
    reduced_homogeneous_transpose_contraction!(v_kernel, weights, cf, ctx, θ0, layout, Bscratch, Tscratch)
    v_direct = Gred' * (cf.SW[1:W] .* weights)
    maxdiff = maximum(abs.(v_kernel .- v_direct))
    relscale = maximum(abs.(v_direct))
    println("trial $trial: max|v_kernel - v_direct| = $maxdiff (scale ~$relscale)")
    @assert maxdiff < 1e-8 * max(1.0, relscale) "TEST 2 FAILED"
end
println("PASS")

println("\n" * "="^78); println("TEST 3: reduced kernel == OLD FULL kernel at the zero-anchor-embedded point (exact equivalence, cross-checks reduced_homogeneous_contraction against homogeneous_contraction_2026-07-31.jl independently of the dense-G reference)"); println("="^78)
for trial in 1:3
    β = randn(rng, n_red) .* 0.25
    β_full = expand_reduced_beta_to_full(β, layout, ncolI_full)
    t_reduced = reduced_homogeneous_dual_contraction(β, cf, ctx, θ0, layout)
    t_full = homogeneous_dual_contraction(β_full, cf, ctx, θ0)
    maxdiff = maximum(abs.(t_reduced .- t_full))
    println("trial $trial (forward): max|t_reduced - t_full(zero-anchor)| = $maxdiff")
    @assert maxdiff < 1e-10 "TEST 3 (forward) FAILED"
end
Bscratch2 = zeros(D, Ddest); Tscratch2 = zeros(Ddest)
for trial in 1:3
    weights = randn(rng, W) .* 0.25
    v_reduced = zeros(n_red)
    reduced_homogeneous_transpose_contraction!(v_reduced, weights, cf, ctx, θ0, layout, Bscratch2, Tscratch2)
    v_full = zeros(ncolI_full)
    homogeneous_transpose_contraction!(v_full, weights, cf, ctx, θ0, Bscratch2, Tscratch2)
    v_full_restricted = [v_full[layout.retained_full_factual_j[k]] for k in eachindex(layout.retained_full_factual_j)]
    v_reduced_bilateral = v_reduced[1:length(layout.retained_full_factual_j)]
    maxdiff = maximum(abs.(v_reduced_bilateral .- v_full_restricted))
    println("trial $trial (transpose, bilateral): max diff = $maxdiff")
    @assert maxdiff < 1e-10 "TEST 3 (transpose bilateral) FAILED"
    if has_france
        d2 = abs(v_reduced[layout.france_ratio_reduced_j] - v_full[cf.cf_col])
        println("trial $trial (transpose, france): diff = $d2")
        @assert d2 < 1e-10 "TEST 3 (transpose france) FAILED"
    end
end
println("PASS -- reduced kernels are algebraically identical to the full kernel with the anchor coordinate's beta forced to exactly 0.0, confirming this is a genuine dimension reduction, not a new derivation")

println("\n" * "="^78); println("TEST 4: reduced forward/transpose are exact mutual adjoints"); println("="^78)
β = randn(rng, n_red) .* 0.2
weights = randn(rng, W) .* 0.2
t_kernel = reduced_homogeneous_dual_contraction(β, cf, ctx, θ0, layout)
v_kernel = zeros(n_red)
reduced_homogeneous_transpose_contraction!(v_kernel, weights, cf, ctx, θ0, layout, Bscratch, Tscratch)
lhs = dot(weights, t_kernel)
rhs = dot(β, v_kernel)
println("dot(weights, t) = $lhs   dot(β, v) = $rhs   diff = $(abs(lhs-rhs))")
@assert abs(lhs - rhs) < 1e-6 * max(1.0, abs(lhs))
println("PASS")

println("\n" * "="^78); println("ALL TESTS PASSED"); println("="^78)
