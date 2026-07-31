#!/usr/bin/env julia
# Regression check for the 2026-07-31 legacy-H audit fix to evaluate_melitz_delta_from_solution
# (delta_star.jl): store_G=true used to call obj_like.moments! directly (a guaranteed
# FieldError for MelitzCCBundle, which has no .moments! field) -- now dispatches through
# melitz_bundle_dense_G_at_theta like its sibling evaluate_melitz_delta already did. This
# script proves the fixed path actually returns a correct dense G for MelitzCCBundle, not
# just that the file compiles.
using Pkg
Pkg.activate(dirname(@__DIR__))
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO, LinearAlgebra

data4 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj4, theta4 = build_melitz_psi_bundle(data4; policy=CappedEvaluation(10.0))
@assert obj4 isa MelitzCCBundle

val, x, nStatus = melitz_bundle_inner_loop(obj4, theta4)
@assert nStatus == 0

# BEFORE the fix, this line would throw:
#   FieldError: type MelitzCCBundle has no field `moments!`
r = evaluate_melitz_delta_from_solution(theta4, obj4.γ, obj4, val, x, nStatus; store_G=true)
println("store_G=true succeeded for MelitzCCBundle: typeof(r.G) = ", typeof(r.G), "  size = ", size(r.G))
@assert r.G isa Matrix{Float64}
@assert size(r.G) == (obj4.op.W, obj4.d)

# cross-check against the already-trusted melitz_bundle_dense_G_at_theta / melitz_dense_G_from_operator
G_ref = melitz_dense_G_from_operator(obj4.op)
maxdiff = maximum(abs, r.G .- G_ref)
println("max|G_from_fix - G_from_known_diagnostic_path| = ", maxdiff)
@assert maxdiff == 0.0

println("PASS: evaluate_melitz_delta_from_solution(store_G=true) now works for MelitzCCBundle and matches the reference dense reconstruction exactly.")
