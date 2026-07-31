# ============================================================================
# Task §6 round-trip + integration gate for relative_a_coordinate_2026-07-31.jl.
# ADDITIVE ONLY -- see that file's header for the design rationale.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
using Random, LinearAlgebra

ctx = d4_exact_setup()
D = ctx.D; Ddest = D   # square D=4 test context (no exclude_row here)
Aod_offset = ctx.Aod_offset
θ0 = copy(ctx.θ0_up)
rng = MersenneTwister(2026_07_31)

println("="^78); println("SETUP: AnchorSpec with a non-own anchor (D=4 analog of Korea->Brazil)"); println("="^78)
# destination 3's anchor is origin 1 (non-own), every other destination is own-cell,
# exactly mirroring the real manifest's "own-cell default, one explicit non-own override" shape.
spec = default_anchor_spec(D, Ddest; overrides = Dict(3 => 1))
println("anchor_origin = ", spec.anchor_origin, "  (destination 3 -> origin 1, non-own)")
println("n_retained(spec) = ", n_retained(spec), "  (expect D*Ddest-Ddest = $(D*Ddest-Ddest))")
@assert n_retained(spec) == D * Ddest - Ddest
ridx = retained_linear_indices(spec)
aidx = anchor_linear_indices(spec)
@assert length(ridx) == n_retained(spec)
@assert length(aidx) == Ddest
@assert isempty(intersect(ridx, aidx))
@assert sort(vcat(ridx, aidx)) == collect(1:D*Ddest)
println("PASS: retained/anchor index partition is exact")

println("\n" * "="^78); println("TEST 1: pure round-trip, r -> decode -> encode -> r"); println("="^78)
for trial in 1:5
    r_test = randn(rng, n_retained(spec))
    gauge_test = randn(rng, Ddest)
    z = decode_relative_A(r_test, spec, gauge_test)
    r_back = encode_relative_A(z, spec, gauge_test)
    maxdiff = maximum(abs.(r_test .- r_back))
    println("trial $trial: max|r_test - r_back| = $maxdiff")
    @assert maxdiff < 1e-13
    for d in 1:Ddest
        @assert z[spec.anchor_origin[d], d] == gauge_test[d]
    end
end
println("PASS")

println("\n" * "="^78); println("TEST 2: gauge built from GENUINE calibration (theta0_up), decode(encode(z_calib)) == z_calib exactly"); println("="^78)
Aod_θ0 = reshape(θ0[Aod_offset+1:Aod_offset+D^2], (D, D))
z_calib = log.(Aod_θ0)
gauge = build_anchor_gauge(z_calib, spec)
println("gauge (from calibration) = ", gauge)
r_calib = encode_relative_A(z_calib, spec, gauge)
z_rebuilt = decode_relative_A(r_calib, spec, gauge)
maxdiff2 = maximum(abs.(z_calib .- z_rebuilt))
println("max|z_calib - z_rebuilt| = $maxdiff2")
@assert maxdiff2 < 1e-13
println("PASS -- exact at genuine calibration, not the zfree=0 reference point")

println("\n" * "="^78); println("TEST 3: end-to-end integration -- decode into a real theta vector, feed ctx.obj.moments!, confirm bit-identical to direct evaluation"); println("="^78)
θ_rebuilt = copy(θ0)
θ_rebuilt[Aod_offset+1:Aod_offset+D^2] .= vec(exp.(z_rebuilt))
W = size(ctx.U, 1)
K_direct = zeros(W); G_direct = zeros(W, ctx.nTotalMoments)
K_rebuilt = zeros(W); G_rebuilt = zeros(W, ctx.nTotalMoments)
ctx.obj.moments!(K_direct, G_direct, θ0, ctx.U, ctx.obj)
ctx.obj.moments!(K_rebuilt, G_rebuilt, θ_rebuilt, ctx.U, ctx.obj)
maxdiff3 = maximum(abs.(G_direct .- G_rebuilt))
maxdiffK = maximum(abs.(K_direct .- K_rebuilt))
println("max|G_direct - G_rebuilt| = $maxdiff3   max|K_direct - K_rebuilt| = $maxdiffK")
@assert maxdiff3 < 1e-10 && maxdiffK < 1e-10
println("PASS -- relative-A round-trip through decode->theta->moments! reproduces genuine calibration bit-for-bit")

println("\n" * "="^78); println("TEST 4: perturbing ONLY retained coordinates leaves anchor cells fixed at gauge, and moments! runs cleanly"); println("="^78)
r_perturbed = r_calib .+ 0.05 .* randn(rng, n_retained(spec))
z_perturbed = decode_relative_A(r_perturbed, spec, gauge)
for d in 1:Ddest
    @assert z_perturbed[spec.anchor_origin[d], d] == gauge[d] "anchor cell moved under a retained-coordinate-only perturbation"
end
θ_perturbed = copy(θ0)
θ_perturbed[Aod_offset+1:Aod_offset+D^2] .= vec(exp.(z_perturbed))
K_p = zeros(W); G_p = zeros(W, ctx.nTotalMoments)
ctx.obj.moments!(K_p, G_p, θ_perturbed, ctx.U, ctx.obj)   # must not throw
println("moments! evaluated cleanly at a retained-only perturbation; anchor cells confirmed unmoved")
println("PASS")

println("\n" * "="^78); println("ALL TESTS PASSED"); println("="^78)
