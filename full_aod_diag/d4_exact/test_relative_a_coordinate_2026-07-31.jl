# ============================================================================
# Task §6 round-trip + integration gate for relative_a_coordinate_2026-07-31.jl.
# ADDITIVE ONLY -- see that file's header for the design rationale.
#
# CORRECTED 2026-07-31 (same day, user stop): Tests 3-4's "end-to-end
# integration" originally called the LEGACY dense `ctx.obj.moments!`
# (G/K matrix). Rebuilt to use `build_compressed_factual` -- see
# recover_full_a_2026-07-31.jl's header for the full correction rationale.
# `ctx = d4_exact_setup()` is still used only as a DATA/economy builder; its
# dense `ctx.obj` is never read.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
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

println("\n" * "="^78); println("TEST 3: end-to-end integration -- decode into a real theta vector, feed build_compressed_factual, confirm bit-identical to direct evaluation"); println("="^78)
θ_rebuilt = copy(θ0)
θ_rebuilt[Aod_offset+1:Aod_offset+D^2] .= vec(exp.(z_rebuilt))
cf_direct = build_compressed_factual(θ0, ctx; check_ties = false)
cf_rebuilt = build_compressed_factual(θ_rebuilt, ctx; check_ties = false)
maxdiff_wval = maximum(abs.(cf_direct.wval .- cf_rebuilt.wval))
maxdiff_cfraw = maximum(abs.(cf_direct.cf_raw .- cf_rebuilt.cf_raw))
n_winner_mismatch = sum(cf_direct.winner .!= cf_rebuilt.winner)
println("max|wval_direct - wval_rebuilt| = $maxdiff_wval   max|cf_raw_direct - cf_raw_rebuilt| = $maxdiff_cfraw   winner mismatches = $n_winner_mismatch")
@assert maxdiff_wval < 1e-10 && maxdiff_cfraw < 1e-10 && n_winner_mismatch == 0
println("PASS -- relative-A round-trip through decode->theta->build_compressed_factual reproduces genuine calibration bit-for-bit")

println("\n" * "="^78); println("TEST 4: perturbing ONLY retained coordinates leaves anchor cells fixed at gauge, and build_compressed_factual runs cleanly"); println("="^78)
r_perturbed = r_calib .+ 0.05 .* randn(rng, n_retained(spec))
z_perturbed = decode_relative_A(r_perturbed, spec, gauge)
for d in 1:Ddest
    @assert z_perturbed[spec.anchor_origin[d], d] == gauge[d] "anchor cell moved under a retained-coordinate-only perturbation"
end
θ_perturbed = copy(θ0)
θ_perturbed[Aod_offset+1:Aod_offset+D^2] .= vec(exp.(z_perturbed))
build_compressed_factual(θ_perturbed, ctx; check_ties = false)   # must not throw
println("build_compressed_factual evaluated cleanly at a retained-only perturbation; anchor cells confirmed unmoved")
println("PASS")

println("\n" * "="^78); println("ALL TESTS PASSED"); println("="^78)
