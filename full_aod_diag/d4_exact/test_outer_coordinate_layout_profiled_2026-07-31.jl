# ============================================================================
# Task §6/§10 gate: profiled outer-vector decode/encode, KNITRO-facing shape.
# ADDITIVE ONLY -- see outer_coordinate_layout_profiled_2026-07-31.jl header.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(@__DIR__, "gravity_pivot_on_retained_2026-07-31.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout_profiled_2026-07-31.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
using Random

ctx = d4_exact_setup()
D = ctx.D; Ddest = D
Aod_offset = ctx.Aod_offset
θ0 = copy(ctx.θ0_up)
rng = MersenneTwister(3107202601)

spec = default_anchor_spec(D, Ddest; overrides = Dict(3 => 1))
Aod_θ0 = reshape(θ0[Aod_offset+1:Aod_offset+D^2], (D, D))
z_calib = log.(Aod_θ0)
gauge = build_anchor_gauge(z_calib, spec)
pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
gp_calib = θ0[3+D]

println("="^78); println("TEST 1: dimension -- profiled outer vector is exactly Ddest shorter than full"); println("="^78)
full_outer_dim = D * Ddest   # decode_outer_unified's own fixed-mode outer_dim(layout,D,Ddest)
prof_dim = outer_dim_profiled(pe)
println("full outer_dim = $full_outer_dim   profiled outer_dim = $prof_dim   difference = $(full_outer_dim - prof_dim) (expect Ddest=$Ddest)")
@assert full_outer_dim - prof_dim == Ddest
println("PASS")

println("\n" * "="^78); println("TEST 2: round trip, (gp,z_full) -> w_profiled -> decode -> xf reproduces calibration"); println("="^78)
w_profiled_calib = reduce_to_w_profiled(gp_calib, z_calib, pe)
println("length(w_profiled_calib) = ", length(w_profiled_calib), " (expect $prof_dim)")
@assert length(w_profiled_calib) == prof_dim
result = decode_outer_profiled(w_profiled_calib, ctx, pe)
println("length(xf) = ", length(result.xf), " (expect 1+D^2=$(1+D^2), matching decode_outer_unified's own fixed-mode xf shape)")
@assert length(result.xf) == 1 + D^2
maxdiff_A = maximum(abs.(result.Aod_levels .- vec(Aod_θ0)))
println("max|Aod_levels_recovered - Aod_theta0| = $maxdiff_A")
@assert maxdiff_A < 1e-10
@assert result.gp == gp_calib
println("PASS -- profiled decode reproduces genuine calibration's full Aod_theta levels exactly")

println("\n" * "="^78); println("TEST 3: xf feeds build_compressed_factual cleanly (production-shaped input)"); println("="^78)
θ_from_xf = copy(θ0)
θ_from_xf[3+D] = result.xf[1]
θ_from_xf[Aod_offset+1:Aod_offset+D^2] .= result.xf[2:end]
cf = build_compressed_factual(θ_from_xf, ctx; check_ties = false)
println("build_compressed_factual succeeded: D_dest=$(cf.D_dest), W=$(cf.W)")
println("PASS")

println("\n" * "="^78); println("TEST 4: perturbed round trip at a non-calibration point"); println("="^78)
n_r_free = outer_dim_profiled(pe) - 1
for trial in 1:5
    r_free_test = randn(rng, n_r_free) .* 0.2
    w_test = vcat(gp_calib * (1 + 0.01 * randn(rng)), r_free_test)
    res = decode_outer_profiled(w_test, ctx, pe)
    w_back = reduce_to_w_profiled(res.gp, res.z_full, pe)
    maxdiff = maximum(abs.(w_test .- w_back))
    println("trial $trial: max round-trip diff = $maxdiff")
    @assert maxdiff < 1e-8
    g = gravity_from_logz(res.z_full, ctx)
    @assert abs(g) < 1e-9 "gravity not feasible at decoded profiled point"
end
println("PASS -- round trip and gravity-feasibility hold at random profiled points, not just calibration")

println("\n" * "="^78); println("ALL TESTS PASSED"); println("="^78)
