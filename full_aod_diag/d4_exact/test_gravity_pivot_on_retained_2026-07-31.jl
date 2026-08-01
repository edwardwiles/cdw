# ============================================================================
# Task §7 composition gate: anchor reduction + gravity pivot together.
# ADDITIVE ONLY -- see gravity_pivot_on_retained_2026-07-31.jl header.
#
# CORRECTED 2026-07-31 (same day, user stop): Test 6 originally called the
# LEGACY dense `ctx.obj.moments!` (G/K matrix). Rebuilt to use
# `build_compressed_factual` -- see recover_full_a_2026-07-31.jl's header for
# the full correction rationale.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(@__DIR__, "gravity_pivot_on_retained_2026-07-31.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
using Random

ctx = d4_exact_setup()
D = ctx.D; Ddest = D
Aod_offset = ctx.Aod_offset
θ0 = copy(ctx.θ0_up)
rng = MersenneTwister(20260731)

spec = default_anchor_spec(D, Ddest; overrides = Dict(3 => 1))
Aod_θ0 = reshape(θ0[Aod_offset+1:Aod_offset+D^2], (D, D))
z_calib = log.(Aod_θ0)
gauge = build_anchor_gauge(z_calib, spec)

println("="^78); println("SETUP + TEST 1: dimension audit"); println("="^78)
println("D=$D Ddest=$Ddest  active A = $(D*Ddest)")
println("n_retained(spec) = $(n_retained(spec))  (active A - Ddest = $(D*Ddest-Ddest))")
pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
n_free_composed = n_retained(spec) - 1
println("free-A after anchor reduction + gravity pivot = $(n_free_composed)  (= D*Ddest - Ddest - 1 = $(D*Ddest-Ddest-1))")
@assert n_free_composed == D * Ddest - Ddest - 1
println("PASS")

println("\n" * "="^78); println("TEST 2: composed pivot is never an anchor cell"); println("="^78)
ridx = retained_linear_indices(spec)
pivot_full_lin = ridx[pe.pivot_pos]
aidx = anchor_linear_indices(spec)
println("pivot (full-space linear index) = $pivot_full_lin ; anchor indices = $aidx")
@assert !(pivot_full_lin in aidx)
println("PASS")

println("\n" * "="^78); println("TEST 3: round trip, r_free -> expand -> reduce -> r_free"); println("="^78)
for trial in 1:5
    r_free_test = randn(rng, n_free_composed)
    r_expanded = pivot_expand_on_retained(r_free_test, pe)
    r_free_back = pivot_reduce_on_retained(r_expanded, pe)
    maxdiff = maximum(abs.(r_free_test .- r_free_back))
    println("trial $trial: max diff = $maxdiff")
    @assert maxdiff < 1e-10
end
println("PASS")

println("\n" * "="^78); println("TEST 4: gravity residual EXACTLY zero at random composed points"); println("="^78)
for trial in 1:5
    r_free_test = randn(rng, n_free_composed) .* 0.3
    z_full = decode_full_z_on_retained(r_free_test, pe)
    g = gravity_from_logz(z_full, ctx)
    println("trial $trial: gravity residual = $g")
    @assert abs(g) < 1e-9 "gravity residual not at machine precision for composed anchor+pivot point"
end
println("PASS -- anchor reduction + gravity pivot compose exactly")

println("\n" * "="^78); println("TEST 5: anchor cells remain fixed at gauge under every composed point"); println("="^78)
for trial in 1:3
    r_free_test = randn(rng, n_free_composed) .* 0.3
    z_full = decode_full_z_on_retained(r_free_test, pe)
    for d in 1:Ddest
        @assert z_full[spec.anchor_origin[d], d] == gauge[d] "anchor cell moved under composed decode"
    end
end
println("PASS")

println("\n" * "="^78); println("TEST 6: end-to-end -- composed point feeds build_compressed_factual cleanly (no throw)"); println("="^78)
r_free_test = randn(rng, n_free_composed) .* 0.2
z_full = decode_full_z_on_retained(r_free_test, pe)
θ_test = copy(θ0)
θ_test[Aod_offset+1:Aod_offset+D^2] .= vec(exp.(z_full))
build_compressed_factual(θ_test, ctx; check_ties = false)
println("build_compressed_factual evaluated cleanly at a fully-composed (anchor+gravity-pivot) point")
println("PASS")

println("\n" * "="^78); println("ALL TESTS PASSED"); println("="^78)
