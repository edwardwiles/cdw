# ============================================================================
# Task §16 / theory doc section 2.2 numerical gate: exact full-A recovery.
# ADDITIVE ONLY -- see recover_full_a_2026-07-31.jl header.
#
# CORRECTED 2026-07-31 (same day, user stop): Test 2 originally called the
# LEGACY dense `ctx.obj.moments!` (G/K matrix) directly. Rebuilt to use
# `destination_Q_od`/`destination_M_d` (both already corrected to use
# `build_compressed_factual`) -- see recover_full_a_2026-07-31.jl's header
# for the full correction rationale.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "recover_full_a_2026-07-31.jl"))
using Random, Statistics

ctx = d4_exact_setup()
D = ctx.D; Ddest = D
Aod_offset = ctx.Aod_offset
θ0 = copy(ctx.θ0_up)
rng = MersenneTwister(31072026)

spec = default_anchor_spec(D, Ddest; overrides = Dict(3 => 1))
Aod_θ0 = reshape(θ0[Aod_offset+1:Aod_offset+D^2], (D, D))
z_calib = log.(Aod_θ0)
gauge = build_anchor_gauge(z_calib, spec)
r_calib = encode_relative_A(z_calib, spec, gauge)

println("="^78); println("SETUP: build a working-gauge (NOT gamma-normalized) point"); println("="^78)
r_perturbed = r_calib .+ 0.4 .* randn(rng, n_retained(spec))   # a real, non-calibration perturbation
z_working = decode_relative_A(r_perturbed, spec, gauge)
θ_working = copy(θ0)
θ_working[Aod_offset+1:Aod_offset+D^2] .= vec(exp.(z_working))

M_before = destination_M_d(θ_working, ctx)
gamma_tilde_before = Dict(d => mean(M_before[d]) / (ctx.γ.wHat[d] * ctx.γ.L[d]) for d in 1:D)
println("gamma_tilde BEFORE recovery (expect scattered away from 1, since this is a random perturbation):")
for d in 1:D
    println("  d=$d: gamma_tilde = $(gamma_tilde_before[d])")
end

println("\n" * "="^78); println("TEST 1: recovery drives gamma_tilde to 1 for every destination"); println("="^78)
z_full, c, gamma_tilde_computed = recover_gamma_normalized_full_A(θ_working, ctx)
θ_recovered = copy(θ0)
θ_recovered[Aod_offset+1:Aod_offset+D^2] .= vec(exp.(z_full))
M_after = destination_M_d(θ_recovered, ctx)
maxdev = 0.0
for d in 1:D
    gt_after = mean(M_after[d]) / (ctx.γ.wHat[d] * ctx.γ.L[d])
    println("  d=$d: c[d]=$(c[d])  gamma_tilde_predicted(before recovery)=$(gamma_tilde_computed[d])  gamma_tilde_AFTER=$gt_after")
    global maxdev = max(maxdev, abs(gt_after - 1.0))
end
println("max|gamma_tilde_after - 1| over all d = $maxdev")
@assert maxdev < 1e-8 "TEST 1 FAILED: recovery did not drive gamma_tilde to 1"
println("PASS -- recovery is exact (not just approximately gamma-normalized)")

println("\n" * "="^78); println("TEST 2: winner identities and share ratios unchanged by recovery (per theory section 2.1)"); println("="^78)
maxratiodiff = 0.0
for d in 1:D
    Qw = destination_Q_od(θ_working, ctx, d)
    Qr = destination_Q_od(θ_recovered, ctx, d)
    Mw = vec(sum(Qw, dims = 2)); Mr = vec(sum(Qr, dims = 2))
    winw = [argmax(@view Qw[w, :]) for w in 1:size(Qw, 1)]
    winr = [argmax(@view Qr[w, :]) for w in 1:size(Qr, 1)]
    nmis = sum(winw .!= winr)
    rdiff = maximum(abs.((Qw ./ Mw) .- (Qr ./ Mr)))
    global maxratiodiff = max(maxratiodiff, rdiff)
    println("  d=$d: winner mismatches=$nmis  max ratio diff=$rdiff")
    @assert nmis == 0 "TEST 2 FAILED at d=$d: winner identity changed by recovery"
end
@assert maxratiodiff < 1e-6 "TEST 2 FAILED: share ratios changed by recovery"
println("PASS -- recovery changes only the destination-column SCALE, nothing economically meaningful")

println("\n" * "="^78); println("ALL TESTS PASSED"); println("="^78)
