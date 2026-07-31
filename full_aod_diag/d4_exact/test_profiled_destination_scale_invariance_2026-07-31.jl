# ============================================================================
# Task §2.1/§2.2/§2.1(c) numerical falsification gate, D=4.
#
# ADDITIVE ONLY (repo convention): does not modify context.jl, moments_gammanorm.jl,
# gravity_elimination.jl, or any other trusted production file. Calls the EXISTING,
# already-validated `ctx.obj.moments!` (== EK_moments_gammanorm_directgp!, confirmed
# in context.jl) and `gravity_elimination.jl` functions exactly as production does,
# at the GENUINE calibration point `ctx.θ0_up` (NOT the gravity-elimination pivot's
# `zfree=0` reference point -- see this repo's standing CLAUDE.md warning) perturbed
# by an explicit, documented destination-column rescale.
#
# What this tests, from PROFILED_DESTINATION_SCALE_THEORY_2026-07-31.md:
#   (A) winner identities unchanged under a common rescale of one destination's A column
#   (B) factual per-draw share ratios Q_od(w)/M_d(w) exactly unchanged (theory §2.1a/b)
#   (C) M_d(w) rescales by exactly kappa^(mu*(sigma-1)) for EVERY draw w (the exact
#       homogeneity exponent claim, theory §0/§2.1/§2.2)
#   (D) gravity residual exactly unchanged (theory §2.1c, closed-form proof)
#   (E) whether E_F[M_d_raw]/denom[d] == 1 at genuine calibration (checks whether the
#       task's "E_F[M_d]=1" normalization already holds automatically at theta0_up,
#       informing whether it is a live constraint being profiled out or already-implied)
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
using SpecialFunctions: gamma as spgamma
using Random, Statistics, LinearAlgebra

ctx = d4_exact_setup()
D = ctx.D
Aod_offset = ctx.Aod_offset
θ0 = copy(ctx.θ0_up)
μ = θ0[1]; σ = θ0[2]
W = size(ctx.U, 1)

println("="^78); println("SETUP"); println("="^78)
println("D=$D  mu=$μ  sigma=$σ  W=$W  Aod_offset=$Aod_offset  baseIndex=$(ctx.bi)")

d_test = 2  # arbitrary non-baseIndex destination to shift; re-run with d_test=ctx.bi separately below
κ = 1.7     # destination-column rescale factor (Aod_theta[:,d_test] *= kappa)
e_exponent = μ * (σ - 1)
println("Testing destination d_test=$d_test with rescale kappa=$κ, predicted exponent mu*(sigma-1)=$e_exponent")

# ---- baseline evaluation at GENUINE calibration (theta0_up itself, not zfree=0) ----
K0 = zeros(W); G0 = zeros(W, ctx.nTotalMoments)
ctx.obj.moments!(K0, G0, θ0, ctx.U, ctx.obj)

# ---- shifted evaluation: multiply Aod_theta[:, d_test] (LEVEL) by kappa, everything else fixed ----
θ1 = copy(θ0)
for o in 1:D
    idx = Aod_offset + (d_test - 1) * D + o
    θ1[idx] *= κ
end
K1 = zeros(W); G1 = zeros(W, ctx.nTotalMoments)
ctx.obj.moments!(K1, G1, θ1, ctx.U, ctx.obj)

# ---- reconstruct Q_od(w), M_d(w) from G at both points ----
lambda = reshape(ctx.γ.P, (D, D))'          # lambda[o,d], data
denom_d = ctx.γ.wHat[d_test] * ctx.γ.L[d_test]  # denom[d] under gamma_d==1 (forced in this formulation)
gammafac = spgamma(μ * (1 - σ) + 1)
SW = ctx.γ.SamplingWeights[1:W]
uniform_SW = maximum(abs.(SW .- SW[1])) < 1e-12
println("SamplingWeights uniform: $uniform_SW (value=$(SW[1]))")
wscale = SW ./ gammafac   # per-draw scalar; == a single constant if SW uniform

function extract_Q_M(G, d, D, lambda, denom_d, wscale)
    W = size(G, 1)
    Q = zeros(W, D)
    for o in 1:D
        d1 = d + (o - 1) * D
        @. Q[:, o] = G[:, d1] / wscale + lambda[o, d] * denom_d
    end
    M = vec(sum(Q, dims = 2))
    return Q, M
end

Q0, M0 = extract_Q_M(G0, d_test, D, lambda, denom_d, wscale)
Q1, M1 = extract_Q_M(G1, d_test, D, lambda, denom_d, wscale)

println("\n" * "="^78); println("TEST A: winner identity unchanged (argmax_o Q_od(w) same o, every draw)"); println("="^78)
winner0 = [argmax(@view Q0[w, :]) for w in 1:W]
winner1 = [argmax(@view Q1[w, :]) for w in 1:W]
n_mismatch = sum(winner0 .!= winner1)
println("mismatches out of $W draws: $n_mismatch")
@assert n_mismatch == 0 "TEST A FAILED: winner identity changed under destination-column rescale"
println("PASS")

println("\n" * "="^78); println("TEST B: per-draw share ratio Q_od(w)/M_d(w) exactly invariant"); println("="^78)
ratio0 = Q0 ./ M0
ratio1 = Q1 ./ M1
maxdiff = maximum(abs.(ratio0 .- ratio1))
println("max|ratio1 - ratio0| over all (w,o) = $maxdiff")
@assert maxdiff < 1e-8 "TEST B FAILED: share ratios not invariant"
println("PASS")

println("\n" * "="^78); println("TEST C: M_d(w) rescales by EXACTLY kappa^(mu*(sigma-1)) for every draw"); println("="^78)
predicted = κ^e_exponent
empirical_ratio = M1 ./ M0
println("predicted M1/M0 = $predicted")
println("empirical M1/M0: min=$(minimum(empirical_ratio))  max=$(maximum(empirical_ratio))  mean=$(mean(empirical_ratio))")
maxdiff_C = maximum(abs.(empirical_ratio .- predicted))
println("max|empirical - predicted| = $maxdiff_C")
@assert maxdiff_C < 1e-6 "TEST C FAILED: homogeneity exponent mu*(sigma-1) does not match production code"
println("PASS -- exponent mu*(sigma-1) confirmed numerically, not just from reading code")

println("\n" * "="^78); println("TEST D: gravity residual exactly unchanged under the same shift"); println("="^78)
Aod_θ0 = reshape(θ0[Aod_offset+1:Aod_offset+D^2], (D, D))
Aod_θ1 = reshape(θ1[Aod_offset+1:Aod_offset+D^2], (D, D))
z0 = log.(Aod_θ0); z1 = log.(Aod_θ1)
g0 = gravity_from_logz(z0, ctx)
g1 = gravity_from_logz(z1, ctx)
println("g_gravity(base) = $g0")
println("g_gravity(shifted) = $g1")
println("abs diff = $(abs(g1 - g0))")
@assert abs(g1 - g0) < 1e-9 "TEST D FAILED: gravity residual changed under a pure destination-column shift"
println("PASS -- confirms theory doc section 2.1(c)'s closed-form proof numerically")

println("\n" * "="^78); println("TEST E: is E_F[M_d_raw]/denom[d] == 1 at genuine calibration (theta0_up)?"); println("="^78)
EF_M0_over_denom = mean(M0) / denom_d
println("mean(M_d(w)) / denom[d] at genuine calibration = $EF_M0_over_denom  (task's 'E_F[M_d]=1' claim)")
println("(informational -- not asserted; see script header note E)")

println("\n" * "="^78); println("Re-running Tests A-D for d_test = baseIndex ($(ctx.bi)) to confirm the France-analog destination too"); println("="^78)
d_test2 = ctx.bi
θ2 = copy(θ0)
for o in 1:D
    idx = Aod_offset + (d_test2 - 1) * D + o
    θ2[idx] *= κ
end
K2 = zeros(W); G2 = zeros(W, ctx.nTotalMoments)
ctx.obj.moments!(K2, G2, θ2, ctx.U, ctx.obj)
denom_d2 = ctx.γ.wHat[d_test2] * ctx.γ.L[d_test2]
Q0b, M0b = extract_Q_M(G0, d_test2, D, lambda, denom_d2, wscale)
Q2, M2 = extract_Q_M(G2, d_test2, D, lambda, denom_d2, wscale)
winner0b = [argmax(@view Q0b[w, :]) for w in 1:W]
winner2 = [argmax(@view Q2[w, :]) for w in 1:W]
@assert sum(winner0b .!= winner2) == 0 "baseIndex destination: winner identity changed"
maxdiff_b = maximum(abs.((Q0b ./ M0b) .- (Q2 ./ M2)))
@assert maxdiff_b < 1e-8 "baseIndex destination: share ratios not invariant"
maxdiff_Cb = maximum(abs.((M2 ./ M0b) .- κ^e_exponent))
@assert maxdiff_Cb < 1e-6 "baseIndex destination: homogeneity exponent mismatch"
z2 = log.(reshape(θ2[Aod_offset+1:Aod_offset+D^2], (D, D)))
g2 = gravity_from_logz(z2, ctx)
@assert abs(g2 - g0) < 1e-9 "baseIndex destination: gravity changed"
println("PASS for baseIndex destination too (winner/ratio/exponent/gravity all confirmed)")

println("\n" * "="^78); println("ALL TESTS PASSED"); println("="^78)
