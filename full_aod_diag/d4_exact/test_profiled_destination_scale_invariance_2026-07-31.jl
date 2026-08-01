# ============================================================================
# Task §2.1/§2.2/§2.1(c) numerical falsification gate, D=4.
#
# CORRECTED 2026-07-31 (same day, user stop): the original version of this
# file read the LEGACY dense G/K moment matrix via `ctx.obj.moments!`, which
# only exists on the pre-hardening `PsiObjectiveBundleImplicit` bundle
# `d4_exact_setup()` happens to attach. This repo's production stack uses
# genuinely no-H/no-G/no-K `OperatorPsiBundle`s -- rebuilt to read
# `build_compressed_factual`'s `winner`/`wval`/`Pmat`/`denom` fields directly,
# the actual winner-compressed representation production uses (see
# recover_full_a_2026-07-31.jl's header for the full correction rationale).
# `ctx = d4_exact_setup()` is still used only as a DATA/economy builder
# (gamma, U, theta0_up, etc.) -- its dense `ctx.obj` is never read.
#
# ADDITIVE ONLY (repo convention): does not modify context.jl,
# compressed_moments.jl, gravity_elimination.jl, or any other trusted file.
# Evaluated at the GENUINE calibration point `ctx.θ0_up` (NOT the
# gravity-elimination pivot's `zfree=0` reference point -- see this repo's
# standing CLAUDE.md warning) perturbed by an explicit, documented
# destination-column rescale.
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
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
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

function winner_and_M(θ, ctx, d)
    cf = build_compressed_factual(θ, ctx; check_ties = false)
    s = dest_slot(ctx, d)
    return cf.winner[:, s], cf.wval[:, s]   # (winner origin per draw, M_d(w)=winning value per draw)
end

function Q_matrix(θ, ctx, d)
    D = ctx.D
    winner, wval = winner_and_M(θ, ctx, d)
    Q = zeros(length(winner), D)
    @inbounds for w in eachindex(winner)
        Q[w, winner[w]] = wval[w]
    end
    return Q
end

# ---- baseline evaluation at GENUINE calibration (theta0_up itself, not zfree=0) ----
winner0, M0 = winner_and_M(θ0, ctx, d_test)
Q0 = Q_matrix(θ0, ctx, d_test)

# ---- shifted evaluation: multiply Aod_theta[:, d_test] (LEVEL) by kappa, everything else fixed ----
θ1 = copy(θ0)
for o in 1:D
    idx = Aod_offset + (d_test - 1) * D + o
    θ1[idx] *= κ
end
winner1, M1 = winner_and_M(θ1, ctx, d_test)
Q1 = Q_matrix(θ1, ctx, d_test)

println("\n" * "="^78); println("TEST A: winner identity unchanged (same winning origin, every draw)"); println("="^78)
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
println("PASS -- exponent mu*(sigma-1) confirmed numerically against the compressed/operator-representative path")

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
println("(gravity_elimination.jl is unchanged/shared regardless of dense-G vs operator bundle -- it only")
println(" reads Aod_theta levels via gravity_from_logz, never touches G/H/K at all)")

println("\n" * "="^78); println("TEST E: is E_F[M_d_raw]/denom[d] == 1 at genuine calibration (theta0_up)?"); println("="^78)
cf0 = build_compressed_factual(θ0, ctx; check_ties = false)
s_test = dest_slot(ctx, d_test)
EF_M0_over_denom = mean(M0) / cf0.denom[s_test]
println("mean(M_d(w)) / denom[d] at genuine calibration = $EF_M0_over_denom  (task's 'E_F[M_d]=1' claim)")
println("(informational -- not asserted; see script header note E)")

println("\n" * "="^78); println("Re-running Tests A-D for d_test = baseIndex ($(ctx.bi)) to confirm the France-analog destination too"); println("="^78)
d_test2 = ctx.bi
θ2 = copy(θ0)
for o in 1:D
    idx = Aod_offset + (d_test2 - 1) * D + o
    θ2[idx] *= κ
end
winner0b, M0b = winner_and_M(θ0, ctx, d_test2)
Q0b = Q_matrix(θ0, ctx, d_test2)
winner2, M2 = winner_and_M(θ2, ctx, d_test2)
Q2 = Q_matrix(θ2, ctx, d_test2)
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
