# 2026-08-05 (user-directed, beyond original task brief): validate the theoretical-Frechet-
# quantile cutoff derivation (theoretical_u_threshold) and the closed-form truncated-moment
# formula (eq36_theoretical_truncated_moment) via large-W Monte Carlo convergence -- NO KNITRO,
# NO context construction needed at all, pure synthetic Exp(1) draws + arithmetic. This is the
# critical empirical check on the sign/direction derivation in theoretical_u_threshold's own
# docstring (a subtle decreasing-transform argument) -- if the derivation had a sign error, this
# convergence check would fail cleanly.
const D4X = @__DIR__
include(joinpath(D4X, "context.jl"))
include(joinpath(D4X, "winners.jl"))
include(joinpath(D4X, "oracle.jl"))
include(joinpath(D4X, "common_marginals_moments.jl"))
using Random, Statistics, Printf

Random.seed!(20260805)
μ = 0.15
σ = 2.5
k = 1 - σ   # eq.36's own exponent

println("="^100)
println("Check 1: eq36_theoretical_truncated_moment reduces to the untruncated Γ(1-μk) as z_ℓ→∞")
println("="^100)
for zl in (10.0, 100.0, 1e4, 1e8, 1e15)
    val = eq36_theoretical_truncated_moment(zl, k, μ)
    @printf("  z_ℓ=%.0e  ->  %.10f\n", zl, val)
end
println("  Γ(1-μk) untruncated = ", gamma(1 - μ*k))

println("\n" * "="^100)
println("Check 2: Monte Carlo convergence, several (W, L, l) combinations")
println("="^100)
for W in (10_000, 1_000_000, 20_000_000)
    U = -log.(rand(W))   # Exp(1)
    for p in (0.1, 0.5, 0.9)
        u_thresh = theoretical_u_threshold(p)
        z = U .^ (-μ)                 # true Frechet productivity draw
        zl = u_thresh^(-μ)            # the Frechet-space cutoff this p/threshold corresponds to
        # Direct Monte Carlo estimate of E[z^k * 1{z<z_l}], computed INDEPENDENTLY of the
        # existing-code-structure sign-flip trick (literal population definition, in z-space):
        mc_direct = mean((z .^ k) .* (z .< zl))
        # The EXISTING-code-structure computation (1{U<=u_thresh}, weight z^k) -- what
        # precalc_common_marginals_cdf ACTUALLY computes internally (same sign as the real code):
        mc_codepath = mean((z .^ k) .* (U .<= u_thresh))
        theo = eq36_theoretical_truncated_moment(zl, k, μ)
        @printf("  W=%8d p=%.1f  u_thresh=%.6f  z_l=%.6f | MC(direct z<zl)=%.6f  MC(code U<=u)=%.6f  theory=%.6f  |diff_direct|=%.2e  |diff_code|=%.2e\n",
                W, p, u_thresh, zl, mc_direct, mc_codepath, theo, abs(mc_direct-theo), abs(mc_codepath-theo))
    end
end

println("\n" * "="^100)
println("Check 3: full precalc_common_marginals_cdf raw feature (uniform-weight mean) vs closed form, at moderate W/L")
println("="^100)
W = 2_000_000
D = 4
refIndex1 = 1
Random.seed!(4242)
U2 = -log.(rand(W, D))
L = 5
CM, z, origins = precalc_common_marginals_cdf(U2, refIndex1, L; include_truncated_moment = true, σHat = σ, μHat = μ, contrasts = :anchored)
@printf("  z cutoffs (theoretical): %s\n", z)
nO = length(origins)
tm_offset = nO*L
for l in 1:L
    zl = z[l]^(-μ)
    theo = eq36_theoretical_truncated_moment(zl, k, μ)
    for (oi, o) in enumerate(origins)
        col_pow = tm_offset + (l-1)*nO + oi
        col_cdf = (l-1)*nO + oi
        # CM's power column is a CONTRAST/anchored DIFFERENCE (origin minus reference) -- both
        # sides should independently converge to the SAME theo value (exchangeable origins), so
        # the raw column mean should be close to 0 (difference of two things near `theo`), while
        # E[z_o^k*1{z_o<zl}] alone (not differenced) should match `theo` directly -- check both.
        raw_o = mean((U2[:,o].^(-μ)).^k .* (U2[:,o].^(-μ) .< zl))
        @printf("    l=%d o=%d: E[z_o^k*1{z_o<zl}]=%.6f  theory=%.6f  |diff|=%.2e | CM_pow_col_mean(diff)=%.6f  CM_cdf_col_mean(diff)=%.6f\n",
                l, o, raw_o, theo, abs(raw_o-theo), mean(CM[:,col_pow]), mean(CM[:,col_cdf]))
    end
end
