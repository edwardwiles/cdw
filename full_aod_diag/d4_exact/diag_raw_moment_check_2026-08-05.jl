# 2026-08-05: pure-arithmetic raw-moment check, REDONE after finding+fixing the raw-U-vs-Frechet-z
# bug (precalc_common_marginals_cdf's eq.36 feature now uses z=U^(-mu), via frechet_power_feature,
# instead of raw U^(1-sigma)). NO KNITRO SOLVE ANYWHERE in this script -- context construction only
# (d20_real_setup builds the PsiObjectiveBundleImplicit struct but does not call KN_solve), then
# plain array arithmetic.
const D4X = @__DIR__
for f in ["draw_design.jl","context_real_d20.jl","winners.jl","oracle.jl","common_marginals_moments.jl"]
    include(joinpath(D4X, f))
end
using Statistics, Printf

W = 5_000
ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
D = ctx.D
σ = ctx.σ
μ = ctx.μHat
refIndex1 = ctx.γ.refIndex1
U = ctx.U
println("D=$D  σ=$σ  μ=$μ  refIndex1=$refIndex1  size(U)=$(size(U))")
println("U column summaries (min, 1%ile, mean, max) per origin (first 6 + refIndex1):")
for o in vcat(1:6, refIndex1)
    col = U[:, o]
    @printf("  o=%2d  min=%.6g  p1=%.6g  mean=%.6g  max=%.6g\n", o, minimum(col), quantile(col,0.01), mean(col), maximum(col))
end

pw = 1 - σ   # z-space exponent k = 1-sigma
zexp = μ * (σ - 1)   # CORRECT exponent applied directly to U: z^(1-σ) = U^(-μ(1-σ)) = U^(μ(σ-1))
println("\nz-space exponent (1-σ) = $pw ; CORRECTED U-space exponent μ(σ-1) = $zexp (was WRONGLY (1-σ)=$pw before the fix)")
zpow = U .^ zexp   # independent hand computation, NOT calling frechet_power_feature
println("z^(1-σ)=U^(μ(σ-1)) column summaries (min, mean, max) per origin (first 6 + refIndex1):")
for o in vcat(1:6, refIndex1)
    col = zpow[:, o]
    @printf("  o=%2d  min=%.6g  mean=%.6g  max=%.6g  any_inf=%s  any_nan=%s\n",
            o, minimum(col), mean(col), maximum(col), any(isinf,col), any(isnan,col))
end

L = 10
probs = collect(range(1/L, (L-1)/L, length=L))
z = quantile(U[:, refIndex1], probs)
println("\nz cutoffs (L=$L): ", z)

CM, z2, origins = precalc_common_marginals_cdf(U, refIndex1, L; include_truncated_moment = true, σHat = σ, μHat = μ, contrasts = :anchored, probs = probs)
@assert z2 == z
nO = length(origins)

println("\n" * "="^100)
println("Raw (uniform-weight) moment comparison: implementation column-mean vs INDEPENDENT hand-computed, per (o,l)")
println("="^100)
@printf("%-4s %-4s | %14s %14s %10s | %14s %14s %10s\n",
        "o", "l", "CDF_impl_mean", "CDF_hand", "match?", "POW_impl_mean", "POW_hand", "match?")
for (oi, o) in enumerate(origins[1:min(5,nO)])
    for l in (1, div(L,2), L)
        col_cdf = (l-1)*nO + oi
        col_pow = nO*L + (l-1)*nO + oi
        impl_cdf = mean(CM[:, col_cdf])
        hand_cdf = mean(U[:,o] .<= z[l]) - mean(U[:,refIndex1] .<= z[l])
        impl_pow = mean(CM[:, col_pow])
        hand_pow = mean(zpow[:,o] .* (U[:,o] .<= z[l])) - mean(zpow[:,refIndex1] .* (U[:,refIndex1] .<= z[l]))
        @printf("%-4d %-4d | %14.6g %14.6g %10s | %14.6g %14.6g %10s\n",
                o, l, impl_cdf, hand_cdf, isapprox(impl_cdf,hand_cdf;atol=1e-12) ? "EXACT" : "MISMATCH",
                impl_pow, hand_pow, isapprox(impl_pow,hand_pow;atol=1e-9,rtol=1e-9) ? "EXACT" : "MISMATCH")
    end
end

println("\n" * "="^100)
println("Magnitude comparison across ALL (o,l): is the POW residual now well-behaved (same order as CDF), not divergent?")
println("="^100)
cdf_cols = 1:(nO*L)
pow_cols = (nO*L+1):(2*nO*L)
cdf_means = [mean(CM[:,j]) for j in cdf_cols]
pow_means = [mean(CM[:,j]) for j in pow_cols]
@printf("CDF raw moment (uniform weight):  max|mean|=%.6g  mean(|mean|)=%.6g\n", maximum(abs, cdf_means), mean(abs, cdf_means))
@printf("POW raw moment (uniform weight):  max|mean|=%.6g  mean(|mean|)=%.6g\n", maximum(abs, pow_means), mean(abs, pow_means))
@printf("Ratio of scales: mean(|POW|)/mean(|CDF|) = %.6g  (WAS ~610,000 before the fix -- should now be a modest, single-digit-to-low-double-digit ratio)\n", mean(abs,pow_means)/mean(abs,cdf_means))

println("\nFor context, mean(z^(1-σ)) magnitude across origins (this is the 'unit scale' the POW residual should be judged against; should be a modest, similar-across-origins number now, not wildly different by orders of magnitude):")
for o in vcat(origins[1:min(5,nO)], refIndex1)
    @printf("  o=%2d  mean(z^(1-σ))=%.6g\n", o, mean(zpow[:,o]))
end
