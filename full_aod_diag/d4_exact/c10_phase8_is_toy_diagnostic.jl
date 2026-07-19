# ============================================================================
# Continuation 10, Phase 8 (importance-sampling ASSESSMENT ONLY -- per task
# instruction, this is NOT integrated into the real D=20 pipeline, and is a
# strictly separate, higher-risk investigation from Phase 7's QMC work. Toy/
# reduced-D prototype: D_TOY origins, one destination, standalone script.
# Reuses the REAL cc_algo/Psi.jl divergence functions (Psi!/dPsi!/ddPsi!,
# exactly what the production dual objective calls) -- not reimplemented --
# but everything else (the toy "price"/"winner" model, the toy dual-like
# statistic) is a deliberately simplified stand-in for the real D=20 model,
# built only to probe IS mechanics (ESS, weight tails, conditioning), not to
# produce any number that could be mistaken for a real kappa/Delta result.
#
# Question this answers: if we deliberately oversample a rare-winner origin's
# shock (the kind of bilateral winner Phase 7's zero-winner-incidence
# measurements can identify as under-represented under vanilla MC), what does
# the resulting importance-weight distribution look like -- effective sample
# size, tail heaviness, and its effect on a Hessian-like second-moment matrix
# built from the (correctly Psi!-weighted) per-draw statistic?
# ============================================================================
using Random, Statistics, LinearAlgebra, Printf
const D4X_ROOT3 = dirname(dirname(@__DIR__))
include(joinpath(D4X_ROOT3, "cc_algo", "Psi.jl"))   # -> Psi!, dPsi!, ddPsi! (the REAL divergence functions)

println("="^80); println("Phase 8: toy importance-sampling diagnostic"); println("="^80)

const D_TOY = 6
const W_TOY = 20_000
const MU = 0.3
Random.seed!(20260719)

# ---- toy "economy": destination-specific trade costs c_o (o* is the expensive/rare-winner one) ----
c = [1.0, 1.05, 1.1, 1.15, 1.2, 3.0]   # origin 6 (o*=6) is much more expensive -> rarely wins
const OSTAR = 6

price(U::AbstractVector) = c ./ (U .^ MU)   # smaller price wins; larger U -> smaller price (more competitive)

function winner_shares(Umat::Matrix{Float64})
    W = size(Umat, 1)
    counts = zeros(Int, D_TOY)
    for i in 1:W
        p = price(@view Umat[i, :])
        counts[argmin(p)] += 1
    end
    return counts ./ W
end

# ---- Baseline: plain MC from F* = Exp(1)^D_TOY (matches the real model's per-origin Exp(1) draws) ----
U_base = -log.(1.0 .- rand(W_TOY, D_TOY))
shares_base = winner_shares(U_base)
println("[1] Baseline (F*=Exp(1) MC) winner shares: ", round.(shares_base, digits=4))
println("    o*=", OSTAR, " (rare-winner origin) share=", shares_base[OSTAR])

# ---- IS proposal Q: identical to F* except origin o*'s column is Exp(rate=s), s<1 (mean 1/s>1,
# shifts mass to LARGER U_o* -> smaller price_o* -> o* wins more often). Importance weight per
# draw is the likelihood ratio for JUST that one column (all others are drawn identically under
# F* and Q, so they contribute a ratio of 1 and cancel) -- w_i = f*(u_i)/q(u_i) = exp(-u_i) /
# (s*exp(-s*u_i)).
const S_RATE = 0.15   # mean 1/s ~= 6.67 under Q vs mean 1 under F*
U_is = copy(U_base)   # reuse origins 1:D_TOY-1 draws for a fair, common-random-number-style comparison
U_is[:, OSTAR] = -log.(1.0 .- rand(W_TOY)) ./ S_RATE   # Exp(rate=S_RATE): -log(1-u)/S_RATE
w_is = exp.(-U_is[:, OSTAR]) ./ (S_RATE .* exp.(-S_RATE .* U_is[:, OSTAR]))

shares_is_raw = winner_shares(U_is)
println("\n[2] IS proposal (o*'s shock boosted, unweighted raw shares under Q): ", round.(shares_is_raw, digits=4))
println("    o* share under Q (should be MUCH higher than under F*): ", shares_is_raw[OSTAR])

# reweighted estimate of the ORIGINAL F*-shares using importance weights: should recover shares_base
function winner_shares_weighted(Umat::Matrix{Float64}, w::Vector{Float64})
    W = size(Umat, 1)
    wsum = zeros(D_TOY)
    for i in 1:W
        p = price(@view Umat[i, :])
        wsum[argmin(p)] += w[i]
    end
    return wsum ./ sum(w)
end
shares_is_reweighted = winner_shares_weighted(U_is, w_is)
println("    o* share, IS-REWEIGHTED (should match baseline F* share, ", shares_base[OSTAR], "): ", shares_is_reweighted[OSTAR])
println("    all-origin max abs diff (reweighted-IS vs baseline-MC): ", maximum(abs.(shares_is_reweighted .- shares_base)),
        " (unbiasedness check -- small diff confirms the IS weight formula is correctly applied)")

# ---- Effective sample size + weight tail diagnostics ----
ESS = sum(w_is)^2 / sum(w_is .^ 2)
ess_frac = ESS / W_TOY
tail_ratio = maximum(w_is) / mean(w_is)
sorted_w = sort(w_is, rev = true)
top1pct_share = sum(sorted_w[1:cld(W_TOY, 100)]) / sum(w_is)
println("\n[3] Importance-weight diagnostics:")
@printf("    ESS = %.1f / %d draws (%.2f%% of nominal W)\n", ESS, W_TOY, 100*ess_frac)
@printf("    max(w)/mean(w) = %.2f  (tail heaviness -- 1.0 would be uniform/no variance inflation)\n", tail_ratio)
@printf("    top-1%%-of-draws share of total weight = %.2f%%  (vs 1%% under uniform weights)\n", 100*top1pct_share)

# ---- Toy "Hessian" conditioning: build a per-draw feature vector x_i = [1, U_i,o*] and a toy
# dual-like second-derivative weight ddPsi!(arg0_i) using the REAL cc_algo/Psi.jl ddPsi! (not
# reimplemented) at an arbitrary interior arg0 (a stand-in "k+zeta+lambda'g" combination scaled by
# -eta) -- then compare the condition number of (1/W) sum w_i * d2_i * x_i x_i' (IS-weighted) vs
# the unweighted plain-MC analogue (1/W) sum d2_i * x_i x_i' (from F*-baseline draws), i.e. does
# the IS weighting concentrate curvature mass onto a few draws and worsen conditioning.
function toy_hessian(Umat::Matrix{Float64}, weights::Union{Nothing,Vector{Float64}})
    W = size(Umat, 1)
    arg0 = 0.3 .* (Umat[:, OSTAR] .- mean(Umat[:, OSTAR]))   # arbitrary interior stand-in argument
    d2 = zeros(W); ddPsi!(d2, arg0)
    H = zeros(2, 2)
    for i in 1:W
        x = [1.0, Umat[i, OSTAR]]
        wi = weights === nothing ? 1.0 : weights[i]
        H .+= wi .* d2[i] .* (x * x')
    end
    H ./= (weights === nothing ? W : sum(weights))
    return H
end
H_plain = toy_hessian(U_base, nothing)
H_is    = toy_hessian(U_is, w_is)
cond_plain = cond(H_plain)
cond_is = cond(H_is)
println("\n[4] Toy Hessian conditioning (2x2, feature=[1, U_o*]):")
println("    plain-MC (F*, uniform weights): cond=", round(cond_plain, digits=3), "  H=", H_plain)
println("    IS-weighted (Q, importance weights): cond=", round(cond_is, digits=3), "  H=", H_is)
@printf("    conditioning ratio (IS/plain) = %.2fx  (>1 means IS weighting worsened conditioning here)\n", cond_is/cond_plain)

println("\n", "="^80)
println("SUMMARY: ESS_frac=", round(ess_frac, digits=4), " tail_ratio=", round(tail_ratio, digits=2),
        " top1pct_weight_share=", round(top1pct_share, digits=4),
        " unbiasedness_max_abs_diff=", round(maximum(abs.(shares_is_reweighted .- shares_base)), digits=5),
        " cond_ratio=", round(cond_is/cond_plain, digits=3))
println("="^80)
