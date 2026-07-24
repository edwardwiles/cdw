# ============================================================================
# Explicit hard-winner recovery for the full-A D=4 factual moment system.
#
# moments/hFunction.jl never returns winner indices directly -- it only fills
# G using an inline MinInd! call whose result (pricesInd, a 0/1 indicator
# vector) is immediately discarded per draw. Sections 11-12 of the task brief
# need the actual winner(o,d,draw) array (tie thresholds, switch counts,
# switch mass), so this file REPLICATES hFunction!'s exact price formula
# (UoModel==1 branch only, matching this investigation's fixed AD_PARAMS) to
# recover it, and is validated against hFunction!'s own G output below (not
# assumed correct by construction).
# ============================================================================

"""
    factual_prices(θ_full, ctx) -> (price::Array{Float64,3}, Aod::Matrix, AodPow::Matrix)

`price[ω, o, d]` = origin o's factual delivered price to destination d on draw ω
(UoModel==1: origin draws only, `U[ω,o]`, no destination subscript). Matches
`moments/hFunction.jl`'s `pricesTemp[o] = constCons[o,d] / UPow[ω,o]` exactly.
"""
function factual_prices(θ_full::AbstractVector, ctx)
    @unpack wHat, τ, cHat, P = ctx.γ
    D = ctx.D; U = ctx.U; W = size(U, 1)
    Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D   # Part A, 2026-07-23
    μ = θ_full[1]
    lambda = reshape(P, (Ddest, D))'
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], (D, Ddest))
    Aod = Aod_θ .* cHat .* (((wHat .* τ) ./ (wHat[1, 1] .* τ[1, :]')) .^ (1 / μ)) .* (lambda ./ lambda[1, :]')
    AodPow = (Aod ./ cHat) .^ (-μ)
    constCons = [wHat[o] * AodPow[o, d] * τ[o, d] for o in 1:D, d in 1:Ddest]
    UPow = U .^ (-μ)   # W x D, θConstant != 1 branch (matches EK_moments_gammanorm_directgp!)
    price = Array{Float64}(undef, W, D, Ddest)
    @inbounds for d in 1:Ddest, o in 1:D, ω in 1:W
        price[ω, o, d] = constCons[o, d] / UPow[ω, o]
    end
    return price, Aod, AodPow
end

"""
    compute_winners(θ_full, ctx) -> (winner::Matrix{Int}, price::Array{Float64,3}, gap::Matrix{Float64})

`winner[ω,d]` = argmin_o price[ω,o,d] (the origin serving destination d on draw
ω under the FACTUAL competitiveness matrix). `gap[ω,d]` = price of the runner-up
minus price of the winner (>=0; the literal tie-threshold-relevant quantity
used by section 11's diagnostics).
"""
function compute_winners(θ_full::AbstractVector, ctx)
    price, Aod, AodPow = factual_prices(θ_full, ctx)
    W = size(price, 1); Ddest = size(price, 3)
    winner = Matrix{Int}(undef, W, Ddest)
    gap = Matrix{Float64}(undef, W, Ddest)
    @inbounds for d in 1:Ddest, ω in 1:W
        col = @view price[ω, :, d]
        wmin, wo = findmin(col)
        s = sort(col)
        gap[ω, d] = length(s) > 1 ? (s[2] - s[1]) : Inf
        winner[ω, d] = wo
    end
    return winner, price, gap
end

"""
    validate_winners_against_hFunction(θ_full, ctx; W_check=200) -> Bool

Cross-check: recompute G[:, d1] from `compute_winners`' own winner array using
the SAME `pricesTempσ`/`denom` formula hFunction.jl uses, and compare to the
ACTUAL G returned by `EK_moments_gammanorm_directgp!` for the same theta. This
validates the replicated price/winner logic is correct, not merely
plausible -- required before trusting any hard-winner diagnostic built on it.
"""
function validate_winners_against_hFunction(θ_full::AbstractVector, ctx; rtol = 1e-10)
    D = ctx.D; U = ctx.U; W = size(U, 1)
    Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D   # Part A, 2026-07-23
    K = zeros(W); G = zeros(W, ctx.nTotalMoments)
    EK_moments_gammanorm_directgp!(K, G, θ_full, U, ctx.obj)

    @unpack wHat, τ, cHat, P = ctx.γ
    σ = θ_full[2]; μ = θ_full[1]
    lambda = reshape(P, (Ddest, D))'
    winner, price, gap = compute_winners(θ_full, ctx)
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], (D, Ddest))
    Aod = Aod_θ .* cHat .* (((wHat .* τ) ./ (wHat[1, 1] .* τ[1, :]')) .^ (1 / μ)) .* (lambda ./ lambda[1, :]')
    AodPow = (Aod ./ cHat) .^ (-μ)
    wPow = wHat .^ (1 - σ)
    constConsσ = [wPow[o] * (AodPow[o, d] * τ[o, d])^(1 - σ) for o in 1:D, d in 1:Ddest]
    # hFunction!'s "Uσ" parameter actually RECEIVES UσPow = ctx.γ.Uσ .^ (-μ) from the caller
    # (EK_moments_gammanorm_directgp!'s θConstant!=1 branch) -- not ctx.γ.Uσ directly.
    UσPow = ctx.γ.Uσ .^ (-μ)
    denom = [1.0^σ * (wHat[d] * ctx.γ.L[d]) for d in 1:Ddest]  # gamma[d]==1 under this gauge

    # EK_moments_gammanorm_directgp! applies TWO post-hFunction! rescalings this replication must
    # match: (1) division by gamma(mu*(1-sigma)+1) for every trade-share/price-index column
    # (theta_constant!=1 branch), (2) per-draw multiplication by SamplingWeights. Both are simple
    # scalars/vectors, not part of the winner-selection logic itself, but must be applied for a
    # bit-exact cross-check against the real G.
    gammafac = SpecialFunctions.gamma(μ * (1 - σ) + 1)
    SW = ctx.γ.SamplingWeights

    maxerr = 0.0
    @inbounds for d in 1:Ddest, ω in 1:W
        o = winner[ω, d]
        d1 = d + (o - 1) * Ddest
        pTσ = constConsσ[o, d] / UσPow[ω, o]
        predicted = ((pTσ - P[d1] * denom[d]) / gammafac) * SW[ω]
        maxerr = max(maxerr, abs(predicted - G[ω, d1]))
    end
    return maxerr, maxerr < rtol
end
