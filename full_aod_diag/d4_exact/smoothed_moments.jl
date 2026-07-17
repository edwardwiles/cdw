# ============================================================================
# Task §12E-F: derivative-only / consistent smoothing. Builds a smoothed
# variant of the factual trade-share moments (the ONLY place MinInd!'s hard
# branch appears, per docs/fullA_d4_code_audit.md sec 6 -- the counterfactual
# focal price-index moment is already smooth, no argmin at all under
# counterType==1) using misc/smoothMinIndNew!.jl's existing softmax function,
# UNCHANGED, at a configurable temperature. Purely additive: does not modify
# moments/hFunction.jl or the production EK_moments_gammanorm_directgp!.
#
# tuner sign convention (verified from smoothMinIndNew!.jl's own formula, not
# the possibly-stale inline comment which reads backwards): xInd[i] =
# exp((x[i]-xmin)*tuner)/sum(...) is a softmax over (x-xmin)*tuner; since only
# the true minimizer has x[i]-xmin=0, tuner MUST be NEGATIVE for the softmax
# to concentrate weight there (matches hFunction!'s own commented-out
# `tuner=-100.0`). tuner -> -Inf recovers the hard MinInd! exactly; tuner -> 0
# gives a uniform (maximally smoothed) split across origins.
# ============================================================================

"""
    smoothed_factual_G(θ_full, ctx, tuner::Float64) -> Matrix{Float64} (W x D^2)

PER-DRAW (not averaged) factual trade-share moments, using smoothMinIndNew!
(tuner<0, more negative = sharper/closer to hard) in place of MinInd!. Kept
per-draw (not collapsed to a mean) specifically so callers can reweight by a
frozen per-draw m_s* exactly as frozen_adjoint_Q does -- collapsing to
mean(G) first and THEN applying lambda* would silently compute
lambda*'*mean(G) instead of mean(m*.*lambda*'G), a DIFFERENT (wrong) object
since m_s* varies by draw; caught and fixed while writing this file.
"""
function smoothed_factual_G(θ_full::AbstractVector, ctx, tuner::Float64)
    D = ctx.D; W = size(ctx.U, 1)
    price, Aod, AodPow = factual_prices(θ_full, ctx)   # winners.jl, already validated bit-exact
    σ = θ_full[2]; μ = θ_full[1]
    wPow = ctx.γ.wHat .^ (1 - σ)
    constConsσ = [wPow[o] * (AodPow[o, d] * ctx.τ[o, d])^(1 - σ) for o in 1:D, d in 1:D]
    UσPow = ctx.γ.Uσ .^ (-μ)
    denom = [1.0^σ * (ctx.γ.wHat[d] * ctx.γ.L[d]) for d in 1:D]
    P = ctx.γ.P
    gammafac = SpecialFunctions.gamma(μ * (1 - σ) + 1)

    G = zeros(W, D^2)
    xInd = zeros(D)
    @inbounds for d in 1:D, ω in 1:W
        @views smoothMinIndNew!(xInd, price[ω, :, d], D, tuner)
        for o in 1:D
            d1 = d + (o - 1) * D
            G[ω, d1] = ((constConsσ[o, d] / UσPow[ω, o]) * xInd[o] - P[d1] * denom[d]) / gammafac
        end
    end
    # SamplingWeights not applied: this diagnostic assumes SamplingWeights==1 for all draws (true
    # for this investigation's synthetic economy; NOT re-verified for a general economy).
    return G
end

"""
    smoothed_frozen_adjoint_Q(x_free, ctx, base::BaseDualState, tuner) -> Float64

Task §12E ("derivative-only smoothing"): same construction as
frozen_adjoint_Q (per-draw m_s*, lambda* frozen from the base HARD solve),
but the perturbed G_s(x) is computed via smoothed_factual_G instead of a
fresh hard moments! call. Labeled heuristic/inexact per the task's explicit
instruction.
"""
function smoothed_frozen_adjoint_Q(x_free::AbstractVector, ctx, base::BaseDualState, tuner::Float64)
    θ_full = CS.reconstruct_full(x_free, ctx.m)
    G = smoothed_factual_G(θ_full, ctx, tuner)
    oci = ctx.obj.outer_constr_index
    W = size(G, 1)
    acc = 0.0
    @inbounds for s in 1:W
        acc += base.m_star[s] * dot(base.λstar, @view(G[s, 1:oci-1]))
    end
    return (acc / W) - base.ζstar
end
