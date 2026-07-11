# Reduced focal-only moment builder (Phase 2a).
#
# The reduced CC problem the spec assumes: only the focal destination's (baseIndex) objects enter.
# Outer θ_reduced = [μ, σ, γ_focal, γ'_focal, A[1..D, focal]]  (length D+4; A[1,focal] pinned = 1).
# Inner moments = the D focal trade-share moments + 1 focal (autarky) counterfactual price index.
# The omitted destinations' A columns and trade-share moments are dropped entirely (they enter
# neither κ nor the focal shares); they are recovered later by inversion in the sequential loop.
#
# This mirrors, for the focal column only, the algebra of moments!.jl / hFunction.jl /
# hFunctionCounter.jl (autarky, UoModel=1, θConstant=0), and is eltype(θ)-generic for ForwardDiff.
# Correctness is unit-tested against the full EK_moments! focal columns in test_focal_moments.jl.
#
# Signature matches the pipeline's moments! so the existing CC solver can call it unchanged:
#   EK_moments_focal!(K, G, θ, U, obj)   with obj.γ the standard data NamedTuple.

using SpecialFunctions: gamma

"Number of reduced inner moments: D focal trade shares + 1 focal counterfactual."
n_focal_moments(D::Int) = D + 1

"Reduced θ length: μ, σ, γ_focal, γ'_focal, A[1..D,focal]."
n_focal_theta(D::Int) = D + 4

"""
    build_focal_theta(θ_full_initial, D, focal)

Extract the reduced θ = [μ, σ, γ_focal, γ'_focal, A[:,focal]] from the full autarky θ_initial
[μ, σ, γ(D), γ'_focal, Aod(D²)] (Aod column-major, Aod[o,d] at index (d-1)*D+o).
"""
function build_focal_theta(θ_full::AbstractVector, D::Int, focal::Int)
    μ = θ_full[1]; σ = θ_full[2]
    γ_focal = θ_full[2 + focal]         # γ(D) occupies indices 3..2+D
    γp_focal = θ_full[3 + D]            # single γ'_focal under autarky
    Aod = θ_full[(4 + D):(3 + D + D^2)] # D² A entries
    Acol = [Aod[(focal - 1) * D + o] for o in 1:D]
    return vcat(μ, σ, γ_focal, γp_focal, Acol)
end

"""
    EK_moments_focal!(K, G, θ, U, obj)

Fill K (W-vector, constant counterfactual κ) and G (W×(D+1)) for the reduced focal problem.
G[:,1:D] = focal trade-share moments (origin o), G[:,D+1] = focal counterfactual price index.
"""
function EK_moments_focal!(K, G, θ, U, obj)
    γobj = obj.γ
    wHat = γobj.wHat; L = γobj.L; LPrime = γobj.LPrime
    τ = γobj.τ; τPrime = γobj.τPrime; P = γobj.P; μHat = γobj.μHat
    focal = γobj.baseIndex
    D = size(τ, 1); W = size(U, 1)
    T = eltype(θ)

    μ = θ[1]; σ = θ[2]
    γf_θ = θ[3]; γpf_θ = θ[4]
    Acol = @view θ[5:4+D]                         # A[:,focal]; A[1,focal] pinned = 1
    lambda = reshape(P, (D, D))'                  # lambda[o,d] (matches moments!.jl)
    Γ = gamma(μ * (1 - σ) + 1)
    gdp = wHat .* L

    # focal-column effective competitiveness AodPow[o] (= 1/A_od structural), incl. the μ-varying
    # Δ^A adjustment that keeps the model matching data under Frechet at Acol=1 (moments!.jl:75,83)
    AodPow = Vector{T}(undef, D)
    @inbounds for o in 1:D
        base = Acol[o] * ((wHat[o] * τ[o, focal]) / (wHat[1] * τ[1, focal]))^(1 / μ) *
               (lambda[o, focal] / lambda[1, focal])
        AodPow[o] = base^(-μ)
    end

    # γ_focal and γ'_focal with the Δγ adjustments (moments!.jl:90-105, focal column only)
    ΔγA = one(T)
    @inbounds for o in 1:D
        ΔγA *= Acol[o]^(lambda[o, focal] * μ * (σ - 1) / σ)
    end
    Δγμ = (gamma(μ * (1 - σ) + 1) / gamma(μHat * (1 - σ) + 1))^(1 / σ) *
          lambda[1, focal]^((1 - σ) * (μ - μHat) / σ)
    γf = γf_θ * ΔγA * Δγμ

    ΔγA_p = Acol[focal]^(μ * (σ - 1) / σ)
    Δγμ_p = (gamma(μ * (1 - σ) + 1) / gamma(μHat * (1 - σ) + 1))^(1 / σ) *
            (lambda[1, focal] / lambda[focal, focal])^((1 - σ) * (μ - μHat) / σ)
    γpf = γpf_θ * ΔγA_p * Δγμ_p

    # implicit counterfactual κ = 1 − (γ'/γ)^{σ/(σ-1)}  (constant across draws)
    counterVal = 1 - (γpf / γf)^(σ / (σ - 1))
    @. K = counterVal

    denomf = γf^σ * gdp[focal]
    wPow = wHat .^ (1 - σ)
    constConsσ = Vector{T}(undef, D)
    @inbounds for o in 1:D
        constConsσ[o] = wPow[o] * (AodPow[o] * τ[o, focal])^(1 - σ)
    end
    # counterfactual (autarky) domestic price-index pieces; wPrime[focal]=1, τPrime[focal,focal]=1
    cc_prime = (AodPow[focal] * τPrime[focal, focal])^(1 - σ)
    denom_prime = γpf^σ * LPrime[focal]

    @inbounds for ω in 1:W
        # winner = argmin_o level price  w_o·AodPow_o·τ_o·U^{μ}
        best = T(Inf); bo = 1
        for o in 1:D
            price = (wHat[o] * AodPow[o] * τ[o, focal]) * U[ω, o]^μ
            if price < best; best = price; bo = o; end
        end
        for o in 1:D
            share_mag = (o == bo) ? constConsσ[o] * U[ω, o]^(μ * (1 - σ)) : zero(T)
            G[ω, o] = (share_mag - lambda[o, focal] * denomf) / Γ
        end
        G[ω, D + 1] = (cc_prime * U[ω, focal]^(μ * (1 - σ)) - denom_prime) / Γ
    end
    return nothing
end
