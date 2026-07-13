# Direct-γ' objective variant of EK_moments_focal_norm! (focal_moments.jl), matching this
# session's EK_moments_gammanorm_directgp! for the full-A model: the ONLY change is
# `counterVal = γpf` (γ'_focal DIRECTLY) instead of `1 - (γpf/γf)^(σ/(σ-1))`. Since γf≡1 already
# (this is the γ_focal≡1-normalized reduced model) and κ = 1-(γpf/γf)^(σ/(σ-1)) is a strictly
# monotone (decreasing) transform of γpf alone, extremizing γpf directly and converting κ
# afterward gives identical bounds. Every OTHER moment (the D focal trade shares, the
# counterfactual price-index moment) is byte-identical — copied verbatim from
# focal_moments.jl::EK_moments_focal_norm!, only the K/counterVal line differs.
function EK_moments_focal_norm_directgp!(K, G, θ, U, obj)
    γobj = obj.γ
    wHat = γobj.wHat; L = γobj.L; LPrime = γobj.LPrime
    τ = γobj.τ; τPrime = γobj.τPrime; P = γobj.P; μHat = γobj.μHat
    focal = γobj.baseIndex
    D = size(τ, 1); W = size(U, 1)
    T = eltype(θ)

    μ = θ[1]; σ = θ[2]
    γp_target = θ[3]
    Acol = @view θ[4:3+D]
    lambda = reshape(P, (D, D))'
    Γ = gamma(μ * (1 - σ) + 1)
    gdp = wHat .* L

    AodPow = Vector{T}(undef, D)
    @inbounds for o in 1:D
        base = Acol[o] * ((wHat[o] * τ[o, focal]) / (wHat[1] * τ[1, focal]))^(1 / μ) *
               (lambda[o, focal] / lambda[1, focal])
        AodPow[o] = base^(-μ)
    end

    γf = one(T)
    γpf = γp_target
    counterVal = γpf                                # THE ONLY CHANGE vs EK_moments_focal_norm!
    @. K = counterVal

    denomf = γf^σ * gdp[focal]
    wPow = wHat .^ (1 - σ)
    constConsσ = Vector{T}(undef, D)
    @inbounds for o in 1:D
        constConsσ[o] = wPow[o] * (AodPow[o] * τ[o, focal])^(1 - σ)
    end
    cc_prime = (AodPow[focal] * τPrime[focal, focal])^(1 - σ)
    denom_prime = γpf^σ * LPrime[focal]

    @inbounds for ω in 1:W
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
