# Diagnostics-only variant of EK_moments_gammanorm_directgp_ts! testing whether Enzyme's
# NaN-gradient bug traces to `Aod_θ = reshape(vcat(θ[range]), (D,D))` specifically. Builds
# Aod_θ with a plain preallocated-Matrix + loop instead — no reshape, no vcat, no
# intermediate materialization of a slice. If this is ALSO more efficient (it should be:
# vcat(θ[range]) both slices AND copies, reshape then aliases that copy — a loop into a
# preallocated Matrix does one copy, not two, and needs no temporary), that is itself a
# reason to prefer it regardless of the Enzyme outcome.
function EK_moments_gammanorm_directgp_noreshapevcat!(K, G, θ, U, obj)
    @unpack wHat, L, LPrime, τ, τPrime, P, baseIndex, Uσ, cHat = obj.γ
    @unpack counterType, θConstant, gravMoment, OuterScaling, UoModel = obj.γ.indicators
    W = size(U,1); D = size(τ,1); T = eltype(θ)
    μ = θ[1]; σ = θ[2]
    wPrime = copy(obj.γ.wPrimeHat); insert!(wPrime, baseIndex, 1)
    Aod = ones(T,D,D); AodPow = ones(T,D,D)
    Aod_offset = 3+D
    # THE ONLY CHANGE vs EK_moments_gammanorm_directgp_ts!: no reshape(vcat(...)).
    Aod_θ = Matrix{T}(undef, D, D)
    @inbounds for dd in 1:D, o in 1:D
        Aod_θ[o, dd] = θ[Aod_offset + o + (dd - 1) * D]
    end
    lambda = reshape(P,(D,D))'
    Aod = Aod_θ .* cHat .* (((wHat.*τ)./(wHat[1,1].*τ[1,:]')).^(1/μ)) .* (lambda./lambda[1,:]')
    @. AodPow[:,:] = (Aod[:,:] ./ cHat[:,:]) .^ (-μ)
    γ = ones(T,D); γ_prime = ones(T,D); γ_prime[baseIndex] = θ[3+D]
    @. K[:] = γ_prime[baseIndex]
    N_local = size(U,1)
    UPow = zeros(T,size(U)); UσPow = zeros(T,size(U))
    @. UPow = U ^ (-μ); UσPow .= Uσ[1:N_local,:] .^ (-μ)
    hFunction!(G, UPow, UσPow, wHat, τ, σ, γ, AodPow, L, P, counterType, gravMoment, 0, 0, 0, 0.0, UoModel)
    hFunctionCounter!(K, G, UPow, UσPow, wPrime, τPrime, σ, γ_prime, AodPow, LPrime, counterType, baseIndex, UoModel)
    newGravityMoment_typestable!(G, τ, D, W, γ, AodPow, U, 0, UoModel)
    simple_end = D^2+1
    @. G[:,1:simple_end] /= gamma(μ*(1-σ)+1)
    return nothing
end
