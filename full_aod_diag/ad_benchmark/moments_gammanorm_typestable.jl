# Diagnostics-only copy of EK_moments_gammanorm_directgp! that calls
# newGravityMoment_typestable! instead of newGravityMoment! (the ONE-token type-stability
# fix, `meanτ = 0` -> `meanτ = zero(eltype(τ))`, verified primal-identical, dead code for
# UoModel==1). Needed for Enzyme (Method C): combined with enzyme_gamma_rule.jl (the custom
# gamma derivative rule bypassing the broken digamma JIT symbol, Enzyme.jl issue #2890),
# these are the TWO real, independent, minimal fixes required to get Enzyme compiling
# through the full moment map. Neither touches production; both are additive/diagnostics-
# only per the audit's own rules.
function EK_moments_gammanorm_directgp_ts!(K, G, θ, U, obj)
    @unpack wHat, L, LPrime, τ, τPrime, P, baseIndex, Uσ, cHat = obj.γ
    @unpack counterType, θConstant, gravMoment, OuterScaling, UoModel = obj.γ.indicators
    W = size(U,1); D = size(τ,1); T = eltype(θ)
    μ = θ[1]; σ = θ[2]
    wPrime = copy(obj.γ.wPrimeHat); insert!(wPrime, baseIndex, 1)
    Aod = ones(T,D,D); AodPow = ones(T,D,D)
    Aod_offset = 3+D
    Aod_θ = reshape(vcat(θ[Aod_offset+1:Aod_offset+D^2]), (D,D))
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
