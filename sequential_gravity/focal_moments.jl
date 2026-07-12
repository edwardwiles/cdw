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

# ================================================================================================
# γ_focal≡1 NORMALIZED reparameterization
# ================================================================================================
#
# γf enters ONLY through denomf=γf^σ*gdp[focal], the common scale that the D trade-share moments'
# targets (lambda[o,focal]*denomf) must match — i.e. γf sets E_F[Σ_o share_mag] = denomf, the
# MODEL-IMPLIED aggregate expenditure at the focal destination. National-accounts consistency
# (total expenditure at d = gdp[d], an accounting IDENTITY, not a model prediction) requires this
# aggregate to equal gdp[focal] exactly — i.e. γf≡1 structurally, always, for the BASELINE
# (observed) equilibrium. Letting γf_θ range freely (the original parameterization) treats this
# accounting identity as an extra free/searchable degree of freedom, which is what let the outer
# search reach economically-impossible (γ'>γ, negative-GT) combinations: γf and γ'_focal are NOT
# symmetric — γf describes the FULLY-OBSERVED baseline (must satisfy the identity exactly, so
# should be PINNED), while γ'_focal describes the counterfactual under a fixed-wage approximation
# (not a solved GE, so genuinely uncertain — this axis should stay free, but BOUNDED to what the
# model can structurally deliver).
#
# Closed-form verification (autarky, wPrime[focal]=1=wHat[focal], LPrime[focal]=L[focal],
# τPrime[focal,focal]=1): from computeGamma.jl's Phi construction, Phi'_focal = c_focal*w_focal^{-θ}
# (only the domestic term survives τ'_od=∞ for o≠focal) and Phi_focal = Σ_o c_o*(τ_o,focal*w_o)^{-θ}
# (full baseline sum); by the EK gravity identity λ_od = [c_o(τ_od w_o)^{-θ}]/Phi_d, the domestic
# share λ_focal,focal = Phi'_focal/Phi_focal EXACTLY. Combined with γ = (Phi^{-(1-σ)/θ}Γ(·)/(wL))^{1/σ}
# and θ=1/μ:
#     γ'_focal/γ_focal = λ_dd^{(σ-1)/(θσ)} = λ_dd^{μ(σ-1)/σ}     (λ_dd = λ[focal,focal], DATA)
#     κ = 1-(γ'_focal/γ_focal)^{σ/(σ-1)} = 1 - λ_dd^μ
# μ ranges over (0, 1/(σ-1)] (existing outer bound, Frechet regularity), and κ=1-λ_dd^μ is strictly
# increasing in μ (since 0<λ_dd<1), so:
#     κ ∈ [0, 1-λ_dd^{1/(σ-1)}]   ⟺   (γ_focal=1 normalization) γ'_focal ∈ [λ_dd^{1/σ}, 1]
# Numerically verified against θ_initial: γ'_focal/γ_focal = 0.8634603.../0.8985346... = 0.96110...
# = λData[focal,focal]^{μHat(σ-1)/σ} to machine precision (λ_dd≈0.6715, matching the independently
# cross-checked ceiling κ_max=1-0.6715^{1/1.5}=0.2331 from EXPERIMENTS_FINDINGS.md's earlier,
# separately-derived all-A audit — two independent derivations agree).
#
# Reduced θ_norm layout: [μ, σ, γ'_focal(DIRECT, ADJUSTED value — not the raw _θ multiplier),
# A[1..D,focal]]  (length D+3; one fewer than the original D+4, since γf_θ is no longer a free
# slot — it is SOLVED so that γf≡1 identically for whatever (Acol,μ) the search proposes).

"γ_dd (domestic/own trade share) for the focal country — the data constant the theoretical bound uses."
lambda_dd(γobj) = begin
    D = size(γobj.τ, 1); focal = γobj.baseIndex
    reshape(γobj.P, (D, D))'[focal, focal]
end

"Theoretical κ bounds (κ_min, κ_max) = (0, 1-λ_dd^{1/(σ-1)}), and the implied γ'_focal bounds
 (λ_dd^{1/σ}, 1) under the γ_focal≡1 normalization."
function theoretical_kappa_bounds(γobj, σ::Real)
    λdd = lambda_dd(γobj)
    κ_max = 1 - λdd^(1 / (σ - 1))
    γp_lo = λdd^(1 / σ)     # γ'_focal at κ=κ_max
    γp_hi = 1.0             # γ'_focal at κ=0 (=γ_focal)
    return (κ_min = 0.0, κ_max = κ_max, γp_lo = γp_lo, γp_hi = γp_hi)
end

"Reduced θ_norm length: μ, σ, γ'_focal (direct), A[1..D,focal]."
n_focal_theta_norm(D::Int) = D + 3

"""
    build_focal_theta_norm(θ_full_initial, D, focal)

Extract the NORMALIZED reduced θ = [μ, σ, γ'_focal(adjusted), A[:,focal]] from the full autarky
θ_initial. γ'_focal(adjusted) is recovered from the ORIGINAL θ_initial's γ_focal,γ'_focal via the
ratio (γ'_focal/γ_focal) — since at θ_initial the moments are matched EXACTLY (F=F*), this ratio
IS the model-consistent γ'_focal under the γ_focal≡1 gauge, by the derivation above.
"""
function build_focal_theta_norm(θ_full::AbstractVector, D::Int, focal::Int)
    μ = θ_full[1]; σ = θ_full[2]
    γ_focal = θ_full[2 + focal]
    γp_focal = θ_full[3 + D]
    Aod = θ_full[(4 + D):(3 + D + D^2)]
    Acol = [Aod[(focal - 1) * D + o] for o in 1:D]
    γp_norm = γp_focal / γ_focal
    return vcat(μ, σ, γp_norm, Acol)
end

"""
    EK_moments_focal_norm!(K, G, θ, U, obj)

Same as `EK_moments_focal!` but with θ = [μ, σ, γ'_focal(direct), A[:,focal]] (length D+3):
γ_focal is FORCED to 1 (γf_θ solved internally, not a free parameter) and γ'_focal enters directly
as the adjusted value (γpf_θ solved internally so the realized γpf equals θ[3] exactly).
"""
function EK_moments_focal_norm!(K, G, θ, U, obj)
    γobj = obj.γ
    wHat = γobj.wHat; L = γobj.L; LPrime = γobj.LPrime
    τ = γobj.τ; τPrime = γobj.τPrime; P = γobj.P; μHat = γobj.μHat
    focal = γobj.baseIndex
    D = size(τ, 1); W = size(U, 1)
    T = eltype(θ)

    μ = θ[1]; σ = θ[2]
    γp_target = θ[3]                              # γ'_focal DIRECT (adjusted value; γ_focal forced to 1)
    Acol = @view θ[4:3+D]                          # A[:,focal]; A[1,focal] pinned = 1
    lambda = reshape(P, (D, D))'
    Γ = gamma(μ * (1 - σ) + 1)
    gdp = wHat .* L

    AodPow = Vector{T}(undef, D)
    @inbounds for o in 1:D
        base = Acol[o] * ((wHat[o] * τ[o, focal]) / (wHat[1] * τ[1, focal]))^(1 / μ) *
               (lambda[o, focal] / lambda[1, focal])
        AodPow[o] = base^(-μ)
    end

    ΔγA = one(T)
    @inbounds for o in 1:D
        ΔγA *= Acol[o]^(lambda[o, focal] * μ * (σ - 1) / σ)
    end
    Δγμ = (gamma(μ * (1 - σ) + 1) / gamma(μHat * (1 - σ) + 1))^(1 / σ) *
          lambda[1, focal]^((1 - σ) * (μ - μHat) / σ)
    γf = one(T)                                    # FORCED normalization (was γf_θ*ΔγA*Δγμ)
    γf_θ = 1 / (ΔγA * Δγμ)                          # solved so that γf_θ*ΔγA*Δγμ ≡ 1 (unused below,
    #                                                  kept only for diagnostic symmetry with the original)

    ΔγA_p = Acol[focal]^(μ * (σ - 1) / σ)
    Δγμ_p = (gamma(μ * (1 - σ) + 1) / gamma(μHat * (1 - σ) + 1))^(1 / σ) *
            (lambda[1, focal] / lambda[focal, focal])^((1 - σ) * (μ - μHat) / σ)
    γpf = γp_target                                 # γ'_focal enters DIRECTLY as the adjusted value
    γpf_θ = γp_target / (ΔγA_p * Δγμ_p)              # solved so that γpf_θ*ΔγA_p*Δγμ_p ≡ γp_target

    counterVal = 1 - (γpf / γf)^(σ / (σ - 1))       # = 1 - γp_target^{σ/(σ-1)}, since γf≡1
    @. K = counterVal

    denomf = γf^σ * gdp[focal]                      # = gdp[focal] exactly (γf≡1)
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
