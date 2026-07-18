# ============================================================================
# Focal-autarky counterfactual price-index moment — V2 cached-base-vector path.
#
# ADDITIVE / OPT-IN, layered ON TOP OF autarky_cf.jl (which must be `include`d
# first — this file reuses its `autarky_cf_scalars`). Follows the established
# enable_pow_cache! / enable_autarky_cf! discipline (single mutable-field rebind,
# bit-or-tight equivalence tested before adoption).
#
# ---------------------------------------------------------------------------
# WHAT THIS ADDS OVER autarky_cf.jl
# ---------------------------------------------------------------------------
# autarky_cf.jl already removed the generic hFunctionCounter! call and does the
# raw column as a single O(W) broadcast:
#
#     G[s, D^2+1] = cf_num / UσPow[s, o1] - cf_denom            (autarky_cf.jl)
#
# with UσPow[s,o1] = Uσ[s,o1]^(-μ) re-materialized every build. V2 observes that
# μ, σ, the focal σ-draw column Uσ[:,o1], the domestic wage wPrime[bi]≡1, the
# domestic trade cost τPrime[bi,bi], the focal labour LPrime[bi] and the entire
# draw matrix are PROVABLY FIXED for a ctx's whole outer optimization
# (context.jl pins θ_lo[1]==θ_hi[1] (μ), θ_lo[2]==θ_hi[2] (σ), and holds indices
# 1,2,3:2+D out of the free-param map). The only outer-free quantities that reach
# this column are:
#   * A_dd  = Aod_θ[bi,bi] = θ[3+D + (bi-1)*D + bi]   (one diagonal free-A entry)
#   * γ'_bi = θ[3+D]                                  (the objective variable)
# and NOTHING ELSE in the A_od matrix — the column reads only AodPow[bi,bi],
# which depends on Aod_θ[bi,bi] alone (verified in test_autarky_cf_v2.jl).
#
# So we precompute ONCE the draw-level reciprocal vector
#
#     inv_uσ[s] = 1 / UσPow[s, o1] = 1 / Uσ[s, o1]^(-μ)          (μ, Uσ, o1 fixed)
#
# and at every outer point the column is a single fused multiply-subtract with
# TWO scalars and NO per-draw divide, power, log, exp, min, or branch:
#
#     G[s, D^2+1] = cf_num * inv_uσ[s] - cf_denom                (V2)
#
# where cf_num, cf_denom come from autarky_cf.jl::autarky_cf_scalars VERBATIM
# (so the A_dd/γ' dependence, the exponent and the sign are byte-identical to the
# already-adopted path — we do not re-derive them here).
#
# ---------------------------------------------------------------------------
# THE SCALAR AND ITS EXACT EXPONENT (verified, see test_autarky_cf_v2.jl)
# ---------------------------------------------------------------------------
#   cf_num = wPrime[bi]^(1-σ) * (AodPow[bi,bi]*τPrime[bi,bi])^(1-σ)
#   AodPow[bi,bi] = (Aod[bi,bi]/cHat[bi,bi])^(-μ),  Aod[bi,bi]/cHat[bi,bi] = A_dd * B_bi
#   ⇒ cf_num = C_num_fixed * A_dd^(-μ(1-σ)) = C_num_fixed * A_dd^(μ(σ-1))
# So on the RAW free variable A_dd = Aod_θ[bi,bi] the exponent is  μ(σ-1)
#   (NOT the task-hint's bare (σ-1); the μ factor is real and must be kept).
# On the code's power-convention quantity AodPow[bi,bi] the factor is
#   AodPow[bi,bi]^(1-σ), i.e. exponent (1-σ) = -(σ-1).
# cf_denom = γ'_bi^σ * wPrime[bi] * LPrime[bi]  is A_dd-INDEPENDENT (a pure
# additive constant that shifts with the objective variable γ'_bi only).
#
# EQUIVALENCE NOTE: because inv_uσ is the reciprocal of UσPow[s,o1] formed once,
# `cf_num * inv_uσ[s]` is NOT bit-identical to `cf_num / UσPow[s,o1]` — it differs
# by the single extra rounding of the reciprocal (~1 ULP). V2 therefore targets
# DOCUMENTED-TIGHT equivalence (measured max rel-err ~1e-16), unlike autarky_cf.jl
# which is exactly bit-identical. This is the deliberate cost of turning the
# per-draw divide into a per-draw multiply against a once-computed cache.
# ============================================================================

"""
    AutarkyCFBase

Immutable-after-build cache of the draw-level reciprocal vector
`inv_uσ[s] = 1 / (Uσ[s,o1]^(-μ))` for the focal-autarky CF column, plus the
(μ, o1) it was built at so reuse can be validated (never assumed). `valid=false`
forces one build on first use; if ever asked for a DIFFERENT (μ,o1) it rebuilds
(degrades gracefully, never returns a stale vector for the wrong μ).
"""
mutable struct AutarkyCFBase
    μ::Float64
    o1::Int
    inv_uσ::Vector{Float64}
    valid::Bool
    n_build::Int
    n_reuse::Int
end
AutarkyCFBase(W::Int) = AutarkyCFBase(NaN, 0, zeros(W), false, 0, 0)

"""
    get_autarky_cf_base!(base, Uσ, μ, o1) -> inv_uσ

Populate (if stale / different μ or o1) or reuse the cached reciprocal vector
`inv_uσ[s] = 1 / Uσ[s,o1]^(-μ)`, computed with the SAME `^(-μ)` as production
UσPow so the only float difference vs the divide-based paths is the reciprocal.
"""
function get_autarky_cf_base!(base::AutarkyCFBase, Uσ::AbstractMatrix, μ::Float64, o1::Int)
    if !base.valid || base.μ !== μ || base.o1 != o1
        W = size(Uσ, 1)
        length(base.inv_uσ) == W || (base.inv_uσ = zeros(W))
        @inbounds @simd for s in 1:W
            base.inv_uσ[s] = 1.0 / (Uσ[s, o1]^(-μ))
        end
        base.μ = μ
        base.o1 = o1
        base.valid = true
        base.n_build += 1
    else
        base.n_reuse += 1
    end
    return base.inv_uσ
end

"""
    fill_autarky_cf_column_v2!(G, inv_uσ, cf_num, cf_denom, col)

Write the RAW focal-autarky CF column (index `col == D^2+1`) as a single per-draw
multiply-subtract against the cached reciprocal vector:
`G[:, col] = cf_num * inv_uσ - cf_denom`. No per-draw divide/power/log/exp/min/
branch. Post-processing (/gammafac, *SamplingWeights) is applied by the caller
exactly as for every other column.
"""
@inline function fill_autarky_cf_column_v2!(G::AbstractMatrix, inv_uσ::AbstractVector,
                                            cf_num, cf_denom, col::Int)
    @inbounds @simd for s in eachindex(inv_uσ)
        G[s, col] = cf_num * inv_uσ[s] - cf_denom
    end
    return nothing
end

"""
    EK_moments_gammanorm_directgp_autarkyCF_v2!(K, G, θ, U, obj; base, pow_cache=nothing)

Mirror of `EK_moments_gammanorm_directgp_autarkyCF!` (autarky_cf.jl) EXCEPT the
CF column is produced by `fill_autarky_cf_column_v2!` against the once-built
`base::AutarkyCFBase` reciprocal vector instead of a live `./ UσPow[:,o1]`.
The FACTUAL bilateral block (`hFunction!`), the UPow/UσPow materialization it
needs, and ALL post-processing are called verbatim/identical, so output matches
the trusted path to tight tolerance (see test_autarky_cf_v2.jl).
counterType==1 (autarky) only.
"""
function EK_moments_gammanorm_directgp_autarkyCF_v2!(K, G, θ, U, obj;
                                                     base::AutarkyCFBase,
                                                     pow_cache = nothing)
    @unpack wHat, L, LPrime, τ, τPrime, P, σ_Moments, baseIndex, refIndex1, indicators, Uσ, μHat, CDF_Moments, Ind_Moments, cHat, IndCDF_Cells, Ū, numMomentsSimple, SamplingWeights, PMM, moments_without_var, UPow_scratch, UσPow_scratch = obj.γ
    @unpack counterExplicit, counterType, θConstant, gravMoment, localGravityMoment,
    GravityMomentFirstApproach, sameMarginalsMoment, independenceMoment, momentOrder,
    momentOrderForBaseIndex, IndMomentOrder, OuterScaling, usePMM, UoModel, NormalizeMoments = indicators

    counterType == 1 || error("EK_moments_gammanorm_directgp_autarkyCF_v2! only implements counterType==1 (autarky)")

    W = size(U, 1)
    D = size(τ, 1)
    T = eltype(θ)

    μ = θ[1]
    σ = θ[2]

    Aod = ones(T, D, D)
    AodPow = ones(T, D, D)
    Aod_θ = ones(T, D, D)
    Aod_offset = 3 + D
    if OuterScaling == 1
        if independenceMoment == 1
            Aod_offset += 1
        end
        Aod_θ = reshape(vcat(θ[Aod_offset+1:Aod_offset+D^2]), (D, D))
    end

    lambda = reshape(P, (D, D))'

    if θConstant != 1
        Aod = Aod_θ .* cHat .* (((wHat .* τ) ./ (wHat[1, 1] .* τ[1, :]')) .^ (1 / μ)) .* (lambda ./ lambda[1, :]')
    else
        Aod = Aod_θ
    end
    @. AodPow[:, :] = (Aod[:, :] ./ cHat[:, :]) .^ (-μ)

    γ = ones(T, D)
    γ_prime_bi = θ[3+D]

    if counterExplicit == 0
        @. K[:] = γ_prime_bi
    end

    # ---- specialized autarky CF scalars (reused verbatim from autarky_cf.jl) ----
    cf_num, cf_denom, cf_o1 = autarky_cf_scalars(obj, AodPow, σ, γ_prime_bi)
    cf_col = D^2 + 1

    if θConstant != 1
        if pow_cache !== nothing && eltype(γ) === Float64 && T === Float64
            UPow, UσPow = get_upow!(pow_cache, U, Uσ, μ)
        elseif eltype(γ) === Float64 && size(UPow_scratch, 1) == size(U, 1)
            UPow = UPow_scratch
            UσPow = UσPow_scratch
            Th0 = Threads.nthreads()
            Threads.@threads for t ∈ 1:Th0
                ix0 = round(Int, (t - 1) / Th0 * W) + 1
                ix1 = round(Int, t / Th0 * W)
                @. UPow[ix0:ix1, :] = U[ix0:ix1, :] .^ (-μ)
                @. UσPow[ix0:ix1, :] = Uσ[ix0:ix1, :] .^ (-μ)
            end
        else
            UPow = zeros(eltype(γ), size(U))
            UσPow = zeros(eltype(γ), size(U))
            @. UPow = U ^ (-μ)
            @. UσPow = Uσ ^ (-μ)
        end
        Th = Threads.nthreads()
        Threads.@threads for t ∈ 1:Th
            ix0 = round(Int, (t - 1) / Th * W) + 1
            ix1 = round(Int, t / Th * W)
            hFunction!(@view(G[ix0:ix1, :]), @view(UPow[ix0:ix1, :]), @view(UσPow[ix0:ix1, :]), wHat, τ, σ, γ, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, independenceMoment, μHat, UoModel)
        end
        # counterfactual column: cached reciprocal, single per-draw multiply-subtract
        inv_uσ = get_autarky_cf_base!(base, Uσ, μ, cf_o1)
        fill_autarky_cf_column_v2!(G, inv_uσ, cf_num, cf_denom, cf_col)
    else
        Th = Threads.nthreads()
        Threads.@threads for t ∈ 1:Th
            ix0 = round(Int, (t - 1) / Th * W) + 1
            ix1 = round(Int, t / Th * W)
            hFunction!(@view(G[ix0:ix1, :]), @view(U[ix0:ix1, :]), @view(Uσ[ix0:ix1, :]), wHat, τ, σ, γ, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, independenceMoment, μHat, UoModel)
        end
        # θConstant path divides by raw Uσ (not UσPow); build the base at μ=1 so
        # inv_uσ[s] = 1/Uσ[s,o1]^(-1) is the reciprocal of Uσ used there.
        inv_uσ = get_autarky_cf_base!(base, Uσ, 1.0, cf_o1)
        fill_autarky_cf_column_v2!(G, inv_uσ, cf_num, cf_denom, cf_col)
    end

    if gravMoment == 1
        newGravityMoment!(G, τ, D, W, γ, AodPow, U, GravityMomentFirstApproach, UoModel)
    end

    GravityMomentFirstApproach == 0 || error("gammanorm variant: GravityMomentFirstApproach not implemented")
    sameMarginalsMoment == 0 || error("gammanorm variant: sameMarginalsMoment not implemented")
    independenceMoment == 0 || error("gammanorm variant: independenceMoment not implemented")

    if θConstant != 1
        simple_end = D^2 + 1
        @. G[:, 1:simple_end] /= gamma(μ * (1 - σ) + 1)
    end

    if usePMM == 1
        for im ∈ 1:numMomentsSimple
            @. G[:, im] -= PMM[im]
        end
    end

    if NormalizeMoments == 1
        for im ∈ 1:numMomentsSimple-GravityMomentFirstApproach-independenceMoment
            if im ∉ moments_without_var
                @. G[:, im] *= 1 ./ σ_Moments[im]
            end
        end
    end

    for im ∈ 1:numMomentsSimple
        @. G[:, im] *= SamplingWeights[1:W]
    end
    @. K[:] *= SamplingWeights[1:W]

    return nothing
end

"""
    enable_autarky_cf_v2!(ctx; pow_cache=nothing) -> AutarkyCFBase

Opt-in wiring of the V2 cached-base focal-autarky CF path into the LIVE oracle
(same single-field-mutation mechanism as enable_pow_cache!/enable_autarky_cf!).
Returns the per-ctx `AutarkyCFBase` so callers can inspect `.n_build`/`.n_reuse`
(the closure retains its own reference). Pass a `MuSigmaPowCache` to compose with
the pow cache for the factual block. Requires autarky_cf.jl to be `include`d
(for `autarky_cf_scalars`).
"""
function enable_autarky_cf_v2!(ctx; pow_cache = nothing)
    obj = ctx.obj
    W = size(obj.U, 1)
    base = AutarkyCFBase(W)
    obj.moments! = (K, G, θ, U, o) ->
        EK_moments_gammanorm_directgp_autarkyCF_v2!(K, G, θ, U, o; base = base, pow_cache = pow_cache)
    return base
end
