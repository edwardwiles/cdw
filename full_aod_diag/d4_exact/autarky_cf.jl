# ============================================================================
# Specialized focal-autarky counterfactual price-index moment construction.
#
# ADDITIVE / OPT-IN, following the established enable_pow_cache! pattern
# (moments_fast.jl) and the fullA_pow_cache_wiring.md discipline: build a
# specialized fast path behind a helper, verify bit-identical vs the trusted
# path, only then wire it in. The generic hFunctionCounter! remains the
# reference/fallback and is used verbatim for every OTHER counterfactual.
#
# ---- Why a specialized path exists (audit, verified against code + data) ----
#
# Under counterType==1 (autarky) the focal destination's counterfactual supplier
# is MECHANICALLY the focal country itself on every draw: p^aut_bi(s) = p_{bi,bi}(s).
# So the counterfactual price-index moment column (G[:, D^2+1]) is, EXACTLY
# (moments/hFunction.jl:201, the active autarky branch):
#
#     G[s, D^2+1]  =  constConsσ_cf[bi,bi] / UσPow[s,bi]  -  denom_cf      (raw)
#       constConsσ_cf[bi,bi] = wPrime[bi]^(1-σ) * (AodPow[bi,bi]*τPrime[bi,bi])^(1-σ)
#       denom_cf             = γ'[bi]^σ * (wPrime[bi] * LPrime[bi])
#       UσPow[s,bi]          = Uσ[s,bi]^(-μ)   (origin bi's σ-draw term)
#     then /gammafac (cols ≤ D^2+1) and *SamplingWeights[s], like every column.
#
# There is NO counterfactual minimization, NO winner search, NO price vector for
# o≠bi. The generic `hFunctionCounter!` autarky branch already produces exactly
# this via a single broadcast -- BUT the CALL still (a) copies+inserts wPrime,
# (b) allocates pricesTemp/pricesTempσ/pricesInd/pricesCounterVec(D^2)/denom/
# constCons(D^2)/constConsσ(D^2)/wPow EVERY thread-chunk, and (c) recomputes the
# FULL D×D constCons (never used at all under autarky) and constConsσ (only the
# [bi,bi] entry is used) and the full denom vector (only [bi] used). This
# specialized path computes only the two scalars it needs and does the O(W)
# broadcast directly, skipping hFunctionCounter! entirely under autarky.
#
# DOMESTIC SCALAR NOTE (verified empirically, autarky_cf_audit.jl): in this
# calibration wPrime[bi]=wHat[bi]=1, τPrime[bi,bi]=τ[bi,bi]=1, LPrime[bi]=L[bi],
# so the counterfactual CES NUMERATOR is bit-identical to the FACTUAL domestic
# numerator (constConsσ_cf[bi,bi] == constConsσ_fac[bi,bi] == AodPow[bi,bi]^(1-σ)).
# The ONLY thing that differs from the factual domestic term is the constant:
# denom_cf = γ'[bi]^σ · LPrime[bi]  vs  denom_fac = 1 · L[bi]  (the free γ'^σ factor
# and, in general, LPrime vs L). This function computes the CF scalar from the
# PRIMED quantities exactly (wPrime/τPrime/LPrime/γ'), so it stays correct even
# if a future calibration breaks the w'=w=1, τ'=τ, L'=L coincidences.
# ============================================================================
using UnPack

"""
    autarky_cf_scalars(obj, AodPow, σ, γ_prime_bi) -> (cf_num, cf_denom, o1)

The two draw-independent scalars (and the σ-draw column index `o1`) for the
focal-autarky counterfactual price-index column, computed exactly as
hFunctionCounter!'s autarky branch would, but WITHOUT its per-chunk allocations
or the unused D×D constCons/constConsσ/denom construction.
"""
@inline function autarky_cf_scalars(obj, AodPow::AbstractMatrix, σ, γ_prime_bi)
    γo = obj.γ
    bi = γo.baseIndex
    D = size(AodPow, 1)
    UoModel = γo.indicators.UoModel
    # wPrime[bi] is the focal counterfactual wage: hFunctionCounter! builds wPrime by
    # inserting 1 at baseIndex into wPrimeHat, so wPrime[bi] == 1 exactly.
    wPrime_bi = 1.0
    τPrime_bi = γo.τPrime[bi, bi]
    LPrime_bi = γo.LPrime[bi]
    cf_num   = wPrime_bi^(1 - σ) * (AodPow[bi, bi] * τPrime_bi)^(1 - σ)
    cf_denom = γ_prime_bi^σ * (wPrime_bi * LPrime_bi)
    o1 = (UoModel == 1) ? bi : bi + (bi - 1) * D
    return cf_num, cf_denom, o1
end

"""
    fill_autarky_cf_column!(G, UσPow, cf_num, cf_denom, o1)

Write the RAW focal-autarky counterfactual price-index column
`G[:, D^2+1] = cf_num / UσPow[:,o1] - cf_denom` (O(W), no min, no allocation,
reads only column o1 of the already-materialized UσPow). Post-processing
(/gammafac, *SamplingWeights) is applied by the caller exactly as for the
generic path, so the two are bit-identical.
"""
@inline function fill_autarky_cf_column!(G::AbstractMatrix, UσPow::AbstractMatrix,
                                         cf_num, cf_denom, o1::Int)
    D2p1 = size(UσPow, 2)^2 + 1
    @. @views G[:, D2p1] = cf_num / UσPow[:, o1] - cf_denom
    return nothing
end

"""
    EK_moments_gammanorm_directgp_autarkyCF!(K, G, θ, U, obj; pow_cache=nothing)

Byte-for-byte mirror of `EK_moments_gammanorm_directgp!`
(full_aod_diag/moments_gammanorm.jl) EXCEPT the counterfactual price-index
column (G[:, D^2+1]) is produced by the specialized O(W) autarky broadcast above
instead of a per-chunk `hFunctionCounter!` call. The FACTUAL bilateral block
(`hFunction!`) and ALL post-processing are called verbatim/identical, so output
is bit-identical (see test_autarky_cf.jl). If `pow_cache::MuSigmaPowCache` is
supplied, UPow/UσPow are served from it (composes with enable_pow_cache!).

counterType==1 (autarky) only, matching the original.
"""
function EK_moments_gammanorm_directgp_autarkyCF!(K, G, θ, U, obj; pow_cache = nothing)
    @unpack wHat, L, LPrime, τ, τPrime, P, σ_Moments, baseIndex, refIndex1, indicators, Uσ, μHat, CDF_Moments, Ind_Moments, cHat, IndCDF_Cells, Ū, numMomentsSimple, SamplingWeights, PMM, moments_without_var, UPow_scratch, UσPow_scratch = obj.γ
    @unpack counterExplicit, counterType, θConstant, gravMoment, localGravityMoment,
    GravityMomentFirstApproach, sameMarginalsMoment, independenceMoment, momentOrder,
    momentOrderForBaseIndex, IndMomentOrder, OuterScaling, usePMM, UoModel, NormalizeMoments = indicators

    counterType == 1 || error("EK_moments_gammanorm_directgp_autarkyCF! only implements counterType==1 (autarky)")

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

    # ---- specialized autarky CF scalars (no hFunctionCounter!, no per-chunk alloc) ----
    cf_num, cf_denom, cf_o1 = autarky_cf_scalars(obj, AodPow, σ, γ_prime_bi)

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
        # counterfactual column: single O(W) broadcast over the full UσPow column
        fill_autarky_cf_column!(G, UσPow, cf_num, cf_denom, cf_o1)
    else
        Th = Threads.nthreads()
        Threads.@threads for t ∈ 1:Th
            ix0 = round(Int, (t - 1) / Th * W) + 1
            ix1 = round(Int, t / Th * W)
            hFunction!(@view(G[ix0:ix1, :]), @view(U[ix0:ix1, :]), @view(Uσ[ix0:ix1, :]), wHat, τ, σ, γ, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, independenceMoment, μHat, UoModel)
        end
        fill_autarky_cf_column!(G, Uσ, cf_num, cf_denom, cf_o1)
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
    enable_autarky_cf!(ctx; pow_cache=nothing) -> nothing

Opt-in wiring of the specialized focal-autarky CF path into the LIVE oracle path
(same single-field-mutation mechanism as enable_pow_cache!). Pass a
`MuSigmaPowCache` to compose with the pow cache. Leaves d4_exact_setup()
unchanged; existing callers/tests unaffected unless they call this.
"""
function enable_autarky_cf!(ctx; pow_cache = nothing)
    obj = ctx.obj
    obj.moments! = (K, G, θ, U, o) -> EK_moments_gammanorm_directgp_autarkyCF!(K, G, θ, U, o; pow_cache = pow_cache)
    return nothing
end

# ============================================================================
# fix/zc-profile-focal-sigmaminus1-mean-2026-08-07: derived focal k=(sigma-1) mean value
# nu_star = cf_denom/cf_num and its EXACT Jacobian w.r.t. every economic coordinate that affects
# it (gp AND the focal A_dd term -- task Section 3; the 2026-08-05 merged Variant-C fix only
# chain-ruled through gp, silently dropping the A_dd term, since cf_num depends on AodPow[bi,bi]).
# Both derivatives verified against direct finite differences at real D20 data, sigma=3
# (docs/audits/zc-profile-focal-sigmaminus1-mean-2026-08-07/MASTER.md,
# diagnostics/02_dnu_star_jacobian_d20.jl): <1e-9 relative error.
# ============================================================================

"""
    nu_star_value_and_dgrad(θ_full, ctx) -> (nu_star, d_nu_d_gp, d_lognu_d_adid)

Pure algebraic function of `theta_full` (NO inner solve -- `nu_star` is an algebraic function of
theta, not a fixed point of the inner problem). Reconstructs `AodPow` exactly as
`EK_moments_gammanorm_directgp_autarkyCF!` does, then calls `autarky_cf_scalars` (UNCHANGED,
above) to get `cf_num`/`cf_denom`.

    nu_star = cf_denom / cf_num
    d(nu_star)/d(gp)       = nu_star * sigma / gp     (cf_denom = gp^sigma*(...), cf_num has NO
                                                         gp dependence -- confirmed by inspection
                                                         of autarky_cf_scalars above)
    d(log nu_star)/d(a_dd) = sigma - 1                 (a_dd := log(AodPow[bi,bi]); cf_num propto
                                                         AodPow[bi,bi]^(1-sigma), cf_denom has NO
                                                         AodPow[bi,bi] dependence)
"""
function nu_star_value_and_dgrad(θ_full::AbstractVector{Float64}, ctx)
    D = ctx.D
    Ddest = _ctx_ddest(ctx)
    μ = θ_full[1]; σ = θ_full[2]
    Aod_offset = 3 + D
    Aod_θ = reshape(θ_full[Aod_offset+1:Aod_offset+D*Ddest], (D, Ddest))
    γo = ctx.γ
    lambda = reshape(γo.P, (Ddest, D))'
    Aod_lvl = Aod_θ .* γo.cHat .* (((γo.wHat .* ctx.τ) ./ (γo.wHat[1, 1] .* ctx.τ[1, :]')) .^ (1 / μ)) .* (lambda ./ lambda[1, :]')
    AodPow = (Aod_lvl ./ γo.cHat) .^ (-μ)
    γ_prime_bi = θ_full[3+D]
    cf_num, cf_denom, _ = autarky_cf_scalars(ctx.obj, AodPow, σ, γ_prime_bi)
    nu_star = cf_denom / cf_num
    d_nu_d_gp = nu_star * σ / γ_prime_bi
    d_lognu_d_adid = σ - 1
    return nu_star, d_nu_d_gp, d_lognu_d_adid
end

"""
    FocalKStarDerivativeInfo

Precomputed (ONCE per driver setup -- not per callback) description of which `z_free` coordinate
governs the focal country's own diagonal cell `z[bi,bi] = log(Aod_theta[bi,bi])`, needed to
propagate `d(nu_star)/dx` into the outer gradient (task Section 3/11) through the EXISTING
gravity-pivot chain-rule machinery (`gravity_elimination.jl`'s `PivotGravityElim`), not a new one.
Handles BOTH cases the pivot choice could put `(bi,bi)` in -- confirmed live at real D20 production
data that `(bi,bi)` is an ORDINARY free coordinate (`is_pivot=false`), but this is data-dependent
(the pivot is chosen as `argmax|c|` over ALL D*Ddest cells, task-brief-unspecified whether `(bi,bi)`
could ever coincide with it under some other config), so both branches are implemented, not assumed.
"""
struct FocalKStarDerivativeInfo
    is_pivot::Bool
    j0::Int                      # valid iff !is_pivot: position of (bi,bi) in z_free/pe.other_idx
    dz_dzfree::Vector{Float64}   # valid iff is_pivot: length(z_free), = -pe.c[pe.other_idx]/pe.c[pe.pivot_lin]
end

"""
    build_focal_kstar_derivative_info(ctx, pe) -> FocalKStarDerivativeInfo

`pe::PivotGravityElim` is the driver's own (already-built) pivot-elimination object. Locates
`(bi,bi)`'s linear (column-major) index in the `D x Ddest` matrix and checks it against
`pe.pivot_lin`/`pe.other_idx` -- read directly off `pe`'s own fields, no re-derivation of the pivot
choice itself.
"""
function build_focal_kstar_derivative_info(ctx, pe)
    D = ctx.D; bi = ctx.bi
    lin_bd = bi + (bi - 1) * D
    if lin_bd == pe.pivot_lin
        dz_dzfree = -pe.c[pe.other_idx] ./ pe.c[pe.pivot_lin]
        return FocalKStarDerivativeInfo(true, -1, dz_dzfree)
    else
        j0 = findfirst(==(lin_bd), pe.other_idx)
        j0 === nothing && error("build_focal_kstar_derivative_info: (bi,bi) linear index $lin_bd not found in pe.other_idx or as the pivot -- inconsistent pivot elimination structure")
        return FocalKStarDerivativeInfo(false, j0, Float64[])
    end
end

"""
    apply_focal_kstar_chain_rule!(gfull, θ_full, ctx, info::FocalKStarDerivativeInfo, D2_econ, coeff)

Adds `coeff * d(nu_star)/dx` into `gfull` at every economic coordinate `x` (`gp` always at
`gfull[1]`; the `z_free` coordinate(s) governing the focal A_dd term at `gfull[2:D2_econ]`, in
Z-SPACE units). MUST be called BEFORE any `A_coordinate_mode=:powered_aspace` rescale
(`gfull[2:D2_econ] .*= -theta_cm`) -- that existing, UNCHANGED rescale applies uniformly to
whatever is in `gfull[2:D2_econ]` at the time it runs, so adding this z-space contribution first
lets it get carried through by the SAME existing machinery, no separate a-space derivative needed.
`coeff` is task Section 11's pair-lambda sum (`d_delta_d_nu_star`, from
`d_delta_dual_d_eta_active_and_nustar`/its CM+ZC analog).
"""
function apply_focal_kstar_chain_rule!(gfull::AbstractVector{Float64}, θ_full::AbstractVector{Float64}, ctx,
                                        info::FocalKStarDerivativeInfo, D2_econ::Int, coeff::Float64)
    nu_star, d_nu_d_gp, d_lognu_d_adid = nu_star_value_and_dgrad(θ_full, ctx)
    μ = θ_full[1]
    d_lognu_d_z_at_focal = d_lognu_d_adid * (-μ)   # d(a_dd)/d(z[bi,bi]) = -mu, FD-verified
    d_nu_d_z_at_focal = nu_star * d_lognu_d_z_at_focal
    gfull[1] += coeff * d_nu_d_gp
    if info.is_pivot
        @inbounds for j in eachindex(info.dz_dzfree)
            gfull[1+j] += coeff * d_nu_d_z_at_focal * info.dz_dzfree[j]
        end
    else
        gfull[1+info.j0] += coeff * d_nu_d_z_at_focal
    end
    return gfull
end
