# ============================================================================
# Shared setup for the A_od profile-optimality diagnostics (see
# A_profile_optimality_report.md). Builds the SAME full-A (all D^2 A_od free,
# gamma_d===1 gauge, direct-gamma'-focal objective) economy used by the
# production driver `full_aod_diag/run_fullA_D10_production.jl`, D configurable
# via DVAL (default 5, per the user's request to start at D=5 or D=10).
#
# This file only builds context/helpers; it does not run any diagnostic.
# ============================================================================
using Parameters, Base.Threads, Random, Dates, DelimitedFiles, Printf
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, JLD2

const DIAG_ROOT = dirname(@__DIR__)
include(joinpath(DIAG_ROOT, "setup/include_setup.jl"))
include(joinpath(DIAG_ROOT, "prestep/include_prestep.jl"))
include(joinpath(DIAG_ROOT, "prepare_cc/include_prepare_cc.jl"))
include(joinpath(DIAG_ROOT, "moments/include_moments.jl"))
include(joinpath(DIAG_ROOT, "cc_algo/include_cc_algo.jl"))
include(joinpath(DIAG_ROOT, "lfd/include_lfd.jl"))
include(joinpath(DIAG_ROOT, "misc/include_misc.jl"))
using .CounterfactualSensitivity
const CS = CounterfactualSensitivity
include(joinpath(DIAG_ROOT, "full_aod_diag", "moments_gammanorm.jl"))
CS.include(joinpath(DIAG_ROOT, "full_aod_diag", "PsiObjectiveBundleImplicitMethodB_fullA.jl"))
include(joinpath(DIAG_ROOT, "full_aod_diag", "gravity_tariff.jl"))
include(joinpath(DIAG_ROOT, "full_aod_diag", "ad_benchmark", "derivative_core.jl"))

"""
    build_diag_context(; D=5, W=8000)

Builds the full-A gammanorm-gauge context at the Frechet benchmark (Aod_theta
identically 1). Returns a NamedTuple with everything the diagnostics need.
"""
function build_diag_context(; D::Int=5, W::Int=8000, gravMoment::Int=1)
    params = (server=1, user=2, fakeData=1, DFake=D, seedFakeData=889, counterType=1, counterExplicit=0,
        θHat=0, σHat=2.5, baseIndex=2, W=W, seedU=888, importanceSampling=0, importanceSamplingFactor=2,
        stratifiedSampling=0, IndMomentOrder=5, θConstant=0, gravMoment=gravMoment, localGravityMoment=0,
        localGravityCrossMoment=0, GravityMomentFirstApproach=0, sameMarginalsMoment=0,
        NoScalingforSameMartingale=1, useCDFforMarginalMatching=0, independenceMoment=0, momentOrder=5,
        momentOrderForBaseIndex=50, ForceFrechetMarginal=0, OuterScaling=1, useParallel=0, usePMM=0,
        PMMGammaOnly=0, NormalizeMoments=0, useConfidenceIntervals=0, ConfidenceLevel=0.05, δGridType=0,
        δ_ref=1, refIndex1=1, OuterLoop=1, UoModel=1, use_Jacobian=0, calc_δ_star_initial=1, Jac_W=W,
        theta_init=0, runLFD=1, runLFDCounterFactual=1)

    so = master_setup(params)
    up = (; params..., D=so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
    checkParams(up)
    ps = master_prestep(so.data, so.counters, up)
    pp = master_prepare_cc(so.data, so.counters, ps, up)
    @unpack θ_initial, θ_initial_up, U, γ, outer_constr_index, nTotalMoments, complement_index, inequality_index = pp

    @assert so.D == D
    bi = params.baseIndex
    σ = params.σHat
    μHat = γ.μHat
    Aod_offset = 3 + D

    θ0_up = build_theta_gammanorm(θ_initial_up, D, bi, μHat, σ)
    bounds = theoretical_gammaprime_bounds(γ, σ)
    θ0_up[3+D] = clamp(θ0_up[3+D], bounds.γp_lo, bounds.γp_hi)

    μ0 = θ0_up[1]
    β = (1 / μ0) / (σ - 1)   # β = θ* / (σ-1), θ* = 1/μ

    λ = reshape(γ.P, (D, D))'   # λ[o,d], data trade shares (Frechet-consistent)
    q_tilde, N_obs = precompute_q_tilde(γ.τ)
    @assert N_obs == D^2

    return (params=params, so=so, pp=pp, γobj=γ, U=U, D=D, W=W,
        outer_constr_index=outer_constr_index, nTotalMoments=nTotalMoments,
        complement_index=complement_index, inequality_index=inequality_index,
        baseIndex=bi, σ=σ, μHat=μHat, Aod_offset=Aod_offset,
        θ0_up=θ0_up, bounds=bounds, β=β, λ=λ, q_tilde=q_tilde, N_obs=N_obs)
end

"Index of moment column for (o,d) in mean-G / H output ordering (matches hFunction!'s d1=d+(o-1)*D)."
idx_out(o::Int, d::Int, D::Int) = d + (o - 1) * D

"Index of Aod_theta entry (j,d) in the theta vector's flattened A-block (matches reshape(.,(D,D)) column-major)."
idx_in(j::Int, d::Int, D::Int) = j + (d - 1) * D

"""
    make_full_theta(θbase, Aodvec, Aod_offset, D)

θbase is the Float64 benchmark vector; Aodvec (length D^2, eltype T, any T
ForwardDiff can promote) overwrites the A_od block. Everything else (μ, σ,
old-γ slots, γ'_focal) stays at θbase's value. Output eltype is
promote_type(eltype(θbase), eltype(Aodvec)).
"""
function make_full_theta(θbase::Vector{Float64}, Aodvec::AbstractVector{T}, Aod_offset::Int, D::Int) where {T}
    θ = Vector{T}(undef, length(θbase))
    @inbounds for i in eachindex(θbase)
        θ[i] = θbase[i]
    end
    @inbounds for k in 1:D^2
        θ[Aod_offset+k] = Aodvec[k]
    end
    return θ
end

"""
Mean over draws of the D^2 RAW trade-flow moments (H columns 3:2+D^2), for any
θ (Dual-safe). NOTE: production's G_od is a DEVIATION object,
`G_od = pricesTempσ[o]*pricesInd[o] - λ_od*denom[d]` (moments/hFunction.jl:86),
so E_F[G_od] = 0 at the Frechet benchmark, NOT λ_od -- see
`tradeshare_share_means` below for the object whose expectation IS λ_od
(matching the user's economic-setup definition of g_od).
"""
function tradeshare_means(θ::AbstractVector, U::AbstractMatrix, γobj, D::Int, nTotalMoments::Int)
    H = moment_map(θ, U, γobj, nTotalMoments)
    return vec(mean(view(H, :, 3:2+D^2), dims=1))
end

"denom[d] = gdp[d] = wHat[d]*L[d], the destination-d GDP scaling factor hFunction! uses (see hFunction.jl:43)."
denom_vec(γobj, D::Int) = [γobj.wHat[d] * γobj.L[d] for d in 1:D]

"""
    tradeshare_share_means(θ, U, γobj, D, nTotalMoments)

Recovers E_F[g_od] in the user's economic-setup units (`g_od = b_od X_o
1{o=argmax}`, with E_F[g_od]=λ_od at the Frechet benchmark) from production's
raw deviation moments: share_od = G_od/denom[d] + λ_od. Since λ_od*denom[d] is
a CONSTANT (data), ∂share_od/∂θ = (1/denom[d]) * ∂G_od/∂θ exactly -- so
Jacobians of this object and of the raw G are related by a per-destination
row rescaling only (see `to_share_jacobian`).
"""
function tradeshare_share_means(θ::AbstractVector, U::AbstractMatrix, γobj, D::Int, nTotalMoments::Int)
    Gmean = tradeshare_means(θ, U, γobj, D, nTotalMoments)
    λ = reshape(γobj.P, (D, D))'
    denom = denom_vec(γobj, D)
    shares = similar(Gmean)
    for d in 1:D, o in 1:D
        k = idx_out(o, d, D)
        shares[k] = Gmean[k] / denom[d] + λ[o, d]
    end
    return shares
end

"Full (un-meaned) H = [K const G] at θ (Float64 only; used for per-draw winner detection)."
function full_H(θ::Vector{Float64}, U::AbstractMatrix, γobj, nTotalMoments::Int)
    W = size(U, 1)
    H = zeros(Float64, W, nTotalMoments + 2)
    moment_map!(H, θ, U, γobj)
    return H
end

"""
    winners_at_destination(H, γobj, D, d_focus)

Recovers, for every draw, the winning origin at destination `d_focus`, from
the ALREADY-COMPUTED moment matrix H (no re-derivation of AodPow/UPow chains):
for a loser o, G[ω,idx_out(o,d)] = -λ[o,d]*denom[d] exactly (a known data
constant); for the winner, G[ω,idx_out(o,d)] = pricesTempσ[o] - λ[o,d]*denom[d]
> that constant (since pricesTempσ>0). So argmax over o of
(G[ω,idx_out(o,d)] + λ[o,d]*denom[d]) recovers the winner.
"""
function winners_at_destination(H::Matrix{Float64}, γobj, D::Int, d_focus::Int)
    λ = reshape(γobj.P, (D, D))'
    denom_d = γobj.wHat[d_focus] * γobj.L[d_focus]
    W = size(H, 1)
    winners = Vector{Int}(undef, W)
    @inbounds for ω in 1:W
        best_o = 1
        best_val = -Inf
        for o in 1:D
            val = H[ω, 2+idx_out(o, d_focus, D)] + λ[o, d_focus] * denom_d
            if val > best_val
                best_val = val
                best_o = o
            end
        end
        winners[ω] = best_o
    end
    return winners
end
