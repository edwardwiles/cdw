# ============================================================================
# Genuinely CONSISTENT smoothed full-A moment kernel + inner/outer solve
# machinery for the D=4 exact investigation (diag/fullA-d4-exact-smoothed-
# consistent branch).
#
# WHY THIS FILE EXISTS (Step 1 of the task): full_aod_diag/d4_exact/
# smoothed_moments.jl's `smoothed_frozen_adjoint_Q` is documented (docs/
# fullA_d4_final_report.md sec 6) to be CALL-HISTORY-DEPENDENT -- repeated
# calls at bit-identical arguments return different values (1.5e-11 to 7.8e-2,
# plus NaN/Inf), root cause not identified in that session. This session's own
# investigation (see docs/fullA_smoothed_consistent_experiment.md sec 1)
# ruled out the two most obvious culprits by direct code reading:
#   - BaseDualState (three_way_derivatives.jl) deep-copies m_star/lambda* at
#     construction (`copy(obj.arg1)`, `collect(inner_x[2:end])`) -- NOT an
#     aliased view into obj's mutable scratch, so later hard-oracle calls
#     elsewhere in the process cannot silently mutate a `base` object under
#     smoothed_frozen_adjoint_Q's feet.
#   - hFunction!'s (and this file's own hFunction_smoothed!'s) per-thread
#     Threads.@threads partitioning writes DISJOINT ω-ranges of G with NO
#     cross-thread reduction -- thread-scheduling nondeterminism cannot
#     explain a per-call-history-dependent VALUE (there is no shared
#     accumulator to race on).
# Root cause not otherwise isolated within this session's budget. Per the
# task's explicit instruction, this file BYPASSES the bug entirely: every
# function below allocates its OWN fresh buffers on every call (no shared
# mutable scratch across calls, no closures capturing mutable state beyond a
# fixed Float64 `tuner`/`rho`) and never calls into smoothed_moments.jl's
# code path. Determinism is re-verified explicitly in test_smoothed_consistent.jl
# (Step 1/2 requirement), not assumed.
#
# CONSISTENCY (the task's central methodological requirement): this is NOT
# derivative-only smoothing (smoothed_moments.jl's Method E, which computed a
# smoothed VALUE but the task brief flags "retaining hard origin indicators"
# as insufficient). Verified by direct code audit of moments/hFunction.jl,
# moments/hFunction.jl::hFunctionCounter!, and moments/newGravityMoment!.jl
# (docs/fullA_smoothed_consistent_experiment.md sec 0) that, for THIS
# investigation's fixed AD_PARAMS (counterType=1 i.e. autarky, UoModel=1,
# localGravityMoment=0, gravMoment=1, independenceMoment=0), there is EXACTLY
# ONE hard branch anywhere in the active moment map: hFunction!'s `MinInd!`
# call building the factual trade-share moments G[:,1:D^2]. Specifically:
#   - hFunctionCounter!'s counterType==1 branch (moments/hFunction.jl:192-216)
#     is a closed-form broadcast in Aod[baseIndex,baseIndex] and gamma'_focal
#     ALONE -- no MinInd! call at all. Verified by reading the branch: it
#     never touches `pricesInd`.
#   - newGravityMoment!'s UoModel==1 branch (moments/newGravityMoment!.jl:17-31)
#     is a pure function of AodPow (=smooth in theta) and tau (data) -- no
#     per-draw winner selection, no U dependence at all.
# So a genuinely consistent smoothing for THIS setup = replace hFunction!'s
# MinInd! with smoothMinIndNew! (misc/smoothMinIndNew!.jl, UNCHANGED,
# genuine softmax supplier probabilities, not a smoothed scalar bolted onto a
# hard indicator) and change NOTHING else -- hFunctionCounter! and
# newGravityMoment! are reused verbatim (imported, not modified, not copied)
# because they are already smooth for this configuration.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "winners_v2.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
using ForwardDiff, SpecialFunctions, LinearAlgebra, Statistics

"""
    hFunction_smoothed!(G, UPow, Uσ, w, τ, σ, γ, Aod, L, P, tuner)

Fresh, from-scratch (own buffers, allocated every call) analogue of
`moments/hFunction.jl::hFunction!`'s factual trade-share block, generalized
to ANY `eltype(G)` (including ForwardDiff.Dual -- unlike smoothed_moments.jl's
`smoothed_factual_G`, which hardcodes `G = zeros(W, D^2)` and is therefore
NOT differentiable via ForwardDiff through this function). UoModel==1
hardcoded (o1=o, matching this investigation's fixed AD_PARAMS, verified in
docs/fullA_d4_code_audit.md and re-verified directly here). Single-threaded
by construction (no Threads.@threads) -- a deliberate simplification for
maximum determinism-by-inspection at this diagnostic's W=8000/D=4 scale,
documented as a known perf-vs-safety trade-off, not a bug.
"""
function hFunction_smoothed!(G::AbstractMatrix{T}, UPow::AbstractMatrix, Uσ::AbstractMatrix,
                              w::AbstractVector, τ::AbstractMatrix, σ, γ::AbstractVector,
                              Aod::AbstractMatrix, L::AbstractArray, P::AbstractArray,
                              tuner::Float64) where {T}
    D = size(τ, 1); W = size(UPow, 1)
    gdp = w .* L
    wPowσ = w .^ (1 - σ)
    pricesTemp = zeros(T, D)
    pricesTempσ = zeros(T, D)
    pricesInd = zeros(T, D)
    denom = zeros(T, D)
    constCons = zeros(T, D, D)
    constConsσ = zeros(T, D, D)
    for d in 1:D
        denom[d] = γ[d]^σ * gdp[d]
        for o in 1:D
            constCons[o, d] = w[o] * Aod[o, d] * τ[o, d]
            constConsσ[o, d] = wPowσ[o] * (Aod[o, d] * τ[o, d])^(1 - σ)
        end
    end
    @inbounds for d in 1:D
        for ω in 1:W
            for o in 1:D
                pricesTemp[o] = constCons[o, d] / UPow[ω, o]
                pricesTempσ[o] = constConsσ[o, d] / Uσ[ω, o]
            end
            smoothMinIndNew!(pricesInd, pricesTemp, D, tuner)   # softmax supplier probabilities (misc/smoothMinIndNew!.jl, unchanged)
            for o in 1:D
                d1 = d + (o - 1) * D
                G[ω, d1] = pricesTempσ[o] * pricesInd[o] - P[d1] * denom[d]
            end
        end
    end
    return nothing
end

"""
    smoothed_moments!(K, G, θ, U, obj; tuner)

Consistent-smoothing analogue of `full_aod_diag/moments_gammanorm.jl::
EK_moments_gammanorm_directgp!`: identical post-processing (gammafac
division, SamplingWeights), identical calls to the EXISTING (unmodified)
`hFunctionCounter!` and `newGravityMoment!` (both already smooth for this
config, see file header), but calls THIS file's `hFunction_smoothed!` in
place of `moments/hFunction.jl::hFunction!`. Errors loudly (does not
silently mis-apply) if the live AD_PARAMS indicator configuration differs
from the one this file's consistency argument was verified against.
"""
function smoothed_moments!(K::AbstractVector, G::AbstractMatrix, θ::AbstractVector, U::AbstractMatrix, obj; tuner::Float64)
    γobj = obj.γ
    ind = γobj.indicators
    ind.counterType == 1 || error("smoothed_moments!: only counterType==1 (autarky) verified consistent -- see file header")
    ind.localGravityMoment == 0 || error("smoothed_moments!: localGravityMoment!=0 not verified consistent (uses its own smoothMinIndNew! call inside moments/localGravityMoment!.jl, not audited here)")
    ind.UoModel == 1 || error("smoothed_moments!: only UoModel==1 verified (o1=o hardcoded in hFunction_smoothed!)")
    ind.independenceMoment == 0 || error("smoothed_moments!: independenceMoment!=0 not implemented")
    ind.GravityMomentFirstApproach == 0 || error("smoothed_moments!: GravityMomentFirstApproach!=0 not implemented")
    ind.usePMM == 0 || error("smoothed_moments!: usePMM!=0 not implemented")
    ind.NormalizeMoments == 0 || error("smoothed_moments!: NormalizeMoments!=0 not implemented")

    D = size(γobj.τ, 1); W = size(U, 1)
    μ = θ[1]; σ = θ[2]
    T = eltype(θ)

    wPrime = copy(γobj.wPrimeHat); insert!(wPrime, γobj.baseIndex, 1)
    Aod_offset = 3 + D
    Aod_θ = reshape(θ[Aod_offset+1:Aod_offset+D^2], (D, D))
    lambda = reshape(γobj.P, (D, D))'
    Aod = Aod_θ .* γobj.cHat .* (((γobj.wHat .* γobj.τ) ./ (γobj.wHat[1, 1] .* γobj.τ[1, :]')) .^ (1 / μ)) .* (lambda ./ lambda[1, :]')
    AodPow = (Aod ./ γobj.cHat) .^ (-μ)

    γvec = ones(T, D)
    γ_prime = ones(T, D); γ_prime[γobj.baseIndex] = θ[3+D]
    @. K = γ_prime[γobj.baseIndex]

    UPow = U .^ (-μ)
    UσPow = γobj.Uσ .^ (-μ)

    hFunction_smoothed!(G, UPow, UσPow, γobj.wHat, γobj.τ, σ, γvec, AodPow, γobj.L, γobj.P, tuner)
    hFunctionCounter!(K, G, UPow, UσPow, wPrime, γobj.τPrime, σ, γ_prime, AodPow, γobj.LPrime, ind.counterType, γobj.baseIndex, ind.UoModel)

    if ind.gravMoment == 1
        newGravityMoment!(G, γobj.τ, D, W, γvec, AodPow, U, ind.GravityMomentFirstApproach, ind.UoModel)
    end

    gammafac = SpecialFunctions.gamma(μ * (1 - σ) + 1)
    simple_end = D^2 + 1
    @. G[:, 1:simple_end] /= gammafac

    for im in 1:γobj.numMomentsSimple
        @. G[:, im] *= γobj.SamplingWeights[1:W]
    end
    @. K *= γobj.SamplingWeights[1:W]
    return nothing
end

"Closure factory: fixes `tuner` so the result matches the (K,G,θ,U,obj) `moments!` signature PsiObjectiveBundleImplicit expects."
make_smoothed_moments(tuner::Float64) = (K, G, θ, U, obj) -> smoothed_moments!(K, G, θ, U, obj; tuner = tuner)

"tuner (smoothMinIndNew! convention) <-> rho (CC/sequential-methodology.tex convention: soft-max via rho*log(sum(exp(./rho)))). tuner = -1/rho; rho -> 0 recovers the hard model, matching sequential_methodology.tex sec 4's rho=2e-3 production default."
rho_to_tuner(rho::Float64) = -1.0 / rho
tuner_to_rho(tuner::Float64) = -1.0 / tuner

"""
    smoothed_obj_for(ctx, tuner) -> PsiObjectiveBundleImplicit

Fresh objective bundle reusing every field of `ctx.obj` EXCEPT `moments!`
(swapped to `make_smoothed_moments(tuner)`). Reuses `CS.inner_loop_internal`
and the KNITRO inner-dual machinery completely unmodified -- per task
instruction, does not reimplement the inner CC solve, only the moment
kernel it calls.
"""
function smoothed_obj_for(ctx, tuner::Float64)
    return CS.PsiObjectiveBundleImplicit(δ = ctx.δ, find_smallest = ctx.find_smallest, γ = ctx.γ,
        (moments!) = make_smoothed_moments(tuner), moments_jacobian! = error, d = ctx.nTotalMoments,
        outer_constr_index = ctx.outer_constr_index, inequality_index = ctx.pp.inequality_index,
        complement_index = ctx.pp.complement_index, l = ctx.l_full, U = ctx.U, N = AD_PARAMS.Jac_W,
        lower_limit = -50, use_cached_x = true,
        outer_loop_opt = ctx.obj.outer_loop_opt, inner_loop_opt = ctx.obj.inner_loop_opt)
end

# ============================================================================
# BaseDualState-analogue + the three matched scalar objects (mirrors
# three_way_derivatives.jl's structural pattern exactly, per task instruction
# "your smoothed analogues should follow the same structural pattern").
# ============================================================================

struct SmoothedBaseDualState
    x_free0::Vector{Float64}
    θ_full0::Vector{Float64}
    ζstar::Float64
    λstar::Vector{Float64}
    m_star::Vector{Float64}
    inner_status::Int
    tuner::Float64
end

"Solves the SMOOTHED inner CC dual (fixed tuner) at x_free0 via the unmodified CS.inner_loop_internal, then freezes (ζ*,λ*,m*) by DEEP COPY (verified copy()/collect(), not a view into obj_smooth's mutable scratch -- this is exactly what rules out one candidate root cause for smoothed_moments.jl's original non-determinism, see file header)."
function solve_smoothed_base_state(x_free0::AbstractVector, ctx, obj_smooth, tuner::Float64)
    θ_full0 = CS.reconstruct_full(x_free0, ctx.m)
    K, inner_x, nStatus = CS.inner_loop_internal(obj_smooth, θ_full0)
    nStatus in (0, -100, -101, -103) || error("solve_smoothed_base_state: inner solve failed, nStatus=$nStatus")
    ncon = obj_smooth.d - obj_smooth.outer_constr_index + 2
    cbuf = zeros(ncon)
    obj_smooth(inner_x, constr = @view(cbuf[1:ncon]))
    m_star = copy(obj_smooth.arg1)
    return SmoothedBaseDualState(collect(x_free0), θ_full0, inner_x[1], collect(inner_x[2:end]), m_star, nStatus, tuner)
end

"""
    smoothed_fixed_dual_L(x_free, ctx, obj_smooth, base) -> Float64

L_fix(x;y*) with the SMOOTHED moments!, zeta*/lambda* frozen at `base`
(itself a smoothed-inner solve). ForwardDiff-differentiable in x_free (K,G
freshly allocated as eltype(x_free), matching three_way_derivatives.jl's
fixed_dual_L pattern exactly -- NOT obj_smooth.H, which is hardcoded
Float64).
"""
function smoothed_fixed_dual_L(x_free::AbstractVector, ctx, obj_smooth, base::SmoothedBaseDualState)
    θ_full = CS.reconstruct_full(x_free, ctx.m)
    W = size(obj_smooth.U, 1); d = obj_smooth.d
    K = zeros(eltype(θ_full), W); G = zeros(eltype(θ_full), W, d)
    obj_smooth.moments!(K, G, θ_full, obj_smooth.U, obj_smooth)
    oci = obj_smooth.outer_constr_index
    q = [-base.ζstar - dot(base.λstar, @view(G[s, 1:oci-1])) for s in 1:W]
    Psi_q = similar(q)
    CS.Psi!(Psi_q, q)
    return -(sum(Psi_q) / W + base.ζstar)
end

"Smoothed frozen-adjoint Q (linear in G, does not re-evaluate Psi) -- kept for the gradient-benchmark's directional comparison, matching three_way_derivatives.jl::frozen_adjoint_Q's sign convention exactly (see that function's docstring for the sign derivation)."
function smoothed_frozen_adjoint_Q(x_free::AbstractVector, ctx, obj_smooth, base::SmoothedBaseDualState)
    θ_full = CS.reconstruct_full(x_free, ctx.m)
    W = size(obj_smooth.U, 1); d = obj_smooth.d
    K = zeros(eltype(θ_full), W); G = zeros(eltype(θ_full), W, d)
    obj_smooth.moments!(K, G, θ_full, obj_smooth.U, obj_smooth)
    oci = obj_smooth.outer_constr_index
    acc = zero(eltype(θ_full))
    @inbounds for s in 1:W
        acc += base.m_star[s] * dot(@view(base.λstar[:]), @view(G[s, 1:oci-1]))
    end
    return (acc / W) - base.ζstar
end

"""
    smoothed_optimized_Delta(x_free, ctx, obj_smooth) -> (Delta, inner_x, nStatus)

Fully re-solved smoothed Delta(x) = min_y L(x,y) at fixed tuner (matched
value, not a hard value with smoothed gradient). Mirrors oracle.jl's sign
convention (Delta = -f) exactly.
"""
function smoothed_optimized_Delta(x_free::AbstractVector, ctx, obj_smooth)
    θ_full = CS.reconstruct_full(x_free, ctx.m)
    K, inner_x, nStatus = CS.inner_loop_internal(obj_smooth, θ_full)
    if !(nStatus in (0, -100, -101, -103))
        return NaN, inner_x, nStatus
    end
    ncon = obj_smooth.d - obj_smooth.outer_constr_index + 2
    cbuf = zeros(ncon)
    obj_smooth(inner_x, constr = @view(cbuf[1:ncon]))
    Delta_dual = cbuf[1] / 1e10
    return Delta_dual, inner_x, nStatus
end

# ============================================================================
# Diagnostics: winner-probability entropy / boundary mass (Step 5).
# ============================================================================

"""
    winner_softmax_probs(θ_full, ctx, tuner) -> Array{Float64,3} (W x D x D)

`probs[ω,o,d]` = softmax supplier probability of origin o winning
destination d on draw ω, at the given tuner. Own fresh computation (not
reusing smoothed_moments.jl).
"""
function winner_softmax_probs(θ_full::AbstractVector, ctx, tuner::Float64)
    price, Aod, AodPow = factual_prices(θ_full, ctx)   # winners.jl, validated bit-exact vs hFunction!
    D = ctx.D; W = size(price, 1)
    probs = Array{Float64}(undef, W, D, D)
    xInd = zeros(D)
    @inbounds for d in 1:D, ω in 1:W
        @views smoothMinIndNew!(xInd, price[ω, :, d], D, tuner)
        probs[ω, :, d] .= xInd
    end
    return probs
end

"Shannon entropy (nats) of a probability vector, 0 for degenerate (one-hot) vectors."
function entropy_nats(p::AbstractVector)
    s = 0.0
    for pi in p
        pi > 0 && (s -= pi * log(pi))
    end
    return s
end

"""
    winner_entropy_diagnostics(probs) -> (mean_entropy, max_entropy, frac_boundary)

`mean_entropy`/`max_entropy` over all (ω,d) softmax distributions (nats, max
possible = log(D)). `frac_boundary` = fraction of (ω,d) draws whose top-1
probability mass is < 0.99 (i.e. NOT cleanly resolved into a 0/1 split) --
the task's "effective boundary mass" diagnostic.
"""
function winner_entropy_diagnostics(probs::Array{Float64,3})
    W, D, _ = size(probs)
    ents = Float64[]
    top1 = Float64[]
    for d in 1:D, ω in 1:W
        p = @view probs[ω, :, d]
        push!(ents, entropy_nats(p))
        push!(top1, maximum(p))
    end
    return (mean_entropy = mean(ents), max_entropy = maximum(ents), max_possible_entropy = log(D),
            frac_boundary_lt_099 = mean(top1 .< 0.99), mean_top1 = mean(top1))
end
