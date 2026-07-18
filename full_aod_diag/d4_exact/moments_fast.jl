# ============================================================================
# Continuation 5, Priority 2 addendum: audit + optimize the structural
# moment-construction hot path (moments!/hFunction!), per explicit user
# request. Purely additive -- mirrors, never modifies,
# full_aod_diag/moments_gammanorm.jl::EK_moments_gammanorm_directgp! (the
# ACTUAL moments! this investigation's ctx.obj uses, confirmed from
# context.jl's `(moments!) = EK_moments_gammanorm_directgp!`) and
# moments/hFunction.jl (unmodified, called verbatim).
#
# ---- 1. Call chain (confirmed by direct code reading, this session) ----
#
# evaluate_fullA / solve_base_state
#   -> CS.inner_loop_internal(obj, θ_full)
#        -> obj.moments!(K, G, θ_full, obj.U, obj)   ==  EK_moments_gammanorm_directgp!
#             1. unpack obj.γ (wHat, L, τ, P, Uσ, cHat, UPow_scratch/UσPow_scratch, ...)
#             2. mu = θ[1], sigma = θ[2]   -- READ from θ, but in THIS
#                investigation's ctx (context.jl: free_idx = [3+D, Aod_offset+1:...])
#                indices 1,2 (mu,sigma) are NEVER in free_idx -- they are
#                box-constrained to a single fixed value (theta_lo[1]==theta_hi[1])
#                for the ENTIRE lifetime of one ctx/obj. CONFIRMED (not assumed)
#                by direct query: free_idx=[7..23] at D=4, so indices 1,2 are
#                always fixed_vals, never touched by any outer-loop perturbation.
#             3. build Aod (level) from Aod_theta (the free outer param) + a
#                closed-form gauge transform (cHat, wHat, tau, lambda) -- O(D^2),
#                cheap, INDEPENDENT of the draws (correctly hoisted outside any
#                draw loop already).
#             4. AodPow = (Aod./cHat).^(-mu)            -- O(D^2), cheap.
#             5. gamma/gamma_prime setup (gammanorm: forced to 1 except focal) -- O(D).
#             6. K[:] = gamma_prime[baseIndex]            -- O(W) assignment, trivial.
#             7. **UPow = U.^(-mu); UσPow = Uσ.^(-mu)**  -- O(W*D) power ops,
#                RECOMPUTED EVERY CALL despite mu being PROVABLY CONSTANT for
#                the ctx's entire lifetime (see item 2). THE finding this
#                addendum's item 2 asks to check for -- see "2. Fixed-mu,sigma
#                audit" below for the measured verdict.
#             8. Threads.@threads over Th=nthreads() draw CHUNKS (already
#                thread-parallel over draws in production -- see "5. Threading"
#                below), each chunk calls:
#                  hFunction!(view(G,chunk,:), view(UPow,chunk,:), view(UσPow,chunk,:), ...)
#                    -> for each destination d, for each origin o: pricesTemp[o] =
#                       constCons[o,d]/UPow[ω,o] (O(D) per draw), MinInd!
#                       (O(D) hard-min), then fills D moment columns.
#                  hFunctionCounter!(view(K,chunk), view(G,chunk,:), ...)
#                    -> same shape, autarky (counterType==1) branch is a single
#                       VECTORIZED broadcast (moments/hFunction.jl:201, no draw
#                       loop at all) -- already about as cheap as this can be.
#             9. gravMoment==1: newGravityMoment!(...) -- confirmed cheap
#                elsewhere (docs/fullA_smoothed_consistent_experiment.md sec 0:
#                gravity_value bit-identical across every theta tried).
#            10. post-processing: divide by gamma(mu*(1-sigma)+1) (O(W*D),
#                one scalar divide per moment cell), PMM subtract (usePMM==1?
#                -- NOT ACTIVE for this investigation, see audit below),
#                NormalizeMoments (O(W*moments)), SamplingWeights multiply
#                (O(W*moments)).
#
# Diagnostic-only code NOT on this path (confirmed inactive for this
# investigation's AD_PARAMS, not touched): localGravityMoment!/
# localGravityCrossMoment! (localGravityMoment=0), smoothMinIndNew! (commented
# out in hFunction.jl, MinInd! is what's actually called), the
# counterType!=1 branches of hFunction!/hFunctionCounter! (counterType=1,
# autarky, throughout).
# ============================================================================
# NOTE: does NOT include(context.jl) -- per this directory's established convention (see
# composite_gradient.jl, which likewise assumes the caller already loaded context.jl/context_scaled.jl
# first). Re-including context.jl from two different top-level scripts in the same process causes a
# genuine binding-ambiguity error (Julia's `include` is not idempotent -- caught directly this session
# via profile_moments_fast.jl's D=4/W=80000 benchmark, which needs context_scaled.jl, itself already
# include-ing context.jl).
using UnPack

"""
    MuSigmaPowCache

Per-ctx cache for `U.^(-mu)` / `Uσ.^(-mu)` -- the ONE fixed-mu,sigma
redundant computation this audit found (see moments_fast.jl module docstring
item 2). `mu`/`sigma` are provably invariant for a ctx's lifetime in this
investigation (context.jl pins theta_lo[1]==theta_hi[1], theta_lo[2]==
theta_hi[2]); `valid=false` forces one recompute on first use (or if the
cache is ever asked for a DIFFERENT mu -- checked, not assumed, so this
cache is safe to use even outside this investigation's fixed-mu convention:
it degrades to "recompute every time mu changes," never returns a wrong
value for the wrong mu).
"""
mutable struct MuSigmaPowCache
    mu::Float64
    UPow::Matrix{Float64}
    UσPow::Matrix{Float64}
    valid::Bool
    n_recompute::Int
    n_reuse::Int
end
MuSigmaPowCache(U::AbstractMatrix, Uσ::AbstractMatrix) =
    MuSigmaPowCache(NaN, zeros(size(U)), zeros(size(Uσ)), false, 0, 0)

"Populate (if stale) or reuse (if valid at this mu) the cache's UPow/UσPow. Returns (UPow, UσPow) views into the cache's OWN storage -- caller must not mutate them."
function get_upow!(cache::MuSigmaPowCache, U::AbstractMatrix, Uσ::AbstractMatrix, μ::Float64)
    if !cache.valid || cache.mu !== μ
        Th = Threads.nthreads()
        W = size(U, 1)
        Threads.@threads for t in 1:Th
            ix0 = round(Int, (t - 1) / Th * W) + 1
            ix1 = round(Int, t / Th * W)
            @. cache.UPow[ix0:ix1, :] = U[ix0:ix1, :] ^ (-μ)
            @. cache.UσPow[ix0:ix1, :] = Uσ[ix0:ix1, :] ^ (-μ)
        end
        cache.mu = μ
        cache.valid = true
        cache.n_recompute += 1
    else
        cache.n_reuse += 1
    end
    return cache.UPow, cache.UσPow
end

"""
    EK_moments_gammanorm_directgp_fast!(K, G, θ, U, obj, pow_cache)

Byte-for-byte mirror of `EK_moments_gammanorm_directgp!`
(full_aod_diag/moments_gammanorm.jl), EXCEPT step 7 (UPow/UσPow) is served
from `pow_cache` (a `MuSigmaPowCache`) instead of recomputed unconditionally.
`hFunction!`/`hFunctionCounter!`/`newGravityMoment!` are called VERBATIM,
unmodified, imported not copied. Equivalence with the original is mandatory
before use -- see test_moments_fast.jl.
"""
function EK_moments_gammanorm_directgp_fast!(K, G, θ, U, obj, pow_cache::MuSigmaPowCache)
    @unpack wHat, L, LPrime, τ, τPrime, P, σ_Moments, baseIndex, refIndex1, indicators, Uσ, μHat, CDF_Moments, Ind_Moments, cHat, IndCDF_Cells, Ū, numMomentsSimple, SamplingWeights, PMM, moments_without_var, UPow_scratch, UσPow_scratch = obj.γ
    @unpack counterExplicit,
    counterType,
    θConstant,
    gravMoment,
    localGravityMoment,
    GravityMomentFirstApproach,
    sameMarginalsMoment,
    independenceMoment,
    momentOrder,
    momentOrderForBaseIndex,
    IndMomentOrder,
    OuterScaling,
    usePMM,
    UoModel,
    NormalizeMoments = indicators

    counterType == 1 || error("EK_moments_gammanorm_directgp_fast! only implements counterType==1 (autarky), matches the original")

    W = size(U, 1)
    D = size(τ, 1)
    T = eltype(θ)

    μ = θ[1]
    σ = θ[2]

    wPrime = copy(obj.γ.wPrimeHat)
    insert!(wPrime, baseIndex, 1)

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
    γ_prime = ones(T, D)
    γ_prime[baseIndex] = θ[3+D]

    if counterExplicit == 0
        counterVal = γ_prime[baseIndex]
        @. K[:] = counterVal
    end

    if θConstant != 1
        # ---- THE FIX: serve UPow/UσPow from pow_cache instead of recomputing
        # unconditionally (only path that changed vs the original function) ----
        if eltype(γ) === Float64 && T === Float64
            UPow, UσPow = get_upow!(pow_cache, U, Uσ, μ)
        else
            # non-Float64 eltype (e.g. a ForwardDiff Dual path perturbing mu itself)
            # -- NOT this investigation's path (mu is always fixed Float64 here,
            # confirmed above), but kept correct rather than erroring, matching
            # the original function's own eltype(γ)===Float64 branch discipline.
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
            hFunctionCounter!(@view(K[ix0:ix1, :]), @view(G[ix0:ix1, :]), @view(UPow[ix0:ix1, :]), @view(UσPow[ix0:ix1, :]), wPrime, τPrime, σ, γ_prime, AodPow, LPrime, counterType, baseIndex, UoModel)
        end
    else
        Th = Threads.nthreads()
        Threads.@threads for t ∈ 1:Th
            ix0 = round(Int, (t - 1) / Th * W) + 1
            ix1 = round(Int, t / Th * W)
            hFunction!(@view(G[ix0:ix1, :]), @view(U[ix0:ix1, :]), @view(Uσ[ix0:ix1, :]), wHat, τ, σ, γ, AodPow, L, P, counterType, gravMoment, localGravityMoment, GravityMomentFirstApproach, independenceMoment, μHat, UoModel)
            hFunctionCounter!(@view(K[ix0:ix1, :]), @view(G[ix0:ix1, :]), @view(U[ix0:ix1, :]), @view(Uσ[ix0:ix1, :]), wPrime, τPrime, σ, γ_prime, AodPow, LPrime, counterType, baseIndex, UoModel)
        end
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
    enable_pow_cache!(ctx) -> MuSigmaPowCache

Opt-in wiring of the fixed-mu,sigma `MuSigmaPowCache` into the LIVE oracle path.

`ctx.obj` is a `PsiObjectiveBundleImplicit` (`cc_algo/PsiObjectiveBundle.jl`), a
**mutable** `@with_kw` struct whose `moments!::Function` field is invoked verbatim
by every live consumer -- `inner_loop_internal` (`oracle.jl`, `oracle_fast.jl`,
`oracle_profiled.jl`), `lfix_incremental.jl`, and the KNITRO F+G callback path all
call `obj.moments!(K, G, θ, obj.U, obj)`. Rebinding that ONE field to a closure that
dispatches to `EK_moments_gammanorm_directgp_fast!` (with a per-ctx `MuSigmaPowCache`)
therefore threads the cache through the entire live path with a single field mutation
-- no context-builder mirror, no change to any call site.

This is a strict, equivalence-tested refactor of the moment build (byte-identical
output; see `test_moments_fast.jl` and `verify_pow_cache_wiring.jl`): the ONLY change
is that `U.^(-μ)`/`Uσ.^(-μ)` are served from a value cache keyed on `μ` instead of
being recomputed on every call. `μ`/`σ` are provably invariant for a ctx's lifetime in
this investigation (`context.jl` pins `θ_lo[1]==θ_hi[1]`, `θ_lo[2]==θ_hi[2]`; `free_idx`
never touches indices 1,2), so after the first call the cache is pure reuse; `get_upow!`
still recomputes if ever asked for a different `μ`, so the wiring is safe even outside
that convention.

Opt-in by design: `d4_exact_setup()` is left unchanged, so existing callers/tests are
unaffected unless they explicitly call this. Returns the `MuSigmaPowCache` so callers
can inspect `.n_recompute`/`.n_reuse` (the closure retains its own reference).

    ctx = d4_exact_setup()
    pow_cache = enable_pow_cache!(ctx)   # every subsequent live moments! build uses the cache
"""
function enable_pow_cache!(ctx)
    obj = ctx.obj
    pow_cache = MuSigmaPowCache(obj.U, obj.γ.Uσ)
    obj.moments! = (K, G, θ, U, o) -> EK_moments_gammanorm_directgp_fast!(K, G, θ, U, o, pow_cache)
    return pow_cache
end
