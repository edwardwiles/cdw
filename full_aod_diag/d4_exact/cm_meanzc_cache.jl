# ============================================================================
# Exact-point cache for the CM+moments(+ZC) extension (task brief 9.2/4/11),
# extending the current pattern (CMEvalKey/SafeExactCache{CMEvalKey}/
# cm_cache_key, cm_config.jl) rather than forking a new mechanism. Reuses
# SafeExactCache (oracle.jl) UNCHANGED -- it is already generic over the key
# type. A CMMeanZCEvalKey point must NEVER collide with a CMEvalKey
# (CM-only) or FullAEvalKey (unrestricted) point: each extension/config gets
# its OWN physically separate SafeExactCache instance (cm_meanzc_oracle_cache_for
# below), never a shared Dict keyed by a union type.
#
# "Accepted-point reuse requires exact equality of the full outer vector
# including eta_nu" (task brief 9.2): eta_nu (equivalently nu) is part of the
# KEY itself, alongside x_free -- a cache hit at the same x_free but a
# DIFFERENT nu is, correctly, a cache MISS (they are different points in the
# outer decision space, not the same point re-evaluated).
# ============================================================================

"""
    CMMeanZCEvalKey

Exact-point cache key for the CM+moments(+ZC) path -- `x_free` AND `eta_nu`
(the full outer vector `w_ext = [gp; zfree; eta_nu]`'s economic and nu
components respectively) plus δ/find_smallest/inner_loop_opt/context
fingerprint (mirrors `CMEvalKey`'s own fields) plus a `meanzc::NamedTuple`
fragment (`cm_meanzc_cache_key`) identifying the FULL configuration: K_mean,
K_pair, meanzc_basis, CM grid size/cutpoints/contrasts, and a draw checksum.
Never let two different configurations (e.g. K_mean=1 vs K_mean=2, or
different CM cutpoints) collide in the same cache -- `meanzc` being part of
the key/hash makes any such difference a guaranteed miss, not silently wrong
data.
"""
struct CMMeanZCEvalKey
    x_free::Vector{Float64}
    eta_nu::Vector{Float64}
    δ::Float64
    find_smallest::Bool
    inner_loop_opt::String
    meanzc::NamedTuple
    ctx_fingerprint::String
end
Base.:(==)(a::CMMeanZCEvalKey, b::CMMeanZCEvalKey) = a.x_free == b.x_free && a.eta_nu == b.eta_nu &&
    a.δ == b.δ && a.find_smallest == b.find_smallest && a.inner_loop_opt == b.inner_loop_opt &&
    a.meanzc == b.meanzc && a.ctx_fingerprint == b.ctx_fingerprint
Base.hash(k::CMMeanZCEvalKey, h::UInt) = hash((k.x_free, k.eta_nu, k.δ, k.find_smallest, k.inner_loop_opt, k.meanzc, k.ctx_fingerprint), h)

"Fresh, empty CM+moments(+ZC) exact-point cache -- mirrors cm_config.jl's cm_oracle_cache_for, scoped to CMMeanZCEvalKey. NEVER share this Dict/lock with a CMEvalKey or FullAEvalKey cache."
cm_meanzc_oracle_cache_for(pcx) = SafeExactCache{CMMeanZCEvalKey}()

"""
    cm_meanzc_cache_key(K_mean, K_pair, meanzc_basis, L, probs, contrasts, draw_checksum; backend=:structured) -> NamedTuple

Hashable, comparable key fragment identifying a CM+moments(+ZC) configuration
(everything about the configuration EXCEPT the evaluation point itself,
which lives in `CMMeanZCEvalKey.x_free`/`.eta_nu`). `draw_checksum` should be
a cheap deterministic hash of `ctx.U` (or a stored `draw_seed`) -- NOT
recomputed here, matching `cm_cache_key`'s own convention. `backend` mirrors
`CMEvalKey`'s own inclusion of `cm_hessian_backend`: even though Architecture
C's structured Hessian is validated to reproduce the dense reference to
~1e-15 (so a backend change alone cannot legitimately change the cached
VALUE), it is still included in the key so a backend change is a guaranteed
cache MISS rather than a silent reuse across an unvalidated future backend --
only `:structured` (Architecture C) is wired for this extension today.
"""
function cm_meanzc_cache_key(K_mean::Int, K_pair::Int, meanzc_basis::Symbol, L::Int,
                              probs::AbstractVector{Float64}, contrasts::Symbol, draw_checksum;
                              backend::Symbol = :structured)
    return (K_mean = K_mean, K_pair = K_pair, meanzc_basis = meanzc_basis,
            L = L, cutpoints = Tuple(probs), contrasts = contrasts, backend = backend, draw_checksum = draw_checksum)
end

"""
    cm_meanzc_production_value_verified_cached(x_free0, νvec, pcx; cache=nothing, use_cache=true,
        probs, draw_checksum=nothing) -> (K, base, verify)

Cache-aware analogue of `cm_meanzc_production_value_verified`. `cache` is a
`SafeExactCache{CMMeanZCEvalKey}` (`cm_meanzc_oracle_cache_for`); when
supplied and `use_cache=true`, an exact repeat of `(x_free0, νvec, δ,
find_smallest, inner_loop_opt, K_mean, K_pair, meanzc_basis, L, probs,
contrasts, draw_checksum, ctx_fingerprint)` returns without invoking the
inner KNITRO solve at all. Only a `is_verified_success` result is ever
stored (same cacheability contract as `oracle.jl`'s exact-point cache for the
unrestricted/CM-only paths) -- an infeasible/unverified point is never
cached as if it were a valid answer.
"""
function cm_meanzc_production_value_verified_cached(x_free0::AbstractVector, νvec::AbstractVector{Float64}, pcx;
        cache = nothing, use_cache::Bool = true,
        probs::AbstractVector{Float64} = pcx.aug.z, draw_checksum = nothing, backend::Symbol = :structured)
    obj = pcx.ctx_cm.obj
    key = nothing
    if cache !== nothing && use_cache
        meanzc = cm_meanzc_cache_key(pcx.aug.K_mean, pcx.aug.K_pair, pcx.aug.meanzc_basis, pcx.aug.L,
                                      probs, pcx.aug.contrasts, draw_checksum; backend = backend)
        key = CMMeanZCEvalKey(collect(x_free0), collect(νvec), obj.δ, obj.find_smallest, obj.inner_loop_opt,
                               meanzc, context_fingerprint(pcx.ctx_cm))
        hit = _cache_lookup(cache, key)
        if hit !== nothing
            return hit.K, hit.base, hit.verify
        end
    end

    K, base, verify = cm_meanzc_production_value_verified(x_free0, νvec, pcx)

    if key !== nothing && is_verified_success(verify)
        _cache_store!(cache, key, (K = K, base = base, verify = verify))
    end
    return K, base, verify
end
