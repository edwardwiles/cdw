# ================================================================================================
# Phase C remediation (production-audit continuation, 2026-07-26): exact-point cache for the four
# restricted families' PUBLIC drivers. A complete generic cache primitive already exists
# (SafeExactCache{K}, oracle.jl) and is wired into the UNRESTRICTED driver by default -- but
# `cm_config.jl`'s own CMEvalKey/cm_production_value_v2 is built against a DIFFERENT, incompatible
# `pcx` shape (`pcx.cfg::CMConfig`, `cm_base_state_v2`) than what `run_cm_upper_checkpointed`/
# `run_originzc_upper_checkpointed` actually build (`build_cm_production_context`'s own
# `(ctx_cm, aug, bins, cctx)`, `archC_verified_state`) -- so it is not a drop-in for the real
# production drivers, despite existing in the tree. This file defines a NEW key type against the
# REAL production `pcx`/`ctx`/`cctx` shapes, reusing SafeExactCache/_cache_lookup/_cache_store!/
# context_fingerprint unchanged (oracle.jl), not re-deriving any of that machinery.
#
# Key design follows the SAME "key on the canonical decoded x_free, not the raw outer search
# coordinate" convention the UNRESTRICTED family's own FullAEvalKey already uses (oracle.jl) --
# audited and confirmed SAFE in the baseline static audit (a hit across A_coordinate_mode values
# is intentional, correct reuse of an equal mathematical point, not aliasing).
# ================================================================================================

"""
    CMProductionEvalKey

Exact-point cache key for the four restricted families' real public drivers
(run_cm_upper_checkpointed / run_originzc_upper_checkpointed). Deliberately includes every field
that changes the MATHEMATICAL value at this point -- same discipline as oracle.jl's FullAEvalKey.
`nu` is the restriction-parameter vector (mean/pair eta_nu for CM+ZC, eta_origin for origin-ZC;
empty for plain flexible CM / common-Frechet, which have no free restriction parameters).
`family_tag` distinguishes flexible_cm / common_frechet / cm_meanzc / origin_zc so a point from one
family/config can never collide with another even if x_free happened to coincide numerically.
"""
struct CMProductionEvalKey
    x_free::Vector{Float64}
    nu::Vector{Float64}
    delta::Float64
    find_smallest::Bool
    inner_loop_opt::String
    family_tag::Symbol
    L::Int
    contrasts::Symbol
    K_mean::Int
    K_pair::Int
    A_coordinate_mode::Symbol
    ctx_fingerprint::String
end
Base.:(==)(a::CMProductionEvalKey, b::CMProductionEvalKey) =
    a.x_free == b.x_free && a.nu == b.nu && a.delta == b.delta && a.find_smallest == b.find_smallest &&
    a.inner_loop_opt == b.inner_loop_opt && a.family_tag == b.family_tag && a.L == b.L &&
    a.contrasts == b.contrasts && a.K_mean == b.K_mean && a.K_pair == b.K_pair &&
    a.A_coordinate_mode == b.A_coordinate_mode && a.ctx_fingerprint == b.ctx_fingerprint
Base.hash(k::CMProductionEvalKey, h::UInt) = hash((k.x_free, k.nu, k.delta, k.find_smallest,
    k.inner_loop_opt, k.family_tag, k.L, k.contrasts, k.K_mean, k.K_pair, k.A_coordinate_mode,
    k.ctx_fingerprint), h)

"Fresh, empty exact-point cache for the restricted-family production drivers -- one per run, mirrors oracle.jl's oracle_cache_for."
cm_production_exact_cache() = SafeExactCache{CMProductionEvalKey}()

"""
    CMExactCacheCounters

Runtime evidence (task Phase C, required public counters): lookups/hits/misses/store_rejections.
A `store_rejection` is an attempted store of a result that should never be cached (mirrors
oracle.jl's own is_cacheable_result discipline -- only genuinely feasible, verified solves are
cached; anything else is a rejection, not a silent no-op).
"""
mutable struct CMExactCacheCounters
    lookups::Int
    hits::Int
    misses::Int
    store_rejections::Int
end
CMExactCacheCounters() = CMExactCacheCounters(0, 0, 0, 0)
const CM_EXACT_CACHE_COUNTERS = Ref(CMExactCacheCounters())
reset_cm_exact_cache_counters!() = (CM_EXACT_CACHE_COUNTERS[] = CMExactCacheCounters())

"Prints the live exact-cache counters (task Phase C required public counters). Call AFTER a
solve/benchmark, not at startup (mirrors print_core_hessian_counters' own discipline)."
function print_cm_exact_cache_counters(c::CMExactCacheCounters = CM_EXACT_CACHE_COUNTERS[])
    println("[cm-exact-cache] lookups=", c.lookups, " hits=", c.hits, " misses=", c.misses,
            " store_rejections=", c.store_rejections,
            " hit_rate=", c.lookups == 0 ? "n/a" : round(c.hits / c.lookups, digits = 4))
end

"""
    cm_cache_lookup_or_compute!(cache, key, compute_fn) -> (base, verify)

`compute_fn()` must return `(base, verify)`. On a cache hit, returns the stored pair without
calling `compute_fn` (the real inner-solve skip this cache exists for) and increments `hits`;
on a miss, calls `compute_fn`, stores `(base, verify)` iff `verify.inner_status` is a genuine
feasible/verified code (same discipline as `archC_verified_state`'s own `nStatus in (0,-100,-101,-103)`
gate -- never cache an infeasible/failed solve as if it were reusable), and increments `misses`
(+`store_rejections` if the store was skipped for a non-feasible result).
`cache === nothing` (use_exact_cache=false) always calls `compute_fn` directly, zero overhead.
"""
function cm_cache_lookup_or_compute!(cache, key::Union{Nothing,CMProductionEvalKey}, compute_fn)
    if cache === nothing || key === nothing
        return compute_fn()
    end
    CM_EXACT_CACHE_COUNTERS[].lookups += 1
    hit = _cache_lookup(cache, key)
    if hit !== nothing
        CM_EXACT_CACHE_COUNTERS[].hits += 1
        return hit.base, hit.verify
    end
    CM_EXACT_CACHE_COUNTERS[].misses += 1
    base, verify = compute_fn()
    if verify.inner_status in (0, -100, -101, -103)
        _cache_store!(cache, key, (base = base, verify = verify))
    else
        CM_EXACT_CACHE_COUNTERS[].store_rejections += 1
    end
    return base, verify
end
