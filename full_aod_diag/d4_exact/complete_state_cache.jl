# ============================================================================
# Overnight task (2026-07-22), Priority 2: complete-state exact/cross-delta cache.
#
# See docs/COMPLETE_STATE_CACHE_DESIGN_2026-07-22.md for the full field-by-field mutability
# audit and fingerprint design this file implements. Summary: `BaseDualState` (three_way_
# derivatives.jl) and the `verify` NamedTuple (archC_verified_state, cm_production_bundle.jl)
# are ALREADY fully-owned, non-aliased data (every array field is a fresh `collect`/`copy`, never
# a scratch-buffer reference) -- proven by direct code read, not assumed, and already exploited
# by cm_checkpoint.jl's own `last_F_state[]` pattern. This lets the cache store/return them by
# direct reference with no extra defensive copying.
#
# PURELY ADDITIVE, OPT-IN, DEFAULT OFF: does not modify archC_base_state, archC_verified_state,
# cm_production_value_verified, cm_production_gradient, or cm_production_gradient_cplus. Nothing
# in cm_checkpoint.jl's default (:reference, no cache) call path touches this file.
# ============================================================================
using SHA

"One complete-state cache entry: a verified BaseDualState + its verify NamedTuple, stored by
direct reference (see this file's header for why that is safe), plus bookkeeping for LRU/bytes."
struct CompleteStateEntry
    base::BaseDualState
    verify::NamedTuple
    bytes::Int
    stored_at::Float64
end

"""
    CompleteStateCache(; max_entries=64)

Bounded, opt-in cache mapping `(context_fingerprint, outer_point_key) -> CompleteStateEntry`.
`enabled=false` by default at the CALL SITE (this struct itself has no "disabled" mode -- callers
gate at `complete_state_lookup!`/`complete_state_store!`'s own call sites, matching every other
opt-in feature in this codebase's own convention of a caller-level `Union{Nothing,X}` rather than
an internal enabled flag duplicated inside X). LRU via a monotonic access counter + O(n) scan on
eviction (n = max_entries, expected small -- see design doc §6 for why no external
OrderedDict/DataStructures.jl dependency is introduced).
"""
mutable struct CompleteStateCache
    max_entries::Int
    entries::Dict{UInt64,CompleteStateEntry}
    access::Dict{UInt64,Int}
    access_counter::Int
    hits::Int
    misses::Int
    evictions::Int
    inner_solves_avoided::Int
    bytes_current::Int
    bytes_peak::Int
    hit_restore_wall::Float64
    fresh_base_solve_wall_counterfactual::Float64
end
CompleteStateCache(; max_entries::Int = 64) =
    CompleteStateCache(max_entries, Dict{UInt64,CompleteStateEntry}(), Dict{UInt64,Int}(), 0,
                        0, 0, 0, 0, 0, 0, 0.0, 0.0)

"sha256 hex digest of an arbitrary printable value -- used for the opt-file-content fragment of the fingerprint (a filename alone must not stand in for its contents, see design doc §4)."
sha256_hex_of_string(s::AbstractString) = bytes2hex(SHA.sha256(s))

"""
    complete_state_fingerprint(ctx, L, contrasts, probs, cm_basis, cm_hessian_backend,
                                find_smallest, opt_file_path, knitro_version;
                                schema::Int=1) -> UInt64

Context fingerprint per design doc §4: draws (checksum_uniform/checksum_transformed), W, D,
sigma, CM config (L, contrasts, probs, basis, Hessian backend), find_smallest, the .opt file's
own CONTENT hash (not just its path), the KNITRO release string, and a schema version for this
cache itself. Does NOT include `delta` (Delta*(theta) is budget-independent, design doc §3) or
`cm_gradient_backend` (both outer-gradient backends read the same cached base, design doc §4).
"""
function complete_state_fingerprint(ctx, L::Int, contrasts::Symbol, probs::AbstractVector{Float64},
        cm_basis::Symbol, cm_hessian_backend::Symbol, find_smallest::Bool,
        opt_file_path::AbstractString, knitro_version::AbstractString; schema::Int = 1)
    opt_content_hash = isfile(opt_file_path) ? sha256_hex_of_string(read(opt_file_path, String)) : "MISSING:$opt_file_path"
    W, D = size(ctx.U)
    # Real D=20 production contexts (d20_real_setup_design, draw_design.jl) carry a `draw_meta`
    # NamedTuple with pre-computed SHA256 checksums of the draws (checksum_uniform/
    # checksum_transformed) -- reused here, not re-hashed, matching the SAME checksums
    # cm_checkpoint.jl's own resume-validation already trusts. D=4 test/diagnostic contexts
    # (d4_exact_setup, context.jl) have no `draw_meta` field at all; fall back to hashing `ctx.U`
    # directly (`sha256_of_matrix`, oracle.jl, the SAME hash primitive draw_meta itself is built
    # from) so the fingerprint is still a genuine draws fingerprint, not a placeholder, at D=4.
    checksum_u, checksum_t = hasproperty(ctx, :draw_meta) ?
        (ctx.draw_meta.checksum_uniform, ctx.draw_meta.checksum_transformed) :
        (sha256_of_matrix(ctx.U), "no_draw_meta_field")
    return hash((schema, checksum_u, checksum_t,
                 W, D, ctx.σ, L, contrasts, collect(probs), cm_basis, cm_hessian_backend,
                 find_smallest, opt_content_hash, knitro_version))
end

"Outer-point key: the outer point IS the cache key within a fixed context (design doc §4) -- rounded to 12 digits so bit-noise in an otherwise-identical KNITRO-reported point does not force a spurious miss."
outer_point_key(x_free0::AbstractVector) = hash(round.(x_free0, digits = 12))

"""
    complete_state_lookup!(cache, fp, x_free0) -> Union{Nothing,CompleteStateEntry}

Returns the cached entry on a hit (bumping `hits`/`inner_solves_avoided`/LRU access order), or
`nothing` on a miss (bumping `misses`) -- caller falls back to a fresh solve. `fp` is the
CALLER's own `complete_state_fingerprint(...)` for its current context; a fingerprint mismatch is
structurally impossible to observe here (a different fingerprint is a different Dict key, i.e. a
miss, not a "wrong hit") -- see `test_complete_state_cache.jl` for an explicit test of exactly
this (changed draws / changed CM config / changed option-file content / changed KNITRO release
each produce a miss, not a stale hit).
"""
function complete_state_lookup!(cache::CompleteStateCache, fp::UInt64, x_free0::AbstractVector)
    key = hash((fp, outer_point_key(x_free0)))
    entry = get(cache.entries, key, nothing)
    if entry === nothing
        cache.misses += 1
        return nothing
    end
    t0 = time()
    cache.access_counter += 1
    cache.access[key] = cache.access_counter
    cache.hits += 1
    cache.inner_solves_avoided += 1
    cache.hit_restore_wall += time() - t0
    return entry
end

"""
    complete_state_store!(cache, fp, x_free0, base, verify; solve_wall=NaN)

Stores a verified `(base, verify)` pair. Caller MUST have already confirmed
`is_verified_success(verify)` (oracle.jl) -- this function does not re-check it (design doc §5:
enforced structurally by only ever being called from inside the same try/catch that narrows to
`CMExpectedSolveFailure`, not by a second, potentially-divergent runtime check here). Evicts the
least-recently-used entry first if `max_entries` would be exceeded.
"""
function complete_state_store!(cache::CompleteStateCache, fp::UInt64, x_free0::AbstractVector,
        base::BaseDualState, verify::NamedTuple; solve_wall::Float64 = NaN)
    key = hash((fp, outer_point_key(x_free0)))
    haskey(cache.entries, key) && return cache.entries[key]   # already cached, no-op (idempotent)
    if length(cache.entries) >= cache.max_entries
        evict_key = argmin(cache.access)
        evicted = pop!(cache.entries, evict_key)
        pop!(cache.access, evict_key)
        cache.bytes_current -= evicted.bytes
        cache.evictions += 1
    end
    nbytes = Base.summarysize(base) + Base.summarysize(verify)
    entry = CompleteStateEntry(base, verify, nbytes, time())
    cache.entries[key] = entry
    cache.access_counter += 1
    cache.access[key] = cache.access_counter
    cache.bytes_current += nbytes
    cache.bytes_peak = max(cache.bytes_peak, cache.bytes_current)
    isfinite(solve_wall) && (cache.fresh_base_solve_wall_counterfactual += solve_wall)
    return entry
end

"""
    archC_verified_state_cached!(cache, fp, x_free0, ctx_cm, cctx) -> (base, verify, from_cache::Bool)

Opt-in cache-aware wrapper around `archC_verified_state` (cm_production_bundle.jl, UNMODIFIED).
On a hit, returns the cached `(base, verify)` directly (no inner solve). On a miss, calls the
real `archC_verified_state`, and stores the result ONLY if `is_verified_success(verify)` (never
stores an unverified/failed/ExactInfeasible result, design doc §5). `cache === nothing` (the
default everywhere this is not explicitly opted into) makes this IDENTICAL to calling
`archC_verified_state` directly, plus the `from_cache=false` tag.
"""
function archC_verified_state_cached!(cache::Union{Nothing,CompleteStateCache}, fp::UInt64,
        x_free0::AbstractVector, ctx_cm, cctx)
    if cache !== nothing
        hit = complete_state_lookup!(cache, fp, x_free0)
        hit !== nothing && return hit.base, hit.verify, true
    end
    t0 = time()
    base, verify = archC_verified_state(x_free0, ctx_cm, cctx)
    solve_wall = time() - t0
    if cache !== nothing && is_verified_success(verify)
        complete_state_store!(cache, fp, x_free0, base, verify; solve_wall = solve_wall)
    end
    return base, verify, false
end
