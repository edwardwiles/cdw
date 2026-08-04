# task §7 (profiled-outer-ab-readiness-2026-08-04): safe REDUCED bandwidth-search reuse.
# ADDITIVE ONLY -- profiled_lfix_incremental_2026-08-01.jl's own profiled_select_bandwidth is NOT
# modified; this file provides an opt-in cache wrapper the coordinate loop can consult.
#
# Deliberately NOT a copy of FULL's own h_mode=:cached (composite_gradient_fast.jl): that cache is
# keyed ONLY by coordinate index `k`, populated once, and reused for the ENTIRE outer KNITRO run
# regardless of how far the economic state has since moved -- correct for FULL only because FULL's
# own outer steps are typically small relative to its bandwidth-selection window (an assumption
# never tested for REDUCED's own, differently-scaled retained-relative coordinate geometry). Task
# §7 explicitly warns against inheriting that rule untested. This cache instead keys on the FULL
# semantic context (manifest/family/formulation/coordinate-mode/layout/nu-generation/coordinate
# index) AND validates spatial locality on every lookup via an explicit `validity_radius` -- a
# cached bandwidth is only reused if the CURRENT `w_profiled` is within `validity_radius` (L-inf)
# of the point it was accepted at; otherwise it is treated as a miss and recomputed.

isdefined(Main, :ProfiledLFixCache) ||
    error("profiled_reduced_bandwidth_cache_2026-08-04.jl requires profiled_lfix_incremental_2026-08-01.jl to be included first.")
isdefined(Main, :stable_layout_digest) ||
    error("profiled_reduced_bandwidth_cache_2026-08-04.jl requires profiled_stable_layout_digest_2026-08-01.jl to be included first.")

"""
    ReducedBandwidthCacheKey

Identifies WHICH bandwidth-search result a cache entry represents. Two entries with different
keys are never confused for each other regardless of validity-radius proximity -- the radius check
only applies WITHIN a matching key.
"""
struct ReducedBandwidthCacheKey
    manifest_hash::UInt64        # hash of the ScientificManifest content (W/sigma/draw_seed/... )
    family::Symbol
    formulation::Symbol          # :profiled_destination_scales always, for this cache -- kept explicit per task's own key list
    coordinate_mode::Symbol      # :profiled_pivot_anchor_relative | :profiled_powered_relative_A
    layout_digest::String        # stable_layout_digest(fctx) -- family/layout identity
    nu_generation::UInt64        # nu_generation_id(eta_nu) for ZC families, 0 for families with no free nu
    coordinate_idx::Int
end

mutable struct ReducedBandwidthCacheEntry
    h::Float64
    mass::Float64
    w_at_accept::Vector{Float64}   # full w_profiled at acceptance, for the validity-radius check
end

"""
    ReducedBandwidthCache(validity_radius=0.02)

`validity_radius`: L-infinity distance (in outer-coordinate units) a NEW `w_profiled` may be from
a cache entry's own `w_at_accept` before that entry is treated as stale. Default `0.02` is a
conservative starting point (an order of magnitude tighter than `profiled_select_bandwidth`'s own
default `h_ceil=0.1`) -- NOT claimed optimal, deliberately left tunable per task §7's own "do not
copy FULL's rule untested" instruction; the gate tests below measure whether it is safe and
whether it provides material gain at this value, rather than asserting it does.
"""
mutable struct ReducedBandwidthCache
    entries::Dict{ReducedBandwidthCacheKey,ReducedBandwidthCacheEntry}
    validity_radius::Float64
    hits::Int
    misses::Int
    stale_evictions::Int
    lk::ReentrantLock
end
ReducedBandwidthCache(validity_radius::Float64 = 0.02) =
    ReducedBandwidthCache(Dict{ReducedBandwidthCacheKey,ReducedBandwidthCacheEntry}(), validity_radius, 0, 0, 0, ReentrantLock())

"content hash of a ScientificManifest -- in-process only (not claimed cross-process-stable; this cache is never persisted, unlike CMCheckpointV11/stable_layout_digest)."
reduced_bandwidth_manifest_hash(sci) = hash((sci.W, sci.sigma, sci.draw_design, sci.draw_seed, sci.destination_sample,
    sci.exclude_diagonal_gravity, sci.gravity_exclude_cells, sci.K_mean, sci.K_pair, sci.L, sci.dataset_checksum))

"""
    reduced_bandwidth_cache_key(cache_ctx, coordinate_idx) -> ReducedBandwidthCacheKey

`cache_ctx` is a NamedTuple `(manifest_hash, family, coordinate_mode, layout_digest, nu_generation)`
-- built once per outer run (task §7's own key components that do NOT change per coordinate),
combined here with the one per-coordinate field.
"""
reduced_bandwidth_cache_key(cache_ctx, coordinate_idx::Int) = ReducedBandwidthCacheKey(
    cache_ctx.manifest_hash, cache_ctx.family, :profiled_destination_scales, cache_ctx.coordinate_mode,
    cache_ctx.layout_digest, cache_ctx.nu_generation, coordinate_idx)

"""
    reduced_select_bandwidth_cached(cache, cache_ctx, plfix, ctx, spec, pe, w0, coord_idx; kwargs...) -> (h, mass, hit::Bool)

Cache-aware wrapper around `profiled_select_bandwidth`. On a hit (matching key AND `w0` within
`cache.validity_radius` of the entry's own `w_at_accept`), returns the cached `(h, mass)` with NO
bandwidth search. On a miss (no entry, OR entry present but stale by distance -- counted separately
via `stale_evictions`), runs the real `profiled_select_bandwidth` and stores/overwrites the entry.
Thread-safe (`cache.lk` guards the dict read/write only -- the expensive search itself runs outside
the lock, mirroring FULL's own `bandwidth_cache_lock` discipline in composite_gradient_fast.jl).
"""
function reduced_select_bandwidth_cached(cache::ReducedBandwidthCache, cache_ctx, plfix::ProfiledLFixCache, ctx,
        spec::AnchorSpec, pe::PivotGravityElimOnRetained, w0::AbstractVector{Float64}, coord_idx::Int; kwargs...)
    key = reduced_bandwidth_cache_key(cache_ctx, coord_idx)
    w0c = collect(Float64, w0)
    local entry, is_hit, is_stale
    lock(cache.lk) do
        entry = get(cache.entries, key, nothing)
        if entry === nothing
            is_hit = false; is_stale = false
        else
            d = maximum(abs.(w0c .- entry.w_at_accept))
            is_stale = d > cache.validity_radius
            is_hit = !is_stale
        end
    end
    if is_hit
        lock(cache.lk) do; cache.hits += 1; end
        return entry.h, entry.mass, true
    end
    lock(cache.lk) do
        is_stale && (cache.stale_evictions += 1)
        cache.misses += 1
    end
    h, m, _selmeta = profiled_select_bandwidth(plfix, ctx, spec, pe, w0c, coord_idx; kwargs...)
    lock(cache.lk) do
        cache.entries[key] = ReducedBandwidthCacheEntry(h, m, w0c)
    end
    return h, m, false
end

"Reset hit/miss/eviction counters (does NOT clear entries) -- call between independent measurement windows."
function reduced_bandwidth_cache_reset_counters!(cache::ReducedBandwidthCache)
    cache.hits = 0; cache.misses = 0; cache.stale_evictions = 0
    return nothing
end

"""
    profiled_composite_gradient_from_cache_bwcache(cache, ctx, spec, pe, w_profiled, ev, bwcache, cache_ctx; threaded=false) -> (g, meta)

Cache-aware variant of `profiled_composite_gradient_from_cache` (unchanged, not modified) -- same
gp-analytic + per-coordinate central-FD structure, but each coordinate's bandwidth comes from
`reduced_select_bandwidth_cached` instead of an unconditional `profiled_select_bandwidth` call.
Threading-compatible: `bwcache`'s own internal lock (not the coordinate-loop's `g`/`h_used`
arrays) is the only shared mutable state touched inside `do_coord!`, exactly mirroring how FULL's
own `h_mode=:cached` path composes with `threaded=true` (composite_gradient_fast.jl's own
`bandwidth_cache_lock`).
"""
function profiled_composite_gradient_from_cache_bwcache(plfix::ProfiledLFixCache, ctx, spec::AnchorSpec,
        pe::PivotGravityElimOnRetained, w_profiled::AbstractVector{Float64}, ev, bwcache::ReducedBandwidthCache,
        cache_ctx; threaded::Bool = false)
    n_total = outer_dim_profiled(pe)
    g = zeros(n_total)
    g[1] = profiled_gp_component_analytic(plfix, w_profiled, ev, ctx)
    h_used = zeros(n_total); switch_mass = zeros(n_total); cache_hit = falses(n_total)

    function do_coord!(coord_idx::Int)
        h, m, hit = reduced_select_bandwidth_cached(bwcache, cache_ctx, plfix, ctx, spec, pe, w_profiled, coord_idx)
        h_used[coord_idx] = h; switch_mass[coord_idx] = m; cache_hit[coord_idx] = hit
        Lp = profiled_lfix_incremental_at(plfix, ctx, spec, pe, w_profiled, coord_idx, w_profiled[coord_idx] + h)
        Lm = profiled_lfix_incremental_at(plfix, ctx, spec, pe, w_profiled, coord_idx, w_profiled[coord_idx] - h)
        g[coord_idx] = (Lp - Lm) / (2h)
        return nothing
    end

    if threaded
        Main.CS.guard_enter_coord_pool!()
        try
            Threads.@threads for coord_idx in 2:n_total
                do_coord!(coord_idx)
            end
        finally
            Main.CS.guard_exit_coord_pool!()
        end
    else
        @inbounds for coord_idx in 2:n_total
            do_coord!(coord_idx)
        end
    end
    return g, (cache = plfix, w0 = collect(Float64, w_profiled), h_used = h_used, switch_mass = switch_mass,
        threaded = threaded, cache_hit = cache_hit)
end
