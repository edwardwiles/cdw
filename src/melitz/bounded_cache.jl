# Continuation session (2026-07-23, "make the optimized architecture scalable in memory and
# D") Section 4: shared bounded-LRU machinery for the Melitz exact-point/heavy-state caches.
#
# Why a hand-rolled LRU rather than a package dependency: both caches this backs
# (`MelitzExactPointCache`, `finite_delta_outer.jl`; `MelitzDeltaEvalCache`, `delta_star.jl`)
# are keyed by `Vector{Float64}` (an outer `theta_free` point) and sized in the tens-to-low-
# hundreds of entries at most (bounded BY DESIGN, per this session's own governing prompt --
# "do not retain every full moment matrix indefinitely") -- an O(n) touch/evict per access at
# that scale is negligible next to a single KNITRO inner solve (milliseconds) or moment build
# it is there to avoid repeating, so there is no reason to add an external LRU dependency
# for asymptotic behavior this code path will never approach.

"""
    MelitzLRUOrder

Access-order tracker for a bounded cache keyed by `Vector{Float64}`: `keys` holds every
LIVE key, least-recently-used FIRST, most-recently-used LAST. `Dict` itself has no access-
order concept in Julia, hence this companion structure.
"""
mutable struct MelitzLRUOrder
    keys::Vector{Vector{Float64}}
end
MelitzLRUOrder() = MelitzLRUOrder(Vector{Float64}[])

"""
    melitz_lru_touch!(order, key)

Marks `key` most-recently-used: moves it to the end of `order.keys` (inserting it there if
not already present -- the caller is responsible for having just read or written `key` in
the backing `Dict`).
"""
function melitz_lru_touch!(order::MelitzLRUOrder, key::Vector{Float64})
    idx = findfirst(==(key), order.keys)
    idx === nothing || deleteat!(order.keys, idx)
    push!(order.keys, key)
    return nothing
end

"""
    melitz_lru_forget!(order, key)

Removes `key` from the order tracker with no eviction bookkeeping (used by the stale-
context guard below, which deletes an entry outright rather than counting it as an LRU
eviction).
"""
function melitz_lru_forget!(order::MelitzLRUOrder, key::Vector{Float64})
    idx = findfirst(==(key), order.keys)
    idx === nothing || deleteat!(order.keys, idx)
    return nothing
end

"""
    melitz_lru_evict_until!(order, store, max_size) -> n_evicted

Pops the least-recently-used keys (the FRONT of `order.keys`) and deletes them from `store`
until `length(order.keys) <= max_size`. Returns the number of entries evicted (0 if already
within capacity). `max_size <= 0` is treated as "unbounded" (a no-op) purely so a caller can
pass a configured `max_size` of `0`/negative to mean "disable bounding" explicitly rather
than needing a separate flag -- every PRODUCTION default in this session is a small positive
integer, never this escape hatch.
"""
function melitz_lru_evict_until!(order::MelitzLRUOrder, store::Dict, max_size::Int)
    max_size <= 0 && return 0
    n_evicted = 0
    while length(order.keys) > max_size
        oldest = popfirst!(order.keys)
        delete!(store, oldest)
        n_evicted += 1
    end
    return n_evicted
end

"""
    melitz_context_fingerprint(ctx, U=nothing) -> UInt

Closure-audit session (2026-07-24), Phase D: a stable CONTENT-based fingerprint for a
Melitz outer context `ctx`, replacing `objectid(ctx)` as the SOLE cache-staleness key
(`MelitzExactPointCache`, `finite_delta_outer.jl`). `objectid` changes across every fresh
reconstruction of an otherwise-identical `ctx` (a new NamedTuple is a new object even with
byte-identical fields), so it can never recognize a "recreated context" as the SAME context
for caching purposes -- this function can, because it hashes the fields that actually
determine the economic problem: `D`/`sigma`/`theta_star`/`target_country` (economic
version/dimension), `tau`/`w`/`X_data` (data), `outer_parameterization` (parameterization),
`inner_loop_opt`/`outer_loop_opt` (solver options). `U` (the Monte Carlo draws) is threaded
through as a SEPARATE optional argument rather than read off `ctx`, because `U` lives on the
bundle (`obj.U`), not on `ctx`, in this codebase's data model -- a caller with only a bare
`ctx` still gets a genuine content fingerprint over everything else; a caller that also
supplies `U` (as the one production call site in `finite_delta_outer.jl` does) gets full
coverage including the draws.

Falls back to `objectid(ctx)` when `ctx` does not expose the expected field set (checked via
`hasproperty` on `:D`/`:X_data`/`:tau`/`:sigma`/`:outer_parameterization`/`:inner_loop_opt`/
`:outer_loop_opt`) -- covers this repo's own synthetic cache-mechanics test doubles (plain
`Ref{Symbol}` placeholders in `test/melitz/runtests.jl`'s LRU/eviction tests), which were
never meant to model a real Melitz context and have no stable content to hash.
"""
function melitz_context_fingerprint(ctx, U::Union{Nothing,AbstractMatrix}=nothing)
    has_shape = hasproperty(ctx, :D) && hasproperty(ctx, :X_data) && hasproperty(ctx, :tau) &&
                hasproperty(ctx, :sigma) && hasproperty(ctx, :outer_parameterization) &&
                hasproperty(ctx, :inner_loop_opt) && hasproperty(ctx, :outer_loop_opt)
    has_shape || return UInt(objectid(ctx))
    h = hash(:melitz_ctx_fingerprint_v1)
    h = hash(ctx.D, h)
    h = hash(ctx.sigma, h)
    h = hash(ctx.theta_star, h)
    h = hash(ctx.target_country, h)
    h = hash(ctx.outer_parameterization, h)
    h = hash(ctx.inner_loop_opt, h)
    h = hash(ctx.outer_loop_opt, h)
    h = hash(ctx.tau, h)
    h = hash(ctx.w, h)
    h = hash(ctx.X_data, h)
    U !== nothing && (h = hash(U, h))
    return UInt(h)
end
