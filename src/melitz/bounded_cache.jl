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
