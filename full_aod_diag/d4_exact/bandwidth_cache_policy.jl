# ============================================================================
# Continuation 9, Phase 5: staleness-aware bandwidth cache policy.
#
# `composite_gradient_at_fast`'s existing `h_mode=:cached` (composite_gradient_fast.jl,
# Continuation 5) already provides the MECHANISM -- a caller-owned
# `Dict{Int,Float64}` reused across calls, computed once on a miss -- but
# DELIBERATELY has no invalidation policy of its own ("this file does not
# impose a revalidation POLICY, only the cache mechanism" -- that file's own
# header). This file supplies the policy the standing brief requires ("never
# silently use stale bandwidths after large parameter moves... SOME staleness
# detection, not unconditional reuse forever"), as a thin wrapper AROUND the
# existing Dict, not a modification of composite_gradient_fast.jl.
#
# Two triggers, matching the task's methods 2 ("cached... reused across nearby
# outer iterations") and 4 ("fixed within one quasi-Newton run, restart when
# the schedule changes") -- same mechanism, different threshold tuning, so
# implemented as ONE policy with two configurable knobs rather than two
# separate structs:
#   - `max_iters_since_anchor`: iteration-count budget (method 4's "one
#     quasi-Newton run" is, operationally, "until N outer iterates have
#     passed since the bandwidths were last (re)computed").
#   - `max_move`: Euclidean distance in the reduced w-coordinate the anchor
#     point may drift before the WHOLE cache is invalidated (method 2's
#     "nearby outer iterations" bound, expressed directly in parameter space
#     rather than iteration count, since a large single step and many small
#     steps are both real ways a cache can go stale).
# On EITHER trigger, the ENTIRE Dict is cleared and every coordinate is
# recomputed at the new anchor on next use (not a per-coordinate partial
# invalidation) -- deliberately simple and conservative: partial invalidation
# would need a per-coordinate staleness metric this investigation has not
# derived, and the task's own explicit hazard ("switching cheap/expensive
# gradient source... unsafe with quasi-Newton Hessians", reused here by
# analogy for bandwidth CHOICE, not gradient source) argues for an
# all-or-nothing anchor rather than a coordinate-by-coordinate patchwork that
# could leave the effective FD stencil inconsistent across the gradient
# vector within one call.
# ============================================================================
using LinearAlgebra: norm

mutable struct BandwidthCachePolicy
    cache::Dict{Int,Float64}
    anchor_w::Union{Nothing,Vector{Float64}}
    iters_since_anchor::Int
    max_iters_since_anchor::Int
    max_move::Float64
    n_invalidations::Int
    n_hits::Int
    n_misses::Int
    log::Vector{NamedTuple}
end

"""
    BandwidthCachePolicy(; max_iters_since_anchor=5, max_move=0.05) -> BandwidthCachePolicy

`max_move` is measured in the reduced w-coordinate (A-block entries only,
`w[2:end]`, the SAME coordinates `select_bandwidth`/`select_bandwidth_quantile`
operate in) -- 0.05 is a starting default on the same order as
`select_bandwidth`'s own `h_ceil=0.1` (a move bigger than the bandwidth ceiling
itself is a reasonable "definitely re-derive" trigger; tuned/validated
empirically in docs/fullA_D20_bandwidth_optimization_report.md, not asserted
correct a priori).
"""
BandwidthCachePolicy(; max_iters_since_anchor::Int = 5, max_move::Float64 = 0.05) =
    BandwidthCachePolicy(Dict{Int,Float64}(), nothing, 0, max_iters_since_anchor, max_move, 0, 0, 0, NamedTuple[])

"""
    maybe_invalidate!(policy, w0) -> (invalidated::Bool, reason::Symbol)

Call ONCE per outer-loop gradient evaluation, before using `policy.cache` as
the `bandwidth_cache` argument to `composite_gradient_at_fast(...; h_mode=:cached)`.
Checks both triggers against the CURRENT anchor; on a trigger, clears the
Dict, resets the anchor to `w0`, and returns `true`. On the very first call
(`anchor_w === nothing`) always "invalidates" (there is nothing to keep) and
sets the anchor -- this is not a real invalidation event but is logged as one
for a uniform record.
"""
function maybe_invalidate!(policy::BandwidthCachePolicy, w0::AbstractVector)
    a_block = @view w0[2:end]
    if policy.anchor_w === nothing
        policy.anchor_w = collect(a_block)
        policy.iters_since_anchor = 0
        push!(policy.log, (event = :init, move = 0.0, iters = 0))
        return true, :init
    end
    move = norm(a_block .- policy.anchor_w)
    reason = :none
    do_inval = false
    if move > policy.max_move
        do_inval = true; reason = :move_threshold
    elseif policy.iters_since_anchor >= policy.max_iters_since_anchor
        do_inval = true; reason = :iter_budget
    end
    if do_inval
        empty!(policy.cache)
        policy.anchor_w = collect(a_block)
        policy.iters_since_anchor = 0
        policy.n_invalidations += 1
        push!(policy.log, (event = reason, move = move, iters_at_invalidation = policy.iters_since_anchor))
        return true, reason
    else
        policy.iters_since_anchor += 1
        push!(policy.log, (event = :reuse, move = move, iters = policy.iters_since_anchor))
        return false, :none
    end
end

"record hit/miss counts AFTER a composite_gradient_at_fast(...; h_mode=:cached) call, by diffing meta.cache_hits against the policy's own running tallies -- call once per gradient evaluation, after maybe_invalidate!."
function record_hits!(policy::BandwidthCachePolicy, cache_hits::AbstractVector{Bool})
    policy.n_hits += count(cache_hits)
    policy.n_misses += count(!, cache_hits)
    return nothing
end
