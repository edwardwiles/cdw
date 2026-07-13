# ============================================================================
# Generic free/fixed outer-parameter map.
#
# Every outer θ vector in this codebase mixes genuinely free coordinates
# (optimized by KNITRO, differentiated by ForwardDiff) with coordinates that
# are fixed/calibrated/normalized (mu, sigma, normalization slots) but were
# historically still carried inside θ with equal lower/upper KNITRO bounds and
# still differentiated (as zero-partial Dual entries) by ForwardDiff. That
# means every ForwardDiff call paid for `length(theta_full)` partials per Dual
# number even though only `length(free_idx)` of them were ever nonzero.
#
# FreeParamMap makes the split explicit and mechanical:
#   x_free = theta_full[free_idx]                      (pack, KNITRO sees this)
#   theta_full = reconstruct(x_free, fixed_vals, ...)   (unpack, moments! sees this)
# ForwardDiff.gradient is then called on `x -> f(reconstruct_full(x, m))`,
# i.e. with `x_free` (length(free_idx)) as the actual AD input — so every Dual
# number produced downstream carries `length(free_idx)`-length partial tuples,
# not `length(theta_full)`-length ones. This is what "differentiate only the
# free coordinates" means computationally, not just conceptually.
# ============================================================================

"""
    FreeParamMap(l_full, free_idx, fixed_idx, fixed_vals)

- `l_full`     : length of the full economic parameter vector theta.
- `free_idx`   : indices (into the length-`l_full` vector) that are genuinely
                 free / optimized / differentiated, in the ORDER they appear
                 in the packed free vector x_free.
- `fixed_idx`  : the complementary indices, held at `fixed_vals`.
- `fixed_vals` : the constant value for each entry of `fixed_idx` (same order).

`free_idx` and `fixed_idx` must partition `1:l_full` exactly (checked in the
constructor) — every full-vector coordinate is either free or fixed, never
both, never neither.
"""
struct FreeParamMap
    l_full::Int
    free_idx::Vector{Int}
    fixed_idx::Vector{Int}
    fixed_vals::Vector{Float64}

    function FreeParamMap(l_full::Int, free_idx::Vector{Int}, fixed_idx::Vector{Int}, fixed_vals::Vector{Float64})
        length(fixed_idx) == length(fixed_vals) || error("FreeParamMap: fixed_idx and fixed_vals must have equal length")
        all_idx = sort(vcat(free_idx, fixed_idx))
        all_idx == collect(1:l_full) || error(
            "FreeParamMap: free_idx ∪ fixed_idx must partition 1:$l_full exactly " *
            "(got $(length(free_idx)) free + $(length(fixed_idx)) fixed = $(length(all_idx)) indices, " *
            "with $(length(unique(all_idx))) unique) — check for overlaps or gaps.")
        new(l_full, free_idx, fixed_idx, fixed_vals)
    end
end

n_free(m::FreeParamMap) = length(m.free_idx)

"pack_free(theta_full, m) -> x_free. Extracts the free coordinates, in free_idx order."
pack_free(theta_full::AbstractVector, m::FreeParamMap) = theta_full[m.free_idx]

"""
    reconstruct_full(x_free, m::FreeParamMap) -> theta_full

Builds the full economic parameter vector from the free coordinates plus the
map's constant `fixed_vals`. `eltype` follows `x_free` (so under ForwardDiff,
`theta_full` becomes a Dual-typed vector whose Dual partials tuples have
length `length(x_free)` — the fixed entries become `Dual(fixed_vals[i], 0,...,0)`,
i.e. correctly-zero-derivative constants, WITHOUT inflating the partial-tuple
length to `l_full` the way differentiating the full old-style theta vector did).
"""
function reconstruct_full(x_free::AbstractVector{T}, m::FreeParamMap) where {T}
    theta_full = Vector{T}(undef, m.l_full)
    @inbounds for (k, i) in enumerate(m.free_idx)
        theta_full[i] = x_free[k]
    end
    @inbounds for (k, i) in enumerate(m.fixed_idx)
        theta_full[i] = T(m.fixed_vals[k])
    end
    return theta_full
end

"In-place variant: writes into a preallocated `theta_full` buffer (must already have the right eltype/length)."
function reconstruct_full!(theta_full::AbstractVector{T}, x_free::AbstractVector, m::FreeParamMap) where {T}
    length(theta_full) == m.l_full || error("reconstruct_full!: buffer length $(length(theta_full)) != l_full $(m.l_full)")
    @inbounds for (k, i) in enumerate(m.free_idx)
        theta_full[i] = x_free[k]
    end
    @inbounds for (k, i) in enumerate(m.fixed_idx)
        theta_full[i] = T(m.fixed_vals[k])
    end
    return theta_full
end

"""
    pack_bounds_free(theta_lower_full, theta_upper_full, m::FreeParamMap) -> (lo_free, hi_free)

Extracts KNITRO bounds for the free-only vector. Errors if any FIXED index's
bounds are not both exactly equal to `m.fixed_vals` (catches the old "leave it
in the vector with a degenerate bound" pattern silently drifting out of sync
with the map).
"""
function pack_bounds_free(theta_lower_full::AbstractVector, theta_upper_full::AbstractVector, m::FreeParamMap)
    for (k, i) in enumerate(m.fixed_idx)
        (theta_lower_full[i] == theta_upper_full[i] == m.fixed_vals[k]) || error(
            "pack_bounds_free: full-vector bounds at fixed index $i ($(theta_lower_full[i]), $(theta_upper_full[i])) " *
            "do not match fixed_vals[$k]=$(m.fixed_vals[k]) — map and bounds have drifted out of sync.")
    end
    return pack_free(theta_lower_full, m), pack_free(theta_upper_full, m)
end

"""
    round_trip_check(theta_full, m::FreeParamMap) -> Bool

`full -> pack_free -> reconstruct_full -> full` must be EXACT (not approximate)
on every coordinate, since fixed coordinates are literal constants and free
coordinates are copied, not recomputed. Section 5's required round-trip test.
"""
function round_trip_check(theta_full::AbstractVector, m::FreeParamMap)
    x = pack_free(theta_full, m)
    theta_back = reconstruct_full(x, m)
    return theta_back == theta_full
end
