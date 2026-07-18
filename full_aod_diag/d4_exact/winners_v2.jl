# ============================================================================
# Phase 1B (continuation 3): additive, allocation-free replacement for
# winners.jl::compute_winners' per-column `sort(col)` call. `sort(col)`
# computes a full sorted order just to extract the min and runner-up value --
# this file replaces that with a direct two-pass min/second-min scan that
# computes EXACTLY the same `wmin`/`wo`/`gap` triple, with the SAME
# tie-breaking convention as `findmin` (first occurrence of the minimum wins,
# via strict `<` comparisons only -- matches Julia's `isless`-based total
# order, so this scan reproduces `findmin`'s own index choice bit-for-bit,
# not just "a" valid minimum). winners.jl is NOT modified; this is a mirror,
# equivalence-tested in test_winners_v2.jl before being trusted.
# ============================================================================

"""
    min_and_secondmin(col) -> (wmin, wo, gap)

Single pass, allocation-free (no temporary array, unlike `sort(col)`).
`wmin` = minimum value, `wo` = its 1-based index (first occurrence, matching
`findmin`), `gap` = second-smallest minus smallest (`Inf` if `length(col)<2`
or if a second finite/comparable value never displaces `Inf`). NaN handling
matches `findmin`/`isless`-based ordering: a NaN entry can only become `wmin`
if EVERY entry is NaN (since `isless(x, NaN)` is `true` for any non-NaN `x`,
NaN never displaces a real minimum, exactly as `sort`'s default total order
treats NaN as maximal).
"""
function min_and_secondmin(col)
    n = length(col)
    @assert n >= 1 "min_and_secondmin: empty column"
    m1 = col[1]; idx1 = 1
    m2 = oftype(m1, Inf)
    # Invariant maintained at the top of every iteration: m1 <= m2 (in isless order).
    # So when a new global min is found (v < m1), the OLD m1 is guaranteed to be the
    # new runner-up -- no comparison against the old m2 is needed.
    @inbounds for i in 2:n
        v = col[i]
        if isless(v, m1)
            m2 = m1
            m1 = v; idx1 = i
        elseif isless(v, m2)
            m2 = v
        end
    end
    gap = n >= 2 ? (m2 - m1) : oftype(m1, Inf)
    return m1, idx1, gap
end

"""
    compute_winners_fast(θ_full, ctx) -> (winner::Matrix{Int}, price::Array{Float64,3}, gap::Matrix{Float64})

Same signature/semantics as `winners.jl::compute_winners`, allocation-free
inner loop (`min_and_secondmin` instead of `sort`). Reuses `factual_prices`
unchanged (that part was never the allocation hotspot per
`docs/fullA_performance_profile.md` -- only the per-column `sort()` was).
"""
function compute_winners_fast(θ_full::AbstractVector, ctx)
    price, Aod, AodPow = factual_prices(θ_full, ctx)
    D = ctx.D; W = size(price, 1)
    winner = Matrix{Int}(undef, W, D)
    gap = Matrix{Float64}(undef, W, D)
    @inbounds for d in 1:D, ω in 1:W
        col = @view price[ω, :, d]
        wmin, wo, g = min_and_secondmin(col)
        gap[ω, d] = g
        winner[ω, d] = wo
    end
    return winner, price, gap
end
