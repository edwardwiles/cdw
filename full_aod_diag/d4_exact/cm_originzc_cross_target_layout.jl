# ============================================================================
# OZC-CROSS target layout: the SAME nu_{o,k} outer parameters as OriginByPowerLayout (`n_eta`,
# `target_index`, `mean_targets` all identical formulas -- the mean block does not change), but a
# DIFFERENT `pair_targets` formula spanning the full K_pair^2 ordered-level-pair grid instead of
# the diagonal-only k1=k2 case. Implemented as a new concrete `MeanZCTargetLayout` subtype (not a
# flag bolted onto `OriginByPowerLayout`) so `pair_targets` dispatches differently via ordinary
# Julia multiple dispatch -- the SAME technique `cm_originzc_target_layout.jl` itself already uses
# to add `OriginByPowerLayout` alongside `SharedByPowerLayout` without touching either's existing
# code, applied one level further out.
# ============================================================================

isdefined(Main, :OriginByPowerLayout) || include(joinpath(@__DIR__, "cm_originzc_target_layout.jl"))
isdefined(Main, :cross_pair_level_index) || include(joinpath(@__DIR__, "cm_originzc_cross_moments.jl"))

"""
    OriginByPowerCrossLayout(D, K_mean, K_pair)

Identical outer-parameter space to `OriginByPowerLayout(D, K_mean, K_pair)` (SAME `n_eta =
K_mean*D`, SAME `target_index`/`mean_targets` formulas) -- `K_pair` here means "the cross-pair grid
spans levels 1:K_pair" (`K_pair^2` restrictions per origin pair), not "K_pair diagonal levels" as
in the base layout. Its `pair_targets` method (below) returns the `K_pair^2`-long cross-product
target vector, keyed by `cross_pair_level_index(K_pair)`'s flat `klin` ordering (matches
`build_raw_cross_pair_matrix_levels`'s block order exactly, cm_originzc_cross_moments.jl).
"""
struct OriginByPowerCrossLayout <: MeanZCTargetLayout
    D::Int
    K_mean::Int
    K_pair::Int
    function OriginByPowerCrossLayout(D::Int, K_mean::Int, K_pair::Int)
        D >= 2 || error("OriginByPowerCrossLayout: D must be >= 2, got $D")
        K_mean >= 1 || error("OriginByPowerCrossLayout: K_mean must be >= 1, got $K_mean")
        0 <= K_pair <= K_mean || error("OriginByPowerCrossLayout: K_pair must satisfy 0 <= K_pair <= K_mean, got K_pair=$K_pair, K_mean=$K_mean")
        return new(D, K_mean, K_pair)
    end
end

"n_eta/target_index: byte-identical formulas to OriginByPowerLayout (cm_originzc_target_layout.jl) -- the mean/nu space is completely unaffected by the cross-pair extension."
n_eta(layout::OriginByPowerCrossLayout) = layout.K_mean * layout.D
target_index(layout::OriginByPowerCrossLayout, o::Int, k::Int) = (k - 1) * layout.D + o

"""
    pair_targets(layout::OriginByPowerCrossLayout, νfull, klin, D) -> Vector{Float64}   (length D*(D-1)/2)

Cross-pair target: `nu_{o,k1} * nu_{p,k2}` for every unordered pair `o<p` (`packed_pair_index`
ordering, UNCHANGED), where `(k1,k2) = cross_pair_level_index(layout.K_pair)[klin]`. Power `k1`
attaches to the LOWER-indexed origin `o`, `k2` to the HIGHER-indexed origin `p` -- the same
canonical-order convention `build_raw_cross_pair_matrix_levels` uses to build the matching raw
feature columns (cm_originzc_cross_moments.jl). Reduces to the base family's diagonal-only
`pair_targets` (cm_originzc_target_layout.jl) exactly when `k1==k2`. Dispatches ahead of the
generic abstract-type `pair_targets(layout::MeanZCTargetLayout, ...)` method by ordinary Julia
method specificity (concrete type beats abstract type) -- that generic method is never called for
this layout.
"""
function pair_targets(layout::OriginByPowerCrossLayout, νfull::AbstractVector{Float64}, klin::Int, D::Int)
    pairs = packed_pair_index(D)
    levels = cross_pair_level_index(layout.K_pair)
    k1, k2 = levels[klin]
    return [νfull[target_index(layout, o, k1)] * νfull[target_index(layout, p, k2)] for (o, p) in pairs]
end

layout_name(::OriginByPowerCrossLayout) = :origin_by_power_cross
layout_D(layout::OriginByPowerCrossLayout) = layout.D
