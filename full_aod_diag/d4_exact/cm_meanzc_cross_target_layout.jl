# ============================================================================
# CM+ZC-CROSS target layout (2026-08-09): the SAME shared-across-origins nu_k outer parameters as
# `SharedByPowerLayout` (`n_eta = K_mean`, `target_index(layout,o,k) = k`, `mean_targets` all
# identical formulas -- the mean block does not change), but a DIFFERENT `pair_targets` formula
# spanning the full K_pair^2 ordered-level-pair grid instead of the diagonal-only k1=k2 case.
#
# Exactly the same technique `cm_originzc_cross_target_layout.jl` uses for OZC-CROSS, one level
# further in: a new concrete `MeanZCTargetLayout` subtype (not a flag bolted onto
# `SharedByPowerLayout`) so `pair_targets` dispatches differently via ordinary Julia multiple
# dispatch, leaving every existing layout's code byte-untouched.
#
# WHY THE CROSS TARGET IS SIMPLER HERE THAN FOR OZC-CROSS: under common marginals every origin
# shares ONE marginal and therefore ONE nu_k per level (cm_meanzc_moments.jl's file header), so the
# cross-pair target `E[z_o^k1 * z_p^k2] = nu_{o,k1} * nu_{p,k2}` collapses to the origin-free
# `nu_k1 * nu_k2` -- the SAME scalar for every unordered pair (o,p) at a given `klin`. Consequences
# that matter downstream and are relied on throughout the CM+ZC-CROSS code:
#   * `n_eta` stays `K_mean` (NOT `K_mean*D`) -- no new outer parameters, exactly as for OZC-CROSS.
#   * `target_index(layout,o,k) == k`, hence `ActiveMeanLayout.dense_omit_idx == aml.kstar`.
#   * nu_{k*} (the Variant D focal level) appears in cross-pair rows at BOTH `(k*,k2)` for every k2
#     AND `(k1,k*)` for every k1 -- strictly more terms than the diagonal family's single
#     `(k*,k*)` block. See `d_delta_dual_d_nu_cross_vec` (cm_meanzc_cross_moments.jl) for how the
#     accumulate-into-both-slots pattern handles that, and the D4 FD gate for its verification.
# ============================================================================

isdefined(Main, :SharedByPowerLayout) || include(joinpath(@__DIR__, "cm_originzc_target_layout.jl"))
isdefined(Main, :cross_pair_level_index) || include(joinpath(@__DIR__, "cm_originzc_cross_moments.jl"))

"""
    SharedByPowerCrossLayout(K_mean, K_pair)

Identical outer-parameter space to `SharedByPowerLayout(K_mean, K_pair)` (SAME `n_eta = K_mean`,
SAME `target_index`/`mean_targets` formulas) -- `K_pair` here means "the cross-pair grid spans
levels 1:K_pair" (`K_pair^2` restrictions per origin pair), not "K_pair diagonal levels" as in the
base layout. Its `pair_targets` method (below) returns the `K_pair^2`-indexed cross-product target
vector, keyed by `cross_pair_level_index(K_pair)`'s flat `klin` ordering (matching
`build_raw_cross_pair_matrix_levels`'s block order exactly, cm_originzc_cross_moments.jl -- that
builder is reused VERBATIM by this family; the raw cross-power feature columns are origin-indexed
and therefore completely independent of whether nu is shared or origin-specific).

`D`-agnostic by construction, exactly like `SharedByPowerLayout` (one target per level-pair
regardless of D) -- hence `layout_D` returns 0 and the constructor takes no `D`.
"""
struct SharedByPowerCrossLayout <: MeanZCTargetLayout
    K_mean::Int
    K_pair::Int
    function SharedByPowerCrossLayout(K_mean::Int, K_pair::Int)
        K_mean >= 1 || error("SharedByPowerCrossLayout: K_mean must be >= 1, got $K_mean")
        0 <= K_pair <= K_mean || error("SharedByPowerCrossLayout: K_pair must satisfy 0 <= K_pair <= K_mean, got K_pair=$K_pair, K_mean=$K_mean")
        return new(K_mean, K_pair)
    end
end

"n_eta/target_index: byte-identical formulas to SharedByPowerLayout (cm_originzc_target_layout.jl) -- the mean/nu space is completely unaffected by the cross-pair extension."
n_eta(layout::SharedByPowerCrossLayout) = layout.K_mean
target_index(layout::SharedByPowerCrossLayout, o::Int, k::Int) = k

"""
    pair_targets(layout::SharedByPowerCrossLayout, νfull, klin, D) -> Vector{Float64}   (length D*(D-1)/2)

Cross-pair target: `nu_{k1} * nu_{k2}` for every unordered pair `o<p` (`packed_pair_index`
ordering, UNCHANGED), where `(k1,k2) = cross_pair_level_index(layout.K_pair)[klin]`. Under the
shared-nu layout the value is the SAME for every pair (no origin dependence), so this is a constant
vector -- but it is returned as a full-length vector, not a scalar, because `refresh_zc_targets!`
(zc_restriction_operator.jl, UNCHANGED) writes it into a `(npair, K_pair^2)` target buffer column.
Reduces to the base family's diagonal-only `pair_targets(::MeanZCTargetLayout, ...)` -> `nu_k^2`
exactly when `k1==k2`. Dispatches ahead of that generic abstract-type method by ordinary Julia
method specificity (concrete type beats abstract type).
"""
function pair_targets(layout::SharedByPowerCrossLayout, νfull::AbstractVector{Float64}, klin::Int, D::Int)
    levels = cross_pair_level_index(layout.K_pair)
    k1, k2 = levels[klin]
    npair = div(D * (D - 1), 2)
    return fill(νfull[k1] * νfull[k2], npair)
end

layout_name(::SharedByPowerCrossLayout) = :shared_by_power_cross
layout_D(::SharedByPowerCrossLayout) = 0
