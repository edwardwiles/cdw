# ============================================================================
# OZC-CROSS: full K_pair^2 cross-power extension of origin-specific ZC's pairwise-zero-covariance
# restriction block. See cm_originzc_moments.jl / zc_restriction_operator.jl for the base family
# this extends -- EVERY piece of shared inner-loop machinery there (ZCRestrictionOperator,
# restriction_forward!/restriction_transpose! (the FG callback), zc_restriction_gram! (H_ZZ),
# winner_pair_cross_hessian_zc_block! (H_EZ)) is column-count-agnostic (confirmed by direct
# reading, 2026-08-09: each operates on "however many restriction columns exist", never on K_pair's
# meaning) and is reused HERE completely UNCHANGED; this file supplies ONLY the two things that
# differ from the base family: (1) the raw cross-power feature matrices (Zpairraw_all, built below),
# (2) the cross target formula (via OriginByPowerCrossLayout's `pair_targets` method,
# cm_originzc_cross_target_layout.jl).
#
# Restriction (confirmed with user 2026-08-09): for every unordered origin pair (o,p), o<p
# (packed_pair_index, cm_meanzc_moments.jl, UNCHANGED canonical order), and every ORDERED level
# pair (k1,k2) in {1,...,K_pair}x{1,...,K_pair} (K_pair^2 total per origin pair, including the
# pre-existing diagonal k1=k2):
#     E_F[z_o(w)^k1 * z_p(w)^k2] = nu_{o,k1} * nu_{p,k2}
# No new outer parameters -- nu_{o,k} is the SAME quantity origin-ZC's mean block already defines
# (n_eta(layout) is UNCHANGED, K_mean*D); only the pair-restriction COUNT (K_pair^2*npair instead
# of K_pair*npair) and the cross-term math change. No duplication: (o,k1;p,k2) and (o,k2;p,k1) are
# genuinely distinct restrictions for k1!=k2 (o,p are specific, non-interchangeable origins with
# different nu's) -- they are NOT the same restriction listed twice, precisely because pairs stay
# canonically ordered o<p and are never separately revisited as (p,o).
# ============================================================================

isdefined(Main, :OriginByPowerLayout) || include(joinpath(@__DIR__, "cm_originzc_target_layout.jl"))

"""
    cross_pair_level_index(K_pair::Int) -> Vector{Tuple{Int,Int}}

Canonical linear ordering of the `K_pair^2` ordered level pairs `(k1,k2)`, `k1,k2 = 1:K_pair`,
row-major (`k1` outer, `k2` inner): `(1,1),(1,2),...,(1,K_pair),(2,1),...,(K_pair,K_pair)`. The
SAME ordering is used by `build_raw_cross_pair_matrix_levels` below (to fix `Zpairraw_all`'s block
order) and by `pair_targets` (`OriginByPowerCrossLayout` method, cm_originzc_cross_target_layout.jl)
-- both index by the same flat integer `klin = (k1-1)*K_pair + k2`, matching
`ZCRestrictionOperator`'s own `(k-1)*npair+1:k*npair` flat-block convention (zc_restriction_operator.jl,
UNCHANGED) with `k=klin`.
"""
function cross_pair_level_index(K_pair::Int)
    K_pair >= 1 || error("cross_pair_level_index: K_pair must be >= 1, got $K_pair")
    pairs = Vector{Tuple{Int,Int}}(undef, K_pair^2)
    idx = 0
    for k1 in 1:K_pair, k2 in 1:K_pair
        idx += 1
        pairs[idx] = (k1, k2)
    end
    return pairs
end

"""
    n_originzc_cross_moments(D::Int, K_mean::Int, K_pair::Int) -> Int

Total new inner moments for OZC-CROSS: `K_mean*D` (mean block, UNCHANGED formula/meaning) plus
`K_pair^2 * D(D-1)/2` (cross-pair block -- `K_pair^2` ordered level-pair combinations per unordered
origin pair, see file header). Contrast `n_originzc_moments`/`n_meanzc_moments`'s
`K_pair*D(D-1)/2` (diagonal-only, base family).
"""
function n_originzc_cross_moments(D::Int, K_mean::Int, K_pair::Int)
    K_mean >= 1 || error("n_originzc_cross_moments: K_mean must be >= 1, got $K_mean")
    0 <= K_pair <= K_mean || error("n_originzc_cross_moments: K_pair must satisfy 0 <= K_pair <= K_mean, got K_pair=$K_pair, K_mean=$K_mean")
    return K_mean * D + K_pair^2 * div(D * (D - 1), 2)
end

"""
    build_raw_cross_pair_matrix_levels(Zraw_all::Vector{Matrix{Float64}}, K_pair::Int) -> Vector{Matrix{Float64}}

Builds the `K_pair^2` cross-power raw pair-feature matrices, reusing the ALREADY-COMPUTED per-level
Fréchet-power features `Zraw_all[k] = z.^k` (`build_raw_mean_pair_matrix_levels`, UNCHANGED,
`cm_meanzc_moments.jl`) -- no new `frechet_power_feature` calls, no re-derivation of the raw draw
data. For linear index `klin` (canonical order `cross_pair_level_index(K_pair)`, `(k1,k2) =
cross_pair_level_index(K_pair)[klin]`), column `j` (pair `(o,p) = packed_pair_index(D)[j]`, `o<p`
canonical) is `Zraw_all[k1][:,o] .* Zraw_all[k2][:,p]` -- power `k1` on the LOWER-indexed origin
`o`, power `k2` on the HIGHER-indexed origin `p`, matching `pair_targets`'s
(`OriginByPowerCrossLayout` method) identical convention. Reduces to the base family's
`build_raw_mean_pair_matrices`' pair block exactly when `k1==k2`.
"""
function build_raw_cross_pair_matrix_levels(Zraw_all::Vector{Matrix{Float64}}, K_pair::Int)
    K_pair >= 1 || error("build_raw_cross_pair_matrix_levels: K_pair must be >= 1, got $K_pair")
    K_pair <= length(Zraw_all) || error("build_raw_cross_pair_matrix_levels: K_pair=$K_pair exceeds length(Zraw_all)=$(length(Zraw_all))")
    W, D = size(Zraw_all[1])
    pairs = packed_pair_index(D)
    levels = cross_pair_level_index(K_pair)
    Zpairraw_all = Vector{Matrix{Float64}}(undef, K_pair^2)
    for (klin, (k1, k2)) in enumerate(levels)
        Zk1 = Zraw_all[k1]; Zk2 = Zraw_all[k2]
        Zpk = Matrix{Float64}(undef, W, length(pairs))
        @inbounds for (j, (o, p)) in enumerate(pairs)
            @views Zpk[:, j] .= Zk1[:, o] .* Zk2[:, p]
        end
        Zpairraw_all[klin] = Zpk
    end
    return Zpairraw_all
end
