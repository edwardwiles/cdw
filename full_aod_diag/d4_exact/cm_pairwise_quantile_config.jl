# ================================================================================================
# Configuration surface + grid/bin/mass machinery for the CM + PAIRWISE-QUANTILE family (family #7,
# 2026-08-12): common marginals (CDW eq.35, optionally +eq.36) combined with the pairwise-quantile
# independence restriction (draft eq.32) whose bin masses are now SHARED across origins.
#
# WHERE THIS SITS. The four-cell table the family fills in:
#
#                          | per-origin free params | shared params (CM-combined)
#   powers-based (ZC)      | origin-ZC              | CM+ZC          (cm_meanzc_*)
#   quantile-based (PQ)    | pairwise-quantile      | CM+PQ  <-- THIS FILE
#
# THE MATH, in the order the code needs it.
#
# CM (unchanged, not re-derived here) imposes `F_o = F_ref` for every origin on a grid of
# probability levels, and is THETA-INDEPENDENT (built once from the fixed baseline draws). It pins
# the marginals TO EACH OTHER but says nothing about the LEVEL of the reference marginal. So this
# family adds:
#
#   (a) L-1 free outer parameters `mu_a` -- the binned masses of the REFERENCE marginal on the
#       pairwise-quantile grid, with `mu_L = 1 - sum_a mu_a` implicit (ONE simplex, not D of them).
#   (b) L-1 LEVEL moment rows pinning the reference marginal's binned masses to `mu`:
#             1{b_ref = a} - mu_a                                a = 1..L-1
#   (c) the pair rows (the restriction proper), keyed on the SHARED `mu`:
#             1{b_o = a, b_p = b} - mu_a * mu_b                  all C(D,2) pairs, all (a,b)
#
# and DROPS the standalone family's per-origin marginal rows `1{b_o=a} - mu_{o,a}` (o = 1..D): under
# CM + (b) those are implied, so keeping them makes the stacked moment matrix EXACTLY rank
# deficient. The identity that makes them redundant is
#
#       [1{b_o=a} - mu_a]  =  [1{b_o=a} - 1{b_ref=a}]  +  [1{b_ref=a} - mu_a]
#                              \_____ in CM's row span ____/   \___ row (b) ___/
#
# and the first bracket is in CM's row span ONLY IF each PQ bin is a union of CM grid cells, i.e.
# only if CM's grid CONTAINS the PQ cutoffs. That superset property is the load-bearing precondition
# of this entire family, and it is NOT automatic -- see the next block.
#
# ⚠️ THE SUPERSET CONDITION NEEDS THE k/G GRID, WHICH IS *NOT* CM'S CURRENT DEFAULT GRID.
# The intended CM grid (user, 2026-08-12) is the plain `k/G` one: at G=50, `0.02, 0.04, ..., 0.98`.
# On that grid the condition really is `L | G`, giving `L in {2,5,10,25}` at G=50, exactly as the
# handover doc for this family (docs/CM_PLUS_PAIRWISE_QUANTILE_HANDOVER_2026-08-12.md §1) says.
# CM's DEFAULT grid, however, is not that grid, and the difference is silent -- checked numerically
# before this file was written (2026-08-12), not reasoned about:
#
#   `precalc_common_marginals_cdf`'s default grid is `range(1/G, (G-1)/G, length=G)`, which at G=50
#   runs 0.02, 0.039592, 0.059184, ... 0.98 -- spacing 0.0195918, NOT 0.02. It is NOT the k/G grid
#   (that file's own docstring flags this: "the default is NOT literally k/L for k=1:L"). Under it,
#   the required levels are present for NO value of L whatsoever -- not L=5, not even L=2 (0.5 is
#   not on the grid). Measured: all four of L in {2,5,10,25} FAIL the superset test on the default
#   grid, and all four PASS on an explicit k/50 grid.
#
# So this family does not rely on CM's default. It builds CM on the EXPLICIT k/G grid
# `probs = (1:(G-1))/G` via `precalc_common_marginals_cdf`'s already-existing `probs=` kwarg (the
# same mechanism `nested_quantile_grids.jl` uses -- no new machinery, and no change to CM required
# for this family; passing it explicitly also insulates the family from any later change to CM's own
# default). `cm_pq_probs_grid`/`resolve_cm_pairwise_quantile_config` below enforce `L | G` and
# hard-error on an L that does not divide G, with a message that says why.
#
# WHY G-1 LEVELS AND NOT G (i.e. 49 at G=50, not 50). `p=1` must be excluded: its CDF contrast
# `1{U_o<=Inf} - 1{U_ref<=Inf}` is identically zero, a structurally ZERO moment column and hence a
# singular KKT, not merely an uninformative row. CM's own default grid excludes `p=1` for the same
# reason. Keeping 50 levels instead by moving to `k/51` would force `L | 51`, i.e. `L in {3,17}` --
# so 49 levels at `k/50` is the right shape, and `L in {2,5,10,25}` stays available.
#
# ⚠️ AND THE CUTOFFS ARE *SELECTED FROM* CM'S OWN THRESHOLD ARRAY, NEVER RECOMPUTED.
# The PQ cutoff at z-quantile level `r/L` and the CM threshold at U-CDF level `1 - r/L` are the same
# number in exact arithmetic (both `-log(r/L)` in U space; see `cm_pq_u_cutoffs_from_cm_grid`). They
# are NOT the same Float64 if computed by the two different routes: `1 - (1 - 0.2) == 0.19999999999999996`,
# not `0.2` (measured). A draw landing between the two values would be binned inconsistently by CM
# and PQ, silently breaking the superset identity for that draw. So the PQ cutoffs are obtained by
# INDEXING CM's own computed `z` array (`cm_pq_u_cutoffs_from_cm_grid`), and the PQ bin assignment is
# ALSO derivable from CM's own bin indices by a pure integer map (`cm_pq_bin_map`) with no float
# comparison at all. `assert_cm_pq_bin_consistency` checks the two routes agree for every draw and
# hard-errors if not -- the "same quantity by two routes" discipline this codebase already applies to
# `q0` (pairwise_quantile_cplus.jl) and the stick-breaking inverse (`uniform_mass_raw`).
#
# NO DEFAULTS on any field that changes which economic/statistical problem is solved (CLAUDE.md):
# every field of `CMPairwiseQuantileConfig` is a REQUIRED keyword argument, and `L`/`cm_grid_size`/
# `cm_moment_families`/`contrasts`/`min_bin_count`/`mass_start` all qualify.
#
# Requires: pairwise_quantile_bin_context.jl (PairwiseQuantileOperator, n_pair_rows),
# pairwise_quantile_mass_transform.jl (decode_origin_masses!, raw_from_origin_masses!,
# mass_jacobian_block!, pq_logit), common_marginals_moments.jl (theoretical_u_threshold).
# ================================================================================================

"""
    CMPairwiseQuantileConfig(; L, cm_grid_size, cm_moment_families, contrasts, min_bin_count,
                              mass_start)

- `L::Int`: number of pairwise-quantile BINS (`L>=2`). MUST divide `cm_grid_size` -- see this file's
  header for why that is the family's load-bearing precondition rather than a convenience.
- `cm_grid_size::Int`: the `G` of the explicit `k/G` CM probability grid (CM gets `G-1` levels,
  `p_k = k/G`, `k = 1..G-1`). Production CM uses `G=50`.
- `cm_moment_families::Int`: `1` = CM eq.35 (CDF) only; `2` = eq.35 + eq.36 (truncated
  `(sigma-1)`-power companion), the corrected production CM spec (see
  `precalc_common_marginals_cdf`). Mirrors `CMConfig.cm_moment_families`, which is itself required
  with no default.
- `contrasts::Symbol`: `:anchored` | `:orthonormal` -- CM's own contrast basis, passed straight
  through to CM's machinery.
- `min_bin_count::Int`: non-degeneracy floor asserted on every PQ marginal bin AND every PQ joint
  cell (`assert_pairwise_quantile_bins_nondegenerate`). Read against the campaign's `W` and `L`: a
  joint cell holds only ~W/L^2 draws in expectation.
- `mass_start::Symbol`: `:uniform` (`mu_a = 1/L`) | `:empirical` (the unweighted draws' own
  REFERENCE-ORIGIN bin frequencies under the fixed cutoffs). Note the difference from the standalone
  family: there is one shared simplex here, so `:empirical` reads origin `refIndex1`'s frequencies,
  not each origin's own.

There is deliberately NO `cutoff_source` field. The standalone family's `:empirical_quantile` option
puts each origin's cutoffs at its OWN empirical quantiles, which (i) are not CM grid points, so the
superset identity above fails outright, and (ii) differ across origins, so a shared `mu` would not
even be well defined. This family's cutoffs are always the CM-grid-derived theoretical ones; that is
a structural property, not a choice, so it is not offered as one.
"""
Base.@kwdef struct CMPairwiseQuantileConfig
    L::Int
    cm_grid_size::Int
    cm_moment_families::Int
    contrasts::Symbol
    min_bin_count::Int
    mass_start::Symbol
end

const CM_PQ_MASS_STARTS = (:uniform, :empirical)
const CM_PQ_CONTRASTS = (:anchored, :orthonormal)

"""
    resolve_cm_pairwise_quantile_config(cfg::CMPairwiseQuantileConfig) -> NamedTuple

Validates every field and returns the derived dimensions the rest of the family reads:
`(L, G, n_cm_levels, cm_probs, ratio, n_families, contrasts, min_bin_count, mass_start)`.

`n_cm_levels = G-1` is what CM's `L` argument must be set to (CM's own parameter named `L` is its
NUMBER OF GRID LEVELS, unrelated to this family's `L` = number of PQ bins -- a genuine name
collision between two established APIs, disambiguated here once so no downstream site has to).
"""
function resolve_cm_pairwise_quantile_config(cfg::CMPairwiseQuantileConfig)
    cfg.L >= 2 || error("CMPairwiseQuantileConfig: L (PQ bins) must be >= 2, got $(cfg.L)")
    cfg.cm_grid_size >= 2 ||
        error("CMPairwiseQuantileConfig: cm_grid_size (G) must be >= 2, got $(cfg.cm_grid_size)")
    if cfg.cm_grid_size % cfg.L != 0
        error("CMPairwiseQuantileConfig: L=$(cfg.L) does not divide cm_grid_size=$(cfg.cm_grid_size). " *
              "This family's per-origin marginal rows are DROPPED as implied by CM plus the " *
              "reference-level rows, and that implication holds only if every PQ bin is a union of " *
              "CM grid cells -- i.e. only if the PQ cutoff levels {r/L} are all CM grid levels " *
              "{k/G}, which requires L | G exactly. At L=$(cfg.L), G=$(cfg.cm_grid_size) the level " *
              "r/L = $(1/cfg.L) is not of the form k/$(cfg.cm_grid_size), so the restriction this " *
              "family would impose is NOT 'CM + shared level' -- it would silently be a strictly " *
              "weaker object with the origins' PQ-binned marginals left unpinned. Choose an L " *
              "dividing $(cfg.cm_grid_size) (for G=50: L in {2,5,10,25}) or change cm_grid_size.")
    end
    cfg.cm_moment_families in (1, 2) ||
        error("CMPairwiseQuantileConfig: cm_moment_families must be 1 (eq.35) or 2 (eq.35+eq.36), " *
              "got $(cfg.cm_moment_families)")
    cfg.contrasts in CM_PQ_CONTRASTS ||
        error("CMPairwiseQuantileConfig: contrasts must be one of $CM_PQ_CONTRASTS, got $(cfg.contrasts)")
    cfg.min_bin_count >= 1 ||
        error("CMPairwiseQuantileConfig: min_bin_count must be >= 1, got $(cfg.min_bin_count)")
    cfg.mass_start in CM_PQ_MASS_STARTS ||
        error("CMPairwiseQuantileConfig: mass_start must be one of $CM_PQ_MASS_STARTS, got $(cfg.mass_start)")
    G = cfg.cm_grid_size
    return (L = cfg.L, G = G, n_cm_levels = G - 1, cm_probs = cm_pq_probs_grid(G),
            ratio = div(G, cfg.L), n_families = cfg.cm_moment_families, contrasts = cfg.contrasts,
            min_bin_count = cfg.min_bin_count, mass_start = cfg.mass_start)
end

"""
    cm_pq_probs_grid(G::Int) -> Vector{Float64}

The EXPLICIT `k/G` CM probability grid, `p_k = k/G` for `k = 1..G-1` (length `G-1`).

This REPLACES `precalc_common_marginals_cdf`'s default grid for this family, and the replacement is
the point: the default `range(1/G,(G-1)/G,length=G)` is not a `k/G` grid at all (at G=50 its spacing
is 0.0195918, not 0.02), so no PQ cutoff level `r/L` lands on it for ANY `L` -- verified numerically
2026-08-12, see this file's header. Passed through CM's own `probs=` kwarg, so CM's machinery is used
completely unmodified.

`p=1` is excluded deliberately (`k` stops at `G-1`): its CDF contrast `1{U_o<=Inf} - 1{U_ref<=Inf}`
is identically zero, which is a structurally zero moment column, not merely an uninformative one.
"""
cm_pq_probs_grid(G::Int) = collect((1:(G-1)) ./ G)

"""
    cm_pq_grid_index(G::Int, L::Int, r::Int) -> Int

Index into `cm_pq_probs_grid(G)` of the CM grid level corresponding to the `r`-th PQ cutoff,
`r = 1..L-1`.

Derivation. The PQ cutoff `r` sits at Fréchet-`z` quantile level `r/L`: `P(z <= q_r) = r/L`. Since
`z = U^(-mu)` is strictly DECREASING in `U`, the same event in `U` space is an UPPER tail,
`{z <= q_r} = {U >= c_r}` with `P(U >= c_r) = r/L`, i.e. `P(U <= c_r) = 1 - r/L`. CM's grid level `k`
sits at `P(U <= z_k) = k/G`, so the matching index solves `k/G = 1 - r/L`, giving

    k = G - r*(G/L)                    (an integer exactly because L | G)

which is decreasing in `r`: `cm_pq_grid_index(50,5,1) = 40`, then 30, 20, 10. Bounds: `r=L-1` gives
`k = G/L >= 1`, and `r=1` gives `k = G - G/L <= G-1`, so every index is inside `1:(G-1)`.
"""
function cm_pq_grid_index(G::Int, L::Int, r::Int)
    G % L == 0 || error("cm_pq_grid_index: L=$L must divide G=$G (see resolve_cm_pairwise_quantile_config)")
    1 <= r <= L - 1 || error("cm_pq_grid_index: r must be in 1:(L-1)=$(L-1), got $r")
    k = G - r * div(G, L)
    1 <= k <= G - 1 || error("cm_pq_grid_index: internal error, k=$k outside 1:$(G-1) at (G=$G,L=$L,r=$r)")
    return k
end

"""
    cm_pq_u_cutoffs_from_cm_grid(z_cm::AbstractVector{Float64}, G::Int, L::Int) -> Vector{Float64}

The `L-1` PQ cutoffs in `U` space, obtained by INDEXING CM's own already-computed threshold array
`z_cm` (`= theoretical_u_threshold.(cm_pq_probs_grid(G))`, i.e. whatever floats CM is actually
using) at the indices `cm_pq_grid_index` selects -- never by evaluating a formula a second time.

Returned in the order `r = 1..L-1`, i.e. DECREASING in `U` (`c_1 > c_2 > ... > c_{L-1}`), because
`r` counts up the `z` quantiles and `z` is decreasing in `U`. `cm_pq_z_cutoffs_from_u` flips this
into the ascending `z`-space cutoff matrix `PairwiseQuantileOperator` wants.

Why not `theoretical_u_threshold(1 - r/L)`: `1 - (1 - 0.2)` is `0.19999999999999996`, so that route
can produce a threshold one ulp away from the CM threshold at the same mathematical level, which
would bin a draw falling between them into inconsistent CM and PQ cells and silently break the
superset identity the whole family rests on.
"""
function cm_pq_u_cutoffs_from_cm_grid(z_cm::AbstractVector{Float64}, G::Int, L::Int)
    length(z_cm) == G - 1 ||
        error("cm_pq_u_cutoffs_from_cm_grid: length(z_cm)=$(length(z_cm)) != G-1=$(G-1) -- z_cm must be " *
              "CM's threshold array for the k/G grid (cm_pq_probs_grid(G))")
    c = Vector{Float64}(undef, L - 1)
    @inbounds for r in 1:(L-1)
        c[r] = z_cm[cm_pq_grid_index(G, L, r)]
    end
    @inbounds for r in 2:(L-1)
        c[r] < c[r-1] ||
            error("cm_pq_u_cutoffs_from_cm_grid: CM thresholds at the selected indices are not strictly " *
                  "decreasing in r (c[$(r-1)]=$(c[r-1]), c[$r]=$(c[r])) -- CM's z array is not ascending, " *
                  "or the index map is wrong.")
    end
    return c
end

"""
    cm_pq_z_cutoffs_from_u(c_u::AbstractVector{Float64}, D::Int; mu_frechet::Float64) -> Matrix{Float64}

The `(L-1) x D` ASCENDING Fréchet-`z` cutoff matrix `PairwiseQuantileOperator` requires, from the
`U`-space cutoffs `c_u` (descending in `r`, as `cm_pq_u_cutoffs_from_cm_grid` returns them):
`Q[r,o] = c_u[r]^(-mu_frechet)` for every origin.

Identical across origins by construction -- every `z_o` has the same calibrated Fréchet marginal, so
the cutoffs are common; that is NOT the same thing as the MASSES being common (which is this
family's own added restriction, imposed through moment rows, not through the bins).

`mu_frechet` is REQUIRED with no default (CLAUDE.md): it is `ctx.muHat`, and the bins are meaningless
without recording which one was used.
"""
function cm_pq_z_cutoffs_from_u(c_u::AbstractVector{Float64}, D::Int; mu_frechet::Float64)
    mu_frechet > 0 || error("cm_pq_z_cutoffs_from_u: mu_frechet must be > 0, got $mu_frechet")
    D >= 2 || error("cm_pq_z_cutoffs_from_u: D must be >= 2, got $D")
    nc = length(c_u)
    Q = Matrix{Float64}(undef, nc, D)
    @inbounds for r in 1:nc
        # r counts UP the z quantiles while c_u counts DOWN in U, and x -> x^(-mu) is decreasing,
        # so Q is automatically ASCENDING in r. Asserted below rather than assumed.
        qr = c_u[r]^(-mu_frechet)
        for o in 1:D
            Q[r, o] = qr
        end
    end
    @inbounds for r in 2:nc
        Q[r, 1] > Q[r-1, 1] ||
            error("cm_pq_z_cutoffs_from_u: z-space cutoffs are not strictly ascending in r " *
                  "(Q[$(r-1)]=$(Q[r-1,1]), Q[$r]=$(Q[r,1])) -- the U->z orientation flip is wrong.")
    end
    return Q
end

"""
    cm_pq_bin_map(G::Int, L::Int) -> Vector{Int}

Pure INTEGER map from a CM bin index to a PQ bin index: `map[k]` is the PQ bin containing CM bin `k`,
for `k = 1..G` (CM has `G-1` thresholds, hence `G` bins, `compute_bin_indices`' own convention:
bin `k` is `(z_{k-1}, z_k]`, bin 1 is `(-Inf, z_1]`, bin `G` is `(z_{G-1}, Inf)`).

This is the constructive statement of the superset property -- each PQ bin is a UNION of consecutive
CM bins -- and it involves no floating-point comparison whatsoever, which is exactly why it is worth
having alongside the `searchsortedfirst`-on-`Q` route (`assert_cm_pq_bin_consistency` cross-checks
them).

Construction. PQ bin `a` is `{c_a <= U < c_{a-1}}` (`c_0 := +Inf`, `c_L := 0`), with
`c_a = z_cm[k_a]`, `k_a = cm_pq_grid_index(G,L,a)` decreasing in `a`. CM bins strictly above `k_a`
and at or below `k_{a-1}` are precisely those inside PQ bin `a`, so

    map[k] = a   for   k_a < k <= k_{a-1}          (`k_0 := G`, `k_L := 0`)

Boundaries differ by a single point (CM's cells are right-closed, PQ's are left-closed), a
probability-zero event for a.s.-continuous draws -- and `assert_cm_pq_bin_consistency` will catch it
loudly rather than let it pass, if it ever happens.
"""
function cm_pq_bin_map(G::Int, L::Int)
    G % L == 0 || error("cm_pq_bin_map: L=$L must divide G=$G")
    map = zeros(Int, G)
    @inbounds for a in 1:L
        k_hi = a == 1 ? G : cm_pq_grid_index(G, L, a - 1)
        k_lo = a == L ? 0 : cm_pq_grid_index(G, L, a)
        for k in (k_lo+1):k_hi
            map[k] = a
        end
    end
    all(>(0), map) ||
        error("cm_pq_bin_map: CM bins $(findall(iszero, map)) were not assigned to any PQ bin -- " *
              "the index map does not tile 1:$G.")
    return map
end

"""
    assert_cm_pq_bin_consistency(op::PairwiseQuantileOperator, Bidx::AbstractMatrix{<:Integer},
                                 G::Int) -> NamedTuple

Hard-errors unless the PQ bin assignment reached by the two INDEPENDENT routes agrees for every one
of the `W*D` (draw, origin) cells:

  1. `op.bin` -- `searchsortedfirst` on the `z`-space cutoffs `Q` (built by
     `PairwiseQuantileOperator`, i.e. the standalone family's own unmodified code path);
  2. `cm_pq_bin_map(G,L)[Bidx[w,o]]` -- CM's own bin index pushed through the pure integer map.

A mismatch means the superset identity fails for that draw, so the per-origin marginal rows this
family DROPS are not in fact implied, and the restriction being solved is not the one intended. That
is a hard error, not a warning: the failure is silent in every downstream number.

Returns the (draw, origin) count checked, for the run record.
"""
function assert_cm_pq_bin_consistency(op::PairwiseQuantileOperator, Bidx::AbstractMatrix{<:Integer}, G::Int)
    W = op.W; D = op.D; L = op.L
    size(Bidx) == (W, D) ||
        error("assert_cm_pq_bin_consistency: size(Bidx)=$(size(Bidx)) != (W,D)=($W,$D)")
    bmap = cm_pq_bin_map(G, L)
    nbad = 0; first_bad = (0, 0, 0, 0)
    @inbounds for o in 1:D, w in 1:W
        k = Int(Bidx[w, o])
        (1 <= k <= G) ||
            error("assert_cm_pq_bin_consistency: CM bin index $k at (w=$w,o=$o) outside 1:$G -- CM was " *
                  "not built on the $(G-1)-level k/G grid this family requires.")
        a_map = bmap[k]
        a_op = Int(op.bin[w, o])
        if a_map != a_op
            nbad += 1
            first_bad == (0, 0, 0, 0) && (first_bad = (w, o, a_op, a_map))
        end
    end
    nbad == 0 ||
        error("assert_cm_pq_bin_consistency: $nbad of $(W*D) (draw,origin) cells disagree between the " *
              "z-space searchsortedfirst bin and CM's own bin pushed through the integer map " *
              "(first: w=$(first_bad[1]), o=$(first_bad[2]), op.bin=$(first_bad[3]), " *
              "map(Bidx)=$(first_bad[4])). The PQ bins are therefore NOT unions of CM grid cells, so " *
              "the per-origin marginal rows this family drops are not implied by CM plus the " *
              "reference-level rows, and the restriction actually imposed is weaker than intended.")
    return (n_checked = W * D, L = L, G = G)
end

# ------------------------------------------------------------------------------------------------
# Row layout of the restriction block: [level rows (L-1) | pair rows (L-1)^2 * C(D,2)]
#
# Deliberately the SAME internal (a,b,pidx) ordering as the standalone family's `pair_row`
# (`... + (pidx-1)*nc^2 + (b-1)*nc + a`), so `reshape(v, nc, nc, npair)`'s natural column-major
# layout indexes the pair duals with no transpose -- the property `dual_index!` relies on, and the
# one whose marginal-block analogue caused a real bug on 2026-08-09 (see
# pairwise_quantile_production.jl's own comment). Only the OFFSET differs: `(L-1)` reference-level
# rows here, versus `(L-1)*D` per-origin marginal rows there.
# ------------------------------------------------------------------------------------------------

"Number of reference-LEVEL moment rows: `L-1` (one per free bin of the single shared simplex)."
n_cmpq_level_rows(L::Int) = L - 1

"Number of pair moment rows: `(L-1)^2 * C(D,2)` -- identical to the standalone family's `n_pair_rows`."
n_cmpq_pair_rows(D::Int, L::Int) = n_pair_rows(D, L)

"Total rows in this family's OWN restriction block (excludes CM's block, which is counted separately
by `n_cm_moments`)."
n_cmpq_restr_rows(D::Int, L::Int) = n_cmpq_level_rows(L) + n_cmpq_pair_rows(D, L)

"Row index of the level row for free bin `a` within this family's restriction block."
cmpq_level_row(a::Integer) = Int(a)

"Row index of pair cell `(pidx,a,b)` within this family's restriction block."
function cmpq_pair_row(D::Integer, pidx::Integer, a::Integer, b::Integer, L::Integer)
    nc = Int(L) - 1
    return n_cmpq_level_rows(Int(L)) + (Int(pidx) - 1) * nc^2 + (Int(b) - 1) * nc + Int(a)
end

# ------------------------------------------------------------------------------------------------
# The ONE shared simplex: state + decode. Reuses the standalone family's stick-breaking transform
# FUNCTIONS verbatim (`decode_origin_masses!`, `raw_from_origin_masses!`, `mass_jacobian_block!`,
# `pq_logit` -- all of which already operate on ONE origin's row and take plain vectors, so they need
# no generalization at all); only the STATE shrinks from `D x (L-1)` to length `L-1`.
#
# `PairwiseQuantileMassLayout` is deliberately NOT reused: it requires `D >= 2` (correctly, for the
# standalone family) and its `n_raw`/`raw_index` are about the origin-major stacking of D simplices,
# which is exactly what this family does not have.
# ------------------------------------------------------------------------------------------------

"""
    CMPQMassState(L::Int)

Mutable per-outer-point state for the SINGLE shared reference-marginal simplex:
  - `mu[a]`, `a = 1..L-1` -- the free masses (shared by every origin, which is the whole point);
  - `mu_last` -- the implied remainder `mu_L = 1 - sum(mu) > 0`, never a free coordinate;
  - `Pcum[a] = sum_{j<=a} mu[j]` -- cumulative masses, the form the math note states the
    equivalence in, and what the verifier reports against.

Refreshed exactly ONCE per outer point (`set_cmpq_masses!`), never inside an FG/Hessian callback --
the same lifecycle contract `set_pairwise_quantile_masses!` carries, for the same reason: everything
downstream reads `mu` as a constant of the inner problem.
"""
mutable struct CMPQMassState
    mu::Vector{Float64}
    mu_last::Float64
    Pcum::Vector{Float64}
end

CMPQMassState(L::Int) = (L >= 2 || error("CMPQMassState: L must be >= 2, got $L");
                         CMPQMassState(zeros(L - 1), 0.0, zeros(L - 1)))

"Number of raw (unconstrained-real) outer coordinates this family's restriction adds: `L-1`.
Contrast the standalone family's `(L-1)*D` -- collapsing that is the OUTER-dimension win this family
exists for (80 -> 4 at D=20, L=5)."
n_cmpq_raw(L::Int) = L - 1

"""
    set_cmpq_masses!(state::CMPQMassState, raw::AbstractVector{Float64}) -> state

Decode the `L-1` raw KNITRO coordinates into the shared free masses via the standalone family's own
`decode_origin_masses!` (ONE simplex, so this is a single call, not a loop over origins), then fill
the derived remainder/cumulative tables.

Asserts the open-simplex invariant rather than assuming it: a violation means the decode is wrong and
every centering constant downstream would be silently wrong with it (same assertion, same reason, as
`set_pairwise_quantile_masses!`).
"""
function set_cmpq_masses!(state::CMPQMassState, raw::AbstractVector{Float64})
    nb = length(state.mu)
    length(raw) == nb ||
        error("set_cmpq_masses!: length(raw)=$(length(raw)) != L-1=$nb")
    state.mu_last = decode_origin_masses!(state.mu, raw)
    acc = 0.0
    @inbounds for a in 1:nb
        acc += state.mu[a]
        state.Pcum[a] = acc
    end
    (state.mu_last > 0.0 && acc < 1.0) ||
        error("set_cmpq_masses!: decoded masses are off the open simplex (sum of free masses = $acc, " *
              "remainder = $(state.mu_last))")
    return state
end

"""
    cmpq_uniform_mass_raw(L::Int) -> Vector{Float64}

Raw coordinates for `mu_a = 1/L` on every free bin -- the theoretical Fréchet-quantile masses, i.e.
the point at which this restriction says the reference marginal is EXACTLY the calibrated one. Built
through `raw_from_origin_masses!` and cross-checked against the closed form `-log(L-a)`, the same
two-independent-routes check `uniform_mass_raw` makes.
"""
function cmpq_uniform_mass_raw(L::Int)
    L >= 2 || error("cmpq_uniform_mass_raw: L must be >= 2, got $L")
    nb = L - 1
    buf = Vector{Float64}(undef, nb)
    raw_from_origin_masses!(buf, fill(1.0 / L, nb))
    @inbounds for a in 1:nb
        closed = -log(float(L - a))
        isapprox(buf[a], closed; atol = 1e-12, rtol = 1e-12) ||
            error("cmpq_uniform_mass_raw: stick-breaking inverse ($(buf[a])) disagrees with the closed " *
                  "form -log(L-a)=$closed at a=$a, L=$L -- one of the two is wrong.")
    end
    return buf
end

"""
    cmpq_empirical_mass_raw(bin_counts::AbstractMatrix{<:Real}, refIndex1::Int, L::Int) -> Vector{Float64}

Raw coordinates placing the SHARED masses at the reference origin's own unweighted bin frequencies
under the fixed cutoffs (`bin_counts[refIndex1,a]/W`), from `pairwise_quantile_bin_counts`.

The reference origin specifically -- not an average over origins -- because the level rows this
family imposes are the REFERENCE origin's (`1{b_ref=a} - mu_a`), so this is the point at which those
rows are satisfied exactly by the unweighted draws, leaving the restriction as a pure dependence
restriction with no level slack to absorb. (Under exact CM the origins agree anyway; they differ here
only by the draws' own Monte Carlo error, which is precisely what the CM rows are absorbing.)
"""
function cmpq_empirical_mass_raw(bin_counts::AbstractMatrix{<:Real}, refIndex1::Int, L::Int)
    L >= 2 || error("cmpq_empirical_mass_raw: L must be >= 2, got $L")
    size(bin_counts, 2) == L ||
        error("cmpq_empirical_mass_raw: size(bin_counts,2)=$(size(bin_counts,2)) != L=$L")
    1 <= refIndex1 <= size(bin_counts, 1) ||
        error("cmpq_empirical_mass_raw: refIndex1=$refIndex1 outside 1:$(size(bin_counts,1))")
    tot = sum(@view bin_counts[refIndex1, :])
    tot > 0 || error("cmpq_empirical_mass_raw: reference origin $refIndex1 has no draws at all")
    nb = L - 1
    mu_row = Vector{Float64}(undef, nb)
    @inbounds for a in 1:nb
        mu_row[a] = bin_counts[refIndex1, a] / tot
    end
    buf = Vector{Float64}(undef, nb)
    raw_from_origin_masses!(buf, mu_row)
    return buf
end

"""
    cmpq_default_raw_mass_bounds(bin_counts::AbstractMatrix{<:Real}, refIndex1::Int, L::Int)
        -> Vector{NTuple{2,Float64}}

Data-derived KNITRO box for the `L-1` shared raw coordinates, centred on the reference origin's own
conditional bin frequencies and widened by an explicit margin in LOG-ODDS (the coordinate KNITRO
actually moves) -- the single-simplex analogue of `default_raw_mass_bounds`, using the same
`ODDS_MARGIN = 16.0` and the same `v in [1/W, 1-1/W]` resolvability clamp, so the two families' boxes
are the same object in the same units and a cross-family comparison is not confounded by the box.
"""
function cmpq_default_raw_mass_bounds(bin_counts::AbstractMatrix{<:Real}, refIndex1::Int, L::Int)
    L >= 2 || error("cmpq_default_raw_mass_bounds: L must be >= 2, got $L")
    size(bin_counts, 2) == L ||
        error("cmpq_default_raw_mass_bounds: size(bin_counts,2)=$(size(bin_counts,2)) != L=$L")
    ODDS_MARGIN = 16.0
    W_o = sum(@view bin_counts[refIndex1, :])
    W_o > 0 || error("cmpq_default_raw_mass_bounds: reference origin $refIndex1 has no draws at all")
    v_floor = 1.0 / W_o
    v_ceil = 1.0 - 1.0 / W_o
    v_floor < v_ceil ||
        error("cmpq_default_raw_mass_bounds: W=$W_o is too small to resolve any conditional bin probability")
    bounds = Vector{NTuple{2,Float64}}(undef, L - 1)
    remaining = float(W_o)
    @inbounds for a in 1:(L-1)
        remaining > 0 ||
            error("cmpq_default_raw_mass_bounds: reference origin has no draws at or above bin $a")
        vhat = clamp(bin_counts[refIndex1, a] / remaining, v_floor, v_ceil)
        c = pq_logit(vhat)
        lo = max(c - log(ODDS_MARGIN), pq_logit(v_floor))
        hi = min(c + log(ODDS_MARGIN), pq_logit(v_ceil))
        lo < hi ||
            error("cmpq_default_raw_mass_bounds: degenerate box at bin=$a (lo=$lo hi=$hi, vhat=$vhat, W=$W_o)")
        bounds[a] = (lo, hi)
        remaining -= bin_counts[refIndex1, a]
    end
    return bounds
end
