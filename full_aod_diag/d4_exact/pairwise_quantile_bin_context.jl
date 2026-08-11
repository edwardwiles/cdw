# ================================================================================================
# Fixed-cutoff bin assignment + free-mass state for the pairwise-quantile-independence restriction
# (draft eq. 32), version B (free-mass reparameterization, 2026-08-10).
#
# Two structs, mirroring `zc_restriction_operator.jl`'s ZCRestrictionOperator/*Workspace split:
#   - PairwiseQuantileOperator: IMMUTABLE, campaign-lifetime. Under version B this now also owns
#     the CUTOFFS `Q` and the decoded bin assignment `bin[w,o]`, because both are campaign
#     constants: the cutoffs are fixed once per campaign (that is the whole point of the
#     reparameterization) and `ctx.U` never changes, so `bin` is computed exactly ONCE, in this
#     struct's constructor, rather than once per outer point.
#   - PairwiseQuantileMassState: MUTABLE, refreshed exactly once per OUTER point (never inside an
#     FG/Hessian callback). Holds the decoded free bin masses `mu[o,a]`, the implied remainder
#     `mu_last[o] = mu_{o,L}`, and the cumulative masses `Pcum[o,a]` the verifier reports against.
#
# WHAT VERSION A HAD HERE AND WHY IT IS GONE. Version A stored `sorted_z`/`sorted_idx` (each origin's
# draws presorted, plus the permutation) on the operator, existing SOLELY to let the cutoff-crossing
# gradient find, in O(log W + k), the draws whose bin membership changes when a cutoff moves. With
# the cutoffs fixed, no cutoff ever moves, that gradient does not exist, and the presorted arrays
# have no consumer -- so they are deleted with it, not carried "just in case"
# (docs/PAIRWISE_QUANTILE_FREE_MASS_REPARAMETERIZATION_HANDOVER_2026-08-10.md section 3.3). One
# genuine consumer of sorting remains, `pairwise_quantile_fixed_cutoffs` under
# `:empirical_quantile`, and it sorts locally at context-build time.
#
# Requires pairwise_quantile_mass_transform.jl (PairwiseQuantileMassLayout, decode_all_masses!) and
# cm_meanzc_moments.jl (packed_pair_index, pair_oi_to_lin) to already be included.
# ================================================================================================

"""
    pairwise_quantile_fixed_cutoffs(U::AbstractMatrix{Float64}, L::Int; cutoff_source::Symbol) -> Matrix{Float64}

Builds the `(L-1) x D` FIXED cutoff matrix `Q[r,o] = q_{o,r}` that defines this restriction's bins
for the entire campaign. `cutoff_source` is REQUIRED with no default (CLAUDE.md's no-silent-defaults
rule): the whole meaning of the restriction depends on where its bins are, and a run recorded
without it is not reproducible.

  - `:frechet_theoretical` -- the THEORETICAL population quantiles of each origin's calibrated
    marginal. In this codebase `ctx.U` holds Exp(1) draws (`draw_design.jl`: `U = -log(1-U01)`,
    one shared transform for every design), and the Fréchet productivity is `z_o = U_o^{-mu_hat}`,
    a strictly DECREASING bijection of `U_o`. So the population `r/L` quantile of `U_o` is
    `-log(1 - r/L)`, identical across origins, and binning `U` there is the SAME partition of
    draws as binning `z` at its own Fréchet quantiles `(-log(r/L))^{-mu_hat}` -- only the bin
    LABELS are reversed, and this restriction is invariant to relabelling bins (it constrains bin
    masses and the independence of bin memberships, neither of which depends on the labels).
    Cutoffs common across origins is NOT the same thing as masses common across origins; the
    latter would silently add a marginal restriction and is forbidden (see
    `PairwiseQuantileMassState`).
  - `:empirical_quantile` -- each origin's OWN empirical `r/L` quantile of `U[:,o]`, using the same
    `sorted[clamp(round(Int, (r/L)*W),1,W)]` convention version A's `pairwise_quantile_start_cutoffs`
    used. This is the setting that makes version B reproduce version A exactly at `mu = 1/L`
    (the equivalence anchor).

Both choices are legitimate; neither is a default.
"""
function pairwise_quantile_fixed_cutoffs(U::AbstractMatrix{Float64}, L::Int; cutoff_source::Symbol)
    W, D = size(U)
    L >= 2 || error("pairwise_quantile_fixed_cutoffs: L (n_bins) must be >= 2, got $L")
    nc = L - 1
    Q = Matrix{Float64}(undef, nc, D)
    if cutoff_source === :frechet_theoretical
        for r in 1:nc
            qr = -log(1.0 - r / L)
            for o in 1:D
                Q[r, o] = qr
            end
        end
    elseif cutoff_source === :empirical_quantile
        for o in 1:D
            s = sort(collect(@view U[:, o]))
            for r in 1:nc
                Q[r, o] = s[clamp(round(Int, (r / L) * W), 1, W)]
            end
        end
    else
        error("pairwise_quantile_fixed_cutoffs: cutoff_source must be :frechet_theoretical|:empirical_quantile, " *
              "got :$cutoff_source (no default -- see this function's docstring)")
    end
    @inbounds for o in 1:D
        for r in 2:nc
            Q[r, o] > Q[r-1, o] ||
                error("pairwise_quantile_fixed_cutoffs(:$cutoff_source): cutoffs $(r-1) and $r coincide or " *
                      "invert at origin=$o (q=$(Q[r-1,o]) then $(Q[r,o])) -- the draws are too coarse for " *
                      "L=$L bins at this W.")
        end
    end
    return Q
end

"""
    PairwiseQuantileOperator(U::Matrix{Float64}, L::Int, Q::Matrix{Float64})

`U` is `ctx.U` (W x D raw draws, immutable for the whole campaign), `L` the number of quantile BINS
per origin (`L>=2`), `Q` the `(L-1) x D` FIXED cutoffs from `pairwise_quantile_fixed_cutoffs`.
`L` and `Q` are both REQUIRED arguments -- no defaults (CLAUDE.md).

Builds, ONCE:
  - `bin[w,o] in 1:L` (`W x D`, `UInt8`) via `searchsortedfirst(Q[:,o], U[w,o])` -- the SAME
    convention `cm_hessian_architectures.jl::compute_bin_indices` /
    `common_marginals_interval.jl::compute_bin_indices` already use (bin k covers
    `(Q[k-1,o], Q[k,o]]`, bin 1 covers `(-Inf, Q[1,o]]`, bin L covers `(Q[L-1,o], Inf)`), reused
    rather than reinvented. `U_o` is a.s.-continuous, so the `<=`-vs-`<` boundary convention is a
    probability-zero event and does not affect the math note's equivalence proofs.
  - `pairs = packed_pair_index(D)` (REUSED from `cm_meanzc_moments.jl`, `(o,p)` with `o<p`,
    o-outer-loop-major) -- the SAME ordering convention every other pair-indexed quantity in this
    codebase uses.
  - the T3/T4 Hessian combo registries (campaign-lifetime, depend only on D/pairs).

Because `bin` is campaign-constant under version B, EVERY downstream consumer reads `op.bin`, and
the whole class of "some later solve overwrote the bin state" hazards that version A had to guard
against (`ensure_pq_bins!`) is gone by construction. The equivalent hazard now attaches to the
MASSES instead, and is guarded the same way (`ensure_pq_masses!`).
"""
struct PairwiseQuantileOperator
    D::Int
    L::Int
    W::Int
    npair::Int
    pairs::Vector{Tuple{Int,Int}}
    Q::Matrix{Float64}         # (L-1) x D, FIXED cutoffs
    bin::Matrix{UInt8}         # W x D, campaign-constant bin assignment
    # ---- Hessian table-combo registries (Section 5) -- campaign-lifetime, depend only on D/pairs,
    # never on draws/duals/masses, so precomputed ONCE here rather than rebuilt inside the Hessian
    # file. `triple_lookup[o,pidx]` = 1-based index into the T3 combo axis if origin `o` is disjoint
    # from `pairs[pidx]`, else 0 (sentinel: NOT a Dict -- a plain D x npair Int matrix, O(1)
    # lookup). `triple_combos[k] = (o,pidx)` is the inverse map.
    triple_lookup::Matrix{Int}
    triple_combos::Vector{Tuple{Int,Int}}
    # `quad_lookup[pidx1,pidx2]` = 1-based index into the T4 combo axis if `pairs[pidx1]` and
    # `pairs[pidx2]` share NO origin, for pidx1<pidx2 ONLY (canonical order -- the disjoint block's
    # transpose is obtained by transposing the SAME stored sub-block, never a second copy).
    quad_lookup::Matrix{Int}
    quad_combos::Vector{Tuple{Int,Int}}
end

function PairwiseQuantileOperator(U::AbstractMatrix{Float64}, L::Int, Q::AbstractMatrix{Float64})
    W, D = size(U)
    D >= 2 || error("PairwiseQuantileOperator: D must be >= 2, got $D")
    L >= 2 || error("PairwiseQuantileOperator: L (n_bins) must be >= 2, got $L")
    size(Q) == (L - 1, D) || error("PairwiseQuantileOperator: size(Q)=$(size(Q)) != ($(L-1),$D)")
    pairs = packed_pair_index(D)
    npair = length(pairs)
    npair == div(D * (D - 1), 2) || error("PairwiseQuantileOperator: internal pair-count mismatch")

    Qc = Matrix{Float64}(Q)   # own copy: the operator is immutable, and a caller mutating its Q
                              # afterwards would silently desynchronize it from `bin`.
    bin = Matrix{UInt8}(undef, W, D)
    @inbounds for o in 1:D
        qcol = @view Qc[:, o]
        for w in 1:W
            bin[w, o] = UInt8(searchsortedfirst(qcol, U[w, o]))
        end
    end

    triple_lookup = zeros(Int, D, npair)
    triple_combos = Tuple{Int,Int}[]
    for pidx in 1:npair
        (p, q) = pairs[pidx]
        for o in 1:D
            (o == p || o == q) && continue
            push!(triple_combos, (o, pidx))
            triple_lookup[o, pidx] = length(triple_combos)
        end
    end

    quad_lookup = zeros(Int, npair, npair)
    quad_combos = Tuple{Int,Int}[]
    for pidx1 in 1:npair-1
        (o1, o2) = pairs[pidx1]
        for pidx2 in pidx1+1:npair
            (o3, o4) = pairs[pidx2]
            if o1 != o3 && o1 != o4 && o2 != o3 && o2 != o4
                push!(quad_combos, (pidx1, pidx2))
                quad_lookup[pidx1, pidx2] = length(quad_combos)
            end
        end
    end

    return PairwiseQuantileOperator(D, L, W, npair, pairs, Qc, bin,
        triple_lookup, triple_combos, quad_lookup, quad_combos)
end

"pair_index(op, o, p) -> Int: O(1) linear index into the packed pair convention, o,p in either order."
pair_index(op::PairwiseQuantileOperator, o::Int, p::Int) = pair_oi_to_lin(o, p, op.D)

"n_marginal_rows(D,L) = (L-1)*D; n_pair_rows(D,L) = (L-1)^2*C(D,2); n_total_rows(D,L) = their sum.
`L` is the number of quantile bins."
n_marginal_rows(D::Int, L::Int) = (L - 1) * D
n_pair_rows(D::Int, L::Int) = (L - 1)^2 * div(D * (D - 1), 2)
n_total_rows(D::Int, L::Int) = n_marginal_rows(D, L) + n_pair_rows(D, L)

"""
    pairwise_quantile_bin_counts(op::PairwiseQuantileOperator) -> Matrix{Int}

`counts[o,a]` = number of draws in bin `a` of origin `o` (all `L` bins), from the operator's
campaign-constant `bin`. Cheap (`O(W*D)`), computed at context-build time only -- it feeds the
non-degeneracy gate below, the data-derived mass box (`default_raw_mass_bounds`) and the
empirical-mass starting point (`empirical_mass_raw`).
"""
function pairwise_quantile_bin_counts(op::PairwiseQuantileOperator)
    D = op.D; L = op.L; W = op.W
    counts = zeros(Int, D, L)
    bin = op.bin
    @inbounds for o in 1:D, w in 1:W
        counts[o, bin[w, o]] += 1
    end
    return counts
end

"""
    pairwise_quantile_joint_min_count(op::PairwiseQuantileOperator) -> (min_count, argmin_cell)

Smallest occupancy over ALL `L^2` joint cells of ALL `npair` pairs, with the (pidx,a,b) achieving
it. `O(W*npair)`, context-build time only. A marginal bin can be perfectly healthy while some
joint cell is empty -- and an empty joint cell means that pair's `(a,b)` moment row is the constant
`-mu_{o,a}*mu_{p,b}` on every draw, i.e. a row with no draw-side variation at all, which is exactly
the degeneracy the handover asks be caught loudly rather than handed to KNITRO.
"""
function pairwise_quantile_joint_min_count(op::PairwiseQuantileOperator)
    D = op.D; L = op.L; W = op.W; npair = op.npair
    bin = op.bin; pairs = op.pairs
    tab = zeros(Int, L, L, npair)
    @inbounds for w in 1:W, pidx in 1:npair
        (o, p) = pairs[pidx]
        tab[bin[w, o], bin[w, p], pidx] += 1
    end
    mn = typemax(Int); cell = (0, 0, 0)
    @inbounds for pidx in 1:npair, b in 1:L, a in 1:L
        if tab[a, b, pidx] < mn
            mn = tab[a, b, pidx]; cell = (pidx, a, b)
        end
    end
    return (mn, cell)
end

"""
    assert_pairwise_quantile_bins_nondegenerate(op; min_bin_count::Int) -> NamedTuple

Hard-errors unless EVERY marginal bin and EVERY joint cell holds at least `min_bin_count` draws.
`min_bin_count` is REQUIRED with no default: it is a genuine per-campaign choice (it must be read
against `W` and `L` -- at `W=100,000, L=10` a joint cell holds ~1,000 draws in expectation, at
`W=20,000, L=10` only ~200), and the handover asks explicitly that degenerate bins fail loudly
rather than be discovered as a mysterious inner-solve failure later.

Returns the measured occupancy summary for logging/checkpointing, so a run records what it actually
had rather than only that it passed.
"""
function assert_pairwise_quantile_bins_nondegenerate(op::PairwiseQuantileOperator; min_bin_count::Int)
    min_bin_count >= 1 ||
        error("assert_pairwise_quantile_bins_nondegenerate: min_bin_count must be >= 1, got $min_bin_count")
    counts = pairwise_quantile_bin_counts(op)
    mmin, midx = findmin(counts)
    if mmin < min_bin_count
        (o, a) = Tuple(midx)
        error("assert_pairwise_quantile_bins_nondegenerate: marginal bin (origin=$o, bin=$a) holds only " *
              "$mmin draws, below min_bin_count=$min_bin_count (W=$(op.W), L=$(op.L)). The fixed cutoffs " *
              "leave this bin too sparse for its moment row to carry information -- raise W, lower L, or " *
              "choose a different cutoff_source.")
    end
    (jmin, jcell) = pairwise_quantile_joint_min_count(op)
    if jmin < min_bin_count
        (pidx, a, b) = jcell
        (o, p) = op.pairs[pidx]
        error("assert_pairwise_quantile_bins_nondegenerate: joint cell (origins=($o,$p), cell=($a,$b)) holds " *
              "only $jmin draws, below min_bin_count=$min_bin_count (W=$(op.W), L=$(op.L)). That pair-" *
              "independence moment row has essentially no draw-side variation -- raise W, lower L, or " *
              "choose a different cutoff_source.")
    end
    return (min_marginal_count = mmin, min_joint_count = jmin, min_joint_cell = jcell,
            bin_counts = counts)
end

"""
    PairwiseQuantileMassState(D::Int, L::Int)

Mutable per-outer-point state: the decoded FREE bin masses.
  - `mu[o,a]` (`D x (L-1)`) -- the free masses, indexed exactly as `lambda_M[o,a]` is.
  - `mu_last[o]` -- the implied remainder `mu_{o,L} = 1 - sum_a mu[o,a] > 0`, never a free
    coordinate, carried for the verifier's probability report.
  - `Pcum[o,a] = sum_{j<=a} mu[o,j]` -- the CUMULATIVE masses. The math note
    (PAIRWISE_QUANTILE_INDEPENDENCE_MATH_NOTE_2026-08-09.md Sections 2-4) states the whole
    equivalence in cumulative form (`P(z_o<q_r)=p_r`, `F(r,s)=p_r p_s`), so the verifier's
    factorization residuals are computed against these rather than against `r/L`.

MASSES ARE PER-ORIGIN AND MUST STAY THAT WAY. Making `mu` common across origins would silently
convert this from a pure DEPENDENCE restriction into "independence AND all origins share one set of
bin masses" -- a strictly stronger, different restriction (the earlier
`pairwise_grid_common_marginal` restriction in `trade_robustness_modular_perf` did exactly that; it
is prior art, not the same object). Nothing in this file or downstream may collapse the `o` index.
"""
mutable struct PairwiseQuantileMassState
    mu::Matrix{Float64}        # D x (L-1)
    mu_last::Vector{Float64}   # D
    Pcum::Matrix{Float64}      # D x (L-1)
end

PairwiseQuantileMassState(D::Int, L::Int) =
    PairwiseQuantileMassState(zeros(D, L - 1), zeros(D), zeros(D, L - 1))

"""
    set_pairwise_quantile_masses!(state, raw, layout) -> state

ONE per-outer-point refresh: decode the `n_raw(layout)` raw KNITRO coordinates into free masses
(`decode_all_masses!`, the stick-breaking transform) and fill the derived remainder/cumulative
tables.

Caller's responsibility: call this ONCE per outer point, BEFORE starting the inner KNITRO dual
solve -- NEVER from inside an FG or Hessian callback. This is the same lifecycle contract version A
imposed on `refresh_pairwise_quantile_bins!`, and for the same reason: everything downstream reads
`state.mu` as a constant of the inner problem.
"""
function set_pairwise_quantile_masses!(state::PairwiseQuantileMassState, raw::AbstractVector{Float64},
                                       layout::PairwiseQuantileMassLayout)
    D = layout.D; nb = n_free_bins(layout)
    size(state.mu) == (D, nb) ||
        error("set_pairwise_quantile_masses!: size(state.mu)=$(size(state.mu)) != ($D,$nb)")
    decode_all_masses!(state.mu, state.mu_last, raw, layout)
    @inbounds for o in 1:D
        acc = 0.0
        for a in 1:nb
            acc += state.mu[o, a]
            state.Pcum[o, a] = acc
        end
        # Structural invariant of the stick-breaking transform, asserted rather than assumed: a
        # violation here means the decode is wrong, and every centering constant downstream would
        # be silently wrong with it.
        (state.mu_last[o] > 0.0 && acc < 1.0) ||
            error("set_pairwise_quantile_masses!: decoded masses at origin=$o are off the open simplex " *
                  "(sum of free masses = $acc, remainder = $(state.mu_last[o]))")
    end
    return state
end
