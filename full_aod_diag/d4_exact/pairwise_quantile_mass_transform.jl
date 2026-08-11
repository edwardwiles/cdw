# ================================================================================================
# Simplex (stick-breaking) transform for the pairwise-quantile-independence restriction's FREE BIN
# MASSES -- version B of this restriction (free-mass reparameterization, 2026-08-10).
#
# WHAT CHANGED AND WHY (read this before touching anything downstream). Version A of this
# restriction made the quantile CUTOFFS `q_{o,r}` the free outer coordinates and pinned the moment
# targets at the constants `1/L`, `1/L^2`:
#
#     g^M_{o,a}(z) = 1{b_o(z)=a} - 1/L                 g^P_{op,ab}(z) = 1{...} - 1/L^2
#
# Cutoffs live INSIDE indicator functions, so `Delta*` was a genuine STEP function of every outer
# coordinate: its exact derivative is zero between draw crossings and undefined at them, and the
# outer gradient had to be a bandwidth-selected secant that could only ever be validated to ~20%.
#
# Version B fixes the cutoffs once per campaign and makes the bin MASSES `mu_{o,a}` free:
#
#     g^M_{o,a}(z) = 1{b_o(z)=a} - mu_{o,a}           g^P_{op,ab}(z) = 1{...} - mu_{o,a}*mu_{p,b}
#
# The outer parameters now enter ONLY through a per-row constant shift, so `Delta*` is smooth in
# them and the outer gradient is an exact closed-form envelope derivative
# (`pairwise_quantile_mass_gradient.jl`) -- structurally the SAME formula as origin-ZC's
# production-validated `d_delta_dual_d_eta_origin_vec`. Outer coordinate count is unchanged,
# `(L-1)*D` either way. See docs/PAIRWISE_QUANTILE_FREE_MASS_REPARAMETERIZATION_HANDOVER_2026-08-10.md.
#
# THE PARAMETERIZATION (choice #2 of the two the handover requires be made explicit).
# `mu_{o,\cdot}` must live on the open simplex (`mu_{o,a} > 0`, `sum_{a<L} mu_{o,a} < 1`, bin `L`
# taking the remainder) -- a plain KNITRO box on `mu` itself cannot express that. This file uses
# STICK-BREAKING on unconstrained reals, chosen over the softmax alternative because:
#   - it is the cumulative/one-increment-at-a-time construction the math note
#     (PAIRWISE_QUANTILE_INDEPENDENCE_MATH_NOTE_2026-08-09.md Sections 2-3) is ALREADY written in
#     (`P(z_o<q_r)=p_r`, `F(r,s)=p_r p_s`), so those equivalence proofs carry over verbatim with
#     `p_r` free instead of `r/L`;
#   - it is the direct structural analog of the ordered-cutoff transform it replaces (successive
#     increments of a monotone cumulative, one raw real per increment, lower-triangular Jacobian);
#   - it has NO redundant dimension: `(L-1)` raw reals per origin map bijectively onto the
#     `(L-1)`-dimensional simplex interior, unlike softmax over `L` reals (whose Jacobian is rank
#     deficient by construction and needs a gauge fixed by hand).
#
# Per origin `o`, `L-1` raw KNITRO reals `s_{o,1},...,s_{o,L-1}` decode as
#
#     v_{o,a} = logistic(s_{o,a}) in (0,1)                  (conditional prob. of bin a | bin >= a)
#     R_{o,0} = 1,  R_{o,a} = R_{o,a-1} * (1 - v_{o,a})     (mass still unallocated after bin a)
#     mu_{o,a} = v_{o,a} * R_{o,a-1}       a = 1..L-1
#     mu_{o,L} = R_{o,L-1}                                  (the implicit remainder bin, never free)
#
# so `mu_{o,a} > 0` and `sum_{a=1}^{L-1} mu_{o,a} = 1 - R_{o,L-1} < 1` hold for EVERY real vector,
# by construction, never by a KNITRO constraint -- exactly the discipline the softplus-ordered
# cutoff transform used for the ordering it enforced.
#
# Cumulative form (the one the math note speaks in): `P_{o,a} = sum_{j<=a} mu_{o,j} = 1 - R_{o,a}`,
# strictly increasing in `a`, `0 < P_{o,1} < ... < P_{o,L-1} < 1`.
#
# Outer coordinate layout: ORIGIN-MAJOR, `raw_index(layout,o,k) = (o-1)*(L-1) + k` -- carried over
# unchanged from the cutoff transform, and for the same reason: the Jacobian below is exactly
# block-diagonal across origins under this layout, so each origin's own `(L-1)x(L-1)` block is
# contiguous and a caller mapping a mass-gradient back to raw coordinates for origin `o` never
# touches another origin's coordinates.
# ================================================================================================

"Numerically stable logistic sigmoid 1/(1+exp(-x)), computed without overflow for large |x|."
pq_logistic(x::Float64) = x >= 0.0 ? 1.0 / (1.0 + exp(-x)) : (e = exp(x); e / (1.0 + e))

"Inverse of `pq_logistic`: log(v/(1-v)), defined for v in (0,1)."
function pq_logit(v::Float64)
    (0.0 < v < 1.0) || error("pq_logit: v must be in (0,1), got $v")
    return log(v) - log1p(-v)
end

"""
    PairwiseQuantileMassLayout(D::Int, L::Int)

Layout descriptor: `D` origins, `L` quantile bins per origin (`L>=2`, so
`n_free_bins(layout) = L-1` free mass coordinates per origin -- bin `L` is the simplex remainder
and is never a free coordinate). `n_raw(layout) = (L-1)*D`, origin-major.

`L` has NO default (repo's own no-silent-defaults rule, CLAUDE.md) -- callers must choose it
explicitly, alongside this restriction's other genuine modelling choices (`cutoff_source`,
`min_bin_count`, `W`, `draw_seed`, ...).
"""
struct PairwiseQuantileMassLayout
    D::Int
    L::Int
    function PairwiseQuantileMassLayout(D::Int, L::Int)
        D >= 2 || error("PairwiseQuantileMassLayout: D must be >= 2, got $D")
        L >= 2 || error("PairwiseQuantileMassLayout: L (n_bins) must be >= 2, got $L")
        return new(D, L)
    end
end

n_free_bins(layout::PairwiseQuantileMassLayout) = layout.L - 1
n_raw(layout::PairwiseQuantileMassLayout) = n_free_bins(layout) * layout.D
raw_index(layout::PairwiseQuantileMassLayout, o::Int, k::Int) = (o - 1) * n_free_bins(layout) + k

"""
    decode_origin_masses!(mu_row::AbstractVector{Float64}, raw::AbstractVector{Float64}) -> Float64

Stick-breaking decode for ONE origin: `raw` (length `n = L-1`) -> `mu_row` (length `n`, the FREE
bin masses `mu_{o,1..L-1}`), returning the remainder `mu_{o,L} = 1 - sum(mu_row) > 0`. Writes into
the caller-supplied `mu_row` -- no allocation.
"""
function decode_origin_masses!(mu_row::AbstractVector{Float64}, raw::AbstractVector{Float64})
    n = length(raw)
    length(mu_row) == n || error("decode_origin_masses!: length(mu_row)=$(length(mu_row)) != length(raw)=$n")
    n >= 1 || error("decode_origin_masses!: need at least 1 raw coordinate, got $n")
    rem = 1.0
    @inbounds for a in 1:n
        v = pq_logistic(raw[a])
        mu_row[a] = v * rem
        rem *= (1.0 - v)
    end
    return rem
end

"""
    decode_all_masses!(MU::AbstractMatrix{Float64}, mu_last::AbstractVector{Float64},
                       raw::AbstractVector{Float64}, layout::PairwiseQuantileMassLayout) -> MU

Fills `MU` (`D x (L-1)`, `MU[o,a] = mu_{o,a}`) and `mu_last` (length `D`, `mu_last[o] = mu_{o,L}`)
from the length-`n_raw(layout)` raw outer vector.

`MU` is `D x (L-1)` (origin-first) deliberately: that is the SAME index order the restriction's
marginal dual block `lambda_M[o,a]` uses (`pairwise_quantile_production.jl::dual_index!`), so every
centering site downstream reads `MU[o,a]` next to `lambda_M[o,a]` with no transpose in between.

Called ONCE per outer point, never inside an FG/Hessian callback -- the same lifecycle position
`refresh_pairwise_quantile_bins!` occupied in version A.
"""
function decode_all_masses!(MU::AbstractMatrix{Float64}, mu_last::AbstractVector{Float64},
                            raw::AbstractVector{Float64}, layout::PairwiseQuantileMassLayout)
    D = layout.D; nb = n_free_bins(layout)
    size(MU) == (D, nb) || error("decode_all_masses!: size(MU)=$(size(MU)) != ($D, $nb)")
    length(mu_last) == D || error("decode_all_masses!: length(mu_last)=$(length(mu_last)) != D=$D")
    length(raw) == n_raw(layout) ||
        error("decode_all_masses!: length(raw)=$(length(raw)) != n_raw(layout)=$(n_raw(layout))")
    buf = Vector{Float64}(undef, nb)
    @inbounds for o in 1:D
        base = raw_index(layout, o, 1)
        mu_last[o] = decode_origin_masses!(buf, @view raw[base:base+nb-1])
        for a in 1:nb
            MU[o, a] = buf[a]
        end
    end
    return MU
end

"""
    raw_from_origin_masses!(raw::AbstractVector{Float64}, mu_row::AbstractVector{Float64}) -> raw

Exact inverse of `decode_origin_masses!` for ONE origin: given free masses `mu_row` with
`mu_row .> 0` and `sum(mu_row) < 1`, returns the raw reals that decode to them.
`v_a = mu_a / R_{a-1}`, `raw_a = logit(v_a)`.

Used to construct data-derived or theory-derived STARTING points (`uniform_mass_raw`,
`empirical_mass_raw`) in mass space and then hand KNITRO the corresponding raw coordinates --
rather than hand-deriving raw values, which is exactly the sort of place an off-by-one hides.
"""
function raw_from_origin_masses!(raw::AbstractVector{Float64}, mu_row::AbstractVector{Float64})
    n = length(mu_row)
    length(raw) == n || error("raw_from_origin_masses!: length(raw)=$(length(raw)) != length(mu_row)=$n")
    rem = 1.0
    @inbounds for a in 1:n
        mu_row[a] > 0.0 ||
            error("raw_from_origin_masses!: mass $a is non-positive ($(mu_row[a])) -- not on the open simplex")
        v = mu_row[a] / rem
        v < 1.0 ||
            error("raw_from_origin_masses!: masses 1..$a already exhaust the simplex (v=$v >= 1) -- " *
                  "sum(mu_row)=$(sum(mu_row)) must be < 1 strictly (bin L takes the remainder)")
        raw[a] = pq_logit(v)
        rem -= mu_row[a]
    end
    return raw
end

"""
    uniform_mass_raw(layout::PairwiseQuantileMassLayout) -> Vector{Float64}

The canonical starting point `mu_{o,a} = 1/L` for every origin and free bin -- i.e. equal bin
masses, which is EXACTLY version A's fixed target. Combined with fixed cutoffs set to the draws'
own empirical quantiles, this reproduces version A's moment matrix identically; that is the
version-A/version-B equivalence anchor (handover section 4.3), and the reason this function is
named for the mass value rather than for "the start".

Built through `raw_from_origin_masses!` (the general inverse) and cross-checked against the closed
form `raw_{o,a} = -log(L-a)` -- two independent routes to the same vector, so a mistake in either
shows up here rather than silently downstream.
"""
function uniform_mass_raw(layout::PairwiseQuantileMassLayout)
    D = layout.D; L = layout.L; nb = n_free_bins(layout)
    raw = zeros(n_raw(layout))
    mu_row = fill(1.0 / L, nb)
    buf = Vector{Float64}(undef, nb)
    raw_from_origin_masses!(buf, mu_row)
    for a in 1:nb
        closed = -log(float(L - a))
        isapprox(buf[a], closed; atol = 1e-12, rtol = 1e-12) ||
            error("uniform_mass_raw: stick-breaking inverse ($(buf[a])) disagrees with the closed form " *
                  "-log(L-a)=$closed at a=$a, L=$L -- one of the two is wrong.")
    end
    for o in 1:D
        base = raw_index(layout, o, 1)
        raw[base:base+nb-1] .= buf
    end
    return raw
end

"""
    mass_jacobian_block!(J, raw::AbstractVector{Float64}, mu_row::AbstractVector{Float64}) -> J

Fills the `(L-1)x(L-1)` Jacobian `J[a,k] = d mu_{o,a} / d raw_{o,k}` for ONE origin (`raw` = that
origin's own raw coordinates, `mu_row` = the masses they decode to, via `decode_origin_masses!` on
the SAME `raw`). Block-diagonal across origins by construction -- this function only ever touches
one origin's own block, so the full outer Jacobian is `D` independent blocks, never materialized.

Derivation (`v_k = logistic(raw_k)`, `R_{a-1} = mu_a / v_a`, `dv_k/draw_k = v_k(1-v_k)`):
    k > a :  0                        (bin `a`'s mass does not depend on later stick breaks)
    k = a :  R_{a-1} * v_a(1-v_a) = mu_a * (1 - v_a)
    k < a :  d mu_a/d v_k = -mu_a/(1-v_k), times v_k(1-v_k)  =>  -mu_a * v_k
so `J` is LOWER-triangular, and every entry is available from `(mu_row, v)` alone -- no extra state.

Gated against finite differences of `decode_origin_masses!` itself
(`test_pairwise_quantile_d4_dense_oracle.jl` CHECK 1), mirroring the FD test the ordered-cutoff
transform's own `cutoff_jacobian_block!` carried.
"""
function mass_jacobian_block!(J::AbstractMatrix{Float64}, raw::AbstractVector{Float64},
                              mu_row::AbstractVector{Float64})
    nb = length(raw)
    size(J) == (nb, nb) || error("mass_jacobian_block!: size(J)=$(size(J)) != ($nb,$nb)")
    length(mu_row) == nb || error("mass_jacobian_block!: length(mu_row)=$(length(mu_row)) != $nb")
    fill!(J, 0.0)
    @inbounds for a in 1:nb
        mua = mu_row[a]
        for k in 1:a-1
            J[a, k] = -mua * pq_logistic(raw[k])
        end
        J[a, a] = mua * (1.0 - pq_logistic(raw[a]))
    end
    return J
end

"""
    empirical_mass_raw(bin_counts::AbstractMatrix{<:Real}, layout) -> Vector{Float64}

Raw coordinates placing every origin's masses at the UNWEIGHTED draws' own bin frequencies under
the (fixed) cutoffs: `mu_{o,a} = n_{o,a} / W`, where `n_{o,a} = #{w : b_o(w) = a}`
(`bin_counts[o,a]`, all `L` bins, as produced by `pairwise_quantile_bin_counts`).

This is the version-B analog of "start the cutoffs at the empirical quantiles": a data-derived
starting point at which the restriction's marginal moments are satisfied by the unweighted draws
exactly, so the restriction begins as a pure DEPENDENCE restriction with no marginal slack to
absorb. Under `cutoff_source=:empirical_quantile` it coincides with `uniform_mass_raw` up to the
rounding of `W/L` to whole draws; under `cutoff_source=:frechet_theoretical` it does not, and the
difference is exactly the Monte Carlo error in the draws' own marginals.
"""
function empirical_mass_raw(bin_counts::AbstractMatrix{<:Real}, layout::PairwiseQuantileMassLayout)
    D = layout.D; L = layout.L; nb = n_free_bins(layout)
    size(bin_counts) == (D, L) ||
        error("empirical_mass_raw: size(bin_counts)=$(size(bin_counts)) != ($D,$L)")
    raw = zeros(n_raw(layout))
    buf = Vector{Float64}(undef, nb)
    mu_row = Vector{Float64}(undef, nb)
    for o in 1:D
        tot = sum(@view bin_counts[o, :])
        tot > 0 || error("empirical_mass_raw: origin $o has no draws at all (total count $tot)")
        for a in 1:nb
            mu_row[a] = bin_counts[o, a] / tot
        end
        raw_from_origin_masses!(buf, mu_row)
        base = raw_index(layout, o, 1)
        raw[base:base+nb-1] .= buf
    end
    return raw
end

"""
    default_raw_mass_bounds(bin_counts::AbstractMatrix{<:Real}, layout) -> Vector{NTuple{2,Float64}}

DATA-DERIVED per-raw-coordinate KNITRO box (length `n_raw(layout)`), the version-B replacement for
`default_raw_cutoff_bounds`, and built on the same discipline that function inherited from
`cm_originzc_config.jl::originzc_default_nu_bounds`: centre the box on what the draws themselves
say, widen it by an explicit documented margin, never write a hand-tuned magic level.

The raw coordinate `s_{o,a}` is `logit` of the CONDITIONAL bin probability
`v_{o,a} = mu_{o,a} / (1 - sum_{j<a} mu_{o,j})` (see this file's header), so the natural centre is
the draws' own conditional frequency
`vhat_{o,a} = n_{o,a} / (W - sum_{j<a} n_{o,j})` and the natural width is a margin in LOG-ODDS:

    s_{o,a} in [ logit(vhat_{o,a}) - log(ODDS_MARGIN) , logit(vhat_{o,a}) + log(ODDS_MARGIN) ]

with `ODDS_MARGIN = 16.0`, i.e. any conditional bin probability within a 16x odds ratio of the
data's own is reachable. That is deliberately much wider than any plausible optimum -- the point of
the box is only that KNITRO has one, not to encode a belief about where `mu*` lies -- and it is
symmetric in log-odds, which is the coordinate KNITRO actually moves.

The box is then CLAMPED so that no coordinate can request a conditional probability the `W` draws
cannot resolve at all: `v` is kept inside `[1/W, 1 - 1/W]`. That second bound is what stops a wide
margin from turning into an effectively infinite one for the near-empty conditional bins that occur
at large `L`; it is derived from `W`, not chosen.

Non-degeneracy of the bins themselves is a SEPARATE, harder gate applied at context-build time
(`assert_pairwise_quantile_bins_nondegenerate`), not something this box is asked to enforce.
"""
function default_raw_mass_bounds(bin_counts::AbstractMatrix{<:Real}, layout::PairwiseQuantileMassLayout)
    D = layout.D; L = layout.L; nb = n_free_bins(layout)
    size(bin_counts) == (D, L) ||
        error("default_raw_mass_bounds: size(bin_counts)=$(size(bin_counts)) != ($D,$L)")
    ODDS_MARGIN = 16.0
    bounds = Vector{NTuple{2,Float64}}(undef, n_raw(layout))
    for o in 1:D
        W_o = sum(@view bin_counts[o, :])
        W_o > 0 || error("default_raw_mass_bounds: origin $o has no draws at all")
        v_floor = 1.0 / W_o
        v_ceil = 1.0 - 1.0 / W_o
        v_floor < v_ceil ||
            error("default_raw_mass_bounds: W=$W_o is too small to resolve any conditional bin probability")
        remaining = float(W_o)
        for a in 1:nb
            remaining > 0 ||
                error("default_raw_mass_bounds: origin $o has no draws left at or above bin $a -- " *
                      "the fixed cutoffs leave bin $a and everything after it empty")
            vhat = clamp(bin_counts[o, a] / remaining, v_floor, v_ceil)
            c = pq_logit(vhat)
            lo = max(c - log(ODDS_MARGIN), pq_logit(v_floor))
            hi = min(c + log(ODDS_MARGIN), pq_logit(v_ceil))
            lo < hi ||
                error("default_raw_mass_bounds: degenerate box at origin=$o bin=$a (lo=$lo hi=$hi, " *
                      "vhat=$vhat, W=$W_o)")
            bounds[raw_index(layout, o, a)] = (lo, hi)
            remaining -= bin_counts[o, a]
        end
    end
    return bounds
end

