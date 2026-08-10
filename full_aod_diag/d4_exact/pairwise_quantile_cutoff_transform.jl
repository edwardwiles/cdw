# ================================================================================================
# Ordered-cutoff transform for the pairwise-quantile-independence restriction (draft eq. 32),
# prototype/pairwise-quantile-independence-2026-08-09.
#
# Confirmed by investigation (see docs/PAIRWISE_QUANTILE_INDEPENDENCE_MATH_NOTE_2026-08-09.md and
# the plan this branch was built from): no ordered/monotone-threshold transform exists anywhere in
# this codebase to reuse -- CM's own bin cutpoints (`common_marginals_interval.jl`,
# `cm_hessian_architectures.jl::compute_bin_indices`) are data-derived and never optimized by
# KNITRO; ZC's `nu`/`eta` outer coordinates are independent boxed reals with no ordering
# constraint between them. This file is therefore genuinely new machinery, per the task's own
# "otherwise use a smooth monotone transform in log-z space" fallback instruction.
#
# Per origin `o`, `L-1` raw KNITRO outer reals (c_{o,1}, delta_{o,2}, ..., delta_{o,L-1}) decode
# to `L-1` strictly-ordered physical cutoffs q_{o,1}<...<q_{o,L-1} (same units as ctx.U, the raw
# positive productivity draws) -- `L` (>=2) is the number of quantile BINS, an `L`-generic
# generalization of the task's own draft (which used `L=5`, quintiles) added 2026-08-09:
#
#     logq_{o,1} = c_{o,1}
#     logq_{o,r} = logq_{o,r-1} + softplus(delta_{o,r})            r = 2,...,L-1
#     q_{o,r}    = exp(logq_{o,r})
#
# softplus(x) > 0 for every real x, so each step strictly increases logq -- ordering is guaranteed
# by construction, never by a KNITRO inequality constraint. (L-1) raw reals/origin * D origins =
# n_raw outer coordinates (80 at the task's own D=20, L=5).
#
# Outer coordinate layout: ORIGIN-MAJOR, i.e. raw_index(o,k) = (o-1)*(L-1) + k for k=1:(L-1) --
# chosen (not the level-major/origin-minor convention `cm_originzc_target_layout.jl::
# OriginByPowerLayout` uses) because the cutoff Jacobian below is exactly block-diagonal across
# origins under this layout: each origin's own (L-1)x(L-1) Jacobian block is contiguous, and a
# caller mapping a physical cutoff-gradient back to raw outer coordinates for origin `o` only ever
# needs that one block, never touches another origin's raw coordinates.
# ================================================================================================

"Numerically stable softplus: log(1+exp(x)), computed without overflow for large |x|."
softplus(x::Float64) = x > 0.0 ? x + log1p(exp(-x)) : log1p(exp(x))

"Derivative of softplus = the logistic sigmoid, 1/(1+exp(-x)) -- stable for every real x."
dsoftplus(x::Float64) = 1.0 / (1.0 + exp(-x))

"""
    PairwiseQuantileCutoffLayout(D::Int, L::Int)

Layout descriptor: `D` origins, `L` quantile bins per origin (`L>=2`, so `n_cutoffs(layout)=L-1`
raw coordinates per origin), origin-major (`raw_index(layout,o,k) = (o-1)*(L-1)+k`).
`n_raw(layout) = (L-1)*D`. `L` has NO default (repo's own no-silent-defaults convention) --
callers must choose it explicitly, matching every other genuine modeling choice in this
restriction (`min_crossed`, `W`, `draw_seed`, ...).
"""
struct PairwiseQuantileCutoffLayout
    D::Int
    L::Int
    function PairwiseQuantileCutoffLayout(D::Int, L::Int)
        D >= 2 || error("PairwiseQuantileCutoffLayout: D must be >= 2, got $D")
        L >= 2 || error("PairwiseQuantileCutoffLayout: L (n_bins) must be >= 2, got $L")
        return new(D, L)
    end
end

n_cutoffs(layout::PairwiseQuantileCutoffLayout) = layout.L - 1
n_raw(layout::PairwiseQuantileCutoffLayout) = n_cutoffs(layout) * layout.D
raw_index(layout::PairwiseQuantileCutoffLayout, o::Int, k::Int) = (o - 1) * n_cutoffs(layout) + k

"""
    decode_origin_logcutoffs!(logq::AbstractVector{Float64}, raw::AbstractVector{Float64}) -> logq

`raw = (c, delta_2, ..., delta_{n})` for ONE origin (`n = length(raw) = length(logq)` raw
coordinates, `n = L-1`) -> `logq`, strictly increasing by construction (each step adds
`softplus(delta_r) > 0`). Writes into caller-supplied `logq` -- no allocation.
"""
function decode_origin_logcutoffs!(logq::AbstractVector{Float64}, raw::AbstractVector{Float64})
    n = length(raw)
    length(logq) == n || error("decode_origin_logcutoffs!: length(logq)=$(length(logq)) != length(raw)=$n")
    n >= 1 || error("decode_origin_logcutoffs!: need at least 1 raw coordinate, got $n")
    @inbounds begin
        logq[1] = raw[1]
        for r in 2:n
            logq[r] = logq[r-1] + softplus(raw[r])
        end
    end
    return logq
end

"""
    decode_all_cutoffs!(Q::Matrix{Float64}, raw::AbstractVector{Float64}, layout::PairwiseQuantileCutoffLayout) -> Q

Fills `Q` (`(L-1) x D`, `Q[r,o] = q_{o,r}`, physical/z-space cutoffs) from the length-`n_raw(layout)`
raw outer vector. Called once per outer point (Section 3 of the implementation plan) -- never
inside an FG/Hessian callback.
"""
function decode_all_cutoffs!(Q::Matrix{Float64}, raw::AbstractVector{Float64}, layout::PairwiseQuantileCutoffLayout)
    D = layout.D; nc = n_cutoffs(layout)
    size(Q) == (nc, D) || error("decode_all_cutoffs!: size(Q)=$(size(Q)) != ($nc, $D)")
    length(raw) == n_raw(layout) || error("decode_all_cutoffs!: length(raw)=$(length(raw)) != n_raw(layout)=$(n_raw(layout))")
    @inbounds for o in 1:D
        base = raw_index(layout, o, 1)
        rawo = @view raw[base:base+nc-1]
        Qcol = @view Q[:, o]
        decode_origin_logcutoffs!(Qcol, rawo)
        for r in 1:nc
            Q[r, o] = exp(Qcol[r])
        end
    end
    return Q
end

"""
    cutoff_jacobian_block!(J::Matrix{Float64}, raw::AbstractVector{Float64}, Qcol::AbstractVector{Float64}) -> J

Fills the `(L-1)x(L-1)` Jacobian `J[r,k] = d q_{o,r} / d raw_{o,k}` for ONE origin (`raw` = that
origin's own `n_cutoffs(layout)` raw coordinates, `Qcol = Q[:,o]`, already decoded via
`decode_all_cutoffs!`/`decode_origin_logcutoffs!` for the SAME `raw`). Block-diagonal across
origins by construction (this function only ever touches one origin's own block) -- different
origins never interact, so the full outer-coordinate Jacobian is `D` independent `(L-1)x(L-1)`
blocks, never materialized densely.

Derivation: `logq_r = c + sum_{k=2}^{r} softplus(delta_k)`, so
`d logq_r / d c = 1` for every r (a shift in `c` moves every downstream cutoff together),
`d logq_r / d delta_k = softplus'(delta_k) = dsoftplus(delta_k)` for `k<=r`, else 0.
Chain rule through `q_r = exp(logq_r)`: `d q_r / d raw_k = q_r * d logq_r / d raw_k`.
"""
function cutoff_jacobian_block!(J::Matrix{Float64}, raw::AbstractVector{Float64}, Qcol::AbstractVector{Float64})
    nc = length(raw)
    size(J) == (nc, nc) || error("cutoff_jacobian_block!: size(J)=$(size(J)) != ($nc,$nc)")
    length(Qcol) == nc || error("cutoff_jacobian_block!: length(Qcol)=$(length(Qcol)) != $nc")
    @inbounds begin
        dlog1 = 1.0
        for r in 1:nc
            qr = Qcol[r]
            J[r, 1] = qr * dlog1
        end
        for k in 2:nc
            dlogk = dsoftplus(raw[k])
            for r in 1:nc
                qr = Qcol[r]
                J[r, k] = k <= r ? qr * dlogk : 0.0
            end
        end
    end
    return J
end

"""
    default_raw_cutoff_bounds(ctx, layout::PairwiseQuantileCutoffLayout) -> Vector{NTuple{2,Float64}}

Deliberately wide, DATA-DERIVED per-raw-coordinate KNITRO box (length `n_raw(layout)`), mirroring
`cm_originzc_config.jl::originzc_default_nu_bounds`'s own "4x safety margin around the empirical
draw range, never a hand-tuned magic constant" discipline (repo convention: no silently-chosen
scientific default -- see CLAUDE.md's no-defaults rule). For origin `o`:
  - `c_{o,1}` (= logq_{o,1}) bounded to `(log(lo_o/4), log(hi_o*4))`, `lo_o=minimum(ctx.U[:,o])`,
    `hi_o=maximum(ctx.U[:,o])` -- identical construction to `originzc_default_nu_bounds`'s own
    `k=1` case, since `c_{o,1}` plays exactly the same role (a single free log-scale coordinate).
  - `delta_{o,2:(L-1)}` (softplus-gap coordinates) bounded to `(-20.0, log(hi_o/lo_o) + log(4.0))` --
    the lower end (`softplus(-20)~2e-9`) allows a cutoff gap small enough to be effectively zero;
    the upper end allows a single gap to span the ENTIRE empirical log-range of the origin's own
    draws plus a 4x safety margin, i.e. wide enough that no genuinely finite optimum is ever
    boxed out, without being unboundedly wide (`Inf` bounds are not a KNITRO box).
"""
function default_raw_cutoff_bounds(ctx, layout::PairwiseQuantileCutoffLayout)
    D = layout.D; nc = n_cutoffs(layout)
    bounds = Vector{NTuple{2,Float64}}(undef, n_raw(layout))
    @inbounds for o in 1:D
        Uo = @view ctx.U[:, o]
        lo_o = minimum(Uo); hi_o = maximum(Uo)
        lo_o > 0 || error("default_raw_cutoff_bounds: non-positive lower bound at origin=$o (lo=$lo_o) -- log(q) undefined")
        c_bounds = (log(lo_o / 4), log(hi_o * 4))
        delta_hi = log(hi_o / lo_o) + log(4.0)
        bounds[raw_index(layout, o, 1)] = c_bounds
        for k in 2:nc
            bounds[raw_index(layout, o, k)] = (-20.0, delta_hi)
        end
    end
    return bounds
end
