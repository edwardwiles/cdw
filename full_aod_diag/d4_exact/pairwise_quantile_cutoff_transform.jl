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
# Per origin `o`, 4 raw KNITRO outer reals (c_{o,1}, delta_{o,2}, delta_{o,3}, delta_{o,4}) decode
# to 4 strictly-ordered physical cutoffs q_{o,1}<q_{o,2}<q_{o,3}<q_{o,4} (same units as ctx.U, the
# raw positive productivity draws):
#
#     logq_{o,1} = c_{o,1}
#     logq_{o,r} = logq_{o,r-1} + softplus(delta_{o,r})            r = 2,3,4
#     q_{o,r}    = exp(logq_{o,r})
#
# softplus(x) > 0 for every real x, so each step strictly increases logq -- ordering is guaranteed
# by construction, never by a KNITRO inequality constraint. 4 raw reals/origin * D origins = 80 raw
# outer coordinates at D=20, matching `outer_cutoff_params=80`.
#
# Outer coordinate layout: ORIGIN-MAJOR, i.e. raw_index(o,k) = (o-1)*4 + k for k=1:4 -- chosen
# (not the level-major/origin-minor convention `cm_originzc_target_layout.jl::OriginByPowerLayout`
# uses) because the cutoff Jacobian below is exactly block-diagonal across origins under this
# layout: each origin's own 4x4 Jacobian block is contiguous, and a caller mapping a physical
# cutoff-gradient back to raw outer coordinates for origin `o` only ever needs that one 4x4 block,
# never touches another origin's raw coordinates.
# ================================================================================================

"Numerically stable softplus: log(1+exp(x)), computed without overflow for large |x|."
softplus(x::Float64) = x > 0.0 ? x + log1p(exp(-x)) : log1p(exp(x))

"Derivative of softplus = the logistic sigmoid, 1/(1+exp(-x)) -- stable for every real x."
dsoftplus(x::Float64) = 1.0 / (1.0 + exp(-x))

"""
    PairwiseQuantileCutoffLayout(D::Int)

Trivial layout descriptor: `D` origins, 4 raw coordinates each, origin-major
(`raw_index(layout,o,k) = (o-1)*4+k`). `n_raw(layout) = 4*D`.
"""
struct PairwiseQuantileCutoffLayout
    D::Int
    function PairwiseQuantileCutoffLayout(D::Int)
        D >= 2 || error("PairwiseQuantileCutoffLayout: D must be >= 2, got $D")
        return new(D)
    end
end

n_raw(layout::PairwiseQuantileCutoffLayout) = 4 * layout.D
raw_index(layout::PairwiseQuantileCutoffLayout, o::Int, k::Int) = (o - 1) * 4 + k

"""
    decode_origin_logcutoffs(raw4::NTuple{4,Float64}) -> NTuple{4,Float64}

`raw4 = (c, delta2, delta3, delta4)` for ONE origin -> `(logq1,logq2,logq3,logq4)`, strictly
increasing by construction (each step adds `softplus(delta_r) > 0`).
"""
function decode_origin_logcutoffs(raw4::NTuple{4,Float64})
    c, d2, d3, d4 = raw4
    logq1 = c
    logq2 = logq1 + softplus(d2)
    logq3 = logq2 + softplus(d3)
    logq4 = logq3 + softplus(d4)
    return (logq1, logq2, logq3, logq4)
end

"""
    decode_all_cutoffs!(Q::Matrix{Float64}, raw::AbstractVector{Float64}, layout::PairwiseQuantileCutoffLayout) -> Q

Fills `Q` (4 x D, `Q[r,o] = q_{o,r}`, physical/z-space cutoffs) from the length-`n_raw(layout)`
raw outer vector. Called once per outer point (Section 3 of the implementation plan) -- never
inside an FG/Hessian callback.
"""
function decode_all_cutoffs!(Q::Matrix{Float64}, raw::AbstractVector{Float64}, layout::PairwiseQuantileCutoffLayout)
    D = layout.D
    size(Q) == (4, D) || error("decode_all_cutoffs!: size(Q)=$(size(Q)) != (4, $D)")
    length(raw) == n_raw(layout) || error("decode_all_cutoffs!: length(raw)=$(length(raw)) != n_raw(layout)=$(n_raw(layout))")
    @inbounds for o in 1:D
        base = raw_index(layout, o, 1)
        raw4 = (raw[base], raw[base+1], raw[base+2], raw[base+3])
        logq1, logq2, logq3, logq4 = decode_origin_logcutoffs(raw4)
        Q[1, o] = exp(logq1)
        Q[2, o] = exp(logq2)
        Q[3, o] = exp(logq3)
        Q[4, o] = exp(logq4)
    end
    return Q
end

"""
    cutoff_jacobian_block!(J::Matrix{Float64}, raw4::NTuple{4,Float64}, Qcol::AbstractVector{Float64}) -> J

Fills the 4x4 Jacobian `J[r,k] = d q_{o,r} / d raw_{o,k}` for ONE origin (`Qcol = Q[:,o]`, already
decoded via `decode_all_cutoffs!`/`decode_origin_logcutoffs` for the SAME `raw4`). Block-diagonal
across origins by construction (this function only ever touches one origin's own 4x4 block) --
`cutoff_jacobian_block!` for different origins never interact, so the full 80x80 outer-coordinate
Jacobian is 20 independent 4x4 blocks (320 nonzero entries), never materialized densely.

Derivation: `logq_r = c + sum_{k=2}^{r} softplus(delta_k)`, so
`d logq_r / d c = 1` for every r (a shift in `c` moves every downstream cutoff together),
`d logq_r / d delta_k = softplus'(delta_k) = dsoftplus(delta_k)` for `k<=r`, else 0.
Chain rule through `q_r = exp(logq_r)`: `d q_r / d raw_k = q_r * d logq_r / d raw_k`.
"""
function cutoff_jacobian_block!(J::Matrix{Float64}, raw4::NTuple{4,Float64}, Qcol::AbstractVector{Float64})
    size(J) == (4, 4) || error("cutoff_jacobian_block!: size(J)=$(size(J)) != (4,4)")
    length(Qcol) == 4 || error("cutoff_jacobian_block!: length(Qcol)=$(length(Qcol)) != 4")
    _, d2, d3, d4 = raw4
    ds2 = dsoftplus(d2); ds3 = dsoftplus(d3); ds4 = dsoftplus(d4)
    dlog = (1.0, ds2, ds3, ds4)   # dlog[k] = d logq_r/d raw_k, valid for k<=r (else 0 below)
    @inbounds for r in 1:4
        qr = Qcol[r]
        for k in 1:4
            J[r, k] = k <= r ? qr * dlog[k] : 0.0
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
  - `delta_{o,2:4}` (softplus-gap coordinates) bounded to `(-20.0, log(hi_o/lo_o) + log(4.0))` --
    the lower end (`softplus(-20)~2e-9`) allows a cutoff gap small enough to be effectively zero;
    the upper end allows a single gap to span the ENTIRE empirical log-range of the origin's own
    draws plus a 4x safety margin, i.e. wide enough that no genuinely finite optimum is ever
    boxed out, without being unboundedly wide (`Inf` bounds are not a KNITRO box).
"""
function default_raw_cutoff_bounds(ctx, layout::PairwiseQuantileCutoffLayout)
    D = layout.D
    bounds = Vector{NTuple{2,Float64}}(undef, n_raw(layout))
    @inbounds for o in 1:D
        Uo = @view ctx.U[:, o]
        lo_o = minimum(Uo); hi_o = maximum(Uo)
        lo_o > 0 || error("default_raw_cutoff_bounds: non-positive lower bound at origin=$o (lo=$lo_o) -- log(q) undefined")
        c_bounds = (log(lo_o / 4), log(hi_o * 4))
        delta_hi = log(hi_o / lo_o) + log(4.0)
        bounds[raw_index(layout, o, 1)] = c_bounds
        for k in 2:4
            bounds[raw_index(layout, o, k)] = (-20.0, delta_hi)
        end
    end
    return bounds
end
