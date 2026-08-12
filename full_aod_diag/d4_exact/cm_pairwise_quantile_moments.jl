# ================================================================================================
# Moment kernels for the CM + pairwise-quantile family (family #7, 2026-08-12): forward, transpose,
# target vector, and the EXACT closed-form outer gradient w.r.t. the SHARED bin masses.
#
# WHAT IS ACTUALLY NEW HERE, stated precisely so nothing gets re-derived that already exists.
# Against the standalone pairwise-quantile family (pairwise_quantile_operator.jl /
# pairwise_quantile_mass_gradient.jl) exactly two things change:
#
#   1. THE MARGINAL ROWS. `(L-1)*D` per-origin rows `1{b_o=a} - mu_{o,a}` become `L-1` REFERENCE-LEVEL
#      rows `1{b_ref=a} - mu_a`. The other origins' rows are dropped as implied by CM plus these (see
#      cm_pairwise_quantile_config.jl's header for the identity and its precondition). So `lambda_M`
#      (`D x (L-1)`) becomes `lambda_L` (length `L-1`).
#   2. THE TARGET IS A SINGLE SHARED VECTOR. `mu_{o,a} * mu_{p,b}` becomes `mu_a * mu_b` -- both
#      factors drawn from the SAME length-`(L-1)` vector.
#
# Everything else is identical and is REUSED, not re-implemented: the bin lookups, the threaded
# 1-way/2-way histogram builder (`build_pairwise_quantile_tables_threaded!`), the sign convention, the
# per-draw-constant hoisting of the centering term, and the `PairwiseQuantileOperator`/`Q`/`bin`
# machinery. The cost bounds are unchanged: `O(D+npair)` lookups per draw, no allocation, no
# `W x n_rows` object ever built.
#
# > SIGN. `cm_pq_forward!` SUBTRACTS into its accumulator (`arg0[w] -= Rw`), matching the economic
# > block's own `arg0 .-= econ_buf` and CM's own `arg0 .-= cm_contrib`, i.e.
# > `r = -zeta - E*lambda_E - G_R*lambda_R - G_CM*lambda_CM`. Every derivation below is stated against
# > that convention. A sign error against it survived a test that passed to 1e-16 once already in this
# > restriction's history (memory `feedback-self-cancelling-test-convention-cannot-gate-a-sign`).
#
# Requires pairwise_quantile_bin_context.jl (PairwiseQuantileOperator), pairwise_quantile_operator.jl
# (build_pairwise_quantile_tables_threaded!, PairwiseQuantileThreadScratch,
# PairwiseQuantileTransposeScratch), pairwise_quantile_mass_transform.jl (mass_jacobian_block!) and
# cm_pairwise_quantile_config.jl (CMPQMassState, cmpq_level_row, cmpq_pair_row, n_cmpq_*).
# ================================================================================================

"""
    cm_pq_forward!(arg0, lambda_L, lambda_P, op, state, ref) -> arg0

ACCUMULATES `-R_w` into `arg0[w]` for every draw, where

    R_w = lambda_L[b_ref(w)] + sum_{(o,p)=pairs[pidx]} lambda_P[b_o(w),b_p(w),pidx] - C_lambda

with omitted `L`-th-bin/cell dual entries treated as the implicit zero (never stored -- `lambda_L` is
length `L-1`, `lambda_P` is `(L-1) x (L-1) x npair`), and the per-draw-CONSTANT centering term

    C_lambda = sum_a lambda_L[a]*mu_a  +  sum_pidx sum_{a,b} lambda_P[a,b,pidx]*mu_a*mu_b

`mu` does not depend on `w`, so `C_lambda` is computed ONCE per call, `O(n_rows)`, and hoisted out of
the draw loop -- exactly as the standalone family does it.

`ref` is the reference origin (`refIndex1`, CM's own anchor) whose bins the LEVEL rows are stated on.
It is a REQUIRED positional argument with no default: which origin carries the level rows is part of
the restriction's definition, and CM's anchor is the only choice under which the dropped per-origin
rows are implied by the CM rows in the cleanest form.
"""
function cm_pq_forward!(arg0::AbstractVector{Float64}, lambda_L::AbstractVector{Float64},
        lambda_P::AbstractArray{Float64,3}, op::PairwiseQuantileOperator, state::CMPQMassState, ref::Int)
    D = op.D; npair = op.npair; W = op.W; L = op.L; nc = L - 1
    length(lambda_L) == nc || error("cm_pq_forward!: length(lambda_L)=$(length(lambda_L)) != L-1=$nc")
    size(lambda_P) == (nc, nc, npair) ||
        error("cm_pq_forward!: size(lambda_P)=$(size(lambda_P)) != ($nc,$nc,$npair)")
    length(arg0) == W || error("cm_pq_forward!: length(arg0)=$(length(arg0)) != W=$W")
    length(state.mu) == nc || error("cm_pq_forward!: length(state.mu)=$(length(state.mu)) != L-1=$nc")
    1 <= ref <= D || error("cm_pq_forward!: ref=$ref outside 1:D=$D")

    nlast = UInt8(nc)   # last ACTIVE bin index; bin L (implicit zero) is > nlast
    mu = state.mu
    pairs = op.pairs
    C_lambda = 0.0
    @inbounds for a in 1:nc
        C_lambda += lambda_L[a] * mu[a]
    end
    # Pair part: sum_pidx mu' * Lambda_pidx * mu -- a quadratic form in the SINGLE shared mu, which is
    # the whole structural difference from the standalone family's mu[o,:]' * Lambda * mu[p,:].
    @inbounds for pidx in 1:npair
        for b in 1:nc
            mub = mu[b]
            mub == 0.0 && continue
            acc = 0.0
            for a in 1:nc
                acc += lambda_P[a, b, pidx] * mu[a]
            end
            C_lambda += acc * mub
        end
    end

    bin = op.bin
    @inbounds for w in 1:W
        Rw = -C_lambda
        aref = bin[w, ref]
        aref <= nlast && (Rw += lambda_L[aref])
        for pidx in 1:npair
            (o, p) = pairs[pidx]
            a = bin[w, o]; b = bin[w, p]
            (a <= nlast && b <= nlast) && (Rw += lambda_P[a, b, pidx])
        end
        arg0[w] -= Rw
    end
    return arg0
end

"""
    cm_pq_target_vector!(t, op, state) -> t

Fills `t` (length `n_cmpq_restr_rows(D,L)`) with the per-row centering constant `c_I`: `mu_a` at
`cmpq_level_row(a)`, `mu_a*mu_b` at `cmpq_pair_row(D,pidx,a,b,L)`.

ONE definition of the target vector, shared by every site that centers this family's raw indicator
sums (the Hessian block and the economic cross block), for the same reason the standalone family has
one: with a mass-dependent, row-dependent target, duplicated inline constants are exactly where a
silent inconsistency lives.
"""
function cm_pq_target_vector!(t::AbstractVector{Float64}, op::PairwiseQuantileOperator, state::CMPQMassState)
    D = op.D; npair = op.npair; L = op.L; nc = L - 1
    nrow = n_cmpq_restr_rows(D, L)
    length(t) == nrow || error("cm_pq_target_vector!: length(t)=$(length(t)) != nrow=$nrow")
    length(state.mu) == nc || error("cm_pq_target_vector!: length(state.mu)=$(length(state.mu)) != $nc")
    mu = state.mu
    @inbounds for a in 1:nc
        t[cmpq_level_row(a)] = mu[a]
    end
    @inbounds for pidx in 1:npair
        for b in 1:nc
            mub = mu[b]
            for a in 1:nc
                t[cmpq_pair_row(D, pidx, a, b, L)] = mu[a] * mub
            end
        end
    end
    return t
end

"""
    cm_pq_transpose!(g_L, g_P, draw_weights, op, state, ref, tls, scratch) -> (g_L, g_P)

Writes `g_L[a] = -(Mraw[ref,a] - S*mu_a)/W` and
`g_P[a,b,pidx] = -(Praw[a,b,pidx] - S*mu_a*mu_b)/W`, with `S = sum(draw_weights)` -- the "subtract the
targets analytically" step, applied ONCE on the aggregated sums, never per draw. `G[w,I] = ind_I(w) -
c_I` gives `G'w|_I = (weighted histogram)_I - c_I*sum_w w_w`, and only `c_I` differs from the
standalone family.

`Mraw`/`Praw` come from the SAME shared threaded pass the standalone family uses
(`build_pairwise_quantile_tables_threaded!`) over `draw_weights = Psi'(q_w)` (`dPsi!`'s output --
caller's responsibility, matching `zc_restriction_operator.jl::restriction_transpose!`'s convention).

DELIBERATELY REUSED AS-IS, with a known small inefficiency: that builder fills `Mraw` for all `D`
origins while only row `ref` is read here. The marginal pass is `O(W*D)` against the pair pass's
`O(W*npair)`, i.e. ~10% of it at D=20 (20 vs 190 increments per draw), and skipping it would mean
adding a flag to a kernel shared with the standalone family and its Hessian. Left as one accumulator
with several consumers; revisit only if a profile says this 10% matters.
"""
function cm_pq_transpose!(g_L::AbstractVector{Float64}, g_P::AbstractArray{Float64,3},
        draw_weights::AbstractVector{Float64}, op::PairwiseQuantileOperator, state::CMPQMassState,
        ref::Int, tls::PairwiseQuantileThreadScratch, scratch::PairwiseQuantileTransposeScratch)
    D = op.D; npair = op.npair; W = op.W; L = op.L; nc = L - 1
    length(g_L) == nc || error("cm_pq_transpose!: length(g_L)=$(length(g_L)) != L-1=$nc")
    size(g_P) == (nc, nc, npair) || error("cm_pq_transpose!: size(g_P)=$(size(g_P)) != ($nc,$nc,$npair)")
    length(state.mu) == nc || error("cm_pq_transpose!: length(state.mu)=$(length(state.mu)) != $nc")
    1 <= ref <= D || error("cm_pq_transpose!: ref=$ref outside 1:D=$D")

    build_pairwise_quantile_tables_threaded!(scratch.Mraw, scratch.Praw, tls, op, draw_weights)
    S = sum(draw_weights)
    Mraw = scratch.Mraw; Praw = scratch.Praw
    mu = state.mu
    @inbounds for a in 1:nc
        g_L[a] = -(Mraw[ref, a] - S * mu[a]) / W
    end
    @inbounds for pidx in 1:npair
        for b in 1:nc
            Smub = S * mu[b]
            for a in 1:nc
                g_P[a, b, pidx] = -(Praw[a, b, pidx] - Smub * mu[a]) / W
            end
        end
    end
    return g_L, g_P
end

"""
    cm_pq_dC_dmu(lambda_L, lambda_P, mu, op) -> Vector{Float64}

`A_c := dC_lambda/dmu_c` for `c = 1..L-1`, the quantity every mass-derivative in this family is built
from:

    A_c = lambda_L[c]  +  sum_pidx [ sum_b lambda_P[c,b,pidx]*mu_b  +  sum_a lambda_P[a,c,pidx]*mu_a ]

⚠️ THIS IS THE TWO-SLOT TERM, and it is the single most error-prone line in the family.
With a SHARED `mu`, the pair target is `mu_a*mu_b` with BOTH factors from the same vector, so the
product rule gives TWO contributions per pair row:

    d(mu_a*mu_b)/d(mu_c) = delta_{ac}*mu_b + delta_{bc}*mu_a

The standalone family's `d_delta_dual_d_mu` already writes exactly this, but into TWO DIFFERENT
origins' slots (`d_mu[o,a] -= mu[p,b]*lp; d_mu[p,b] -= mu[o,a]*lp`), where a transcription slip lands
in another origin's row and is at least structurally visible. Here both slots collapse onto the SAME
length-`(L-1)` vector, so:
  - dropping the second term (keeping only the `a`-slot) halves the diagonal contribution and is
    otherwise plausible-looking;
  - writing `mu[c]` in place of `mu[b]`/`mu[a]` (the partner's mass at the PARTNER's own bin index)
    is the same natural error `d_delta_dual_d_mu`'s docstring warns about;
  - and BOTH are numerically INVISIBLE at `mu_a = 1/L`, where every mass is equal.
Hence the gates for this function are run at a deliberately NON-UNIFORM `mu` and carry an explicit
NEGATIVE CONTROL that must FAIL when the partner index is written the natural wrong way
(memory `pairwise-quantile-free-mass-reparam-2026-08-10`, fact 2). This is the shared-parameter
analogue of `cm_meanzc_moments.jl::d_delta_dual_d_nu_vec`'s own `d_pair_dnu` term -- read that one
alongside this.

Note this is the derivative of the FIXED-DUAL objective's per-draw constant, exact at ANY
`(zeta,lambda)`, not only at the optimum; the envelope theorem is what promotes it to the reoptimized
derivative (see `d_delta_dual_d_mu_shared`).
"""
function cm_pq_dC_dmu(lambda_L::AbstractVector{Float64}, lambda_P::AbstractArray{Float64,3},
                       mu::AbstractVector{Float64}, op::PairwiseQuantileOperator)
    npair = op.npair; L = op.L; nc = L - 1
    length(lambda_L) == nc || error("cm_pq_dC_dmu: length(lambda_L)=$(length(lambda_L)) != $nc")
    size(lambda_P) == (nc, nc, npair) || error("cm_pq_dC_dmu: size(lambda_P)=$(size(lambda_P)) != ($nc,$nc,$npair)")
    length(mu) == nc || error("cm_pq_dC_dmu: length(mu)=$(length(mu)) != $nc")
    A = zeros(nc)
    @inbounds for c in 1:nc
        A[c] = lambda_L[c]
    end
    @inbounds for pidx in 1:npair
        for b in 1:nc, a in 1:nc
            lp = lambda_P[a, b, pidx]
            lp == 0.0 && continue
            A[a] += mu[b] * lp     # a-slot: partner's mass at the PARTNER's bin index b
            A[b] += mu[a] * lp     # b-slot: the same row contributes AGAIN, at index b
        end
    end
    return A
end

"""
    d_delta_dual_d_mu_shared(lambda_L, lambda_P, mu, op; mean_m::Float64) -> Vector{Float64}

`d(Delta_dual)/d(mu_c)` for the `L-1` shared free masses.

DERIVATION (four lines, same as the standalone family's -- do it yourself before trusting this).
The inner solve minimizes `f = (1/W) sum_w Psi(r_w) + zeta`, the reported divergence is
`Delta* = -f*`, and `mu` enters `r` ONLY through the per-draw constant `C_lambda`:

    dr_w/dmu_c = +A_c          (draw-INDEPENDENT: `r = ... - (G_R lambda_R)_w` and `cm_pq_forward!`
                                SUBTRACTS, so `-d(-C_lambda) = +A`)
    df/dmu_c   = (1/W) sum_w Psi'(r_w) * A_c = A_c * mean_m
    d(Delta*)/dmu_c = -mean_m * A_c                      (envelope theorem at `(zeta*, lambda*)`)

`mean_m = (1/W) sum_w Psi'(r_w)` is `verify.m_mean`, the same scalar origin-ZC and the standalone PQ
family both pass. No bandwidth, no secant: `mu` shifts a moment TARGET smoothly and never reassigns a
draw between bins, so this is exact (see pairwise_quantile_mass_gradient.jl's header for why version A
of that family needed a secant and version B does not).
"""
function d_delta_dual_d_mu_shared(lambda_L::AbstractVector{Float64}, lambda_P::AbstractArray{Float64,3},
                                   mu::AbstractVector{Float64}, op::PairwiseQuantileOperator;
                                   mean_m::Float64)
    A = cm_pq_dC_dmu(lambda_L, lambda_P, mu, op)
    A .*= -mean_m
    return A
end

"""
    chain_cmpq_mass_gradient_to_raw(d_mu, raw, mu) -> Vector{Float64}

Chain-rules `d(Delta_dual)/d(mu_a)` through the stick-breaking transform to the `L-1` raw KNITRO
coordinates: `g[k] = sum_a d_mu[a] * (d mu_a / d raw_k)`, with the Jacobian from the standalone
family's own `mass_jacobian_block!` (reused unchanged -- it is already a single-simplex function).

ONE block, not `D` of them: that collapse from `D*(L-1)` to `L-1` outer coordinates is the entire
point of this family.
"""
function chain_cmpq_mass_gradient_to_raw(d_mu::AbstractVector{Float64}, raw::AbstractVector{Float64},
                                          mu::AbstractVector{Float64})
    nb = length(d_mu)
    length(raw) == nb || error("chain_cmpq_mass_gradient_to_raw: length(raw)=$(length(raw)) != $nb")
    length(mu) == nb || error("chain_cmpq_mass_gradient_to_raw: length(mu)=$(length(mu)) != $nb")
    J = zeros(nb, nb)
    mass_jacobian_block!(J, raw, mu)
    g = zeros(nb)
    @inbounds for k in 1:nb
        acc = 0.0
        for a in 1:nb
            acc += d_mu[a] * J[a, k]
        end
        g[k] = acc
    end
    return g
end

"""
    reshape_cmpq_duals(x, op, ncore1) -> (lambda_L, lambda_P, lambda_CM)

Views into the inner KNITRO variable vector for this family's blocks, given `ncore1` = the number of
economic (pre-gravity) inner columns.

LAYOUT: `x = [zeta; lambda_E(ncore1); lambda_L(L-1); lambda_P((L-1)^2*npair); lambda_CM(ncm)]`,
i.e. this family's own restriction rows sit BEFORE the CM-grid block -- the SAME ordering CM+ZC uses
(`[E | mean | pair | CM-grid | gravity]`, cm_meanzc_lookup_kernels.jl's header), so the `ncore1`
semantics and the CM offset arithmetic carry over unchanged.

`lambda_P`'s `reshape(v, nc, nc, npair)` matches `cmpq_pair_row`'s `(b-1)*nc+a` ordering directly.
`lambda_L` needs no reshape at all -- and note that this is where the standalone family had its
2026-08-09 bug (`reshape(v,D,nc)` is column-major/a-major while `marginal_row` is o-major, needing
`reshape(v,nc,D)'`): with ONE shared simplex there is no `(o,a)` matrix to transpose, so that entire
class of error is gone by construction rather than by getting the transpose right.
"""
function reshape_cmpq_duals(x::AbstractVector{Float64}, op::PairwiseQuantileOperator, ncore1::Int)
    D = op.D; npair = op.npair; L = op.L; nc = L - 1
    nL = n_cmpq_level_rows(L)
    nP = nc * nc * npair
    off = 1 + ncore1
    lambda_L = @view x[off+1 : off+nL]
    lambda_P = reshape(@view(x[off+nL+1 : off+nL+nP]), nc, nc, npair)
    lambda_CM = @view x[off+nL+nP+1 : end]
    return lambda_L, lambda_P, lambda_CM
end
