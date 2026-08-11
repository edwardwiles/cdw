# ================================================================================================
# Forward / transpose operator for the pairwise-quantile-independence restriction (draft eq. 32),
# version B (fixed cutoffs + FREE bin masses, 2026-08-10).
#
# Same ROLE as zc_restriction_operator.jl's restriction_forward!/restriction_transpose! pair (same
# FG-callback convention: forward "ACCUMULATES -(Rλ) into arg0", transpose writes -(1/W)*(raw sum -
# target*S) into caller-owned gradient buffers), but bin-LOOKUP based rather than BLAS-gemv based,
# since this restriction's features are discrete bin/cell indicators, not continuous columns.
#
# VERSION A -> VERSION B, PRECISELY: the moment rows differ from version A ONLY in the constant
# subtracted per row -- `1/L -> mu_{o,a}` for marginal rows, `1/L^2 -> mu_{o,a}*mu_{p,b}` for pair
# rows. The indicator machinery, the bin lookups, the threaded histogram builder and every cost
# bound below are IDENTICAL. Concretely, in this file the change is confined to the `C_lambda`
# expression in `pairwise_quantile_forward!` and the two centering lines in
# `pairwise_quantile_transpose!`. That is the whole edit here.
#
# `L`-generic: every table/dimension below is sized off `op.L` (the number of quantile bins,
# `op.L>=2`), not hardcoded to the draft's `L=5`.
#
# Requires pairwise_quantile_bin_context.jl (PairwiseQuantileOperator, PairwiseQuantileMassState).
# ================================================================================================

"""
    pairwise_quantile_forward!(arg0, lambda_M, lambda_P, op, state) -> arg0

ACCUMULATES `-R_w` into `arg0[w]` for every draw, where
`R_w = sum_o lambda_M[o,b_o(w)] + sum_{(o,p)} lambda_P[b_o(w),b_p(w),pidx] - C_lambda`,
omitted `L`-th-bin dual entries treated as the implicit zero the task specifies (never stored --
`lambda_M`/`lambda_P` only carry bins/cells `1:(L-1)`). `lambda_M` is `D x (L-1)`, `lambda_P` is
`(L-1) x (L-1) x npair` (`op.pairs` ordering).

The per-draw centering constant is now MASS-weighted (version B):

    C_lambda = sum_{o,a} lambda_M[o,a]*mu[o,a]
             + sum_{(o,p)=pairs[pidx]} sum_{a,b} lambda_P[a,b,pidx]*mu[o,a]*mu[p,b]

(version A had the target-free `sum(lambda_M)/L + sum(lambda_P)/L^2`). It is still a per-draw
CONSTANT -- `mu` does not depend on `w` -- so it is still computed ONCE per call, `O(n_rows)`, and
hoisted out of the `w` loop exactly as before. Cost per draw is unchanged at `O(D+npair)` lookups,
no allocation, no `W x n_rows` object ever constructed.

> SIGN. This function SUBTRACTS into its accumulator (`arg0[w] -= Rw`), matching the economic
> block's own `arg0 .-= econ_buf`, i.e. `r = -zeta - E*lambda_E - G_R*lambda_R`. A sign error
> against this convention was one of two real bugs found on 2026-08-10 and it survived a test that
> passed to 1e-16 (see memory `feedback-self-cancelling-test-convention-cannot-gate-a-sign`).
> Every downstream derivation -- the Hessian centering, the HVP, the closed-form mass gradient's
> `+A_{o,a}` sign -- is stated against this convention.
"""
function pairwise_quantile_forward!(arg0::AbstractVector{Float64}, lambda_M::AbstractMatrix{Float64},
        lambda_P::AbstractArray{Float64,3}, op::PairwiseQuantileOperator, state::PairwiseQuantileMassState)
    D = op.D; npair = op.npair; W = op.W; L = op.L; nc = L - 1
    size(lambda_M) == (D, nc) || error("pairwise_quantile_forward!: size(lambda_M)=$(size(lambda_M)) != ($D,$nc)")
    size(lambda_P) == (nc, nc, npair) || error("pairwise_quantile_forward!: size(lambda_P)=$(size(lambda_P)) != ($nc,$nc,$npair)")
    length(arg0) == W || error("pairwise_quantile_forward!: length(arg0)=$(length(arg0)) != W=$W")
    size(state.mu) == (D, nc) || error("pairwise_quantile_forward!: size(state.mu)=$(size(state.mu)) != ($D,$nc)")

    nlast = UInt8(nc)   # last ACTIVE bin index; bin L (implicit zero) is > nlast
    mu = state.mu
    pairs = op.pairs
    C_lambda = 0.0
    @inbounds for a in 1:nc, o in 1:D
        C_lambda += lambda_M[o, a] * mu[o, a]
    end
    @inbounds for pidx in 1:npair
        (o, p) = pairs[pidx]
        for b in 1:nc
            mub = mu[p, b]
            mub == 0.0 && continue
            acc = 0.0
            for a in 1:nc
                acc += lambda_P[a, b, pidx] * mu[o, a]
            end
            C_lambda += acc * mub
        end
    end

    bin = op.bin
    @inbounds for w in 1:W
        Rw = -C_lambda
        for o in 1:D
            a = bin[w, o]
            a <= nlast && (Rw += lambda_M[o, a])
        end
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
    pairwise_quantile_target_vector!(t::AbstractVector{Float64}, op, state) -> t

Fills `t` (length `n_total_rows(D,L)`) with the per-row centering constant `c_I`:
`mu[o,a]` at `marginal_row(o,a,L)`, `mu[o,a]*mu[p,b]` at `pair_row(D,pidx,a,b,L)` with
`(o,p)=op.pairs[pidx]`.

ONE definition of the version-B target vector, shared by every site that centers this restriction's
raw indicator sums -- the Hessian block (`center_and_scale_pairwise_quantile_hessian!`) and the
economic cross block (`pairwise_quantile_cross_hessian_block!`). Both previously wrote the version-A
constants `1/L`/`1/L^2` inline in several places each; with a mass-dependent, row-dependent target
that duplication is exactly where a silent inconsistency would live, so there is now one function.
`pairwise_quantile_forward!`/`pairwise_quantile_transpose!` deliberately do NOT call it: they need
the same constants but never as a materialized `n_rows` vector, and building one per FG call would
be pointless allocation on the hottest path.

Requires `marginal_row`/`pair_row` (pairwise_quantile_hessian.jl) -- resolved at call time, so the
include order between these two files is not constrained.
"""
function pairwise_quantile_target_vector!(t::AbstractVector{Float64}, op::PairwiseQuantileOperator,
                                          state::PairwiseQuantileMassState)
    D = op.D; npair = op.npair; L = op.L; nc = L - 1
    nrow = n_total_rows(D, L)
    length(t) == nrow || error("pairwise_quantile_target_vector!: length(t)=$(length(t)) != nrow=$nrow")
    size(state.mu) == (D, nc) ||
        error("pairwise_quantile_target_vector!: size(state.mu)=$(size(state.mu)) != ($D,$nc)")
    mu = state.mu
    @inbounds for o in 1:D, a in 1:nc
        t[marginal_row(o, a, L)] = mu[o, a]
    end
    @inbounds for pidx in 1:npair
        (o, p) = op.pairs[pidx]
        for b in 1:nc
            mub = mu[p, b]
            for a in 1:nc
                t[pair_row(D, pidx, a, b, L)] = mu[o, a] * mub
            end
        end
    end
    return t
end

"Per-thread scratch for the 1-way (Mtab, D x L) and 2-way (Ptab, L x L x npair) weighted histogram
tables. Mirrors cm_hessian_threaded.jl::ThreadLocalBinScratch's shape/lifecycle exactly, generalized
from ONE shared-z bin index to per-origin bins (`L` bins/origin instead of shared levels)."
struct PairwiseQuantileThreadScratch
    Mtab::Vector{Matrix{Float64}}    # [tid] -> D x L
    Ptab::Vector{Array{Float64,3}}   # [tid] -> L x L x npair
end

function build_pairwise_quantile_thread_scratch(D::Int, npair::Int, L::Int)
    nt = Threads.nthreads()
    return PairwiseQuantileThreadScratch([zeros(D, L) for _ in 1:nt], [zeros(L, L, npair) for _ in 1:nt])
end

"""
    build_pairwise_quantile_tables_threaded!(Mtab, Ptab, tls, op, weight) -> (Mtab, Ptab)

ONE threaded pass building the weighted 1-way (`Mtab[o,a] = sum_w weight[w]*1{b_o(w)=a}`, ALL `L`
bins) and 2-way (`Ptab[a,b,pidx] = sum_w weight[w]*1{b_o(w)=a,b_p(w)=b}`, `(o,p)=op.pairs[pidx]`,
ALL `LxL` cells) raw histograms, via the SAME static-chunk-then-fixed-order-reduction discipline as
`cm_hessian_threaded.jl::build_bin_tables_threaded!` (no atomics; deterministic reduction order
`1:nt`, not completion order). `weight` is EITHER the transpose's `draw_weights` (`=dPsi!`'s output,
`Ψ'(q_w)`) or the Hessian's `h_w=Ψ''(q_w)` -- ONE builder shared by both `pairwise_quantile_
transpose!` (below) and `pairwise_quantile_hessian.jl`'s `T1`/`T2` raw-table build: one generalized
accumulator, several consumers, not several copies.

TARGET-INDEPENDENT BY CONSTRUCTION, and this is load-bearing for version B: these are histograms of
the RAW indicators, so nothing here depends on `mu` (or, in version A, on `1/L`). The mass targets
enter only where these raw sums are centered -- in `pairwise_quantile_transpose!` below, in
`center_and_scale_pairwise_quantile_hessian!`, and in the cross-Hessian block. Cost:
`O(W*(D+npair))` total.

Reads `op.bin` directly: under version B the bin assignment is a campaign constant owned by the
operator, so this builder no longer takes a mutable state argument at all.
"""
function build_pairwise_quantile_tables_threaded!(Mtab::AbstractMatrix{Float64}, Ptab::AbstractArray{Float64,3},
        tls::PairwiseQuantileThreadScratch, op::PairwiseQuantileOperator, weight::AbstractVector{Float64})
    D = op.D; npair = op.npair; W = op.W; L = op.L
    size(Mtab) == (D, L) || error("build_pairwise_quantile_tables_threaded!: size(Mtab)=$(size(Mtab)) != ($D,$L)")
    size(Ptab) == (L, L, npair) || error("build_pairwise_quantile_tables_threaded!: size(Ptab)=$(size(Ptab)) != ($L,$L,$npair)")
    length(weight) == W || error("build_pairwise_quantile_tables_threaded!: length(weight)=$(length(weight)) != W=$W")
    bin = op.bin
    pairs = op.pairs
    nt = Threads.nthreads()

    for t in 1:nt
        fill!(tls.Mtab[t], 0.0)
        fill!(tls.Ptab[t], 0.0)
    end

    Threads.@threads :static for tid in 1:nt
        lo = 1 + div((tid - 1) * W, nt)
        hi = div(tid * W, nt)
        Mloc = tls.Mtab[tid]; Ploc = tls.Ptab[tid]
        @inbounds for w in lo:hi
            ws = weight[w]
            for o in 1:D
                Mloc[o, bin[w, o]] += ws
            end
            for pidx in 1:npair
                (o, p) = pairs[pidx]
                Ploc[bin[w, o], bin[w, p], pidx] += ws
            end
        end
    end

    fill!(Mtab, 0.0); fill!(Ptab, 0.0)
    for tid in 1:nt   # fixed order 1:nt (not completion order) -> deterministic
        Mtab .+= tls.Mtab[tid]
        Ptab .+= tls.Ptab[tid]
    end
    return Mtab, Ptab
end

"""
    PairwiseQuantileTransposeScratch(D::Int, npair::Int, L::Int)

Persistent (campaign-lifetime, reused every callback -- NOT reallocated per call) raw-table buffers
for `pairwise_quantile_transpose!`. Tiny at D=20/L=5 (`Mraw`: 20x5=100 doubles; `Praw`: 5x5x190=4750
doubles, ~38KB) -- but still allocated ONCE, not per FG callback, per the task's zero-hot-path-
allocation requirement.
"""
mutable struct PairwiseQuantileTransposeScratch
    Mraw::Matrix{Float64}     # D x L
    Praw::Array{Float64,3}    # L x L x npair
end
PairwiseQuantileTransposeScratch(D::Int, npair::Int, L::Int) = PairwiseQuantileTransposeScratch(zeros(D, L), zeros(L, L, npair))

"""
    pairwise_quantile_transpose!(g_M, g_P, draw_weights, op, state, tls, scratch) -> (g_M, g_P)

Writes `g_M[o,a] = -(1/W)*(Mraw[o,a] - S*mu[o,a])` and
`g_P[a,b,pidx] = -(1/W)*(Praw[a,b,pidx] - S*mu[o,a]*mu[p,b])` (`S=sum(draw_weights)`,
`(o,p)=op.pairs[pidx]`) -- the "subtract the targets analytically" step, applied ONCE on the
aggregated sums, never per-draw. This is the version-B form of the same two lines: `G[w,I] =
ind_I(w) - c_I`, so `G'w|_I = (weighted histogram)_I - c_I*sum_w w_w`, and only `c_I` changed
(`1/L -> mu[o,a]`, `1/L^2 -> mu[o,a]*mu[p,b]`).

`Mraw`/`Praw` (full `1:L` tables) come from ONE threaded pass
(`build_pairwise_quantile_tables_threaded!`) over `draw_weights = Ψ'(q_w)` (`dPsi!`'s output --
caller's responsibility, matching `zc_restriction_operator.jl::restriction_transpose!`'s own
convention). `g_M`/`g_P` are the ACTIVE `(L-1)`/`(L-1)x(L-1)` subset only; the omitted `L`-th
bin/cell is never returned as a gradient component, matching `pairwise_quantile_forward!`.
"""
function pairwise_quantile_transpose!(g_M::AbstractMatrix{Float64}, g_P::AbstractArray{Float64,3},
        draw_weights::AbstractVector{Float64}, op::PairwiseQuantileOperator, state::PairwiseQuantileMassState,
        tls::PairwiseQuantileThreadScratch, scratch::PairwiseQuantileTransposeScratch)
    D = op.D; npair = op.npair; W = op.W; L = op.L; nc = L - 1
    size(g_M) == (D, nc) || error("pairwise_quantile_transpose!: size(g_M)=$(size(g_M)) != ($D,$nc)")
    size(g_P) == (nc, nc, npair) || error("pairwise_quantile_transpose!: size(g_P)=$(size(g_P)) != ($nc,$nc,$npair)")
    size(state.mu) == (D, nc) || error("pairwise_quantile_transpose!: size(state.mu)=$(size(state.mu)) != ($D,$nc)")

    build_pairwise_quantile_tables_threaded!(scratch.Mraw, scratch.Praw, tls, op, draw_weights)
    S = sum(draw_weights)
    Mraw = scratch.Mraw; Praw = scratch.Praw
    mu = state.mu
    pairs = op.pairs
    @inbounds for a in 1:nc, o in 1:D
        g_M[o, a] = -(Mraw[o, a] - S * mu[o, a]) / W
    end
    @inbounds for pidx in 1:npair
        (o, p) = pairs[pidx]
        for b in 1:nc
            Smub = S * mu[p, b]
            for a in 1:nc
                g_P[a, b, pidx] = -(Praw[a, b, pidx] - Smub * mu[o, a]) / W
            end
        end
    end
    return g_M, g_P
end
