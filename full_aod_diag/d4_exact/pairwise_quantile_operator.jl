# ================================================================================================
# Forward / transpose operator for the pairwise-quantile-independence restriction (draft eq. 32).
#
# Same ROLE as zc_restriction_operator.jl's restriction_forward!/restriction_transpose! pair (same
# FG-callback convention: forward "ACCUMULATES -(Rλ) into arg0", transpose writes -(1/W)*(raw sum -
# target*S) into caller-owned gradient buffers), but bin-LOOKUP based rather than BLAS-gemv based,
# since this restriction's features are discrete bin/cell indicators, not continuous columns.
#
# `L`-generic (2026-08-09): every table/dimension below is sized off `op.L` (the number of
# quantile bins, `op.L>=2`), not hardcoded to the task's own draft value `L=5` (quintiles).
#
# Requires pairwise_quantile_bin_context.jl (PairwiseQuantileOperator, PairwiseQuantileBinState) to
# already be included.
# ================================================================================================

"""
    pairwise_quantile_forward!(arg0, lambda_M, lambda_P, op, state) -> arg0

ACCUMULATES `-R_w` into `arg0[w]` for every draw, where
`R_w = sum_o lambda_M[o,bin[w,o]] + sum_{(o,p)} lambda_P[bin[w,o],bin[w,p],pidx] - C_lambda`
(task's own formula), omitted `L`-th-bin dual entries treated as the implicit zero the task
specifies (never stored -- `lambda_M`/`lambda_P` only carry bins/cells `1:(L-1)`). `lambda_M` is
`D x (L-1)`, `lambda_P` is `(L-1) x (L-1) x npair` (`op.pairs` ordering).
`C_lambda = sum(lambda_M)/L + sum(lambda_P)/L^2` is precomputed ONCE per call (`O(n_rows)`, not per
draw). Cost per draw: `O(D+npair)` lookups, matching the task's stated complexity -- no allocation,
no `W x n_rows` object ever constructed.
"""
function pairwise_quantile_forward!(arg0::AbstractVector{Float64}, lambda_M::AbstractMatrix{Float64},
        lambda_P::AbstractArray{Float64,3}, op::PairwiseQuantileOperator, state::PairwiseQuantileBinState)
    D = op.D; npair = op.npair; W = op.W; L = op.L; nc = L - 1
    size(lambda_M) == (D, nc) || error("pairwise_quantile_forward!: size(lambda_M)=$(size(lambda_M)) != ($D,$nc)")
    size(lambda_P) == (nc, nc, npair) || error("pairwise_quantile_forward!: size(lambda_P)=$(size(lambda_P)) != ($nc,$nc,$npair)")
    length(arg0) == W || error("pairwise_quantile_forward!: length(arg0)=$(length(arg0)) != W=$W")

    nlast = UInt8(nc)   # last ACTIVE bin index; bin L (implicit zero) is > nlast
    C_lambda = sum(lambda_M) / L + sum(lambda_P) / L^2
    bin = state.bin
    pairs = op.pairs
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
    build_pairwise_quantile_tables_threaded!(Mtab, Ptab, tls, op, state, weight) -> (Mtab, Ptab)

ONE threaded pass building the weighted 1-way (`Mtab[o,a] = sum_w weight[w]*1{bin[w,o]=a}`, ALL `L`
bins) and 2-way (`Ptab[a,b,pidx] = sum_w weight[w]*1{bin[w,o]=a,bin[w,p]=b}`, `(o,p)=op.pairs[pidx]`,
ALL `LxL` cells) raw histograms, via the SAME static-chunk-then-fixed-order-reduction discipline as
`cm_hessian_threaded.jl::build_bin_tables_threaded!` (no atomics; deterministic reduction order
`1:nt`, not completion order). `weight` is EITHER the transpose's `draw_weights` (`=dPsi!`'s output,
`Ψ'(q_w)`) or the Hessian's `h_w=Ψ''(q_w)` -- ONE builder shared by both `pairwise_quantile_
transpose!` (below) and `pairwise_quantile_hessian.jl`'s `T1`/`T2` raw-table build, per the task's
"reuse current thread-local histogram infrastructure" instruction: one generalized accumulator, two
consumers, not two copies. Cost: `O(W*(D+npair))` total -- same complexity class as the forward
pass's own `O(D+npair)`-per-draw cost, since every bin/cell touched by a draw is touched exactly
once here too (the full `L`-way marginal and `LxL`-way pair tables, not just the active `(L-1)x(L-1)`/
`(L-1)` subset, so the SAME tables also serve as the verifier's "report all bin/cell probabilities"
need, Section 8).
"""
function build_pairwise_quantile_tables_threaded!(Mtab::AbstractMatrix{Float64}, Ptab::AbstractArray{Float64,3},
        tls::PairwiseQuantileThreadScratch, op::PairwiseQuantileOperator,
        state::PairwiseQuantileBinState, weight::AbstractVector{Float64})
    D = op.D; npair = op.npair; W = op.W; L = op.L
    size(Mtab) == (D, L) || error("build_pairwise_quantile_tables_threaded!: size(Mtab)=$(size(Mtab)) != ($D,$L)")
    size(Ptab) == (L, L, npair) || error("build_pairwise_quantile_tables_threaded!: size(Ptab)=$(size(Ptab)) != ($L,$L,$npair)")
    length(weight) == W || error("build_pairwise_quantile_tables_threaded!: length(weight)=$(length(weight)) != W=$W")
    bin = state.bin
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

Writes `g_M[o,a] = -(1/W)*(Mraw[o,a] - S/L)` and `g_P[a,b,pidx] = -(1/W)*(Praw[a,b,pidx] - S/L^2)`
(`S=sum(draw_weights)`) -- the "subtract the 1/L,1/L^2 targets analytically" step (task Section 4),
applied ONCE on the aggregated sums, never per-draw. `Mraw`/`Praw` (full `1:L` tables) come from ONE
threaded pass (`build_pairwise_quantile_tables_threaded!`) over `draw_weights = Ψ'(q_w)` (`dPsi!`'s
output -- caller's responsibility, matching `zc_restriction_operator.jl::restriction_transpose!`'s
own convention). `g_M`/`g_P` are the ACTIVE `(L-1)x(L-1)`/`(L-1)` subset only (columns/cells
`1:(L-1)`) -- the omitted `L`-th bin/cell is never returned as a gradient component, matching
`pairwise_quantile_forward!`.
"""
function pairwise_quantile_transpose!(g_M::AbstractMatrix{Float64}, g_P::AbstractArray{Float64,3},
        draw_weights::AbstractVector{Float64}, op::PairwiseQuantileOperator, state::PairwiseQuantileBinState,
        tls::PairwiseQuantileThreadScratch, scratch::PairwiseQuantileTransposeScratch)
    D = op.D; npair = op.npair; W = op.W; L = op.L; nc = L - 1
    size(g_M) == (D, nc) || error("pairwise_quantile_transpose!: size(g_M)=$(size(g_M)) != ($D,$nc)")
    size(g_P) == (nc, nc, npair) || error("pairwise_quantile_transpose!: size(g_P)=$(size(g_P)) != ($nc,$nc,$npair)")

    build_pairwise_quantile_tables_threaded!(scratch.Mraw, scratch.Praw, tls, op, state, draw_weights)
    S = sum(draw_weights)
    Mraw = scratch.Mraw; Praw = scratch.Praw
    @inbounds for a in 1:nc, o in 1:D
        g_M[o, a] = -(Mraw[o, a] - S / L) / W
    end
    @inbounds for pidx in 1:npair, b in 1:nc, a in 1:nc
        g_P[a, b, pidx] = -(Praw[a, b, pidx] - S / L^2) / W
    end
    return g_M, g_P
end
