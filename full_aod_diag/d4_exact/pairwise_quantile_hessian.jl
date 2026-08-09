# ================================================================================================
# Hessian (H_MM/H_MP/H_PP) for the pairwise-quantile-independence restriction (draft eq. 32).
#
# Implements Section 5 of the implementation plan: every raw (uncentered) Hessian block reduces to
# exactly one of four small table families (T1 1-way, T2 2-way, T3 3-way, T4 4-way), two of which
# (T2, T3) are each consumed by TWO different named blocks -- see the table in the plan / the
# per-block derivations in this file's own comments. Centering uses the task's mandatory identity
#     sum_w h_w(x_w-t)(x_w-t)' = X'WX - r t' - t r' + S t t'
# applied ONCE, densely, over the (small, D+npair*16-sized, NEVER draw-indexed) restriction-only
# Hessian block -- this is the ordinary "small dense Hessian scratch" every other family already
# builds (CM's own Hfull is the same order of magnitude), not the forbidden W x n_rows dense G.
#
# Requires pairwise_quantile_bin_context.jl (PairwiseQuantileOperator, n_marginal_rows/n_pair_rows/
# n_total_rows) and pairwise_quantile_operator.jl (build_pairwise_quantile_tables_threaded!,
# PairwiseQuantileThreadScratch) to already be included -- T1/T2 reuse that SAME threaded builder
# (weighted by h_w=Psi''(q_w) instead of the transpose's Psi'(q_w)), per the task's "reuse current
# thread-local histogram infrastructure" instruction: one generalized accumulator, multiple
# consumers, never a second copy.
# ================================================================================================

"Row index of marginal (origin o, bin a) within the restriction-only Hfull block."
marginal_row(o::Int, a::Int) = (o - 1) * 4 + a

"Row index of pair cell (pidx, bin a of pairs[pidx][1], bin b of pairs[pidx][2]) within the block."
pair_row(D::Int, pidx::Int, a::Int, b::Int) = n_marginal_rows(D) + (pidx - 1) * 16 + (b - 1) * 4 + a

"origin_slot(op,pidx,o): 1 if o==pairs[pidx][1], 2 if o==pairs[pidx][2], else 0 (disjoint)."
function origin_slot(op::PairwiseQuantileOperator, pidx::Int, o::Int)
    (p, q) = op.pairs[pidx]
    o == p && return 1
    o == q && return 2
    return 0
end

"""
    PairwiseQuantileHessianTables(op::PairwiseQuantileOperator)

Persistent (campaign-lifetime shape, rebuilt in VALUE every Hessian callback -- never reallocated)
raw table storage: `T1` (D x 4, `r_M`/H_MM-diagonal), `T2` (4x4xnpair, H_MM-cross / H_MP-same-
origin / H_PP-same-pair-diagonal), `T3` (4x4x4xncombo3, H_MP-disjoint / H_PP-shared-origin), `T4`
(4x4x4x4xncombo4, H_PP-fully-disjoint ONLY). `S = sum(h_w)` (the shared scalar the centering
identity needs). Sizes at D=20: T1 80 doubles, T2 12160, T3 ~219k (`ncombo3=D*(D-2)*... `see
`op.triple_combos`), T4 ~3.7M (`ncombo4=length(op.quad_combos)~14535`) -- a few tens of MB total,
`O(#combos)`, independent of W, NOT the forbidden W x n_rows dense G.
"""
mutable struct PairwiseQuantileHessianTables
    T1::Matrix{Float64}
    T2::Array{Float64,3}
    T3::Array{Float64,4}
    T4::Array{Float64,5}
    S::Float64
    # precomputed flat (origin...) tuples, avoiding repeated op.pairs[] indexing inside the hot
    # per-draw scatter loops below -- derived once from `op` (campaign-lifetime), cached here since
    # this struct is already the natural home for anything Hessian-table-build-specific.
    triple_opq::Vector{NTuple{3,Int}}    # [(o,p,q)] matches op.triple_combos order
    quad_oooo::Vector{NTuple{4,Int}}     # [(o1,o2,o3,o4)] matches op.quad_combos order
end

function PairwiseQuantileHessianTables(op::PairwiseQuantileOperator)
    D = op.D; npair = op.npair
    ncombo3 = length(op.triple_combos)
    ncombo4 = length(op.quad_combos)
    triple_opq = [(o, op.pairs[pidx][1], op.pairs[pidx][2]) for (o, pidx) in op.triple_combos]
    quad_oooo = [(op.pairs[pidx1][1], op.pairs[pidx1][2], op.pairs[pidx2][1], op.pairs[pidx2][2])
                 for (pidx1, pidx2) in op.quad_combos]
    # T1/T2 are sized (D,5)/(5,5,npair) -- matching `build_pairwise_quantile_tables_threaded!`'s
    # full-5-bin output shape (SAME shared builder the transpose uses, see that function's own
    # docstring) -- only the [1:4,...] slice is ever READ by the block-fill/centering code below
    # (bin 5's dual is the implicit zero the task specifies), but the table itself is built over
    # all 5 bins so no second, differently-shaped builder is needed.
    return PairwiseQuantileHessianTables(zeros(D, 5), zeros(5, 5, npair), zeros(4, 4, 4, ncombo3),
        zeros(4, 4, 4, 4, ncombo4), 0.0, triple_opq, quad_oooo)
end

"""
    build_pairwise_quantile_hessian_tables!(tabs, op, state, h, tls) -> tabs

ONE per-Hessian-callback refresh of all four raw table families, weighted by `h = Psi''(q_w)`
(caller-supplied, matching every other family's `obj.ddPsi!(arg2,arg0)` convention). `T1`/`T2` are
built via the SAME threaded static-chunk/fixed-order-reduction builder the transpose uses
(`build_pairwise_quantile_tables_threaded!`), just weighted by `h` instead of `Psi'(q_w)`. `T3`/`T4`
are built via a direct single-threaded scatter pass (baseline, correctness-first per the task's
"profile first, optimize only the measured bottleneck" instruction -- Section 11 of the plan is
where any threading/BLAS-batching decision for these two gets made, informed by real D20 numbers,
not guessed here).
"""
function build_pairwise_quantile_hessian_tables!(tabs::PairwiseQuantileHessianTables, op::PairwiseQuantileOperator,
        state::PairwiseQuantileBinState, h::AbstractVector{Float64}, tls::PairwiseQuantileThreadScratch)
    build_pairwise_quantile_tables_threaded!(tabs.T1, tabs.T2, tls, op, state, h)
    tabs.S = sum(h)

    bin = state.bin
    W = op.W
    T3 = tabs.T3; fill!(T3, 0.0)
    triples = tabs.triple_opq
    @inbounds for w in 1:W
        hw = h[w]
        for k in 1:length(triples)
            (o, p, q) = triples[k]
            a = bin[w, o]
            a > 0x04 && continue
            b = bin[w, p]
            b > 0x04 && continue
            c = bin[w, q]
            c > 0x04 && continue
            T3[a, b, c, k] += hw
        end
    end

    T4 = tabs.T4; fill!(T4, 0.0)
    quads = tabs.quad_oooo
    @inbounds for w in 1:W
        hw = h[w]
        for k in 1:length(quads)
            (o1, o2, o3, o4) = quads[k]
            a = bin[w, o1]
            a > 0x04 && continue
            b = bin[w, o2]
            b > 0x04 && continue
            c = bin[w, o3]
            c > 0x04 && continue
            d = bin[w, o4]
            d > 0x04 && continue
            T4[a, b, c, d, k] += hw
        end
    end
    return tabs
end

"""
    fill_pairwise_quantile_hessian_raw!(HfullR, op, tabs) -> HfullR

Fills the DENSE, symmetric, UNCENTERED (`n_rows x n_rows`) restriction-only Hessian block from
`tabs`'s raw tables, exploiting the raw structure exactly as the task specifies:
  - H_MM same-origin: diagonal, from `T1`.
  - H_MM cross-origin / H_PP same-pair diagonal / H_MP same-origin: all read `T2` (three different
    slicings of the SAME 2-way table -- no block here needs its own separate table).
  - H_MP disjoint-origin / H_PP shared-one-origin: both read `T3` (the SAME 3-way (marginal-origin,
    other-pair) table -- see the plan's table for the algebraic identity that makes these two named
    blocks collapse onto one underlying joint distribution).
  - H_PP fully-disjoint: `T4` only.
`n_rows x n_rows` is NEVER draw-indexed (no `W` dimension at all), so this is the ordinary small
dense Hessian scratch every family already builds, not the forbidden dense G.
"""
function fill_pairwise_quantile_hessian_raw!(HfullR::AbstractMatrix{Float64}, op::PairwiseQuantileOperator,
        tabs::PairwiseQuantileHessianTables)
    D = op.D; npair = op.npair
    nrow = n_total_rows(D)
    size(HfullR) == (nrow, nrow) || error("fill_pairwise_quantile_hessian_raw!: size(HfullR)=$(size(HfullR)) != ($nrow,$nrow)")
    fill!(HfullR, 0.0)
    T1 = tabs.T1; T2 = tabs.T2; T3 = tabs.T3

    # ---- H_MM same-origin (diagonal, from T1) ----
    @inbounds for o in 1:D, a in 1:4
        i = marginal_row(o, a)
        HfullR[i, i] = T1[o, a]
    end

    # ---- H_MM cross-origin (dense 4x4, from T2) ----
    @inbounds for pidx in 1:npair
        (o, p) = op.pairs[pidx]
        for b in 1:4, a in 1:4
            i = marginal_row(o, a); j = marginal_row(p, b)
            v = T2[a, b, pidx]
            HfullR[i, j] = v
            HfullR[j, i] = v
        end
    end

    # ---- H_PP same-pair (diagonal in the 16 (a,b) cells, from T2 itself) ----
    @inbounds for pidx in 1:npair, b in 1:4, a in 1:4
        i = pair_row(D, pidx, a, b)
        HfullR[i, i] = T2[a, b, pidx]
    end

    # ---- H_MP: marginal origin o vs pair pidx=(p,q), p<q ----
    @inbounds for pidx in 1:npair
        for o in 1:D
            slot = origin_slot(op, pidx, o)
            if slot == 1                       # o == p: Raw[a,(b,c)] = delta(a,b)*T2[a,c,pidx]
                for c in 1:4, a in 1:4
                    v = T2[a, c, pidx]
                    i = marginal_row(o, a); j = pair_row(D, pidx, a, c)
                    HfullR[i, j] = v; HfullR[j, i] = v
                end
            elseif slot == 2                   # o == q: Raw[a,(b,c)] = delta(a,c)*T2[b,a,pidx]
                for b in 1:4, a in 1:4
                    v = T2[b, a, pidx]
                    i = marginal_row(o, a); j = pair_row(D, pidx, b, a)
                    HfullR[i, j] = v; HfullR[j, i] = v
                end
            else                                # disjoint: genuine 3-way, Raw[a,(b,c)] = T3[a,b,c]
                tidx = op.triple_lookup[o, pidx]
                for c in 1:4, b in 1:4, a in 1:4
                    v = T3[a, b, c, tidx]
                    i = marginal_row(o, a); j = pair_row(D, pidx, b, c)
                    HfullR[i, j] = v; HfullR[j, i] = v
                end
            end
        end
    end

    # ---- H_PP: pidx1 vs pidx2, pidx1<pidx2 ----
    @inbounds for pidx1 in 1:npair-1
        (o1, o2) = op.pairs[pidx1]
        for pidx2 in pidx1+1:npair
            (o3, o4) = op.pairs[pidx2]
            s1 = (o1 == o3 || o1 == o4) ? o1 : 0
            s2 = (o2 == o3 || o2 == o4) ? o2 : 0
            nshared = (s1 != 0 ? 1 : 0) + (s2 != 0 ? 1 : 0)
            if nshared == 0
                combo = op.quad_lookup[pidx1, pidx2]
                combo == 0 && error("fill_pairwise_quantile_hessian_raw!: missing quad combo for disjoint ($pidx1,$pidx2)")
                T4 = tabs.T4
                for d in 1:4, c in 1:4, b in 1:4, a in 1:4
                    v = T4[a, b, c, d, combo]
                    i = pair_row(D, pidx1, a, b); j = pair_row(D, pidx2, c, d)
                    HfullR[i, j] = v; HfullR[j, i] = v
                end
            elseif nshared == 1
                s = s1 != 0 ? s1 : s2
                other1 = o1 == s ? o2 : o1
                other2 = o3 == s ? o4 : o3
                opidx = pair_index(op, other1, other2)
                tidx = op.triple_lookup[s, opidx]
                (u, _v) = op.pairs[opidx]   # u < _v, sorted order of (other1,other2)
                for d in 1:4, c in 1:4, b in 1:4, a in 1:4
                    bin_s_1 = (o1 == s) ? a : b
                    bin_s_2 = (o3 == s) ? c : d
                    bin_s_1 == bin_s_2 || continue
                    other1_val = (o1 == s) ? b : a
                    other2_val = (o3 == s) ? d : c
                    bin_u = (other1 == u) ? other1_val : other2_val
                    bin_v = (other1 == u) ? other2_val : other1_val
                    val = T3[bin_s_1, bin_u, bin_v, tidx]
                    i = pair_row(D, pidx1, a, b); j = pair_row(D, pidx2, c, d)
                    HfullR[i, j] = val; HfullR[j, i] = val
                end
            else
                error("fill_pairwise_quantile_hessian_raw!: pidx1=$pidx1 pidx2=$pidx2 share $nshared>1 origins -- impossible for distinct pairs")
            end
        end
    end

    return HfullR
end

"""
    center_and_scale_pairwise_quantile_hessian!(HfullR, op, tabs) -> HfullR

Applies the task's mandatory centering identity `X'WX - r t' - t r' + S t t'` IN PLACE over the
whole (small, non-draw-indexed) raw block just filled by `fill_pairwise_quantile_hessian_raw!`,
then scales by `1/M` (`M=W`, matching `zc_restriction_gram!`'s own `1/M` convention). `r_I` is
`T1[o,a]` for a marginal row, or `T2[a,b,pidx]` itself for a pair row (a pair indicator's own raw
first moment IS its own diagonal `T2` entry -- no separate accumulator needed). `t_I` is `1/5` for
marginal rows, `1/25` for pair rows.
"""
function center_and_scale_pairwise_quantile_hessian!(HfullR::AbstractMatrix{Float64}, op::PairwiseQuantileOperator,
        tabs::PairwiseQuantileHessianTables)
    D = op.D; npair = op.npair
    nrow = n_total_rows(D)
    r = Vector{Float64}(undef, nrow)
    t = Vector{Float64}(undef, nrow)
    @inbounds for o in 1:D, a in 1:4
        i = marginal_row(o, a)
        r[i] = tabs.T1[o, a]; t[i] = 0.2
    end
    @inbounds for pidx in 1:npair, b in 1:4, a in 1:4
        j = pair_row(D, pidx, a, b)
        r[j] = tabs.T2[a, b, pidx]; t[j] = 0.04
    end
    S = tabs.S
    invM = 1.0 / op.W
    @inbounds for J in 1:nrow
        tJ = t[J]; rJ = r[J]
        for I in 1:nrow
            HfullR[I, J] = (HfullR[I, J] - tJ * r[I] - t[I] * rJ + S * t[I] * tJ) * invM
        end
    end
    return HfullR
end

"""
    pack_upper_pairwise_quantile_hessian!(hvec, HfullR, nrow) -> hvec

Row-major upper-triangular packing, mirroring `cm_hessian_architectures.jl::pack_upper_cm_hessian!`'s
own convention exactly (`hvec[k] = 0.5*(HfullR[i,j]+HfullR[j,i])` for `i<=j` -- defensive
symmetrization even though `HfullR` is already built symmetric by construction here, same
"mirror both triangles" discipline flagged in this codebase's own Hessian-symmetry memory note).
`length(hvec) == nrow*(nrow+1)/2`.
"""
function pack_upper_pairwise_quantile_hessian!(hvec::AbstractVector{Float64}, HfullR::AbstractMatrix{Float64}, nrow::Int)
    length(hvec) == div(nrow * (nrow + 1), 2) || error("pack_upper_pairwise_quantile_hessian!: length(hvec) mismatch")
    k = 0
    @inbounds for i in 1:nrow, j in i:nrow
        k += 1
        hvec[k] = 0.5 * (HfullR[i, j] + HfullR[j, i])
    end
    return hvec
end
