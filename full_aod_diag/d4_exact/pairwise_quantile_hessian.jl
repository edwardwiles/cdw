# ================================================================================================
# Hessian (H_MM/H_MP/H_PP) for the pairwise-quantile-independence restriction (draft eq. 32).
#
# Implements Section 5 of the implementation plan: every raw (uncentered) Hessian block reduces to
# exactly one of four small table families (T1 1-way, T2 2-way, T3 3-way, T4 4-way), two of which
# (T2, T3) are each consumed by TWO different named blocks -- see the table in the plan / the
# per-block derivations in this file's own comments. Centering uses the task's mandatory identity
#     sum_w h_w(x_w-t)(x_w-t)' = X'WX - r t' - t r' + S t t'
# applied ONCE, densely, over the (small, D+npair*(L-1)^2-sized, NEVER draw-indexed) restriction-
# only Hessian block -- this is the ordinary "small dense Hessian scratch" every other family
# already builds (CM's own Hfull is the same order of magnitude), not the forbidden W x n_rows
# dense G. `L`-generic (2026-08-09): every table dimension is sized off `op.L` (the number of
# quantile bins), not hardcoded to the task's own draft value `L=5`.
#
# Requires pairwise_quantile_bin_context.jl (PairwiseQuantileOperator, n_marginal_rows/n_pair_rows/
# n_total_rows) and pairwise_quantile_operator.jl (build_pairwise_quantile_tables_threaded!,
# PairwiseQuantileThreadScratch) to already be included -- T1/T2 reuse that SAME threaded builder
# (weighted by h_w=Psi''(q_w) instead of the transpose's Psi'(q_w)), per the task's "reuse current
# thread-local histogram infrastructure" instruction: one generalized accumulator, multiple
# consumers, never a second copy.
# ================================================================================================

"Row index of marginal (origin o, bin a) within the restriction-only Hfull block. `L` is the
number of quantile bins (`nc = L-1` active marginal columns per origin)."
marginal_row(o::Integer, a::Integer, L::Integer) = (Int(o) - 1) * (Int(L) - 1) + Int(a)

"Row index of pair cell (pidx, bin a of pairs[pidx][1], bin b of pairs[pidx][2]) within the block.
`L` is the number of quantile bins (`nc = L-1` active cells per axis, `nc^2` per pair)."
function pair_row(D::Integer, pidx::Integer, a::Integer, b::Integer, L::Integer)
    nc = Int(L) - 1
    return n_marginal_rows(Int(D), Int(L)) + (Int(pidx) - 1) * nc^2 + (Int(b) - 1) * nc + Int(a)
end

"""
    _lo_write!(H, i, j, v)

Store `v` at the LOWER-triangle position of the unordered index pair `(i,j)` -- `H[max(i,j), min(i,j)]`
-- and nowhere else.

WHY ONLY ONE TRIANGLE (2026-08-10). `H_RR` is symmetric, and its only consumer is the packed
upper-triangular vector KNITRO is handed (`KN_DENSE_ROWMAJOR`), which reads each unordered pair
exactly once. Mirroring every value into both triangles therefore did ~2x the stores and ~2x the
centering arithmetic for a half that nothing ever read. The LOWER triangle is the one kept because
the packed write walks it by COLUMN (`HRR[j, i]` with `j >= i`), which is the contiguous direction
in Julia's column-major layout -- see the packed-write comment in
`pairwisequantile_hess_cb_builder`. Diagonal entries (`i == j`) are written once, as before.
"""
@inline function _lo_write!(H::AbstractMatrix{Float64}, i::Int, j::Int, v::Float64)
    @inbounds H[max(i, j), min(i, j)] = v
    return nothing
end

"origin_slot(op,pidx,o): 1 if o==pairs[pidx][1], 2 if o==pairs[pidx][2], else 0 (disjoint)."
function origin_slot(op::PairwiseQuantileOperator, pidx::Int, o::Int)
    (p, q) = op.pairs[pidx]
    o == p && return 1
    o == q && return 2
    return 0
end

"""
    _assign_canonical_combos(tuples::Vector{NTuple{N,Int}}) -> (canon_idx, perm, canon_list)

Dedup helper (Hessian-optimization pass, 2026-08-09; handover doc "Verified finding #1"): for a
fixed UNORDERED N-tuple of origins, `tuples` (e.g. `triple_opq`/`quad_oooo`) contains N! / (N-1)! =
N entries that are the SAME underlying joint bin-probability table, just axis-permuted (each entry
picks a different origin to play the "marginal"/"first pair" role). `sortperm` on each entry gives
the exact axis permutation relating it to the CANONICAL (origin-sorted) version of that same
N-tuple; `canon_idx[k]` groups all entries sharing a canonical tuple onto ONE stored table;
`canon_list[c]` is that canonical (origin-sorted) N-tuple itself, in first-seen order -- this is
the SCATTER-time representative (`build_pairwise_quantile_hessian_tables!` scatters each canonical
tuple's OWN sorted origins directly, exactly once per draw, never the N redundant role-variants --
scattering all N would triple/quadruple-count, see this function's own correctness note below).

`perm[k]` (an `NTuple{N,Int}`) is used ONLY at READ time (`read_T3`/`read_T4`), to map an arbitrary
caller-supplied (o,p,q,...)-ordered bin tuple back onto `canon_list[canon_idx[k]]`'s own sorted
axis order: `canonical_vals = (vals[perm[1]], ..., vals[perm[N]])`.

CORRECTNESS NOTE (bug found + fixed live in this same pass, via `debug_pq_t3t4_dedup_isolate.jl`):
an earlier version of this dedup scattered EVERY one of the N redundant `tuples` entries (not just
`canon_list`) into the shared canonical table, which is a real N-way OVERCOUNT, not merely wasted
work -- confirmed via a synthetic brute-force check (max abs error ~3x the correct value at N=3,
localized instantly to the T3/T4 SCATTER loop, not the read-side permutation math, which was
already correct). The fix is scattering `canon_list` (one representative per canonical tuple) with
its own identity ordering, never the full redundant list.

Construction-time only (`O(#tuples)`, a few thousand at D=20) -- the `Dict` here is NOT a hot-path
structure (unlike `op.triple_lookup`/`op.quad_lookup`, which stay flat-matrix O(1) lookups); it
runs once per campaign, not once per draw or per Hessian callback.
"""
function _assign_canonical_combos(tuples::Vector{NTuple{N,Int}}) where {N}
    canon_map = Dict{NTuple{N,Int},Int}()
    canon_list = NTuple{N,Int}[]
    canon_idx = Vector{Int}(undef, length(tuples))
    perm = Vector{NTuple{N,Int}}(undef, length(tuples))
    for (k, vals) in enumerate(tuples)
        ord = Tuple(sortperm(collect(vals)))
        sorted_vals = ntuple(i -> vals[ord[i]], N)
        idx = get!(canon_map, sorted_vals) do
            push!(canon_list, sorted_vals)
            length(canon_list)
        end
        canon_idx[k] = idx
        perm[k] = ord
    end
    return canon_idx, perm, canon_list
end

"""
    PairwiseQuantileHessianTables(op::PairwiseQuantileOperator)

Persistent (campaign-lifetime shape, rebuilt in VALUE every Hessian callback -- never reallocated)
raw table storage: `T1` (D x L, `r_M`/H_MM-diagonal), `T2` (LxLxnpair, H_MM-cross / H_MP-same-
origin / H_PP-same-pair-diagonal), `T3` ((L-1)^3 x ncanon3, H_MP-disjoint / H_PP-shared-origin, ONE
entry per UNORDERED triple -- see `_assign_canonical_combos`), `T4` ((L-1)^4 x ncanon4,
H_PP-fully-disjoint ONLY, ONE entry per UNORDERED quadruple). `S = sum(h_w)` (the shared scalar the
centering identity needs). `ncanon3=C(D,3)`, `ncanon4=C(D,4)` -- a genuine 3x reduction from the
raw `length(op.triple_combos)=3*ncanon3`/`length(op.quad_combos)=3*ncanon4` combo counts (handover
doc "Verified finding #1": each canonical N-tuple was previously stored N times, axis-permuted).
`L=op.L` (the number of quantile bins) is `L`-generic throughout, not hardcoded to the task's own
draft value `L=5`.
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
    # dedup bookkeeping (Hessian-optimization pass): canonical index + axis permutation per ORIGINAL
    # (redundant) combo, used at READ (fill) time; `triple_canonical_list`/`quad_canonical_list` are
    # the ncanon3/ncanon4 SORTED representative tuples, used at SCATTER (build) time -- see
    # `_assign_canonical_combos`'s own correctness note for why scatter must use ONLY these, never
    # the redundant `triple_opq`/`quad_oooo` lists.
    triple_canon_idx::Vector{Int}
    triple_perm::Vector{NTuple{3,Int}}
    triple_canonical_list::Vector{NTuple{3,Int}}
    quad_canon_idx::Vector{Int}
    quad_perm::Vector{NTuple{4,Int}}
    quad_canonical_list::Vector{NTuple{4,Int}}
end

function PairwiseQuantileHessianTables(op::PairwiseQuantileOperator)
    D = op.D; npair = op.npair; L = op.L; nc = L - 1
    triple_opq = [(o, op.pairs[pidx][1], op.pairs[pidx][2]) for (o, pidx) in op.triple_combos]
    quad_oooo = [(op.pairs[pidx1][1], op.pairs[pidx1][2], op.pairs[pidx2][1], op.pairs[pidx2][2])
                 for (pidx1, pidx2) in op.quad_combos]

    triple_canon_idx, triple_perm, triple_canonical_list = _assign_canonical_combos(triple_opq)
    quad_canon_idx, quad_perm, quad_canonical_list = _assign_canonical_combos(quad_oooo)
    ncanon3 = length(triple_canonical_list); ncanon4 = length(quad_canonical_list)
    ncanon3 == binomial(D, 3) || error("PairwiseQuantileHessianTables: ncanon3=$ncanon3 != C(D,3)=$(binomial(D,3))")
    ncanon4 == binomial(D, 4) || error("PairwiseQuantileHessianTables: ncanon4=$ncanon4 != C(D,4)=$(binomial(D,4))")

    # T1/T2 are sized (D,L)/(L,L,npair) -- matching `build_pairwise_quantile_tables_threaded!`'s
    # full-L-bin output shape (SAME shared builder the transpose uses, see that function's own
    # docstring) -- only the [1:nc,...] slice is ever READ by the block-fill/centering code below
    # (bin L's dual is the implicit zero the task specifies), but the table itself is built over
    # all L bins so no second, differently-shaped builder is needed.
    return PairwiseQuantileHessianTables(zeros(D, L), zeros(L, L, npair), zeros(nc, nc, nc, ncanon3),
        zeros(nc, nc, nc, nc, ncanon4), 0.0, triple_opq, quad_oooo,
        triple_canon_idx, triple_perm, triple_canonical_list, quad_canon_idx, quad_perm, quad_canonical_list)
end

"Read T3 at ORIGINAL combo index `tidx` and its own (a,b,c) bin triple (matching `triple_opq[tidx]`'s
own origin order) -- applies `triple_perm[tidx]` to reach the deduped canonical table. Inverse of
the scatter-time transform in `build_pairwise_quantile_hessian_tables!`, so scatter and read always
agree by construction (same stored perm, never re-derived)."
@inline function read_T3(tabs::PairwiseQuantileHessianTables, tidx::Int, a::Integer, b::Integer, c::Integer)
    perm = tabs.triple_perm[tidx]
    vals = (a, b, c)
    @inbounds return tabs.T3[vals[perm[1]], vals[perm[2]], vals[perm[3]], tabs.triple_canon_idx[tidx]]
end

"Read T4 at ORIGINAL combo index `combo` and its own (a,b,c,d) bin quadruple (matching
`quad_oooo[combo]`'s own origin order) -- applies `quad_perm[combo]`, mirroring `read_T3` above."
@inline function read_T4(tabs::PairwiseQuantileHessianTables, combo::Int, a::Integer, b::Integer, c::Integer, d::Integer)
    perm = tabs.quad_perm[combo]
    vals = (a, b, c, d)
    @inbounds return tabs.T4[vals[perm[1]], vals[perm[2]], vals[perm[3]], vals[perm[4]], tabs.quad_canon_idx[combo]]
end

"""
    build_pairwise_quantile_hessian_tables!(tabs, op, h, tls) -> tabs

ONE per-Hessian-callback refresh of all four raw table families, weighted by `h = Psi''(q_w)`
(caller-supplied, matching every other family's `obj.ddPsi!(arg2,arg0)` convention). `T1`/`T2` are
built via the SAME threaded static-chunk/fixed-order-reduction builder the transpose uses
(`build_pairwise_quantile_tables_threaded!`), just weighted by `h` instead of `Psi'(q_w)`.

`T3`/`T4`: scatter into the DEDUPED canonical tables (`tabs.T3`/`tabs.T4`, sized
`ncanon3=C(D,3)`/`ncanon4=C(D,4)`, ~3x fewer combos than the raw `triple_opq`/`quad_oooo` lists),
with COMBOS on the outer loop and threads partitioned over COMBOS rather than draws -- see the
block comment at the loop for the measured op and memory counts that motivate both choices, and for
the (deliberate, gated) change in summation order.
"""
function build_pairwise_quantile_hessian_tables!(tabs::PairwiseQuantileHessianTables, op::PairwiseQuantileOperator,
        h::AbstractVector{Float64}, tls::PairwiseQuantileThreadScratch)
    build_pairwise_quantile_tables_threaded!(tabs.T1, tabs.T2, tls, op, h)
    tabs.S = sum(h)

    bin = op.bin
    W = op.W
    nlast = UInt8(op.L - 1)   # last ACTIVE bin index; bin L (implicit zero) is > nlast

    # ---- T3 / T4 scatter: COMBOS OUTER, DRAWS INNER, threaded over COMBOS ----------------------
    #
    # This loop order and threading strategy are both deliberate, and both are the opposite of what
    # the first version did (draws outer, combos inner, threaded over draws with a private full copy
    # of every table per thread). Measured op/memory counts at D=20, W=100,000 that motivated the
    # change:
    #
    #                                       L=5              L=10
    #   scatter adds  W*(C(D,3)+C(D,4))     598.5 M          598.5 M     (L-independent)
    #   per-thread reduction adds            21.0 M          521.9 M     <- pure overhead
    #   per-thread scratch (x16 threads)      0.16 GB          3.89 GB   <- most of the RSS
    #
    # 1. LOOP ORDER. With combos inner, the write target `T4[a,b,c,d,k]` is a random cell of the
    #    WHOLE table -- a 248.9 MB working set at L=10, so essentially every one of ~485M writes
    #    missed cache. With combos outer, a thread works inside ONE combo's slice at a time:
    #    `(L-1)^4` = 6561 doubles = 52 KB, which sits in L2. The draw loop then reads `bin[:,o]`
    #    columns and `h` contiguously.
    #
    # 2. THREADING OVER COMBOS, NOT DRAWS. Each combo `k` owns table slice `[..., k]` exclusively,
    #    so threads write to DISJOINT memory and no per-thread copy and no reduction are needed at
    #    all -- deleting `PairwiseQuantileHessThreadScratch` and, at L=10, 521.9 M reduction adds
    #    (46% of the total add count) and 3.89 GB of scratch. `:dynamic` because the per-combo cost
    #    is uniform but the combo counts (1140 / 4845) do not divide evenly across threads.
    #
    # ⚠️ SUMMATION ORDER CHANGED, so this is NOT bit-identical to the previous version: each cell is
    # now one sequential sum over `w in 1:W` instead of `nt` partial sums merged in a fixed order.
    # It is still fully DETERMINISTIC and now independent of thread count (which the old fixed-order
    # reduction only achieved for a fixed `nt`). The gate is `test_pairwise_quantile_d4_dense_oracle.jl`,
    # which compares the centered block against an INDEPENDENT dense reference at 1e-8 -- the right
    # gate for a summation-order change, and a stronger one than self-comparison.
    T3 = tabs.T3; fill!(T3, 0.0)
    triples_canon = tabs.triple_canonical_list
    Threads.@threads :dynamic for k in 1:length(triples_canon)
        (o, p, q) = triples_canon[k]
        @inbounds for w in 1:W
            a = bin[w, o]
            a > nlast && continue
            b = bin[w, p]
            b > nlast && continue
            c = bin[w, q]
            c > nlast && continue
            T3[a, b, c, k] += h[w]
        end
    end

    T4 = tabs.T4; fill!(T4, 0.0)
    quads_canon = tabs.quad_canonical_list
    Threads.@threads :dynamic for k in 1:length(quads_canon)
        (o1, o2, o3, o4) = quads_canon[k]
        @inbounds for w in 1:W
            a = bin[w, o1]
            a > nlast && continue
            b = bin[w, o2]
            b > nlast && continue
            c = bin[w, o3]
            c > nlast && continue
            d = bin[w, o4]
            d > nlast && continue
            T4[a, b, c, d, k] += h[w]
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
    D = op.D; npair = op.npair; L = op.L; nc = L - 1
    nrow = n_total_rows(D, L)
    size(HfullR) == (nrow, nrow) || error("fill_pairwise_quantile_hessian_raw!: size(HfullR)=$(size(HfullR)) != ($nrow,$nrow)")
    # Threaded zero: at L=10 this is 1.81 GiB, and a serial `fill!` measured 0.315 s of this
    # function's 1.262 s. Columns are disjoint, so this is embarrassingly parallel.
    Threads.@threads :static for j in 1:nrow
        @inbounds @simd for i in 1:nrow
            HfullR[i, j] = 0.0
        end
    end
    T1 = tabs.T1; T2 = tabs.T2

    # ---- H_MM same-origin (diagonal, from T1) ----
    @inbounds for o in 1:D, a in 1:nc
        i = marginal_row(o, a, L)
        HfullR[i, i] = T1[o, a]
    end

    # ---- H_MM cross-origin (dense nc x nc, from T2) ----
    @inbounds for pidx in 1:npair
        (o, p) = op.pairs[pidx]
        for b in 1:nc, a in 1:nc
            i = marginal_row(o, a, L); j = marginal_row(p, b, L)
            _lo_write!(HfullR, i, j, T2[a, b, pidx])
        end
    end

    # ---- H_PP same-pair (diagonal in the nc^2 (a,b) cells, from T2 itself) ----
    @inbounds for pidx in 1:npair, b in 1:nc, a in 1:nc
        i = pair_row(D, pidx, a, b, L)
        HfullR[i, i] = T2[a, b, pidx]
    end

    # ---- H_MP: marginal origin o vs pair pidx=(p,q), p<q ----
    # Threaded over pidx. `pair_row` is strictly increasing in pidx and every marginal row index is
    # below every pair row index, so here i (marginal) < j (pair) always, and `_lo_write!` therefore
    # stores at [j, i] -- row from the PAIR, column from the MARGINAL. Two iterations with different
    # pidx write disjoint ROW blocks, so there is no race.
    Threads.@threads :dynamic for pidx in 1:npair
        @inbounds for o in 1:D
            slot = origin_slot(op, pidx, o)
            if slot == 1                       # o == p: Raw[a,(b,c)] = delta(a,b)*T2[a,c,pidx]
                for c in 1:nc, a in 1:nc
                    i = marginal_row(o, a, L); j = pair_row(D, pidx, a, c, L)
                    _lo_write!(HfullR, i, j, T2[a, c, pidx])
                end
            elseif slot == 2                   # o == q: Raw[a,(b,c)] = delta(a,c)*T2[b,a,pidx]
                for b in 1:nc, a in 1:nc
                    i = marginal_row(o, a, L); j = pair_row(D, pidx, b, a, L)
                    _lo_write!(HfullR, i, j, T2[b, a, pidx])
                end
            else                                # disjoint: genuine 3-way, Raw[a,(b,c)] = T3[a,b,c]
                tidx = op.triple_lookup[o, pidx]
                for c in 1:nc, b in 1:nc, a in 1:nc
                    i = marginal_row(o, a, L); j = pair_row(D, pidx, b, c, L)
                    _lo_write!(HfullR, i, j, read_T3(tabs, tidx, a, b, c))
                end
            end
        end
    end

    # ---- H_PP: pidx1 vs pidx2, pidx1<pidx2 ----
    # The dominant block of the whole assembly: npair*(npair-1)/2 pair-of-pairs, each nc^4 cells
    # (at D=20/L=10 that is 17,955 x 6,561 ~ 118M writes).
    #
    # Threaded over pidx1. Since pidx1 < pidx2 and `pair_row` is strictly increasing in pidx, i < j
    # always, so `_lo_write!` stores at [j, i] -- row from pidx2, COLUMN from pidx1. Distinct pidx1
    # therefore own disjoint column blocks and cannot race. `:dynamic` because the pidx2 loop is
    # triangular: pidx1=1 does npair-1 inner iterations and pidx1=npair-1 does exactly one, so
    # static chunks would leave most threads idle.
    Threads.@threads :dynamic for pidx1 in 1:npair-1
        (o1, o2) = op.pairs[pidx1]
        @inbounds for pidx2 in pidx1+1:npair
            (o3, o4) = op.pairs[pidx2]
            s1 = (o1 == o3 || o1 == o4) ? o1 : 0
            s2 = (o2 == o3 || o2 == o4) ? o2 : 0
            nshared = (s1 != 0 ? 1 : 0) + (s2 != 0 ? 1 : 0)
            if nshared == 0
                combo = op.quad_lookup[pidx1, pidx2]
                combo == 0 && error("fill_pairwise_quantile_hessian_raw!: missing quad combo for disjoint ($pidx1,$pidx2)")
                for d in 1:nc, c in 1:nc, b in 1:nc, a in 1:nc
                    i = pair_row(D, pidx1, a, b, L); j = pair_row(D, pidx2, c, d, L)
                    _lo_write!(HfullR, i, j, read_T4(tabs, combo, a, b, c, d))
                end
            elseif nshared == 1
                s = s1 != 0 ? s1 : s2
                other1 = o1 == s ? o2 : o1
                other2 = o3 == s ? o4 : o3
                opidx = pair_index(op, other1, other2)
                tidx = op.triple_lookup[s, opidx]
                (u, _v) = op.pairs[opidx]   # u < _v, sorted order of (other1,other2)
                # The shared origin forces bin_s_1 == bin_s_2, so of the nc^4 (a,b,c,d) cells only
                # nc^3 are ever written -- the old form enumerated all nc^4 and threw away 8/9 of
                # them on a `continue`. Enumerate the surviving cells directly instead: pick the
                # SHARED origin's bin `bs` once, then the two non-shared origins' bins freely, and
                # reconstruct (a,b,c,d). Identical writes, nc^3 iterations instead of nc^4.
                s_is_first_1 = (o1 == s)    # loop-invariant: which slot of pidx1 carries s
                s_is_first_2 = (o3 == s)    # ditto for pidx2
                o1_is_u      = (other1 == u)
                for bs in 1:nc, ov1 in 1:nc, ov2 in 1:nc
                    a = s_is_first_1 ? bs  : ov1
                    b = s_is_first_1 ? ov1 : bs
                    c = s_is_first_2 ? bs  : ov2
                    d = s_is_first_2 ? ov2 : bs
                    bin_u = o1_is_u ? ov1 : ov2
                    bin_v = o1_is_u ? ov2 : ov1
                    i = pair_row(D, pidx1, a, b, L); j = pair_row(D, pidx2, c, d, L)
                    _lo_write!(HfullR, i, j, read_T3(tabs, tidx, bs, bin_u, bin_v))
                end
            else
                error("fill_pairwise_quantile_hessian_raw!: pidx1=$pidx1 pidx2=$pidx2 share $nshared>1 origins -- impossible for distinct pairs")
            end
        end
    end

    return HfullR
end

"""
    center_and_scale_pairwise_quantile_hessian!(HfullR, op, state, tabs) -> HfullR

Applies the task's mandatory centering identity `X'WX - r t' - t r' + S t t'` IN PLACE over the
whole (small, non-draw-indexed) raw block just filled by `fill_pairwise_quantile_hessian_raw!`,
then scales by `1/M` (`M=W`, matching `zc_restriction_gram!`'s own `1/M` convention). `r_I` is
`T1[o,a]` for a marginal row, or `T2[a,b,pidx]` itself for a pair row (a pair indicator's own raw
first moment IS its own diagonal `T2` entry -- no separate accumulator needed).

VERSION B: `t_I` is now `mu[o,a]` for marginal rows and `mu[o,a]*mu[p,b]` for pair rows
(version A: the constants `1/L`, `1/L^2`). That substitution is the ENTIRE Hessian edit, and the
expansion below is why -- writing `H_RR[I,J] = (1/W) sum_w h_w (ind_I - c_I)(ind_J - c_J)` out,

    (1/W)[ sum_w h*ind_I*ind_J  -  c_J sum_w h*ind_I  -  c_I sum_w h*ind_J  +  c_I c_J sum_w h ]
            \\_______________/
             the T1-T4 tables: ~78% of inner-solve wall-clock, and INDEPENDENT of the targets

the expensive part (`build_pairwise_quantile_hessian_tables!` /
`fill_pairwise_quantile_hessian_raw!`) is untouched by the reparameterization. This also means the
reparameterization buys EXACTNESS of the outer gradient, not inner-solve speed: the row count and
the T1-T4 cost are identical to version A's.
"""
function center_and_scale_pairwise_quantile_hessian!(HfullR::AbstractMatrix{Float64}, op::PairwiseQuantileOperator,
        state::PairwiseQuantileMassState, tabs::PairwiseQuantileHessianTables)
    D = op.D; npair = op.npair; L = op.L; nc = L - 1
    nrow = n_total_rows(D, L)
    size(state.mu) == (D, nc) ||
        error("center_and_scale_pairwise_quantile_hessian!: size(state.mu)=$(size(state.mu)) != ($D,$nc)")
    r = Vector{Float64}(undef, nrow)
    t = Vector{Float64}(undef, nrow)
    pairwise_quantile_target_vector!(t, op, state)   # the ONE definition of c_I (see its docstring)
    @inbounds for o in 1:D, a in 1:nc
        r[marginal_row(o, a, L)] = tabs.T1[o, a]
    end
    @inbounds for pidx in 1:npair, b in 1:nc, a in 1:nc
        r[pair_row(D, pidx, a, b, L)] = tabs.T2[a, b, pidx]
    end
    S = tabs.S
    invM = 1.0 / op.W
    # LOWER TRIANGLE ONLY (`I >= J`) -- the upper half of `HfullR` is never written by
    # `fill_pairwise_quantile_hessian_raw!` and never read by anything (see `_lo_write!`). Threaded
    # over `J` for the same reason the packed write is: columns are independent, there is no
    # reduction, and the schedule therefore cannot change the result by an ulp. `:dynamic` because
    # the column lengths are triangular and static chunks would be badly imbalanced.
    Threads.@threads :dynamic for J in 1:nrow
        tJ = t[J]; rJ = r[J]
        StJ = S * tJ
        @inbounds @simd for I in J:nrow
            HfullR[I, J] = (HfullR[I, J] - tJ * r[I] - t[I] * rJ + t[I] * StJ) * invM
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

Reads the LOWER triangle (`HfullR[j, i]`, `j >= i`), which since 2026-08-10 is the only half
`fill_pairwise_quantile_hessian_raw!` populates -- so the previous defensive
`0.5*(HfullR[i,j]+HfullR[j,i])` symmetrization is gone: it would now average a real value with a
structural zero. The symmetry it was defending is instead enforced upstream, by `_lo_write!` storing
each unordered pair exactly once.
"""
function pack_upper_pairwise_quantile_hessian!(hvec::AbstractVector{Float64}, HfullR::AbstractMatrix{Float64}, nrow::Int)
    length(hvec) == div(nrow * (nrow + 1), 2) || error("pack_upper_pairwise_quantile_hessian!: length(hvec) mismatch")
    k = 0
    @inbounds for i in 1:nrow, j in i:nrow
        k += 1
        hvec[k] = HfullR[j, i]
    end
    return hvec
end
