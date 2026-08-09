# ================================================================================================
# Bin decode + presorted draws for the pairwise-quantile-independence restriction (draft eq. 32).
#
# Two structs, mirroring `zc_restriction_operator.jl`'s ZCRestrictionOperator/*Workspace split:
#   - PairwiseQuantileOperator: IMMUTABLE, campaign-lifetime (the draws `ctx.U` never change once
#     the scientific context is built). Holds the presorted-per-origin draws (Section 3 of the
#     plan -- needed by the O(log W) cutoff-crossing gradient method, pairwise_quantile_cutoff_
#     gradient.jl) and the pair-index convention (REUSED verbatim from cm_meanzc_moments.jl's
#     packed_pair_index/pair_oi_to_lin -- no new pair-ordering scheme invented here).
#   - PairwiseQuantileBinState: MUTABLE, rebuilt exactly once per OUTER point (never inside an
#     FG/Hessian callback -- this is the task's own load-bearing "decode once" requirement). Holds
#     the decoded cutoffs Q and the persistent bin[w,o]::UInt8 array.
#
# Requires pairwise_quantile_cutoff_transform.jl (decode_all_cutoffs!, PairwiseQuantileCutoffLayout)
# and cm_meanzc_moments.jl (packed_pair_index, pair_oi_to_lin) to already be included.
# ================================================================================================

"""
    PairwiseQuantileOperator(U::Matrix{Float64})

`U` is `ctx.U` (W x D raw productivity draws, immutable for the whole campaign). Builds, ONCE:
  - `sorted_z[:,o]` = `U[:,o]` sorted ascending; `sorted_idx[:,o]` = the corresponding `sortperm`
    (so `U[sorted_idx[k,o],o] == sorted_z[k,o]`) -- used by the cutoff-crossing gradient method to
    find, in `O(log W + k)`, exactly the draws whose bin membership changes when `q_{o,r}` moves.
  - `pairs = packed_pair_index(D)` (REUSED from `cm_meanzc_moments.jl`, `(o,p)` with `o<p`,
    `o` outer-loop-major) -- `unordered_pairs = npair = D*(D-1)/2`, the SAME ordering convention
    every other pair-indexed quantity in this codebase already uses.
"""
struct PairwiseQuantileOperator
    D::Int
    W::Int
    npair::Int
    pairs::Vector{Tuple{Int,Int}}
    sorted_z::Matrix{Float64}
    sorted_idx::Matrix{Int}
    # ---- Hessian table-combo registries (Section 5) -- campaign-lifetime, depend only on D/pairs,
    # never on draws/duals, so precomputed ONCE here rather than rebuilt inside the Hessian file.
    # `triple_lookup[o,pidx]` = 1-based index into the T3 combo axis if origin `o` is disjoint from
    # `pairs[pidx]` (i.e. a genuine 3-way (marginal,pair) combo exists), else 0 (sentinel: NOT a
    # Dict -- a plain D x npair Int matrix, O(1) lookup, matches the task's "no dictionaries"
    # instruction). `triple_combos[k] = (o,pidx)` is the inverse map.
    triple_lookup::Matrix{Int}
    triple_combos::Vector{Tuple{Int,Int}}
    # `quad_lookup[pidx1,pidx2]` = 1-based index into the T4 combo axis if `pairs[pidx1]` and
    # `pairs[pidx2]` share NO origin, for pidx1<pidx2 ONLY (canonical order -- the disjoint block's
    # transpose, pidx1>pidx2, is obtained by transposing the SAME stored 16x16 sub-block when
    # filling Hfull, never by storing a second copy). `quad_combos[k]=(pidx1,pidx2)`, pidx1<pidx2.
    quad_lookup::Matrix{Int}
    quad_combos::Vector{Tuple{Int,Int}}
end

function PairwiseQuantileOperator(U::AbstractMatrix{Float64})
    W, D = size(U)
    D >= 2 || error("PairwiseQuantileOperator: D must be >= 2, got $D")
    pairs = packed_pair_index(D)
    npair = length(pairs)
    npair == div(D * (D - 1), 2) || error("PairwiseQuantileOperator: internal pair-count mismatch")
    sorted_z = Matrix{Float64}(undef, W, D)
    sorted_idx = Matrix{Int}(undef, W, D)
    @inbounds for o in 1:D
        idx = sortperm(@view U[:, o])
        sorted_idx[:, o] .= idx
        sorted_z[:, o] .= @view U[idx, o]
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

    return PairwiseQuantileOperator(D, W, npair, pairs, sorted_z, sorted_idx,
        triple_lookup, triple_combos, quad_lookup, quad_combos)
end

"pair_index(op, o, p) -> Int: O(1) linear index into the 190-pair convention, o,p in either order."
pair_index(op::PairwiseQuantileOperator, o::Int, p::Int) = pair_oi_to_lin(o, p, op.D)

"n_marginal_rows(D) = 4*D; n_pair_rows(D) = 16*C(D,2); n_total_rows(D) = their sum. D=20 gives the
task's own asserted counts: 80, 3040, 3120."
n_marginal_rows(D::Int) = 4 * D
n_pair_rows(D::Int) = 16 * div(D * (D - 1), 2)
n_total_rows(D::Int) = n_marginal_rows(D) + n_pair_rows(D)

"""
    PairwiseQuantileBinState(W::Int, D::Int)

Mutable, rebuilt once per outer point via `refresh_pairwise_quantile_bins!` below.
`Q[r,o] = q_{o,r}` (4xD physical cutoffs); `bin[w,o] in 1:5` (Wx D, `UInt8`).
"""
mutable struct PairwiseQuantileBinState
    Q::Matrix{Float64}
    bin::Matrix{UInt8}
end

PairwiseQuantileBinState(W::Int, D::Int) = PairwiseQuantileBinState(zeros(4, D), zeros(UInt8, W, D))

"""
    refresh_pairwise_quantile_bins!(state, op, U, raw, layout) -> state

ONE per-outer-point refresh: decode the 80 raw KNITRO coordinates into physical cutoffs `Q`
(`decode_all_cutoffs!`, `pairwise_quantile_cutoff_transform.jl`), then assign every draw's bin via
`searchsortedfirst` on each origin's own (now up to date) 4-cutoff column -- the SAME
`searchsortedfirst(z, u)` convention `cm_hessian_architectures.jl::compute_bin_indices`/
`common_marginals_interval.jl::compute_bin_indices` already use (returns the smallest k with
`Q[k,o] >= u`, i.e. bin k covers `(Q[k-1,o], Q[k,o]]`, bin 1 covers `(-Inf, Q[1,o]]`, bin 5 covers
`(Q[4,o], Inf)`) -- reused verbatim, not reinvented, per the task's "reuse current ... histogram
infrastructure" instruction. `z_o` is a.s.-continuous, so the `<=`-vs-`<` boundary convention is a
probability-zero event and does not affect any of the equivalence proofs in the math note.

Caller's responsibility (Section 3 of the plan): call this ONCE per outer point, before starting
the inner KNITRO dual solve -- NEVER from inside an FG or Hessian callback.
"""
function refresh_pairwise_quantile_bins!(state::PairwiseQuantileBinState, op::PairwiseQuantileOperator,
                                          U::AbstractMatrix{Float64}, raw::AbstractVector{Float64},
                                          layout::PairwiseQuantileCutoffLayout)
    D = op.D
    decode_all_cutoffs!(state.Q, raw, layout)
    bin = state.bin
    Q = state.Q
    @inbounds for o in 1:D
        qcol = @view Q[:, o]
        for w in 1:op.W
            bin[w, o] = UInt8(searchsortedfirst(qcol, U[w, o]))
        end
    end
    return state
end
