# ================================================================================================
# OPTIMIZED economic x pairwise-quantile cross Hessian (H_E,R) for family #7, 2026-08-12.
#
# WHY. At real D=20/W=100,000 this block MEASURED 2.556 s of a 5.514 s Hessian callback -- 46.8%,
# the largest single cost, and nearly four times the entire new CM x PQ block. The existing
# `pairwise_quantile_cross_hessian_block!` (pairwise_quantile_cross_hessian.jl) already exploits the
# two structural facts that matter -- the WINNER assignment (each draw contributes to exactly one
# economic row per slot, not all `ncolI`) and the PQ SPARSITY (each draw scatters into the
# `D + npair = 210` cells it is active in, never a full row of 3120) -- but it was written serial and
# never received the threading its two siblings did (`winner_pair_cross_hessian_fill_threaded!` and
# `winner_pair_cross_hessian_zc_block_threaded!`, threaded_cross_hessian.jl). This file closes that,
# plus two memory-layout problems in the same loop.
#
# THREE CHANGES, and each is a distinct problem:
#
#  1. THREADING, over `slot`. For a fixed `slot`, `j = slot + (o-1)*Ddest`, so `j ≡ slot (mod Ddest)`
#     -- different slots write DISJOINT sets of `j`. No reduction, no atomics, no per-thread copy of
#     the output. And because each `j` is touched by exactly one slot with the `w` order unchanged,
#     the result is BIT-IDENTICAL to the serial version, not merely equal to 1e-14. That is what
#     makes this gateable by exact comparison rather than by tolerance.
#
#  2. SCATTER LOCALITY -- accumulate TRANSPOSED. The serial version scatters into
#     `bilateral_block[j, x]`, a row of a COLUMN-MAJOR `NCORE x n_rows` view: consecutive `x` stride
#     by `NCORE = 382` doubles, so essentially every one of ~399M accumulations touches a different
#     cache line, over a 9.5 MB block. Accumulating into `Bt[x, j]` instead makes the working set for
#     one `j` a CONTIGUOUS 3120-double column (25 KB, L1/L2-resident), and the transpose is paid once
#     per callback at 1.2M entries instead of per-increment.
#
#  3. INDEX WORK -- hoist this draw's bins once. `bin[w, o]` is a `W x D` `UInt8` matrix, so
#     `bin[w, p]` for varying `p` strides by `W`. The serial inner loop reads it TWICE PER PAIR --
#     380 strided loads per `(slot, w)` -- to recover only `D = 20` distinct values. Copying
#     `bin[w, 1:D]` into a `D`-length local first turns those 380 strided loads into 20, and the pair
#     loop then reads from a 20-byte buffer that stays in L1.
#
# WHAT IS *NOT* CHANGED, deliberately: the algebra. Row 1, the `Snu`-weighted correction tables, the
# centering convention, the `v_winner_sum` per-`j` centering, the `pi_vec` rank-1 correction and the
# cf row are all carried over verbatim from the serial function, which carries two live-found bug
# fixes (row 1's missing centering, and the per-`j` rather than global `v_winner_sum`). This is a
# loop-structure change only.
#
# WHY IT IS A SEPARATE FUNCTION RATHER THAN AN EDIT TO THE SHARED ONE. That file is the STANDALONE
# pairwise-quantile family's, and that family is in production. This version is called only by
# family #7 until it has been gated against the standalone family's own results too. The cost of the
# split is a ~100-line duplicate that could drift -- mitigated by `test_cm_pairwise_quantile_real_d4_hessian.jl`
# asserting BIT-IDENTITY between the two on every run, which is only possible because of property 1.
#
# `v` is gone entirely: the serial version materializes `v[w] = Snu[w]*y[w,slot]` into a length-W
# buffer, then reads it back one element at a time in the very next loop. Computed inline here, which
# also removes the need for any per-thread scratch in the scatter.
# ================================================================================================

isdefined(Main, :PairwiseQuantileCrossHessScratch) ||
    include(joinpath(@__DIR__, "pairwise_quantile_cross_hessian.jl"))

"""
    CMPQCrossHessFastScratch(n_rows, nbilateral)

Campaign-lifetime buffers for `cmpq_pq_cross_hessian_block_fast!`.

`Bt` is the TRANSPOSED accumulator (`n_rows x nbilateral`, 9.5 MB at D=20/L=5) -- see change 2 in
this file's header. It is the whole point of the restructure and is not optional scratch.

SIZED TO `ncolI`, NOT `nbilateral`, and the fill accepts any `nbilateral <= ncolI`. `nbilateral` is
`ncolI - 1` exactly when the gravity common-factor column is present (`wctx.has_cf`), which is a
property of the per-outer-point `CompressedFactual` and is NOT known when this campaign-lifetime
buffer is built. Allocating the upper bound and checking `>=` is the fix; sizing it exactly at build
time was wrong and failed on the first real D=20 run (`(3120,381)` vs an expected `(3120,380)`).
"""
mutable struct CMPQCrossHessFastScratch
    Bt::Matrix{Float64}          # n_rows x nbilateral, transposed accumulator
    n_rows::Int
    nbilateral::Int
end
CMPQCrossHessFastScratch(n_rows::Int, nbilateral::Int) =
    CMPQCrossHessFastScratch(zeros(n_rows, nbilateral), n_rows, nbilateral)

"""
    cmpq_pq_cross_hessian_block_fast!(HEQ, wctx, ws, op, state, tls, S, cross_hess_scratch, fast)
        -> HEQ

Drop-in replacement for `pairwise_quantile_cross_hessian_block!` with the identical signature plus
the extra `fast::CMPQCrossHessFastScratch`. Produces a BIT-IDENTICAL `HEQ`.

Precondition unchanged: `winner_pair_cross_hessian_zc_prep!(ws, wctx, S)` must have been called this
Hessian callback (fills `ws.Snu`).
"""
function cmpq_pq_cross_hessian_block_fast!(HEQ::AbstractMatrix{Float64}, wctx, ws,
        op::PairwiseQuantileOperator, state::PairwiseQuantileMassState,
        tls::PairwiseQuantileThreadScratch, S::AbstractVector{Float64},
        cross_hess_scratch::PairwiseQuantileCrossHessScratch, fast::CMPQCrossHessFastScratch)
    D = op.D; npair = op.npair; W = op.W; L = op.L; nc = L - 1
    nrow = n_total_rows(D, L)
    ncolI = wctx.ncolI
    size(HEQ) == (ncolI + 1, nrow) ||
        error("cmpq_pq_cross_hessian_block_fast!: size(HEQ)=$(size(HEQ)) != ($(ncolI+1),$nrow)")
    length(S) == W || error("cmpq_pq_cross_hessian_block_fast!: length(S)=$(length(S)) != W=$W")
    M = W
    tvec = cross_hess_scratch.tvec
    pairwise_quantile_target_vector!(tvec, op, state)
    nlast = UInt8(nc)

    # ---- row 1 (S-only) -- unchanged, and already threaded inside the table builder --------------
    Mtab_S = cross_hess_scratch.Mtab_S; Ptab_S = cross_hess_scratch.Ptab_S
    fill!(Mtab_S, 0.0); fill!(Ptab_S, 0.0)
    build_pairwise_quantile_tables_threaded!(Mtab_S, Ptab_S, tls, op, S)
    S_sum = sum(S)
    row1 = @view HEQ[1, :]
    @inbounds for o in 1:D, a in 1:nc
        x = marginal_row(o, a, L)
        row1[x] = (Mtab_S[o, a] - tvec[x] * S_sum) / M
    end
    @inbounds for pidx in 1:npair, b in 1:nc, a in 1:nc
        x = pair_row(D, pidx, a, b, L)
        row1[x] = (Ptab_S[a, b, pidx] - tvec[x] * S_sum) / M
    end

    # ---- Snu-weighted correction tables -- unchanged ---------------------------------------------
    Mtab_Snu = cross_hess_scratch.Mtab_Snu; Ptab_Snu = cross_hess_scratch.Ptab_Snu
    fill!(Mtab_Snu, 0.0); fill!(Ptab_Snu, 0.0)
    build_pairwise_quantile_tables_threaded!(Mtab_Snu, Ptab_Snu, tls, op, ws.Snu)
    Snu_sum = sum(ws.Snu)
    @inbounds for o in 1:D, a in 1:nc
        Mtab_Snu[o, a] -= tvec[marginal_row(o, a, L)] * Snu_sum
    end
    @inbounds for pidx in 1:npair, b in 1:nc, a in 1:nc
        Ptab_Snu[a, b, pidx] -= tvec[pair_row(D, pidx, a, b, L)] * Snu_sum
    end

    nbilateral = wctx.has_cf ? ncolI - 1 : ncolI
    (fast.n_rows == nrow && fast.nbilateral >= nbilateral) ||
        error("cmpq_pq_cross_hessian_block_fast!: fast scratch is ($(fast.n_rows),$(fast.nbilateral)), " *
              "needs ($nrow, >= $nbilateral)")
    Bt = fast.Bt
    fill!(Bt, 0.0)
    v_winner_sum = cross_hess_scratch.v_winner_sum
    fill!(v_winner_sum, 0.0)

    y = wctx.y; winner = wctx.winner; Ddest = wctx.Ddest
    pairs = op.pairs
    bin = op.bin
    Snu = ws.Snu
    nmarg = n_marginal_rows(D, L)

    # ---- THE SCATTER: threaded over slot, transposed accumulation, bins hoisted per draw ----------
    Threads.@threads :dynamic for slot in 1:Ddest
        # One D-length bin buffer per task (19 tiny allocations per callback, not per draw).
        b = Vector{UInt8}(undef, D)
        @inbounds for w in 1:W
            o_ = winner[w, slot]
            j = slot + (o_ - 1) * Ddest
            vw = Snu[w] * y[w, slot]
            v_winner_sum[j] += vw          # j is unique to this slot -- no race
            col = @view Bt[:, j]           # contiguous 3120-double column
            for o in 1:D
                b[o] = bin[w, o]           # 20 strided loads, once per (slot, draw)
            end
            for o in 1:D
                a = b[o]
                a <= nlast && (col[(o - 1) * nc + a] += vw)     # == marginal_row(o,a,L)
            end
            for pidx in 1:npair
                (p, q) = pairs[pidx]
                a = b[p]; bq = b[q]
                if a <= nlast && bq <= nlast
                    col[nmarg + (pidx - 1) * nc * nc + (bq - 1) * nc + a] += vw   # == pair_row(...)
                end
            end
        end
    end

    # ---- centering + pi_vec correction, threaded over j (disjoint rows of HEQ) --------------------
    pi_vec = wctx.pi_vec
    Threads.@threads :dynamic for j in 1:nbilateral
        pij = pi_vec[j]
        vwj = v_winner_sum[j]
        @inbounds begin
            colj = @view Bt[:, j]
            for o in 1:D, a in 1:nc
                x = marginal_row(o, a, L)
                HEQ[j+1, x] = invM_apply(colj[x] - tvec[x] * vwj, pij, Mtab_Snu[o, a], M)
            end
            for pidx in 1:npair, b2 in 1:nc, a in 1:nc
                x = pair_row(D, pidx, a, b2, L)
                HEQ[j+1, x] = invM_apply(colj[x] - tvec[x] * vwj, pij, Ptab_Snu[a, b2, pidx], M)
            end
        end
    end

    # ---- cf row -- unchanged --------------------------------------------------------------------
    if wctx.has_cf
        jcf = ncolI
        row_cf = @view HEQ[jcf+1, :]
        crsbuf = ws.crs_buf
        Mtab_cf = cross_hess_scratch.Mtab_cf; Ptab_cf = cross_hess_scratch.Ptab_cf
        fill!(Mtab_cf, 0.0); fill!(Ptab_cf, 0.0)
        build_pairwise_quantile_tables_threaded!(Mtab_cf, Ptab_cf, tls, op, crsbuf)
        crs_sum = sum(crsbuf)
        @inbounds for o in 1:D, a in 1:nc
            x = marginal_row(o, a, L)
            row_cf[x] = (Mtab_cf[o, a] - tvec[x] * crs_sum) / M
        end
        @inbounds for pidx in 1:npair, b2 in 1:nc, a in 1:nc
            x = pair_row(D, pidx, a, b2, L)
            row_cf[x] = (Ptab_cf[a, b2, pidx] - tvec[x] * crs_sum) / M
        end
    end
    return HEQ
end
