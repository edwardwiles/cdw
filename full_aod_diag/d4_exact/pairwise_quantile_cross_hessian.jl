# ================================================================================================
# Economic x pairwise-quantile Hessian cross-block (H_E,new), Section 6 of the implementation plan.
#
# Reuses `WinnerPairHessCtx`/`winner_pair_hessian!` (H_EE, `core_exact_hessian.jl`) and
# `winner_pair_cross_hessian_zc_prep!`/`WinnerZCCrossScratch` (`winner_pair_cross_hessian.jl`)
# COMPLETELY UNCHANGED -- those are restriction-agnostic (they only depend on `wctx`'s winner
# assignments/CES-kernel state and the current dual weights `S`, never on which restriction is
# active). H_EE itself is never touched by this file.
#
# The existing GENERIC `winner_pair_cross_hessian_zc_block!` (`winner_pair_cross_hessian.jl:493`)
# requires a materialized centered `Z::Matrix{Float64}` (W x nx) argument -- calling it with
# nx=n_total_rows(D)=3120 at D=20 would require exactly the forbidden dense W x 3120 matrix, so it
# is NOT reusable as-is here (confirmed during planning, not assumed). This file instead adapts
# that function's OWN scatter-accumulate discipline (winner-slot loop x draw loop, one v[w] per
# slot/draw) to read the restriction's feature value via `op.bin` LOOKUP -- exactly this
# restriction's own O(D+npair)-per-draw forward/transpose discipline -- rather than a materialized
# Z column read. This changes the per-draw feature ACCESSOR only; it does not rewrite H_EE, does
# not rewrite `winner_pair_hessian!`, and does not introduce a second copy of the winner-pair CES
# kernel.
#
# SCOPE NOTE (disclosed): this file is written directly against the researched `WinnerPairHessCtx`/
# `WinnerZCCrossScratch` field contracts (`core_exact_hessian.jl`, `winner_pair_cross_hessian.jl`)
# but has NOT been exercised against a live `wctx` in this session (that requires a real
# `CompressedFactual` from the full D20/W=100k production context, out of reach standalone). See
# the session status doc for what remains to validate this block end-to-end.
#
# Requires pairwise_quantile_bin_context.jl and pairwise_quantile_operator.jl (for
# build_pairwise_quantile_tables_threaded!/PairwiseQuantileThreadScratch, REUSED here for the
# S-weighted and Snu-weighted marginal/pair tables) to already be included, plus the production
# WinnerPairHessCtx/WinnerZCCrossScratch/winner_pair_cross_hessian_zc_prep! definitions.
# ================================================================================================

"""
    PairwiseQuantileCrossHessScratch(D, npair, W, nbilateral)

Persistent (campaign-lifetime shape, rebuilt in VALUE every Hessian callback -- never reallocated)
scratch for `pairwise_quantile_cross_hessian_block!`. Fixes the ~900KB/call allocation the handover
doc's "Two smaller, lower-risk fixes" #2 flagged: `Mtab_S`/`Ptab_S`/`Mtab_Snu`/`Ptab_Snu`/
`Mtab_cf`/`Ptab_cf`/`v`/`v_winner_sum` were previously built fresh (`zeros(...)`) every call, unlike
every other block in this codebase (zero- or near-zero-allocation already).
"""
mutable struct PairwiseQuantileCrossHessScratch
    Mtab_S::Matrix{Float64}
    Ptab_S::Array{Float64,3}
    Mtab_Snu::Matrix{Float64}
    Ptab_Snu::Array{Float64,3}
    Mtab_cf::Matrix{Float64}
    Ptab_cf::Array{Float64,3}
    v::Vector{Float64}
    v_winner_sum::Vector{Float64}
    # version B: the per-row centering constants c_I are now mass-dependent (mu[o,a],
    # mu[o,a]*mu[p,b]) instead of the two scalars 1/L, 1/L^2, so they are materialized once per
    # call into this n_rows-long buffer via the SHARED `pairwise_quantile_target_vector!` rather
    # than recomputed inline at each of this block's four centering sites.
    tvec::Vector{Float64}
    # TRANSPOSED accumulator for the bilateral scatter, `n_total_rows x nbilateral` (9.5 MB at
    # D=20/L=5). Added 2026-08-12 with the scatter restructure: the old code accumulated into a ROW
    # of the column-major output view, so consecutive columns strided by NCORE=382 doubles and
    # essentially every one of ~399M accumulations touched a fresh cache line across a 9.5 MB block.
    # Accumulating transposed makes one `j`'s working set a contiguous 25 KB column, and the
    # transpose is paid once per callback on 1.2M entries instead of per increment.
    Bt::Matrix{Float64}
end

function PairwiseQuantileCrossHessScratch(D::Int, npair::Int, W::Int, nbilateral::Int, L::Int)
    return PairwiseQuantileCrossHessScratch(
        zeros(D, L), zeros(L, L, npair), zeros(D, L), zeros(L, L, npair),
        zeros(D, L), zeros(L, L, npair), Vector{Float64}(undef, W), zeros(nbilateral),
        zeros(n_total_rows(D, L)), zeros(n_total_rows(D, L), nbilateral))
end

"""
    pairwise_quantile_cross_hessian_block!(HEQ, wctx, ws, op, state, tls, S, cross_hess_scratch) -> HEQ

Fills `HEQ` (`(wctx.ncolI+1) x n_total_rows(D)`) = `(1/M) * E' * diag(S) * G`, where `G`'s columns
are this restriction's own RAW (uncentered indicator) marginal/pair columns -- `G` is NEVER
materialized; every entry is built from `op.bin` lookups and the SAME small table-builder
(`build_pairwise_quantile_tables_threaded!`) the forward/transpose/Hessian files already use.
Requires `winner_pair_cross_hessian_zc_prep!(ws, wctx, S)` to have been called THIS Hessian
callback first (fills `ws.Snu` -- NOT redone here, matching `winner_pair_cross_hessian_zc_block!`'s
own precondition). `cross_hess_scratch` (`PairwiseQuantileCrossHessScratch`) supplies all per-call
buffers -- built ONCE per campaign, reused (via `fill!`, not reallocated) every call.

Row 1 (the zeta-paired "ones" row, S-only, no nu-correction) and the rank-1 `NuZ`-style correction
term both reduce to the SAME (S-weighted / Snu-weighted respectively) marginal+pair tables the
Hessian file's `T1`/`T2` already compute -- built here via TWO extra calls to
`build_pairwise_quantile_tables_threaded!` (weight=`S`, weight=`ws.Snu`), not a new kernel.

The winner-slot bilateral scatter (`HEQ[j+1,x] += v[w]*x_w` for the restriction's own `x`) is
genuinely new per-draw work, but costs `O(Ddest*W*(D+npair))` -- the SAME `O(D+npair)`-per-draw
complexity as this restriction's forward pass, NOT `O(Ddest*W*n_rows)` (n_rows=3120), since each
draw only scatters into the handful of cells it is ACTIVE in, never a full row of length n_rows.
"""
function pairwise_quantile_cross_hessian_block!(HEQ::AbstractMatrix{Float64}, wctx, ws,
        op::PairwiseQuantileOperator, state::PairwiseQuantileMassState,
        tls::PairwiseQuantileThreadScratch, S::AbstractVector{Float64},
        cross_hess_scratch::PairwiseQuantileCrossHessScratch)
    D = op.D; npair = op.npair; W = op.W; L = op.L; nc = L - 1
    nrow = n_total_rows(D, L)
    ncolI = wctx.ncolI
    size(HEQ) == (ncolI + 1, nrow) ||
        error("pairwise_quantile_cross_hessian_block!: size(HEQ)=$(size(HEQ)) != ($(ncolI+1),$nrow)")
    length(S) == W || error("pairwise_quantile_cross_hessian_block!: length(S)=$(length(S)) != W=$W")
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
    Bt = cross_hess_scratch.Bt
    (size(Bt, 1) == nrow && size(Bt, 2) >= nbilateral) ||
        error("pairwise_quantile_cross_hessian_block!: scratch Bt is $(size(Bt)), needs ($nrow, >= $nbilateral)")
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

"HEQ[j+1,x] = (1/M)*(bilateral_raw[j,x] - pi_j*NuX[x]) -- same rank-1 pi_vec correction convention
as winner_pair_cross_hessian_zc_block!'s uncorrected (`use_profiled_correction=false`) branch."
@inline invM_apply(raw::Float64, pij::Float64, nux::Float64, M::Int) = (raw - pij * nux) / M
