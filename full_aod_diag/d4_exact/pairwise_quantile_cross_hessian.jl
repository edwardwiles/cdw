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
# slot/draw) to read the restriction's feature value via `state.bin` LOOKUP -- exactly this
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
    pairwise_quantile_cross_hessian_block!(HEQ, wctx, ws, op, state, tls, S) -> HEQ

Fills `HEQ` (`(wctx.ncolI+1) x n_total_rows(D)`) = `(1/M) * E' * diag(S) * G`, where `G`'s columns
are this restriction's own RAW (uncentered indicator) marginal/pair columns -- `G` is NEVER
materialized; every entry is built from `state.bin` lookups and the SAME small table-builder
(`build_pairwise_quantile_tables_threaded!`) the forward/transpose/Hessian files already use.
Requires `winner_pair_cross_hessian_zc_prep!(ws, wctx, S)` to have been called THIS Hessian
callback first (fills `ws.Snu` -- NOT redone here, matching `winner_pair_cross_hessian_zc_block!`'s
own precondition).

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
        op::PairwiseQuantileOperator, state::PairwiseQuantileBinState, tls::PairwiseQuantileThreadScratch,
        S::AbstractVector{Float64})
    D = op.D; npair = op.npair; W = op.W
    nrow = n_total_rows(D)
    ncolI = wctx.ncolI
    size(HEQ) == (ncolI + 1, nrow) || error("pairwise_quantile_cross_hessian_block!: size(HEQ)=$(size(HEQ)) != ($(ncolI+1),$nrow)")
    length(S) == W || error("pairwise_quantile_cross_hessian_block!: length(S)=$(length(S)) != W=$W")

    M = W

    # ---- row 1: S-only, no nu -- reduces to the plain S-weighted marginal/pair tables ----
    Mtab_S = zeros(D, 5); Ptab_S = zeros(5, 5, npair)
    build_pairwise_quantile_tables_threaded!(Mtab_S, Ptab_S, tls, op, state, S)
    row1 = @view HEQ[1, :]
    @inbounds for o in 1:D, a in 1:4
        row1[marginal_row(o, a)] = Mtab_S[o, a] / M
    end
    @inbounds for pidx in 1:npair, b in 1:4, a in 1:4
        row1[pair_row(D, pidx, a, b)] = Ptab_S[a, b, pidx] / M
    end

    # ---- Snu-weighted tables: feed BOTH the NuZ-style correction AND (implicitly, via Snu itself
    # having already been computed by winner_pair_cross_hessian_zc_prep!) the bilateral scatter's
    # per-slot weight v[w]=Snu[w]*y[w,slot]. ----
    Mtab_Snu = zeros(D, 5); Ptab_Snu = zeros(5, 5, npair)
    build_pairwise_quantile_tables_threaded!(Mtab_Snu, Ptab_Snu, tls, op, state, ws.Snu)

    nbilateral = wctx.has_cf ? ncolI - 1 : ncolI
    bilateral_block = @view HEQ[2:1+nbilateral, :]
    fill!(bilateral_block, 0.0)

    y = wctx.y; winner = wctx.winner; Ddest = wctx.Ddest
    pairs = op.pairs
    v = Vector{Float64}(undef, W)
    @inbounds for slot in 1:Ddest
        for w in 1:W
            v[w] = ws.Snu[w] * y[w, slot]
        end
        wcol = @view winner[:, slot]
        for w in 1:W
            o_ = wcol[w]
            j = slot + (o_ - 1) * Ddest
            vw = v[w]
            for o in 1:D
                a = state.bin[w, o]
                a <= 0x04 && (bilateral_block[j, marginal_row(o, a)] += vw)
            end
            for pidx in 1:npair
                (p, q) = pairs[pidx]
                a = state.bin[w, p]; b = state.bin[w, q]
                (a <= 0x04 && b <= 0x04) && (bilateral_block[j, pair_row(D, pidx, a, b)] += vw)
            end
        end
    end

    pi_vec = wctx.pi_vec
    @inbounds for j in 1:nbilateral
        pij = pi_vec[j]
        for o in 1:D, a in 1:4
            x = marginal_row(o, a)
            HEQ[j+1, x] = invM_apply(bilateral_block[j, x], pij, Mtab_Snu[o, a], M)
        end
        for pidx in 1:npair, b in 1:4, a in 1:4
            x = pair_row(D, pidx, a, b)
            HEQ[j+1, x] = invM_apply(bilateral_block[j, x], pij, Ptab_Snu[a, b, pidx], M)
        end
    end

    if wctx.has_cf
        jcf = ncolI
        row_cf = @view HEQ[jcf+1, :]
        crsbuf = ws.crs_buf
        Mtab_cf = zeros(D, 5); Ptab_cf = zeros(5, 5, npair)
        build_pairwise_quantile_tables_threaded!(Mtab_cf, Ptab_cf, tls, op, state, crsbuf)
        @inbounds for o in 1:D, a in 1:4
            row_cf[marginal_row(o, a)] = Mtab_cf[o, a] / M
        end
        @inbounds for pidx in 1:npair, b in 1:4, a in 1:4
            row_cf[pair_row(D, pidx, a, b)] = Ptab_cf[a, b, pidx] / M
        end
    end

    return HEQ
end

"HEQ[j+1,x] = (1/M)*(bilateral_raw[j,x] - pi_j*NuX[x]) -- same rank-1 pi_vec correction convention
as winner_pair_cross_hessian_zc_block!'s uncorrected (`use_profiled_correction=false`) branch."
@inline invM_apply(raw::Float64, pij::Float64, nux::Float64, M::Int) = (raw - pij * nux) / M
