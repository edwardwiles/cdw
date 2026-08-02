# Genuine-cold ZC Hessian K=3 closeout task (2026-08-01), Section 6: H_EZ drawmajor-v2.
#
# Root cause of drawmajor-v1's modest/non-monotone speedup (see HCZ_HEZ_CANDIDATES_2026-08-01.md):
# v1's hot (x-tile outer, w inner) loop still does, for EVERY feature column x (630 times at
# production width), a strided read of `winner[w,slot]`/`v[w,slot]` (both `(W,Ddest)` column-major)
# to recover `j(w,slot)` and the accumulation weight -- even though neither depends on `x`. That is
# an Ddest-times-x redundant re-read of a small strided slice.
#
# v2 fix: each worker, ONCE per callback (inside its own spawned task, so the fix stays fully
# threaded), transposes its OWN draw chunk's `winner`/`Snu.*y` values into a persistent local
# `(Ddest, chunklen)` layout -- `rowidx_local[slot,iw]`/`v_local[slot,iw]` -- so `slot` varies
# CONTIGUOUSLY for fixed local draw `iw`. The x-tile loop then reads `rl[:,iw]`/`vl[:,iw]` as
# contiguous columns instead of re-deriving them from the strided global `winner`/`y` every x.
# Total FLOP count and target cells are unchanged from v1 -- this is purely a memory-traffic
# reduction (the one-time transpose cost replaces v1's Ddest-per-x redundant strided reads).

mutable struct WinnerZCDrawMajorV2Scratch
    W::Int
    Ddest::Int
    nbilateral::Int
    max_nx::Int
    workers::Int
    chunk_len_max::Int
    rowidx_local::Array{Int,3}     # (Ddest, chunk_len_max, workers)
    v_local::Array{Float64,3}      # (Ddest, chunk_len_max, workers)
    local_tabs::Array{Float64,3}   # (nbilateral, max_nx, workers)
    tasks::Vector{Task}
end
function WinnerZCDrawMajorV2Scratch(W::Int, Ddest::Int, nbilateral::Int, max_nx::Int, workers::Int)
    chunk_len_max = cld(W, workers)
    WinnerZCDrawMajorV2Scratch(W, Ddest, nbilateral, max_nx, workers, chunk_len_max,
        zeros(Int, Ddest, chunk_len_max, workers), zeros(Ddest, chunk_len_max, workers),
        zeros(nbilateral, max_nx, workers), Vector{Task}(undef, workers))
end
function ensure_winner_zc_drawmajor_v2_scratch!(dm::Union{Nothing,WinnerZCDrawMajorV2Scratch},
        W::Int, Ddest::Int, nbilateral::Int, max_nx::Int, workers::Int)
    if dm === nothing || dm.W != W || dm.Ddest != Ddest || dm.nbilateral != nbilateral ||
       dm.max_nx != max_nx || dm.workers != workers
        return WinnerZCDrawMajorV2Scratch(W, Ddest, nbilateral, max_nx, workers)
    end
    return dm
end

"""
    winner_pair_cross_hessian_zc_block_drawmajor_v2!(HEZ, wctx, ws, dm, S, Z, M; workers, xtile=64, use_profiled_correction=false) -> HEZ

drawmajor-v2: same public contract as `winner_pair_cross_hessian_zc_block_drawmajor!` (v1), fixing
v1's identified redundant per-x strided winner/v re-read via a per-worker one-time transpose.

Performance closeout task (2026-08-02), Section 6: `use_profiled_correction` mirrors
`winner_pair_cross_hessian_zc_block!`'s own keyword EXACTLY (same formula, same `ws.SnuWval`/
`ws.TZ_buf`/`wctx.target_slot`/`wctx.Lam_homog`/`wctx.denom_cf_scaled` fields, same single
`BLAS.gemm!('T','N',...)` TZ computation) -- the expensive winner-conditioned W-scale scatter loop
above (lines computing `dm.local_tabs`) is completely untouched by this keyword; only the final
O(nbilateral*nx)/O(nx) correction loop below (already present, tiny relative to the W-loop) branches
between the old (NuZ/pi_vec) and profiled (TZ/Lam_homog) corrections, exactly as the serial and
threaded siblings already do. No second W-scale pass, no dense G, no duplicated Z weighting.
"""
function winner_pair_cross_hessian_zc_block_drawmajor_v2!(HEZ::AbstractMatrix{Float64}, wctx::WinnerPairHessCtx,
        ws::WinnerZCCrossScratch, dm::WinnerZCDrawMajorV2Scratch, S::AbstractVector{Float64},
        Z::AbstractMatrix{Float64}, M::Real; workers::Int, xtile::Int = 64,
        use_profiled_correction::Bool = false)
    workers <= nthreads() || error("winner_pair_cross_hessian_zc_block_drawmajor_v2!: workers=$workers exceeds Threads.nthreads()=$(nthreads())")
    workers == dm.workers || error("winner_pair_cross_hessian_zc_block_drawmajor_v2!: workers=$workers != scratch's dm.workers=$(dm.workers) -- rebuild scratch")
    W = wctx.W; Ddest = wctx.Ddest
    ncolI = wctx.ncolI
    nx = size(Z, 2)
    size(HEZ) == (ncolI + 1, nx) || error("winner_pair_cross_hessian_zc_block_drawmajor_v2!: size(HEZ)=$(size(HEZ)) != ($(ncolI + 1), $nx)")
    length(S) == W || error("winner_pair_cross_hessian_zc_block_drawmajor_v2!: length(S)=$(length(S)) != wctx.W=$W")
    size(Z, 1) == W || error("winner_pair_cross_hessian_zc_block_drawmajor_v2!: size(Z,1)=$(size(Z, 1)) != wctx.W=$W")
    nx <= dm.max_nx || error("winner_pair_cross_hessian_zc_block_drawmajor_v2!: nx=$nx exceeds dm.max_nx=$(dm.max_nx) -- rebuild scratch")
    use_profiled_correction && ws.Ddest != Ddest &&
        error("winner_pair_cross_hessian_zc_block_drawmajor_v2!: use_profiled_correction=true requires ws.Ddest=$(ws.Ddest) == wctx.Ddest=$Ddest -- rebuild scratch via ensure_winner_zc_cross_scratch!(...; Ddest)")

    winner = wctx.winner; pi_vec = wctx.pi_vec; y = wctx.y; target_slot = wctx.target_slot; Lam_homog = wctx.Lam_homog
    has_cf = wctx.has_cf; jcf = ncolI
    Snu = ws.Snu
    invM = 1.0 / M

    row1 = @view HEZ[1, :]
    BLAS.gemv!('T', invM, Z, S, 0.0, row1)

    nbilateral = has_cf ? ncolI - 1 : ncolI
    nbilateral == dm.nbilateral || error("winner_pair_cross_hessian_zc_block_drawmajor_v2!: nbilateral=$nbilateral != dm.nbilateral=$(dm.nbilateral) -- rebuild scratch")
    bilateral_block = @view HEZ[2:1+nbilateral, :]
    fill!(bilateral_block, 0.0)

    local_tabs = @view dm.local_tabs[:, 1:nx, :]
    fill!(local_tabs, 0.0)
    w_chunks = cross_hessian_chunk_ranges(W, workers)
    tasks = dm.tasks
    for wk in 1:workers
        wr = w_chunks[wk]
        clen = length(wr)
        rl_full = @view dm.rowidx_local[:, :, wk]
        vl_full = @view dm.v_local[:, :, wk]
        lt = @view dm.local_tabs[:, :, wk]
        tasks[wk] = Threads.@spawn begin
            # One-time per-callback transpose of this worker's OWN draw chunk (parallel across
            # workers, since each only touches its own scratch slice): winner[w,slot]/Snu[w]*y[w,slot]
            # -> rl[slot,iw]/vl[slot,iw], so slot varies contiguously for fixed local draw iw.
            @inbounds for (iw, w) in enumerate(wr)
                snuw = Snu[w]
                for slot in 1:Ddest
                    o = winner[w, slot]
                    rl_full[slot, iw] = slot + (o - 1) * Ddest
                    vl_full[slot, iw] = snuw * y[w, slot]
                end
            end
            rl = @view rl_full[:, 1:clen]
            vl = @view vl_full[:, 1:clen]
            @inbounds for xb in 1:xtile:nx
                xe = min(xb + xtile - 1, nx)
                for x in xb:xe
                    Zx = @view Z[:, x]
                    for iw in 1:clen
                        w = wr[iw]
                        zxw = Zx[w]
                        for slot in 1:Ddest
                            j = rl[slot, iw]
                            lt[j, x] += vl[slot, iw] * zxw
                        end
                    end
                end
            end
        end
    end
    for wk in 1:workers
        fetch(tasks[wk])
    end

    @inbounds for wk in 1:workers
        for x in 1:nx, j in 1:nbilateral
            HEZ[j+1, x] += dm.local_tabs[j, x, wk]
        end
    end

    NuZ = @view ws.NuZ_buf[1:nx]
    BLAS.gemv!('T', 1.0, Z, Snu, 0.0, NuZ)

    # Profiled economic block port (2026-08-02, mirrors winner_pair_cross_hessian_zc_block!'s own
    # use_profiled_correction=true branch exactly): TZ[d,x] = Σ_w SnuWval[w,d]*Z[w,x], ONE
    # BLAS.gemm! call, only computed when actually needed -- see
    # PROFILED_CROSS_BLOCK_FORMULAS_2026-08-01.md §3. Not a second W-scale draw-major pass: this
    # gemm operates on the already-filled (W x Ddest) SnuWval built once per callback by
    # winner_pair_cross_hessian_zc_prep!, shared with the serial/threaded kernels.
    TZ = nothing
    if use_profiled_correction
        TZ = @view ws.TZ_buf[:, 1:nx]
        BLAS.gemm!('T', 'N', 1.0, ws.SnuWval, Z, 0.0, TZ)
    end

    @inbounds for j in 1:nbilateral
        if use_profiled_correction
            d = target_slot[j]
            lamj = Lam_homog[j]
            for x in 1:nx
                HEZ[j+1, x] = invM * (HEZ[j+1, x] - lamj * TZ[d, x])
            end
        else
            pij = pi_vec[j]
            for x in 1:nx
                HEZ[j+1, x] = invM * (HEZ[j+1, x] - pij * NuZ[x])
            end
        end
    end

    if has_cf
        row_cf = @view HEZ[jcf+1, :]
        BLAS.gemv!('T', invM, Z, ws.crs_buf, 0.0, row_cf)
        if use_profiled_correction && target_slot[jcf] != 0
            d_cf = target_slot[jcf]
            lamcf = Lam_homog[jcf]
            dcfscaled = wctx.denom_cf_scaled
            @inbounds for x in 1:nx
                row_cf[x] += invM * (dcfscaled * NuZ[x] - lamcf * TZ[d_cf, x])
            end
        else
            pij = pi_vec[jcf]
            @inbounds for x in 1:nx
                row_cf[x] -= invM * pij * NuZ[x]
            end
        end
    end

    return HEZ
end
