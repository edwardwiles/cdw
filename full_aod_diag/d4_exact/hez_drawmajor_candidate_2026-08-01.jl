# Genuine-cold ZC Hessian K=3 optimization task (2026-08-01), Section 12: H_EZ surgical candidate.
#
# The current production backend (`winner_pair_cross_hessian_zc_block_threaded!`,
# threaded_cross_hessian.jl) threads by DESTINATION SLOT: each worker owns a disjoint set of
# `Ddest` slots, and for EACH of its slots scans the FULL `Z` (`W x nx`) once per feature column
# `x` (`for x in 1:nx: Zx = view(Z,:,x); for w in 1:W: ... Zx[w] ...`). Since slots partition
# `Ddest`, this means `Z` (at nx=630, W=100,000: ~504MB, far larger than any cache) gets scanned in
# full `Ddest` times in total across all workers -- exactly the inefficiency the task brief's
# "draw-major" suggestion targets.
#
# This candidate instead partitions DRAWS (`w`) across workers. For a fixed draw `w` and feature
# `x`, the value `Z[w,x]` contributes to ALL `Ddest` destination rows at once (one per slot, each
# with its own weight `v[w,slot] = Snu[w]*y[w,slot]` and target row `j(w,slot) =
# slot+(winner[w,slot]-1)*Ddest`) -- so `Z[w,x]` needs to be read only ONCE per (w,x) pair instead
# of `Ddest` times. `v` (`W x Ddest`) is precomputed ONCE per callback (O(W*Ddest), cheap) so the
# main loop never recomputes `Snu[w]*y[w,slot]` inside the (x,w) double loop. The main loop keeps
# `x` as the tiled OUTER dimension (so `Z[:,x]` stays a contiguous column view) with `w` as the
# INNER dimension over each worker's disjoint draw chunk -- same cache-friendly-access rationale as
# the H_CZ reordered candidate (`hcz_reordered_candidate_2026-08-01.jl`).
#
# Total FLOP count is UNCHANGED (same O(W*Ddest*nx) accumulation) -- this is a memory-traffic
# reduction (up to Ddest-fold fewer reads of Z), not an arithmetic reduction, exactly matching the
# task brief's own framing. Correctness: same target cells, tolerance-level (not bit-exact) vs the
# slot-major reference since the summation order differs.

"""
    WinnerZCDrawMajorScratch

Persistent scratch for the draw-major H_EZ candidate. `v` (`W x Ddest`) holds `Snu.*y`,
recomputed once per callback via `precompute_winner_zc_drawmajor_v!`. `local_tabs`
(`nbilateral x max_nx x workers`) accumulates each worker's disjoint draw-chunk contribution
before a final serial reduction into `HEZ`.
"""
mutable struct WinnerZCDrawMajorScratch
    W::Int
    Ddest::Int
    nbilateral::Int
    max_nx::Int
    workers::Int
    v::Matrix{Float64}
    local_tabs::Array{Float64,3}
    tasks::Vector{Task}
end
function WinnerZCDrawMajorScratch(W::Int, Ddest::Int, nbilateral::Int, max_nx::Int, workers::Int)
    WinnerZCDrawMajorScratch(W, Ddest, nbilateral, max_nx, workers,
        zeros(W, Ddest), zeros(nbilateral, max_nx, workers), Vector{Task}(undef, workers))
end
function ensure_winner_zc_drawmajor_scratch!(dm::Union{Nothing,WinnerZCDrawMajorScratch},
        W::Int, Ddest::Int, nbilateral::Int, max_nx::Int, workers::Int)
    if dm === nothing || dm.W != W || dm.Ddest != Ddest || dm.nbilateral != nbilateral ||
       dm.max_nx != max_nx || dm.workers != workers
        return WinnerZCDrawMajorScratch(W, Ddest, nbilateral, max_nx, workers)
    end
    return dm
end

"`v[w,slot] = Snu[w]*y[w,slot]` for every (w,slot) -- O(W*Ddest), computed once per callback so the main (x,w) loop below never recomputes it."
function precompute_winner_zc_drawmajor_v!(dm::WinnerZCDrawMajorScratch, wctx::WinnerPairHessCtx, Snu::AbstractVector{Float64})
    W = wctx.W; Ddest = wctx.Ddest; y = wctx.y
    v = dm.v
    @inbounds for slot in 1:Ddest
        for w in 1:W
            v[w, slot] = Snu[w] * y[w, slot]
        end
    end
    return dm
end

"""
    winner_pair_cross_hessian_zc_block_drawmajor!(HEZ, wctx, ws, dm, S, Z, M; workers, xtile=64) -> HEZ

Draw-major candidate for H_EZ. Same public contract/output as
`winner_pair_cross_hessian_zc_block!`/`_threaded!` -- requires `winner_pair_cross_hessian_zc_prep!`
already called this callback (for `ws.Snu`/`ws.crs_buf`), same as its siblings.
"""
function winner_pair_cross_hessian_zc_block_drawmajor!(HEZ::AbstractMatrix{Float64}, wctx::WinnerPairHessCtx,
        ws::WinnerZCCrossScratch, dm::WinnerZCDrawMajorScratch, S::AbstractVector{Float64},
        Z::AbstractMatrix{Float64}, M::Real; workers::Int, xtile::Int = 64)
    workers <= nthreads() || error("winner_pair_cross_hessian_zc_block_drawmajor!: workers=$workers exceeds Threads.nthreads()=$(nthreads())")
    workers == dm.workers || error("winner_pair_cross_hessian_zc_block_drawmajor!: workers=$workers != scratch's dm.workers=$(dm.workers) -- rebuild scratch")
    W = wctx.W; Ddest = wctx.Ddest
    ncolI = wctx.ncolI
    nx = size(Z, 2)
    size(HEZ) == (ncolI + 1, nx) || error("winner_pair_cross_hessian_zc_block_drawmajor!: size(HEZ)=$(size(HEZ)) != ($(ncolI + 1), $nx)")
    length(S) == W || error("winner_pair_cross_hessian_zc_block_drawmajor!: length(S)=$(length(S)) != wctx.W=$W")
    size(Z, 1) == W || error("winner_pair_cross_hessian_zc_block_drawmajor!: size(Z,1)=$(size(Z, 1)) != wctx.W=$W")
    nx <= dm.max_nx || error("winner_pair_cross_hessian_zc_block_drawmajor!: nx=$nx exceeds dm.max_nx=$(dm.max_nx) -- rebuild scratch")

    winner = wctx.winner; pi_vec = wctx.pi_vec
    has_cf = wctx.has_cf; jcf = ncolI
    Snu = ws.Snu
    invM = 1.0 / M

    row1 = @view HEZ[1, :]
    BLAS.gemv!('T', invM, Z, S, 0.0, row1)

    nbilateral = has_cf ? ncolI - 1 : ncolI
    nbilateral == dm.nbilateral || error("winner_pair_cross_hessian_zc_block_drawmajor!: nbilateral=$nbilateral != dm.nbilateral=$(dm.nbilateral) -- rebuild scratch")
    bilateral_block = @view HEZ[2:1+nbilateral, :]
    fill!(bilateral_block, 0.0)

    precompute_winner_zc_drawmajor_v!(dm, wctx, Snu)
    v = dm.v

    local_tabs = @view dm.local_tabs[:, 1:nx, :]
    fill!(local_tabs, 0.0)
    w_chunks = cross_hessian_chunk_ranges(W, workers)
    tasks = dm.tasks
    for wk in 1:workers
        wr = w_chunks[wk]
        lt = @view dm.local_tabs[:, :, wk]
        tasks[wk] = Threads.@spawn begin
            @inbounds for xb in 1:xtile:nx
                xe = min(xb + xtile - 1, nx)
                for x in xb:xe
                    Zx = @view Z[:, x]
                    for w in wr
                        zxw = Zx[w]
                        for slot in 1:Ddest
                            o = winner[w, slot]
                            j = slot + (o - 1) * Ddest
                            lt[j, x] += v[w, slot] * zxw
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

    @inbounds for j in 1:nbilateral
        pij = pi_vec[j]
        for x in 1:nx
            HEZ[j+1, x] = invM * (HEZ[j+1, x] - pij * NuZ[x])
        end
    end

    if has_cf
        row_cf = @view HEZ[jcf+1, :]
        BLAS.gemv!('T', invM, Z, ws.crs_buf, 0.0, row_cf)
        pij = pi_vec[jcf]
        @inbounds for x in 1:nx
            row_cf[x] -= invM * pij * NuZ[x]
        end
    end

    return HEZ
end

"""
    ZC_EZ_BACKEND_DEFAULT

`:winner_bin` (former production default, dispatched on `cctx.cross_hessian_threaded`) |
`:drawmajor` (this file's candidate, v1 -- superseded, see below) |
`:drawmajor_v2` (`hez_drawmajor_v2_candidate_2026-08-01.jl` -- fixes v1's redundant per-x strided
winner/v re-read via a per-worker one-time transpose; beats reference at EVERY worker count
including workers=1, unlike v1 which was slower than reference there).

Flipped to `:drawmajor_v2` (2026-08-01, ZC Hessian backend production integration): 2.31x faster
than `:winner_bin` in isolation at workers=10 (production width), `~1.5e-10` absolute correctness
(scale ~2112), complete-packed-Hessian ALL PASS for BOTH cm_meanzc (16/16) and origin-ZC (3/3,
previously blocked by an unrelated harness bug, now resolved -- see
`ORIGIN_ZC_HARNESS_ROOT_CAUSE_2026-08-01.md`), and part of a validated 22-44% real single-inner-
solve speedup at W=100,000/500,000 for both families (genuine-cold, JIT-warm, `julia -t 10`,
matched against an independent cross-session baseline) -- see
`ZC_HESSIAN_BACKEND_CLOSEOUT_MASTER_2026-08-01.md`. `:winner_bin` remains available as a selectable
diagnostic fallback (unchanged behavior, not removed); `:drawmajor` (v1) remains available too but
is superseded by v2 for all practical purposes.
"""
const ZC_EZ_BACKEND_DEFAULT = Ref{Symbol}(:drawmajor_v2)
const HEZ_DRAWMAJOR_XTILE_DEFAULT = Ref{Int}(64)
