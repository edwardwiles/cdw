# Genuine-cold ZC Hessian K=3 optimization task (2026-08-01), Section 11: H_CZ surgical candidate.
#
# The current default (`:draw_chunk_thread_local`, hcz_drawchunk_candidate_2026-07-29.jl) fills
# `local_tabs[x, j, b, wk] += ZcS[w, j]` with loop order w (outer, chunked) -> x -> j (inner).
# `ZcS` is `(W, nx)` column-major (`zc_restriction_operator.jl`'s `ZCCenteredScratch.ZcS =
# zeros(W, max_nx)`) -- for FIXED w, varying j, `ZcS[w, j]` has STRIDE W (at W=100,000, each step
# jumps 800KB, far larger than any cache, i.e. this access pattern is close to a cache miss on
# every single element at nx=630's inner loop). This candidate reorders the SAME accumulation
# (same targets, same total FLOPs, same reduction structure) to x (outer, per worker's draw chunk)
# -> j (tiled) -> w (inner), so both `ZcS[:, j]` and `Bidx[:, x]` are read via CONTIGUOUS column
# views inside the innermost `w` loop -- trading the bad ZcS access for a scatter-write into
# `local_tabs[x, j, :, wk]` across the b-dimension (stride D*nz elements per b, ~50 distinct
# targets spanning ~5MB -- L2-hostile but far cheaper than a stride-W main-memory-class read).
#
# Correctness: bit-identical target cells (`local_tabs[x,j,b,wk]`), same per-worker draw-chunk
# partition, same summation ORDER within a chunk is NOT preserved (w-then-x-then-j reference vs
# x-then-j-then-w here) -- floating point addition is not associative, so this is a
# tolerance-level (not bit-exact) candidate, same `HCZ_CANDIDATE_TOL` discipline as its sibling.

"""
    bin_zc_cross_hessian_fill_drawchunk_reordered!(ws, dc, Bidx, ZcS; workers, jtile=64) -> ws

Cache-access-pattern candidate for H_CZ prep. Same public contract as
`bin_zc_cross_hessian_fill_drawchunk!` (fills `ws.ZBinTab`/`ws.ZBinCScum`), reuses the SAME
`BinZCrossDrawChunkScratch` (no new scratch type). `jtile` bounds how many `nz` columns are
processed together before moving to the next batch of origins `x` -- purely a cache-tuning knob,
does not change the result (only benchmarked, not gated, across a couple of `jtile` values).
"""
function bin_zc_cross_hessian_fill_drawchunk_reordered!(ws::BinZCrossScratch, dc::BinZCrossDrawChunkScratch,
        Bidx::AbstractMatrix{<:Integer}, ZcS::AbstractMatrix{Float64}; workers::Int, jtile::Int = 64)
    workers <= nthreads() || error("bin_zc_cross_hessian_fill_drawchunk_reordered!: workers=$workers exceeds Threads.nthreads()=$(nthreads())")
    workers == dc.workers || error("bin_zc_cross_hessian_fill_drawchunk_reordered!: workers=$workers != scratch's dc.workers=$(dc.workers) -- rebuild scratch")
    D = ws.D; L = ws.L; nz = ws.nz
    W = size(ZcS, 1)
    size(Bidx, 1) == W || error("bin_zc_cross_hessian_fill_drawchunk_reordered!: size(Bidx,1)=$(size(Bidx,1)) != size(ZcS,1)=$W")
    size(ZcS, 2) >= nz || error("bin_zc_cross_hessian_fill_drawchunk_reordered!: size(ZcS,2)=$(size(ZcS,2)) < ws.nz=$nz")

    local_tabs = dc.local_tabs
    fill!(local_tabs, 0.0)
    w_chunks = cross_hessian_chunk_ranges(W, workers)
    tasks = dc.tasks
    for wk in 1:workers
        wr = w_chunks[wk]
        lt = @view local_tabs[:, :, :, wk]
        tasks[wk] = Threads.@spawn begin
            @inbounds for x in 1:D
                bx = @view Bidx[:, x]
                for jb in 1:jtile:nz
                    je = min(jb + jtile - 1, nz)
                    for j in jb:je
                        zcol = @view ZcS[:, j]
                        for w in wr
                            b = bx[w]
                            lt[x, j, b] += zcol[w]
                        end
                    end
                end
            end
        end
    end
    for wk in 1:workers
        fetch(tasks[wk])
    end

    ZBinTab = ws.ZBinTab
    fill!(ZBinTab, 0.0)
    @inbounds for wk in 1:workers
        for b in 1:(L+1), j in 1:nz, x in 1:D
            ZBinTab[x, j, b] += local_tabs[x, j, b, wk]
        end
    end

    ZBinCScum = ws.ZBinCScum
    @inbounds for x in 1:D, j in 1:nz
        acc = 0.0
        for l in 1:L
            acc += ZBinTab[x, j, l]
            ZBinCScum[x, j, l] = acc
        end
    end
    return ws
end
