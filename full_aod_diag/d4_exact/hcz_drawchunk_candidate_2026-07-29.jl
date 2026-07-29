# Part C, Candidate 2 — draw-chunk thread-local H_CZ prep (2026-07-29).
#
# Alternative to bin_zc_cross_hessian_fill_threaded! (threaded_cross_hessian.jl), which chunks the
# outer loop by ORIGIN (x): every worker still scans all W draws for its owned origins, so the
# single largest array in this computation (ZcS, W x n_z) gets streamed through cache `workers`-fold
# redundantly (see docs/PART_A_LABEL_MAPPING_2026-07-29.md / PART_C_HCZ_FORMULA_2026-07-29.md for
# the full derivation). This candidate instead chunks the W draws across workers: each worker reads
# its OWN disjoint slice of ZcS exactly once, accumulating into a persistent thread-local D x n_z x
# (L+1) table (no cross-worker writes -- disjoint by construction), then a final serial reduction
# sums the per-worker tables into ws.ZBinTab.
#
# NOT bit-identical to the serial/origin-chunked reference: floating-point addition is not
# associative, and this candidate sums in a different order (per-chunk partial sums, then combined)
# than the reference's strict w=1:W accumulation. Correctness gates below compare via a relative
# tolerance (matching this file's own `HCZ_CANDIDATE_TOL`), not exact equality.

"""
    BinZCrossDrawChunkScratch

Persistent per-worker `D x n_z x (L+1)` accumulator tables (`local_tabs[:, :, :, wk]`, disjoint
across workers -- no atomics) plus the `Threads.@spawn` task buffer, sized once per `(D, L, nz,
workers)` combination.
"""
mutable struct BinZCrossDrawChunkScratch
    D::Int
    L::Int
    nz::Int
    workers::Int
    local_tabs::Array{Float64,4}
    tasks::Vector{Task}
end
function BinZCrossDrawChunkScratch(D::Int, L::Int, nz::Int, workers::Int)
    BinZCrossDrawChunkScratch(D, L, nz, workers, zeros(D, nz, L + 1, workers), Vector{Task}(undef, workers))
end
function ensure_bin_zc_drawchunk_scratch!(dc::Union{Nothing,BinZCrossDrawChunkScratch}, D::Int, L::Int, nz::Int, workers::Int)
    if dc === nothing || dc.D != D || dc.L != L || dc.nz != nz || dc.workers != workers
        return BinZCrossDrawChunkScratch(D, L, nz, workers)
    end
    return dc
end

const HCZ_CANDIDATE_TOL = 1e-9   # relative tolerance for non-bit-identical reduction-order candidates

"""
    bin_zc_cross_hessian_fill_drawchunk!(ws, dc, Bidx, ZcS; workers) -> ws

Draw-chunked drop-in replacement for `bin_zc_cross_hessian_fill!`/`_threaded!`. Fills the SAME
`ws.ZBinTab`/`ws.ZBinCScum` outputs (`winner_pair_cross_hessian.jl`'s `BinZCrossScratch`) via
per-worker draw-chunk ownership instead of origin ownership.
"""
function bin_zc_cross_hessian_fill_drawchunk!(ws::BinZCrossScratch, dc::BinZCrossDrawChunkScratch,
        Bidx::AbstractMatrix{<:Integer}, ZcS::AbstractMatrix{Float64}; workers::Int)
    workers <= nthreads() || error("bin_zc_cross_hessian_fill_drawchunk!: workers=$workers exceeds Threads.nthreads()=$(nthreads())")
    workers == dc.workers || error("bin_zc_cross_hessian_fill_drawchunk!: workers=$workers != scratch's dc.workers=$(dc.workers) -- rebuild scratch")
    D = ws.D; L = ws.L; nz = ws.nz
    W = size(ZcS, 1)
    size(Bidx, 1) == W || error("bin_zc_cross_hessian_fill_drawchunk!: size(Bidx,1)=$(size(Bidx,1)) != size(ZcS,1)=$W")
    size(ZcS, 2) >= nz || error("bin_zc_cross_hessian_fill_drawchunk!: size(ZcS,2)=$(size(ZcS,2)) < ws.nz=$nz")

    local_tabs = dc.local_tabs
    fill!(local_tabs, 0.0)
    w_chunks = cross_hessian_chunk_ranges(W, workers)
    tasks = dc.tasks
    for wk in 1:workers
        wr = w_chunks[wk]
        tasks[wk] = Threads.@spawn begin
            @inbounds for w in wr
                for x in 1:D
                    b = Bidx[w, x]
                    for j in 1:nz
                        local_tabs[x, j, b, wk] += ZcS[w, j]
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
