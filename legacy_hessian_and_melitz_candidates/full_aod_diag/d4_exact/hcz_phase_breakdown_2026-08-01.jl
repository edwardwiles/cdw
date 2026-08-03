# Genuine-cold ZC Hessian K=3 closeout task (2026-08-01), Section 5: H_CZ phase-level timing
# breakdown + a surgical parallel-reduction test for the reordered candidate's serial reduce step.
#
# Instruments `bin_zc_cross_hessian_fill_drawchunk_reordered!` (hcz_reordered_candidate_2026-08-01.jl)
# into 3 timed phases without changing its numerics: (1) local-table fill -- the threaded
# per-worker accumulation into `local_tabs`, including the fetch/join; (2) local-table reduction --
# the serial `for wk, for b,j,x: ZBinTab += local_tabs` sum; (3) cumulative-bin construction -- the
# `ZBinCScum` cumsum pass. ("Assembly and packing" from the task brief's own phase list has no
# separate step inside this specific prep function -- it fills `ws.ZBinTab`/`ws.ZBinCScum` only;
# any further packed-Hessian assembly happens in a different, calling function and is out of scope
# for this isolated H_CZ-prep timing.)

"Timed clone of `bin_zc_cross_hessian_fill_drawchunk_reordered!` -- same numerics, returns (ws, phase_times::NamedTuple)."
function bin_zc_cross_hessian_fill_drawchunk_reordered_timed!(ws::BinZCrossScratch, dc::BinZCrossDrawChunkScratch,
        Bidx::AbstractMatrix{<:Integer}, ZcS::AbstractMatrix{Float64}; workers::Int, jtile::Int = 64)
    D = ws.D; L = ws.L; nz = ws.nz
    W = size(ZcS, 1)

    local_tabs = dc.local_tabs
    fill!(local_tabs, 0.0)
    w_chunks = cross_hessian_chunk_ranges(W, workers)
    tasks = dc.tasks

    t0 = time_ns()
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
    t_fill = (time_ns() - t0) / 1e9

    t1 = time_ns()
    ZBinTab = ws.ZBinTab
    fill!(ZBinTab, 0.0)
    @inbounds for wk in 1:workers
        for b in 1:(L+1), j in 1:nz, x in 1:D
            ZBinTab[x, j, b] += local_tabs[x, j, b, wk]
        end
    end
    t_reduce = (time_ns() - t1) / 1e9

    t2 = time_ns()
    ZBinCScum = ws.ZBinCScum
    @inbounds for x in 1:D, j in 1:nz
        acc = 0.0
        for l in 1:L
            acc += ZBinTab[x, j, l]
            ZBinCScum[x, j, l] = acc
        end
    end
    t_cumsum = (time_ns() - t2) / 1e9

    return ws, (fill_and_join_s = t_fill, reduction_s = t_reduce, cumsum_s = t_cumsum,
                total_s = t_fill + t_reduce + t_cumsum)
end

"""
    bin_zc_cross_hessian_fill_drawchunk_reordered_parreduce!(ws, dc, Bidx, ZcS; workers, jtile=64, reduce_workers=workers) -> (ws, phase_times)

Same as the timed clone above, EXCEPT the local-table reduction is itself parallelized over
disjoint `j` (nz-column) ranges across `reduce_workers` tasks -- each task owns a disjoint slice
of `ZBinTab[:, jrange, :]` and sums every worker's `local_tabs[:, jrange, :, wk]` into it, so no two
tasks ever write the same output cell (safe without atomics/locks). Only worth using if the serial
reduction phase above is shown to be material relative to the fill phase.
"""
function bin_zc_cross_hessian_fill_drawchunk_reordered_parreduce!(ws::BinZCrossScratch, dc::BinZCrossDrawChunkScratch,
        Bidx::AbstractMatrix{<:Integer}, ZcS::AbstractMatrix{Float64}; workers::Int, jtile::Int = 64,
        reduce_workers::Int = workers)
    D = ws.D; L = ws.L; nz = ws.nz
    W = size(ZcS, 1)

    local_tabs = dc.local_tabs
    fill!(local_tabs, 0.0)
    w_chunks = cross_hessian_chunk_ranges(W, workers)
    tasks = dc.tasks

    t0 = time_ns()
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
    t_fill = (time_ns() - t0) / 1e9

    t1 = time_ns()
    ZBinTab = ws.ZBinTab
    fill!(ZBinTab, 0.0)
    j_chunks = cross_hessian_chunk_ranges(nz, reduce_workers)
    rtasks = Vector{Task}(undef, reduce_workers)
    for rk in 1:reduce_workers
        jr = j_chunks[rk]
        rtasks[rk] = Threads.@spawn begin
            @inbounds for wk in 1:workers, b in 1:(L+1), j in jr, x in 1:D
                ZBinTab[x, j, b] += local_tabs[x, j, b, wk]
            end
        end
    end
    for rk in 1:reduce_workers
        fetch(rtasks[rk])
    end
    t_reduce = (time_ns() - t1) / 1e9

    t2 = time_ns()
    ZBinCScum = ws.ZBinCScum
    @inbounds for x in 1:D, j in 1:nz
        acc = 0.0
        for l in 1:L
            acc += ZBinTab[x, j, l]
            ZBinCScum[x, j, l] = acc
        end
    end
    t_cumsum = (time_ns() - t2) / 1e9

    return ws, (fill_and_join_s = t_fill, reduction_s = t_reduce, cumsum_s = t_cumsum,
                total_s = t_fill + t_reduce + t_cumsum)
end
