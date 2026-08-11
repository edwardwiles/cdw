# H_CZ prep candidate: BIN-MAJOR thread-local accumulator (2026-08-11).
#
# MOTIVATION (measured, not guessed). The 2026-08-11 outer-eval profile
# (profile_cross_outer_eval_2026-08-11.jl) found that after the 2026-08-10 BLAS-thread gate fix,
# `H_CZ_prep` OVERTOOK `H_ZZ` as the largest block of the CM+ZC-CROSS Hessian callback:
#
#     H_CZ_prep  2.122 s/call  (33.9% of the callback)     <-- now #1
#     H_ZZ       1.745 s/call  (27.9%)                     <-- was 7.524 s at 1 BLAS thread
#
# `bin_zc_cross_hessian_fill_drawchunk_reordered!` (hcz_reordered_candidate_2026-08-01.jl) already
# fixed the READ side: its `x -> j -> w` order makes `ZcS[:, j]` a contiguous column read, replacing
# the stride-W (800KB/step) row access of the older `:draw_chunk_thread_local`.
#
# But the WRITE side is still hostile, and that file's own docstring says so:
#     "trading the bad ZcS access for a scatter-write into `local_tabs[x, j, :, wk]` across the
#      b-dimension (stride D*nz elements per b, ~50 distinct targets spanning ~5MB -- L2-hostile)"
# At the real CM+ZC-CROSS K=3/3 shape (D=20, nz=1770, L=50) the b-stride is D*nz = 35,400 elements
# = 283 KB, and the L+1 = 51 live targets span 14.4 MB -- L3-class, so essentially every one of the
# D*nz*W = 3.5e9 accumulates can miss L2.
#
# THIS CANDIDATE: permute the thread-local accumulator to `(L+1, D, nz, workers)` -- bin index
# FASTEST -- so that for a fixed `(x, j)` the 51 live targets are 51 CONTIGUOUS Float64 = 408 bytes
# = 7 cache lines, resident in L1 for the whole inner `w` loop (10,000 iterations per worker chunk).
# Everything else is unchanged: same loop order, same per-worker draw partition, same arithmetic.
#
# CORRECTNESS: this is a pure memory-layout permutation. The accumulation ORDER into each logical
# cell (x, j, b) is IDENTICAL to `_reordered!` (same w sequence within the same worker chunk), and
# the cross-worker reduction is done in the same `wk` order, so the result must be **BIT-IDENTICAL**
# -- not merely within HCZ_CANDIDATE_TOL. That is a far stronger gate than the tolerance-level one
# its sibling candidates need, and the test asserts exactly that.

isdefined(Main, :BinZCrossDrawChunkScratch) || include(joinpath(@__DIR__, "hcz_drawchunk_candidate_2026-07-29.jl"))

"""
    BinZCrossBTransposeScratch(D, L, nz, workers)

Thread-local accumulator for `bin_zc_cross_hessian_fill_drawchunk_btranspose!`, dimensioned
`(L+1, D, nz, workers)` -- the SAME total element count as `BinZCrossDrawChunkScratch`'s
`(D, nz, L+1, workers)`, only permuted so the bin index is fastest-varying. Separate type (rather
than reusing the sibling's) because the two layouts are not interchangeable and silently handing one
kernel the other's buffer would produce wrong answers rather than an error.
"""
mutable struct BinZCrossBTransposeScratch
    D::Int
    L::Int
    nz::Int
    workers::Int
    local_tabs::Array{Float64,4}       # (L+1, D, nz, workers)
    local_tabs_pow::Array{Float64,4}
    tasks::Vector{Task}
end
function BinZCrossBTransposeScratch(D::Int, L::Int, nz::Int, workers::Int)
    BinZCrossBTransposeScratch(D, L, nz, workers, zeros(L + 1, D, nz, workers), zeros(L + 1, D, nz, workers),
                               Vector{Task}(undef, workers))
end
function ensure_bin_zc_btranspose_scratch!(dcb::Union{Nothing,BinZCrossBTransposeScratch}, D::Int, L::Int, nz::Int, workers::Int)
    if dcb === nothing || dcb.D != D || dcb.L != L || dcb.nz != nz || dcb.workers != workers
        return BinZCrossBTransposeScratch(D, L, nz, workers)
    end
    return dcb
end

"""
    BIN_ZC_BTRANSPOSE_SCRATCH

Campaign-lifetime scratch slot for the `:draw_chunk_btranspose` dispatch arm. Deliberately a
module-level slot rather than a new `CMBinHessCtx` field: this is a *candidate* backend, and adding
a field to that struct means touching its (long, positional) constructor and every call site, which
is not warranted until the candidate is adopted. Rebuilt only when `(D, L, nz, workers)` changes,
i.e. once per campaign in practice -- the same reuse discipline the sibling scratches get from
their `cctx` fields. Written only from `hcz_prep_dispatch!`, which is called from the single thread
that then spawns this kernel's workers.
"""
const BIN_ZC_BTRANSPOSE_SCRATCH = Ref{Union{Nothing,BinZCrossBTransposeScratch}}(nothing)

"""
    bin_zc_cross_hessian_fill_drawchunk_btranspose!(ws, dcb, Bidx, ZcS; workers, jtile=64, Pow=nothing) -> ws

Bin-major analog of `bin_zc_cross_hessian_fill_drawchunk_reordered!`. Same public contract (fills
`ws.ZBinTab`/`ws.ZBinCScum`, and `_pow` when `Pow !== nothing`), same loop order, same partition --
the ONLY difference is that the thread-local accumulator is indexed `[b, x, j]` instead of
`[x, j, b]`. Result is bit-identical to `_reordered!`.
"""
function bin_zc_cross_hessian_fill_drawchunk_btranspose!(ws::BinZCrossScratch, dcb::BinZCrossBTransposeScratch,
        Bidx::AbstractMatrix{<:Integer}, ZcS::AbstractMatrix{Float64}; workers::Int, jtile::Int = 64,
        Pow::Union{Nothing,AbstractMatrix{Float64}} = nothing)
    workers <= nthreads() || error("bin_zc_cross_hessian_fill_drawchunk_btranspose!: workers=$workers exceeds Threads.nthreads()=$(nthreads())")
    workers == dcb.workers || error("bin_zc_cross_hessian_fill_drawchunk_btranspose!: workers=$workers != scratch's dcb.workers=$(dcb.workers) -- rebuild scratch")
    D = ws.D; L = ws.L; nz = ws.nz
    W = size(ZcS, 1)
    size(Bidx, 1) == W || error("bin_zc_cross_hessian_fill_drawchunk_btranspose!: size(Bidx,1)=$(size(Bidx,1)) != size(ZcS,1)=$W")
    size(ZcS, 2) >= nz || error("bin_zc_cross_hessian_fill_drawchunk_btranspose!: size(ZcS,2)=$(size(ZcS,2)) < ws.nz=$nz")
    fam2 = Pow !== nothing

    local_tabs = dcb.local_tabs
    local_tabs_pow = dcb.local_tabs_pow
    fill!(local_tabs, 0.0)
    fam2 && fill!(local_tabs_pow, 0.0)
    w_chunks = cross_hessian_chunk_ranges(W, workers)
    tasks = dcb.tasks
    for wk in 1:workers
        wr = w_chunks[wk]
        lt = @view local_tabs[:, :, :, wk]              # (L+1, D, nz)
        lt_pow = fam2 ? (@view local_tabs_pow[:, :, :, wk]) : nothing
        tasks[wk] = Threads.@spawn begin
            @inbounds for x in 1:D
                bx = @view Bidx[:, x]
                powx = fam2 ? (@view Pow[:, x]) : nothing
                for jb in 1:jtile:nz
                    je = min(jb + jtile - 1, nz)
                    for j in jb:je
                        zcol = @view ZcS[:, j]
                        # lt[:, x, j] is a length-(L+1) CONTIGUOUS slice: 408 bytes, L1-resident
                        # for the whole w loop below. This is the entire point of the candidate.
                        for w in wr
                            b = bx[w]
                            lt[b, x, j] += zcol[w]
                            fam2 && (lt_pow[b, x, j] += zcol[w] * powx[w])
                        end
                    end
                end
            end
        end
    end
    for wk in 1:workers
        fetch(tasks[wk])
    end

    # Cross-worker reduction + transpose back into the public (D, nz, L+1) layout. `wk` OUTER, to
    # match _reordered!'s accumulation order exactly (bit-identity). Cost is
    # D*nz*(L+1)*workers ~ 1.8e7 element-ops against the inner loop's D*nz*W ~ 3.5e9 -- 0.5%.
    ZBinTab = ws.ZBinTab
    fill!(ZBinTab, 0.0)
    @inbounds for wk in 1:workers
        for j in 1:nz, x in 1:D, b in 1:(L+1)
            ZBinTab[x, j, b] += local_tabs[b, x, j, wk]
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

    if fam2
        ZBinTab_pow = ws.ZBinTab_pow
        fill!(ZBinTab_pow, 0.0)
        @inbounds for wk in 1:workers
            for j in 1:nz, x in 1:D, b in 1:(L+1)
                ZBinTab_pow[x, j, b] += local_tabs_pow[b, x, j, wk]
            end
        end
        ZBinCScum_pow = ws.ZBinCScum_pow
        @inbounds for x in 1:D, j in 1:nz
            acc = 0.0
            for l in 1:L
                acc += ZBinTab_pow[x, j, l]
                ZBinCScum_pow[x, j, l] = acc
            end
        end
    end
    return ws
end

# ================================================================================================
# SECOND candidate (2026-08-11): parallelise over RESTRICTION COLUMNS j instead of over draws w.
#
# The draw-parallel design above (inherited from every sibling backend) needs a per-worker
# accumulator because all workers accumulate into the same (x,j,b) cells, then a serial reduction.
# That costs (L+1)*D*nz*workers*8*2 bytes -- 275 MB at nz=1770/workers=10 -- and the reduction
# grows with worker count.
#
# But different j write to DISJOINT output cells. Partitioning over j therefore needs NO
# thread-local accumulator and NO cross-worker reduction at all; each worker owns a slice of the
# output outright. nz=1770 >> workers, so load balance is fine.
#
# It also fixes a SECOND cache problem the x-outer order has: with `for x, for j, for w`, the column
# ZcS[:,j] (781 KB) is re-streamed once per (x,j) PAIR, i.e. D*nz = 35,400 column reads = 26.4 GB of
# traffic per call. Ordering `for j, for x, for w` reads each column once and reuses it across all
# D=20 origins while it is still L2-resident: 1.32 GB, a 20x reduction.
#
# Per-(x,j) accumulation uses a length-(L+1) stack-local buffer (408 bytes, L1) and writes the 51
# results out to ZBinTab[x,j,:] once -- D*nz*(L+1) = 1.8e6 strided writes total, negligible against
# the 3.5e9 accumulates.
#
# NOT bit-identical to `_reordered!`: each cell is summed over ALL w in one sequential pass rather
# than per-worker chunks added in wk order. Floating-point addition is not associative, so this is a
# HCZ_CANDIDATE_TOL-level candidate, the same discipline its draw-parallel siblings use.
# ================================================================================================

"""
    bin_zc_cross_hessian_fill_jparallel!(ws, Bidx, ZcS; workers, Pow=nothing) -> ws

Column-parallel H_CZ prep. Same public contract as `bin_zc_cross_hessian_fill_drawchunk_reordered!`
(fills `ws.ZBinTab`/`ws.ZBinCScum`, plus `_pow` when `Pow !== nothing`), but partitions over `j`,
needs no thread-local accumulator, and reads each `ZcS` column once instead of `D` times.
"""
function bin_zc_cross_hessian_fill_jparallel!(ws::BinZCrossScratch,
        Bidx::AbstractMatrix{<:Integer}, ZcS::AbstractMatrix{Float64}; workers::Int,
        Pow::Union{Nothing,AbstractMatrix{Float64}} = nothing)
    workers <= nthreads() || error("bin_zc_cross_hessian_fill_jparallel!: workers=$workers exceeds Threads.nthreads()=$(nthreads())")
    D = ws.D; L = ws.L; nz = ws.nz
    W = size(ZcS, 1)
    size(Bidx, 1) == W || error("bin_zc_cross_hessian_fill_jparallel!: size(Bidx,1)=$(size(Bidx,1)) != size(ZcS,1)=$W")
    size(ZcS, 2) >= nz || error("bin_zc_cross_hessian_fill_jparallel!: size(ZcS,2)=$(size(ZcS,2)) < ws.nz=$nz")
    fam2 = Pow !== nothing
    ZBinTab = ws.ZBinTab
    ZBinTab_pow = ws.ZBinTab_pow

    j_chunks = cross_hessian_chunk_ranges(nz, workers)
    # This kernel ASSIGNS (`ZBinTab[x,j,b] = acc[b]`) rather than accumulating, precisely so it needs
    # no zero-fill and no cross-worker reduction -- but that makes complete coverage of 1:nz a
    # correctness requirement, not just an efficiency one: any uncovered j would silently retain the
    # PREVIOUS callback's values instead of erroring. cross_hessian_chunk_ranges tiles correctly
    # today (including workers > nz, where the surplus ranges are simply empty), so this is a guard
    # against a future change to it, checked once per call at O(workers) cost.
    sum(length, j_chunks) == nz ||
        error("bin_zc_cross_hessian_fill_jparallel!: j chunks cover $(sum(length, j_chunks)) columns, not nz=$nz -- " *
              "cross_hessian_chunk_ranges no longer tiles 1:nz, which would leave stale values in ZBinTab")
    tasks = Vector{Task}(undef, workers)
    for wk in 1:workers
        jr = j_chunks[wk]
        tasks[wk] = Threads.@spawn begin
            acc = zeros(Float64, L + 1)          # 408 B, L1-resident for the whole w loop
            acc_pow = fam2 ? zeros(Float64, L + 1) : Float64[]
            @inbounds for j in jr
                zcol = @view ZcS[:, j]           # read ONCE, reused across all D origins below
                for x in 1:D
                    bx = @view Bidx[:, x]
                    fill!(acc, 0.0); fam2 && fill!(acc_pow, 0.0)
                    if fam2
                        powx = @view Pow[:, x]
                        for w in 1:W
                            b = bx[w]; z = zcol[w]
                            acc[b] += z
                            acc_pow[b] += z * powx[w]
                        end
                    else
                        for w in 1:W
                            acc[bx[w]] += zcol[w]
                        end
                    end
                    for b in 1:(L+1)
                        ZBinTab[x, j, b] = acc[b]
                        fam2 && (ZBinTab_pow[x, j, b] = acc_pow[b])
                    end
                end
            end
        end
    end
    for wk in 1:workers
        fetch(tasks[wk])
    end

    ZBinCScum = ws.ZBinCScum
    @inbounds for x in 1:D, j in 1:nz
        acc = 0.0
        for l in 1:L
            acc += ZBinTab[x, j, l]
            ZBinCScum[x, j, l] = acc
        end
    end
    if fam2
        ZBinCScum_pow = ws.ZBinCScum_pow
        @inbounds for x in 1:D, j in 1:nz
            acc = 0.0
            for l in 1:L
                acc += ZBinTab_pow[x, j, l]
                ZBinCScum_pow[x, j, l] = acc
            end
        end
    end
    return ws
end
