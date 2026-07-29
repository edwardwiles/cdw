# optimize/structured-cross-hessian-ZC-CM-2026-07-28: threaded (output-ownership, no-atomics)
# variants of the four raw-table-fill cross-Hessian primitives already established in
# `winner_pair_cross_hessian.jl` / `zc_restriction_operator.jl` (H_EC, H_EZ, H_CZ, H_ZZ). Those
# primitives are already exact, shared across families, and allocation-free after warm-up -- the
# ONLY gap this file closes is that every one of their raw-table-fill loops is single-threaded
# (confirmed by direct code reading: no `Threads.@threads`/`@spawn` anywhere in either file), while
# this codebase's own diagnostic (`inner_timing_and_termination_diagnostic_2026-07-28.zip`,
# corrected 2026-07-28) found this exact class of work ("crossprep"/H_ER) is the dominant unthreaded
# cost (39-79% of the Hessian callback) once H_EE and the CM bin tables are already threaded
# (production default since `port/shared-winner-pair-core-hessian-production-2026-07-25` and
# `hessian_cm_structured_v2!`/`cctx.use_threaded_bins`, respectively).
#
# Parallelization idiom, matching `core_exact_hessian.jl::hessian_core_winner_pair!`'s own
# established pattern EXACTLY (same file this task's provenance doc identifies as the shared H_EE
# threaded kernel): OUTPUT-ROW ownership, not draw-range ownership with a reduce. Each worker owns a
# *disjoint* slice of the output (a contiguous range of `slot` values for H_EC/H_EZ, a contiguous
# range of `origin`/`x` values for H_CZ, a contiguous range of Gram *columns* for H_ZZ) and
# re-scans ALL `W` draws for its own owned slice. Because every output entry is written by exactly
# one worker, there is no reduction step and no atomics -- and because the summation order over `w`
# for any given output entry is IDENTICAL to the serial version (both accumulate `w = 1:W` in
# order, just executed by a different worker), results are BIT-IDENTICAL to the serial kernels, not
# merely "agrees to floating-point tolerance" -- this is checked directly in the D=4 correctness
# gates (`test_threaded_cross_hessian_d4.jl`), not assumed.
#
# `workers` is capped at `Threads.nthreads()` by every entry point below (mirrors
# `hessian_core_winner_pair!`'s own `workers` bound) -- passing more workers than threads would
# silently oversubscribe, not error, so callers must not do this; enforced here instead.

using Base.Threads: nthreads

"""
    resolve_cross_hessian_workers_default() :: Int

Same piecewise worker-count policy `core_exact_hessian.jl::resolve_core_hessian_workers_default()`
already established for H_EE (`core_exact_hessian.jl`'s own docstring: 20 workers if
`nthreads()>=20`, else 10 if `>=10`, else all available) -- reused here rather than re-derived, and
re-validated (not merely assumed) by this task's own worker sweep
(`CROSS_HESSIAN_WORKER_SWEEP_2026-07-28.csv`).
"""
function resolve_cross_hessian_workers_default()
    n = nthreads()
    n >= 20 && return 20
    n >= 10 && return 10
    return max(1, n)
end

"""
    CROSS_HESSIAN_THREADED_DEFAULT / CROSS_HESSIAN_WORKERS_DEFAULT

Selective-merge release (2026-07-28): flipped `true`. Governs `build_cm_bin_ctx` (flexible_cm,
`cm_production_bundle.jl`; common_frechet, `cm_frechet_level.jl`) and origin_zc's
`OriginZCCoreHessCtx` constructor (`cm_hessian_architectures.jl` ~L1437) -- all three validated by
real D=20/W=100,000 matched-driver complete-inner-solve A/Bs (bit-exact packed Hessian, identical
KNITRO status/n_eval/n_grad/kappa, 1.42x complete-callback for flexible_cm/common_frechet, 4.5x
H_EZ for origin_zc -- see `docs/COMPLETE_INNER_SOLVE_BEFORE_AFTER_2026-07-28.csv` and
`docs/SELECTIVE_STRUCTURED_HESSIAN_RELEASE_MASTER_2026-07-28.md`).

**Also now affects CM+ZC (`cm_meanzc`)**, as of `docs/CM_MEANZC_HEC_HEZ_ISOLATED_GATE_2026-07-28.csv`
passing (bit-exact packed Hessian in isolation, no KNITRO-concurrency errors, 1.26x wall time at
the cleanly-matched point). `build_cm_meanzc_bin_ctx` (`cm_meanzc_production.jl`) reads this same
Ref like the other three families. Caveat unique to this family: H_EC/H_EZ/H_CZ share this ONE
toggle (no independent per-block switch exists) -- H_CZ's own PERFORMANCE evidence is still the
prior session's weaker number (wins only at t=20, loses to serial at t=4/t=8; its CORRECTNESS was
covered by the bit-exact full-packed-Hessian check). Flipped anyway on explicit user sign-off
(2026-07-28) to accept that as a tradeoff rather than hold back the validated H_EC/H_EZ win.
"""
const CROSS_HESSIAN_THREADED_DEFAULT = Ref{Bool}(true)
const CROSS_HESSIAN_WORKERS_DEFAULT = Ref{Int}(resolve_cross_hessian_workers_default())

"""
    cross_hessian_chunk_ranges(n::Int, workers::Int) -> Vector{UnitRange{Int}}

Contiguous partition of `1:n` into `workers` disjoint chunks (last chunk absorbs any remainder),
SAME balanced-chunk convention `cm_hessian_threaded.jl::build_bin_tables_threaded!`'s own
`lo = 1+div((tid-1)*W,nt); hi = div(tid*W,nt)` already uses (just factored out here so all four
threaded kernels below share one implementation instead of four independent copies). A chunk may
legally be empty (`workers > n`, e.g. more Julia threads than CM origins) -- callers must tolerate
an empty range, never assume `workers` chunks are all nonempty.
"""
function cross_hessian_chunk_ranges(n::Int, workers::Int)
    return [1 + div((wk - 1) * n, workers) : div(wk * n, workers) for wk in 1:workers]
end

# ============================================================================
# H_EC: threaded winner_pair_cross_hessian_fill! (winner_pair_cross_hessian.jl)
#
# The two raw-table passes are independent accumulators with DIFFERENT natural output-ownership
# axes:
#   - NuTab/SOnlyTab/QCfTab are indexed (x=origin, bin) only -- no slot/j dependence at all --
#     so these are partitioned by ORIGIN (x in 1:D) ownership.
#   - QTab is indexed (j, x=origin, bin), j = slot + (o-1)*Ddest for o = winner(w,slot) -- for a
#     FIXED slot, j only ever takes the Ddest-strided values {slot, slot+Ddest, slot+2*Ddest, ...}
#     (one per possible origin o), so partitioning by SLOT (destination) ownership gives each worker a
#     fully disjoint set of QTab *rows* (no two slots ever produce the same j). This is exactly the
#     task brief's own suggested design ("one destination... per worker; each worker: owns disjoint
#     output rows; scans draws; reads winner for its destination; accumulates").
# ============================================================================

"""
    winner_pair_cross_hessian_fill_threaded!(wctx, ws, obj, Bidx; workers) -> ws

Threaded drop-in replacement for `winner_pair_cross_hessian_fill!` -- writes the SAME
`ws.QTab`/`ws.NuTab`/`ws.SOnlyTab`/`ws.QCfTab`/`ws.EsumEcon`/`ws.QCScum`/`ws.NuCScum`/
`ws.SOnlyCScum`/`ws.QCfCScum` fields, bit-identical to the serial version (see file header). The
`Threads.@spawn` tasks are stored into `ws.tasks_ec` (persistent, sized at scratch construction) so
no `Vector{Task}` is allocated per call. Cumulative prefix-sum step (`O(D*L)`/`O(D*ncolI*L)`) is
left serial -- negligible next to the raw-table fill it follows (confirmed in this task's own
sub-block profile, not assumed).
"""
function winner_pair_cross_hessian_fill_threaded!(wctx::WinnerPairHessCtx, ws::WinnerBinCrossScratch,
        obj, Bidx::AbstractMatrix{<:Integer}; workers::Int)
    workers <= nthreads() || error("winner_pair_cross_hessian_fill_threaded!: workers=$workers exceeds Threads.nthreads()=$(nthreads())")
    workers <= length(ws.tasks_ec) || error("winner_pair_cross_hessian_fill_threaded!: workers=$workers exceeds scratch's tasks_ec capacity=$(length(ws.tasks_ec)) -- rebuild scratch")
    ddPsi! = obj.ddPsi!
    ddPsi!(obj.arg2, obj.arg0)
    S = obj.arg2
    Ddest = wctx.Ddest; W = wctx.W
    nu = wctx.nu; y = wctx.y; winner = wctx.winner
    D = ws.D; L = ws.L

    QTab = ws.QTab; NuTab = ws.NuTab; SOnlyTab = ws.SOnlyTab; QCfTab = ws.QCfTab; EsumEcon = ws.EsumEcon
    fill!(QTab, 0.0); fill!(NuTab, 0.0); fill!(SOnlyTab, 0.0); fill!(QCfTab, 0.0); fill!(EsumEcon, 0.0)

    has_cf = wctx.has_cf
    crs = wctx.cf_raw_scaled
    tasks = ws.tasks_ec

    # --- pass 1: NuTab/SOnlyTab/QCfTab, ORIGIN(x)-owned ---
    x_chunks = cross_hessian_chunk_ranges(D, workers)
    for wk in 1:workers
        xr = x_chunks[wk]
        tasks[wk] = Threads.@spawn begin
            @inbounds for w in 1:W
                Sw = S[w]; nuw = nu[w]
                snu = Sw * nuw
                snucf = has_cf ? snu * crs[w] : 0.0
                for x in xr
                    b = Bidx[w, x]
                    NuTab[x, b] += snu
                    SOnlyTab[x, b] += Sw
                    has_cf && (QCfTab[x, b] += snucf)
                end
            end
        end
    end
    for wk in 1:workers
        fetch(tasks[wk])
    end

    # --- pass 2: QTab + EsumEcon, SLOT(destination)-owned ---
    slot_chunks = cross_hessian_chunk_ranges(Ddest, workers)
    for wk in 1:workers
        sr = slot_chunks[wk]
        tasks[wk] = Threads.@spawn begin
            @inbounds for slot in sr
                for w in 1:W
                    o = winner[w, slot]
                    j = slot + (o - 1) * Ddest
                    snuy = (S[w] * nu[w]) * y[w, slot]
                    EsumEcon[j] += snuy
                    for x in 1:D
                        QTab[j, x, Bidx[w, x]] += snuy
                    end
                end
            end
        end
    end
    for wk in 1:workers
        fetch(tasks[wk])
    end

    # --- serial cumulative prefix sum (unchanged from winner_pair_cross_hessian_fill!) ---
    QCScum = ws.QCScum; NuCScum = ws.NuCScum; SOnlyCScum = ws.SOnlyCScum; QCfCScum = ws.QCfCScum
    @inbounds for x in 1:D
        acc = 0.0; accS = 0.0; accCf = 0.0
        for l in 1:L
            acc += NuTab[x, l]
            NuCScum[x, l] = acc
            accS += SOnlyTab[x, l]
            SOnlyCScum[x, l] = accS
            accCf += QCfTab[x, l]
            QCfCScum[x, l] = accCf
        end
    end
    @inbounds for x in 1:D, j in 1:ws.ncolI
        acc = 0.0
        for l in 1:L
            acc += QTab[j, x, l]
            QCScum[j, x, l] = acc
        end
    end
    return ws
end

# ============================================================================
# H_EZ: threaded winner_pair_cross_hessian_zc_block! (winner_pair_cross_hessian.jl)
#
# Same SLOT-ownership idea as H_EC's pass 2 above: for fixed `slot`, `j = slot + (o-1)*Ddest`
# ranges only over the Ddest-strided rows belonging to that slot, so partitioning `1:Ddest` across
# workers gives disjoint `HEZ` row ranges. Row 1 (ones/zeta) and the cf row are NOT
# winner-conditioned (plain gemv over all W draws) -- left as single-thread BLAS `gemv!` calls
# (cheap relative to the O(W*Ddest*n_x) scatter, confirmed in the sub-block profile), executed
# once, not chunked.
# ============================================================================

"""
    winner_pair_cross_hessian_zc_block_threaded!(HEZ, wctx, ws, S, Z, M; workers) -> HEZ

Threaded drop-in replacement for `winner_pair_cross_hessian_zc_block!`. Requires
`winner_pair_cross_hessian_zc_prep!(ws, wctx, S)` to have been called already this callback (same
precondition as the serial version). Bit-identical output (same accumulation order per row, just
executed by a different worker for disjoint row ranges).
"""
function winner_pair_cross_hessian_zc_block_threaded!(HEZ::AbstractMatrix{Float64}, wctx::WinnerPairHessCtx,
        ws::WinnerZCCrossScratch, S::AbstractVector{Float64}, Z::AbstractMatrix{Float64}, M::Real; workers::Int)
    workers <= nthreads() || error("winner_pair_cross_hessian_zc_block_threaded!: workers=$workers exceeds Threads.nthreads()=$(nthreads())")
    workers <= length(ws.tasks_ez) || error("winner_pair_cross_hessian_zc_block_threaded!: workers=$workers exceeds scratch's tasks_ez capacity=$(length(ws.tasks_ez)) -- rebuild scratch")
    W = wctx.W; Ddest = wctx.Ddest
    ncolI = wctx.ncolI
    nx = size(Z, 2)
    size(HEZ) == (ncolI + 1, nx) || error("winner_pair_cross_hessian_zc_block_threaded!: size(HEZ)=$(size(HEZ)) != ($(ncolI + 1), $nx)")
    length(S) == W || error("winner_pair_cross_hessian_zc_block_threaded!: length(S)=$(length(S)) != wctx.W=$W")
    size(Z, 1) == W || error("winner_pair_cross_hessian_zc_block_threaded!: size(Z,1)=$(size(Z, 1)) != wctx.W=$W")

    y = wctx.y; winner = wctx.winner; pi_vec = wctx.pi_vec
    has_cf = wctx.has_cf; jcf = ncolI
    Snu = ws.Snu
    invM = 1.0 / M

    row1 = @view HEZ[1, :]
    BLAS.gemv!('T', invM, Z, S, 0.0, row1)

    nbilateral = has_cf ? ncolI - 1 : ncolI
    bilateral_block = @view HEZ[2:1+nbilateral, :]
    fill!(bilateral_block, 0.0)

    tasks = ws.tasks_ez
    slot_chunks = cross_hessian_chunk_ranges(Ddest, workers)
    for wk in 1:workers
        sr = slot_chunks[wk]
        v = ws.thread_scratch_ez[wk]   # persistent per-worker-slot W-length scratch, no per-call allocation
        tasks[wk] = Threads.@spawn begin
            @inbounds for slot in sr
                for w in 1:W
                    v[w] = Snu[w] * y[w, slot]
                end
                wcol = @view winner[:, slot]
                for x in 1:nx
                    Zx = @view Z[:, x]
                    for w in 1:W
                        o = wcol[w]
                        j = slot + (o - 1) * Ddest
                        HEZ[j+1, x] += v[w] * Zx[w]
                    end
                end
            end
        end
    end
    for wk in 1:workers
        fetch(tasks[wk])
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

# ============================================================================
# H_CZ: threaded bin_zc_cross_hessian_fill! (winner_pair_cross_hessian.jl)
#
# No winner-selection at all -- ZBinTab[x,j,b] is indexed purely by (origin x, restriction column
# j, bin b), so ORIGIN(x)-ownership gives disjoint output rows directly, exactly the task brief's
# own suggested design ("one CM origin per worker... There are 20 origins, a natural 20-thread
# decomposition").
# ============================================================================

"""
    bin_zc_cross_hessian_fill_threaded!(ws, Bidx, ZcS; workers) -> ws

Threaded drop-in replacement for `bin_zc_cross_hessian_fill!`. Bit-identical output (each origin's
`ZBinTab[x,:,:]` row accumulates over `w=1:W` in the same order regardless of which worker owns it).
"""
function bin_zc_cross_hessian_fill_threaded!(ws::BinZCrossScratch, Bidx::AbstractMatrix{<:Integer},
        ZcS::AbstractMatrix{Float64}; workers::Int)
    workers <= nthreads() || error("bin_zc_cross_hessian_fill_threaded!: workers=$workers exceeds Threads.nthreads()=$(nthreads())")
    workers <= length(ws.tasks_cz) || error("bin_zc_cross_hessian_fill_threaded!: workers=$workers exceeds scratch's tasks_cz capacity=$(length(ws.tasks_cz)) -- rebuild scratch")
    D = ws.D; L = ws.L; nz = ws.nz
    W = size(ZcS, 1)
    size(Bidx, 1) == W || error("bin_zc_cross_hessian_fill_threaded!: size(Bidx,1)=$(size(Bidx,1)) != size(ZcS,1)=$W")
    size(ZcS, 2) >= nz || error("bin_zc_cross_hessian_fill_threaded!: size(ZcS,2)=$(size(ZcS,2)) < ws.nz=$nz")
    ZBinTab = ws.ZBinTab
    fill!(ZBinTab, 0.0)

    tasks = ws.tasks_cz
    x_chunks = cross_hessian_chunk_ranges(D, workers)
    for wk in 1:workers
        xr = x_chunks[wk]
        tasks[wk] = Threads.@spawn begin
            @inbounds for w in 1:W
                for x in xr
                    b = Bidx[w, x]
                    for j in 1:nz
                        ZBinTab[x, j, b] += ZcS[w, j]
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
    return ws
end

# ============================================================================
# H_ZZ: threaded zc_restriction_gram! (zc_restriction_operator.jl), Candidate B (column-block
# ownership) per the task brief's own Section 7 menu. Deliberately pure-Julia (no nested BLAS call
# inside a `Threads.@spawn` task) -- avoids any question of per-thread BLAS reentrancy under this
# codebase's mandatory `OPENBLAS_NUM_THREADS=1`; the single-thread-BLAS `gemm!` path
# (`zc_restriction_gram!`) remains available as Candidate C and is benchmarked against this
# directly (see CROSS_HESSIAN_WORKER_SWEEP_2026-07-28.csv) -- not assumed to lose.
# ============================================================================

"""
    zc_restriction_gram_threaded!(HZZ, cs, op, M; workers) -> HZZ

Threaded drop-in replacement for `zc_restriction_gram!`. Each worker owns a disjoint contiguous
range of Gram COLUMNS (`j2`) and computes the full upper-triangle entries `HZZ[j1,j2]` for
`j1<=j2` in that range directly from `cs.Zc`/`cs.ZcS` (already centered/`S`-weighted by
`refresh_zc_centered!`, precondition unchanged from the serial version). Lower triangle mirrored
serially after the parallel section (cheap, `O(nx^2)`, not `O(nx^2*W)`).
"""
function zc_restriction_gram_threaded!(HZZ::AbstractMatrix{Float64}, cs::ZCCenteredScratch,
        op::ZCRestrictionOperator, M::Real; workers::Int)
    workers <= nthreads() || error("zc_restriction_gram_threaded!: workers=$workers exceeds Threads.nthreads()=$(nthreads())")
    nx = n_restriction(op)
    size(HZZ) == (nx, nx) || error("zc_restriction_gram_threaded!: size(HZZ)=$(size(HZZ)) != ($nx, $nx)")
    Zc = @view cs.Zc[:, 1:nx]
    ZcS = @view cs.ZcS[:, 1:nx]
    W = size(Zc, 1)
    invM = 1.0 / M

    tasks = Vector{Task}(undef, workers)
    col_chunks = cross_hessian_chunk_ranges(nx, workers)
    for wk in 1:workers
        cols = col_chunks[wk]
        tasks[wk] = Threads.@spawn begin
            @inbounds for j2 in cols
                zcs_j2 = @view ZcS[:, j2]
                for j1 in 1:j2
                    zc_j1 = @view Zc[:, j1]
                    acc = 0.0
                    for w in 1:W
                        acc += zc_j1[w] * zcs_j2[w]
                    end
                    HZZ[j1, j2] = acc * invM
                end
            end
        end
    end
    for wk in 1:workers
        fetch(tasks[wk])
    end
    @inbounds for j2 in 1:nx, j1 in 1:j2-1
        HZZ[j2, j1] = HZZ[j1, j2]
    end
    return HZZ
end
