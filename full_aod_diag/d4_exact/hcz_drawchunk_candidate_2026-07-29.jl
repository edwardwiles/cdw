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

2026-08-05 root-cause fix #3 (paired-basis-preconditioning pilot, CM+ZC follow-up): `local_tabs_pow`
is the per-worker companion of `local_tabs`, needed for the two-family (eq.36 truncated-power)
family's own H_CZ block -- both `bin_zc_cross_hessian_fill_drawchunk!` (this file) and
`bin_zc_cross_hessian_fill_drawchunk_reordered!` (hcz_reordered_candidate_2026-08-01.jl, the
production default via `HCZ_PREP_BACKEND_DEFAULT[]`) had NO Pow/fam2 handling at all until this
fix -- the same missing-companion-table failure mode as root causes #1 (cm_hessian_threaded.jl)
and #2 (threaded_cross_hessian.jl), just in CM+ZC's own additional widened-row H_CZ prep. Always
allocated (same reasoning as `WinnerBinCrossScratch`'s/`BinZCrossScratch`'s own unconditional
"_pow" fields) -- modest fixed cost, not worth threading a family-count flag through this struct's
several call sites.
"""
mutable struct BinZCrossDrawChunkScratch
    D::Int
    L::Int
    nz::Int
    workers::Int
    local_tabs::Array{Float64,4}
    local_tabs_pow::Array{Float64,4}
    tasks::Vector{Task}
end
function BinZCrossDrawChunkScratch(D::Int, L::Int, nz::Int, workers::Int)
    BinZCrossDrawChunkScratch(D, L, nz, workers, zeros(D, nz, L + 1, workers), zeros(D, nz, L + 1, workers),
                               Vector{Task}(undef, workers))
end
function ensure_bin_zc_drawchunk_scratch!(dc::Union{Nothing,BinZCrossDrawChunkScratch}, D::Int, L::Int, nz::Int, workers::Int)
    if dc === nothing || dc.D != D || dc.L != L || dc.nz != nz || dc.workers != workers
        return BinZCrossDrawChunkScratch(D, L, nz, workers)
    end
    return dc
end

const HCZ_CANDIDATE_TOL = 1e-9   # relative tolerance for non-bit-identical reduction-order candidates

"""
    bin_zc_cross_hessian_fill_drawchunk!(ws, dc, Bidx, ZcS; workers, Pow=nothing) -> ws

Draw-chunked drop-in replacement for `bin_zc_cross_hessian_fill!`/`_threaded!`. Fills the SAME
`ws.ZBinTab`/`ws.ZBinCScum` outputs (`winner_pair_cross_hessian.jl`'s `BinZCrossScratch`) via
per-worker draw-chunk ownership instead of origin ownership.

2026-08-05 root-cause fix #3: `Pow` (two-family only) additionally accumulates `dc.local_tabs_pow`
in the SAME per-(w,x) loop, one extra `Pow[w,x]` factor -- direct port of the serial
`bin_zc_cross_hessian_fill!`'s own "_pow" formula (winner_pair_cross_hessian.jl), just executed
per-draw-chunk instead of per-draw. See `BinZCrossDrawChunkScratch`'s own docstring for the
incident writeup.
"""
function bin_zc_cross_hessian_fill_drawchunk!(ws::BinZCrossScratch, dc::BinZCrossDrawChunkScratch,
        Bidx::AbstractMatrix{<:Integer}, ZcS::AbstractMatrix{Float64}; workers::Int,
        Pow::Union{Nothing,AbstractMatrix{Float64}} = nothing)
    workers <= nthreads() || error("bin_zc_cross_hessian_fill_drawchunk!: workers=$workers exceeds Threads.nthreads()=$(nthreads())")
    workers == dc.workers || error("bin_zc_cross_hessian_fill_drawchunk!: workers=$workers != scratch's dc.workers=$(dc.workers) -- rebuild scratch")
    D = ws.D; L = ws.L; nz = ws.nz
    W = size(ZcS, 1)
    size(Bidx, 1) == W || error("bin_zc_cross_hessian_fill_drawchunk!: size(Bidx,1)=$(size(Bidx,1)) != size(ZcS,1)=$W")
    size(ZcS, 2) >= nz || error("bin_zc_cross_hessian_fill_drawchunk!: size(ZcS,2)=$(size(ZcS,2)) < ws.nz=$nz")
    fam2 = Pow !== nothing

    local_tabs = dc.local_tabs
    local_tabs_pow = dc.local_tabs_pow
    fill!(local_tabs, 0.0)
    fam2 && fill!(local_tabs_pow, 0.0)
    w_chunks = cross_hessian_chunk_ranges(W, workers)
    tasks = dc.tasks
    for wk in 1:workers
        wr = w_chunks[wk]
        lt_pow = fam2 ? (@view local_tabs_pow[:, :, :, wk]) : nothing
        tasks[wk] = Threads.@spawn begin
            @inbounds for w in wr
                for x in 1:D
                    b = Bidx[w, x]
                    px = fam2 ? Pow[w, x] : 0.0
                    for j in 1:nz
                        local_tabs[x, j, b, wk] += ZcS[w, j]
                        fam2 && (lt_pow[x, j, b] += ZcS[w, j] * px)
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

    if fam2
        ZBinTab_pow = ws.ZBinTab_pow
        fill!(ZBinTab_pow, 0.0)
        @inbounds for wk in 1:workers
            for b in 1:(L+1), j in 1:nz, x in 1:D
                ZBinTab_pow[x, j, b] += local_tabs_pow[x, j, b, wk]
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

"""
    HCZ_PREP_BACKEND_DEFAULT

`:origin_owned` (existing `bin_zc_cross_hessian_fill!`/`_threaded!`, unchanged) |
`:draw_chunk_thread_local` (this file's candidate, 12-14x faster at real D=20/W=100,000 per
`docs/PART_C_HCZ_CANDIDATE_RESULTS_2026-07-29.md`, tolerance-level correct -- NOT bit-identical,
see `HCZ_CANDIDATE_TOL`).

Flipped to `:draw_chunk_thread_local` 2026-07-29 after a complete-inner-solve gate through the
real `run_cm_upper_checkpointed` driver (`hcz_complete_inner_solve_smoke_2026-07-29.jl`, real
D=20/W=100,000, `draw_seed=20260719`): both backends complete real KNITRO inner solves with
feasible status (`nStatus=-401`), no failure signature of any kind (contrast with the H_ZZ
backend bake-off, `docs/HZZ_BACKEND_BAKEOFF_VERDICT_2026-07-29.md`, where 4/4 non-reference
backends failed identically) -- `:draw_chunk_thread_local` completed MORE outer evaluations in
LESS wall time (99.0s/4 evals vs 166.6s/3 evals for `:origin_owned`), consistent with its
isolated-kernel speedup. Also gated bit-identical (max|Δ|=0.0, 3 random states) through the
complete packed Hessian callback (`hcz_wired_complete_hessian_gate_2026-07-29.jl`) and 56/56
correctness rows at D=4 (rectangular/square, K variants) and real D=20
(`docs/PART_C_HCZ_CANDIDATE_RESULTS_2026-07-29.md`).

Flipped again to `:draw_chunk_reordered` (2026-08-01, ZC Hessian backend production integration):
`hcz_reordered_candidate_2026-08-01.jl`'s memory-access-pattern fix (contiguous `ZcS`/`Bidx` column
reads instead of `:draw_chunk_thread_local`'s stride-W inner loop) is 2.65x faster in isolation at
real K=3/W=100,000 production width, EXACT correctness (`max|Δ|=0.0`) at every tested
(workers,jtile) combination, and part of a validated 22-26% real single-inner-solve speedup
(genuine-cold, JIT-warm, `julia -t 10`, matched against an independent cross-session baseline) at
both W=100,000 and W=500,000 -- see `ZC_HESSIAN_BACKEND_CLOSEOUT_MASTER_2026-08-01.md` and
`ZC_COMPILE_FREE_BACKEND_AB_2026-08-01.csv`. `:draw_chunk_thread_local` remains available as a
selectable diagnostic fallback (unchanged behavior, not removed). Applies to CM+ZC only -- origin-ZC
has no CM-grid block, hence no H_CZ.
"""
const HCZ_PREP_BACKEND_DEFAULT = Ref{Symbol}(:draw_chunk_reordered)
const HCZ_PREP_DRAWCHUNK_WORKERS_DEFAULT = Ref{Int}(resolve_cross_hessian_workers_default())

"""
    hcz_prep_dispatch!(bin_zc_ws, backend, Bidx, ZcS, cctx; workers, Pow=nothing) -> ws

ONE dispatcher for H_CZ prep, shared by the serial and threaded call sites. `cctx` supplies/owns
the `BinZCrossDrawChunkScratch` (`cctx.bin_zc_drawchunk`, lazily built/resized here) for
`:draw_chunk_thread_local`; `:origin_owned` ignores it entirely.

2026-08-05 root-cause fix #3: `Pow` (two-family only) is now threaded through to whichever backend
is selected -- all three (`:origin_owned`, `:draw_chunk_thread_local`, `:draw_chunk_reordered`) now
support it; see `BinZCrossDrawChunkScratch`'s own docstring for the incident writeup (this
dispatcher previously had no `Pow` parameter at all, so no backend could ever have received it even
after the individual fill functions gained the kwarg).
"""
function hcz_prep_dispatch!(bin_zc_ws::BinZCrossScratch, backend::Symbol,
        Bidx::AbstractMatrix{<:Integer}, ZcS::AbstractMatrix{Float64}, cctx; workers::Int, threaded::Bool,
        Pow::Union{Nothing,AbstractMatrix{Float64}} = nothing)
    if backend === :origin_owned
        if threaded
            bin_zc_cross_hessian_fill_threaded!(bin_zc_ws, Bidx, ZcS; workers = workers, Pow = Pow)
        else
            bin_zc_cross_hessian_fill!(bin_zc_ws, Bidx, ZcS; Pow = Pow)
        end
    elseif backend === :draw_chunk_thread_local
        cctx.bin_zc_drawchunk = ensure_bin_zc_drawchunk_scratch!(cctx.bin_zc_drawchunk, bin_zc_ws.D, bin_zc_ws.L, bin_zc_ws.nz, workers)
        bin_zc_cross_hessian_fill_drawchunk!(bin_zc_ws, cctx.bin_zc_drawchunk, Bidx, ZcS; workers = workers, Pow = Pow)
    elseif backend === :draw_chunk_reordered
        # ZC Hessian backend production integration (2026-08-01): cache-access-pattern candidate,
        # hcz_reordered_candidate_2026-08-01.jl -- reuses the SAME BinZCrossDrawChunkScratch as
        # :draw_chunk_thread_local. This branch was missing from the initial port (only the
        # docstring/default Ref were ported, not this dispatch arm) -- found live via Gate 3's own
        # real-KNITRO-driver run throwing KN_RC_CALLBACK_ERR("hcz_prep_dispatch!: unknown backend
        # :draw_chunk_reordered"), which Gate 2's direct (non-threaded_bins) call path had not
        # exercised. Fixed before merge.
        cctx.bin_zc_drawchunk = ensure_bin_zc_drawchunk_scratch!(cctx.bin_zc_drawchunk, bin_zc_ws.D, bin_zc_ws.L, bin_zc_ws.nz, workers)
        bin_zc_cross_hessian_fill_drawchunk_reordered!(bin_zc_ws, cctx.bin_zc_drawchunk, Bidx, ZcS; workers = workers, Pow = Pow)
    else
        error("hcz_prep_dispatch!: unknown backend :$backend (must be :origin_owned|:draw_chunk_thread_local|:draw_chunk_reordered)")
    end
    return bin_zc_ws
end
