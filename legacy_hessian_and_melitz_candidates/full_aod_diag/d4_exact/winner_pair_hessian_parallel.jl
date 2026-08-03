# diag/compressed-hessian-operator-audit-2026-07-25 continuation: destination-
# PAIR ownership parallel winner-pair Hessian. See
# docs/WINNER_PAIR_PARALLEL_DESIGN_2026-07-25.md for the full design
# rationale -- summary: the serial kernel (winner_pair_hessian.jl) is already
# pair-outer/draw-inner; different unordered destination-group pairs touch
# PROVABLY DISJOINT (j,jp) coordinates (column index j encodes its owning
# group via j mod Ddest), so partitioning pairs across workers needs no
# reduction, no atomics, no per-worker private QQ copy -- each worker writes
# directly into the shared, persistent QQ/u/r arrays at coordinates only it
# can ever touch. The counterfactual (cf) column, when active, is folded in
# as an ordinary size-1 group (group index Ddest+1), not a second code path.
#
# BENCHMARK/CANDIDATE CODE ONLY -- not wired into any production driver.
# The serial kernel (winner_pair_hessian.jl, untouched) remains the
# correctness oracle and production fallback.
using Base.Threads: nthreads

# ============================================================================
# Group/pair precomputation (theta-fixed, built once per outer point).
# ============================================================================

"One destination-group: either a bilateral destination slot (D possible winning origins) or the counterfactual column (1 'origin', i.e. no winner structure)."
struct WPGroup
    is_cf::Bool
    slot::Int              # bilateral slot index (1..Ddest); unused (0) for cf
    cols::Vector{Int}       # global column indices this group owns, length D (bilateral) or 1 (cf)
end

"Balanced partition of `1:N` into `nchunks` contiguous, size-differ-by-at-most-1 ranges (empty ranges allowed if nchunks>N) -- every group and every pair costs O(W) uniformly, so equal COUNT per chunk is equal WORK per chunk."
function wp_balanced_ranges(N::Int, nchunks::Int)
    base_len, rem = divrem(N, nchunks)
    ranges = Vector{UnitRange{Int}}(undef, nchunks)
    start = 1
    for t in 1:nchunks
        len = base_len + (t <= rem ? 1 : 0)
        ranges[t] = start:(start + len - 1)   # empty (start:start-1) when len==0
        start += len
    end
    return ranges
end

"""
    WinnerPairParallelWorkspace

Persistent, theta-fixed workspace for the pair-ownership parallel kernel.
Built once per outer point, reused across every Hessian-callback call
within one inner solve. Owns the SAME shared QQ/u/r arrays every worker
writes into directly (no per-worker copies), the precomputed group/pair
structure, precomputed BALANCED PARTITIONS FOR EVERY WORKER COUNT this
workspace will be benchmarked at (so no partitioning work happens inside a
timed callback), the exact packed-index lookup table (built once by
literally replaying the reference packing loop -- not an independently
re-derived formula), and persistent Snu/Snu2/task scratch (never
reallocated after construction).
"""
struct WinnerPairParallelWorkspace
    base::WinnerPairHessCtx   # reuses kappa0/pi_vec/nu/y/winner/cf_raw_scaled/D/Ddest/W/ncolI/has_cf
    ngroups::Int
    groups::Vector{WPGroup}
    pairs::Vector{Tuple{Int,Int}}                       # (g,gp), g<=gp, over 1:ngroups
    QQ::Matrix{Float64}                                 # ncolI x ncolI, upper triangle only ever written
    u::Vector{Float64}
    r::Vector{Float64}
    PackIdx::Matrix{Int}                                # (1+ncolI) x (1+ncolI) packed positions, R<=C only valid
    Snu::Vector{Float64}                                # W, persistent (S[w]*nu[w])
    Snu2::Vector{Float64}                                # W, persistent (S[w]*nu[w]^2)
    draw_ranges_by_workers::Dict{Int,Vector{UnitRange{Int}}}
    group_chunks_by_workers::Dict{Int,Vector{UnitRange{Int}}}
    pair_chunks_by_workers::Dict{Int,Vector{UnitRange{Int}}}
    Ssum_local::Vector{Float64}                         # max_workers, persistent
    t0_local::Vector{Float64}
    s0_local::Vector{Float64}
    tasks::Vector{Task}                                  # max_workers, persistent (rebound each call, not reallocated)
    max_workers::Int
end

"""
    build_winner_pair_parallel_workspace(cf::CompressedFactual; worker_counts, max_workers) -> WinnerPairParallelWorkspace

`worker_counts` (default the standard scaling sweep) is precomputed ONCE so
`hessian_core_winner_pair!` never builds a partition inside a timed call.
"""
function build_winner_pair_parallel_workspace(cf::CompressedFactual;
        worker_counts::Vector{Int} = [1, 2, 4, 8, 10, 19, 20],
        max_workers::Int = max(nthreads(), maximum(worker_counts)))
    base = build_winner_pair_ctx(cf)
    D = base.D; Ddest = base.Ddest; ncolI = base.ncolI; has_cf = base.has_cf; W = base.W
    ngroups = Ddest + (has_cf ? 1 : 0)
    groups = Vector{WPGroup}(undef, ngroups)
    for slot in 1:Ddest
        cols = [slot + (o - 1) * Ddest for o in 1:D]
        groups[slot] = WPGroup(false, slot, cols)
    end
    if has_cf
        groups[Ddest + 1] = WPGroup(true, 0, [ncolI])
    end
    pairs = Tuple{Int,Int}[]
    for g in 1:ngroups, gp in g:ngroups
        push!(pairs, (g, gp))
    end

    n = 1 + ncolI
    PackIdx = fill(0, n, n)
    k = 1
    @inbounds for i in 1:n
        for j in i:n
            PackIdx[i, j] = k
            k += 1
        end
    end

    draw_by_w = Dict{Int,Vector{UnitRange{Int}}}()
    group_by_w = Dict{Int,Vector{UnitRange{Int}}}()
    pair_by_w = Dict{Int,Vector{UnitRange{Int}}}()
    for wc in unique(vcat(worker_counts, 1))
        draw_by_w[wc] = wp_balanced_ranges(W, wc)
        group_by_w[wc] = wp_balanced_ranges(ngroups, wc)
        pair_by_w[wc] = wp_balanced_ranges(length(pairs), wc)
    end

    return WinnerPairParallelWorkspace(base, ngroups, groups, pairs,
        zeros(ncolI, ncolI), zeros(ncolI), zeros(ncolI), PackIdx,
        Vector{Float64}(undef, W), Vector{Float64}(undef, W),
        draw_by_w, group_by_w, pair_by_w,
        zeros(max_workers), zeros(max_workers), zeros(max_workers),
        Vector{Task}(undef, max_workers), max_workers)
end

# ============================================================================
# Per-pair / per-group accumulation kernels (called by each worker for its
# OWNED pairs/groups only -- writes land in disjoint coordinates by
# construction, see design doc). No allocation.
# ============================================================================

@inline function wp_accumulate_pair!(QQ::Matrix{Float64}, g1::WPGroup, g2::WPGroup, Snu2::Vector{Float64},
        winner::Matrix{Int}, y::Matrix{Float64}, cf_raw_scaled::Vector{Float64}, W::Int, Ddest::Int)
    if !g1.is_cf && !g2.is_cf && g1.slot == g2.slot
        slot = g1.slot
        @inbounds for w in 1:W
            o = winner[w, slot]
            j = slot + (o - 1) * Ddest
            QQ[j, j] += Snu2[w] * y[w, slot] * y[w, slot]
        end
    elseif !g1.is_cf && !g2.is_cf
        s1 = g1.slot; s2 = g2.slot
        @inbounds for w in 1:W
            o1 = winner[w, s1]; o2 = winner[w, s2]
            j1 = s1 + (o1 - 1) * Ddest
            j2 = s2 + (o2 - 1) * Ddest
            v = Snu2[w] * y[w, s1] * y[w, s2]
            if j1 <= j2
                QQ[j1, j2] += v
            else
                QQ[j2, j1] += v
            end
        end
    elseif !g1.is_cf && g2.is_cf
        s1 = g1.slot; jcf = g2.cols[1]
        @inbounds for w in 1:W
            o1 = winner[w, s1]
            j1 = s1 + (o1 - 1) * Ddest
            QQ[j1, jcf] += Snu2[w] * y[w, s1] * cf_raw_scaled[w]
        end
    else
        jcf = g1.cols[1]
        @inbounds for w in 1:W
            QQ[jcf, jcf] += Snu2[w] * cf_raw_scaled[w] * cf_raw_scaled[w]
        end
    end
    return nothing
end

@inline function wp_accumulate_group!(u::Vector{Float64}, r::Vector{Float64}, g::WPGroup,
        Snu::Vector{Float64}, Snu2::Vector{Float64}, winner::Matrix{Int}, y::Matrix{Float64},
        cf_raw_scaled::Vector{Float64}, W::Int, Ddest::Int)
    if g.is_cf
        jcf = g.cols[1]
        acc_u = 0.0; acc_r = 0.0
        @inbounds for w in 1:W
            acc_u += Snu[w] * cf_raw_scaled[w]
            acc_r += Snu2[w] * cf_raw_scaled[w]
        end
        u[jcf] += acc_u; r[jcf] += acc_r
    else
        slot = g.slot
        @inbounds for w in 1:W
            o = winner[w, slot]
            j = slot + (o - 1) * Ddest
            yv = y[w, slot]
            u[j] += Snu[w] * yv
            r[j] += Snu2[w] * yv
        end
    end
    return nothing
end

# ============================================================================
# Shared entry point.
# ============================================================================

"""
    hessian_core_winner_pair!(hess_packed, curvature_weights, compressed_core, workspace; workers=1, storage=:full_stride)

Shared serial/parallel interface. `compressed_core` is `obj` (needs
`.arg0`/`.arg2`/`.M`/`.ddPsi!`, matching every other Hessian backend in this
codebase). `curvature_weights` is accepted for interface-contract clarity
(matches `obj.arg2` after `ddPsi!` -- computed internally, same convention
as `winner_pair_hessian!`/`hessian!`/`hessian_cm_structured!`) but not
re-read; kept as a named argument so the call site documents what the
kernel depends on. `workers=1` runs the SAME code path (2 spawn rounds of 1
task each) -- not a separate serial implementation, so there is no risk of
the "fast path" silently drifting from the parallel path.

Two spawn rounds: (1) draw-partitioned Snu/Snu2 computation + scalar partial
sums; (2) group-partitioned u/r + pair-partitioned QQ, combined per worker
into one task each (every worker does its u/r groups then its QQ pairs).
`storage=:full_stride` (default) writes into persistent QQ/u/r, packs once;
`storage=:direct_packed` writes final assembled values directly into
`hess_packed` via the precomputed `PackIdx` table.
"""
function hessian_core_winner_pair!(hess_packed::AbstractVector, curvature_weights, obj, workspace::WinnerPairParallelWorkspace;
        workers::Int = 1, storage::Symbol = :full_stride)
    storage in (:full_stride, :direct_packed) || error("hessian_core_winner_pair!: storage must be :full_stride or :direct_packed, got :$storage")
    haskey(workspace.draw_ranges_by_workers, workers) || error("hessian_core_winner_pair!: workers=$workers not in workspace's precomputed worker_counts")
    CS._enter_callback!(obj)
    try
    base = workspace.base
    ddPsi! = obj.ddPsi!
    ddPsi!(obj.arg2, obj.arg0)
    S = obj.arg2; M = obj.M
    Ddest = base.Ddest; ncolI = base.ncolI; W = base.W
    nu = base.nu; y = base.y; winner = base.winner; cf_raw_scaled = base.cf_raw_scaled
    pi_vec = base.pi_vec
    n = 1 + ncolI
    length(hess_packed) == n * (n + 1) ÷ 2 || error("hessian_core_winner_pair!: length mismatch")

    QQ = workspace.QQ; u = workspace.u; r = workspace.r
    Snu = workspace.Snu; Snu2 = workspace.Snu2
    fill!(QQ, 0.0); fill!(u, 0.0); fill!(r, 0.0)

    Ssum_local = workspace.Ssum_local; t0_local = workspace.t0_local; s0_local = workspace.s0_local
    draw_ranges = workspace.draw_ranges_by_workers[workers]
    group_chunks = workspace.group_chunks_by_workers[workers]
    pair_chunks = workspace.pair_chunks_by_workers[workers]
    tasks = workspace.tasks

    # ---- round 1: Snu/Snu2 (persistent, disjoint by draw range) + scalar partial sums ----
    for wk in 1:workers
        rng = draw_ranges[wk]
        tasks[wk] = Threads.@spawn begin
            Ss = 0.0; t0 = 0.0; s0 = 0.0
            @inbounds for w in rng
                Sw = S[w]; nuw = nu[w]
                snu = Sw * nuw
                Snu[w] = snu
                snu2 = snu * nuw
                Snu2[w] = snu2
                Ss += Sw; t0 += snu; s0 += snu2
            end
            (Ss, t0, s0)
        end
    end
    S_sum = 0.0; t0_tot = 0.0; s0_tot = 0.0
    for wk in 1:workers
        Ss, t0v, s0v = fetch(tasks[wk])
        S_sum += Ss; t0_tot += t0v; s0_tot += s0v
    end

    # ---- round 2: u/r (group-owned) + QQ (pair-owned), fused per worker ----
    for wk in 1:workers
        gr = group_chunks[wk]; pr = pair_chunks[wk]
        tasks[wk] = Threads.@spawn begin
            for gi in gr
                wp_accumulate_group!(u, r, workspace.groups[gi], Snu, Snu2, winner, y, cf_raw_scaled, W, Ddest)
            end
            for pi_ in pr
                (g, gp) = workspace.pairs[pi_]
                wp_accumulate_pair!(QQ, workspace.groups[g], workspace.groups[gp], Snu2, winner, y, cf_raw_scaled, W, Ddest)
            end
            nothing
        end
    end
    for wk in 1:workers
        fetch(tasks[wk])
    end

    # ---- final assembly: identical formula/packing to the serial kernel ----
    invM = 1.0 / M
    if storage == :full_stride
        k = 1
        hess_packed[k] = S_sum * invM; k += 1
        @inbounds for j in 1:ncolI
            hess_packed[k] = (u[j] - t0_tot * pi_vec[j]) * invM
            k += 1
        end
        @inbounds for i in 1:ncolI
            for j in i:ncolI
                val = QQ[i, j] - r[i] * pi_vec[j] - pi_vec[i] * r[j] + s0_tot * pi_vec[i] * pi_vec[j]
                hess_packed[k] = val * invM
                k += 1
            end
        end
    else
        PackIdx = workspace.PackIdx
        hess_packed[PackIdx[1, 1]] = S_sum * invM
        @inbounds for j in 1:ncolI
            hess_packed[PackIdx[1, 1 + j]] = (u[j] - t0_tot * pi_vec[j]) * invM
        end
        @inbounds for i in 1:ncolI
            for j in i:ncolI
                val = QQ[i, j] - r[i] * pi_vec[j] - pi_vec[i] * r[j] + s0_tot * pi_vec[i] * pi_vec[j]
                hess_packed[PackIdx[1 + i, 1 + j]] = val * invM
            end
        end
    end
    return hess_packed
    finally
        CS._exit_callback!(obj)
    end
end
