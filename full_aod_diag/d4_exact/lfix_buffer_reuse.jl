# ============================================================================
# Continuation 10, Section 4: allocation-reuse variant of the L_fix central-FD
# gradient loop. Additive only -- does NOT modify lfix_incremental.jl or
# composite_gradient_fast.jl.
#
# MOTIVATION (found by direct code reading, not assumed): `lfix_incremental_at`
# allocates a fresh `q = copy(cache.q0)` (W-length) on EVERY call, and
# `lfix_from_q` allocates a fresh `Psi_q = similar(q)` (W-length) on EVERY
# call. `a_block_fd_component` calls `lfix_incremental_at` twice per
# coordinate (Lp, Lm) -- so a full D^2-coordinate composite gradient performs
# 2*(D^2-1) calls, each allocating 2 arrays of length W. At D=20/W=80000, that
# is 2*399*2 = 1596 allocations of 80000 Float64 (625KB each) = ~1.0GB of
# allocation churn per gradient call, purely from buffers that are used once
# and immediately discarded -- a textbook "many small scalar-loop-adjacent
# allocations that could be a single reused buffer" pattern (the class of
# thing Section 4 of the continuation-10 brief asks to audit), distinct from
# the Hessian-side rank-one/BLAS work owned by the parallel c10-chunked-hessian
# workstream and distinct from the destination-batching/kernel-v2 work
# Continuation 9 Phase 4 already covered (that was about winner-search
# indexing patterns; this is about the SUM/allocation stage that runs
# downstream of any tier).
#
# This file provides buffer-aware variants (`!`-suffixed) that accept a
# caller-owned `(q_buf, psi_buf)` pair (one per thread, since
# `composite_gradient_at_fast`'s `do_coord!` runs under `Threads.@threads`)
# and reuse them across every probe instead of allocating fresh ones.
# Bit-identical to the original by construction (same arithmetic, same
# summation order -- only the buffer's provenance changes) -- verified in
# `test_lfix_buffer_reuse.jl`.
# ============================================================================
include(joinpath(@__DIR__, "lfix_incremental.jl"))

"lfix_from_q!(psi_buf, q, ζstar) -> Float64 -- buffer-reuse variant of lfix_from_q; writes into caller-owned psi_buf instead of allocating."
function lfix_from_q!(psi_buf::AbstractVector, q::AbstractVector, ζstar::Float64)
    CS.Psi!(psi_buf, q)
    return -(sum(psi_buf) / length(q) + ζstar)
end

"""
    lfix_incremental_at!(q_buf, psi_buf, cache, ctx, pe, w0, coord_idx, new_val; tier=:incremental_o1, multi_method=:top3) -> Float64

Buffer-reuse variant of `lfix_incremental_at`: writes the working `q` into
caller-owned `q_buf` (via `copyto!` instead of `copy`) and reuses `psi_buf`
inside `lfix_from_q!`. Identical arithmetic/summation order to the original --
same function, different buffer provenance only.
"""
function lfix_incremental_at!(q_buf::AbstractVector, psi_buf::AbstractVector,
        cache::LFixBaseCache, ctx, pe, w0::AbstractVector, coord_idx::Int, new_val::Float64;
        tier::Symbol = :incremental_o1, multi_method::Symbol = :top3)
    D = cache.D
    w = copy(w0); w[coord_idx] = new_val   # O(D^2)-length, not O(W) -- not the allocation target here
    z = pivot_expand(w[2:end], pe)
    Aod_theta = exp.(z)
    x_free = vcat(w[1], vec(Aod_theta))
    θ_full = CS.reconstruct_full(x_free, ctx.m)

    cells = affected_cells(pe, coord_idx)
    affected_dests = unique(last.(cells))
    cf_touched = coord_idx == 1 || any(((o, d),) -> o == cache.baseIndex && d == cache.baseIndex, cells)

    copyto!(q_buf, cache.q0)
    for d in affected_dests
        old_contrib = @view cache.contrib0[:, d]
        origins_here = [o for (o, dd) in cells if dd == d]
        new_contrib = tier == :incremental_o1 ? dest_contrib_incremental_o1(cache, ctx, θ_full, d, origins_here; multi_method = multi_method) :
                      tier == :incremental ? dest_contrib_incremental(cache, ctx, θ_full, d, origins_here) :
                      tier == :block_local ? dest_contrib_block_local(cache, ctx, θ_full, d) :
                      error("lfix_incremental_at!: unknown tier=$tier")
        q_buf .-= new_contrib .- old_contrib
    end
    if cf_touched
        new_cf = cf_contrib_at(cache, θ_full, ctx)
        q_buf .-= new_cf .- cache.cf_contrib0
    end

    return lfix_from_q!(psi_buf, q_buf, cache.ζstar)
end

"""
    a_block_fd_component!(q_buf, psi_buf, cache, ctx, pe, w0, coord_idx, h; multi_method=:top3) -> Float64

Buffer-reuse variant of `a_block_fd_component`: the SAME `q_buf`/`psi_buf`
pair is reused across the plus AND minus probe (sequential, not concurrent,
within one coordinate's own call) -- safe because each probe fully
overwrites `q_buf` via `copyto!` before use, and the coordinate's own probes
are never interleaved with another coordinate's on the SAME buffer pair
(callers must supply a distinct buffer pair per concurrent thread, matching
`composite_gradient_at_fast`'s existing `bandwidth_cache_lock`-style
per-call-not-per-coordinate-shared discipline).
"""
function a_block_fd_component!(q_buf::AbstractVector, psi_buf::AbstractVector,
        cache::LFixBaseCache, ctx, pe, w0::AbstractVector, coord_idx::Int, h::Float64; multi_method::Symbol = :top3)
    Lp = lfix_incremental_at!(q_buf, psi_buf, cache, ctx, pe, w0, coord_idx, w0[coord_idx] + h; multi_method = multi_method)
    Lm = lfix_incremental_at!(q_buf, psi_buf, cache, ctx, pe, w0, coord_idx, w0[coord_idx] - h; multi_method = multi_method)
    if isfinite(Lp) && isfinite(Lm)
        return (Lp - Lm) / (2h)
    elseif isfinite(Lp)
        L0 = lfix_incremental_at!(q_buf, psi_buf, cache, ctx, pe, w0, coord_idx, w0[coord_idx]; multi_method = multi_method)
        return (Lp - L0) / h
    elseif isfinite(Lm)
        L0 = lfix_incremental_at!(q_buf, psi_buf, cache, ctx, pe, w0, coord_idx, w0[coord_idx]; multi_method = multi_method)
        return (L0 - Lm) / h
    else
        return 0.0
    end
end

"""
    composite_gradient_at_fast_buffered(x_free0, ctx, pe; base=nothing, threaded=false,
                                         h_mode=:cached, bandwidth_cache) -> (g, meta)

Section-4 candidate: identical to `composite_gradient_at_fast` (h_mode
restricted to :cached/:fixed, the two production-relevant modes -- :adaptive/
:quantile's own extra h/2 diagnostic probe would need a second buffer pair,
not implemented here since :cached is the actual production default per
`c9_phase8_d20_pilot.jl`), except each thread owns ONE persistent
`(q_buf, psi_buf)` pair (sized `Threads.nthreads()`, allocated ONCE per call
-- not per coordinate, not per probe) reused across every coordinate that
thread processes. Equivalence with `composite_gradient_at_fast` verified in
`test_lfix_buffer_reuse.jl`.
"""
function composite_gradient_at_fast_buffered(x_free0::AbstractVector, ctx, pe;
        base::Union{Nothing,BaseDualState} = nothing, threaded::Bool = false,
        h_mode::Symbol = :cached, h0::Float64 = 0.01,
        bandwidth_cache::Union{Nothing,Dict{Int,Float64}} = nothing,
        multi_method::Symbol = :top3)
    h_mode in (:fixed, :cached) || error("composite_gradient_at_fast_buffered: h_mode must be :fixed|:cached, got $h_mode")
    h_mode == :cached && bandwidth_cache === nothing && error("composite_gradient_at_fast_buffered: h_mode=:cached requires a bandwidth_cache Dict")

    base = base === nothing ? solve_base_state(x_free0, ctx) : base
    cache = build_lfix_base_cache(x_free0, ctx, base; validate_dense = false)
    D = ctx.D; D2 = D^2; W = cache.W
    z0 = log.(reshape(x_free0[2:end], D, D))
    w0 = vcat(x_free0[1], pivot_reduce(z0, pe))

    g = zeros(D2)
    g[1] = gamma_component_analytic(cache, base, w0[1])
    h_used = zeros(D2); cache_hits = falses(D2)

    # Threads.nthreads() only counts the :default pool -- a task can also run on the
    # :interactive pool's thread(s) (e.g. the main thread), whose threadid() values are
    # NOT necessarily <= Threads.nthreads(). Confirmed live (this task): with
    # JULIA_NUM_THREADS=20, Threads.nthreads()==20 but a real :static-scheduled task
    # reported threadid()==21, causing a genuine BoundsError the first time this was run
    # -- Threads.maxthreadid() (the documented upper bound on threadid() across ALL
    # pools) is the correct sizing call, not Threads.nthreads().
    nT = Threads.maxthreadid()
    q_bufs = [Vector{Float64}(undef, W) for _ in 1:nT]
    psi_bufs = [Vector{Float64}(undef, W) for _ in 1:nT]

    bandwidth_cache_lock = ReentrantLock()

    function do_coord!(k::Int)
        tid = Threads.threadid()
        q_buf = q_bufs[tid]; psi_buf = psi_bufs[tid]
        if h_mode == :fixed
            h = h0
            h_used[k] = h
            g[k] = a_block_fd_component!(q_buf, psi_buf, cache, ctx, pe, w0, k, h; multi_method = multi_method)
        else # :cached
            local h, is_hit
            lock(bandwidth_cache_lock) do
                is_hit = haskey(bandwidth_cache, k)
                h = is_hit ? bandwidth_cache[k] : NaN
            end
            if is_hit
                cache_hits[k] = true
            else
                h, _, _ = select_bandwidth(cache, ctx, pe, w0, k; multi_method = multi_method)
                lock(bandwidth_cache_lock) do
                    bandwidth_cache[k] = h
                end
                cache_hits[k] = false
            end
            h_used[k] = h
            g[k] = a_block_fd_component!(q_buf, psi_buf, cache, ctx, pe, w0, k, h; multi_method = multi_method)
        end
        return nothing
    end

    if threaded
        # :static, not the default :dynamic -- REQUIRED for correctness here, not just an
        # optimization choice: do_coord! captures tid=Threads.threadid() once and reuses
        # q_bufs[tid]/psi_bufs[tid] across the whole coordinate's work, including across a
        # `lock()` call (bandwidth_cache_lock) that can yield. Under :dynamic scheduling a
        # yielded task can resume on a DIFFERENT OS thread, silently invalidating the
        # threadid()-indexed buffer assignment (a real race, not a theoretical one) -- :static
        # pins each loop chunk to one thread for its entire execution, ruling this out by
        # construction.
        CS.guard_enter_coord_pool!()
        try
            Threads.@threads :static for k in 2:D2
                do_coord!(k)
            end
        finally
            CS.guard_exit_coord_pool!()
        end
    else
        for k in 2:D2
            do_coord!(k)
        end
    end

    return g, (base = base, cache = cache, w0 = w0, h_used = h_used, h_mode = h_mode,
               threaded = threaded, cache_hits = cache_hits, tie_fallback = false)
end
