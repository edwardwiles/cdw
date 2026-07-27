# ============================================================================
# Persistent per-thread gradient workspace (allocation/cache cleanup task, §7).
#
# MOTIVATION (measured, not assumed -- see docs/fullA_allocation_cache_cleanup_handoff.md
# §6): `composite_gradient_at_fast_buffered`'s own `q_bufs`/`psi_bufs` (lfix_buffer_reuse.jl)
# are ALREADY reused across all D^2-1 coordinates within one gradient call -- that part of
# the "buffer reuse" story was done in Continuation 10/11. What is NOT reused is everything
# `a_block_fd_component!` calls THROUGH those buffers: `dest_contrib_incremental_o1` (the
# default, common-case tier) still calls the ALLOCATING `price_and_pTsigma_cell` (not the
# `!`-suffixed in-place version `build_lfix_base_cache` already uses) and allocates a fresh
# `contrib = Vector{Float64}(undef, W)` every single call; `cf_contrib_at` allocates two more
# fresh W-length arrays every time the counterfactual column is touched. Measured live at
# real D20/W=80000/calibration (`bench_grad_alloc.jl`): 7.72MB per coordinate probe, 4.49GB
# for a full serial 400-coordinate gradient call -- matching the brief's own "~4GB/gradient"
# figure. This file adds an in-place path for exactly the fast/common tiers (single-changed-
# origin `dest_contrib_incremental_o1`, `cf_contrib_at`) that a real D20 gradient call
# overwhelmingly uses, and a `GradWorkspacePool` that survives ACROSS gradient calls (not
# just within one), unlike `q_bufs`/`psi_bufs` which are still rebuilt fresh at the top of
# every `composite_gradient_at_fast_buffered` call.
#
# Purely ADDITIVE: does not modify lfix_incremental.jl, lfix_buffer_reuse.jl, or
# composite_gradient_fast.jl. The rare 2-changed-origin-in-one-destination case (top3/generic
# tiers) and the `:block_local`/`:generic` tiers fall back to the existing ALLOCATING
# functions unchanged -- disclosed, not silently different -- matching this file's own
# established "correctness first, rare paths may allocate" precedent (see
# `dest_contrib_incremental_o1`'s own docstring on the 2-changed-origin fallback).
#
# Equivalence with `composite_gradient_at_fast_buffered` is verified bit-for-bit in
# `test_gradient_workspace.jl`.
# ============================================================================
include(joinpath(@__DIR__, "lfix_buffer_reuse.jl"))

"Per-thread scratch for the L_fix coordinate-gradient hot path: 6 W-length buffers, allocated once, reused across every coordinate AND every gradient call that shares the same pool."
struct GradWorkspace
    q::Vector{Float64}
    psi::Vector{Float64}
    price::Vector{Float64}
    pTσ::Vector{Float64}
    contrib::Vector{Float64}
    cf::Vector{Float64}
end
GradWorkspace(W::Int) = GradWorkspace((Vector{Float64}(undef, W) for _ in 1:6)...)

"""
    GradWorkspacePool

Mutable holder for one `GradWorkspace` per thread SLOT (sized via `Threads.maxthreadid()`,
NOT `Threads.nthreads()` -- see `composite_gradient_at_fast_buffered`'s own documented
`maxthreadid` bug catch, the same pitfall applies here). Callers build ONE pool via
`build_grad_workspace_pool` and thread it through every subsequent gradient call (e.g. across
KNITRO outer iterations, across δ-stages) instead of letting each call allocate its own
buffers -- this is the piece `q_bufs`/`psi_bufs` in `lfix_buffer_reuse.jl` do NOT do (those
are still rebuilt fresh at the top of every `composite_gradient_at_fast_buffered` call).
"""
mutable struct GradWorkspacePool
    slots::Vector{GradWorkspace}
    W::Int
end

"`build_grad_workspace_pool(W; nT=Threads.maxthreadid())` -- one-time allocation, `nT` `GradWorkspace`s of length `W` each."
function build_grad_workspace_pool(W::Int; nT::Int = Threads.maxthreadid())
    return GradWorkspacePool([GradWorkspace(W) for _ in 1:nT], W)
end

"""
    resize_pool_if_needed!(pool, W; nT=Threads.maxthreadid()) -> pool

Mutates `pool` in place (rebuilding `slots` if `W` changed or the thread count grew) and
returns the SAME object -- callers holding a reference to `pool` see the update without
needing to reassign. A no-op (zero allocation) when `W`/`nT` already match, which is the
steady-state case across repeated gradient calls at a fixed problem size.
"""
function resize_pool_if_needed!(pool::GradWorkspacePool, W::Int; nT::Int = Threads.maxthreadid())
    if pool.W != W || length(pool.slots) < nT
        pool.slots = [GradWorkspace(W) for _ in 1:nT]
        pool.W = W
    end
    return pool
end

"In-place variant of `cf_contrib_at`: writes into caller-supplied `buf` via ONE fused broadcast (`.=` over the full expression tree) instead of allocating two fresh W-length temporaries (`raw_cf` and the return value). Same formula, byte-for-byte."
function cf_contrib_at!(buf::AbstractVector, cache::LFixBaseCache, θ_full::AbstractVector, ctx)
    bi = cache.baseIndex; σ = cache.σ
    AodPow_bibi = aod_pow_cell(θ_full, ctx, bi, bi)
    γ_prime_bi = θ_full[3+ctx.D]
    constConsσ_bibi = cache.wPrime_bi^(1 - σ) * (AodPow_bibi * cache.τPrime_bi)^(1 - σ)
    denom_cf = γ_prime_bi^σ * wPrime_bi_gdp(cache.wPrime_bi, cache.LPrime_bi)
    buf .= cache.λ_cf .* ((constConsσ_bibi ./ cache.Uσ_bi .- denom_cf) ./ cache.gammafac .* cache.SW)
    return buf
end

"""
    dest_contrib_incremental_o1!(contrib_buf, price_buf, pTσ_buf, cache, ctx, θ_full, d, o)

In-place variant of `dest_contrib_incremental_o1`'s single-changed-origin fast path: uses
`price_and_pTsigma_cell!` (the SAME in-place primitive `build_lfix_base_cache` already uses)
instead of the allocating `price_and_pTsigma_cell`, and writes into caller-supplied
`contrib_buf` instead of allocating a fresh `Vector{Float64}(undef, W)`. Only handles the
`length(changed_origins) == 1` case -- callers with 2+ changed origins must use the existing
allocating `dest_contrib_incremental_top3`/`dest_contrib_incremental` (rare path, disclosed).
"""
function dest_contrib_incremental_o1!(contrib_buf::AbstractVector, price_buf::AbstractVector, pTσ_buf::AbstractVector,
        cache::LFixBaseCache, ctx, θ_full::AbstractVector, d::Int, o::Int)
    W = cache.W
    price_and_pTsigma_cell!(price_buf, pTσ_buf, θ_full, ctx, o, d)
    @inbounds for ω in 1:W
        wo, price_wo, ro, price_ro, _exact = update_winner_o1(
            cache.winner_price0[ω, d], cache.winner0[ω, d],
            cache.runnerup_price0[ω, d], cache.runnerup0[ω, d],
            o, price_buf[ω])
        pTσ_wo = wo == o ? pTσ_buf[ω] : cache.pTσ0[ω, wo, d]
        # BUGFIX (shared outer-A-gradient task, 2026-07-27): this previously read
        # `d1w = d + (wo - 1) * D` using `D = cache.D` (ORIGIN count) as the linear-index stride.
        # Every other site in this codebase that builds this SAME d1/d1w linear index into
        # `λstar`/`γ.P` (build_lfix_base_cache's CONST_d/contrib0, the non-mutating
        # dest_contrib_incremental_o1 in lfix_incremental.jl, hFunction.jl itself) uses
        # `cache.Ddest`/`Ddest` (DESTINATION count) as the stride -- see build_lfix_base_cache's
        # own docstring: "these arrays' own column-major convention has stride Ddest (destination
        # count), NOT D (origin count)". `D == Ddest` for every SQUARE context (D=4 always,
        # D=20/destination_sample=:all_legacy), which silently masked this divergence -- it is
        # WRONG for the current D=20 PRODUCTION DEFAULT (destination_sample=:exclude_row, D=20,
        # Ddest=19), confirmed live: at a real D=20/W=80,000 calibration point, this bug produced
        # contributions differing from the correct (non-mutating) `dest_contrib_incremental_o1` by
        # up to ~98 (vs correct values of order 0.01-0.05) at the very first probed coordinate,
        # propagating into a ~75x-magnitude corruption of composite_gradient_at_fast_pooled's
        # A-block gradient at real D=20 scale (see docs/A_GRADIENT_D20_ALLOCATION_RECONCILIATION_2026-07-27.md
        # for the full repro). This means `composite_gradient_at_fast_pooled` was silently WRONG at
        # the current real-D20 production default the whole time it has existed -- not merely an
        # allocation inefficiency (it was never wired as a hard default anywhere, per the audit,
        # only opt-in, which likely limited exposure, but this is a genuine correctness bug, found
        # and fixed as part of this task, not merely a performance issue).
        d1w = d + (wo - 1) * cache.Ddest
        contrib_buf[ω] = (cache.SW[ω] / cache.gammafac) * (cache.CONST_d[d] + cache.λstar[d1w] * pTσ_wo)
    end
    return contrib_buf
end

"""
    lfix_incremental_at_ws!(ws::GradWorkspace, cache, ctx, pe, w0, coord_idx, new_val) -> Float64

Workspace-aware variant of `lfix_incremental_at!`: routes the common single-changed-origin
case through the in-place `dest_contrib_incremental_o1!`/`cf_contrib_at!` above (zero W-length
allocation); the rare 2-changed-origin-in-one-destination case falls back to the existing
allocating `dest_contrib_incremental_top3` (matches `dest_contrib_incremental_o1`'s own
established fallback, not a new policy). Same arithmetic/summation order as
`lfix_incremental_at!` -- verified bit-for-bit in `test_gradient_workspace.jl`.
"""
function lfix_incremental_at_ws!(ws::GradWorkspace, cache::LFixBaseCache, ctx, pe, w0::AbstractVector, coord_idx::Int, new_val::Float64; multi_method::Symbol = :top3)
    w = copy(w0); w[coord_idx] = new_val   # O(D^2)-length, negligible -- not the allocation target
    z = pivot_expand(w[2:end], pe)
    Aod_theta = exp.(z)
    x_free = vcat(w[1], vec(Aod_theta))
    θ_full = CS.reconstruct_full(x_free, ctx.m)

    cells = affected_cells(pe, coord_idx)
    affected_dests = unique(last.(cells))
    cf_touched = coord_idx == 1 || any(((o, d),) -> o == cache.baseIndex && d == cache.baseIndex, cells)

    copyto!(ws.q, cache.q0)
    for d in affected_dests
        old_contrib = @view cache.contrib0[:, d]
        origins_here = [o for (o, dd) in cells if dd == d]
        if length(origins_here) == 1
            dest_contrib_incremental_o1!(ws.contrib, ws.price, ws.pTσ, cache, ctx, θ_full, d, origins_here[1])
            ws.q .-= ws.contrib .- old_contrib
        else
            # rare (>=2)-changed-origins-in-one-destination case: fall back to the existing
            # ALLOCATING tier, respecting the caller's own multi_method (:top3/:generic) --
            # same policy dest_contrib_incremental_o1 itself uses, not a new one.
            new_contrib = multi_method === :top3 ? dest_contrib_incremental_top3(cache, ctx, θ_full, d, origins_here) :
                          multi_method === :generic ? dest_contrib_incremental(cache, ctx, θ_full, d, origins_here) :
                          error("lfix_incremental_at_ws!: multi_method must be :top3 or :generic, got $multi_method")
            ws.q .-= new_contrib .- old_contrib
        end
    end
    if cf_touched
        cf_contrib_at!(ws.cf, cache, θ_full, ctx)
        ws.q .-= ws.cf .- cache.cf_contrib0
    end

    return lfix_from_q!(ws.psi, ws.q, cache.ζstar)
end

"Workspace-aware variant of `a_block_fd_component!`. Same central-FD formula/fallback structure, routed through `lfix_incremental_at_ws!`."
function a_block_fd_component_ws!(ws::GradWorkspace, cache::LFixBaseCache, ctx, pe, w0::AbstractVector, coord_idx::Int, h::Float64; multi_method::Symbol = :top3)
    Lp = lfix_incremental_at_ws!(ws, cache, ctx, pe, w0, coord_idx, w0[coord_idx] + h; multi_method = multi_method)
    Lm = lfix_incremental_at_ws!(ws, cache, ctx, pe, w0, coord_idx, w0[coord_idx] - h; multi_method = multi_method)
    if isfinite(Lp) && isfinite(Lm)
        return (Lp - Lm) / (2h)
    elseif isfinite(Lp)
        L0 = lfix_incremental_at_ws!(ws, cache, ctx, pe, w0, coord_idx, w0[coord_idx]; multi_method = multi_method)
        return (Lp - L0) / h
    elseif isfinite(Lm)
        L0 = lfix_incremental_at_ws!(ws, cache, ctx, pe, w0, coord_idx, w0[coord_idx]; multi_method = multi_method)
        return (L0 - Lm) / h
    else
        return 0.0
    end
end

"""
    composite_gradient_at_fast_pooled(x_free0, ctx, pe, pool::GradWorkspacePool; base=nothing,
                                       threaded=false, h_mode=:cached, h0=0.01, bandwidth_cache,
                                       multi_method=:top3) -> (g, meta)

Identical to `composite_gradient_at_fast_buffered` (same signature, same defaults, same
:static-scheduling correctness requirement -- see that function's own doc comment on why
:static is REQUIRED, not just an optimization, for the threadid()-indexed buffer pattern used
here too) except it takes a caller-owned, persistent `pool` instead of allocating fresh
`q_bufs`/`psi_bufs` at the top of every call. `resize_pool_if_needed!` is called once (cheap,
a no-op when `W` is unchanged) so the SAME `pool` object can be threaded through every
gradient callback of an entire outer KNITRO run, across δ-stages, etc.
"""
function composite_gradient_at_fast_pooled(x_free0::AbstractVector, ctx, pe, pool::GradWorkspacePool;
        base::Union{Nothing,BaseDualState} = nothing, threaded::Bool = false,
        h_mode::Symbol = :cached, h0::Float64 = 0.01,
        bandwidth_cache::Union{Nothing,Dict{Int,Float64}} = nothing,
        multi_method::Symbol = :top3)
    h_mode in (:fixed, :cached) || error("composite_gradient_at_fast_pooled: h_mode must be :fixed|:cached, got $h_mode")
    h_mode == :cached && bandwidth_cache === nothing && error("composite_gradient_at_fast_pooled: h_mode=:cached requires a bandwidth_cache Dict")

    base = base === nothing ? solve_base_state(x_free0, ctx) : base
    cache = build_lfix_base_cache(x_free0, ctx, base; validate_dense = false)
    D = ctx.D; D2 = D^2; W = cache.W
    z0 = log.(reshape(x_free0[2:end], D, D))
    w0 = vcat(x_free0[1], pivot_reduce(z0, pe))

    g = zeros(D2)
    g[1] = gamma_component_analytic(cache, base, w0[1])
    h_used = zeros(D2); cache_hits = falses(D2)

    nT = Threads.maxthreadid()
    resize_pool_if_needed!(pool, W; nT = nT)

    bandwidth_cache_lock = ReentrantLock()

    function do_coord!(k::Int)
        tid = Threads.threadid()
        ws = pool.slots[tid]
        if h_mode == :fixed
            h = h0
            h_used[k] = h
            g[k] = a_block_fd_component_ws!(ws, cache, ctx, pe, w0, k, h; multi_method = multi_method)
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
            g[k] = a_block_fd_component_ws!(ws, cache, ctx, pe, w0, k, h; multi_method = multi_method)
        end
        return nothing
    end

    if threaded
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
               threaded = threaded, cache_hits = cache_hits, pool = pool)
end
