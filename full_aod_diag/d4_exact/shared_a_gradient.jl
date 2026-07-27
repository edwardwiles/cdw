# ============================================================================
# Shared outer (A)-coordinate gradient task (2026-07-27).
#
# Follow-on to the hot-path allocation audit (diag/hot-path-array-allocation-
# audit-2026-07-27), which found `composite_gradient_at_fast`'s per-A-block-
# coordinate loop is the dominant outer-gradient allocation site in ALL FIVE
# families, and that an existing pooled fix (`composite_gradient_at_fast_pooled`,
# gradient_workspace.jl) is wired as the default for ONLY the unrestricted
# family. This file:
#
#   (1) provides ONE shared production (A)-block gradient entry point,
#       `economic_A_gradient!`, that writes directly into caller-owned
#       storage and accepts a PRE-BUILT `LFixBaseCache` (needed by every
#       restricted family, whose cache folds in a family-specific fixed-dual
#       contribution before this function ever sees it) -- unlike
#       `composite_gradient_at_fast_buffered`/`_pooled`, which build their
#       own plain cache internally and have no `cache=` kwarg;
#   (2) eliminates the LAST two confirmed W-scale allocation sites the
#       existing pooled implementation left behind (see
#       docs/A_GRADIENT_D20_ALLOCATION_RECONCILIATION_2026-07-27.md for the
#       measured, source-level accounting that FOUND these two sites, not
#       just asserted them):
#         (a) `select_bandwidth`'s `mass_at()` closure calls
#             `count_winner_flips`/`count_winner_flips_multi_top3`
#             (composite_gradient.jl), which call the ALLOCATING
#             `price_and_pTsigma_cell` and build a fresh
#             `Dict{Int,Vector{Float64}}` on every bisection iteration for
#             every coordinate whose bandwidth is not already cached --
#             measured at 12.64 MB of the D=4/W=8000 pooled gradient's
#             19.12 MB cold-bandwidth-cache total (66%);
#         (b) `lfix_incremental_at_ws!` (gradient_workspace.jl)'s own
#             documented rare-path fallback: coordinates whose 2 affected
#             (o,d) cells land in the SAME destination fall back to the
#             allocating `dest_contrib_incremental_top3`/`dest_contrib_incremental`
#             tier -- measured directly at D=4: 3 of 15 coordinates (those
#             touching destination d==baseIndex) each cost 0.6464 MB vs
#             0.0041 MB for the fast single-origin path, ~1.94 MB total,
#             matching the pooled gradient's own measured residual almost
#             exactly.
#
# Both sites share the same shape ("which of <=2 changed origins won, using
# the cached top-3, without a Dict"). This file adds ONE fixed 2-slot
# scratch struct (`TwoOriginScratch` -- NEVER a Dict, NEVER more than 2
# W-length buffer pairs, matching task §3's explicit "fixed two-slot
# structure" requirement) used by both call sites, plus mutating
# `!`-suffixed twins of the 4 allocating functions this exposes.
#
# Purely ADDITIVE: does not modify composite_gradient.jl, lfix_incremental.jl,
# lfix_buffer_reuse.jl, composite_gradient_fast.jl, or gradient_workspace.jl.
# Equivalence with the allocating originals (and with
# `composite_gradient_at_fast`/`_buffered`/`_pooled`) is verified bit-for-bit
# in test_shared_a_gradient.jl.
# ============================================================================
include(joinpath(@__DIR__, "gradient_workspace.jl"))

# ----------------------------------------------------------------------------
# Fixed two-slot scratch (task §3: "a fixed two-slot structure or
# caller-owned scratch... Do not allocate a dictionary or W-length vector for
# this case").
# ----------------------------------------------------------------------------

"""
    TwoOriginScratch

Fixed two-slot scratch for the <=2-changed-origin winner/contribution case
that appears in BOTH `select_bandwidth`'s bandwidth-search bisection and
`a_block_fd_component`'s same-destination-2-origin fallback. NEVER a Dict,
NEVER more than these 4 W-length buffers (2 price/pTσ pairs), matching the
dependency graph's own proof (lfix_incremental.jl's header) that at most 2
origins change in any one destination for a single reduced pivot coordinate.
"""
struct TwoOriginScratch
    price1::Vector{Float64}
    pTσ1::Vector{Float64}
    price2::Vector{Float64}
    pTσ2::Vector{Float64}
end
TwoOriginScratch(W::Int) = TwoOriginScratch(Vector{Float64}(undef, W), Vector{Float64}(undef, W),
                                             Vector{Float64}(undef, W), Vector{Float64}(undef, W))

"Per-thread pool of `TwoOriginScratch`, mirroring `GradWorkspacePool`'s own `maxthreadid`-based sizing (same not-`nthreads()` pitfall applies -- see that struct's docstring)."
mutable struct TwoOriginScratchPool
    slots::Vector{TwoOriginScratch}
    W::Int
end
build_two_origin_scratch_pool(W::Int; nT::Int = Threads.maxthreadid()) =
    TwoOriginScratchPool([TwoOriginScratch(W) for _ in 1:nT], W)

function resize_two_origin_pool_if_needed!(pool::TwoOriginScratchPool, W::Int; nT::Int = Threads.maxthreadid())
    if pool.W != W || length(pool.slots) < nT
        pool.slots = [TwoOriginScratch(W) for _ in 1:nT]
        pool.W = W
    end
    return pool
end

# ----------------------------------------------------------------------------
# Mutating twins of count_winner_flips / count_winner_flips_multi_top3
# (composite_gradient.jl) -- used by select_bandwidth!'s bisection.
# ----------------------------------------------------------------------------

"""
    count_winner_flips_multi_top3!(ts, cache, ctx, θ_full, d, changed_origins) -> Int

Mutating twin of `count_winner_flips_multi_top3` (composite_gradient.jl): IDENTICAL top-3-cache
case analysis and IDENTICAL exact-tie convention (lowest origin index wins), but writes the <=2
changed origins' perturbed prices into `ts.price1`/`ts.price2` (via `price_and_pTsigma_cell!`)
instead of allocating a fresh `Dict{Int,Vector{Float64}}` + 2 fresh price/pTσ vector pairs per
call. `pTσ` is computed by `price_and_pTsigma_cell!` (cannot cheaply be skipped, same formula
computes both) but not read here -- winner FLIP COUNTING only needs price, matching the
original's own return contract exactly. Falls back to the (rare, unreachable for D>=4)
`count_winner_flips_multi` for `length(changed_origins) > 2`, same as the original.
"""
function count_winner_flips_multi_top3!(ts::TwoOriginScratch, cache::LFixBaseCache, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
    length(changed_origins) > 2 && return count_winner_flips_multi(cache, ctx, θ_full, d, changed_origins)
    D = cache.D; W = cache.W
    Cd = changed_origins
    o1 = Cd[1]
    price_and_pTsigma_cell!(ts.price1, ts.pTσ1, θ_full, ctx, o1, d)
    n = length(Cd)
    o2 = n == 2 ? Cd[2] : 0
    n == 2 && price_and_pTsigma_cell!(ts.price2, ts.pTσ2, θ_full, ctx, o2, d)

    flips = 0
    @inbounds for ω in 1:W
        r1 = cache.winner0[ω, d]; r2 = cache.runnerup0[ω, d]; r3 = cache.third0[ω, d]
        best_o = 0; best_p = Inf
        if !(r1 in Cd)
            best_o = r1; best_p = cache.winner_price0[ω, d]
        elseif !(r2 in Cd)
            best_o = r2; best_p = cache.runnerup_price0[ω, d]
        elseif r3 != 0 && !(r3 in Cd)
            best_o = r3; best_p = cache.third_price0[ω, d]
        end
        bo = best_o; bp = best_p
        v1 = ts.price1[ω]
        if v1 < bp || (v1 == bp && o1 < bo)
            bp = v1; bo = o1
        end
        if n == 2
            v2 = ts.price2[ω]
            if v2 < bp || (v2 == bp && o2 < bo)
                bp = v2; bo = o2
            end
        end
        if bo == 0
            # extremely defensive: all of top-3 were changed (needs D<=3 & |Cd|>=3), unreachable
            # for |Cd|<=2 with D>=3 -- same defensive fallback as the original. Rare enough (never
            # observed, per the original's own docstring) that a small O(D) allocation here is not
            # worth eliminating.
            col = Vector{Float64}(undef, D)
            for o in 1:D
                col[o] = o == o1 ? v1 : (n == 2 && o == o2 ? ts.price2[ω] : cache.price0[ω, o, d])
            end
            _, bo, _ = min_and_secondmin(col)
        end
        flips += (bo != r1)
    end
    return flips
end

"Mutating twin of `count_winner_flips` (composite_gradient.jl): single-changed-origin case writes into `ts.price1` instead of allocating; 2-origin case delegates to `count_winner_flips_multi_top3!`."
function count_winner_flips!(ts::TwoOriginScratch, cache::LFixBaseCache, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int}; multi_method::Symbol = :top3)
    if length(changed_origins) != 1
        multi_method === :top3 && return count_winner_flips_multi_top3!(ts, cache, ctx, θ_full, d, changed_origins)
        multi_method === :generic && return count_winner_flips_multi(cache, ctx, θ_full, d, changed_origins)
        error("count_winner_flips!: multi_method must be :top3 or :generic, got $multi_method")
    end
    o = changed_origins[1]
    price_and_pTsigma_cell!(ts.price1, ts.pTσ1, θ_full, ctx, o, d)
    flips = 0
    @inbounds for ω in 1:cache.W
        wo, _, _, _, _ = update_winner_o1(cache.winner_price0[ω, d], cache.winner0[ω, d],
                                            cache.runnerup_price0[ω, d], cache.runnerup0[ω, d],
                                            o, ts.price1[ω])
        flips += (wo != cache.winner0[ω, d])
    end
    return flips
end

"""
    select_bandwidth!(ts, cache, ctx, pe, w0, coord_idx; kwargs...) -> (h, switch_mass, meta)

Mutating twin of `select_bandwidth` (composite_gradient.jl): IDENTICAL bisection algorithm,
IDENTICAL default kwargs, only the winner-flip counting is routed through `ts::TwoOriginScratch`
(`count_winner_flips!`) instead of the allocating `count_winner_flips`. Bit-for-bit identical
output -- verified in test_shared_a_gradient.jl.
"""
function select_bandwidth!(ts::TwoOriginScratch, cache::LFixBaseCache, ctx, pe, w0::AbstractVector, coord_idx::Int;
        h0::Float64 = 0.01, h_floor::Float64 = 1e-4, h_ceil::Float64 = 0.1,
        target_mass_frac::Tuple{Float64,Float64} = (0.003, 0.03), max_iter::Int = 6,
        multi_method::Symbol = :top3)

    cells = affected_cells(pe, coord_idx)
    @assert !isempty(cells) "select_bandwidth!: coord_idx=$coord_idx has no affected A_od cells (gamma coordinate uses the analytic path, not this)"
    affected_dests = unique(last.(cells))

    function mass_at(h::Float64)
        w = copy(w0); w[coord_idx] += h
        z = pivot_expand(w[2:end], pe); Aod_theta = exp.(z)
        x_free = vcat(w[1], vec(Aod_theta))
        θ_full = CS.reconstruct_full(x_free, ctx.m)
        total_flips = 0
        for d in affected_dests
            origins_here = [o for (o, dd) in cells if dd == d]
            total_flips += count_winner_flips!(ts, cache, ctx, θ_full, d, origins_here; multi_method = multi_method)
        end
        return total_flips / (cache.W * length(affected_dests))
    end

    h = h0
    lo_frac, hi_frac = target_mass_frac
    m = mass_at(h)
    n_iter = 0
    while n_iter < max_iter
        if m < lo_frac && h < h_ceil
            h = min(h * 2, h_ceil)
        elseif m > hi_frac && h > h_floor
            h = max(h / 2, h_floor)
        else
            break
        end
        m = mass_at(h)
        n_iter += 1
        (h == h_ceil || h == h_floor) && break
    end

    return h, m, (n_iter = n_iter, hit_floor = h == h_floor, hit_ceil = h == h_ceil)
end

# ----------------------------------------------------------------------------
# Mutating twin of dest_contrib_incremental_top3 (lfix_incremental.jl) -- used
# by the a_block_fd coordinate loop's same-destination-2-origin case, which
# `lfix_incremental_at_ws!` (gradient_workspace.jl) documents as a disclosed
# ALLOCATING fallback. This extends that fast path to be allocation-free too.
# ----------------------------------------------------------------------------

"Mutating twin of `dest_contrib_incremental_top3`: writes into `contrib_buf` using `ts::TwoOriginScratch` instead of a Dict + fresh return vector. Falls back (allocating, rare/unreachable for D>=4) to `dest_contrib_incremental` for `length(changed_origins) > 2`."
function dest_contrib_incremental_top3!(contrib_buf::AbstractVector, ts::TwoOriginScratch, cache::LFixBaseCache, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
    if length(changed_origins) > 2
        contrib_buf .= dest_contrib_incremental(cache, ctx, θ_full, d, changed_origins)
        return contrib_buf
    end
    D = cache.D; W = cache.W
    Cd = changed_origins
    o1 = Cd[1]
    price_and_pTsigma_cell!(ts.price1, ts.pTσ1, θ_full, ctx, o1, d)
    n = length(Cd)
    o2 = n == 2 ? Cd[2] : 0
    n == 2 && price_and_pTsigma_cell!(ts.price2, ts.pTσ2, θ_full, ctx, o2, d)

    @inbounds for ω in 1:W
        r1 = cache.winner0[ω, d]; r2 = cache.runnerup0[ω, d]; r3 = cache.third0[ω, d]
        best_o = 0; best_p = Inf; best_pTσ = Inf
        if !(r1 in Cd)
            best_o = r1; best_p = cache.winner_price0[ω, d]; best_pTσ = cache.pTσ0[ω, r1, d]
        elseif !(r2 in Cd)
            best_o = r2; best_p = cache.runnerup_price0[ω, d]; best_pTσ = cache.pTσ0[ω, r2, d]
        elseif r3 != 0 && !(r3 in Cd)
            best_o = r3; best_p = cache.third_price0[ω, d]; best_pTσ = cache.third_pTσ0[ω, d]
        end
        bo = best_o; bp = best_p; bpTσ = best_pTσ
        v1 = ts.price1[ω]
        if v1 < bp || (v1 == bp && o1 < bo)
            bp = v1; bo = o1; bpTσ = ts.pTσ1[ω]
        end
        if n == 2
            v2 = ts.price2[ω]
            if v2 < bp || (v2 == bp && o2 < bo)
                bp = v2; bo = o2; bpTσ = ts.pTσ2[ω]
            end
        end
        if bo == 0
            col = Vector{Float64}(undef, D)
            for o in 1:D
                col[o] = o == o1 ? v1 : (n == 2 && o == o2 ? ts.price2[ω] : cache.price0[ω, o, d])
            end
            _, bo, _ = min_and_secondmin(col)
            bpTσ = bo == o1 ? ts.pTσ1[ω] : (n == 2 && bo == o2 ? ts.pTσ2[ω] : cache.pTσ0[ω, bo, d])
        end
        d1w = d + (bo - 1) * cache.Ddest
        contrib_buf[ω] = (cache.SW[ω] / cache.gammafac) * (cache.CONST_d[d] + cache.λstar[d1w] * bpTσ)
    end
    return contrib_buf
end

"""
    lfix_incremental_at_ws2!(ws, ts, cache, ctx, pe, w0, coord_idx, new_val; multi_method=:top3) -> Float64

Extends `lfix_incremental_at_ws!` (gradient_workspace.jl) to route the same-destination-2-origin
case through the mutating `dest_contrib_incremental_top3!` above (using `ts::TwoOriginScratch`)
instead of that function's disclosed allocating fallback. The single-changed-origin case is
UNCHANGED (still `dest_contrib_incremental_o1!`, already allocation-free); the (D<=3, 3+ changed
origins) defensive case still falls back to the original allocating tiers (unreachable for the
D>=4 production configurations this task covers). Bit-for-bit identical to
`lfix_incremental_at_ws!`/`lfix_incremental_at`/`lfix_incremental_at!` -- verified in
test_shared_a_gradient.jl.
"""
function lfix_incremental_at_ws2!(ws::GradWorkspace, ts::TwoOriginScratch, cache::LFixBaseCache, ctx, pe, w0::AbstractVector, coord_idx::Int, new_val::Float64; multi_method::Symbol = :top3)
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
        elseif length(origins_here) == 2
            dest_contrib_incremental_top3!(ws.contrib, ts, cache, ctx, θ_full, d, origins_here)
            ws.q .-= ws.contrib .- old_contrib
        else
            # (D<=3, 3+ changed origins) defensive case -- unreachable for D>=4, kept allocating
            # (matches every existing tier's own established "correctness first, rare paths may
            # allocate" precedent).
            new_contrib = multi_method === :top3 ? dest_contrib_incremental_top3(cache, ctx, θ_full, d, origins_here) :
                          multi_method === :generic ? dest_contrib_incremental(cache, ctx, θ_full, d, origins_here) :
                          error("lfix_incremental_at_ws2!: multi_method must be :top3 or :generic, got $multi_method")
            ws.q .-= new_contrib .- old_contrib
        end
    end
    if cf_touched
        cf_contrib_at!(ws.cf, cache, θ_full, ctx)
        ws.q .-= ws.cf .- cache.cf_contrib0
    end

    return lfix_from_q!(ws.psi, ws.q, cache.ζstar)
end

"Extends `a_block_fd_component_ws!` to use `lfix_incremental_at_ws2!` (allocation-free same-destination-2-origin case). Same central-FD formula/AUD-12 fallback structure."
function a_block_fd_component_ws2!(ws::GradWorkspace, ts::TwoOriginScratch, cache::LFixBaseCache, ctx, pe, w0::AbstractVector, coord_idx::Int, h::Float64; multi_method::Symbol = :top3)
    Lp = lfix_incremental_at_ws2!(ws, ts, cache, ctx, pe, w0, coord_idx, w0[coord_idx] + h; multi_method = multi_method)
    Lm = lfix_incremental_at_ws2!(ws, ts, cache, ctx, pe, w0, coord_idx, w0[coord_idx] - h; multi_method = multi_method)
    if isfinite(Lp) && isfinite(Lm)
        return (Lp - Lm) / (2h)
    elseif isfinite(Lp)
        L0 = lfix_incremental_at_ws2!(ws, ts, cache, ctx, pe, w0, coord_idx, w0[coord_idx]; multi_method = multi_method)
        return (Lp - L0) / h
    elseif isfinite(Lm)
        L0 = lfix_incremental_at_ws2!(ws, ts, cache, ctx, pe, w0, coord_idx, w0[coord_idx]; multi_method = multi_method)
        return (L0 - Lm) / h
    else
        return 0.0
    end
end

# ----------------------------------------------------------------------------
# Section 1/5: ONE shared production entry point + its persistent workspace.
# ----------------------------------------------------------------------------

"""
    EconomicAGradientWorkspace

One workspace per LIVE outer-solver context (task §5): owns the per-thread
`GradWorkspace` pool (q/psi/price/pTσ/contrib/cf buffers), the per-thread
`TwoOriginScratch` pool (the 2 extra price/pTσ pairs for the same-destination
2-origin case), and the persistent per-coordinate bandwidth cache. Construct
ONE of these per family per continuation/campaign (e.g. once per KNITRO outer
solve, reused across every outer iterate and every δ-stage) -- NEVER
per-gradient-call; `economic_A_gradient!` below only ever calls the
`resize_*_if_needed!` no-ops on it, matching `composite_gradient_at_fast_pooled`'s
own established `GradWorkspacePool` discipline (gradient_workspace.jl).
"""
mutable struct EconomicAGradientWorkspace
    grad_pool::GradWorkspacePool
    two_origin_pool::TwoOriginScratchPool
    bandwidth_cache::Dict{Int,Float64}
end

"`EconomicAGradientWorkspace(W; nT=Threads.maxthreadid())` -- one-time allocation for a problem of destination-count-scale `W`."
function EconomicAGradientWorkspace(W::Int; nT::Int = Threads.maxthreadid())
    return EconomicAGradientWorkspace(build_grad_workspace_pool(W; nT = nT), build_two_origin_scratch_pool(W; nT = nT), Dict{Int,Float64}())
end

"""
    economic_A_gradient!(grad_A, base, ctx, pe, ws::EconomicAGradientWorkspace;
                          cache=nothing, threaded=false, h_mode=:cached, h0=0.01,
                          multi_method=:top3) -> meta

THE shared production (A)-block outer-gradient entry point (shared outer-A-gradient task,
2026-07-27, task §1). Writes the `D^2`-length (A)-block gradient directly into caller-owned
`grad_A` (`length(grad_A) == D*Ddest` required) -- no allocation for the gradient vector itself.

Mathematically and (in `h_mode ∈ (:fixed, :cached)`) BIT-FOR-BIT IDENTICAL to
`composite_gradient_at_fast`/`composite_gradient_at_fast_buffered`/`composite_gradient_at_fast_pooled`
-- same exact C+/L-fix finite-secant calculation, same hard winner switching (top-3-cache exact
case analysis), same transformed-(A)/legacy-(z) pivot-reduced coordinate support (`pe`, the
gravity-pivot map, is threaded through unchanged), same gravity-pivot chain rule
(`pivot_expand`/`pivot_reduce`) -- verified in test_shared_a_gradient.jl. Differs from `_pooled`
ONLY in allocation profile: BOTH of the coordinate-loop's remaining confirmed W-scale allocation
sites (`select_bandwidth`'s Dict-based 2-origin winner-flip counting; `a_block_fd`'s own
2-origin-same-destination dest-contribution fallback) are routed through `ws`'s fixed 2-slot
`TwoOriginScratch`, never a Dict, never more than the 2 extra W-length buffer pairs `ws` already
owns (see docs/A_GRADIENT_D20_ALLOCATION_RECONCILIATION_2026-07-27.md for the measurements that
found these two sites).

`cache`: pass a PRE-BUILT `LFixBaseCache` (e.g. from `build_lfix_base_cache_originzc`/
`build_lfix_base_cache_cm_meanzc`, whose `q0` already folds in that family's fixed-nu/CM
contribution) to reuse it instead of building a plain one internally -- mirrors
`composite_gradient_at_fast`'s own `cache=` kwarg (Continuation 13). REQUIRED for every
restricted-family wrapper below, none of which use a plain (non-restricted) cache.

`h_mode`: `:fixed`/`:cached` only (the two production-relevant modes, matching
`composite_gradient_at_fast_buffered`/`_pooled`'s own restricted scope -- `:adaptive`'s extra
h/2 diagnostic probe is not wired into this shared entry point).
"""
function economic_A_gradient!(grad_A::AbstractVector, base::BaseDualState, ctx, pe, ws::EconomicAGradientWorkspace;
        cache::Union{Nothing,LFixBaseCache} = nothing, threaded::Bool = false,
        h_mode::Symbol = :cached, h0::Float64 = 0.01, multi_method::Symbol = :top3,
        bandwidth_cache::Union{Nothing,Dict{Int,Float64}} = nothing)
    h_mode in (:fixed, :cached) || error("economic_A_gradient!: h_mode must be :fixed|:cached, got $h_mode")

    x_free0 = base.x_free0
    cache = cache === nothing ? build_lfix_base_cache(x_free0, ctx, base; validate_dense = false) : cache
    D = ctx.D
    Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    D2 = D * Ddest
    length(grad_A) == D2 || error("economic_A_gradient!: grad_A must have length D2=$D2, got $(length(grad_A))")
    W = cache.W
    z0 = log.(reshape(x_free0[2:end], D, Ddest))
    w0 = vcat(x_free0[1], pivot_reduce(z0, pe))

    grad_A[1] = gamma_component_analytic(cache, base, w0[1])

    resize_pool_if_needed!(ws.grad_pool, W)
    resize_two_origin_pool_if_needed!(ws.two_origin_pool, W)
    # A caller-supplied `bandwidth_cache` (e.g. a checkpointed, cross-outer-iteration Dict some
    # restricted-family callers already persist -- see cm_originzc_checkpoint.jl) takes precedence
    # over `ws`'s own persistent cache, for drop-in backward compatibility with existing checkpoint
    # schemas. Falls back to `ws.bandwidth_cache` (persistent across calls that share this `ws`,
    # task §5) when the caller does not supply one.
    bandwidth_cache = bandwidth_cache === nothing ? ws.bandwidth_cache : bandwidth_cache
    h_used = zeros(D2); cache_hits = falses(D2)
    bandwidth_cache_lock = ReentrantLock()

    function do_coord!(k::Int)
        tid = Threads.threadid()
        gws = ws.grad_pool.slots[tid]; ts = ws.two_origin_pool.slots[tid]
        if h_mode == :fixed
            h = h0
            h_used[k] = h
        else # :cached
            local h, is_hit
            lock(bandwidth_cache_lock) do
                is_hit = haskey(bandwidth_cache, k)
                h = is_hit ? bandwidth_cache[k] : NaN
            end
            if is_hit
                cache_hits[k] = true
            else
                h, _, _ = select_bandwidth!(ts, cache, ctx, pe, w0, k; multi_method = multi_method)
                lock(bandwidth_cache_lock) do
                    bandwidth_cache[k] = h
                end
                cache_hits[k] = false
            end
            h_used[k] = h
        end
        grad_A[k] = a_block_fd_component_ws2!(gws, ts, cache, ctx, pe, w0, k, h_used[k]; multi_method = multi_method)
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

    return (base = base, cache = cache, w0 = w0, h_used = h_used, h_mode = h_mode,
            threaded = threaded, cache_hits = cache_hits, winner0 = copy(cache.winner0),
            gamma_component = grad_A[1], A_gradient_backend = :shared_inplace_pooled)
end

"""
    economic_A_gradient(base, ctx, pe, ws; kwargs...) -> (grad_A::Vector{Float64}, meta)

Allocating (returns a fresh `Vector{Float64}(D2)`) convenience wrapper around `economic_A_gradient!`
for call sites that want the old `(g, meta)` return-value calling convention (matching
`composite_gradient_at_fast`'s own signature) instead of writing into caller-owned storage. The
D2-length gradient vector allocation here is NOT a hot-path W-scale allocation (D2, not W) -- see
task §9's own distinction between the two -- and every restricted-family wrapper below uses the
in-place `!` form directly, not this wrapper.
"""
function economic_A_gradient(base::BaseDualState, ctx, pe, ws::EconomicAGradientWorkspace; kwargs...)
    D = ctx.D
    Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    grad_A = zeros(D * Ddest)
    meta = economic_A_gradient!(grad_A, base, ctx, pe, ws; kwargs...)
    return grad_A, meta
end
