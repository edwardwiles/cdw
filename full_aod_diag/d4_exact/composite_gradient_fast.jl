# ============================================================================
# Continuation 5, Priority 2: threaded/cached adaptive lfix_composite gradient.
#
# Additive, equivalence-tested extension of composite_gradient.jl -- does not
# modify that file. Motivated directly by Priority 1A/1B's measurements
# (docs/fullA_p1_warmed_profile.md): the live composite-gradient callback
# costs ~141ms warmed, of which only ~10-12ms is the (shareable) base-state
# inner solve; the remaining ~130ms is the SERIAL, per-coordinate
# select_bandwidth (bisection, up to 7 mass_at() evaluations each touching
# O(W) draws) + a_block_fd_component (2-4 O(1)-incremental FD probes per
# coordinate) loop over the 15 A-block coordinates. This file adds three
# independent, composable levers, each individually equivalence-tested
# against the ORIGINAL composite_gradient_at before being trusted:
#
#   1. threaded=true: Threads.@threads over the A-block coordinate loop.
#      Safe by the SAME argument profile_lfix_tiers.jl already established
#      for lfix_incremental_at's incremental_o1 tier (no shared mutable
#      state touched, no obj.moments! call, each probe allocates its own
#      local scratch) -- select_bandwidth's mass_at() and a_block_fd_component
#      both read `cache`/`ctx` only, never mutate them, so parallelizing the
#      outer `for k in 2:D2` loop (disjoint g[k] writes) carries the same
#      safety argument, extended here rather than re-derived from scratch.
#   2. h_mode=:fixed: skip select_bandwidth's bisection AND the h-vs-h/2
#      slope-stability diagnostic entirely (the diagnostic is reported-only,
#      never used to override the chosen h -- composite_gradient.jl's own
#      docstring says so -- so skipping it for a production gradient call
#      changes NOTHING about the returned gradient, only removes 2 of the 4
#      per-coordinate lfix_incremental_at calls). Uses a single fixed h for
#      every coordinate (h0 kwarg, default 0.01, matching the historical
#      FIXED_H baseline this continuation's predecessor flagged as a gap).
#   3. h_mode=:cached: reuse a per-coordinate bandwidth from a caller-owned
#      `Dict{Int,Float64}` (persisted across outer KNITRO iterates by the
#      caller), skipping select_bandwidth's bisection on a cache hit;
#      computes+stores on a miss. The caller decides when to invalidate
#      entries (e.g. periodically, or when the base's winner_hash changes by
#      more than the existing HybridGradientPolicy's winner_jump_frac
#      threshold) -- this file does not impose a revalidation POLICY, only
#      the cache mechanism, matching Priority 2 item 2's "revalidate only
#      when... deteriorate" framing (the policy itself is the caller's
#      choice, kept separate from the mechanism per this investigation's
#      existing HybridGradientPolicy precedent).
#
# Levers compose: h_mode=:fixed/:cached can each be combined with
# threaded=true or false. base-state sharing (Priority 2 item 3, the F/G
# redundancy fix) is handled by the CALLER passing `base` in (already
# supported by the original composite_gradient_at, just never wired into
# run_d4_optimized_fd.jl until this continuation -- see
# run_d4_optimized_fd_fast.jl).
# ============================================================================
include(joinpath(@__DIR__, "composite_gradient.jl"))

"""
    composite_gradient_at_fast(x_free0, ctx, pe; base=nothing, threaded=false,
                                h_mode=:adaptive, h0=0.01, bandwidth_cache=nothing) -> (g, meta)

Drop-in replacement for `composite_gradient_at` with the three levers above.
`h_mode=:adaptive` (default) reproduces the ORIGINAL function's behavior
exactly (same select_bandwidth bisection, same h/2 diagnostic) -- equivalence
with `composite_gradient_at` is verified in `test_composite_gradient_fast.jl`
for every (h_mode, threaded) combination.

Continuation 6: catches `TiedWinnerError` from `build_lfix_base_cache` (an exact price tie between
2+ origins at some draw/destination -- see that error's docstring; NOT a correctness bug, a genuine
edge case the O(1)/O(D) incremental machinery cannot represent) and falls back to a full-rebuild
central-FD gradient (`fixed_dual_L`, always correct since it rebuilds the complete moment matrix via
`obj.moments!` every probe, matching `MinInd!`'s true tie-splitting behavior) for the ENTIRE D2-dim
gradient at that one point -- slower (32 full moment rebuilds instead of ~1) but always correct, per
this file's own "correctness over speed on rare edge cases" precedent. `meta.tie_fallback` reports
whether this path was taken.
"""
function composite_gradient_at_fast(x_free0::AbstractVector, ctx, pe;
        base::Union{Nothing,BaseDualState} = nothing, threaded::Bool = false,
        h_mode::Symbol = :adaptive, h0::Float64 = 0.01,
        bandwidth_cache::Union{Nothing,Dict{Int,Float64}} = nothing,
        tie_fallback_h::Float64 = 0.01)
    h_mode in (:adaptive, :fixed, :cached) || error("composite_gradient_at_fast: h_mode must be :adaptive|:fixed|:cached, got $h_mode")
    h_mode == :cached && bandwidth_cache === nothing && error("composite_gradient_at_fast: h_mode=:cached requires a bandwidth_cache Dict")

    base = base === nothing ? solve_base_state(x_free0, ctx) : base
    local cache
    try
        cache = build_lfix_base_cache(x_free0, ctx, base)
    catch e
        e isa TiedWinnerError || rethrow()
        g_fb, meta_fb = full_rebuild_gradient_fallback(x_free0, ctx, pe, base; h = tie_fallback_h)
        return g_fb, merge(meta_fb, (tie_fallback = true, tie_error = e))
    end
    D = ctx.D; D2 = D^2
    z0 = log.(reshape(x_free0[2:end], D, D))
    w0 = vcat(x_free0[1], pivot_reduce(z0, pe))

    g = zeros(D2)
    g[1] = gamma_component_analytic(cache, base, w0[1])

    h_used = zeros(D2); switch_mass = fill(NaN, D2); slope_ratio = fill(NaN, D2); cache_hits = falses(D2)

    "One coordinate's worth of work -- called either serially or under Threads.@threads, writes only to its OWN index k of the pre-allocated output arrays (thread-safe by construction, no shared mutable state)."
    function do_coord!(k::Int)
        if h_mode == :fixed
            h = h0
            h_used[k] = h
            g[k] = a_block_fd_component(cache, ctx, pe, w0, k, h)
            # slope_ratio/switch_mass intentionally left NaN -- diagnostic-only, skipped for speed
        elseif h_mode == :cached
            if haskey(bandwidth_cache, k)
                h = bandwidth_cache[k]
                cache_hits[k] = true
            else
                h, m, _ = select_bandwidth(cache, ctx, pe, w0, k)
                bandwidth_cache[k] = h
                switch_mass[k] = m
                cache_hits[k] = false
            end
            h_used[k] = h
            g[k] = a_block_fd_component(cache, ctx, pe, w0, k, h)
        else   # :adaptive -- byte-for-byte the ORIGINAL composite_gradient_at computation
            h, m, _ = select_bandwidth(cache, ctx, pe, w0, k)
            h_used[k] = h; switch_mass[k] = m
            g[k] = a_block_fd_component(cache, ctx, pe, w0, k, h)
            g_half = a_block_fd_component(cache, ctx, pe, w0, k, h / 2)
            denom = max(abs(g[k]), abs(g_half), 1e-12)
            slope_ratio[k] = abs(g[k] - g_half) / denom
        end
        return nothing
    end

    if threaded
        Threads.@threads for k in 2:D2
            do_coord!(k)
        end
    else
        for k in 2:D2
            do_coord!(k)
        end
    end

    return g, (base = base, cache = cache, w0 = w0, h_used = h_used, switch_mass = switch_mass,
               slope_ratio = slope_ratio, winner0 = copy(cache.winner0), gamma_component = g[1],
               h_mode = h_mode, threaded = threaded, cache_hits = cache_hits, tie_fallback = false)
end

"""
    full_rebuild_gradient_fallback(x_free0, ctx, pe, base; h=0.01) -> (g, meta)

Always-correct D2-dim central-FD gradient of `L_fix` (`fixed_dual_L`, full moment-matrix rebuild
every probe -- reuses `three_way_derivatives.jl`'s already-validated construction, not a new formula)
in the reduced pivot-eliminated coordinates. Used ONLY as the `TiedWinnerError` fallback -- correct
regardless of ties (since `obj.moments!`/`MinInd!` are called directly, with their true tie-splitting
behavior intact), at the cost of the O(n_free) full-rebuild cost this investigation's incremental
machinery was built to avoid. `threaded` over the 2*D2 probes (safe: each probe is a fresh
`obj.moments!` call into thread-local `K,G` buffers, no shared mutable state touched -- same argument
`profile_lfix_tiers.jl` already established for FD-probe-level threading).
"""
function full_rebuild_gradient_fallback(x_free0::AbstractVector, ctx, pe, base::BaseDualState; h::Float64 = 0.01, threaded::Bool = true)
    D = ctx.D; D2 = D^2
    z0 = log.(reshape(x_free0[2:end], D, D))
    w0 = vcat(x_free0[1], pivot_reduce(z0, pe))
    x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

    g = zeros(D2)
    function do_coord!(k::Int)
        wp = copy(w0); wp[k] += h
        wm = copy(w0); wm[k] -= h
        Lp = fixed_dual_L(x_free_from_w(wp), ctx, base)
        Lm = fixed_dual_L(x_free_from_w(wm), ctx, base)
        g[k] = (Lp - Lm) / (2h)
        return nothing
    end
    if threaded
        Threads.@threads for k in 1:D2
            do_coord!(k)
        end
    else
        for k in 1:D2
            do_coord!(k)
        end
    end
    return g, (base = base, w0 = w0, h_used = fill(h, D2), method = :full_rebuild_fallback, threaded = threaded)
end
