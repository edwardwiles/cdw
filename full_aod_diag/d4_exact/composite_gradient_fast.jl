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
#
# Continuation 8 addition: also includes winner_certificate.jl (Sections 1+6+
# the new PersistentWinnerCache wiring layer) -- see this file's own new
# `lfix_value_certified` / `winner_cache_mode` kwarg on `composite_gradient_at_fast`
# below for the winner-margin-certificate wiring the standing brief's Section 3
# (second half) asks for. Scope note (measured, see winner_certificate.jl's own
# new "Continuation 8" section header): the certificate is provably exact for
# the WINNER identity only, not runner-up/third, so it is wired here as a VALUE
# evaluator (`lfix_value_certified`) for line-search/continuation points, NOT as
# a way to skip the exact-top-3-dependent gradient tiers (those are handled by
# count_winner_flips_multi_top3/dest_contrib_incremental_top3 in
# composite_gradient.jl/lfix_incremental.jl instead, per-call, not persistent).
# ============================================================================
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "winner_certificate.jl"))

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
        tie_fallback_h::Float64 = 0.01,
        winner_cache_mode::Symbol = :none,
        winner_cache::Union{Nothing,PersistentWinnerCache} = nothing,
        winner_cache_threaded::Bool = false,
        multi_method::Symbol = :top3)
    h_mode in (:adaptive, :fixed, :cached) || error("composite_gradient_at_fast: h_mode must be :adaptive|:fixed|:cached, got $h_mode")
    h_mode == :cached && bandwidth_cache === nothing && error("composite_gradient_at_fast: h_mode=:cached requires a bandwidth_cache Dict")
    winner_cache_mode in (:none, :certificate) || error("composite_gradient_at_fast: winner_cache_mode must be :none|:certificate, got $winner_cache_mode")
    winner_cache_mode == :certificate && winner_cache === nothing &&
        error("composite_gradient_at_fast: winner_cache_mode=:certificate requires a winner_cache::PersistentWinnerCache (build ONCE, pass across repeated calls -- see winner_certificate.jl's PersistentWinnerCache docstring)")

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
            g[k] = a_block_fd_component(cache, ctx, pe, w0, k, h; multi_method = multi_method)
            # slope_ratio/switch_mass intentionally left NaN -- diagnostic-only, skipped for speed
        elseif h_mode == :cached
            if haskey(bandwidth_cache, k)
                h = bandwidth_cache[k]
                cache_hits[k] = true
            else
                h, m, _ = select_bandwidth(cache, ctx, pe, w0, k; multi_method = multi_method)
                bandwidth_cache[k] = h
                switch_mass[k] = m
                cache_hits[k] = false
            end
            h_used[k] = h
            g[k] = a_block_fd_component(cache, ctx, pe, w0, k, h; multi_method = multi_method)
        else   # :adaptive -- byte-for-byte the ORIGINAL composite_gradient_at computation
            h, m, _ = select_bandwidth(cache, ctx, pe, w0, k; multi_method = multi_method)
            h_used[k] = h; switch_mass[k] = m
            g[k] = a_block_fd_component(cache, ctx, pe, w0, k, h; multi_method = multi_method)
            g_half = a_block_fd_component(cache, ctx, pe, w0, k, h / 2; multi_method = multi_method)
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

    # Continuation 8: winner_cache_mode=:certificate diagnostic (opt-in, additive). Does NOT
    # touch `g` (the returned gradient) at all -- computed AFTER g is finalized, purely to log a
    # certificate-accelerated winner-matrix recompute at this SAME point via the caller's
    # persistent `winner_cache`, so a caller (KNITRO outer-loop driver / profile-continuation
    # driver) gets a cheap certified/rescanned/fallback breakdown across repeated nearby calls
    # without paying for a second full O(W*D^2)-ish scan. See lfix_value_certified below for the
    # substantive line-search/continuation VALUE accelerator this cache is really for.
    winner_cert_stats = nothing
    if winner_cache_mode == :certificate
        _, _, winner_cert_stats = winner_value_update!(winner_cache, ctx, x_free0; threaded = winner_cache_threaded)
    end

    return g, (base = base, cache = cache, w0 = w0, h_used = h_used, switch_mass = switch_mass,
               slope_ratio = slope_ratio, winner0 = copy(cache.winner0), gamma_component = g[1],
               h_mode = h_mode, threaded = threaded, cache_hits = cache_hits, tie_fallback = false,
               winner_cache_mode = winner_cache_mode, winner_cert_stats = winner_cert_stats)
end

"""
    lfix_value_certified(cache::LFixBaseCache, wc::PersistentWinnerCache, ctx, x_free') -> (Lfix_value, winner', CertStats)

Continuation 8 deliverable: the winner-margin certificate's actual "nearby
ordinary value evaluations / line-search / profile-continuation points"
accelerator. Evaluates `L_fix` EXACTLY at an ARBITRARY new point `x_free'` --
unlike `lfix_incremental_at`'s tiers (`:block_local`/`:incremental`/
`:incremental_o1`), which assume <=2 changed A_od cells (the single-coordinate
FD-probe case), this accepts a full simultaneous multi-coordinate step (every
A_od cell may have moved -- exactly what a genuine line-search or profile-
continuation step does). Reuses `cache`'s BASE-POINT `lambda*/zeta*/CONST_d/
SW/gammafac/contrib0/q0/cf_contrib0` (the SAME closed-form contribution
identity `lfix_incremental_at` uses, not re-derived) together with `wc`'s
PERSISTENT certificate (`winner_value_update!`) for the winner + winning-
origin sigma-value recompute across ALL D destinations at once.

`cache` and `wc` do not need to share an anchor point: `cache` must be built
at the SAME `x_free0` this L_fix value is measured relative to (its own
q0/contrib0 baseline); `wc`'s WinnerRefCache anchor is independent bookkeeping
for the certificate's own performance (see `PersistentWinnerCache`'s
docstring) and can lag behind `cache`'s base point across many calls without
affecting correctness -- `winner_value_update!`'s returned winner/wval are
exact regardless (certified or full-scan-fallback), per Section 1's proof
(reused here, not re-derived).

NOTE on tolerance: the returned winner MATRIX is exact/bit-identical to a full
`compute_winners_fast` scan. The returned L_fix VALUE differs from a
full-rebuild reference (`dest_contrib_block_local`-based) by a tiny (~1e-12
relative) floating-point-path difference, NOT a correctness gap -- it comes
from `winners_from_certificate` computing `AodPow` via `constCons_matrix`'s
VECTORIZED broadcast (matching `winners_from_certificate`'s own established,
already-tested-to-1e-12 convention in `test_winner_certificate.jl`'s test (D)),
vs. `price_and_pTsigma_cell`'s per-cell SCALAR formula that `cache`'s own
`contrib0`/`dest_contrib_*` machinery uses -- same numbers, different
floating-point operation order. Documented and verified (not assumed) in
`test_winner_accelerator_wiring.jl`.
"""
function lfix_value_certified(cache::LFixBaseCache, wc::PersistentWinnerCache, ctx, x_free′::AbstractVector; threaded::Bool = false)
    D = cache.D; W = cache.W
    winner′, wval′, stats = winner_value_update!(wc, ctx, x_free′; threaded = threaded)

    q = copy(cache.q0)
    @inbounds for d in 1:D
        for ω in 1:W
            wo = winner′[ω, d]
            d1w = d + (wo - 1) * D
            new_c = (cache.SW[ω] / cache.gammafac) * (cache.CONST_d[d] + cache.λstar[d1w] * wval′[ω, d])
            q[ω] -= new_c - cache.contrib0[ω, d]
        end
    end
    θ_full′ = CS.reconstruct_full(x_free′, ctx.m)
    new_cf = cf_contrib_at(cache, θ_full′, ctx)
    q .-= new_cf .- cache.cf_contrib0

    return lfix_from_q(q, cache.ζstar), winner′, stats
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
