# ============================================================================
# Continuation 8, Section 3 (winner-margin certificate): equivalence test for
# the PersistentWinnerCache wiring (winner_certificate.jl) and its consumers
# in composite_gradient_fast.jl (lfix_value_certified, winner_cache_mode).
#
#   (A) lfix_value_certified's WINNER MATRIX is bit-identical to the trusted
#       full per-destination rescan (dest_contrib_block_local-based), and its
#       L_FIX VALUE matches to a tight documented tolerance (see
#       lfix_value_certified's own docstring for why it is not literally
#       bit-identical: winners_from_certificate's vectorized constCons_matrix
#       AodPow computation vs. price_and_pTsigma_cell's scalar formula --
#       same already-accepted ~1e-12-class floating-point-path difference
#       test_winner_certificate.jl's test (D) established for wval itself).
#       Checked BOTH cold (fresh PersistentWinnerCache per point) and WARM
#       (one PersistentWinnerCache reused across many perturbed points,
#       simulating a line-search / profile-continuation sweep), across
#       accepted / line-search / continuation / far step magnitudes.
#   (B) composite_gradient_at_fast's RETURNED GRADIENT `g` is bit-identical
#       whether winner_cache_mode=:none or :certificate -- the diagnostic
#       flag must NEVER perturb the actual gradient (cache use only
#       accelerates evaluation, per the standing brief's explicit
#       requirement), only add a `winner_cert_stats` diagnostic to `meta`.
#   (C) Certified/rescanned/fallback fraction breakdown (winner_cache_report)
#       reported for a warm multi-point sweep, for the benchmark's own
#       cross-check.
#
# The TRUSTED reference throughout is dest_contrib_block_local (Tier 1, a
# full O(D) per-draw rescan for every destination -- pre-existing, untouched
# this continuation) chained the SAME way lfix_incremental_at already does.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
using Random, Printf, LinearAlgebra

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2; W = size(ctx.U, 1)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
w_low = [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385]

"Trusted, uncached L_fix VALUE at an ARBITRARY (possibly all-D^2-cells-changed) point, via the pre-existing full per-destination rescan (dest_contrib_block_local, Tier 1) -- the same reference lfix_incremental_at's own tiers are validated against in test_lfix_incremental.jl."
function lfix_value_full_rebuild(cache::LFixBaseCache, ctx, x_free′::AbstractVector)
    D = cache.D
    θ_full′ = CS.reconstruct_full(x_free′, ctx.m)
    q = copy(cache.q0)
    for d in 1:D
        new_contrib = dest_contrib_block_local(cache, ctx, θ_full′, d)
        q .-= new_contrib .- @view(cache.contrib0[:, d])
    end
    new_cf = cf_contrib_at(cache, θ_full′, ctx)
    q .-= new_cf .- cache.cf_contrib0
    return lfix_from_q(q, cache.ζstar)
end

"Trusted winner matrix at x_free' via compute_winners_fast (the same reference winner_certificate.jl's own tests use)."
trusted_winner(x_free′) = compute_winners_fast(CS.reconstruct_full(x_free′, ctx.m), ctx)[1]

all_ok = true
rng = MersenneTwister(20260718)
VALUE_TOL = 1e-8   # generous vs the documented ~1e-12-class floating-point-path difference

println("="^100)
println("WINNER-MARGIN CERTIFICATE WIRING EQUIVALENCE (D=$D, W=$W)")
println("="^100)

# ---- (A) lfix_value_certified vs trusted full rebuild: COLD and WARM ----
println("(A) lfix_value_certified vs trusted full rebuild (dest_contrib_block_local), winner exact + value tight-tolerance")
for (lbl, w0) in (("upper40", w_up40), ("lower", w_low))
    xf0 = x_free_from_w(w0)
    base = solve_base_state(xf0, ctx)
    cache = build_lfix_base_cache(xf0, ctx, base)

    steps = [("accept 1e-3", 1e-3), ("linesrch 2e-2", 2e-2), ("linesrch 5e-2", 5e-2),
             ("continuation 1e-1", 1e-1), ("continuation 2e-1", 2e-1), ("far 5e-1", 5e-1)]

    # -- COLD: fresh PersistentWinnerCache per point (no reuse across calls) --
    worst_winner_cold = 0; worst_val_cold = 0.0
    for (sz_lbl, sz) in steps
        for rep in 1:3
            w′ = copy(w0); w′ .+= sz .* randn(rng, length(w0))
            xf′ = x_free_from_w(w′)
            wc_cold = PersistentWinnerCache()
            Lval, winner′, stats = lfix_value_certified(cache, wc_cold, ctx, xf′)
            Lref = lfix_value_full_rebuild(cache, ctx, xf′)
            wref = trusted_winner(xf′)
            worst_winner_cold = max(worst_winner_cold, maximum(abs.(winner′ .- wref)))
            worst_val_cold = max(worst_val_cold, abs(Lval - Lref) / max(abs(Lref), 1e-8))
        end
    end
    ok_cold = (worst_winner_cold == 0) && (worst_val_cold < VALUE_TOL)
    global all_ok &= ok_cold
    @printf("  [%-8s COLD ] worst_winner_absdiff=%d  worst_value_relerr=%.3e (tol=%.1e)  %s\n",
            lbl, worst_winner_cold, worst_val_cold, VALUE_TOL, ok_cold ? "PASS" : "FAIL")

    # -- WARM: ONE PersistentWinnerCache reused across ALL points below (simulates a
    #    line-search / profile-continuation sweep away from a common base point) --
    wc_warm = PersistentWinnerCache(tol_far = 0.3)
    worst_winner_warm = 0; worst_val_warm = 0.0
    for (sz_lbl, sz) in steps
        for rep in 1:3
            w′ = copy(w0); w′ .+= sz .* randn(rng, length(w0))
            xf′ = x_free_from_w(w′)
            Lval, winner′, stats = lfix_value_certified(cache, wc_warm, ctx, xf′)
            Lref = lfix_value_full_rebuild(cache, ctx, xf′)
            wref = trusted_winner(xf′)
            worst_winner_warm = max(worst_winner_warm, maximum(abs.(winner′ .- wref)))
            worst_val_warm = max(worst_val_warm, abs(Lval - Lref) / max(abs(Lref), 1e-8))
        end
    end
    ok_warm = (worst_winner_warm == 0) && (worst_val_warm < VALUE_TOL)
    global all_ok &= ok_warm
    rpt = winner_cache_report(wc_warm)
    @printf("  [%-8s WARM ] worst_winner_absdiff=%d  worst_value_relerr=%.3e (tol=%.1e)  %s\n",
            lbl, worst_winner_warm, worst_val_warm, VALUE_TOL, ok_warm ? "PASS" : "FAIL")
    @printf("  [%-8s WARM ] n_calls=%d  certified_frac=%.4f  rescanned_frac=%.4f  full_fallback_call_frac=%.4f  n_rebuilds=%d  implied_speedup=%.2fx\n",
            lbl, rpt.n_calls, rpt.certified_frac, rpt.rescanned_frac, rpt.full_fallback_call_frac, rpt.n_rebuilds, rpt.implied_speedup)
end

# ---- (B) winner_cache_mode never perturbs the returned gradient ----
println("-"^100)
println("(B) composite_gradient_at_fast: g bit-identical for winner_cache_mode=:none vs :certificate")
for (lbl, w0) in (("upper40", w_up40), ("lower", w_low))
    xf0 = x_free_from_w(w0)
    base = solve_base_state(xf0, ctx)
    g_none, meta_none = composite_gradient_at_fast(xf0, ctx, pe; base = base, threaded = true, h_mode = :fixed)
    wc = PersistentWinnerCache()
    g_cert, meta_cert = composite_gradient_at_fast(xf0, ctx, pe; base = base, threaded = true, h_mode = :fixed,
                                                     winner_cache_mode = :certificate, winner_cache = wc)
    gdiff = maximum(abs.(g_none .- g_cert))
    stats_present = meta_cert.winner_cert_stats !== nothing && meta_none.winner_cert_stats === nothing
    ok = (gdiff == 0.0) && stats_present
    global all_ok &= ok
    @printf("  [%-8s] max|g_none - g_certificate|=%.3e  stats_only_when_requested=%s  %s\n", lbl, gdiff, stats_present, ok ? "PASS" : "FAIL")
end

# ---- (C) warm multi-point sweep certified/rescanned/fallback breakdown (for the benchmark's cross-check) ----
println("-"^100)
println("(C) warm 30-point line-search-magnitude sweep: certified/rescanned/fallback breakdown")
let
    xf0 = x_free_from_w(w_up40)
    base = solve_base_state(xf0, ctx)
    cache = build_lfix_base_cache(xf0, ctx, base)
    wc = PersistentWinnerCache(tol_far = 0.3)
    for rep in 1:30
        w′ = copy(w_up40); w′ .+= 2e-2 .* randn(rng, length(w_up40))
        xf′ = x_free_from_w(w′)
        lfix_value_certified(cache, wc, ctx, xf′)
    end
    rpt = winner_cache_report(wc)
    @printf("  n_calls=%d  certified_frac=%.4f  rescanned_frac=%.4f  full_fallback_call_frac=%.4f  n_rebuilds=%d\n",
            rpt.n_calls, rpt.certified_frac, rpt.rescanned_frac, rpt.full_fallback_call_frac, rpt.n_rebuilds)
    @printf("  total_cert_s=%.4f  total_full_s=%.4f  implied_speedup=%.2fx\n", rpt.total_cert_s, rpt.total_full_s, rpt.implied_speedup)
end

println("="^100)
if all_ok
    println("ALL WINNER-ACCELERATOR WIRING EQUIVALENCE TESTS PASSED")
else
    println("SOME WINNER-ACCELERATOR WIRING EQUIVALENCE TESTS FAILED"); exit(1)
end
