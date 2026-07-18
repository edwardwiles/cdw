# ============================================================================
# Continuation 8: end-to-end benchmark for the two winner-accelerator wirings
# this continuation adds -- NOT just the isolated winner-computation component
# Continuation 7 already measured (docs/winner_certificate_report.md), but the
# actual FULL hard-L_fix coordinate gradient / value evaluation cost.
#
#   PART 1: coordinate-specialized top-3 update (composite_gradient.jl /
#   lfix_incremental.jl). Measures composite_gradient_at_fast's FULL wall time
#   (all 15 A-block coordinates, including the h-selection bisection) with the
#   NEW default (multi_method=:top3) vs the ORIGINAL O(D) fallback
#   (multi_method=:generic), at threaded=true/h_mode=:fixed and h_mode=
#   :adaptive, real (g,A) points.
#
#   PART 2: winner-margin certificate value evaluator (lfix_value_certified).
#   Simulates a line-search / profile-continuation sweep of N nearby points
#   away from a fixed base cache, comparing a WARM PersistentWinnerCache
#   (reused across all N points) against the trusted uncached full rebuild
#   (dest_contrib_block_local-based, dense O(D) per-destination rescan every
#   point) -- reports wall time, speedup, and the certified/rescanned/
#   full-fallback fraction breakdown (PersistentWinnerCache's own counters).
#
# JULIA_NUM_THREADS should be set >= the number of A-block coordinates (15 at
# D=4) to exercise Part 1's threaded=true lever meaningfully; run on
# demand.mit.edu with .knitro_env.sh sourced (KNITRO-touching: solve_base_state).
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

@printf("JULIA_NUM_THREADS = %d\n", Threads.nthreads())
xf0 = x_free_from_w(w_up40)
base = solve_base_state(xf0, ctx)
cache = build_lfix_base_cache(xf0, ctx, base)

"min over N reps after a warm-up call (JIT-robust timing)."
function time_min(f, N)
    f()
    ts = Float64[]
    for _ in 1:N
        t0 = time_ns()
        f()
        push!(ts, (time_ns() - t0) / 1e9)
    end
    return minimum(ts), sum(ts) / N
end

println("="^100)
println("PART 1: full composite_gradient_at_fast coordinate-gradient speedup, top3 vs original O(D) fallback")
println("="^100)
N = 15
for (mm_lbl, mm) in (("multi_method=:generic (ORIGINAL O(D) fallback)", :generic),
                      ("multi_method=:top3    (NEW default)", :top3))
    for (hm_lbl, hm) in (("h_mode=:fixed", :fixed), ("h_mode=:adaptive", :adaptive))
        for thr in (false, true)
            mn, mu = time_min(() -> composite_gradient_at_fast(xf0, ctx, pe; base = base, threaded = thr, h_mode = hm, multi_method = mm), N)
            @printf("  %-42s %-18s threaded=%-5s  min=%7.3f ms  mean=%7.3f ms\n", mm_lbl, hm_lbl, thr, mn*1000, mu*1000)
        end
    end
end

# explicit speedup ratios at the two production-relevant settings
mn_generic_fixed_thr, _ = time_min(() -> composite_gradient_at_fast(xf0, ctx, pe; base = base, threaded = true, h_mode = :fixed, multi_method = :generic), N)
mn_top3_fixed_thr, _ = time_min(() -> composite_gradient_at_fast(xf0, ctx, pe; base = base, threaded = true, h_mode = :fixed, multi_method = :top3), N)
mn_generic_adapt_thr, _ = time_min(() -> composite_gradient_at_fast(xf0, ctx, pe; base = base, threaded = true, h_mode = :adaptive, multi_method = :generic), N)
mn_top3_adapt_thr, _ = time_min(() -> composite_gradient_at_fast(xf0, ctx, pe; base = base, threaded = true, h_mode = :adaptive, multi_method = :top3), N)
@printf("\n  SPEEDUP (threaded, h_mode=:fixed):    %.3fms -> %.3fms  = %.3fx\n", mn_generic_fixed_thr*1000, mn_top3_fixed_thr*1000, mn_generic_fixed_thr/mn_top3_fixed_thr)
@printf("  SPEEDUP (threaded, h_mode=:adaptive): %.3fms -> %.3fms  = %.3fx\n", mn_generic_adapt_thr*1000, mn_top3_adapt_thr*1000, mn_generic_adapt_thr/mn_top3_adapt_thr)

println()
println("="^100)
println("PART 2: winner-margin certificate value-eval speedup, warm persistent cache vs uncached full rebuild")
println("="^100)

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

rng = MersenneTwister(20260718)
n_points = 40
step_sizes = [1e-3, 5e-3, 1e-2, 2e-2, 5e-2]   # accepted-to-line-search magnitude mix, matching a real KNITRO trajectory / continuation sweep
points = Vector{Vector{Float64}}(undef, n_points)
for i in 1:n_points
    w′ = copy(w_up40); w′ .+= rand(rng, step_sizes) .* randn(rng, length(w_up40))
    points[i] = x_free_from_w(w′)
end

# --- uncached full rebuild baseline, timed over the whole sweep ---
function run_full_rebuild_sweep()
    for xf′ in points
        lfix_value_full_rebuild(cache, ctx, xf′)
    end
end
mn_full, mu_full = time_min(run_full_rebuild_sweep, 8)
@printf("  uncached full-rebuild sweep (%d points): min=%7.3f ms total (%.4f ms/point)  mean=%7.3f ms\n",
        n_points, mn_full*1000, mn_full*1000/n_points, mu_full*1000)

# --- warm persistent-cache sweep (ONE PersistentWinnerCache reused across all n_points) ---
function run_certified_sweep()
    wc = PersistentWinnerCache(tol_far = 0.3)
    for xf′ in points
        lfix_value_certified(cache, wc, ctx, xf′)
    end
    return wc
end
# warm-up (JIT) + discard
run_certified_sweep()
ts = Float64[]
local wc_final
for _ in 1:8
    t0 = time_ns()
    global wc_final = run_certified_sweep()
    push!(ts, (time_ns() - t0) / 1e9)
end
mn_cert = minimum(ts); mu_cert = sum(ts) / length(ts)
@printf("  WARM certified sweep        (%d points): min=%7.3f ms total (%.4f ms/point)  mean=%7.3f ms\n",
        n_points, mn_cert*1000, mn_cert*1000/n_points, mu_cert*1000)

@printf("\n  SPEEDUP (full sweep, warm cache reused across all points): %.3fms -> %.3fms  = %.3fx\n", mn_full*1000, mn_cert*1000, mn_full/mn_cert)

rpt = winner_cache_report(wc_final)
println()
println("  certified/rescanned/fallback breakdown (from the final timed sweep's PersistentWinnerCache):")
@printf("    n_calls=%d  n_cells_total=%d\n", rpt.n_calls, rpt.n_cells_total)
@printf("    certified_frac=%.4f  rescanned_frac=%.4f  full_fallback_call_frac=%.4f  n_rebuilds=%d\n",
        rpt.certified_frac, rpt.rescanned_frac, rpt.full_fallback_call_frac, rpt.n_rebuilds)
@printf("    total_cert_s=%.5f  total_full_s=%.5f (within this ONE sweep's internal accounting)\n", rpt.total_cert_s, rpt.total_full_s)

println()
println("="^100)
println("BENCHMARK COMPLETE")
println("="^100)
