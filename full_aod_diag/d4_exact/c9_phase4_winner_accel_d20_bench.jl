# ============================================================================
# Continuation 9, Phase 4: winner-margin certificate + coordinate-specialized
# top-3 update, tested at REAL D=20/W=80,000 for the first time (Continuation
# 8 validated both only at D=4). Reuses benchmark_winner_accelerator.jl's own
# Part 1/Part 2 structure directly (same functions, same comparison design),
# scaled down in rep-count/point-count given D=20's much larger per-call cost
# (a full composite_gradient_at_fast call is ~4-6.5s at D=20 per the W80k
# microbenchmark, vs ~15-30ms at D=4).
#
# ADDS (per the Phase 4 brief's explicit correctness requirement, not present
# in the D=4 benchmark_winner_accelerator.jl): an ORDER-RANDOMIZED test that
# the SAME set of line-search points evaluated through a warm
# PersistentWinnerCache in two DIFFERENT orders gives IDENTICAL per-point
# values -- cache use must never make results depend on evaluation order.
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))   # -> includes context.jl exactly once
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
using Statistics, Printf, Dates, Random, LinearAlgebra

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c9_phase4_winner_accel_d20")
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...)
    println(LOGIO, xs...)
    flush(stdout); flush(LOGIO)
end
logprint("c9_phase4_winner_accel_d20_bench.jl starting at ", now(), "  commit=", COMMIT, "  nthreads=", Threads.nthreads())

function vmhwm_kb()
    try
        for line in eachline("/proc/self/status")
            startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
        end
    catch
    end
    return -1
end

t0 = time()
ctx = d20_real_setup(W = 80000)
t_setup = time() - t0
D = ctx.D; W = size(ctx.U, 1)
logprint("d20_real_setup(W=80000) wall = ", round(t_setup, digits = 2), "s  VmHWM=", round(vmhwm_kb() / 1e6, digits = 2), " GB")

pe = build_pivot_elimination(ctx)
xf0 = ctx.θ0_up[ctx.free_idx]
base = solve_base_state(xf0, ctx)
cache = build_lfix_base_cache(xf0, ctx, base)
z0 = log.(reshape(xf0[2:end], D, D)); w0 = vcat(xf0[1], pivot_reduce(z0, pe))
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
logprint("D=", D, "  D^2 (A-block coords)=", D^2, "  W=", W)

"min over N reps after a warm-up call."
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

# ============================================================================
# PART 1: full composite_gradient_at_fast, top3 vs generic, at D=20 (399
# A-block coordinates, vs D=4's 15 -- the scale Continuation 8 predicted
# might show a real end-to-end win, not yet tested).
# ============================================================================
logprint("\n", "="^100); logprint("PART 1: full 400-coordinate gradient, top3 vs generic multi_method, D=20"); logprint("="^100)
N1 = 3
part1_rows = NamedTuple[]
for (hm_lbl, hm) in (("h_mode=:fixed", :fixed), ("h_mode=:adaptive", :adaptive))
    mn_gen, mu_gen = time_min(() -> composite_gradient_at_fast(xf0, ctx, pe; base = base, threaded = true, h_mode = hm, multi_method = :generic), N1)
    mn_top3, mu_top3 = time_min(() -> composite_gradient_at_fast(xf0, ctx, pe; base = base, threaded = true, h_mode = hm, multi_method = :top3), N1)
    logprint(@sprintf("  %-18s generic: min=%7.3fs mean=%7.3fs  |  top3: min=%7.3fs mean=%7.3fs  |  speedup(min)=%.3fx",
              hm_lbl, mn_gen, mu_gen, mn_top3, mu_top3, mn_gen / mn_top3))
    push!(part1_rows, (h_mode = string(hm), generic_min_s = mn_gen, generic_mean_s = mu_gen,
                       top3_min_s = mn_top3, top3_mean_s = mu_top3, speedup_min = mn_gen / mn_top3))
end
write_csv_rows(joinpath(OUTDIR, "part1_top3_vs_generic.csv"), part1_rows)

# ============================================================================
# PART 2: winner-margin certificate value-eval speedup, D=20.
# ============================================================================
logprint("\n", "="^100); logprint("PART 2: winner-margin certificate, warm persistent cache vs uncached full rebuild, D=20"); logprint("="^100)

function lfix_value_full_rebuild(cache::LFixBaseCache, ctx, x_free′::AbstractVector)
    Dloc = cache.D
    θ_full′ = CS.reconstruct_full(x_free′, ctx.m)
    q = copy(cache.q0)
    for d in 1:Dloc
        new_contrib = dest_contrib_block_local(cache, ctx, θ_full′, d)
        q .-= new_contrib .- @view(cache.contrib0[:, d])
    end
    new_cf = cf_contrib_at(cache, θ_full′, ctx)
    q .-= new_cf .- cache.cf_contrib0
    return lfix_from_q(q, cache.ζstar)
end

rng = MersenneTwister(20260718)
n_points = 15   # scaled down from D=4's 40 given D=20's much larger per-point cost
step_sizes = [1e-3, 5e-3, 1e-2, 2e-2, 5e-2]
points = Vector{Vector{Float64}}(undef, n_points)
for i in 1:n_points
    w′ = copy(w0); w′ .+= rand(rng, step_sizes) .* randn(rng, length(w0))
    points[i] = x_free_from_w(w′)
end

function run_full_rebuild_sweep()
    for xf′ in points
        lfix_value_full_rebuild(cache, ctx, xf′)
    end
end
mn_full, mu_full = time_min(run_full_rebuild_sweep, 3)
logprint(@sprintf("  uncached full-rebuild sweep (%d points): min=%.3fs total (%.4fs/point)  mean=%.3fs",
          n_points, mn_full, mn_full/n_points, mu_full))

function run_certified_sweep()
    wc = PersistentWinnerCache(tol_far = 0.3)
    vals = Vector{Float64}(undef, n_points)
    for (i, xf′) in enumerate(points)
        vals[i], _, _ = lfix_value_certified(cache, wc, ctx, xf′)
    end
    return wc, vals
end
run_certified_sweep()   # warm-up (JIT)
ts = Float64[]
local wc_final, vals_inorder
for _ in 1:3
    t0 = time_ns()
    global wc_final, vals_inorder
    wc_final, vals_inorder = run_certified_sweep()
    push!(ts, (time_ns() - t0) / 1e9)
end
mn_cert = minimum(ts); mu_cert = sum(ts) / length(ts)
logprint(@sprintf("  WARM certified sweep        (%d points): min=%.3fs total (%.4fs/point)  mean=%.3fs",
          n_points, mn_cert, mn_cert/n_points, mu_cert))
logprint(@sprintf("\n  SPEEDUP (full sweep, warm cache reused across all points): %.3fs -> %.3fs = %.3fx", mn_full, mn_cert, mn_full/mn_cert))

rpt = winner_cache_report(wc_final)
logprint("\n  certified/rescanned/fallback breakdown (final timed sweep's PersistentWinnerCache):")
logprint(@sprintf("    n_calls=%d  n_cells_total=%d", rpt.n_calls, rpt.n_cells_total))
logprint(@sprintf("    certified_frac=%.4f  rescanned_frac=%.4f  full_fallback_call_frac=%.4f  n_rebuilds=%d",
          rpt.certified_frac, rpt.rescanned_frac, rpt.full_fallback_call_frac, rpt.n_rebuilds))
logprint(@sprintf("    total_cert_s=%.5f  total_full_s=%.5f", rpt.total_cert_s, rpt.total_full_s))

write_csv_rows(joinpath(OUTDIR, "part2_certificate_speedup.csv"),
    [(n_points = n_points, full_rebuild_min_s = mn_full, certified_min_s = mn_cert, speedup = mn_full/mn_cert,
      certified_frac = rpt.certified_frac, rescanned_frac = rpt.rescanned_frac,
      full_fallback_call_frac = rpt.full_fallback_call_frac, n_rebuilds = rpt.n_rebuilds)])

# ---- correctness: certified values vs the trusted full rebuild, same points ----
# (wrapped in a function -- a bare top-level `for` loop reassigning an
# already-existing global hit Julia's soft-scope ambiguity in an earlier
# version of this script, caught via UndefVarError, not a silent wrong
# result; fixed the same way as the Phase 3 gradbench script's analogous bug.)
function compute_maxdiff_vs_full(cache, ctx, points, vals_inorder)
    md = 0.0
    for (i, xf′) in enumerate(points)
        v_full = lfix_value_full_rebuild(cache, ctx, xf′)
        md = max(md, abs(vals_inorder[i] - v_full))
    end
    return md
end
maxdiff_vs_full = compute_maxdiff_vs_full(cache, ctx, points, vals_inorder)
logprint("\n  certified-sweep values vs trusted full-rebuild (same points): max|diff| = ", maxdiff_vs_full)

# ============================================================================
# PART 3 (Phase 4 explicit requirement): order-randomization correctness --
# same n_points, evaluated through a warm PersistentWinnerCache in a
# DIFFERENT (permuted) order, must give the SAME per-point value as the
# in-order sweep above. Cache use (certify/rescan/fallback/rebuild, all a
# PURE performance policy) must never change the returned value.
# ============================================================================
logprint("\n", "="^100); logprint("PART 3: order-randomization correctness check"); logprint("="^100)

Random.seed!(31337)
perm = randperm(n_points)
logprint("  permutation: ", perm)

wc_perm = PersistentWinnerCache(tol_far = 0.3)
vals_permorder = Vector{Float64}(undef, n_points)   # indexed by ORIGINAL point id
for i in perm
    v, _, _ = lfix_value_certified(cache, wc_perm, ctx, points[i])
    vals_permorder[i] = v
end

order_maxdiff = maximum(abs.(vals_inorder .- vals_permorder))
logprint("  max|value(in-order) - value(permuted-order)| over all ", n_points, " points = ", order_maxdiff)
order_ok = order_maxdiff == 0.0
logprint("  ORDER-INDEPENDENCE: ", order_ok ? "CONFIRMED (bit-identical regardless of evaluation order)" : "VIOLATED -- investigate before trusting the cache")

rpt_perm = winner_cache_report(wc_perm)
logprint(@sprintf("  permuted-order sweep's own cache stats: certified_frac=%.4f rescanned_frac=%.4f full_fallback_call_frac=%.4f n_rebuilds=%d",
          rpt_perm.certified_frac, rpt_perm.rescanned_frac, rpt_perm.full_fallback_call_frac, rpt_perm.n_rebuilds))
logprint("  (NOTE: the cache's internal STATS above may legitimately differ between orderings -- e.g. which point triggers",
         " the lazy first-build/rebuild -- only the RETURNED VALUES are required to match, which is what order_maxdiff checks.)")

write_csv_rows(joinpath(OUTDIR, "part3_order_randomization.csv"),
    [(n_points = n_points, order_maxdiff = order_maxdiff, order_independent = order_ok,
      inorder_certified_frac = rpt.certified_frac, permorder_certified_frac = rpt_perm.certified_frac)])

logprint("\nVmHWM at end of run = ", round(vmhwm_kb() / 1e6, digits = 2), " GB")
logprint("\nc9_phase4_winner_accel_d20_bench.jl COMPLETE at ", now())
close(LOGIO)
