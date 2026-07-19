# ============================================================================
# Continuation 9, Phase 5: bandwidth-selection optimization benchmark, real
# D=20/W=80,000 economy, France focal, natural-theta calibration point (same
# point as docs/fullA_D20_W80k_microbenchmark.md's Point 1).
#
# Compares select_bandwidth's bisection baseline (measured 15.7s/399 coords in
# the W80k doc) against:
#   - h_mode=:fixed          (composite_gradient_fast.jl, pre-existing)
#   - h_mode=:cached         (pre-existing MECHANISM) + BandwidthCachePolicy
#                              (bandwidth_cache_policy.jl, NEW this session --
#                               the staleness-detection POLICY the mechanism
#                               was missing)
#   - h_mode=:quantile       (bandwidth_quantile.jl, NEW -- closed-form
#                              order-statistic bandwidth selection)
# plus winner-switch-count/weighted-mass logging and directional-secant
# validation against real re-solved evaluate_fullA points, matching
# docs/fullA_D20_W80k_microbenchmark.md sec 3F's own methodology.
#
# SAFETY: per this worktree's standing instruction, a small memory sanity
# check runs immediately after context construction, before any heavier work.
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))
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
include(joinpath(@__DIR__, "bandwidth_cache_policy.jl"))
using Statistics, Printf, Dates, Random, LinearAlgebra

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c9_bandwidth_optimization")
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...)
    println(LOGIO, xs...)
    flush(stdout); flush(LOGIO)
end
logprint("c9_bandwidth_optimization_benchmark.jl starting at ", now(), "  commit=", COMMIT, "  nthreads=", Threads.nthreads())

function vmhwm_kb()
    try
        for line in eachline("/proc/self/status")
            startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
        end
    catch
    end
    return -1
end

const FEASIBLE_CODES = (0, -100, -101, -103)
x_free_from_w(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

# ============================================================================
# PART 0: setup + memory safety check (mandatory before anything heavier)
# ============================================================================
logprint("\n", "="^90); logprint("PART 0: context/setup, W=80000, memory safety check"); logprint("="^90)
t0 = time()
ctx = d20_real_setup(W = 80000)
t_setup = time() - t0
vmhwm1 = vmhwm_kb()
logprint("d20_real_setup(W=80000) wall = ", round(t_setup, digits = 2), "s   VmHWM = ", round(vmhwm1/1e6, digits=2), " GB")
if vmhwm1 / 1e6 > 10.0
    logprint("SAFETY ABORT: VmHWM after setup exceeds 10GB (", round(vmhwm1/1e6,digits=2), " GB) -- ",
             "expected ~2.6GB per docs/fullA_D20_W80k_microbenchmark.md sec 1. Stopping before any heavier work.")
    close(LOGIO)
    exit(1)
end
logprint("Memory safety check PASSED (VmHWM ", round(vmhwm1/1e6, digits=2), " GB, well under the 109GB pre-fix bug's scale).")

D = ctx.D; D2 = D^2
gp0 = ctx.θ0_up[3+D]
xf_nat = ctx.θ0_up[ctx.free_idx]
pe = build_pivot_elimination(ctx)
logprint("D=", D, "  D^2=", D2, "  gamma'_focal=", gp0)

logprint("\nWarming up (untimed)...")
t0 = time()
for _ in 1:2
    evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :dense)
end
base0 = solve_base_state(xf_nat, ctx)
composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = true, h_mode = :adaptive, multi_method = :top3)
composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = true, h_mode = :quantile, multi_method = :top3)
composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = false, h_mode = :fixed, multi_method = :top3)
d1 = Dict{Int,Float64}()
composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = true, h_mode = :cached, bandwidth_cache = d1, multi_method = :top3)
logprint("Warm-up complete, wall = ", round(time() - t0, digits = 1), "s")

r1 = evaluate_fullA(xf_nat, ctx; warm = true)
logprint("Point 1 (calibration): inner_status=", r1.inner_status, "  Delta_dual=", r1.Delta_dual)
@assert r1.inner_status in FEASIBLE_CODES "calibration point infeasible -- cannot proceed"

# ============================================================================
# PART 1: full-gradient wall-clock comparison, threaded=true, N=4/config
# ============================================================================
logprint("\n", "="^90); logprint("PART 1: full ", D2, "-coordinate gradient wall-clock, threaded=true, N=4/config"); logprint("="^90)

base1 = solve_base_state(xf_nat, ctx)
p1_rows = NamedTuple[]
meta_by_mode = Dict{Symbol,Any}()

logprint("\n-- h_mode=:adaptive (BASELINE, bisection) --")
times = Float64[@elapsed composite_gradient_at_fast(xf_nat, ctx, pe; base=base1, threaded=true, h_mode=:adaptive, multi_method=:top3) for _ in 1:4]
g_ad, meta_ad = composite_gradient_at_fast(xf_nat, ctx, pe; base=base1, threaded=true, h_mode=:adaptive, multi_method=:top3)
meta_by_mode[:adaptive] = (g=g_ad, meta=meta_ad)
logprint(@sprintf("  median=%.3fs  reps=%s", median(times), round.(times,digits=2)))
push!(p1_rows, (h_mode="adaptive", median_s=median(times), mean_s=mean(times), min_s=minimum(times), max_s=maximum(times)))

logprint("\n-- h_mode=:quantile (NEW, closed-form order-statistic) --")
times = Float64[@elapsed composite_gradient_at_fast(xf_nat, ctx, pe; base=base1, threaded=true, h_mode=:quantile, multi_method=:top3) for _ in 1:4]
g_q, meta_q = composite_gradient_at_fast(xf_nat, ctx, pe; base=base1, threaded=true, h_mode=:quantile, multi_method=:top3)
meta_by_mode[:quantile] = (g=g_q, meta=meta_q)
logprint(@sprintf("  median=%.3fs  reps=%s", median(times), round.(times,digits=2)))
push!(p1_rows, (h_mode="quantile", median_s=median(times), mean_s=mean(times), min_s=minimum(times), max_s=maximum(times)))

logprint("\n-- h_mode=:fixed (h0=0.01, pre-existing) --")
times = Float64[@elapsed composite_gradient_at_fast(xf_nat, ctx, pe; base=base1, threaded=true, h_mode=:fixed, multi_method=:top3) for _ in 1:4]
g_fx, meta_fx = composite_gradient_at_fast(xf_nat, ctx, pe; base=base1, threaded=true, h_mode=:fixed, multi_method=:top3)
meta_by_mode[:fixed] = (g=g_fx, meta=meta_fx)
logprint(@sprintf("  median=%.3fs  reps=%s", median(times), round.(times,digits=2)))
push!(p1_rows, (h_mode="fixed", median_s=median(times), mean_s=mean(times), min_s=minimum(times), max_s=maximum(times)))

logprint("\n-- h_mode=:cached, COLD (empty Dict, every coord a miss -- should ~= :adaptive minus h/2 diagnostic) --")
times = Float64[]
for _ in 1:4
    dcold = Dict{Int,Float64}()
    push!(times, @elapsed composite_gradient_at_fast(xf_nat, ctx, pe; base=base1, threaded=true, h_mode=:cached, bandwidth_cache=dcold, multi_method=:top3))
end
logprint(@sprintf("  median=%.3fs  reps=%s", median(times), round.(times,digits=2)))
push!(p1_rows, (h_mode="cached_cold", median_s=median(times), mean_s=mean(times), min_s=minimum(times), max_s=maximum(times)))

logprint("\n-- h_mode=:cached, WARM (Dict pre-populated from a prior call at the SAME point -- every coord a hit) --")
dwarm = Dict{Int,Float64}()
composite_gradient_at_fast(xf_nat, ctx, pe; base=base1, threaded=true, h_mode=:cached, bandwidth_cache=dwarm, multi_method=:top3)
times = Float64[@elapsed composite_gradient_at_fast(xf_nat, ctx, pe; base=base1, threaded=true, h_mode=:cached, bandwidth_cache=dwarm, multi_method=:top3) for _ in 1:4]
g_cw, meta_cw = composite_gradient_at_fast(xf_nat, ctx, pe; base=base1, threaded=true, h_mode=:cached, bandwidth_cache=dwarm, multi_method=:top3)
meta_by_mode[:cached_warm] = (g=g_cw, meta=meta_cw)
logprint(@sprintf("  median=%.3fs  reps=%s  (all cache_hits=%s)", median(times), round.(times,digits=2), all(meta_cw.cache_hits[2:end])))
push!(p1_rows, (h_mode="cached_warm", median_s=median(times), mean_s=mean(times), min_s=minimum(times), max_s=maximum(times)))

write_csv_rows(joinpath(OUTDIR, "part1_wallclock.csv"), p1_rows)
logprint("\n-- summary: speedup vs :adaptive baseline --")
base_med = p1_rows[1].median_s
for r in p1_rows
    logprint(@sprintf("  %-14s median=%.3fs   speedup=%.2fx", r.h_mode, r.median_s, base_med/r.median_s))
end

# ============================================================================
# PART 2: winner-switch counts / weighted switching mass, adaptive vs quantile
# ============================================================================
logprint("\n", "="^90); logprint("PART 2: winner-switch counts / weighted switching mass"); logprint("="^90)
function switch_stats(meta)
    sm = filter(!isnan, meta.switch_mass[2:end])
    return (n_coords_with_mass=length(sm), mean_mass=isempty(sm) ? NaN : mean(sm),
            median_mass=isempty(sm) ? NaN : median(sm), min_mass=isempty(sm) ? NaN : minimum(sm),
            max_mass=isempty(sm) ? NaN : maximum(sm))
end
p2_rows = NamedTuple[]
for (label, mk) in (("adaptive", :adaptive), ("quantile", :quantile))
    s = switch_stats(meta_by_mode[mk].meta)
    logprint(@sprintf("  %-10s: n_coords_with_mass=%d  mean_mass=%.5f  median_mass=%.5f  min=%.5f  max=%.5f",
              label, s.n_coords_with_mass, s.mean_mass, s.median_mass, s.min_mass, s.max_mass))
    push!(p2_rows, merge((h_mode=label,), s))
end
n_quantile_fallback = count(bm -> bm isa NamedTuple && get(bm, :method, :x) == :bisection_fallback_same_dest,
                            meta_by_mode[:quantile].meta.bandwidth_meta[2:end])
logprint("  h_mode=:quantile fallback-to-bisection coordinates (same-destination collision): ", n_quantile_fallback, " / ", D2-1)
write_csv_rows(joinpath(OUTDIR, "part2_switch_mass.csv"), p2_rows)

# ============================================================================
# PART 3: per-coordinate gradient agreement, adaptive (bisection) vs quantile
# ============================================================================
logprint("\n", "="^90); logprint("PART 3: per-coordinate A-block gradient agreement, adaptive vs quantile"); logprint("="^90)
ga = g_ad[2:end]; gq = g_q[2:end]
cos_sim = dot(ga, gq) / (norm(ga) * norm(gq))
sign_agree = count((ga .> 0) .== (gq .> 0)) / length(ga)
relerr = abs.(ga .- gq) ./ max.(abs.(ga), abs.(gq), 1e-12)
logprint(@sprintf("  cosine(g_adaptive, g_quantile) [A-block, %d coords] = %.6f", length(ga), cos_sim))
logprint(@sprintf("  sign agreement = %.3f   median relerr = %.3f   mean relerr = %.3f", sign_agree, median(relerr), mean(relerr)))
logprint("  (D=4/W=8000 dev test found cosine~0.83, sign-agreement noisy on a few coords -- checking whether",
          " the 10x-larger-W here (more draws per target mass fraction) tightens this, per the FD-noise hypothesis.)")
write_csv_rows(joinpath(OUTDIR, "part3_percoord_agreement.csv"),
    [(cosine_similarity=cos_sim, sign_agreement=sign_agree, median_relerr=median(relerr), mean_relerr=mean(relerr))])

# ============================================================================
# PART 4: BandwidthCachePolicy staleness-detection demo across a simulated
# short outer-loop trajectory (small steps, matching a KNITRO line-search
# scale), showing hit-rate and wall-time saved, AND that a deliberately large
# jump correctly triggers invalidation (not unconditional reuse forever).
# ============================================================================
logprint("\n", "="^90); logprint("PART 4: BandwidthCachePolicy staleness-detection demo (10 simulated outer points)"); logprint("="^90)
Random.seed!(4242)
zfree_nat = pivot_reduce(log.(reshape(xf_nat[2:end], D, D)), pe)
w_nat = vcat(xf_nat[1], zfree_nat)
policy = BandwidthCachePolicy(max_iters_since_anchor = 5, max_move = 0.05)
p4_rows = NamedTuple[]
w_cur = copy(w_nat)
for it in 1:10
    step = it == 7 ? 0.5 : 0.01   # a deliberate large jump at iterate 7 to test the move-threshold trigger
    dir = randn(length(w_cur) - 1); dir ./= norm(dir)
    global w_cur = copy(w_cur); w_cur[2:end] .+= step .* dir
    xf_it = x_free_from_w(w_cur, pe)
    base_it = solve_base_state(xf_it, ctx)
    z_it = log.(reshape(xf_it[2:end], D, D)); w0_it = vcat(xf_it[1], pivot_reduce(z_it, pe))
    invalidated, reason = maybe_invalidate!(policy, w0_it)
    t_grad = @elapsed g_it, meta_it = composite_gradient_at_fast(xf_it, ctx, pe; base=base_it, threaded=true,
                                                                    h_mode=:cached, bandwidth_cache=policy.cache, multi_method=:top3)
    record_hits!(policy, meta_it.cache_hits[2:end])
    n_hits = count(meta_it.cache_hits[2:end])
    logprint(@sprintf("  iter %2d: step=%.2f invalidated=%-5s reason=%-14s cache_hits=%3d/%d  wall=%.3fs",
              it, step, invalidated, reason, n_hits, D2-1, t_grad))
    push!(p4_rows, (iter=it, step=step, invalidated=invalidated, reason=string(reason), cache_hits=n_hits, wall_s=t_grad))
end
logprint("  policy totals: n_invalidations=", policy.n_invalidations, "  n_hits=", policy.n_hits, "  n_misses=", policy.n_misses)
write_csv_rows(joinpath(OUTDIR, "part4_staleness_demo.csv"), p4_rows)

# ============================================================================
# PART 5: directional secant validation (matching W80k doc sec 3F exactly --
# 5 random unit directions in pivot-reduced z-space, h=0.02, 2 full
# warm-started evaluate_fullA re-solves per direction), comparing the REAL
# re-solved secant against each h_mode's PREDICTED secant (gradient . direction)
# at the base point.
# ============================================================================
logprint("\n", "="^90); logprint("PART 5: directional secant validation (5 random directions, h=0.02)"); logprint("="^90)
Random.seed!(31415)
h_dir = 0.02
p5_rows = NamedTuple[]
for i in 1:5
    dir = randn(length(w_nat) - 1); dir ./= norm(dir)
    wp = copy(w_nat); wp[2:end] .+= h_dir .* dir
    wm = copy(w_nat); wm[2:end] .-= h_dir .* dir
    xfp = x_free_from_w(wp, pe); xfm = x_free_from_w(wm, pe)
    t0 = time(); rp = evaluate_fullA(xfp, ctx; warm=true); t_plus = time()-t0
    t0 = time(); rm = evaluate_fullA(xfm, ctx; warm=true); t_minus = time()-t0
    ok = rp.inner_status in FEASIBLE_CODES && rm.inner_status in FEASIBLE_CODES
    secant = ok ? (rp.Delta_dual - rm.Delta_dual) / (2h_dir) : NaN
    preds = Dict{Symbol,Float64}()
    for (label, gv) in ((:adaptive, g_ad), (:quantile, g_q), (:fixed, g_fx), (:cached_warm, g_cw))
        preds[label] = dot(@view(gv[2:end]), dir)
    end
    logprint(@sprintf("  dir %d: status+=%d status-=%d  secant=%s", i, rp.inner_status, rm.inner_status, string(secant)))
    for (label, pv) in preds
        logprint(@sprintf("      predicted[%-11s] = %-14s  abs_err=%s", label, string(pv), ok ? string(abs(pv-secant)) : "n/a"))
    end
    push!(p5_rows, (dir=i, status_plus=rp.inner_status, status_minus=rm.inner_status, secant=secant,
                    pred_adaptive=preds[:adaptive], pred_quantile=preds[:quantile], pred_fixed=preds[:fixed],
                    pred_cached_warm=preds[:cached_warm], finite_sane=ok && isfinite(secant),
                    wall_plus_s=t_plus, wall_minus_s=t_minus))
end
write_csv_rows(joinpath(OUTDIR, "part5_directional_secants.csv"), p5_rows)
n_sane = count(r -> r.finite_sane, p5_rows)
logprint("\n  ", n_sane, "/5 directional secants finite/sane.")
if n_sane > 0
    sane_rows = filter(r -> r.finite_sane, p5_rows)
    for (label, key) in (("adaptive",:pred_adaptive), ("quantile",:pred_quantile), ("fixed",:pred_fixed), ("cached_warm",:pred_cached_warm))
        errs = [abs(getfield(r, key) - r.secant) for r in sane_rows]
        logprint(@sprintf("  %-12s mean_abs_err=%.6f  median_abs_err=%.6f", label, mean(errs), median(errs)))
    end
end

logprint("\nVmHWM at end of run = ", round(vmhwm_kb()/1e6, digits=2), " GB")
logprint("c9_bandwidth_optimization_benchmark.jl COMPLETE at ", now())
close(LOGIO)
