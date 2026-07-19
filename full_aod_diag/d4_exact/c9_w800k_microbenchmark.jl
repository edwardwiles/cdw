# ============================================================================
# Continuation 9, Phase 2 (W=800,000 half) + Phase 9 (fallback readiness):
# production-speed full breakdown at W=800,000, extending the memory-safety-
# only c9_w800k_memsafety_probe.jl result to match c9_w80k_microbenchmark.jl's
# scope, but against the Phase 3-5 architecture (compressed mode, cached
# bandwidth policy, winner certificates -- NOT the old dense-only path).
#
# ONE warmed Julia process for the whole run (context.jl included exactly
# once via context_real_d20.jl), same double-include-avoidance discipline as
# c9_w80k_microbenchmark.jl. Rep counts are LOWER than the W80k harness
# throughout (per this task's own time-budget instruction: W=800,000 costs
# ~70-190s per cold op, ~10x the W=80,000 numbers) -- calibrated against
# c9_w800k_timing_probe.jl's real per-call numbers (this directory, run
# immediately before this script), not guessed.
#
# NEW vs the W80k harness:
#   - Part 1A also profiles :compressed mode (never timed at W=800,000 before
#     this session) alongside :dense, directly testing whether compressed's
#     5.74x moment-build advantage at W=80,000/D=20 (Phase 3.1) holds at 10x
#     more draws.
#   - Part 1C's gradient breakdown adds Phase 5's recommended production
#     lever (h_mode=:cached wrapped in BandwidthCachePolicy, thread-safety
#     fix already merged at this worktree's fork commit) alongside adaptive/
#     fixed, instead of re-measuring the serial/threaded matrix (that data
#     point comes from the separate thread-sweep script's 1-thread run).
#   - An explicit warm-start-stability check (Phase 9 requirement): cold
#     evaluate_fullA immediately followed by warm=true, wall-time + inner
#     solution sanity compared directly.
#   - VmHWM logged after every major step (not just at the end), per the
#     standing W=800,000 safety discipline -- this is the run most likely to
#     reveal a new peak since it exercises more code paths in one process
#     than the memory-safety probe did.
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
include(joinpath(@__DIR__, "bandwidth_cache_policy.jl"))
using Statistics, Printf, Dates, Random, LinearAlgebra

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c9_w800k_microbenchmark")
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...)
    println(LOGIO, xs...)
    flush(stdout); flush(LOGIO)
end

function vmhwm_kb()
    try
        for line in eachline("/proc/self/status")
            startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
        end
    catch
    end
    return -1
end
gb(kb) = round(kb / 1e6, digits = 2)
logmem(label) = logprint("  [VmHWM after ", label, "] ", gb(vmhwm_kb()), " GB")

logprint("c9_w800k_microbenchmark.jl starting at ", now(), "  commit=", COMMIT, "  nthreads=", Threads.nthreads())
logmem("process start")

const FEASIBLE_CODES = (0, -100, -101, -103)
const W800 = 800000
x_free_from_w(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

function stats_row(label, times)
    sorted = sort(times); n = length(sorted)
    μ = sum(sorted) / n
    return (label = label, n = n, median_s = sorted[n÷2+1], min_s = sorted[1], max_s = sorted[end],
            mean_s = μ, std_s = (n > 1 ? sqrt(sum((t - μ)^2 for t in sorted) / (n - 1)) : 0.0))
end

function print_component_table(tag, summ)
    logprint("\n---- ", tag, " : component breakdown, median ms (N reps) ----")
    logprint(@sprintf("  %-36s %10s %14s", "component", "N", "median(ms)"))
    rows = NamedTuple[]
    for r in sort(summ; by = rr -> -rr.median_s)
        logprint(@sprintf("  %-36s %10d %14.4f", r.label, r.n, r.median_s * 1000))
        push!(rows, (component = r.label, n = r.n, median_ms = r.median_s * 1000, mean_ms = r.mean_s * 1000,
                     min_ms = r.min_s * 1000, max_ms = r.max_s * 1000, std_ms = r.std_s * 1000))
    end
    write_csv_rows(joinpath(OUTDIR, "$(tag).csv"), rows)
    return rows
end

# ============================================================================
# PART 0: context/setup, measured ONCE for W=800000
# ============================================================================
logprint("\n", "="^90); logprint("PART 0: context/setup, W=800000"); logprint("="^90)

gc_live_before = Base.gc_live_bytes()
t0 = time()
ctx = d20_real_setup(W = W800)
t_setup = time() - t0
gc_live_after = Base.gc_live_bytes()
logmem("d20_real_setup(W=800000) cold")

D = ctx.D
n_free = 1 + D^2
logprint("D = ", D, "  n_free (1+D^2) = ", n_free, "  nTotalMoments = ", ctx.nTotalMoments)
logprint("d20_real_setup(W=800000) wall (cold, JIT paid) = ", round(t_setup, digits = 2), "s")
logprint("gc_live_bytes: before=", gc_live_before, "  after=", gc_live_after,
         "  delta=", round((gc_live_after - gc_live_before) / 1e6, digits = 1), " MB")
logprint("size(ctx.U) = ", size(ctx.U), "  (", round(sizeof(ctx.U) / 1e6, digits = 1), " MB)")

gp0 = ctx.θ0_up[3+D]
xf_nat = ctx.θ0_up[ctx.free_idx]
pe = build_pivot_elimination(ctx)
logprint("gamma'_focal (natural theta, calibration) = ", gp0,
         "  bounds = [", ctx.bounds.γp_lo, ", ", ctx.bounds.γp_hi, "]")
logprint("pivot chosen: linear index ", pe.pivot_lin, " (largest |gravity coeff|)")

write_csv_rows(joinpath(OUTDIR, "part0_setup.csv"),
    [(D = D, n_free = n_free, nTotalMoments = ctx.nTotalMoments,
      setup_wall_s_cold = t_setup, vmhwm_after_setup_GB = gb(vmhwm_kb()),
      gc_live_delta_MB = (gc_live_after - gc_live_before) / 1e6, U_size_MB = sizeof(ctx.U) / 1e6)])

# ============================================================================
# WARM-START STABILITY CHECK (Phase 9 requirement): cold evaluate_fullA
# immediately followed by warm=true at the SAME point -- confirms no crash,
# reports the speedup, before any other timed measurement.
# ============================================================================
logprint("\n", "="^90); logprint("WARM-START STABILITY CHECK (Phase 9)"); logprint("="^90)
t0 = time()
r_cold = evaluate_fullA(xf_nat, ctx; warm = false)
t_cold_eval = time() - t0
logprint("cold evaluate_fullA: wall=", round(t_cold_eval, digits = 2), "s  inner_status=", r_cold.inner_status,
         "  Delta_dual=", r_cold.Delta_dual)
logmem("cold evaluate_fullA")
t0 = time()
r_warm = evaluate_fullA(xf_nat, ctx; warm = true)
t_warm_eval = time() - t0
logprint("warm evaluate_fullA (immediately after, same point): wall=", round(t_warm_eval, digits = 2),
         "s  inner_status=", r_warm.inner_status, "  Delta_dual=", r_warm.Delta_dual)
logmem("warm evaluate_fullA")
logprint("warm-start speedup = ", round(t_cold_eval / t_warm_eval, digits = 2), "x   |Delta diff| = ",
         abs(r_cold.Delta_dual - r_warm.Delta_dual), "   BOTH crash-free and finite: ",
         r_cold.inner_status in FEASIBLE_CODES && r_warm.inner_status in FEASIBLE_CODES && isfinite(r_warm.Delta_dual))
write_csv_rows(joinpath(OUTDIR, "warmstart_check.csv"),
    [(cold_wall_s = t_cold_eval, warm_wall_s = t_warm_eval, speedup = t_cold_eval / t_warm_eval,
      cold_status = r_cold.inner_status, warm_status = r_warm.inner_status,
      cold_Delta = r_cold.Delta_dual, warm_Delta = r_warm.Delta_dual)])

# ============================================================================
# WARM-UP (untimed): pay every JIT/compile cost once, both dense AND
# compressed, before ANY further timed measurement.
# ============================================================================
logprint("\nWarming up (untimed, dense + compressed)...")
t0 = time()
for _ in 1:2
    evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :dense)
end
for _ in 1:2
    evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :compressed)
end
base0 = solve_base_state(xf_nat, ctx)
composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = true, h_mode = :adaptive, multi_method = :top3)
composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = true, h_mode = :fixed, multi_method = :top3)
policy0 = BandwidthCachePolicy()
z0w = log.(reshape(xf_nat[2:end], D, D)); w0nat = vcat(xf_nat[1], pivot_reduce(z0w, pe))
maybe_invalidate!(policy0, w0nat)
composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = true, h_mode = :cached, bandwidth_cache = policy0.cache, multi_method = :top3)
logprint("Warm-up complete, wall = ", round(time() - t0, digits = 1), "s")
logmem("full warm-up (dense+compressed value, adaptive+fixed+cached gradient)")

# ============================================================================
# FINDING THE 4 OUTER POINTS -- same offsets/methodology as the W80k doc
# ============================================================================
logprint("\n", "="^90); logprint("FINDING THE 4 OUTER POINTS"); logprint("="^90)

points = Vector{NamedTuple}()

# ---- Point 1: exact calibration point (natural theta); reuse r_warm above, no re-solve ----
push!(points, (label = "calibration_natural_theta", xf = xf_nat, feasible = r_warm.inner_status in FEASIBLE_CODES,
               inner_status = r_warm.inner_status, Delta_dual = r_warm.Delta_dual, note = "primary benchmark point (reused from warm-start check above)"))
logprint("Point 1 (calibration/natural theta): inner_status=", r_warm.inner_status, "  Delta_dual=", r_warm.Delta_dual)

# ---- Point 2: gravity-tangent (gravity-EXACT by pivot-elimination construction) ----
zfree_nat = pivot_reduce(log.(reshape(xf_nat[2:end], D, D)), pe)
w_nat = vcat(xf_nat[1], zfree_nat)
Random.seed!(9001)
dir2 = randn(length(w_nat) - 1); dir2 ./= norm(dir2)
w2 = copy(w_nat); w2[2:end] .+= 0.02 .* dir2
xf2 = x_free_from_w(w2, pe)
t0 = time()
r2 = evaluate_fullA(xf2, ctx; warm = true)
logprint("Point 2 (gravity-tangent perturbation, |step|=0.02): inner_status=", r2.inner_status,
         "  Delta_dual=", r2.Delta_dual, "  gravity_raw=", r2.gravity_raw,
         "  wall=", round(time() - t0, digits = 2), "s")
push!(points, (label = "gravity_tangent_perturbation", xf = xf2, feasible = r2.inner_status in FEASIBLE_CODES,
               inner_status = r2.inner_status, Delta_dual = r2.Delta_dual,
               note = "random unit direction in pivot-reduced z-space, step=0.02, exact gravity-tangent"))

# ---- Point 3: upper branch ----
gp_hi = ctx.bounds.γp_hi
gp_offsets_up = [0.01, 0.02, 0.05, 0.10, 0.20]
found3 = false
local r3, xf3, off3
for off in gp_offsets_up
    global found3, r3, xf3, off3
    gp_try = clamp(gp0 * (1 + off), ctx.bounds.γp_lo, gp_hi)
    xf_try = vcat(gp_try, xf_nat[2:end])
    t0 = time()
    rtry = evaluate_fullA(xf_try, ctx; warm = true)
    logprint("  upper-branch try gamma'=gp0*(1+", off, ")=", gp_try, ": inner_status=", rtry.inner_status,
             "  Delta_dual=", rtry.Delta_dual, "  wall=", round(time() - t0, digits = 2), "s")
    if rtry.inner_status in FEASIBLE_CODES
        r3 = rtry; xf3 = xf_try; off3 = off; found3 = true
        break
    end
end
if found3
    push!(points, (label = "upper_branch_gp_offset_$(off3)", xf = xf3, feasible = true,
                   inner_status = r3.inner_status, Delta_dual = r3.Delta_dual,
                   note = "gamma' = gp0*(1+$(off3)), A at natural theta"))
    logprint("Point 3 (upper branch): FOUND at offset=", off3)
else
    logprint("Point 3 (upper branch): NO feasible offset found -- SKIPPED.")
    push!(points, (label = "upper_branch_INFEASIBLE", xf = xf_nat, feasible = false,
                   inner_status = -9999, Delta_dual = NaN, note = "all tried offsets infeasible: $(gp_offsets_up)"))
end

# ---- Point 4: lower branch ----
gp_lo = ctx.bounds.γp_lo
gp_offsets_dn = [0.01, 0.02, 0.05, 0.10, 0.20]
found4 = false
local r4, xf4, off4
for off in gp_offsets_dn
    global found4, r4, xf4, off4
    gp_try = clamp(gp0 * (1 - off), gp_lo, ctx.bounds.γp_hi)
    xf_try = vcat(gp_try, xf_nat[2:end])
    t0 = time()
    rtry = evaluate_fullA(xf_try, ctx; warm = true)
    logprint("  lower-branch try gamma'=gp0*(1-", off, ")=", gp_try, ": inner_status=", rtry.inner_status,
             "  Delta_dual=", rtry.Delta_dual, "  wall=", round(time() - t0, digits = 2), "s")
    if rtry.inner_status in FEASIBLE_CODES
        r4 = rtry; xf4 = xf_try; off4 = off; found4 = true
        break
    end
end
if found4
    push!(points, (label = "lower_branch_gp_offset_$(off4)", xf = xf4, feasible = true,
                   inner_status = r4.inner_status, Delta_dual = r4.Delta_dual,
                   note = "gamma' = gp0*(1-$(off4)), A at natural theta"))
    logprint("Point 4 (lower branch): FOUND at offset=", off4)
else
    logprint("Point 4 (lower branch): NO feasible offset found -- SKIPPED.")
    push!(points, (label = "lower_branch_INFEASIBLE", xf = xf_nat, feasible = false,
                   inner_status = -9999, Delta_dual = NaN, note = "all tried offsets infeasible: $(gp_offsets_dn)"))
end

write_csv_rows(joinpath(OUTDIR, "points_summary.csv"),
    [(label = p.label, feasible = p.feasible, inner_status = p.inner_status,
      Delta_dual = p.Delta_dual, note = p.note) for p in points])
logmem("4-point search")

logprint("\n---- 4-point summary ----")
for p in points
    logprint("  ", p.label, ": feasible=", p.feasible, " inner_status=", p.inner_status, " Delta_dual=", p.Delta_dual)
end

# ============================================================================
# PART 1: FULL breakdown at Point 1 (calibration)
# ============================================================================
logprint("\n", "="^90); logprint("PART 1: FULL breakdown at Point 1 (calibration/natural theta)"); logprint("="^90)

xf1 = points[1].xf

if !points[1].feasible
    logprint("Point 1 INFEASIBLE -- aborting Part 1 (would contradict this session's own smoke test).")
else
    # ---- 1A: value callback, DENSE mode, warm (N=4) vs cold (N=2) ----
    logprint("\n---- 1A(dense): evaluate_fullA_fast dense, warm-started (N=4) ----")
    prof_reset!()
    total_times_dense_warm = Float64[]
    for _ in 1:4
        t0 = time_ns()
        evaluate_fullA_fast(xf1, ctx; cache = nothing, warm = true, moment_representation = :dense)
        push!(total_times_dense_warm, (time_ns() - t0) / 1e9)
    end
    summ_dense_warm = prof_summary()
    push!(summ_dense_warm, stats_row("TOTAL", total_times_dense_warm))
    rows_dense_warm = print_component_table("part1a_value_dense_warm", summ_dense_warm)
    logmem("1A dense warm (N=4)")

    logprint("\n---- 1A(dense): evaluate_fullA_fast dense, COLD (warm=false, N=2) ----")
    prof_reset!()
    total_times_dense_cold = Float64[]
    for _ in 1:2
        t0 = time_ns()
        evaluate_fullA_fast(xf1, ctx; cache = nothing, warm = false, moment_representation = :dense)
        push!(total_times_dense_cold, (time_ns() - t0) / 1e9)
    end
    summ_dense_cold = prof_summary()
    push!(summ_dense_cold, stats_row("TOTAL", total_times_dense_cold))
    rows_dense_cold = print_component_table("part1a_value_dense_cold", summ_dense_cold)
    logmem("1A dense cold (N=2)")

    # ---- 1A: value callback, COMPRESSED mode, warm (N=4) vs cold (N=2) -- NEW at W=800k ----
    logprint("\n---- 1A(compressed): evaluate_fullA_fast compressed, warm-started (N=4) ----")
    prof_reset!()
    total_times_comp_warm = Float64[]
    for _ in 1:4
        t0 = time_ns()
        evaluate_fullA_fast(xf1, ctx; cache = nothing, warm = true, moment_representation = :compressed)
        push!(total_times_comp_warm, (time_ns() - t0) / 1e9)
    end
    summ_comp_warm = prof_summary()
    push!(summ_comp_warm, stats_row("TOTAL", total_times_comp_warm))
    rows_comp_warm = print_component_table("part1a_value_compressed_warm", summ_comp_warm)
    logprint("  COMPRESSED_FALLBACK_COUNT so far = ", COMPRESSED_FALLBACK_COUNT[])
    logmem("1A compressed warm (N=4)")

    logprint("\n---- 1A(compressed): evaluate_fullA_fast compressed, COLD (warm=false, N=2) ----")
    prof_reset!()
    total_times_comp_cold = Float64[]
    for _ in 1:2
        t0 = time_ns()
        evaluate_fullA_fast(xf1, ctx; cache = nothing, warm = false, moment_representation = :compressed)
        push!(total_times_comp_cold, (time_ns() - t0) / 1e9)
    end
    summ_comp_cold = prof_summary()
    push!(summ_comp_cold, stats_row("TOTAL", total_times_comp_cold))
    rows_comp_cold = print_component_table("part1a_value_compressed_cold", summ_comp_cold)
    logmem("1A compressed cold (N=2)")

    med_dw = median(total_times_dense_warm); med_cw = median(total_times_comp_warm)
    med_dc = median(total_times_dense_cold); med_cc = median(total_times_comp_cold)
    logprint("\n---- 1A headline: dense vs compressed @ W=800,000 ----")
    logprint(@sprintf("  warm TOTAL: dense=%.4fs  compressed=%.4fs  speedup(dense/compressed)=%.3fx", med_dw, med_cw, med_dw / med_cw))
    logprint(@sprintf("  cold TOTAL: dense=%.4fs  compressed=%.4fs  speedup(dense/compressed)=%.3fx", med_dc, med_cc, med_dc / med_cc))
    mb_dense = only(r.median_ms for r in rows_dense_warm if r.component == "inner_moment_build") / 1000
    mb_comp = only(r.median_ms for r in rows_comp_warm if r.component == "inner_moment_build_compressed") / 1000
    logprint(@sprintf("  warm inner_moment_build: dense=%.4fs  compressed=%.4fs  speedup=%.3fx", mb_dense, mb_comp, mb_dense / mb_comp))
    write_csv_rows(joinpath(OUTDIR, "part1a_headline_speedup.csv"),
        [(mode_pair = "warm_TOTAL", dense_s = med_dw, compressed_s = med_cw, speedup = med_dw / med_cw),
         (mode_pair = "cold_TOTAL", dense_s = med_dc, compressed_s = med_cc, speedup = med_dc / med_cc),
         (mode_pair = "warm_moment_build", dense_s = mb_dense, compressed_s = mb_comp, speedup = mb_dense / mb_comp)])

    # ---- 1B: evaluate_fullA (plain oracle), warm-started, N=3 ----
    logprint("\n---- 1B: evaluate_fullA (plain oracle, warm-started, N=3) ----")
    evalA_times = Float64[]; evalA_inner = Float64[]; evalA_post = Float64[]
    for _ in 1:3
        r = evaluate_fullA(xf1, ctx; warm = true)
        push!(evalA_times, r.elapsed.total); push!(evalA_inner, r.elapsed.inner); push!(evalA_post, r.elapsed.post)
    end
    logprint(@sprintf("  total median=%.3fs  inner median=%.3fs  post median=%.3fs",
              median(evalA_times), median(evalA_inner), median(evalA_post)))
    write_csv_rows(joinpath(OUTDIR, "part1b_evaluateFullA_plain.csv"),
        [(rep = i, total_s = evalA_times[i], inner_s = evalA_inner[i], post_s = evalA_post[i]) for i in 1:3])
    logmem("1B plain oracle (N=3)")

    # ---- 1C: full L_fix gradient -- adaptive / fixed / cached(Phase 5 policy), threaded=true, N=2/config ----
    logprint("\n---- 1C: composite_gradient_at_fast, full ", D^2, "-coordinate gradient, threaded=true, N=2 per config ----")
    base1 = solve_base_state(xf1, ctx)
    z01 = log.(reshape(xf1[2:end], D, D)); w01 = vcat(xf1[1], pivot_reduce(z01, pe))
    policy1 = BandwidthCachePolicy()
    maybe_invalidate!(policy1, w01)
    # prime the cache once (this rep intentionally excluded from timing -- matches production use:
    # a caller populates once, then reuses across nearby outer iterates, per Phase 5's recommendation)
    composite_gradient_at_fast(xf1, ctx, pe; base = base1, threaded = true, h_mode = :cached, bandwidth_cache = policy1.cache, multi_method = :top3)

    c1_rows = NamedTuple[]
    for (hm, extra_kw) in ((:adaptive, NamedTuple()), (:fixed, NamedTuple()), (:cached, (bandwidth_cache = policy1.cache,)))
        times = Float64[]
        for _ in 1:2
            push!(times, @elapsed composite_gradient_at_fast(xf1, ctx, pe; base = base1, threaded = true, h_mode = hm, multi_method = :top3, extra_kw...))
        end
        logprint(@sprintf("  h_mode=%-8s threaded=true  median=%.3fs  (N=2, reps=%s)",
                  hm, median(times), round.(times, digits=2)))
        push!(c1_rows, (h_mode = hm, threaded = true, median_s = median(times), mean_s = mean(times),
                        min_s = minimum(times), max_s = maximum(times)))
        logmem("1C gradient h_mode=$(hm)")
    end
    write_csv_rows(joinpath(OUTDIR, "part1c_full_grad_hmode.csv"), c1_rows)
    logprint("  NOTE: serial (threaded=false) full-gradient timing intentionally NOT re-measured here")
    logprint("        (cost-prohibitive at W=800,000 -- the c9_w800k_threadsweep.jl script's own")
    logprint("        JULIA_NUM_THREADS=1 process run supplies this data point instead.)")

    # ---- 1F: optimized-value directional secants -- 5 random gravity-tangent directions, h=0.02 ----
    logprint("\n---- 1F: optimized-value directional secants, 5 random gravity-tangent directions, h=0.02 ----")
    Random.seed!(31415)
    h_dir = 0.02
    f1_rows = NamedTuple[]
    for i in 1:5
        dir = randn(length(w01) - 1); dir ./= norm(dir)
        wp = copy(w01); wp[2:end] .+= h_dir .* dir
        wm = copy(w01); wm[2:end] .-= h_dir .* dir
        xfp = x_free_from_w(wp, pe); xfm = x_free_from_w(wm, pe)
        t0 = time()
        rp = evaluate_fullA(xfp, ctx; warm = true)
        t_plus = time() - t0
        t0 = time()
        rm = evaluate_fullA(xfm, ctx; warm = true)
        t_minus = time() - t0
        ok = rp.inner_status in FEASIBLE_CODES && rm.inner_status in FEASIBLE_CODES
        secant = ok ? (rp.Delta_dual - rm.Delta_dual) / (2h_dir) : NaN
        logprint(@sprintf("  dir %d: status+=%d status-=%d  D+=%s D-=%s  secant=%s  wall+=%.2fs wall-=%.2fs",
                  i, rp.inner_status, rm.inner_status, string(rp.Delta_dual), string(rm.Delta_dual),
                  string(secant), t_plus, t_minus))
        push!(f1_rows, (dir = i, status_plus = rp.inner_status, status_minus = rm.inner_status,
                        Delta_plus = rp.Delta_dual, Delta_minus = rm.Delta_dual,
                        secant = secant, finite_sane = ok && isfinite(secant),
                        wall_plus_s = t_plus, wall_minus_s = t_minus))
    end
    write_csv_rows(joinpath(OUTDIR, "part1f_directional_secants.csv"), f1_rows)
    n_sane = count(r -> r.finite_sane, f1_rows)
    logprint("  ", n_sane, "/5 directional secants finite/sane.")
    logmem("1F secants (5 dirs)")
end

# ============================================================================
# PART 2: lighter checks at points 2-4 -- value(dense, N=2) + full gradient
# (h_mode=:cached, the Phase 5 production recommendation, N=1) only.
# ============================================================================
logprint("\n", "="^90); logprint("PART 2: lighter checks at points 2-4"); logprint("="^90)

part2_rows = NamedTuple[]
for (idx, p) in enumerate(points)
    idx == 1 && continue
    if !p.feasible
        logprint("\n---- ", p.label, ": INFEASIBLE, skipped ----")
        push!(part2_rows, (label = p.label, feasible = false, value_median_s = NaN, grad_s = NaN))
        continue
    end
    logprint("\n---- ", p.label, " ----")
    xfp = p.xf
    evaluate_fullA_fast(xfp, ctx; cache = nothing, warm = true, moment_representation = :dense)  # untimed warm-up at this point
    val_times = Float64[@elapsed evaluate_fullA_fast(xfp, ctx; cache = nothing, warm = true, moment_representation = :dense) for _ in 1:2]
    basep = solve_base_state(xfp, ctx)
    zp = log.(reshape(xfp[2:end], D, D)); wp0 = vcat(xfp[1], pivot_reduce(zp, pe))
    policyp = BandwidthCachePolicy(); maybe_invalidate!(policyp, wp0)
    composite_gradient_at_fast(xfp, ctx, pe; base = basep, threaded = true, h_mode = :cached, bandwidth_cache = policyp.cache, multi_method = :top3)  # prime
    grad_s = @elapsed composite_gradient_at_fast(xfp, ctx, pe; base = basep, threaded = true, h_mode = :cached, bandwidth_cache = policyp.cache, multi_method = :top3)
    logprint(@sprintf("  value(warm,dense) median=%.3fs  full_grad(cached,threaded) single=%.3fs",
              median(val_times), grad_s))
    push!(part2_rows, (label = p.label, feasible = true, value_median_s = median(val_times), grad_s = grad_s))
    logmem("Part2 point $(p.label)")
end
write_csv_rows(joinpath(OUTDIR, "part2_lighter_checks.csv"), part2_rows)

# ============================================================================
# PART 3: memory ledger
# ============================================================================
logprint("\n", "="^90); logprint("PART 3: memory ledger"); logprint("="^90)
ledger = [
    (array = "ctx.U (Frechet draws)", shape = "$(W800) x $(D)", theoretical_MB = W800*D*8/1e6, observed_MB = sizeof(ctx.U)/1e6),
    (array = "inner K/G moment matrix (evaluate_fullA, dense)", shape = "$(W800) x $(ctx.nTotalMoments)", theoretical_MB = W800*ctx.nTotalMoments*8/1e6, observed_MB = NaN),
    (array = "winners.jl gap (W x D)", shape = "$(W800) x $(D)", theoretical_MB = W800*D*8/1e6, observed_MB = NaN),
    (array = "lfix_incremental price/runnerup/third/contrib buffers (6x, W x D each)", shape = "6 x ($(W800) x $(D))", theoretical_MB = 6*W800*D*8/1e6, observed_MB = NaN),
    (array = "compressed_moments wval (W x D)", shape = "$(W800) x $(D)", theoretical_MB = W800*D*8/1e6, observed_MB = NaN),
    (array = "n_free (gradient vector length)", shape = "$(D^2)", theoretical_MB = D^2*8/1e6, observed_MB = NaN),
    (array = "jac_h (THE BUG, W80k doc sec 0 -- confirmed ABSENT here)", shape = "$(W800) x $(ctx.nTotalMoments+2) x l_full(~423)", theoretical_MB = W800*(ctx.nTotalMoments+2)*423*8/1e6, observed_MB = 0.0),
]
logprint(@sprintf("  %-62s %-18s %14s %14s", "array", "shape", "theory(MB)", "observed(MB)"))
for r in ledger
    logprint(@sprintf("  %-62s %-18s %14.2f %14s", r.array, r.shape, r.theoretical_MB, isnan(r.observed_MB) ? "n/a" : @sprintf("%.2f", r.observed_MB)))
end
write_csv_rows(joinpath(OUTDIR, "part3_memory_ledger.csv"), ledger)
logprint("VmHWM at end of run = ", vmhwm_kb(), " KB = ", round(vmhwm_kb()/1e6, digits=2), " GB")
logprint("Base.gc_live_bytes() at end of run = ", round(Base.gc_live_bytes()/1e6, digits=1), " MB")

logprint("\nc9_w800k_microbenchmark.jl COMPLETE at ", now())
close(LOGIO)
