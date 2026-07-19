# ============================================================================
# Continuation 9, Phase 2 (W=80,000 half): production bottleneck + memory
# microbenchmark for the exact full-A_od machinery on the REAL D=20 economy.
#
# Reuses c8_perfprofile_harness.jl's @prof-timer / warmed-process discipline
# DIRECTLY (same instrumentation.jl, same profile_value_mode-style pattern,
# same component/scope names) -- not reinvented. The only new pieces here are
# (a) building the context via `d20_real_setup` (context_real_d20.jl, ported
# this session) instead of the synthetic `d4_exact_setup`/`d_exact_setup_scaled`,
# and (b) a lighter per-point structure matching this task's explicit 4-point
# / time-budget scope (full breakdown at the primary calibration point only;
# lighter checks at the other 3, exactly mirroring c8_perfprofile_harness.jl's
# own Part A (full) vs Part D (lighter grid) split).
#
# ONE warmed Julia process for the whole 4-point grid at W=80000 -- context.jl
# is included exactly once (via context_real_d20.jl's own single include),
# same double-include hazard avoidance as c8_perfprofile_harness.jl (see that
# file's own header for the full explanation, not re-derived here).
#
# Timing probe done first (c9_w80k_timing_probe.jl, this directory) to scope
# rep counts sensibly: at D=20/W=80000, a single evaluate_fullA cold solve is
# ~7-14s (not ~ms like D=4), a single evaluate_fullA_fast warm-ish call is
# ~5-10s, and one FULL 400-dim composite_gradient_at_fast call (threaded,
# adaptive-h) is ~10s. Rep counts below are chosen accordingly (much smaller
# N than the D=4 harness, which could afford N=50).
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
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c9_w80k_microbenchmark")
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...)
    println(LOGIO, xs...)
    flush(stdout); flush(LOGIO)
end
logprint("c9_w80k_microbenchmark.jl starting at ", now(), "  commit=", COMMIT, "  nthreads=", Threads.nthreads())

function vmhwm_kb()
    try
        for line in eachline("/proc/self/status")
            if startswith(line, "VmHWM:")
                return parse(Int, split(line)[2])
            end
        end
    catch
    end
    return -1
end

const FEASIBLE_CODES = (0, -100, -101, -103)
x_free_from_w(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

function stats_row(label, times)
    sorted = sort(times); n = length(sorted)
    μ = sum(sorted) / n
    return (label = label, n = n, median_s = sorted[n÷2+1], min_s = sorted[1], max_s = sorted[end],
            mean_s = μ, std_s = (n > 1 ? sqrt(sum((t - μ)^2 for t in sorted) / (n - 1)) : 0.0))
end

"Paired dense component table (single-mode here -- D=20 dense is the only mode profiled at this scale, per the task's 'evaluate_fullA, dense mode -- trusted reference' instruction)."
function print_component_table(tag, summ)
    logprint("\n---- ", tag, " : component breakdown, median ms (N reps) ----")
    logprint(@sprintf("  %-32s %10s %14s", "component", "N", "median(ms)"))
    rows = NamedTuple[]
    for r in sort(summ; by = rr -> -rr.median_s)
        logprint(@sprintf("  %-32s %10d %14.4f", r.label, r.n, r.median_s * 1000))
        push!(rows, (component = r.label, n = r.n, median_ms = r.median_s * 1000, mean_ms = r.mean_s * 1000,
                     min_ms = r.min_s * 1000, max_ms = r.max_s * 1000, std_ms = r.std_s * 1000))
    end
    write_csv_rows(joinpath(OUTDIR, "$(tag).csv"), rows)
    return rows
end

# ============================================================================
# PART 0: context/setup, measured ONCE for W=80000 (not per point)
# ============================================================================
logprint("\n", "="^90); logprint("PART 0: context/setup, W=80000"); logprint("="^90)

gc_live_before = Base.gc_live_bytes()
t0 = time()
ctx = d20_real_setup(W = 80000)
t_setup = time() - t0
gc_live_after = Base.gc_live_bytes()
vmhwm_after_setup = vmhwm_kb()

D = ctx.D
n_free = 1 + D^2
logprint("D = ", D, "  n_free (1+D^2) = ", n_free, "  nTotalMoments = ", ctx.nTotalMoments)
logprint("d20_real_setup(W=80000) wall = ", round(t_setup, digits = 2), "s")
logprint("gc_live_bytes: before=", gc_live_before, "  after=", gc_live_after,
         "  delta=", round((gc_live_after - gc_live_before) / 1e6, digits = 1), " MB")
logprint("VmHWM after setup = ", vmhwm_after_setup, " KB = ", round(vmhwm_after_setup / 1e6, digits = 2), " GB")
logprint("size(ctx.U) = ", size(ctx.U), "  (", round(sizeof(ctx.U) / 1e6, digits = 1), " MB)")

# repeat setup a 2nd time (fresh call) to separate one-time JIT/compile cost from steady-state cost
t0 = time()
ctx_warm_check = d20_real_setup(W = 80000)
t_setup2 = time() - t0
logprint("d20_real_setup(W=80000) SECOND call (JIT already paid) wall = ", round(t_setup2, digits = 2), "s")
ctx_warm_check = nothing; GC.gc()

write_csv_rows(joinpath(OUTDIR, "part0_setup.csv"),
    [(D = D, n_free = n_free, nTotalMoments = ctx.nTotalMoments,
      setup_wall_s_cold = t_setup, setup_wall_s_warm = t_setup2,
      gc_live_delta_MB = (gc_live_after - gc_live_before) / 1e6,
      vmhwm_after_setup_KB = vmhwm_after_setup, U_size_MB = sizeof(ctx.U) / 1e6)])

gp0 = ctx.θ0_up[3+D]
xf_nat = ctx.θ0_up[ctx.free_idx]
pe = build_pivot_elimination(ctx)
logprint("gamma'_focal (natural theta, calibration) = ", gp0,
         "  bounds = [", ctx.bounds.γp_lo, ", ", ctx.bounds.γp_hi, "]")
logprint("pivot chosen: linear index ", pe.pivot_lin, " (largest |gravity coeff|)")

# ============================================================================
# WARM-UP (untimed): pay every JIT/compile cost once, here, before ANY timed
# measurement below -- matches c8_perfprofile_harness.jl's own discipline.
# ============================================================================
logprint("\nWarming up (untimed)...")
t0 = time()
for _ in 1:2
    evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :dense)
end
base0 = solve_base_state(xf_nat, ctx)
composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = true, h_mode = :adaptive, multi_method = :top3)
composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = false, h_mode = :fixed, multi_method = :top3)
logprint("Warm-up complete, wall = ", round(time() - t0, digits = 1), "s")

# ============================================================================
# FINDING THE 4 OUTER POINTS
# ============================================================================
logprint("\n", "="^90); logprint("FINDING THE 4 OUTER POINTS"); logprint("="^90)

points = Vector{NamedTuple}()

# ---- Point 1: exact calibration point (natural theta) ----
t0 = time()
r1 = evaluate_fullA(xf_nat, ctx; warm = false)
logprint("Point 1 (calibration/natural theta): inner_status=", r1.inner_status,
         "  Delta_dual=", r1.Delta_dual, "  gamma'=", r1.gamma_focal_prime,
         "  wall=", round(time() - t0, digits = 2), "s")
push!(points, (label = "calibration_natural_theta", xf = xf_nat, feasible = r1.inner_status in FEASIBLE_CODES,
               inner_status = r1.inner_status, Delta_dual = r1.Delta_dual, note = "primary benchmark point"))

# ---- Point 2: small gravity-tangent perturbation of point 1's A ----
# The pivot-reduced z_free coordinates (w[2:end]) are gravity-EXACT by
# construction of pivot_expand (not just tangent-to-first-order): for ANY
# z_free, gravity_from_logz(pivot_expand(z_free,pe), ctx) == 0 identically
# (the pivot coordinate is solved exactly to enforce this). So a random
# perturbation of w[2:end] alone (w[1]=gamma' held fixed) is an exact
# gravity-tangent direction, not an approximation -- this is stronger than
# the task's "-ish" allowance, documented as such rather than claimed as new.
zfree_nat = pivot_reduce(log.(reshape(xf_nat[2:end], D, D)), pe)
w_nat = vcat(xf_nat[1], zfree_nat)
Random.seed!(9001)
dir2 = randn(length(w_nat) - 1); dir2 ./= norm(dir2)
w2 = copy(w_nat); w2[2:end] .+= 0.02 .* dir2
xf2 = x_free_from_w(w2, pe)
t0 = time()
r2 = evaluate_fullA(xf2, ctx; warm = true)
logprint("Point 2 (gravity-tangent perturbation, |step|=0.02 in reduced z-space): inner_status=", r2.inner_status,
         "  Delta_dual=", r2.Delta_dual, "  gravity_raw=", r2.gravity_raw,
         "  wall=", round(time() - t0, digits = 2), "s")
push!(points, (label = "gravity_tangent_perturbation", xf = xf2, feasible = r2.inner_status in FEASIBLE_CODES,
               inner_status = r2.inner_status, Delta_dual = r2.Delta_dual,
               note = "random unit direction in pivot-reduced z-space, step=0.02, exact gravity-tangent"))

# ---- Point 3: prospective upper-branch (gamma' above calibration, A at natural theta) ----
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
    logprint("Point 3 (upper branch): NO feasible offset found in ", gp_offsets_up, " -- SKIPPED, recorded explicitly.")
    push!(points, (label = "upper_branch_INFEASIBLE", xf = xf_nat, feasible = false,
                   inner_status = -9999, Delta_dual = NaN, note = "all tried offsets infeasible: $(gp_offsets_up)"))
end

# ---- Point 4: prospective lower-branch (gamma' below calibration, A at natural theta) ----
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
    logprint("Point 4 (lower branch): NO feasible offset found in ", gp_offsets_dn, " -- SKIPPED, recorded explicitly.")
    push!(points, (label = "lower_branch_INFEASIBLE", xf = xf_nat, feasible = false,
                   inner_status = -9999, Delta_dual = NaN, note = "all tried offsets infeasible: $(gp_offsets_dn)"))
end

write_csv_rows(joinpath(OUTDIR, "points_summary.csv"),
    [(label = p.label, feasible = p.feasible, inner_status = p.inner_status,
      Delta_dual = p.Delta_dual, note = p.note) for p in points])

logprint("\n---- 4-point summary ----")
for p in points
    logprint("  ", p.label, ": feasible=", p.feasible, " inner_status=", p.inner_status, " Delta_dual=", p.Delta_dual)
end

# ============================================================================
# PART 1: FULL breakdown at Point 1 (primary benchmark)
# ============================================================================
logprint("\n", "="^90); logprint("PART 1: FULL breakdown at Point 1 (calibration/natural theta)"); logprint("="^90)

xf1 = points[1].xf

if !points[1].feasible
    logprint("Point 1 INFEASIBLE -- cannot run full breakdown. This would be a serious problem (contradicts",
             " this session's own smoke test) -- aborting Part 1.")
else
    # ---- 1A: value callback, dense mode, warm-started vs cold, N reps per this scale's budget ----
    logprint("\n---- 1A: evaluate_fullA_fast dense, warm-started (N=6) ----")
    for _ in 1:2
        evaluate_fullA_fast(xf1, ctx; cache = nothing, warm = true, moment_representation = :dense)
    end
    prof_reset!()
    total_times_warm = Float64[]
    for _ in 1:6
        t0 = time_ns()
        r, meta = evaluate_fullA_fast(xf1, ctx; cache = nothing, warm = true, moment_representation = :dense)
        push!(total_times_warm, (time_ns() - t0) / 1e9)
    end
    summ_warm = prof_summary()
    push!(summ_warm, stats_row("TOTAL", total_times_warm))
    rows_warm = print_component_table("part1a_value_warm", summ_warm)

    logprint("\n---- 1A: evaluate_fullA_fast dense, COLD (warm=false, N=3) ----")
    prof_reset!()
    total_times_cold = Float64[]
    for _ in 1:3
        t0 = time_ns()
        r, meta = evaluate_fullA_fast(xf1, ctx; cache = nothing, warm = false, moment_representation = :dense)
        push!(total_times_cold, (time_ns() - t0) / 1e9)
    end
    summ_cold = prof_summary()
    push!(summ_cold, stats_row("TOTAL", total_times_cold))
    rows_cold = print_component_table("part1a_value_cold", summ_cold)

    # ---- 1B: evaluate_fullA itself (the coarse-grained oracle, "trusted reference" at the
    # top level -- total/inner/post only, no fine @prof breakdown exists in oracle.jl itself) ----
    logprint("\n---- 1B: evaluate_fullA (plain oracle, warm-started, N=5) ----")
    evalA_times = Float64[]; evalA_inner = Float64[]; evalA_post = Float64[]
    for _ in 1:5
        r = evaluate_fullA(xf1, ctx; warm = true)
        push!(evalA_times, r.elapsed.total); push!(evalA_inner, r.elapsed.inner); push!(evalA_post, r.elapsed.post)
    end
    logprint(@sprintf("  total median=%.3fs  inner median=%.3fs  post median=%.3fs",
              median(evalA_times), median(evalA_inner), median(evalA_post)))
    write_csv_rows(joinpath(OUTDIR, "part1b_evaluateFullA_plain.csv"),
        [(rep = i, total_s = evalA_times[i], inner_s = evalA_inner[i], post_s = evalA_post[i]) for i in 1:5])

    # ---- 1C: full L_fix gradient, top3+threaded+adaptive as the primary config, N=4 ----
    logprint("\n---- 1C: composite_gradient_at_fast, full ", D^2, "-coordinate gradient, N=4 per config ----")
    base1 = solve_base_state(xf1, ctx)
    c1_rows = NamedTuple[]
    for hm in (:adaptive, :fixed), th in (true, false)
        for _ in 1:1
            composite_gradient_at_fast(xf1, ctx, pe; base = base1, threaded = th, h_mode = hm, multi_method = :top3)
        end
        times = Float64[]
        for _ in 1:4
            push!(times, @elapsed composite_gradient_at_fast(xf1, ctx, pe; base = base1, threaded = th, h_mode = hm, multi_method = :top3))
        end
        logprint(@sprintf("  h_mode=%-8s threaded=%-5s  median=%.3fs  (N=4, reps=%s)",
                  hm, th, median(times), round.(times, digits=2)))
        push!(c1_rows, (h_mode = hm, threaded = th, median_s = median(times), mean_s = mean(times),
                        min_s = minimum(times), max_s = maximum(times)))
    end
    write_csv_rows(joinpath(OUTDIR, "part1c_full_grad_hmode_threaded.csv"), c1_rows)

    # ---- 1D: component decomposition of the gradient (cache_build, gamma, bandwidth, fd_probes) ----
    logprint("\n---- 1D: gradient component decomposition, h_mode=:adaptive, multi_method=:top3, N=3 ----")
    function decompose_gradient(x_free0, ctx, pe, base; h_mode::Symbol = :adaptive, multi_method::Symbol = :top3)
        Dloc = ctx.D; D2 = Dloc^2
        t_cache0 = time_ns()
        cache = build_lfix_base_cache(x_free0, ctx, base)
        t_cache = (time_ns() - t_cache0) / 1e9

        z0 = log.(reshape(x_free0[2:end], Dloc, Dloc))
        w0 = vcat(x_free0[1], pivot_reduce(z0, pe))

        t_gamma0 = time_ns()
        g1 = gamma_component_analytic(cache, base, w0[1])
        t_gamma = (time_ns() - t_gamma0) / 1e9

        t_bandwidth = 0.0; t_fd = 0.0
        for k in 2:D2
            h, m, _ = select_bandwidth(cache, ctx, pe, w0, k; multi_method = multi_method)
            t_bandwidth += 0.0   # accumulated below via explicit timing to avoid double time_ns() overhead noise
            t0 = time_ns()
            a_block_fd_component(cache, ctx, pe, w0, k, h; multi_method = multi_method)
            a_block_fd_component(cache, ctx, pe, w0, k, h / 2; multi_method = multi_method)
            t_fd += (time_ns() - t0) / 1e9
        end
        return (cache_build = t_cache, gamma_analytic = t_gamma, fd_and_bandwidth = t_fd)
    end
    t_base_solve = median([@elapsed solve_base_state(xf1, ctx) for _ in 1:3])
    decomp = [decompose_gradient(xf1, ctx, pe, base1) for _ in 1:3]
    med_field(v, f) = median([getfield(x, f) for x in v])
    logprint(@sprintf("  base_state solve: %.3fs", t_base_solve))
    for f in (:cache_build, :gamma_analytic, :fd_and_bandwidth)
        logprint(@sprintf("  %-22s median=%.3fs", f, med_field(decomp, f)))
    end
    total_recon = t_base_solve + sum(med_field(decomp, f) for f in (:cache_build, :gamma_analytic, :fd_and_bandwidth))
    logprint(@sprintf("  RECONSTRUCTED TOTAL: %.3fs", total_recon))
    write_csv_rows(joinpath(OUTDIR, "part1d_grad_decomposition.csv"),
        [(base_state_s = t_base_solve, cache_build_s = med_field(decomp, :cache_build),
          gamma_s = med_field(decomp, :gamma_analytic), fd_and_bandwidth_s = med_field(decomp, :fd_and_bandwidth),
          total_s = total_recon)])

    # ---- 1E: per-coordinate cost, a few individual coords (top3 vs generic) ----
    logprint("\n---- 1E: per-coordinate cost, coords {2, D^2÷2, D^2-1} (N=8 reps each) ----")
    function coord_cost(cache, ctx, pe, w0, k; multi_method::Symbol, N::Int = 8)
        for _ in 1:2
            h, m, _ = select_bandwidth(cache, ctx, pe, w0, k; multi_method = multi_method)
            a_block_fd_component(cache, ctx, pe, w0, k, h; multi_method = multi_method)
        end
        times = Float64[]
        for _ in 1:N
            t0 = time_ns()
            h, m, _ = select_bandwidth(cache, ctx, pe, w0, k; multi_method = multi_method)
            a_block_fd_component(cache, ctx, pe, w0, k, h; multi_method = multi_method)
            a_block_fd_component(cache, ctx, pe, w0, k, h / 2; multi_method = multi_method)
            push!(times, (time_ns() - t0) / 1e9)
        end
        return median(times)
    end
    cache1 = build_lfix_base_cache(xf1, ctx, base1)
    z01 = log.(reshape(xf1[2:end], D, D)); w01 = vcat(xf1[1], pivot_reduce(z01, pe))
    e1_rows = NamedTuple[]
    for k in (2, D^2 ÷ 2, D^2 - 1)
        m_top3 = coord_cost(cache1, ctx, pe, w01, k; multi_method = :top3)
        m_gen = coord_cost(cache1, ctx, pe, w01, k; multi_method = :generic)
        logprint(@sprintf("  coord k=%-4d  top3=%.4fms  generic=%.4fms  ratio(gen/top3)=%.3f",
                  k, m_top3*1000, m_gen*1000, m_gen/m_top3))
        push!(e1_rows, (k = k, top3_ms = m_top3*1000, generic_ms = m_gen*1000, ratio_gen_over_top3 = m_gen/m_top3))
    end
    write_csv_rows(joinpath(OUTDIR, "part1e_percoord.csv"), e1_rows)

    # ---- 1F: directional "optimized-value" checks (NOT the incremental machinery) --
    # 5 random gravity-tangent (pivot-reduced z-space) directions, 2 FULL warm-started
    # evaluate_fullA (== optimized_Delta) inner solves per direction (+h, -h), h=0.02.
    # Explicitly does NOT attempt a full D^2-coordinate FD gradient -- out of scope per
    # the task's own instruction.
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
end

# ============================================================================
# PART 2: LIGHTER checks at points 2-4 (feasibility already recorded above;
# here: TOTAL value-callback wall time + TOTAL full-gradient wall time only,
# N=2 reps each -- mirrors c8_perfprofile_harness.jl's Part D "lighter
# breakdown given the time budget" precedent).
# ============================================================================
logprint("\n", "="^90); logprint("PART 2: lighter checks at points 2-4"); logprint("="^90)

part2_rows = NamedTuple[]
for (idx, p) in enumerate(points)
    idx == 1 && continue   # point 1 already fully profiled in Part 1
    if !p.feasible
        logprint("\n---- ", p.label, ": INFEASIBLE, skipped (recorded in points_summary.csv) ----")
        push!(part2_rows, (label = p.label, feasible = false, value_median_s = NaN, grad_median_s = NaN))
        continue
    end
    logprint("\n---- ", p.label, " ----")
    xfp = p.xf
    for _ in 1:1
        evaluate_fullA_fast(xfp, ctx; cache = nothing, warm = true, moment_representation = :dense)
    end
    val_times = Float64[@elapsed evaluate_fullA_fast(xfp, ctx; cache = nothing, warm = true, moment_representation = :dense) for _ in 1:2]
    basep = solve_base_state(xfp, ctx)
    composite_gradient_at_fast(xfp, ctx, pe; base = basep, threaded = true, h_mode = :adaptive, multi_method = :top3)
    grad_times = Float64[@elapsed composite_gradient_at_fast(xfp, ctx, pe; base = basep, threaded = true, h_mode = :adaptive, multi_method = :top3) for _ in 1:2]
    logprint(@sprintf("  value(warm) median=%.3fs  full_grad(top3,threaded,adaptive) median=%.3fs",
              median(val_times), median(grad_times)))
    push!(part2_rows, (label = p.label, feasible = true, value_median_s = median(val_times), grad_median_s = median(grad_times)))
end
write_csv_rows(joinpath(OUTDIR, "part2_lighter_checks.csv"), part2_rows)

# ============================================================================
# PART 3: memory ledger -- major arrays this process allocated, theoretical
# vs observed size where measurable.
# ============================================================================
logprint("\n", "="^90); logprint("PART 3: memory ledger"); logprint("="^90)
W80 = 80000
ledger = [
    (array = "ctx.U (Frechet draws)", shape = "$(W80) x $(D^2)", theoretical_MB = W80*D^2*8/1e6, observed_MB = sizeof(ctx.U)/1e6),
    (array = "inner K/G moment matrix (evaluate_fullA)", shape = "$(W80) x $(ctx.nTotalMoments)", theoretical_MB = W80*ctx.nTotalMoments*8/1e6, observed_MB = NaN),
    (array = "winners.jl gap (W x D)", shape = "$(W80) x $(D)", theoretical_MB = W80*D*8/1e6, observed_MB = NaN),
    (array = "lfix_incremental price/runnerup/third/contrib buffers (6x, W x D each)", shape = "6 x ($(W80) x $(D))", theoretical_MB = 6*W80*D*8/1e6, observed_MB = NaN),
    (array = "compressed_moments wval (W x D)", shape = "$(W80) x $(D)", theoretical_MB = W80*D*8/1e6, observed_MB = NaN),
    (array = "n_free (gradient vector length)", shape = "$(D^2)", theoretical_MB = D^2*8/1e6, observed_MB = NaN),
]
logprint(@sprintf("  %-55s %-16s %14s %14s", "array", "shape", "theory(MB)", "observed(MB)"))
for r in ledger
    logprint(@sprintf("  %-55s %-16s %14.2f %14s", r.array, r.shape, r.theoretical_MB, isnan(r.observed_MB) ? "n/a" : @sprintf("%.2f", r.observed_MB)))
end
write_csv_rows(joinpath(OUTDIR, "part3_memory_ledger.csv"), ledger)
logprint("VmHWM at end of run = ", vmhwm_kb(), " KB = ", round(vmhwm_kb()/1e6, digits=2), " GB")
logprint("Base.gc_live_bytes() at end of run = ", round(Base.gc_live_bytes()/1e6, digits=1), " MB")

logprint("\nc9_w80k_microbenchmark.jl COMPLETE at ", now())
close(LOGIO)
