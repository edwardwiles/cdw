# ============================================================================
# Continuation 8, Wave 2, workstream A: THE canonical performance profile,
# updating docs/fullA_performance_profile_v2.md now that compressed-mode
# (compressed_live.jl) and the winner accelerators (top-3 update +
# winner-margin certificate, wired via composite_gradient_fast.jl /
# lfix_incremental.jl / winner_certificate.jl) exist LIVE (both landed on
# this branch's base commit, d6e3b05, from Wave 1's two parallel
# workstreams -- see docs/compressed_live_integration_report.md and
# docs/winner_accelerator_live_wiring.md, both read in full before writing
# this file).
#
# ONE warmed Julia process, per the standing brief -- JIT/compile costs are
# paid once (the warm-up phase below) and excluded from every timed number.
#
# A note on why this is genuinely ONE process covering D=4/W=8000,
# D=4/W=80000, AND D=6/8(/10), where `audit_jach_d6d8.jl` (the file this
# investigation's own docs point to as "how a D=6/8/10 context is built
# here") deliberately runs in a SEPARATE process from its D=4 counterpart:
# that separation exists because `context.jl` is NOT include()-safe twice in
# one process (re-including it calls `CS.include(...)` again, redefining the
# CounterfactualSensitivity module's types and breaking type identity for
# any object already constructed against the OLD type -- see that script's
# own header comment). The hazard is specifically DOUBLE-INCLUDING
# `context.jl`, not calling `d4_exact_setup`/`d_exact_setup_scaled` multiple
# times with different (D,W) -- those are ordinary function calls. This
# harness includes `context_scaled.jl` ONCE as its sole root include (which
# itself includes `context.jl` exactly once), then calls `d4_exact_setup`
# for the D=4/W=8000 canonical point and `d_exact_setup_scaled` for every
# other (D,W) point in the grid, all safely in the same process. This is a
# genuine (small) infra improvement over the D=4-only / D=6-8-only script
# split used elsewhere in this directory, not a re-derivation of anything.
# ============================================================================
include(joinpath(@__DIR__, "context_scaled.jl"))   # -> d4_exact_setup AND d_exact_setup_scaled; includes context.jl exactly once
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
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c8_perfprofile")
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...)
    println(LOGIO, xs...)
    flush(stdout); flush(LOGIO)
end
logprint("c8_perfprofile_harness.jl starting at ", now(), "  commit=", COMMIT, "  nthreads=", Threads.nthreads())

"Minimal stats_row matching prof_summary()'s exact field set -- mirrors profile_p1a_warmed_and_redundancy.jl's own helper (reused pattern, not re-derived)."
function stats_row(label, times)
    sorted = sort(times); n = length(sorted)
    μ = sum(sorted) / n
    return (label = label, n = n, median_s = sorted[n÷2+1], min_s = sorted[1], max_s = sorted[end],
            p90_s = sorted[clamp(ceil(Int, 0.9 * n), 1, n)], p95_s = sorted[clamp(ceil(Int, 0.95 * n), 1, n)],
            mean_s = μ, std_s = (n > 1 ? sqrt(sum((t - μ)^2 for t in sorted) / (n - 1)) : 0.0),
            mean_alloc_bytes = NaN, total_alloc_bytes = NaN, mean_gc_s = NaN, total_gc_s = NaN)
end

x_free_from_w(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

"""
    find_feasible_point(ctx, pe; tries=6, seed=4200) -> (w0, xf, how)

Generic (D,W)-agnostic feasible-starting-point finder: tries the calibration
point (A_od theta==1) first, then the "natural" A_od theta baked into
ctx.θ0_up (the trick `run_d6_pilot.jl` found necessary at D=6+ -- the
calibration point is cold-infeasible there, per that script's own header
comment and `docs/fullA_block_local_performance.md` sec 7), then a bounded
number of small random perturbations around the natural point (mirroring
`audit_jach_d6d8.jl::try_scaled_feasible`'s own bounded-attempts discipline).
Returns `(nothing, nothing, "NONE_FOUND")` if all attempts fail -- callers
must check for this and skip that (D,W) point, per the task's explicit
"D=10 impractical is an acceptable partial result" allowance.
"""
function find_feasible_point(ctx, pe; tries::Int = 6, seed::Int = 4200)
    D = ctx.D
    gp0 = ctx.θ0_up[3+D]

    zfree_cal = pivot_reduce(zeros(D, D), pe)
    w_cal = vcat(gp0, zfree_cal)
    xf_cal = x_free_from_w(w_cal, pe)
    r_cal = evaluate_fullA(xf_cal, ctx; cache = nothing, warm = false)
    r_cal.inner_status in (0, -100, -101, -103) && return w_cal, xf_cal, "calibration"

    Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2]
    z0 = log.(Aod_theta_natural)
    zfree_nat = pivot_reduce(reshape(z0, D, D), pe)
    w_nat = vcat(gp0, zfree_nat)
    xf_nat = x_free_from_w(w_nat, pe)
    r_nat = evaluate_fullA(xf_nat, ctx; cache = nothing, warm = false)
    r_nat.inner_status in (0, -100, -101, -103) && return w_nat, xf_nat, "natural_theta"

    Random.seed!(seed)
    for attempt in 1:tries
        wp = w_nat .+ 0.02 .* (2 .* rand(length(w_nat)) .- 1)
        xfp = x_free_from_w(wp, pe)
        rp = evaluate_fullA(xfp, ctx; cache = nothing, warm = false)
        rp.inner_status in (0, -100, -101, -103) && return wp, xfp, "random_perturb_$attempt"
    end
    return nothing, nothing, "NONE_FOUND"
end

"""
    profile_value_mode(xf, ctx; mode, warm, N) -> (summ::Vector{NamedTuple}, meta::NamedTuple)

Warms 3 reps (untimed), resets the profiler, then times N reps of
`evaluate_fullA_fast(...; moment_representation=mode)`, in ITS OWN
`prof_reset!()` scope (matching `docs/fullA_performance_profile_v2.md`
sec 3's "warm and cold reps are profiled in separate scopes" methodology,
extended here to also separate dense from compressed, and D=4/W=8000 from
every other grid point -- reused discipline, not a new idea).
"""
function profile_value_mode(xf, ctx; mode::Symbol, warm::Bool, N::Int)
    for _ in 1:3
        evaluate_fullA_fast(xf, ctx; cache = nothing, warm = warm, moment_representation = mode)
    end
    prof_reset!()
    total_times = Float64[]
    n_fg = Int[]; n_hess = Int[]
    for _ in 1:N
        t0 = time_ns()
        r, meta = evaluate_fullA_fast(xf, ctx; cache = nothing, warm = warm, moment_representation = mode)
        push!(total_times, (time_ns() - t0) / 1e9)
        push!(n_fg, meta.n_fg_calls); push!(n_hess, meta.n_hess_calls)
    end
    summ = prof_summary()
    push!(summ, stats_row("TOTAL", total_times))
    return summ, (n_fg_mean = mean(n_fg), n_hess_mean = mean(n_hess), n = N,
                  status_ok = true)
end

"Paired dense-vs-compressed table: strips the `_compressed` suffix so matching labels line up; a dense-only or compressed-only row (e.g. `materialize_dense_for_postproc`, `TOTAL`) prints NaN on the missing side, not an error."
function print_paired_table(tag, summ_dense, summ_compressed)
    dense_by = Dict(r.label => r for r in summ_dense)
    comp_by = Dict(replace(r.label, "_compressed" => "") => r for r in summ_compressed)
    all_keys = union(keys(dense_by), keys(comp_by))
    logprint("\n---- ", tag, " : dense vs compressed, median ms (N reps each) ----")
    logprint(@sprintf("  %-32s %14s %16s %10s", "component", "dense(ms)", "compressed(ms)", "d/c ratio"))
    order = sort(collect(all_keys); by = kk -> -(haskey(dense_by, kk) ? dense_by[kk].median_s : (haskey(comp_by,kk) ? comp_by[kk].median_s : 0.0)))
    rows = NamedTuple[]
    for k in order
        dm = haskey(dense_by, k) ? dense_by[k].median_s * 1000 : NaN
        cm = haskey(comp_by, k) ? comp_by[k].median_s * 1000 : NaN
        ratio = (isfinite(dm) && isfinite(cm) && cm > 0) ? dm / cm : NaN
        logprint(@sprintf("  %-32s %14.4f %16.4f %10.3f", k, dm, cm, ratio))
        push!(rows, (component = k, dense_ms = dm, compressed_ms = cm, ratio_d_over_c = ratio))
    end
    write_csv_rows(joinpath(OUTDIR, "paired_$(tag).csv"), rows)
    return rows
end

# ============================================================================
# SETUP: D=4/W=8000 canonical context + the standing candidate point
# (upper_lfixcomposite_sr1_60s, from candidate_registry.jl -- the SAME point
# every other continuation-8 report profiles at, chosen for comparability,
# not re-derived here).
# ============================================================================
logprint("\n" * "="^90); logprint("SETUP: D=4/W=8000"); logprint("="^90)
ctx4 = d4_exact_setup(find_smallest = true)
pe4 = build_pivot_elimination(ctx4)
D4 = ctx4.D; W4 = size(ctx4.U, 1)
logprint("D=", D4, " W=", W4, " nTotalMoments=", ctx4.nTotalMoments)

const W_UPPER = [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181,
    0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515,
    1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252,
    0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]
xf_upper = x_free_from_w(W_UPPER, pe4)

# ---- warm-up phase (untimed): pay every JIT/compile cost once, here ----
logprint("\nWarming up (untimed)...")
for _ in 1:3
    evaluate_fullA_fast(xf_upper, ctx4; cache = nothing, warm = true, moment_representation = :dense)
    evaluate_fullA_fast(xf_upper, ctx4; cache = nothing, warm = true, moment_representation = :compressed)
end
base0_4 = solve_base_state(xf_upper, ctx4)
for mm in (:top3, :generic), th in (true, false), hm in (:adaptive, :fixed)
    composite_gradient_at_fast(xf_upper, ctx4, pe4; base = base0_4, threaded = th, h_mode = hm, multi_method = mm)
end
wc_warmup = PersistentWinnerCache()
composite_gradient_at_fast(xf_upper, ctx4, pe4; base = base0_4, threaded = true, winner_cache_mode = :certificate, winner_cache = wc_warmup)
cache4_warmup = build_lfix_base_cache(xf_upper, ctx4, base0_4)
lfix_value_certified(cache4_warmup, wc_warmup, ctx4, xf_upper)
logprint("Warm-up complete.")

# ============================================================================
# PART A: exact value callback, dense vs compressed, D=4/W=8000, warm + cold
# ============================================================================
logprint("\n" * "="^90); logprint("PART A: value callback, D=4/W=8000, dense vs compressed"); logprint("="^90)

prof_reset!(); summ_dense_warm, meta_dense_warm = profile_value_mode(xf_upper, ctx4; mode = :dense, warm = true, N = 50)
prof_reset!(); summ_comp_warm, meta_comp_warm = profile_value_mode(xf_upper, ctx4; mode = :compressed, warm = true, N = 50)
rows_warm = print_paired_table("d4w8000_warm", summ_dense_warm, summ_comp_warm)
logprint("  n_fg_calls/call: dense=", meta_dense_warm.n_fg_mean, " compressed=", meta_comp_warm.n_fg_mean,
         "   n_hess_calls/call: dense=", meta_dense_warm.n_hess_mean, " compressed=", meta_comp_warm.n_hess_mean)

prof_reset!(); summ_dense_cold, meta_dense_cold = profile_value_mode(xf_upper, ctx4; mode = :dense, warm = false, N = 15)
prof_reset!(); summ_comp_cold, meta_comp_cold = profile_value_mode(xf_upper, ctx4; mode = :compressed, warm = false, N = 15)
rows_cold = print_paired_table("d4w8000_cold", summ_dense_cold, summ_comp_cold)
logprint("  n_fg_calls/call: dense=", meta_dense_cold.n_fg_mean, " compressed=", meta_comp_cold.n_fg_mean,
         "   n_hess_calls/call: dense=", meta_dense_cold.n_hess_mean, " compressed=", meta_comp_cold.n_hess_mean)

write_csv_rows(joinpath(OUTDIR, "d4w8000_warm_dense_full.csv"), summ_dense_warm)
write_csv_rows(joinpath(OUTDIR, "d4w8000_warm_compressed_full.csv"), summ_comp_warm)

# ============================================================================
# PART B: D=4/W=80000, one point -- does the dense-vs-compressed FG-callback
# verdict flip with 10x more draws?
# ============================================================================
logprint("\n" * "="^90); logprint("PART B: value callback, D=4/W=80000 (one point)"); logprint("="^90)
ctx4b = d_exact_setup_scaled(D = 4, W = 80000, find_smallest = true)
pe4b = build_pivot_elimination(ctx4b)
w0_80k, xf_80k, how_80k = find_feasible_point(ctx4b, pe4b)
if xf_80k === nothing
    logprint("D=4/W=80000: NO feasible point found -- SKIPPED (see find_feasible_point's bounded-attempts discipline).")
else
    logprint("D=4/W=80000: feasible point found via '", how_80k, "'.")
    for _ in 1:2
        evaluate_fullA_fast(xf_80k, ctx4b; cache = nothing, warm = true, moment_representation = :dense)
        evaluate_fullA_fast(xf_80k, ctx4b; cache = nothing, warm = true, moment_representation = :compressed)
    end
    prof_reset!(); summ_dense_80k, meta_dense_80k = profile_value_mode(xf_80k, ctx4b; mode = :dense, warm = true, N = 25)
    prof_reset!(); summ_comp_80k, meta_comp_80k = profile_value_mode(xf_80k, ctx4b; mode = :compressed, warm = true, N = 25)
    rows_80k = print_paired_table("d4w80000_warm", summ_dense_80k, summ_comp_80k)
    logprint("  n_fg_calls/call: dense=", meta_dense_80k.n_fg_mean, " compressed=", meta_comp_80k.n_fg_mean)

    fg_row_d = only(filter(r -> r.component == "inner_dual_fg_callback", rows_80k))
    fg_row_8k = only(filter(r -> r.component == "inner_dual_fg_callback", rows_warm))
    logprint("\n  VERDICT CHECK -- inner_dual_fg_callback dense/compressed ratio: W=8000 -> ", round(fg_row_8k.ratio_d_over_c, digits = 3),
             "   W=80000 -> ", round(fg_row_d.ratio_d_over_c, digits = 3),
             "   (ratio<1 means compressed is SLOWER; flips to >1 if compressed becomes faster at higher W)")
end

# ============================================================================
# PART C: hard L_fix gradient breakdown, D=4/W=8000, at the SAME candidate
# ============================================================================
logprint("\n" * "="^90); logprint("PART C: hard L_fix gradient breakdown, D=4/W=8000"); logprint("="^90)

# ---- C1: full 15-coordinate gradient wall time, top3 vs generic, threaded vs not, adaptive vs fixed-h ----
function bench_full_grad(; multi_method::Symbol, threaded::Bool, h_mode::Symbol, N::Int = 15)
    for _ in 1:2
        composite_gradient_at_fast(xf_upper, ctx4, pe4; base = base0_4, threaded = threaded, h_mode = h_mode, multi_method = multi_method)
    end
    times = Float64[]
    for _ in 1:N
        push!(times, @elapsed composite_gradient_at_fast(xf_upper, ctx4, pe4; base = base0_4, threaded = threaded, h_mode = h_mode, multi_method = multi_method))
    end
    return median(times), times
end

logprint("\n---- C1: full-gradient wall time (median ms, N=15), all 8 (multi_method x threaded x h_mode) combos ----")
c1_rows = NamedTuple[]
for hm in (:adaptive, :fixed), th in (true, false)
    med_top3, _ = bench_full_grad(multi_method = :top3, threaded = th, h_mode = hm)
    med_gen, _ = bench_full_grad(multi_method = :generic, threaded = th, h_mode = hm)
    ratio = med_gen / med_top3
    logprint(@sprintf("  h_mode=%-8s threaded=%-5s   top3=%.4fms  generic=%.4fms  generic/top3=%.3f",
                        hm, th, med_top3 * 1000, med_gen * 1000, ratio))
    push!(c1_rows, (h_mode = hm, threaded = th, top3_ms = med_top3 * 1000, generic_ms = med_gen * 1000, generic_over_top3 = ratio))
end
write_csv_rows(joinpath(OUTDIR, "c1_full_grad_top3_vs_generic.csv"), c1_rows)

# ---- C2: component-level decomposition (external timing around the EXISTING public functions -- no source edits) ----
logprint("\n---- C2: component decomposition, h_mode=:adaptive, multi_method=:top3, N=10 warmed reps ----")
function decompose_gradient(x_free0, ctx, pe, base; h_mode::Symbol = :adaptive, multi_method::Symbol = :top3)
    D = ctx.D; D2 = D^2
    t_cache0 = time_ns()
    cache = build_lfix_base_cache(x_free0, ctx, base)
    t_cache = (time_ns() - t_cache0) / 1e9

    z0 = log.(reshape(x_free0[2:end], D, D))
    w0 = vcat(x_free0[1], pivot_reduce(z0, pe))

    t_gamma0 = time_ns()
    g1 = gamma_component_analytic(cache, base, w0[1])
    t_gamma = (time_ns() - t_gamma0) / 1e9

    t_bandwidth = 0.0; t_fd = 0.0
    for k in 2:D2
        if h_mode == :fixed
            t0 = time_ns()
            a_block_fd_component(cache, ctx, pe, w0, k, 0.01; multi_method = multi_method)
            t_fd += (time_ns() - t0) / 1e9
        else
            t0 = time_ns()
            h, m, _ = select_bandwidth(cache, ctx, pe, w0, k; multi_method = multi_method)
            t_bandwidth += (time_ns() - t0) / 1e9
            t0 = time_ns()
            a_block_fd_component(cache, ctx, pe, w0, k, h; multi_method = multi_method)
            a_block_fd_component(cache, ctx, pe, w0, k, h / 2; multi_method = multi_method)
            t_fd += (time_ns() - t0) / 1e9
        end
    end
    return (cache_build = t_cache, gamma_analytic = t_gamma, bandwidth_selection = t_bandwidth, fd_probes = t_fd)
end

t_base_solve = median([@elapsed solve_base_state(xf_upper, ctx4) for _ in 1:10])
decomp_top3 = [decompose_gradient(xf_upper, ctx4, pe4, base0_4; h_mode = :adaptive, multi_method = :top3) for _ in 1:10]
decomp_generic = [decompose_gradient(xf_upper, ctx4, pe4, base0_4; h_mode = :adaptive, multi_method = :generic) for _ in 1:10]
med_field(v, f) = median([getfield(x, f) for x in v])
logprint(@sprintf("  base_state solve (shareable w/ eval_F): %.4fms", t_base_solve * 1000))
for f in (:cache_build, :gamma_analytic, :bandwidth_selection, :fd_probes)
    logprint(@sprintf("  %-22s  top3=%.4fms  generic=%.4fms", f, med_field(decomp_top3, f) * 1000, med_field(decomp_generic, f) * 1000))
end
total_top3 = t_base_solve + sum(med_field(decomp_top3, f) for f in (:cache_build, :gamma_analytic, :bandwidth_selection, :fd_probes))
total_generic = t_base_solve + sum(med_field(decomp_generic, f) for f in (:cache_build, :gamma_analytic, :bandwidth_selection, :fd_probes))
logprint(@sprintf("  RECONSTRUCTED TOTAL (sum of components): top3=%.4fms  generic=%.4fms", total_top3 * 1000, total_generic * 1000))
write_csv_rows(joinpath(OUTDIR, "c2_component_decomposition.csv"),
    [(multi_method = "top3", base_state_ms = t_base_solve*1000, cache_build_ms = med_field(decomp_top3,:cache_build)*1000,
      gamma_ms = med_field(decomp_top3,:gamma_analytic)*1000, bandwidth_ms = med_field(decomp_top3,:bandwidth_selection)*1000,
      fd_probes_ms = med_field(decomp_top3,:fd_probes)*1000, total_ms = total_top3*1000),
     (multi_method = "generic", base_state_ms = t_base_solve*1000, cache_build_ms = med_field(decomp_generic,:cache_build)*1000,
      gamma_ms = med_field(decomp_generic,:gamma_analytic)*1000, bandwidth_ms = med_field(decomp_generic,:bandwidth_selection)*1000,
      fd_probes_ms = med_field(decomp_generic,:fd_probes)*1000, total_ms = total_generic*1000)])

# ---- C2b: isolate the winner-update (top3 vs generic) contribution AT the 3 known fallback
# coordinates (14,15,16 -- share the gravity pivot's destination, per docs/winner_accelerator_live_wiring.md
# sec "real coordinate sweep") vs an ordinary single-changed-origin coordinate (k=2), directly. ----
logprint("\n---- C2b: per-coordinate cost, fallback coords (14/15/16) vs an ordinary coord (2), top3 vs generic ----")
function coord_cost(cache, ctx, pe, w0, k; multi_method::Symbol, N::Int = 30)
    for _ in 1:3
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
cache4_for_coords = build_lfix_base_cache(xf_upper, ctx4, base0_4)
z0u = log.(reshape(xf_upper[2:end], D4, D4)); w0u = vcat(xf_upper[1], pivot_reduce(z0u, pe4))
c2b_rows = NamedTuple[]
for k in (2, 14, 15, 16)
    m_top3 = coord_cost(cache4_for_coords, ctx4, pe4, w0u, k; multi_method = :top3)
    m_gen = coord_cost(cache4_for_coords, ctx4, pe4, w0u, k; multi_method = :generic)
    logprint(@sprintf("  coord k=%-3d  top3=%.5fms  generic=%.5fms  ratio(gen/top3)=%.3f", k, m_top3*1000, m_gen*1000, m_gen/m_top3))
    push!(c2b_rows, (k = k, top3_ms = m_top3*1000, generic_ms = m_gen*1000, ratio_gen_over_top3 = m_gen/m_top3))
end
write_csv_rows(joinpath(OUTDIR, "c2b_percoord_top3_vs_generic.csv"), c2b_rows)

# ---- C3: winner-margin certificate, IN CONTEXT of composite_gradient_at_fast ----
logprint("\n---- C3: winner-margin certificate overhead/benefit, in composite_gradient_at_fast's own context ----")

# (a) SINGLE full 15-coord gradient eval, cold anchor each time (reset! between reps) vs winner_cache_mode=:none
t_none_single = median([@elapsed composite_gradient_at_fast(xf_upper, ctx4, pe4; base = base0_4, threaded = true, winner_cache_mode = :none) for _ in 1:10])
t_cert_cold_single = Float64[]
for _ in 1:8
    wc = PersistentWinnerCache()
    push!(t_cert_cold_single, @elapsed composite_gradient_at_fast(xf_upper, ctx4, pe4; base = base0_4, threaded = true, winner_cache_mode = :certificate, winner_cache = wc))
end
logprint(@sprintf("  single call, winner_cache_mode=:none                 : %.4fms (median of 10)", t_none_single*1000))
logprint(@sprintf("  single call, winner_cache_mode=:certificate (COLD anchor, N=8): %.4fms (median)", median(t_cert_cold_single)*1000))
logprint(@sprintf("  overhead of the certificate diagnostic on a SINGLE cold call: %.4fms (%.2f%% of the :none baseline)",
                    (median(t_cert_cold_single)-t_none_single)*1000, 100*(median(t_cert_cold_single)-t_none_single)/t_none_single))

# (b) SEQUENCE of K nearby points (A-block perturbations of the same incumbent, magnitudes matching
# Wave 1's own line-search-sweep set), reusing ONE PersistentWinnerCache across the whole sequence --
# isolates whether the certificate's cold-vs-warm trend shows up EMBEDDED inside repeated full-gradient
# calls (not just in the standalone lfix_value_certified sweep Wave 1 already benchmarked).
Random.seed!(777)
const K_SEQ = 10
const STEP_MAGS = [1e-3, 5e-3, 1e-2, 2e-2, 5e-2]
seq_w = Vector{Vector{Float64}}()
for i in 1:K_SEQ
    mag = STEP_MAGS[rand(1:length(STEP_MAGS))]
    dirn = randn(D4^2); dirn ./= norm(dirn)
    wp = copy(W_UPPER); wp[2:end] .+= mag .* dirn[2:end]
    push!(seq_w, wp)
end
# warm-up the sequence path once (untimed) so JIT doesn't pollute rep 1
seq_xf = [x_free_from_w(w, pe4) for w in seq_w]
composite_gradient_at_fast(seq_xf[1], ctx4, pe4; threaded = true, winner_cache_mode = :none)

t_seq_none = Float64[]
for xf in seq_xf
    push!(t_seq_none, @elapsed composite_gradient_at_fast(xf, ctx4, pe4; threaded = true, winner_cache_mode = :none))
end
wc_seq = PersistentWinnerCache()
t_seq_cert = Float64[]
for xf in seq_xf
    push!(t_seq_cert, @elapsed composite_gradient_at_fast(xf, ctx4, pe4; threaded = true, winner_cache_mode = :certificate, winner_cache = wc_seq))
end
logprint("\n  K=", K_SEQ, "-point nearby-perturbation sequence, FULL composite_gradient_at_fast wall time per point (ms):")
logprint("    :none      ", round.(t_seq_none .* 1000, digits = 3))
logprint("    :certificate ", round.(t_seq_cert .* 1000, digits = 3))
logprint(@sprintf("  totals: :none=%.2fms  :certificate=%.2fms  ratio=%.3f (in-context, full-gradient level)",
                    sum(t_seq_none)*1000, sum(t_seq_cert)*1000, sum(t_seq_none)/sum(t_seq_cert)))
rep_cert = winner_cache_report(wc_seq)
logprint("  wc_seq certificate report: ", rep_cert)

# ---- isolate JUST the certificate call's OWN cost trajectory (winner_value_update! alone, no
# gradient math around it) across the same sequence -- cold-vs-warm signal, cleanly, in-context ----
wc_isolated = PersistentWinnerCache()
t_isolated = Float64[]
for xf in seq_xf
    push!(t_isolated, @elapsed winner_value_update!(wc_isolated, ctx4, xf))
end
logprint("\n  ISOLATED winner_value_update! cost alone across the same K=", K_SEQ, " sequence (ms): ", round.(t_isolated .* 1000, digits = 4))
logprint("  first (cold, anchor build) vs subsequent (certified) ratio: ", round(t_isolated[1] / median(t_isolated[2:end]), digits = 2), "x")
rep_isolated = winner_cache_report(wc_isolated)
logprint("  wc_isolated certificate report: ", rep_isolated)

write_csv_rows(joinpath(OUTDIR, "c3_certificate_sequence.csv"),
    [(idx = i, none_ms = t_seq_none[i]*1000, certificate_ms = t_seq_cert[i]*1000, isolated_certificate_call_ms = t_isolated[i]*1000) for i in 1:K_SEQ])

logprint("\nPART A/B/C complete. See ", OUTDIR, " for CSVs.")

# ============================================================================
# PART D: D-scaling grid (D=6, D=8, attempt D=10), W=8000, SAME process --
# reuses `d_exact_setup_scaled` (context_scaled.jl, already included as this
# harness's root include) and the "natural A_od theta" feasible-start trick
# `run_d6_pilot.jl` established (calibration is cold-infeasible at D>=6, per
# that script's own header and docs/fullA_block_local_performance.md sec 7)
# via this file's own `find_feasible_point` helper above -- not rederiving
# context-construction machinery, per the task's explicit instruction.
#
# Lighter breakdown than Parts A-C (TOTAL dense/compressed + the two
# highest-value single-line comparisons -- inner_moment_build and
# inner_dual_fg_callback -- plus a full-gradient wall-clock number), given
# the time budget; D=10 is attempted but SKIPPED with a clear note if no
# feasible point is found within the bounded attempts, per the task's
# explicit "D=4/6/8 partial is acceptable" allowance.
# ============================================================================
logprint("\n" * "="^90); logprint("PART D: D-scaling grid, W=8000"); logprint("="^90)

d_scaling_rows = NamedTuple[]
for D_try in (6, 8, 10)
    logprint("\n---- D=", D_try, " ----")
    local ctxD, peD, w0D, xfD, howD
    t_setup0 = time()
    ctxD = d_exact_setup_scaled(D = D_try, W = 8000, find_smallest = true)
    t_setup = time() - t_setup0
    peD = build_pivot_elimination(ctxD)
    w0D, xfD, howD = find_feasible_point(ctxD, peD)
    if xfD === nothing
        logprint("D=", D_try, ": NO feasible point found (calibration + natural-theta + 6 random perturbations all failed) -- SKIPPED.")
        push!(d_scaling_rows, (D = D_try, status = "skipped_infeasible", setup_s = t_setup,
              total_dense_ms = NaN, total_compressed_ms = NaN, moment_build_dense_ms = NaN, moment_build_compressed_ms = NaN,
              fg_callback_dense_ms = NaN, fg_callback_compressed_ms = NaN, full_grad_ms = NaN))
        continue
    end
    logprint("D=", D_try, ": feasible via '", howD, "', setup wall=", round(t_setup, digits = 2), "s")

    # warm-up (untimed)
    for _ in 1:2
        evaluate_fullA_fast(xfD, ctxD; cache = nothing, warm = true, moment_representation = :dense)
        evaluate_fullA_fast(xfD, ctxD; cache = nothing, warm = true, moment_representation = :compressed)
    end
    N_D = D_try <= 6 ? 20 : 10
    prof_reset!(); summ_dense_D, meta_dense_D = profile_value_mode(xfD, ctxD; mode = :dense, warm = true, N = N_D)
    prof_reset!(); summ_comp_D, meta_comp_D = profile_value_mode(xfD, ctxD; mode = :compressed, warm = true, N = N_D)
    rows_D = print_paired_table("d$(D_try)w8000_warm", summ_dense_D, summ_comp_D)

    total_d = only(filter(r -> r.component == "TOTAL", rows_D))
    mb_row = only(filter(r -> r.component == "inner_moment_build", rows_D))
    fg_row = only(filter(r -> r.component == "inner_dual_fg_callback", rows_D))

    # full-gradient wall-clock, top3, threaded, adaptive-h (fewer reps at larger D)
    baseD = solve_base_state(xfD, ctxD)
    composite_gradient_at_fast(xfD, ctxD, peD; base = baseD, threaded = true, h_mode = :adaptive, multi_method = :top3)   # warm-up
    N_grad = D_try <= 6 ? 8 : 5
    t_grad_D = median([@elapsed composite_gradient_at_fast(xfD, ctxD, peD; base = baseD, threaded = true, h_mode = :adaptive, multi_method = :top3) for _ in 1:N_grad])
    logprint(@sprintf("  full composite_gradient_at_fast (top3, threaded, adaptive-h), D=%d: median=%.2fms (N=%d)", D_try, t_grad_D*1000, N_grad))

    push!(d_scaling_rows, (D = D_try, status = "ok", setup_s = t_setup,
          total_dense_ms = total_d.dense_ms, total_compressed_ms = total_d.compressed_ms,
          moment_build_dense_ms = mb_row.dense_ms, moment_build_compressed_ms = mb_row.compressed_ms,
          fg_callback_dense_ms = fg_row.dense_ms, fg_callback_compressed_ms = fg_row.compressed_ms,
          full_grad_ms = t_grad_D * 1000))
end

logprint("\n---- D-scaling summary table ----")
logprint(@sprintf("  %-4s %-10s %10s %10s %14s %14s %10s", "D", "status", "TOT_dense", "TOT_comp", "fg_dense", "fg_comp", "grad_ms"))
for r in d_scaling_rows
    logprint(@sprintf("  %-4d %-10s %10.3f %10.3f %14.4f %14.4f %10.2f",
              r.D, r.status, r.total_dense_ms, r.total_compressed_ms, r.fg_callback_dense_ms, r.fg_callback_compressed_ms, r.full_grad_ms))
end
write_csv_rows(joinpath(OUTDIR, "d_scaling_grid.csv"), d_scaling_rows)

logprint("\nc8_perfprofile_harness.jl COMPLETE at ", now())
close(LOGIO)
