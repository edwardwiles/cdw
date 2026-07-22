# ============================================================================
# Finalization task Phase 6 (2026-07-22): fair D=20/W=80000 benchmark of every gradient backend
# against the ACTUAL production default (composite_gradient_at_fast_buffered), extending
# c17_production_gate_benchmark_d20.jl's own harness (attributed -- structure, measure_median,
# rss_mb, and the correctness-cross-check-then-benchmark pattern are reused verbatim from that
# file, not re-derived) to add :pooled and :kbplus alongside the existing Reference/A+/C+
# comparison, and to emit machine-readable CSV/JSON per the brief's Phase 7 deliverable list.
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "bandwidth_quantile.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "lfix_buffer_reuse.jl"))
include(joinpath(@__DIR__, "gradient_workspace.jl"))
include(joinpath(@__DIR__, "lfix_base_workspace_pooled.jl"))
include(joinpath(@__DIR__, "lfix_factorized.jl"))
include(joinpath(@__DIR__, "lfix_factorized_workspace.jl"))
include(joinpath(@__DIR__, "lfix_kbplus.jl"))
include(joinpath(@__DIR__, "lfix_kbplus_workspace.jl"))
using Printf, Random, Dates, Statistics

lp(xs...) = (println(xs...); flush(stdout))
const OUTDIR = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c22_phase6_benchmark")
mkpath(OUTDIR)
rss_mb() = parse(Int, split(read(`ps -o rss= -p $(getpid())`, String))[1]) / 1024

function measure_median(f, label; n = 3)
    f()
    times = Float64[]; bytes = Float64[]
    local val
    for _ in 1:n
        GC.gc()
        stats = @timed (val = f())
        push!(times, stats.time); push!(bytes, stats.bytes)
    end
    t = median(times); b = median(bytes)
    lp(@sprintf("    [%-30s] median_wall=%8.4fs  median_bytes=%.3e (%8.1fMB)  rss_now=%9.1fMB", label, t, b, b / 2^20, rss_mb()))
    return val, (label = label, median_wall = t, median_bytes = b, median_mb = b / 2^20)
end

function run_point(δlabel::String, δ::Float64, ctx, pe, grad_pool, ws_a, ws_c, ws_kb)
    lp(""); lp("="^100); lp("POINT: ", δlabel, " (δ=", δ, ")"); lp("="^100)
    D = ctx.D; D2 = D^2
    x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
    Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
    zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, D), pe)
    gp0 = ctx.θ0_up[3+D]
    w0 = vcat(gp0, zfree0)
    xf0 = x_free_from_w(w0)
    base = solve_base_state(xf0, ctx)
    lp("  inner_status=", base.inner_status)

    lp("  --- correctness cross-check (all vs Reference, matching adaptive bandwidth) ---")
    bwc_ref = Dict{Int,Float64}(); bwc_pool = Dict{Int,Float64}(); bwc_a = Dict{Int,Float64}(); bwc_c = Dict{Int,Float64}(); bwc_kb = Dict{Int,Float64}()
    g_ref, _ = composite_gradient_at_fast_buffered(xf0, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bwc_ref)
    g_pool, _ = composite_gradient_at_fast_pooled(xf0, ctx, pe, grad_pool; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bwc_pool)
    g_a, _ = composite_gradient_at_Aplus(xf0, ctx, pe, grad_pool, ws_a; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bwc_a)
    g_c, _ = composite_gradient_at_Cplus(xf0, ctx, pe, grad_pool, ws_c; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bwc_c)
    g_kb, _ = composite_gradient_at_KBplus(xf0, ctx, pe, grad_pool, ws_kb; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bwc_kb)
    d_ref_pool = maximum(abs.(g_ref .- g_pool))
    d_ref_a = maximum(abs.(g_ref .- g_a))
    d_ref_c = maximum(abs.(g_ref .- g_c))
    d_ref_kb = maximum(abs.(g_ref .- g_kb))
    lp(@sprintf("    gradient maxabsdiff  Ref-vs-pooled=%.3e  Ref-vs-A+=%.3e  Ref-vs-C+=%.3e  Ref-vs-kbplus=%.3e",
        d_ref_pool, d_ref_a, d_ref_c, d_ref_kb))

    lp("  --- full gradient benchmark (median of 3, threaded=true, matches production) ---")
    bwc_ref2 = Dict{Int,Float64}()
    _, s_ref = measure_median(() -> composite_gradient_at_fast_buffered(xf0, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bwc_ref2), "Reference (fast_buffered, DEFAULT)")
    bwc_pool2 = Dict{Int,Float64}()
    _, s_pool = measure_median(() -> composite_gradient_at_fast_pooled(xf0, ctx, pe, grad_pool; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bwc_pool2), "pooled (GradWorkspacePool only)")
    bwc_a2 = Dict{Int,Float64}()
    _, s_a = measure_median(() -> composite_gradient_at_Aplus(xf0, ctx, pe, grad_pool, ws_a; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bwc_a2), "A+ (2-tensor+persist+pool)")
    bwc_c2 = Dict{Int,Float64}()
    _, s_c = measure_median(() -> composite_gradient_at_Cplus(xf0, ctx, pe, grad_pool, ws_c; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bwc_c2), "C+ (factorized+persist+pool)")
    bwc_kb2 = Dict{Int,Float64}()
    _, s_kb = measure_median(() -> composite_gradient_at_KBplus(xf0, ctx, pe, grad_pool, ws_kb; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bwc_kb2), "kbplus (ratio+persist+pool)")

    lp(@sprintf("    SPEEDUP vs Reference:  pooled=%.2fx  A+=%.2fx  C+=%.2fx  kbplus=%.2fx",
        s_ref.median_wall / s_pool.median_wall, s_ref.median_wall / s_a.median_wall,
        s_ref.median_wall / s_c.median_wall, s_ref.median_wall / s_kb.median_wall))
    lp(@sprintf("    MEMORY  vs Reference:  pooled=%.2fx less  A+=%.2fx less  C+=%.2fx less  kbplus=%.2fx less",
        s_ref.median_mb / max(s_pool.median_mb, 1e-6), s_ref.median_mb / max(s_a.median_mb, 1e-6),
        s_ref.median_mb / max(s_c.median_mb, 1e-6), s_ref.median_mb / max(s_kb.median_mb, 1e-6)))

    return (δlabel = δlabel, δ = δ,
        ref_wall = s_ref.median_wall, pool_wall = s_pool.median_wall, a_wall = s_a.median_wall, c_wall = s_c.median_wall, kb_wall = s_kb.median_wall,
        ref_mb = s_ref.median_mb, pool_mb = s_pool.median_mb, a_mb = s_a.median_mb, c_mb = s_c.median_mb, kb_mb = s_kb.median_mb,
        d_ref_pool = d_ref_pool, d_ref_a = d_ref_a, d_ref_c = d_ref_c, d_ref_kb = d_ref_kb)
end

lp("=== c22_phase6_fair_benchmark === ", Dates.now())
Random.seed!(20260719)
all_results = NamedTuple[]
for (δlabel, δ) in [("delta1_typical", 1.0), ("delta5_difficult", 5.0)]
    t0 = time()
    ctx = d20_real_setup(W = 80000, δ = δ, find_smallest = true)
    pe = build_pivot_elimination(ctx)
    D = ctx.D; W = 80000
    lp(@sprintf(">>> ctx built (δ=%.1f) in %.1fs. D=%d W=%d, RSS=%.1fMB", δ, time() - t0, D, W, rss_mb()))

    grad_pool = build_grad_workspace_pool(W)
    ws_a = build_lfix_base_workspace(D, W)
    ws_c = build_lfix_factorized_workspace(D, W)
    ws_kb = build_lfix_kbplus_workspace(D, W)

    push!(all_results, run_point(δlabel, δ, ctx, pe, grad_pool, ws_a, ws_c, ws_kb))
end

lp(""); lp("="^100); lp("SUMMARY"); lp("="^100)
lp(@sprintf("%-18s %8s %8s %8s %8s %8s | %7s %7s %7s %7s | %8s %8s %8s %8s %8s",
    "point", "ref_s", "pool_s", "A+_s", "C+_s", "kb_s", "pool_x", "A+_x", "C+_x", "kb_x",
    "ref_MB", "pool_MB", "A+_MB", "C+_MB", "kb_MB"))
for r in all_results
    lp(@sprintf("%-18s %8.3f %8.3f %8.3f %8.3f %8.3f | %6.2fx %6.2fx %6.2fx %6.2fx | %8.1f %8.1f %8.1f %8.1f %8.1f  diffs(pool/A+/C+/kb)=%.1e/%.1e/%.1e/%.1e",
        r.δlabel, r.ref_wall, r.pool_wall, r.a_wall, r.c_wall, r.kb_wall,
        r.ref_wall / r.pool_wall, r.ref_wall / r.a_wall, r.ref_wall / r.c_wall, r.ref_wall / r.kb_wall,
        r.ref_mb, r.pool_mb, r.a_mb, r.c_mb, r.kb_mb,
        r.d_ref_pool, r.d_ref_a, r.d_ref_c, r.d_ref_kb))
end

# Machine-readable output (task Phase 7 deliverable).
csv_path = joinpath(OUTDIR, "phase6_benchmark_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv")
open(csv_path, "w") do io
    println(io, "point,delta,backend,median_wall_s,median_mb,speedup_vs_ref,memory_reduction_vs_ref,maxabsdiff_vs_ref")
    for r in all_results
        println(io, "$(r.δlabel),$(r.δ),buffered,$(r.ref_wall),$(r.ref_mb),1.0,1.0,0.0")
        println(io, "$(r.δlabel),$(r.δ),pooled,$(r.pool_wall),$(r.pool_mb),$(r.ref_wall/r.pool_wall),$(r.ref_mb/max(r.pool_mb,1e-6)),$(r.d_ref_pool)")
        println(io, "$(r.δlabel),$(r.δ),aplus,$(r.a_wall),$(r.a_mb),$(r.ref_wall/r.a_wall),$(r.ref_mb/max(r.a_mb,1e-6)),$(r.d_ref_a)")
        println(io, "$(r.δlabel),$(r.δ),cplus,$(r.c_wall),$(r.c_mb),$(r.ref_wall/r.c_wall),$(r.ref_mb/max(r.c_mb,1e-6)),$(r.d_ref_c)")
        println(io, "$(r.δlabel),$(r.δ),kbplus,$(r.kb_wall),$(r.kb_mb),$(r.ref_wall/r.kb_wall),$(r.ref_mb/max(r.kb_mb,1e-6)),$(r.d_ref_kb)")
    end
end
lp(">>> CSV written: ", csv_path)
lp(">>> done ", Dates.now())
