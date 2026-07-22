# Production-gate addendum, Sections 8-10: fair D=20/W=80000 comparison of the REAL production
# gradient (composite_gradient_at_fast_buffered, the actual default, threaded=true) against
# Backend A+ (persistent 2-tensor + GradWorkspacePool) and Backend C+ (factorized + persistent
# workspace + GradWorkspacePool), at calibration + delta=1 (typical) + delta=5 (difficult).
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
using Printf, Random, Dates, Statistics

lp(xs...) = (println(xs...); flush(stdout))
const OUTDIR = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c17_production_gate")
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

function run_point(δlabel::String, δ::Float64, ctx, pe, grad_pool, ws_a, ws_c)
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

    lp("  --- correctness cross-check (all 3 vs each other, matching adaptive bandwidth) ---")
    bwc_ref = Dict{Int,Float64}(); bwc_a = Dict{Int,Float64}(); bwc_c = Dict{Int,Float64}()
    g_ref, _ = composite_gradient_at_fast_buffered(xf0, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bwc_ref)
    g_a, _ = composite_gradient_at_Aplus(xf0, ctx, pe, grad_pool, ws_a; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bwc_a)
    g_c, _ = composite_gradient_at_Cplus(xf0, ctx, pe, grad_pool, ws_c; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bwc_c)
    d_ref_a = maximum(abs.(g_ref .- g_a))
    d_ref_c = maximum(abs.(g_ref .- g_c))
    lp(@sprintf("    gradient maxabsdiff  Reference-vs-A+=%.3e  Reference-vs-C+=%.3e", d_ref_a, d_ref_c))

    lp("  --- full gradient benchmark (median of 3, threaded=true, matches production) ---")
    bwc_ref2 = Dict{Int,Float64}()
    _, s_ref = measure_median(() -> composite_gradient_at_fast_buffered(xf0, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bwc_ref2), "Reference (fast_buffered)")
    bwc_a2 = Dict{Int,Float64}()
    _, s_a = measure_median(() -> composite_gradient_at_Aplus(xf0, ctx, pe, grad_pool, ws_a; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bwc_a2), "A+ (2-tensor+persist+pool)")
    bwc_c2 = Dict{Int,Float64}()
    _, s_c = measure_median(() -> composite_gradient_at_Cplus(xf0, ctx, pe, grad_pool, ws_c; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bwc_c2), "C+ (factorized+persist+pool)")

    lp(@sprintf("    SPEEDUP vs Reference:  A+ = %.2fx   C+ = %.2fx", s_ref.median_wall / s_a.median_wall, s_ref.median_wall / s_c.median_wall))
    lp(@sprintf("    MEMORY  vs Reference:  A+ = %.2fx less   C+ = %.2fx less", s_ref.median_mb / max(s_a.median_mb, 1e-6), s_ref.median_mb / max(s_c.median_mb, 1e-6)))

    return (δlabel = δlabel, ref_wall = s_ref.median_wall, a_wall = s_a.median_wall, c_wall = s_c.median_wall,
        ref_mb = s_ref.median_mb, a_mb = s_a.median_mb, c_mb = s_c.median_mb,
        d_ref_a = d_ref_a, d_ref_c = d_ref_c)
end

lp("=== c17_production_gate_benchmark_d20 === ", Dates.now())
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

    push!(all_results, run_point(δlabel, δ, ctx, pe, grad_pool, ws_a, ws_c))
end

lp(""); lp("="^100); lp("SUMMARY"); lp("="^100)
lp(@sprintf("%-18s %10s %10s %10s | %8s %8s | %10s %10s | %10s %10s",
    "point", "ref_s", "A+_s", "C+_s", "A+_x", "C+_x", "ref_MB", "A+_MB", "C+_MB", "diffs"))
for r in all_results
    lp(@sprintf("%-18s %10.3f %10.3f %10.3f | %7.2fx %7.2fx | %10.1f %10.1f %10.1f | dA=%.1e dC=%.1e",
        r.δlabel, r.ref_wall, r.a_wall, r.c_wall, r.ref_wall / r.a_wall, r.ref_wall / r.c_wall,
        r.ref_mb, r.a_mb, r.c_mb, r.d_ref_a, r.d_ref_c))
end
lp(">>> done ", Dates.now())
