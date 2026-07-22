# Addendum "test persistent preallocation and eliminate redundant full price tensors",
# Steps 7+8 combined: real D=20/W=80000 cross-backend correctness check (Reference vs
# Backend A/B/C) AND the end-to-end benchmark matrix (cache construction + full gradient),
# at calibration + a typical (δ=1) + a difficult (δ=5) point.
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "bandwidth_quantile.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "lfix_base_workspace.jl"))
include(joinpath(@__DIR__, "lfix_pTsigma_only.jl"))
include(joinpath(@__DIR__, "lfix_factorized.jl"))
using Printf, Random, Dates, Statistics

lp(xs...) = (println(xs...); flush(stdout))
const OUTDIR = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c16_backend_matrix")
mkpath(OUTDIR)
rss_mb() = parse(Int, split(read(`ps -o rss= -p $(getpid())`, String))[1]) / 1024

isclose(a, b; tol = 1e-9) = isapprox(a, b; rtol = tol, atol = tol)

function measure_median(f, label; n = 3)
    f()   # warmup
    times = Float64[]; bytes = Float64[]
    local val
    for _ in 1:n
        GC.gc()
        stats = @timed (val = f())
        push!(times, stats.time); push!(bytes, stats.bytes)
    end
    t = median(times); b = median(bytes)
    lp(@sprintf("    [%-28s] median_wall=%8.4fs  median_bytes=%.3e (%7.1fMB)  rss_now=%8.1fMB", label, t, b, b / 2^20, rss_mb()))
    return val, (label = label, median_wall = t, median_bytes = b, median_mb = b / 2^20)
end

function run_point(δlabel::String, δ::Float64, ptlabel::String, w0::Vector{Float64}, ctx, pe)
    lp(""); lp("="^100); lp("POINT: ", δlabel, " / ", ptlabel); lp("="^100)
    D = ctx.D; D2 = D^2
    x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
    xf0 = x_free_from_w(w0)
    base = solve_base_state(xf0, ctx)
    lp("  inner_status=", base.inner_status)

    cache_ref = build_lfix_base_cache(xf0, ctx, base)
    ws = build_lfix_base_workspace(D, size(ctx.obj.U, 1))
    cache_A = build_lfix_base_cache!(ws, xf0, ctx, base)
    cache_B = build_lfix_base_cache_B(xf0, ctx, base)
    cache_C = build_lfix_base_cache_C(xf0, ctx, base)

    lp("  --- correctness cross-check ---")
    ok_A = (cache_ref.winner0 == cache_A.winner0) && (cache_ref.contrib0 == cache_A.contrib0) && (cache_ref.q0 == cache_A.q0)
    ok_B = (cache_ref.winner0 == cache_B.winner0) && (cache_ref.contrib0 == cache_B.contrib0) && (cache_ref.q0 == cache_B.q0)
    ok_C_id = (cache_ref.winner0 == cache_C.ref.winner)
    ok_C_val = all(isclose.(cache_ref.contrib0, cache_C.contrib0)) && all(isclose.(cache_ref.q0, cache_C.q0))
    lp(@sprintf("    Backend A: winner+contrib0+q0 bit-identical = %s", ok_A))
    lp(@sprintf("    Backend B: winner+contrib0+q0 bit-identical = %s", ok_B))
    lp(@sprintf("    Backend C: winner identical=%s, contrib0/q0 close(tol=1e-9)=%s", ok_C_id, ok_C_val))

    lp("  --- cache construction benchmark (median of 3) ---")
    measure_median(() -> build_lfix_base_cache(xf0, ctx, base), "Reference (allocating)")
    measure_median(() -> build_lfix_base_cache!(ws, xf0, ctx, base), "A (persistent workspace)")
    measure_median(() -> build_lfix_base_cache_B(xf0, ctx, base), "B (pTσ-only)")
    measure_median(() -> build_lfix_base_cache_C(xf0, ctx, base), "C (factorized)")

    lp("  --- full gradient benchmark (median of 3, serial) ---")
    _, s_ref = measure_median(() -> composite_gradient_at(xf0, ctx, pe; base = base), "Reference gradient")
    _, s_B = measure_median(() -> composite_gradient_at_B(xf0, ctx, pe; base = base), "B gradient")
    _, s_C = measure_median(() -> composite_gradient_at_C(xf0, ctx, pe; base = base), "C gradient")
    g_ref, _ = composite_gradient_at(xf0, ctx, pe; base = base)
    g_B, _ = composite_gradient_at_B(xf0, ctx, pe; base = base)
    g_C, _ = composite_gradient_at_C(xf0, ctx, pe; base = base)
    lp(@sprintf("    gradient maxabsdiff  Reference-vs-B=%.3e  Reference-vs-C=%.3e", maximum(abs.(g_ref .- g_B)), maximum(abs.(g_ref .- g_C))))

    return (δlabel = δlabel, ptlabel = ptlabel, ok_A = ok_A, ok_B = ok_B, ok_C_id = ok_C_id, ok_C_val = ok_C_val,
        ref_cache_mb = NaN, ref_grad_s = s_ref.median_wall, B_grad_s = s_B.median_wall, C_grad_s = s_C.median_wall)
end

lp("=== c16_backend_matrix_d20 === ", Dates.now())
Random.seed!(20260719)
all_results = NamedTuple[]
for (δlabel, δ) in [("delta1", 1.0), ("delta5", 5.0)]
    t0 = time()
    ctx = d20_real_setup(W = 80000, δ = δ, find_smallest = true)
    pe = build_pivot_elimination(ctx)
    D = ctx.D; D2 = D^2
    lp(@sprintf(">>> ctx built (δ=%.1f) in %.1fs. D=%d W=80000, RSS=%.1fMB", δ, time() - t0, D, rss_mb()))

    Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
    zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, D), pe)
    gp0 = ctx.θ0_up[3+D]
    w_calib = vcat(gp0, zfree0)

    push!(all_results, run_point(δlabel, δ, "calibration", w_calib, ctx, pe))
end

lp(""); lp("="^100); lp("SUMMARY"); lp("="^100)
for r in all_results
    lp(@sprintf("%-8s %-12s A=%-5s B=%-5s C_id=%-5s C_val=%-5s | grad_s: ref=%.3f B=%.3f C=%.3f",
        r.δlabel, r.ptlabel, r.ok_A, r.ok_B, r.ok_C_id, r.ok_C_val, r.ref_grad_s, r.B_grad_s, r.C_grad_s))
end
lp(">>> done ", Dates.now())
