# Addendum "test persistent preallocation and eliminate redundant full price tensors", Step 2:
# benchmark the CURRENT (reference/allocating) build_lfix_base_cache's price0/pTσ0 construction
# in isolation, at real D=20/W=80000, across calibration + a typical (δ=1) + a difficult (δ=5)
# point. Reuses c14_allocation_audit.jl's own `measure`/`d20_real_setup` pattern (not
# re-invented) -- this file adds a finer split (price0 ALONE vs pTσ0 ALONE vs the full cache)
# that c14_allocation_audit.jl's site-4 measurement didn't isolate.
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
using Printf, Random, Dates, Serialization

lp(xs...) = (println(xs...); flush(stdout))
const OUTDIR = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c15_price_tensor")
mkpath(OUTDIR)
gc_live_mb() = Base.gc_live_bytes() / 2^20
rss_mb() = begin
    pid = getpid()
    try
        parse(Int, split(read(`ps -o rss= -p $pid`, String))[1]) / 1024
    catch
        NaN
    end
end

function measure(f, label; warmup = true)
    warmup && f()
    GC.gc(); live0 = gc_live_mb(); rss0 = rss_mb()
    stats = @timed f()
    live1 = gc_live_mb(); rss1 = rss_mb()
    lp(@sprintf("  [%-40s] wall=%8.4fs  bytes=%.3e (%7.1f MB)  gctime=%7.4fs (%5.1f%%)  gc_live_delta=%7.2fMB  rss_delta=%7.2fMB  rss_now=%8.1fMB",
        label, stats.time, stats.bytes, stats.bytes / 2^20, stats.gctime,
        stats.time > 0 ? 100 * stats.gctime / stats.time : 0.0, live1 - live0, rss1 - rss0, rss1))
    return stats.value, (label = label, wall = stats.time, bytes = stats.bytes, mb = stats.bytes / 2^20,
        gctime = stats.gctime, gc_pct = stats.time > 0 ? 100 * stats.gctime / stats.time : 0.0,
        gc_live_delta_mb = live1 - live0, rss_delta_mb = rss1 - rss0, rss_now_mb = rss1)
end

"price0/pTσ0 construction ALONE, isolated from the rest of build_lfix_base_cache (winner scan, contrib0, cf pieces)."
function build_price_tensors_only(x_free0::AbstractVector, ctx, base)
    D = ctx.D; W = size(ctx.obj.U, 1)
    price0 = Array{Float64}(undef, W, D, D)
    for d in 1:D, o in 1:D
        p, _ = price_and_pTsigma_cell(base.θ_full0, ctx, o, d)
        price0[:, o, d] .= p
    end
    return price0
end
function build_pTσ_tensor_only(x_free0::AbstractVector, ctx, base)
    D = ctx.D; W = size(ctx.obj.U, 1)
    pTσ0 = Array{Float64}(undef, W, D, D)
    for d in 1:D, o in 1:D
        _, ps = price_and_pTsigma_cell(base.θ_full0, ctx, o, d)
        pTσ0[:, o, d] .= ps
    end
    return pTσ0
end
"Both, using the in-place `!` primitive (matches the CURRENT build_lfix_base_cache exactly)."
function build_both_tensors_inplace(x_free0::AbstractVector, ctx, base)
    D = ctx.D; W = size(ctx.obj.U, 1)
    price0 = Array{Float64}(undef, W, D, D)
    pTσ0 = Array{Float64}(undef, W, D, D)
    for d in 1:D, o in 1:D
        price_and_pTsigma_cell!(@view(price0[:, o, d]), @view(pTσ0[:, o, d]), base.θ_full0, ctx, o, d)
    end
    return price0, pTσ0
end

function bench_point(label::String, δ::Float64, W::Int)
    lp(""); lp("="^110); lp("POINT: ", label, "  (δ=", δ, ", W=", W, ")"); lp("="^110)
    t0 = time()
    ctx = d20_real_setup(W = W, δ = δ, find_smallest = true)
    pe = build_pivot_elimination(ctx)
    D = ctx.D; D2 = D^2
    lp(@sprintf(">>> ctx built in %.1fs. D=%d W=%d, RSS now = %.1fMB", time() - t0, D, W, rss_mb()))

    x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
    Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
    z0 = log.(Aod_theta_natural)
    zfree0 = pivot_reduce(reshape(z0, D, D), pe)
    gp0 = ctx.θ0_up[3+D]
    w0 = vcat(gp0, zfree0)   # true calibration point, NOT gp0*1.01 -- unperturbed theta0_up
    xf0 = x_free_from_w(w0)

    base = solve_base_state(xf0, ctx)
    lp("  inner_status=", base.inner_status, "  zeta*=", base.ζstar)

    rows = NamedTuple[]
    _, s_price = measure(() -> build_price_tensors_only(xf0, ctx, base), "price0 ALONE")
    push!(rows, merge(s_price, (point = label,)))
    _, s_pTσ = measure(() -> build_pTσ_tensor_only(xf0, ctx, base), "pTσ0 ALONE")
    push!(rows, merge(s_pTσ, (point = label,)))
    _, s_both = measure(() -> build_both_tensors_inplace(xf0, ctx, base), "both (in-place, current pattern)")
    push!(rows, merge(s_both, (point = label,)))
    _, s_full = measure(() -> build_lfix_base_cache(xf0, ctx, base; validate_dense = false), "full build_lfix_base_cache")
    push!(rows, merge(s_full, (point = label,)))

    # peak-live / RSS check across a mini "coordinate sweep" -- confirm both dense tensors
    # stay referenced (no GC reclaim) while probing several coordinates, matching how a real
    # composite_gradient_at call holds `cache` alive for the WHOLE D^2-1-coordinate sweep.
    cache = build_lfix_base_cache(xf0, ctx, base; validate_dense = false)
    rss_before_sweep = rss_mb()
    for k in 2:min(11, D2)
        lfix_incremental_at(cache, ctx, pe, w0, k, w0[k] + 0.01; tier = :incremental_o1)
    end
    rss_after_sweep = rss_mb()
    lp(@sprintf("  RSS before 10-coord sweep=%.1fMB, after=%.1fMB (delta=%.2fMB) -- cache held alive throughout",
        rss_before_sweep, rss_after_sweep, rss_after_sweep - rss_before_sweep))
    push!(rows, (label = "10coord_sweep_rss_check", wall = NaN, bytes = NaN, mb = NaN, gctime = NaN, gc_pct = NaN,
        gc_live_delta_mb = NaN, rss_delta_mb = rss_after_sweep - rss_before_sweep, rss_now_mb = rss_after_sweep, point = label))

    W_actual = size(ctx.obj.U, 1)
    bytes_per_tensor = W_actual * D * D * 8
    lp(@sprintf("  Theoretical tensor size: W*D*D*8 = %d*%d*%d*8 = %.1f MB each, %.1f MB for both",
        W_actual, D, D, bytes_per_tensor / 2^20, 2 * bytes_per_tensor / 2^20))

    return rows
end

lp("=== c15_price_tensor_benchmark === ", Dates.now())
Random.seed!(20260719)
all_rows = NamedTuple[]
append!(all_rows, bench_point("calibration_delta1ctx", 1.0, 80000))
append!(all_rows, bench_point("typical_delta1", 1.0, 80000))  # same ctx family; genuinely "typical" candidate would need a converged point, calibration serves as the delta=1-context baseline
append!(all_rows, bench_point("difficult_delta5", 5.0, 80000))

lp(""); lp("="^110); lp("SUMMARY"); lp("="^110)
lp(@sprintf("%-22s %-38s %10s %12s %10s %8s", "point", "site", "wall_s", "bytes", "MB", "gc_pct"))
for r in all_rows
    lp(@sprintf("%-22s %-38s %10.4f %12.3e %10.2f %8.2f", r.point, r.label, r.wall, r.bytes, r.mb, r.gc_pct))
end

open(joinpath(OUTDIR, "price_tensor_benchmark.csv"), "w") do io
    println(io, "point,label,wall_s,bytes,mb,gctime_s,gc_pct,gc_live_delta_mb,rss_delta_mb,rss_now_mb")
    for r in all_rows
        println(io, "$(r.point),$(r.label),$(r.wall),$(r.bytes),$(r.mb),$(r.gctime),$(r.gc_pct),$(r.gc_live_delta_mb),$(r.rss_delta_mb),$(r.rss_now_mb)")
    end
end
lp(">>> wrote ", joinpath(OUTDIR, "price_tensor_benchmark.csv"))
