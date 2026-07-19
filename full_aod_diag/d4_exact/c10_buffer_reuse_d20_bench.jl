# ============================================================================
# Continuation 10, Section 4: correctness + D=20/W=80000 benchmark of the
# allocation-reuse composite-gradient variant (lfix_buffer_reuse.jl) against
# the CURRENT production composite_gradient_at_fast (h_mode=:cached,
# threaded=true -- the exact config c9_phase8_d20_pilot.jl uses).
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "lfix_buffer_reuse.jl"))
using Printf, Statistics

println("nthreads=", Threads.nthreads())
ctx = d20_real_setup(W = 80000, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
println("D=", D, " D2=", D2, " W=", ctx.W)

x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2]
z0 = log.(Aod_theta_natural)
zfree0 = pivot_reduce(reshape(z0, D, D), pe)
gp0 = ctx.θ0_up[3+D]
xf = x_free_from_w(vcat(gp0, zfree0))

base = solve_base_state(xf, ctx)

# ---- Correctness: original vs buffered, h_mode=:cached, both threaded and serial ----
println("\n=== Correctness ===")
cache1 = Dict{Int,Float64}()
g_orig, meta_orig = composite_gradient_at_fast(xf, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = cache1)
cache2 = Dict{Int,Float64}()
g_buf, meta_buf = composite_gradient_at_fast_buffered(xf, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = cache2)
maxdiff = maximum(abs.(g_orig .- g_buf))
@printf("threaded, cold-cache: max|g_orig - g_buf| = %.3e  (n=%d)\n", maxdiff, length(g_orig))
@assert maxdiff < 1e-10 "CORRECTNESS FAILURE: buffered gradient disagrees with original beyond floating-point noise"

# second call (warm cache, both caches already populated identically since same coord order/base)
g_orig2, _ = composite_gradient_at_fast(xf, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = cache1)
g_buf2, _ = composite_gradient_at_fast_buffered(xf, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = cache2)
maxdiff2 = maximum(abs.(g_orig2 .- g_buf2))
@printf("threaded, warm-cache: max|g_orig2 - g_buf2| = %.3e\n", maxdiff2)
@assert maxdiff2 < 1e-10 "CORRECTNESS FAILURE (warm cache)"

# serial too
cache3 = Dict{Int,Float64}(); cache4 = Dict{Int,Float64}()
g_orig_s, _ = composite_gradient_at_fast(xf, ctx, pe; base = base, threaded = false, h_mode = :cached, bandwidth_cache = cache3)
g_buf_s, _ = composite_gradient_at_fast_buffered(xf, ctx, pe; base = base, threaded = false, h_mode = :cached, bandwidth_cache = cache4)
maxdiff_s = maximum(abs.(g_orig_s .- g_buf_s))
@printf("serial, cold-cache:   max|g_orig_s - g_buf_s| = %.3e\n", maxdiff_s)
@assert maxdiff_s < 1e-10 "CORRECTNESS FAILURE (serial)"
println("CORRECTNESS: PASS (all 3 checks < 1e-10)")

# ---- Benchmark: warm-cache steady-state gradient call (the realistic in-loop case) ----
# Warm both caches fully first (mimics steady-state where every coordinate's bandwidth is
# already cached, the common case after the first few KNITRO outer iterations).
cache_warm_orig = Dict{Int,Float64}()
composite_gradient_at_fast(xf, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = cache_warm_orig)
cache_warm_buf = Dict{Int,Float64}()
composite_gradient_at_fast_buffered(xf, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = cache_warm_buf)

function timeit(f, N)
    ts = Float64[]
    allocs = Int[]
    for i in 1:N
        stats = @timed f()
        push!(ts, stats.time)
        push!(allocs, stats.bytes)
    end
    return ts, allocs
end

N = 12
println("\n=== Benchmark: warm-cache gradient call, threaded=true, N=$N reps ===")
ts_orig, allocs_orig = timeit(() -> composite_gradient_at_fast(xf, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = cache_warm_orig), N)
ts_buf, allocs_buf = timeit(() -> composite_gradient_at_fast_buffered(xf, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = cache_warm_buf), N)

@printf("original : median=%.4fs mean=%.4fs  median_alloc=%.2f MB\n", median(ts_orig), mean(ts_orig), median(allocs_orig)/1e6)
@printf("buffered : median=%.4fs mean=%.4fs  median_alloc=%.2f MB\n", median(ts_buf), mean(ts_buf), median(allocs_buf)/1e6)
@printf("speedup (median): %.3fx\n", median(ts_orig)/median(ts_buf))
@printf("allocation reduction: %.3fx\n", median(allocs_orig)/median(allocs_buf))

println("\nDONE")
