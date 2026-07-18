# ============================================================================
# Equivalence tests for composite_gradient_fast.jl, mandatory before trusting
# any of its timing/wiring. Three claims checked:
#   1. h_mode=:adaptive, threaded=false reproduces composite_gradient_at
#      (the ORIGINAL, unmodified function) to machine precision -- the fast
#      file is a strict refactor in this mode, not a new formula.
#   2. h_mode=:adaptive, threaded=true reproduces the serial :adaptive result
#      to machine precision (each coordinate's work is independent; only the
#      SCHEDULING changes, not the math).
#   3. h_mode=:fixed and h_mode=:cached (first call) reproduce a
#      composite_gradient_at_fast(h_mode=:adaptive) call in which every
#      select_bandwidth happened to choose h0 -- checked indirectly via a
#      forced-h consistency check: a_block_fd_component itself is a pure
#      function of (cache,ctx,pe,w0,k,h), so calling it directly at h=h0 and
#      comparing against the :fixed-mode gradient is a stronger, more direct
#      equivalence check than re-deriving through select_bandwidth.
#   4. h_mode=:cached populates and reuses the bandwidth_cache correctly
#      (miss then hit), and the reused-h gradient exactly matches what
#      a_block_fd_component(h=cached_h) gives directly.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
using Printf

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2

const W_CAND = [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181,
    0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515,
    1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252,
    0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
xf = x_free_from_w(W_CAND)

all_pass = true

println("="^78); println("TEST 1: h_mode=:adaptive, threaded=false reproduces composite_gradient_at"); println("="^78)
g_orig, meta_orig = composite_gradient_at(xf, ctx, pe)
g_fast, meta_fast = composite_gradient_at_fast(xf, ctx, pe; h_mode = :adaptive, threaded = false)
maxdiff1 = maximum(abs.(g_orig .- g_fast))
h_diff1 = maximum(abs.(meta_orig.h_used .- meta_fast.h_used))
pass1 = maxdiff1 < 1e-13 && h_diff1 == 0.0
println("  max|g_orig - g_fast| = $maxdiff1   max|h_used diff| = $h_diff1   $(pass1 ? "PASS" : "FAIL")")
global all_pass &= pass1

println("\n" * "="^78); println("TEST 2: h_mode=:adaptive, threaded=true reproduces serial :adaptive"); println("="^78)
println("  JULIA_NUM_THREADS = ", Threads.nthreads(), " (re-run with a higher thread count to exercise real parallelism; correctness holds regardless)")
g_thr, meta_thr = composite_gradient_at_fast(xf, ctx, pe; h_mode = :adaptive, threaded = true)
maxdiff2 = maximum(abs.(g_fast .- g_thr))
pass2 = maxdiff2 < 1e-13
println("  max|g_serial - g_threaded| = $maxdiff2   $(pass2 ? "PASS" : "FAIL")")
global all_pass &= pass2

println("\n" * "="^78); println("TEST 3: h_mode=:fixed matches direct a_block_fd_component(h=h0) per coordinate"); println("="^78)
base = solve_base_state(xf, ctx)
cache = build_lfix_base_cache(xf, ctx, base)
z0 = log.(reshape(xf[2:end], D, D))
w0 = vcat(xf[1], pivot_reduce(z0, pe))
const H0 = 0.01
g_fixed, meta_fixed = composite_gradient_at_fast(xf, ctx, pe; h_mode = :fixed, h0 = H0, threaded = false)
direct_check = [a_block_fd_component(cache, ctx, pe, w0, k, H0) for k in 2:D2]
maxdiff3 = maximum(abs.(g_fixed[2:end] .- direct_check))
gamma_diff3 = abs(g_fixed[1] - gamma_component_analytic(cache, base, w0[1]))
pass3 = maxdiff3 < 1e-13 && gamma_diff3 < 1e-13 && all(meta_fixed.h_used[2:end] .== H0)
println("  max|g_fixed[A-block] - direct a_block_fd_component(h0)| = $maxdiff3   gamma diff=$gamma_diff3   all h_used==h0: $(all(meta_fixed.h_used[2:end] .== H0))   $(pass3 ? "PASS" : "FAIL")")
global all_pass &= pass3

println("\n" * "="^78); println("TEST 4: h_mode=:cached -- miss-then-hit correctness"); println("="^78)
bw_cache = Dict{Int,Float64}()
g_c1, meta_c1 = composite_gradient_at_fast(xf, ctx, pe; h_mode = :cached, bandwidth_cache = bw_cache, threaded = false)
n_miss_1 = count(!, meta_c1.cache_hits[2:end])
n_hit_1 = count(meta_c1.cache_hits[2:end])
println("  first call: $n_miss_1 misses (all $D2-1 expected), $n_hit_1 hits, bandwidth_cache now has $(length(bw_cache)) entries")
pass4a = n_miss_1 == D2 - 1 && n_hit_1 == 0 && length(bw_cache) == D2 - 1
# first call's chosen h must equal :adaptive's own choice (identical select_bandwidth call)
h_match_adaptive = maximum(abs.(g_c1[2:end] .- g_fast[2:end])) < 1e-13
println("  first-call gradient matches :adaptive exactly (same select_bandwidth call under the hood): $h_match_adaptive")
g_c2, meta_c2 = composite_gradient_at_fast(xf, ctx, pe; h_mode = :cached, bandwidth_cache = bw_cache, threaded = false)
n_miss_2 = count(!, meta_c2.cache_hits[2:end])
n_hit_2 = count(meta_c2.cache_hits[2:end])
direct_check2 = [a_block_fd_component(cache, ctx, pe, w0, k, bw_cache[k]) for k in 2:D2]
maxdiff4 = maximum(abs.(g_c2[2:end] .- direct_check2))
pass4b = n_miss_2 == 0 && n_hit_2 == D2 - 1 && maxdiff4 < 1e-13
println("  second call (same point): $n_miss_2 misses, $n_hit_2 hits (all cached, expected). max|g_cached - direct(h=cached_h)| = $maxdiff4")
pass4 = pass4a && h_match_adaptive && pass4b
println("  $(pass4 ? "PASS" : "FAIL")")
global all_pass &= pass4

println("\n" * "="^78)
println(all_pass ? "ALL COMPOSITE_GRADIENT_FAST EQUIVALENCE TESTS PASSED" : "SOME TESTS FAILED -- do not trust composite_gradient_at_fast for timing or KNITRO wiring")
println("="^78)
all_pass || error("test_composite_gradient_fast.jl: equivalence checks failed")
