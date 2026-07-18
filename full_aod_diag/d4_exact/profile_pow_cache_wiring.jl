# ============================================================================
# Realized-gain profile of the pow-cache wiring (enable_pow_cache!) on the LIVE
# oracle path. Two measurements, both warmed, both with allocation + GC-time +
# call-count evidence (not wall-clock alone -- per the investigation's
# "verify before causal claims" discipline):
#
#   (A) MOMENT-BUILD COMPONENT in isolation: time the actual obj.moments! FIELD
#       call (what every live consumer invokes), N reps, cache-off ctx vs
#       cache-on ctx. Reports median wall, mean allocations, mean GC time.
#
#   (B) LIVE evaluate_fullA_fast end-to-end: uses the file's own @prof
#       "inner_moment_build" timer (the ONE moments! build per evaluation) plus
#       the total timer, and n_fg_calls/n_hess_calls, to show (i) the realized
#       moment-build gain inside a real evaluation, and (ii) that the inner
#       KNITRO dual-solve path is BIT-FOR-BIT unchanged (identical callback
#       counts) -- i.e. the wall-clock delta is the moment build, not a changed
#       solve trajectory.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "moments_fast.jl"))
using Printf, Statistics

const NTHREADS = Threads.nthreads()

ctx_off = d4_exact_setup(find_smallest = true)          # original moments! (EK_moments_gammanorm_directgp!)
ctx_on  = d4_exact_setup(find_smallest = true)
pow_cache = enable_pow_cache!(ctx_on)                    # fast moments! via MuSigmaPowCache

pe = build_pivot_elimination(ctx_off)
D = ctx_off.D
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
w_upper = [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181, 0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515, 1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252, 0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]
xf = x_free_from_w(w_upper)
θ_full = CS.reconstruct_full(xf, ctx_off.m)
Wn = size(ctx_off.U, 1); d = ctx_off.obj.d

# ============================================================================
# (A) Moment-build component in isolation -- the actual obj.moments! field call
# ============================================================================
K = zeros(Wn); G = zeros(Wn, d)
# warm-up (JIT) both field closures
ctx_off.obj.moments!(K, G, θ_full, ctx_off.obj.U, ctx_off.obj)
ctx_on.obj.moments!(K, G, θ_full, ctx_on.obj.U, ctx_on.obj)

function bench_field(obj, θ, K, G; N = 200)
    times = Float64[]; allocs = Int[]; gcs = Float64[]
    for _ in 1:N
        g0 = Base.gc_num(); t0 = time_ns()
        obj.moments!(K, G, θ, obj.U, obj)
        t1 = time_ns(); gd = Base.GC_Diff(Base.gc_num(), g0)
        push!(times, (t1 - t0)/1e9); push!(allocs, gd.allocd); push!(gcs, gd.total_time/1e9)
    end
    (median_ms = median(times)*1e3, min_ms = minimum(times)*1e3,
     mean_alloc_kb = mean(allocs)/1024, mean_gc_ms = mean(gcs)*1e3)
end

N_field = 300
a_off = bench_field(ctx_off.obj, θ_full, K, G; N = N_field)
a_on  = bench_field(ctx_on.obj,  θ_full, K, G; N = N_field)

println("="^90)
println("(A) MOMENT-BUILD COMPONENT (obj.moments! field call), D=4 W=$Wn, NTHREADS=$NTHREADS, N=$N_field reps, warmed")
println("="^90)
@printf("  %-18s  median=%.4f ms   min=%.4f ms   alloc=%.1f KB/call   gc=%.4f ms/call\n",
        "cache OFF (orig)", a_off.median_ms, a_off.min_ms, a_off.mean_alloc_kb, a_off.mean_gc_ms)
@printf("  %-18s  median=%.4f ms   min=%.4f ms   alloc=%.1f KB/call   gc=%.4f ms/call\n",
        "cache ON  (fast)", a_on.median_ms, a_on.min_ms, a_on.mean_alloc_kb, a_on.mean_gc_ms)
@printf("  --> moment-build speedup (median): %.2fx   |  alloc reduction: %.1f KB/call (%.1f%%)  |  gc reduction: %.4f ms/call\n",
        a_off.median_ms / a_on.median_ms, a_off.mean_alloc_kb - a_on.mean_alloc_kb,
        100*(1 - a_on.mean_alloc_kb/a_off.mean_alloc_kb), a_off.mean_gc_ms - a_on.mean_gc_ms)

# ============================================================================
# (B) Live evaluate_fullA_fast: inner_moment_build timer + total + callback counts
# ============================================================================
function bench_live(ctx; N = 40)
    # warm-up
    evaluate_fullA_fast(xf, ctx; cache = nothing, warm = false)
    prof_reset!()
    n_fg = Int[]; n_hess = Int[]; totals = Float64[]
    for _ in 1:N
        r, meta = evaluate_fullA_fast(xf, ctx; cache = nothing, warm = false)
        push!(n_fg, meta.n_fg_calls); push!(n_hess, meta.n_hess_calls)
        push!(totals, r.elapsed.total)
    end
    # pull the @prof inner_moment_build stats collected across the N reps
    imb_t = PROF_TIMES["inner_moment_build"]; imb_a = PROF_ALLOCS["inner_moment_build"]; imb_g = PROF_GCTIME["inner_moment_build"]
    (imb_median_ms = median(imb_t)*1e3, imb_alloc_kb = mean(imb_a)/1024, imb_gc_ms = mean(imb_g)*1e3,
     total_median_ms = median(totals)*1e3, n_fg = n_fg, n_hess = n_hess, imb_n = length(imb_t))
end

N_live = 50
b_off = bench_live(ctx_off; N = N_live)
b_on  = bench_live(ctx_on;  N = N_live)

println()
println("="^90)
println("(B) LIVE evaluate_fullA_fast, D=4 W=$Wn, NTHREADS=$NTHREADS, N=$N_live reps, warmed, warm=false (deterministic)")
println("="^90)
@printf("  %-18s  inner_moment_build: median=%.4f ms  alloc=%.1f KB  gc=%.4f ms  |  eval total median=%.4f ms  |  n_fg=%s  n_hess=%s\n",
        "cache OFF", b_off.imb_median_ms, b_off.imb_alloc_kb, b_off.imb_gc_ms, b_off.total_median_ms,
        string(sort(unique(b_off.n_fg))), string(sort(unique(b_off.n_hess))))
@printf("  %-18s  inner_moment_build: median=%.4f ms  alloc=%.1f KB  gc=%.4f ms  |  eval total median=%.4f ms  |  n_fg=%s  n_hess=%s\n",
        "cache ON", b_on.imb_median_ms, b_on.imb_alloc_kb, b_on.imb_gc_ms, b_on.total_median_ms,
        string(sort(unique(b_on.n_fg))), string(sort(unique(b_on.n_hess))))
@printf("  --> live inner_moment_build speedup: %.2fx   alloc reduction: %.1f KB   |  eval total: %.4f -> %.4f ms (%.2fx)\n",
        b_off.imb_median_ms / b_on.imb_median_ms, b_off.imb_alloc_kb - b_on.imb_alloc_kb,
        b_off.total_median_ms, b_on.total_median_ms, b_off.total_median_ms / b_on.total_median_ms)

callback_identical = sort(unique(b_off.n_fg)) == sort(unique(b_on.n_fg)) &&
                     sort(unique(b_off.n_hess)) == sort(unique(b_on.n_hess))
println()
println(callback_identical ?
    "INNER-SOLVE PATH UNCHANGED: identical n_fg/n_hess callback counts cache-off vs cache-on -> wall-clock delta is the moment build, NOT a changed solve trajectory." :
    "WARNING: callback counts DIFFER cache-off vs cache-on -- investigate before attributing any wall-clock delta to the cache.")

# ---- machine-readable ----
const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "pow_cache_wiring")
mkpath(OUTDIR)
open(joinpath(OUTDIR, "pow_cache_wiring_profile.csv"), "w") do io
    println(io, "component,cache,median_ms,alloc_kb,gc_ms,extra")
    println(io, "moment_build_field,off,$(a_off.median_ms),$(a_off.mean_alloc_kb),$(a_off.mean_gc_ms),")
    println(io, "moment_build_field,on,$(a_on.median_ms),$(a_on.mean_alloc_kb),$(a_on.mean_gc_ms),")
    println(io, "live_inner_moment_build,off,$(b_off.imb_median_ms),$(b_off.imb_alloc_kb),$(b_off.imb_gc_ms),n_fg=$(sort(unique(b_off.n_fg)))")
    println(io, "live_inner_moment_build,on,$(b_on.imb_median_ms),$(b_on.imb_alloc_kb),$(b_on.imb_gc_ms),n_fg=$(sort(unique(b_on.n_fg)))")
    println(io, "live_eval_total,off,$(b_off.total_median_ms),,,")
    println(io, "live_eval_total,on,$(b_on.total_median_ms),,,")
end
println("\nWrote ", joinpath(OUTDIR, "pow_cache_wiring_profile.csv"))
