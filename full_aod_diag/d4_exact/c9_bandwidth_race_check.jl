# Continuation 9: repro + fix-verification for the h_mode=:cached thread-safety bug found by
# Phase 5 (docs/fullA_D20_bandwidth_optimization_report.md sec 3F). Pre-fix, this reproduced a
# corrupted (under-populated) Dict in ~1.5% of trials (3/200) when populating bandwidth_cache
# via a threaded=true call. Post-fix (ReentrantLock around the dict read/write in
# composite_gradient_fast.jl's h_mode=:cached branch), this should be 0/N.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))

ctx = d4_exact_setup()
pe = build_pivot_elimination(ctx)
xf = ctx.θ0_up[ctx.free_idx]
D2 = ctx.D^2
expected_n = D2 - 1

N = 200
n_corrupted = 0
for trial in 1:N
    bandwidth_cache = Dict{Int,Float64}()
    g, meta = composite_gradient_at_fast(xf, ctx, pe; threaded = true, h_mode = :cached,
        bandwidth_cache = bandwidth_cache)
    if length(bandwidth_cache) != expected_n
        global n_corrupted += 1
        println("  trial ", trial, ": corrupted, length(dict)=", length(bandwidth_cache),
                " expected ", expected_n)
    end
end
println("n_corrupted = ", n_corrupted, " / ", N)
println(n_corrupted == 0 ? "RACE CHECK: PASS (fix verified)" : "RACE CHECK: FAIL (still racy)")
