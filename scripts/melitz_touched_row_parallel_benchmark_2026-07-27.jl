# User-motivated follow-up (2026-07-XX): the SERIAL touched-row backend showed ~5.7x less
# memory traffic but only 1.02x wall-clock gain over the sorted backend -- meaning a single
# thread's wall time is not bottlenecked on memory bandwidth. But the actual PRODUCTION
# configuration uses 20 threads, all simultaneously contending for shared DRAM bandwidth --
# a fundamentally different regime a single-thread test cannot speak to. This script builds
# and validates a parallel touched-row variant, then measures it against the existing
# 20-thread sorted-parallel backend at real D=20/W=80,000 to answer the real question: does
# the memory-traffic reduction matter once 20 cores are contending for bandwidth?

using Pkg
Pkg.activate(dirname(@__DIR__))
using Random, DelimitedFiles, Printf, LinearAlgebra
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

println("Threads.nthreads() = ", Threads.nthreads(), "  BLAS threads = ", LinearAlgebra.BLAS.get_num_threads())
@assert Threads.nthreads() > 1 "run with -t 20 (or similar) -- this script tests the PARALLEL backends specifically"

real_dir = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
focal = findfirst(==("fra"), countries)
observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
z_draws = pareto_draws(80_000, calib.D, calib.theta_star; seed=calib.seed)
p20, eq20, cf20, ctx20 = melitz_calibration_outer_ctx(calib; z_draws=z_draws, moment_backend=:sorted_tail_serial)
theta0 = melitz_reduce_theta(p20, ctx20)
n = length(theta0)
op20 = build_melitz_moment_operator(ctx20.sorted_tail_ctx, ctx20.moment_layout)
obj = build_melitz_cc_bundle(op20, ctx20; mode=:delta, U=z_draws,
    outer_constr_index=ctx20.moment_layout.num_moments + 1,
    lower_limit=-10.0,
    inner_loop_opt=ctx20.inner_loop_opt, outer_loop_opt=ctx20.outer_loop_opt,
    hessian_backend=:structured_parallel)
r0 = evaluate_melitz_delta(theta0, ctx20, obj; cold=true, store_G=false)
x0 = r0.dual_x
println("Delta0=", r0.Delta, " nStatus=", r0.nStatus, " n_theta=", n)

gsorted_p = make_melitz_gradient_delta_direct_sorted_parallel(1e-4)
gtouched_p = make_melitz_gradient_delta_direct_touched_row_parallel(1e-4)

println("\n== Correctness: parallel touched-row vs parallel sorted, real D=20 ==")
g1 = zeros(n); g2 = zeros(n)
gsorted_p(g1, theta0, ctx20, obj, x0)
gtouched_p(g2, theta0, ctx20, obj, x0)
maxrel = maximum(abs.(g1 .- g2) ./ max.(abs.(g1), 1.0))
println("max relative diff: ", maxrel)
@assert maxrel < 1e-8 "parallel touched-row mismatch vs parallel sorted!"

# second call, catch any per-thread stale-generation-stamp bug (each thread's own counter must
# also be monotonic across calls, not just across coordinates within one call)
gtouched_p(g2, theta0, ctx20, obj, x0)
maxrel2 = maximum(abs.(g1 .- g2) ./ max.(abs.(g1), 1.0))
println("second call max relative diff: ", maxrel2)
@assert maxrel2 < 1e-8 "second-call mismatch -- stale per-thread generation stamp bug!"

println("\n== 20-thread wall-clock comparison (warm, post-JIT) ==")
gsorted_p(g1, theta0, ctx20, obj, x0)   # warmup
gtouched_p(g2, theta0, ctx20, obj, x0)  # warmup

n_reps = 5
t_sorted = Float64[]
t_touched = Float64[]
for _ in 1:n_reps
    push!(t_sorted, @elapsed gsorted_p(g1, theta0, ctx20, obj, x0))
    push!(t_touched, @elapsed gtouched_p(g2, theta0, ctx20, obj, x0))
end
b_sorted = @allocated gsorted_p(g1, theta0, ctx20, obj, x0)
b_touched = @allocated gtouched_p(g2, theta0, ctx20, obj, x0)

println(@sprintf("sorted_parallel:   mean=%.4fs  (min=%.4f, max=%.4f)  %d bytes", sum(t_sorted)/n_reps, minimum(t_sorted), maximum(t_sorted), b_sorted))
println(@sprintf("touched_parallel:  mean=%.4fs  (min=%.4f, max=%.4f)  %d bytes", sum(t_touched)/n_reps, minimum(t_touched), maximum(t_touched), b_touched))
println(@sprintf("speedup (sorted/touched): %.3fx", (sum(t_sorted)/n_reps) / (sum(t_touched)/n_reps)))

outfile = joinpath(dirname(@__DIR__), "docs", "key_results", "melitz_touched_row_parallel_benchmark_2026-07-27.csv")
open(outfile, "w") do io
    println(io, "backend,rep,seconds")
    for (i, t) in enumerate(t_sorted)
        println(io, "sorted_parallel,", i, ",", t)
    end
    for (i, t) in enumerate(t_touched)
        println(io, "touched_row_parallel,", i, ",", t)
    end
end
println("\nWrote ", outfile)
