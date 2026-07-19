# ============================================================================
# Continuation 9, Phase 2: thread-count sweep at D=20/W=80000, real data.
# Launched as 4 SEPARATE Julia processes (JULIA_NUM_THREADS=1,5,10,20 via the
# shell env var -- Julia does not support changing this at runtime), each
# paying its own one-time setup/JIT cost. BLAS threads fixed at 1 throughout
# per this investigation's own standing finding (OPENBLAS_NUM_THREADS=1 is
# correct on this machine, Continuation 8 Section 10 -- do NOT vary it here).
#
# Measures: (a) value callback (evaluate_fullA_fast, dense, warm-started,
# N=4), (b) one full composite_gradient_at_fast call (top3, threaded=true iff
# JULIA_NUM_THREADS>1, adaptive-h) if time permits per this script's own
# budget.
#
# Appends its one result row to a shared CSV
# (results/fullA_d4/<commit>/c9_w80k_microbenchmark/threadsweep.csv) so the
# 4 independent process runs accumulate into one table.
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))
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
using Statistics, Printf, Dates

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c9_w80k_microbenchmark")
mkpath(OUTDIR)
const CSVPATH = joinpath(OUTDIR, "threadsweep.csv")

nthreads = Threads.nthreads()
println("c9_w80k_threadsweep.jl starting at ", now(), "  JULIA_NUM_THREADS=", nthreads); flush(stdout)

t0 = time()
ctx = d20_real_setup(W = 80000)
t_setup = time() - t0
println("setup wall = ", round(t_setup, digits=2), "s"); flush(stdout)

D = ctx.D
xf_nat = ctx.θ0_up[ctx.free_idx]
pe = build_pivot_elimination(ctx)

# warm-up (untimed)
t0 = time()
for _ in 1:2
    evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :dense)
end
base0 = solve_base_state(xf_nat, ctx)
composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = (nthreads > 1), h_mode = :adaptive, multi_method = :top3)
t_warmup = time() - t0
println("warm-up wall = ", round(t_warmup, digits=2), "s"); flush(stdout)

# ---- value callback, N=4 ----
val_times = Float64[@elapsed evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :dense) for _ in 1:4]
val_median = median(val_times)
println("value callback (warm, dense) times = ", round.(val_times, digits=3), "  median=", round(val_median, digits=3), "s"); flush(stdout)

# ---- one full gradient, N=3 (threaded=true if nthreads>1, else serial -- an honest single-thread baseline) ----
grad_times = Float64[@elapsed composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = (nthreads > 1), h_mode = :adaptive, multi_method = :top3) for _ in 1:3]
grad_median = median(grad_times)
println("full gradient times = ", round.(grad_times, digits=3), "  median=", round(grad_median, digits=3), "s"); flush(stdout)

# append to shared CSV (create header if new)
newfile = !isfile(CSVPATH)
open(CSVPATH, "a") do io
    newfile && println(io, "julia_num_threads,openblas_num_threads,setup_wall_s,value_median_s,grad_median_s,grad_threaded")
    println(io, nthreads, ",1,", t_setup, ",", val_median, ",", grad_median, ",", (nthreads > 1))
end
println("Appended to ", CSVPATH)
println("c9_w80k_threadsweep.jl COMPLETE at ", now(), "  nthreads=", nthreads)
