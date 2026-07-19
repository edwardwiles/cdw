# ============================================================================
# Continuation 9, Phase 2 (W=800,000 half, resumed): thread-count sweep at
# D=20/W=800000, real data. Direct W=800k analogue of c9_w80k_threadsweep.jl
# -- same discipline (4 SEPARATE Julia processes, one per JULIA_NUM_THREADS
# value, since Julia does not support changing thread count at runtime;
# OPENBLAS_NUM_THREADS=1 fixed throughout per Continuation 8's standing
# finding, not re-tested here).
#
# Value callback (dense, warm) is the mandatory measurement (N=3, lighter
# than W80k's N=4 given ~10x per-call cost at this W). Full gradient is
# "if time permits" per this task's own brief -- included here at N=1 per
# thread count (not W80k's N=3) using h_mode=:adaptive for direct
# comparability with the W80k thread-sweep table (not :cached, which would
# shift the absolute numbers and defeat the cross-W comparison this sweep
# exists to support).
#
# VmHWM checked and logged at every step per the standing W=800,000 safety
# discipline; each process's own peak is independent (separate process ==
# separate address space) so no cross-process aggregation is needed here.
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

function vmhwm_kb()
    try
        for line in eachline("/proc/self/status")
            startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
        end
    catch
    end
    return -1
end
gb(kb) = round(kb / 1e6, digits = 2)

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c9_w800k_microbenchmark")
mkpath(OUTDIR)
const CSVPATH = joinpath(OUTDIR, "threadsweep.csv")

nthreads = Threads.nthreads()
println("c9_w800k_threadsweep.jl starting at ", now(), "  JULIA_NUM_THREADS=", nthreads); flush(stdout)

t0 = time()
ctx = d20_real_setup(W = 800000)
t_setup = time() - t0
println("setup wall = ", round(t_setup, digits=2), "s  VmHWM=", gb(vmhwm_kb()), "GB"); flush(stdout)

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
println("warm-up wall = ", round(t_warmup, digits=2), "s  VmHWM=", gb(vmhwm_kb()), "GB"); flush(stdout)

# ---- value callback, N=3 (dense, warm) ----
val_times = Float64[]
for _ in 1:3
    push!(val_times, @elapsed evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :dense))
end
val_median = median(val_times)
println("value callback (warm, dense) times = ", round.(val_times, digits=3), "  median=", round(val_median, digits=3), "s  VmHWM=", gb(vmhwm_kb()), "GB"); flush(stdout)

# ---- one full gradient, N=1 (threaded=true if nthreads>1, else serial -- honest single-thread baseline) ----
t_g0 = time()
gvec = composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = (nthreads > 1), h_mode = :adaptive, multi_method = :top3)
grad_wall = time() - t_g0
println("full gradient (N=1) wall = ", round(grad_wall, digits=3), "s  length=", length(gvec), "  VmHWM=", gb(vmhwm_kb()), "GB"); flush(stdout)

vmhwm_final = vmhwm_kb()
println("FINAL VmHWM = ", gb(vmhwm_final), "GB")

newfile = !isfile(CSVPATH)
open(CSVPATH, "a") do io
    newfile && println(io, "julia_num_threads,openblas_num_threads,setup_wall_s,value_median_s,grad_wall_s,grad_threaded,vmhwm_final_GB")
    println(io, nthreads, ",1,", t_setup, ",", val_median, ",", grad_wall, ",", (nthreads > 1), ",", gb(vmhwm_final))
end
println("Appended to ", CSVPATH)
println("c9_w800k_threadsweep.jl COMPLETE at ", now(), "  nthreads=", nthreads)
