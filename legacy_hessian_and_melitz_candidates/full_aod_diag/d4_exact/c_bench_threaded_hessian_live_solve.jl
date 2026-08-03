# Diagnostic branch diag/fullA-inner-blas-threading, closes the loop on report sec 3's caveat:
# does the standalone ~2.9x threaded-Hessian speedup (cm_hessian_threaded.jl) translate into a
# real COMPLETE inner-solve wall-time improvement when actually wired into a live KN_solve
# (cm_production_bundle_threaded.jl), or does it wash out against moment construction + genuine
# KNITRO-internal cost + the parts of the Hessian call that stay serial regardless (H_EE, H_EC/H_CC
# assembly, packing)? OPENBLAS_NUM_THREADS=1 throughout to isolate the Julia-threading effect.
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_hessian_threaded.jl"))
include(joinpath(@__DIR__, "cm_production_bundle_threaded.jl"))
using Printf, LinearAlgebra, Statistics

lp(x...) = (println(x...); flush(stdout))
BLAS.set_num_threads(1)
lp("Julia threads = ", Threads.nthreads(), "  BLAS threads (pinned) = ", BLAS.get_num_threads())

W = 80_000
ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true)
x_free_calib = ctx.θ0_up[ctx.free_idx]
snaps = nested_grid_sequence([10, 20, 50])
pcx = build_cm_production_context(ctx, CS; L = 50, contrasts = :anchored, probs = snaps[50])
tls = build_thread_local_scratch(pcx.cctx)

function med_time(f::Function; nrep::Int = 5, nwarm::Int = 1)
    for _ in 1:nwarm
        f()
    end
    ts = Vector{Float64}(undef, nrep)
    local val
    for i in 1:nrep
        t0 = time_ns()
        val = f()
        ts[i] = (time_ns() - t0) / 1e9
    end
    return median(ts), val
end

m_serial, (K1, base1) = med_time(() -> cm_production_value(x_free_calib, pcx))
@printf "SERIAL   (production default) live CM50 cold solve: med=%.4fs  K=%.6f  nStatus=%d\n" m_serial K1 base1.inner_status

m_threaded, (K2, base2) = med_time(() -> cm_production_value_threaded(x_free_calib, pcx, tls))
@printf "THREADED (experimental)      live CM50 cold solve: med=%.4fs  K=%.6f  nStatus=%d\n" m_threaded K2 base2.inner_status

@printf "\nEnd-to-end live-solve speedup: %.2fx (vs standalone-Hessian-only speedup ~2.9x at Julia=%d, report sec 3)\n" (m_serial/m_threaded) Threads.nthreads()
@printf "Correctness: |K_serial - K_threaded| = %.3e (expect ~0, float noise only)\n" abs(K1 - K2)

out_csv = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "threaded_hessian_live_solve.csv")
mkpath(dirname(out_csv))
newfile = !isfile(out_csv)
open(out_csv, "a") do io
    newfile && println(io, "julia_threads,serial_med_s,threaded_med_s,speedup,abs_K_diff")
    println(io, join((Threads.nthreads(), m_serial, m_threaded, m_serial/m_threaded, abs(K1-K2)), ","))
end
lp("DONE")
