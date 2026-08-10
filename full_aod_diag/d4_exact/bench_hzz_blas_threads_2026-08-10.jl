# H_ZZ gram kernel: BLAS-thread scaling microbenchmark (2026-08-10)
#
# Phase 0 of HANDOVER_CROSS_8H_UPPER_BOUND_RUNS_2026-08-10.md. The handover's main lever is a
# BLAS-thread gate fix; before paying for real cold inner solves to measure it, measure the KERNEL
# ITSELF, which is where the time provably goes (H_ZZ = 7.5 s of a 13.6 s Hessian call at K=3/3,
# `key_results/15_iters_vs_per_iteration_2026-08-10.txt`).
#
# This exercises the REAL production kernels (`zc_gram_blas_syrk!`/`zc_gram_blas_gemm!`/
# `zc_gram_threaded_packed!` from zc_gram_blas_candidates.jl) on a synthetic Phi of the REAL
# production shapes -- the kernel cost is shape-determined (dense O(W*nx^2) BLAS-3), not
# value-determined, so synthetic Phi/S measure the same thing real ones do. Correctness is NOT
# tested here (the D4 gates and the Delta_dual gate do that); this is a timing instrument only.
#
# Shapes: W=100_000, nx in {630 (diagonal K=3/3), 1770 (cross K=3/3)}.
# Usage: OPENBLAS_NUM_THREADS=1 julia --project=. -t 10 .../bench_hzz_blas_threads_2026-08-10.jl
const _D4E = @__DIR__
include(joinpath(_D4E, "zc_restriction_operator.jl"))       # ZCRestrictionOperator (type only, for the kernel signatures)
# zc_gram_blas_candidates.jl reads `resolve_cross_hessian_workers_default()` once, at load, only to
# initialise ZC_GRAM_THREADED_WORKERS_DEFAULT[] -- which this benchmark never consults (it passes
# `workers` explicitly). Stubbed rather than pulling in threaded_cross_hessian.jl's whole
# WinnerPairHessCtx include chain, which this timing instrument has no other use for.
resolve_cross_hessian_workers_default() = Threads.nthreads()
include(joinpath(_D4E, "zc_gram_blas_candidates.jl"))
using LinearAlgebra, Printf, Random
lp(xs...) = (println(xs...); flush(stdout))

const W  = 100_000
const NXS = (630, 1770)
const THREADS = (1, 2, 4, 8, 16, 32)
const REPS = 3

"Build a ZCRawWeightedWorkspace directly at (W,nx) without needing a ZCRestrictionOperator."
function synth_ws(W::Int, nx::Int, rng)
    Phi = randn(rng, W, nx)
    return ZCRawWeightedWorkspace(W, nx, Phi, similar(Phi), zeros(nx), zeros(nx, nx),
        randn(rng, nx), Vector{Task}(undef, Threads.nthreads()), Vector{Float64}(undef, W))
end

lp("="^100)
lp("H_ZZ gram kernel BLAS-thread scaling    W=", W, "  julia_threads=", Threads.nthreads(),
   "  ambient BLAS=", BLAS.get_num_threads())
lp("="^100)

rng = MersenneTwister(20260810)
results = Dict{Tuple{Symbol,Int,Int},Float64}()
for nx in NXS
    ws = synth_ws(W, nx, rng)
    S = abs.(randn(rng, W)) .+ 0.1          # S = Psi''(r) >= 0 by construction
    HZZ = zeros(nx, nx)
    gflop_syrk = 1e-9 * W * nx * (nx + 1)    # syrk: W*nx^2 MACs on the triangle = 2*W*nx*(nx+1)/2 flops
    lp("\n---- nx = ", nx, "   (syrk ", @sprintf("%.1f", gflop_syrk), " GFLOP/call) ----")
    # :threaded_packed is deliberately NOT measured here: it needs threaded_cross_hessian.jl's
    # `cross_hessian_chunk_ranges`, i.e. the whole WinnerPairHessCtx include chain this instrument
    # otherwise avoids. It is the handover's option 3, to be tried "only if 1 and 2 disappoint" --
    # and BLAS threading (option 1) does not disappoint, see the numbers below.
    for be in (:blas_syrk, :blas_gemm)
        f = be === :blas_syrk ? zc_gram_blas_syrk! :
            be === :blas_gemm ? zc_gram_blas_gemm! : nothing
        for nt in THREADS
            (be === :threaded_packed && nt != 1) && continue   # Julia-threaded: BLAS count irrelevant
            BLAS.set_num_threads(nt)
            if be === :threaded_packed
                zc_gram_threaded_packed!(HZZ, ws, S, Float64(W); workers = Threads.nthreads())
                t = minimum(@elapsed(zc_gram_threaded_packed!(HZZ, ws, S, Float64(W); workers = Threads.nthreads())) for _ in 1:REPS)
            else
                f(HZZ, ws, S, Float64(W))    # warm/compile
                t = minimum(@elapsed(f(HZZ, ws, S, Float64(W))) for _ in 1:REPS)
            end
            results[(be, nx, nt)] = t
            base = get(results, (be, nx, 1), t)
            @printf("  %-16s blas_threads=%3d   %8.4f s   %7.1f GFLOP/s   speedup_vs_1t = %5.2fx\n",
                    String(be), nt, t, gflop_syrk / t, base / t)
            flush(stdout)
        end
    end
end
BLAS.set_num_threads(1)

lp("\n", "="^100)
lp("SUMMARY (best per nx)")
lp("="^100)
for nx in NXS
    best = argmin(k -> results[k], [k for k in keys(results) if k[2] == nx])
    ref  = results[(:blas_syrk, nx, 1)]
    @printf("  nx=%5d : best = %-16s @ %2d BLAS threads  %8.4f s   (%.2fx vs syrk@1t = %.4f s)\n",
            nx, String(best[1]), best[3], results[best], ref / results[best], ref)
end
flush(stdout)
