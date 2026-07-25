# 2026-07-25 continuation, task §5: confirm the worker default under a REAL 20-thread
# environment (session 1's own worker sweep only had JULIA_NUM_THREADS=10 available, and only at
# an easy P0 point). D=20/W=80,000/:exclude_row/seed=20260719, JULIA_NUM_THREADS=20,
# OPENBLAS_NUM_THREADS=1, through the real production entry point (inner_loop_KNITRO_compressed),
# at P1 (feasible, near delta=1 via a zfree perturbation) and P2 (a larger, harder perturbation),
# comparing serial vs parallel workers in {8, 10, 20}, multiple warmed repetitions.
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "draw_design.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "winner_certificate.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
using Printf, LinearAlgebra, Random, Statistics

const FEASIBLE_CODES = (0, -100, -101, -103)
lp(xs...) = (println(xs...); flush(stdout))

W = 80_000
lp("Building D=20 real context: :exclude_row, W=$W, seed=20260719, JULIA_NUM_THREADS=", Threads.nthreads())
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_seed = 20260719, destination_sample = :exclude_row)
obj = ctx.obj
pe0 = build_pivot_elimination(ctx)
x_free_calib = ctx.θ0_up[ctx.free_idx]

function perturbed_x_free(scale::Float64, seed::Int)
    Aod_block_raw = ctx.θ0_up[ctx.Aod_offset+1 : ctx.Aod_offset + ctx.D*ctx.D_dest]
    zfree_calib = pivot_reduce(reshape(log.(Aod_block_raw), ctx.D, ctx.D_dest), pe0)
    Random.seed!(seed)
    zfree = zfree_calib .+ scale .* randn(length(zfree_calib))
    logA = pivot_expand(zfree, pe0)
    Aod_lvl = exp.(logA)
    A_full = reshape(Aod_lvl, ctx.D, ctx.D_dest)
    x_free = copy(ctx.θ0_up[ctx.free_idx])
    for o in 1:ctx.D, s in 1:ctx.D_dest
        fp = ctx.Aod_free_pos[o, s]
        fp > 0 && (x_free[fp] = A_full[o, s])
    end
    return x_free
end

function run_once(x_free::AbstractVector, backend::Symbol; workers::Int = 10)
    UNRESTRICTED_CORE_HESSIAN_BACKEND[] = backend
    UNRESTRICTED_CORE_HESSIAN_WORKERS[] = workers
    θ_full = CS.reconstruct_full(x_free, ctx.m)
    cf = build_compressed_factual(θ_full, ctx; check_ties = true)
    st = CompressedCBState(obj, cf, 0.0, false)
    t0 = time()
    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_compressed(obj, st)
    dt = time() - t0
    return (nStatus = nStatus, objSol = objSol, dt = dt, n_fg = n_fg, n_hess = n_hess)
end

points = [("P1_near_delta1", perturbed_x_free(0.05, 20260719)),
          ("P2_hard", perturbed_x_free(0.20, 20260720))]

for (plabel, xf) in points
    lp("="^90)
    lp(plabel, ": worker sweep (serial, 1, 2, 4, 8, 10, 20), JULIA_NUM_THREADS=", Threads.nthreads())
    lp("="^90)
    # feasibility check + warm-up (not timed)
    r0 = run_once(xf, :exact_winner_pair_parallel; workers = 10)
    lp(">>> $plabel feasibility check: nStatus=", r0.nStatus, " feasible=", r0.nStatus in FEASIBLE_CODES)
    if !(r0.nStatus in FEASIBLE_CODES)
        lp(">>> WARNING: $plabel infeasible at this perturbation scale -- results below still reported but not a genuine A/B at a feasible point")
    end
    run_once(xf, :dense_reference)   # warm-up

    n_rep = 3
    r_dense = [run_once(xf, :dense_reference) for _ in 1:n_rep]
    dt_dense = minimum(r.dt for r in r_dense)
    @printf("dense-reference       : nStatus=%d n_hess=%d  min_wall=%.4fs  (reps: %s)\n",
        r_dense[1].nStatus, r_dense[1].n_hess, dt_dense, join([@sprintf("%.3f", r.dt) for r in r_dense], ","))

    r_serial = [run_once(xf, :exact_winner_pair_serial) for _ in 1:n_rep]
    dt_serial = minimum(r.dt for r in r_serial)
    @printf("winner-pair serial     : nStatus=%d n_hess=%d  min_wall=%.4fs  (reps: %s)  speedup=%.2fx\n",
        r_serial[1].nStatus, r_serial[1].n_hess, dt_serial, join([@sprintf("%.3f", r.dt) for r in r_serial], ","), dt_dense/dt_serial)

    for wk in [1, 2, 4, 8, 10, 20]
        wk > Threads.nthreads() && continue
        rs = [run_once(xf, :exact_winner_pair_parallel; workers = wk) for _ in 1:n_rep]
        dtw = minimum(r.dt for r in rs)
        @printf("winner-pair workers=%-3d: nStatus=%d n_hess=%d  min_wall=%.4fs  (reps: %s)  speedup=%.2fx\n",
            wk, rs[1].nStatus, rs[1].n_hess, dtw, join([@sprintf("%.3f", r.dt) for r in rs], ","), dt_dense/dtw)
    end
end
lp("="^90)
lp("DONE")
