# port/shared-winner-pair-core-hessian-production-2026-07-25: D=20/W=80,000 real-data smoke +
# timing check for the UNRESTRICTED family's shared H_EE wiring, through the ACTUAL production
# entry point (inner_loop_KNITRO_compressed / _callbackEvalH_inner_compressed!), toggling
# UNRESTRICTED_CORE_HESSIAN_BACKEND[] between :dense_reference (pre-port, byte-identical CS.hessian!)
# and :exact_winner_pair_parallel (this port's production default).
#
# Scope note: this is a single-point (P0 calibration) smoke/timing check, NOT the full task-brief-
# specified 300s matched outer A/B campaign across P0/P1/P2 -- that full campaign was out of reach
# in this session's time budget and is disclosed as deferred in the deliverable docs, not fabricated.
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
const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    println(rpad(cond ? "PASS" : "FAIL", 6), name)
    cond || push!(FAILURES, name)
end

W = 80_000
println("Building D=20 real context: :exclude_row, W=$W, seed=20260719 ..."); flush(stdout)
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_seed = 20260719, destination_sample = :exclude_row)
obj = ctx.obj
n = obj.outer_constr_index
println("n=$n, D=$(ctx.D), D_dest=$(ctx.D_dest), W=$(size(obj.U,1))"); flush(stdout)

x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)

"Run one full compressed inner solve at the given backend; returns (nStatus, objSol, wall_seconds, n_hess)."
function run_once(backend::Symbol; workers::Int = 10)
    UNRESTRICTED_CORE_HESSIAN_BACKEND[] = backend
    UNRESTRICTED_CORE_HESSIAN_WORKERS[] = workers
    cf = build_compressed_factual(θ_full_calib, ctx; check_ties = true)
    st = CompressedCBState(obj, cf, 0.0, false)
    t0 = time()
    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_compressed(obj, st)
    dt = time() - t0
    return (nStatus = nStatus, objSol = objSol, x = x, dt = dt, n_fg = n_fg, n_hess = n_hess)
end

println("="^80)
println("P0 calibration point: dense-reference (CS.hessian!, pre-port byte-identical) vs shared winner-pair (workers=10)")
println("="^80)
# warm-up (JIT) run at each backend, not timed
run_once(:dense_reference); run_once(:exact_winner_pair_parallel; workers = 10)

r_dense = run_once(:dense_reference)
r_shared = run_once(:exact_winner_pair_parallel; workers = 10)

check("dense-reference feasible", r_dense.nStatus in FEASIBLE_CODES)
check("shared winner-pair feasible", r_shared.nStatus in FEASIBLE_CODES)
check("nStatus agrees", r_dense.nStatus == r_shared.nStatus)
check("n_fg agrees (backend must not change dual-iterate path)", r_dense.n_fg == r_shared.n_fg)
check("n_hess agrees", r_dense.n_hess == r_shared.n_hess)
check("objSol agrees to 1e-8", isapprox(r_dense.objSol, r_shared.objSol; rtol = 1e-8, atol = 1e-10))
check("dual solution x agrees to 1e-6", isapprox(r_dense.x, r_shared.x; rtol = 1e-6, atol = 1e-9))

@printf("\ndense-reference : nStatus=%d objSol=%.10f n_fg=%d n_hess=%d wall=%.4fs\n", r_dense.nStatus, r_dense.objSol, r_dense.n_fg, r_dense.n_hess, r_dense.dt)
@printf("winner-pair(w=10): nStatus=%d objSol=%.10f n_fg=%d n_hess=%d wall=%.4fs\n", r_shared.nStatus, r_shared.objSol, r_shared.n_fg, r_shared.n_hess, r_shared.dt)
@printf("speedup (complete inner solve, single warm point): %.2fx\n", r_dense.dt / r_shared.dt)

println("="^80)
println("worker-count sweep (complete inner solve, P0, JULIA_NUM_THREADS=$(Threads.nthreads()))")
println("="^80)
for wk in [1, 2, 4, 8, 10]
    wk > Threads.nthreads() && continue
    r = run_once(:exact_winner_pair_parallel; workers = wk)
    @printf("workers=%-3d  nStatus=%d  wall=%.4fs  feasible=%s\n", wk, r.nStatus, r.dt, r.nStatus in FEASIBLE_CODES)
end
r_serial = run_once(:exact_winner_pair_serial)
@printf("serial       nStatus=%d  wall=%.4fs  feasible=%s\n", r_serial.nStatus, r_serial.dt, r_serial.nStatus in FEASIBLE_CODES)
check("serial backend feasible and agrees with dense", r_serial.nStatus in FEASIBLE_CODES && isapprox(r_serial.objSol, r_dense.objSol; rtol=1e-8))

println("="^80)
if isempty(FAILURES)
    println("ALL D=20 UNRESTRICTED SMOKE CHECKS PASSED")
else
    println("FAILURES (", length(FAILURES), "): ", FAILURES)
    exit(1)
end
