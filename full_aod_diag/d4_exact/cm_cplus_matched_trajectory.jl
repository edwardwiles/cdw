# Part II.3, 2026-07-23: matched short real KNITRO outer-loop trajectories, Reference vs C+,
# same start point / same delta / same wall budget. NOT a resumed/production run -- writes to
# this worktree's own results/ directory, never touches production_runs/. Backend selected via
# `cm_gradient_backend` (a61af66, already wired into run_cm_upper_checkpointed's cb_G!).
#
# Start point: chain1's own unperturbed calibration w0 (CHAIN_PERTURB_SEED=0), matching
# cm_production_stage_runner.jl's own "calibration" mode exactly (W=80000, L=50, pseudorandom
# draws, seed 20260719, :orthonormal contrasts, nested_grid_sequence([10,20,50])[50] probs) --
# same problem instance and same starting point the real campaign used for delta=0.1, chain 1.
include(joinpath(@__DIR__, "draw_design.jl"))
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
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
using Printf, Random, Dates, Serialization, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))

const BACKEND = Symbol(ARGS[1])   # :reference | :cplus
const OUTROOT = ARGS[2]
const MAXTIME = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 100.0
BACKEND in (:reference, :cplus) || error("usage: cm_cplus_matched_trajectory.jl <reference|cplus> <outroot> [maxtime_real]")

const W = 80_000
const DELTA = 0.1
const L = 50
const DRAW_DESIGN = :pseudorandom
const DRAW_SEED = 20260719
const CM_CONTRASTS = :orthonormal

ckpt_dir = joinpath(OUTROOT, string(BACKEND))
mkpath(ckpt_dir)

snaps = nested_grid_sequence([10, 20, 50])
probs = snaps[L]

ctx0 = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true, draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED)
pe0 = build_pivot_elimination(ctx0)
D = ctx0.D
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
w0 = vcat(x_free_calib[1], pivot_reduce(log.(reshape(x_free_calib[2:end], D, D)), pe0))

lp(">>> [", BACKEND, "] matched trajectory start: g=", w0[1], " ||zfree||=", norm(w0[2:end]), " delta=", DELTA, " maxtime=", MAXTIME)

t0 = time()
res = run_cm_upper_checkpointed(w0;
    W = W, delta = DELTA, draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED,
    L = L, contrasts = CM_CONTRASTS, probs = probs,
    maxtime_real = MAXTIME, ckpt_dir = ckpt_dir, run_id = "matched_traj_$(BACKEND)",
    label = "matched_$(BACKEND)", checkpoint_interval_s = 20.0,
    cm_gradient_backend = BACKEND, verbose = true)
wall = time() - t0

lp(">>> [", BACKEND, "] trajectory finished: wall=", round(wall, digits = 1), "s")
lp(">>> [", BACKEND, "] result: ", res)

# Cold-verify the trajectory's own best_feasible (if any) using the OTHER backend's own
# archC_verified_state (backend-independent -- inner solve is orthogonal to outer gradient
# backend, see algebra trace) to confirm the reported incumbent is genuinely feasible/verified,
# not merely self-reported.
pcx = build_cm_production_context(ctx0, CS; L = L, contrasts = CM_CONTRASTS, probs = probs)
if res.best !== nothing
    xf_best = x_free_from_w(res.best.w, pe0)
    base_v, verify_v = archC_verified_state(xf_best, pcx.ctx_cm, pcx.cctx)
    lp(">>> [", BACKEND, "] independent cold re-verify of best: Delta_dual=", verify_v.Delta_dual,
       " (self-reported ", res.best.Delta, ") |diff|=", abs(verify_v.Delta_dual - res.best.Delta),
       " inner_status=", verify_v.inner_status)
end

summary = (backend = BACKEND, wall = wall, n_eval = res.n_eval, n_grad = res.n_grad,
           knitro_status = res.knitro_status, kappa = res.kappa,
           best_delta = res.best === nothing ? NaN : res.best.Delta,
           best_g = res.best === nothing ? NaN : res.best.w[1])
serialize(joinpath(OUTROOT, "summary_$(BACKEND).jls"), summary)
lp(">>> [", BACKEND, "] DONE. summary=", summary)
