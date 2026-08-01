# ============================================================================
# Claude Code task 2026-08-01, follow-up: matched A/B at the GT=5% waypoint
# reached via continuation (continuation_gt_targets_2026-08-01.jl). FULL arm:
# unmodified run_profile_checkpointed, seeded from the continuation's own
# zfree/gp (same coordinate system the continuation itself used).
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Printf, CSV, DataFrames, Serialization

const W = 80_000
const DELTA = 1.0
const MAXIT = parse(Int, get(ENV, "AB_MAXIT", "60"))
const SAFETY_BUDGET = 3600.0

waypoint = deserialize(joinpath(D4X_ROOT, "results", "profiled_ab_2026-08-01", "continuation_gt", "waypoint_GT5.jls"))
println("Loaded waypoint: gp=", waypoint.gp, "  kappa=", waypoint.kappa); flush(stdout)

CKPT_DIR = joinpath(D4X_ROOT, "results", "profiled_ab_2026-08-01", "full_fixediter_gt5")
mkpath(CKPT_DIR)

println("="^90); println("FULL/REFERENCE ARM (fixed-iteration, GT=5% waypoint): maxit=$MAXIT, gp=$(waypoint.gp)"); println("="^90); flush(stdout)
t0 = time()
res = run_profile_checkpointed("full_fixediter_gt5", waypoint.gp, true, waypoint.zfree;
    maxtime_real = SAFETY_BUDGET, W_in = W, delta_in = DELTA, draw_seed_in = 20260719,
    ckpt_dir = CKPT_DIR, checkpoint_interval_s = 60.0, maxit_override = MAXIT)
wall = time() - t0

println("="^90)
@printf("RESULT full_fixediter_gt5: wall=%.1fs n_eval=%d n_grad_calls=%d native_outer_iters=%d\n",
    wall, res.n_eval, res.n_grad_calls, res.native_outer_diag.n_iters)
println("best=", res.best)
println("="^90)

outpath = joinpath(D4X_ROOT, "FULL_VS_PROFILED_OUTER_AB_FIXEDITER_2026-08-01_gt5_FULL_TRACE.csv")
CSV.write(outpath, DataFrame(res.trace))
println("Wrote $outpath")
