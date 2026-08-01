# ============================================================================
# Claude Code task 2026-08-01, §14: matched A/B, FULL/reference arm, upper
# bound, delta=1. Literally `run_profile_checkpointed` (production driver),
# unmodified -- zero new code for this arm. Screens: production defaults
# (pairwise + fused winner-scan + general-range-safety-net all ON -- no
# driver-level kwarg exists to disable the safety net through
# run_profile_checkpointed's own `screened_eval` call, and screen porting is
# explicitly out of scope, task §18) -- documented asymmetry vs the
# screen-free profiled arm, see master doc.
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Printf

const W = 80_000
const DELTA = 1.0
const BUDGET = 1800.0   # 30 minutes

ctx_probe = d20_real_setup(W = W, find_smallest = true, δ = DELTA, destination_sample = :exclude_row)
pe_probe = build_pivot_elimination(ctx_probe)
D = ctx_probe.D
Aod_theta_natural = ctx_probe.θ0_up[ctx_probe.Aod_offset+1:ctx_probe.Aod_offset+ctx_probe.D*ctx_probe.D_dest]
z0 = log.(Aod_theta_natural)
zfree0 = pivot_reduce(reshape(z0, ctx_probe.D, ctx_probe.D_dest), pe_probe)
gp0 = ctx_probe.θ0_up[3+D]

CKPT_DIR = joinpath(D4X_ROOT, "results", "profiled_ab_2026-08-01", "full_upper")
mkpath(CKPT_DIR)

println("="^90); println("FULL/REFERENCE ARM: upper bound, delta=1, W=$W, budget=$(BUDGET)s"); println("="^90); flush(stdout)
t0 = time()
res = run_profile_checkpointed("full_upper", gp0, true, zfree0;
    maxtime_real = BUDGET, W_in = W, delta_in = DELTA, draw_seed_in = 20260719,
    ckpt_dir = CKPT_DIR, checkpoint_interval_s = 60.0)
wall = time() - t0

println("="^90)
@printf("RESULT full_upper: wall=%.1fs n_eval=%d n_grad_calls=%d\n", wall, res.n_eval, res.n_grad_calls)
println("best=", res.best)
println("screen_counts=", res.screen_counts)
println("="^90)

outpath = joinpath(D4X_ROOT, "FULL_VS_PROFILED_OUTER_AB_UPPER_DELTA1_2026-08-01_FULL_TRACE.csv")
using CSV, DataFrames
CSV.write(outpath, DataFrame(res.trace))
println("Wrote $outpath")
