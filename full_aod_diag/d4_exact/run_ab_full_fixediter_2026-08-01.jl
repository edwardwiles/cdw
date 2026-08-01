# ============================================================================
# Claude Code task 2026-08-01, follow-up (live user request): matched A/B on
# a FIXED OUTER-ITERATION BUDGET (not wall-clock) -- isolates "does the
# reduced coordinate system improve the search" from "which gradient
# implementation is faster per call". FULL/reference arm: unmodified
# run_profile_checkpointed, capped at maxit_override=MAXIT with a generous
# wall-clock safety net (should not bind).
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Printf, CSV, DataFrames

const W = 80_000
const DELTA = 1.0
const MAXIT = parse(Int, get(ENV, "AB_MAXIT", "60"))
const SAFETY_BUDGET = 3600.0   # generous wall-clock safety net; should not bind

ctx_probe = d20_real_setup(W = W, find_smallest = true, δ = DELTA, destination_sample = :exclude_row)
pe_probe = build_pivot_elimination(ctx_probe)
D = ctx_probe.D
Aod_theta_natural = ctx_probe.θ0_up[ctx_probe.Aod_offset+1:ctx_probe.Aod_offset+ctx_probe.D*ctx_probe.D_dest]
z0 = log.(Aod_theta_natural)
zfree0 = pivot_reduce(reshape(z0, ctx_probe.D, ctx_probe.D_dest), pe_probe)
gp0 = ctx_probe.θ0_up[3+D]

CKPT_DIR = joinpath(D4X_ROOT, "results", "profiled_ab_2026-08-01", "full_fixediter")
mkpath(CKPT_DIR)

println("="^90); println("FULL/REFERENCE ARM (fixed-iteration): maxit=$MAXIT, W=$W, safety_budget=$(SAFETY_BUDGET)s"); println("="^90); flush(stdout)
t0 = time()
res = run_profile_checkpointed("full_fixediter", gp0, true, zfree0;
    maxtime_real = SAFETY_BUDGET, W_in = W, delta_in = DELTA, draw_seed_in = 20260719,
    ckpt_dir = CKPT_DIR, checkpoint_interval_s = 60.0, maxit_override = MAXIT)
wall = time() - t0

println("="^90)
@printf("RESULT full_fixediter: wall=%.1fs n_eval=%d n_grad_calls=%d native_outer_iters=%d\n",
    wall, res.n_eval, res.n_grad_calls, res.native_outer_diag.n_iters)
println("best=", res.best)
println("="^90)

outpath = joinpath(D4X_ROOT, "FULL_VS_PROFILED_OUTER_AB_FIXEDITER_2026-08-01_FULL_TRACE.csv")
CSV.write(outpath, DataFrame(res.trace))
println("Wrote $outpath")
