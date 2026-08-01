# ============================================================================
# Claude Code task 2026-08-01, follow-up (live user request): matched A/B,
# fixed outer-iteration budget, NON-TRIVIAL start point -- gp = 0.99*calibration
# (Delta_dual ~0.33 vs calibration's ~0.002, ~150x harder, verified to still
# converge cleanly and agree with the reference formulation via recover-then-
# resolve to 2.15e-6 before launching, see /tmp/probe_gp_start.jl). A-block
# start unchanged (calibration values) -- only gp is shifted, for both arms.
# FULL/reference arm: unmodified run_profile_checkpointed.
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Printf, CSV, DataFrames

const W = 80_000
const DELTA = 1.0
const MAXIT = parse(Int, get(ENV, "AB_MAXIT", "60"))
const SAFETY_BUDGET = 3600.0
const GP_FRAC = parse(Float64, get(ENV, "AB_GP_FRAC", "0.99"))
const TAG = replace(@sprintf("gp%.3f", GP_FRAC), "." => "p")

ctx_probe = d20_real_setup(W = W, find_smallest = true, δ = DELTA, destination_sample = :exclude_row)
pe_probe = build_pivot_elimination(ctx_probe)
D = ctx_probe.D
Aod_theta_natural = ctx_probe.θ0_up[ctx_probe.Aod_offset+1:ctx_probe.Aod_offset+ctx_probe.D*ctx_probe.D_dest]
z0 = log.(Aod_theta_natural)
zfree0 = pivot_reduce(reshape(z0, ctx_probe.D, ctx_probe.D_dest), pe_probe)
gp0 = ctx_probe.θ0_up[3+D] * GP_FRAC

CKPT_DIR = joinpath(D4X_ROOT, "results", "profiled_ab_2026-08-01", "full_fixediter_$TAG")
mkpath(CKPT_DIR)

println("="^90); println("FULL/REFERENCE ARM (fixed-iteration, gp=$(GP_FRAC)*calib, tag=$TAG): maxit=$MAXIT, gp0=$gp0"); println("="^90); flush(stdout)
t0 = time()
res = run_profile_checkpointed("full_fixediter_$TAG", gp0, true, zfree0;
    maxtime_real = SAFETY_BUDGET, W_in = W, delta_in = DELTA, draw_seed_in = 20260719,
    ckpt_dir = CKPT_DIR, checkpoint_interval_s = 60.0, maxit_override = MAXIT)
wall = time() - t0

println("="^90)
@printf("RESULT full_fixediter_%s: wall=%.1fs n_eval=%d n_grad_calls=%d native_outer_iters=%d\n",
    TAG, wall, res.n_eval, res.n_grad_calls, res.native_outer_diag.n_iters)
println("best=", res.best)
println("="^90)

outpath = joinpath(D4X_ROOT, "FULL_VS_PROFILED_OUTER_AB_FIXEDITER_2026-08-01_$(TAG)_FULL_TRACE.csv")
CSV.write(outpath, DataFrame(res.trace))
println("Wrote $outpath")
