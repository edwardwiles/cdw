# Closure task Phase 4, INTERRUPT arm: launched with a generous internal budget, killed
# externally via `timeout -s TERM` mid-flight (a genuine external SIGTERM, not a Julia
# exception) after at least one checkpoint has been written. Heartbeat watchdog enabled so
# the resulting log can classify what happened without ambiguity.
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
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
using Printf, Random, Dates

lp(xs...) = (println(xs...); flush(stdout))
const CKPT_ROOT = ARGS[1]
const BUDGET = parse(Float64, ARGS[2])
const DELTA_START = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.0

lp(">>> Julia threads: ", Threads.nthreads())
const DRAW_SEED = 20260719
Random.seed!(DRAW_SEED)
ctx = d20_real_setup(W = 80000, δ = DELTA_START, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D
x_free_calib = ctx.θ0_up[ctx.free_idx]
w_calib = vcat(x_free_calib[1], pivot_reduce(log.(reshape(x_free_calib[2:end], D, D)), pe))
lp(">>> INTERRUPT arm (will be SIGTERM'd externally): internal_budget=", BUDGET, "s delta=", DELTA_START)

snaps = nested_grid_sequence([10, 20, 50])
L = 50
probs = snaps[L]

ckpt_dir = joinpath(CKPT_ROOT, "interrupt")
res = run_cm_upper_checkpointed(w_calib; W = 80000, delta = DELTA_START, draw_design = :pseudorandom,
    draw_seed = DRAW_SEED, L = L, contrasts = :anchored, probs = probs,
    cm_hessian_backend = :structured, cm_grid_rule = :nested_family,
    maxtime_real = BUDGET, ckpt_dir = ckpt_dir, run_id = "phase4_interrupt", label = "interrupt",
    checkpoint_interval_s = 10.0, heartbeat_interval_s = 10.0)

lp(">>> INTERRUPT result (should not normally be reached -- process is meant to be killed first): ",
   "knitro_status=", res.knitro_status, " wall=", round(res.wall,digits=1), " n_eval=", res.n_eval)
