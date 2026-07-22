# Closure task Phase 4 RESUME arm -- fresh process, resumes from the interrupt arm's latest
# CMCheckpoint (current schema=2 only) for the remainder of the total budget.
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
using Printf, Dates

lp(xs...) = (println(xs...); flush(stdout))
const CKPT_ROOT = ARGS[1]
const REMAINING_BUDGET = parse(Float64, ARGS[2])
const resume_path = joinpath(CKPT_ROOT, "interrupt", "interrupt_latest.jls")

lp(">>> Julia threads: ", Threads.nthreads())
isfile(resume_path) || error("c33_phase4_cm_shakedown_resume: no checkpoint found at $resume_path -- interrupt arm must have written at least one checkpoint before being killed")
lp(">>> RESUME arm: resuming from ", resume_path, " remaining_budget=", REMAINING_BUDGET, "s")

res = run_cm_upper_checkpointed(nothing; ckpt_dir = joinpath(CKPT_ROOT, "interrupt"),
    run_id = "phase4_interrupt", label = "interrupt", maxtime_real = REMAINING_BUDGET,
    checkpoint_interval_s = 10.0, resume_from = resume_path, heartbeat_interval_s = 10.0)

lp(">>> RESUME result: knitro_status=", res.knitro_status, " wall=", round(res.wall,digits=1),
   " n_eval=", res.n_eval, " n_grad=", res.n_grad,
   " best=", res.best === nothing ? "nothing" : "gp=$(res.best.gp) Delta=$(res.best.Delta)",
   " kappa=", res.kappa)
lp(">>> resume checkpoint: ", res.ckpt_path)
