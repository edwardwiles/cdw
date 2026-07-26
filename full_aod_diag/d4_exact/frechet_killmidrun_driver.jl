# Phase 0 gate 4 (2026-07-26, production-audit task): genuine kill-mid-run checkpoint test for
# marginal_restriction=:common_frechet through the real public checkpointed driver
# (run_cm_upper_checkpointed). Disclosed gap in the prior session's own verdict: "only graceful
# resume was tested, not a real SIGKILL/process-group-kill mid-write test". This script is the
# LONG-RUNNING half: launched under `setsid`, killed externally with SIGKILL to its process group
# partway through (see frechet_killmidrun_resume.jl for the second half). Production grid L=50.
const D4X = @__DIR__
for f in ["draw_design.jl","context_real_d20.jl","winners.jl","oracle.jl","common_marginals_moments.jl","common_marginals_interval.jl",
          "instrumentation.jl","oracle_fast.jl","gravity_elimination.jl","three_way_derivatives.jl",
          "lfix_incremental.jl","composite_gradient.jl","composite_gradient_fast.jl","cm_lookup_kernels.jl",
          "lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_hessian_threaded.jl","cm_production_bundle.jl","cm_screen_bridge.jl",
          "gradient_workspace.jl","lfix_factorized.jl","lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl","nested_quantile_grids.jl","cm_outer_driver.jl","cm_config.jl",
          "cm_meanzc_moments.jl","cm_meanzc_config.jl","cm_meanzc_production.jl","cm_meanzc_cplus.jl",
          "cm_frechet_level.jl","cm_frechet_hessian.jl","cm_frechet_hessian_threaded.jl","cm_frechet_cplus.jl","cm_checkpoint.jl"]
    include(joinpath(D4X, f))
end
println("=== includes OK ==="); flush(stdout)

CKPT_DIR = joinpath(D4X, "..", "..", "results", "fullA_d4", "frechet_killmidrun_test")
rm(CKPT_DIR; force = true, recursive = true)
mkpath(CKPT_DIR)

W = 80000
L = 50
probs = cm_equal_grid_probs(L)

ctx0 = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
D0 = ctx0.D; Ddest0 = ctx0.D_dest
x_free_calib0 = ctx0.θ0_up[ctx0.free_idx]
w0 = vcat(x_free_calib0[1], pivot_reduce(log.(reshape(x_free_calib0[2:end], D0, Ddest0)), pe0))

println("="^90)
println("KILL-MID-RUN (long half): marginal_restriction=:common_frechet, L=50, checkpoint_interval_s=8")
println("PID=", getpid())
println("="^90)
flush(stdout)
result = run_cm_upper_checkpointed(w0; W = W, delta = 1.0, draw_design = :pseudorandom, draw_seed = 20260719,
    L = L, contrasts = :anchored, probs = probs, cm_hessian_backend = :structured,
    marginal_restriction = :common_frechet,
    ckpt_dir = CKPT_DIR, run_id = "frechet_killtest", label = "frechet_killtest",
    checkpoint_interval_s = 8.0, maxtime_real = 280.0, verbose = true)
println("UNEXPECTED: process was not killed, ran to completion. n_eval=", result.n_eval)
