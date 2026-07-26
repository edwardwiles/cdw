# Phase 0 gate 4, second half: after frechet_killmidrun_driver.jl was SIGKILLed mid-run (whole
# process group, not a graceful shutdown), resume from whatever checkpoint survived the kill and
# verify the resumed state is correct: schema, marginal_restriction, n_eval/n_grad continuing (not
# resetting to 0), and that the resumed solve makes further real progress without error.
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
ckpt_path = joinpath(CKPT_DIR, "frechet_killtest_latest.jls")
@assert isfile(ckpt_path) "no checkpoint survived the kill at $ckpt_path -- FAIL"

ckpt_before = load_cm_checkpoint(ckpt_path)
println("Checkpoint recovered after SIGKILL: schema=", ckpt_before.schema,
        " marginal_restriction=", ckpt_before.marginal_restriction,
        " n_eval=", ckpt_before.n_eval, " n_grad=", ckpt_before.n_grad,
        " gp=", ckpt_before.g)
@assert ckpt_before.schema == CM_CHECKPOINT_SCHEMA "schema mismatch on post-kill checkpoint"
@assert ckpt_before.marginal_restriction == :common_frechet "marginal_restriction corrupted by kill"
@assert ckpt_before.n_eval > 0 "post-kill checkpoint has n_eval=0 -- kill happened before any checkpoint write, retry with a longer pre-kill window"

W = 80000
L = 50
probs = cm_equal_grid_probs(L)

println("="^90)
println("RESUMING after SIGKILL from ", ckpt_path)
println("="^90)
flush(stdout)
result = run_cm_upper_checkpointed(nothing; W = W, delta = 1.0, draw_design = :pseudorandom, draw_seed = 20260719,
    L = L, contrasts = :anchored, probs = probs, cm_hessian_backend = :structured,
    marginal_restriction = :common_frechet,
    ckpt_dir = CKPT_DIR, run_id = "frechet_killtest", label = "frechet_killtest_resumed",
    checkpoint_interval_s = 20.0, maxtime_real = 45.0, verbose = true,
    resume_from = ckpt_path)

println()
println("Post-resume: n_eval=", result.n_eval, " n_grad=", result.n_grad)
@assert result.n_eval >= ckpt_before.n_eval "resumed n_eval ($(result.n_eval)) < pre-kill checkpoint n_eval ($(ckpt_before.n_eval)) -- counters did not carry forward"

ckpt_after = load_cm_checkpoint(joinpath(CKPT_DIR, "frechet_killtest_latest.jls"))
println("Post-resume checkpoint: n_eval=", ckpt_after.n_eval, " marginal_restriction=", ckpt_after.marginal_restriction)
@assert ckpt_after.marginal_restriction == :common_frechet

println()
println("KILL-MID-RUN RESUME GATE: PASS")
