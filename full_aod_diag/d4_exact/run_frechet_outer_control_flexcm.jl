# Part VII: matched direct delta=1 outer control, flexible CM, through the real public
# checkpointed driver, real production scale. Companion script (same methodology) for
# marginal_restriction=:common_frechet is run_frechet_outer_control_frechet.jl -- run
# SEQUENTIALLY, not concurrently (this project's own standing rule: never spawn concurrent
# KNITRO across processes/threads).
const D4X = @__DIR__
for f in ["draw_design.jl","context_real_d20.jl","winners.jl","oracle.jl","common_marginals_moments.jl","common_marginals_interval.jl",
          "instrumentation.jl","oracle_fast.jl","gravity_elimination.jl","three_way_derivatives.jl",
          "lfix_incremental.jl","composite_gradient.jl","composite_gradient_fast.jl","cm_lookup_kernels.jl",
          "lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_production_bundle.jl","cm_screen_bridge.jl",
          "gradient_workspace.jl","lfix_factorized.jl","lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl","nested_quantile_grids.jl","cm_outer_driver.jl","cm_config.jl",
          "cm_meanzc_moments.jl","cm_meanzc_config.jl","cm_meanzc_production.jl","cm_meanzc_cplus.jl",
          "cm_frechet_level.jl","cm_frechet_hessian.jl","cm_frechet_cplus.jl","cm_checkpoint.jl"]
    include(joinpath(D4X, f))
end
println("=== includes OK ==="); flush(stdout)

CKPT_DIR = joinpath(D4X, "..", "..", "results", "fullA_d4", "frechet_outer_control_flexcm")
rm(CKPT_DIR; force = true, recursive = true); mkpath(CKPT_DIR)

W = 80000; L = 10
probs = cm_equal_grid_probs(L)
ctx0 = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
D0 = ctx0.D; Ddest0 = ctx0.D_dest
x_free_calib0 = ctx0.θ0_up[ctx0.free_idx]
w0 = vcat(x_free_calib0[1], pivot_reduce(log.(reshape(x_free_calib0[2:end], D0, Ddest0)), pe0))

println("="^90)
println("FLEXIBLE CM control: delta=1 direct from calibration, real D=20/W=80000/L=10")
println("="^90)
t0 = time()
result = run_cm_upper_checkpointed(w0; W = W, delta = 1.0, draw_design = :pseudorandom, draw_seed = 20260719,
    L = L, contrasts = :anchored, probs = probs, cm_hessian_backend = :structured,
    marginal_restriction = :common_flexible, cm_gradient_backend = :cplus,
    ckpt_dir = CKPT_DIR, run_id = "flexcm_control", label = "flexcm_control",
    checkpoint_interval_s = 60.0, maxtime_real = 600.0, verbose = true)
wall = time() - t0

println()
println("="^90)
println("RESULT: flexible CM control")
println("  wall=", round(wall, digits=1), "s  n_eval=", result.n_eval, " n_grad=", result.n_grad)
println("  best=", result.best)
println("="^90)
