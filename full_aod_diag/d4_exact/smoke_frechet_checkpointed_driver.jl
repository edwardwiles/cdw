# Part V smoke test: exercises marginal_restriction=:common_frechet through the REAL public
# checkpointed driver (run_cm_upper_checkpointed), not a test-script bypass. Small W and short
# maxtime_real for a fast wiring-correctness check -- the full production-scale (W=80,000) gate is
# Part VI.
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
using Printf

CKPT_DIR = joinpath(D4X, "..", "..", "results", "fullA_d4", "frechet_ckpt_smoke_test")
rm(CKPT_DIR; force = true, recursive = true)
mkpath(CKPT_DIR)

W = 80000   # production scale (small W=2000 confirmed to fail identically for plain flexible CM -- a known small-W conditioning issue, not restriction-family-specific)
L = 10
probs = cm_equal_grid_probs(L)

# Real calibration point, same shape run_cm_upper_checkpointed's OWN internal ctx will use
# (destination_sample=:exclude_row default) -- mirrors checkpoint_legacy_toggle_test.jl's own
# w0-construction pattern exactly.
ctx0 = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
D0 = ctx0.D; Ddest0 = ctx0.D_dest
x_free_calib0 = ctx0.θ0_up[ctx0.free_idx]
w0 = vcat(x_free_calib0[1], pivot_reduce(log.(reshape(x_free_calib0[2:end], D0, Ddest0)), pe0))

println("="^90)
println("SMOKE: marginal_restriction=:common_frechet through run_cm_upper_checkpointed")
println("="^90)
result = run_cm_upper_checkpointed(w0; W = W, delta = 1.0, draw_design = :pseudorandom, draw_seed = 20260719,
    L = L, contrasts = :anchored, probs = probs, cm_hessian_backend = :structured,
    marginal_restriction = :common_frechet,
    ckpt_dir = CKPT_DIR, run_id = "frechet_smoke", label = "frechet_smoke",
    checkpoint_interval_s = 30.0, maxtime_real = 150.0, verbose = true)

println()
println("Result keys: ", keys(result))
println("n_eval=", result.n_eval, "  n_grad=", result.n_grad)
println("best=", result.best)

ckpt_path = joinpath(CKPT_DIR, "frechet_smoke_latest.jls")
@assert isfile(ckpt_path) "no checkpoint written at $ckpt_path"
ckpt = load_cm_checkpoint(ckpt_path)
println()
println("Checkpoint: schema=", ckpt.schema, " marginal_restriction=", ckpt.marginal_restriction,
        " n_eval=", ckpt.n_eval, " n_grad=", ckpt.n_grad)
@assert ckpt.schema == CM_CHECKPOINT_SCHEMA "checkpoint schema mismatch"
@assert ckpt.marginal_restriction == :common_frechet "checkpoint marginal_restriction mismatch"

println()
println("SMOKE PASS: fresh run through the real public checkpointed driver, marginal_restriction=:common_frechet")
