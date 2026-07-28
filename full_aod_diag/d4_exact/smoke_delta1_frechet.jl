# Post-merge production smoke test (2026-07-28): delta=1, common Frechet, real D=20/W=80,000.
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Dates
lp(xs...) = (println(xs...); flush(stdout))

const FIND_SMALLEST = length(ARGS) >= 1 ? (ARGS[1] == "true") : true
const DIRECTION = FIND_SMALLEST ? "upper" : "lower"
const W = 80_000
const DELTA = 1.0
const BUDGET = 600.0
const OUT = joinpath(_D4E, "..", "..", "results", "postmerge_smoke_2026-07-28", "frechet_$DIRECTION")
rm(OUT; force = true, recursive = true); mkpath(OUT)

lp("=== COMMON FRECHET smoke ($DIRECTION, find_smallest=$FIND_SMALLEST) ==="); flush(stdout)
ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = FIND_SMALLEST, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
D = ctx0.D; Ddest = ctx0.D_dest
theta0 = cm_fixed_theta(ctx0)
xy0 = precompute_cm_aspace_xy(ctx0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
gp_calib = x_free_calib[1]
z_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, Ddest)), pe0)
a_calib = cm_a_from_z(z_calib, theta0, xy0, pe0)
w_a_calib = vcat(gp_calib, a_calib)

const SNAPS = nested_grid_sequence([10, 20, 50])
const PROBS_L50 = SNAPS[50]

t0 = time()
result = run_cm_upper_checkpointed(copy(w_a_calib);
    W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719, L = 50, contrasts = :orthonormal, probs = PROBS_L50,
    marginal_restriction = :common_frechet, cm_hessian_backend = :structured, cm_gradient_backend = :cplus,
    ckpt_dir = OUT, run_id = "frechet_$DIRECTION", label = "frechet_$DIRECTION",
    checkpoint_interval_s = 60.0, maxtime_real = BUDGET, verbose = true)
wall = time() - t0

lp("="^90)
@printf("RESULT frechet(%s): wall=%.1fs knitro_status=%s n_eval=%d n_grad=%d kappa=%s\n",
    DIRECTION, wall, string(result.knitro_status), result.n_eval, result.n_grad, string(result.kappa))
lp("="^90)
