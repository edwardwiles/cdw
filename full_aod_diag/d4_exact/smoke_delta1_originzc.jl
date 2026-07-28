# Post-merge production smoke test (2026-07-28): delta=1, origin-ZC, real D=20/W=80,000.
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
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl", "postmerge_smoke_diagnostics.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Dates, Statistics
lp(xs...) = (println(xs...); flush(stdout))

const FIND_SMALLEST = length(ARGS) >= 1 ? (ARGS[1] == "true") : true
const DIRECTION = FIND_SMALLEST ? "upper" : "lower"
const W = 100_000
const DELTA = 1.0
const BUDGET = 90.0
const OUT = joinpath(_D4E, "..", "..", "results", "postmerge_smoke_2026-07-28", "originzc_$DIRECTION")
rm(OUT; force = true, recursive = true); mkpath(OUT)

lp("=== ORIGIN-ZC smoke ($DIRECTION, find_smallest=$FIND_SMALLEST) ==="); flush(stdout)
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

nu0_log = begin
    layout0 = OriginByPowerLayout(D, 1, 1)
    nu0 = Vector{Float64}(undef, n_eta(layout0))
    for k in 1:1, o in 1:D
        nu0[target_index(layout0, o, k)] = mean(@view (ctx0.U .^ k)[:, o])
    end
    log.(nu0)
end

lp("D=", D, " Ddest=", Ddest)
print_no_h_bundle_facts("zc_only")
reset_no_h_counters!()

t0 = time()
result = run_originzc_upper_checkpointed(vcat(w_a_calib, nu0_log);
    W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719,
    distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = 1, K_pair = 1,
    ckpt_dir = OUT, run_id = "originzc_$DIRECTION", label = "originzc_$DIRECTION",
    checkpoint_interval_s = 60.0, maxtime_real = BUDGET, verbose = true)
wall = time() - t0

lp("="^90)
@printf("RESULT originzc(%s): wall=%.1fs knitro_status=%s n_eval=%d n_grad=%d kappa=%s\n",
    DIRECTION, wall, string(result.knitro_status), result.n_eval, result.n_grad, string(result.kappa))
print_no_h_counters("zc_only")
lp("="^90)
