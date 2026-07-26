# Five-family finish task, Phase 8 (2026-07-26): live smoke test of the NEW
# A_coordinate_mode=:powered_aspace default (just flipped in cm_checkpoint.jl/
# cm_originzc_checkpoint.jl) through the REAL public driver entry points, for all four restricted
# families, at real D=20/W=80,000. Confirms the flipped default actually produces a real, feasible
# (or genuinely progressing) KNITRO solve from a correctly-constructed a-space w0 -- not just that
# the coordinate-conversion math is correct in isolation (that was already proven by
# test_cm_aspace_coordinate_gates.jl, 6/6 PASS, real D=20 calibration point).
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
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Dates

lp(xs...) = (println(xs...); flush(stdout))
const W = 80_000
const DELTA = 1.0
const BUDGET = 45.0
const OUT = joinpath(_D4E, "..", "..", "results", "phase8_transformed_a_smoke_2026-07-26")
mkpath(OUT)
const FAILURES = String[]
check(name, cond) = (println(rpad(cond ? "PASS" : "FAIL", 6), name); cond || push!(FAILURES, name); flush(stdout); cond)

lp("Building D=20 real context + a-space calibration w0 ..."); flush(stdout)
ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
D = ctx0.D; Ddest = ctx0.D_dest
theta0 = cm_fixed_theta(ctx0)
xy0 = precompute_cm_aspace_xy(ctx0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
gp_calib = x_free_calib[1]
z_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, Ddest)), pe0)
a_calib = cm_a_from_z(z_calib, theta0, xy0, pe0)
w_a_calib = vcat(gp_calib, a_calib)
lp("a-space w0 built: ||a_calib||=", round(norm(a_calib), digits = 4))

const SNAPS = nested_grid_sequence([10, 20, 50])
const PROBS_L50 = SNAPS[50]

function smoke(label, run_fn)
    t0 = time()
    res = run_fn()
    wall = time() - t0
    @printf("[%s] wall=%.1fs knitro_status=%s n_eval=%d kappa=%s\n",
        label, wall, string(res.knitro_status), res.n_eval, string(res.kappa))
    check("$label: A_coordinate_mode=:powered_aspace DEFAULT produces a real, non-crashing KNITRO run", res.n_eval > 0)
    return res
end

lp("="^100, "\nFLEXIBLE CM (default kwargs -- confirms new :powered_aspace default)\n", "="^100)
smoke("flexibleCM", () -> run_cm_upper_checkpointed(copy(w_a_calib);
    W = W, delta = DELTA, draw_seed = 20260719, L = 50, contrasts = :orthonormal, probs = PROBS_L50,
    maxtime_real = BUDGET, ckpt_dir = joinpath(OUT, "flexcm"), label = "flexcm",
    checkpoint_interval_s = 1000.0, verbose = false))

lp("="^100, "\nCOMMON FRECHET (default kwargs)\n", "="^100)
smoke("commonFrechet", () -> run_cm_upper_checkpointed(copy(w_a_calib);
    W = W, delta = DELTA, draw_seed = 20260719, L = 50, contrasts = :orthonormal, probs = PROBS_L50,
    marginal_restriction = :common_frechet, cm_hessian_backend = :structured,
    maxtime_real = BUDGET, ckpt_dir = joinpath(OUT, "frechet"), label = "frechet",
    checkpoint_interval_s = 1000.0, verbose = false))

lp("="^100, "\nCM+mean/ZC (default kwargs)\n", "="^100)
smoke("cmMeanZC", () -> run_cm_upper_checkpointed(vcat(w_a_calib, log.(Float64.(factorial.(1:1))));
    W = W, delta = DELTA, draw_seed = 20260719, L = 50, contrasts = :orthonormal, probs = PROBS_L50,
    cm_extension = :cm_plus_moments, meanzc_K_mean = 1, meanzc_K_pair = 1,
    maxtime_real = BUDGET, ckpt_dir = joinpath(OUT, "meanzc"), label = "meanzc",
    checkpoint_interval_s = 1000.0, verbose = false))

lp("="^100, "\nORIGIN-ZC (default kwargs)\n", "="^100)
nu0_log = begin
    layout0 = OriginByPowerLayout(D, 1, 1)
    nu0 = Vector{Float64}(undef, n_eta(layout0))
    for k in 1:1, o in 1:D
        nu0[target_index(layout0, o, k)] = mean(@view (ctx0.U .^ k)[:, o])
    end
    log.(nu0)
end
smoke("originZC", () -> run_originzc_upper_checkpointed(vcat(w_a_calib, nu0_log);
    W = W, delta = DELTA, draw_seed = 20260719,
    distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = 1, K_pair = 1,
    maxtime_real = BUDGET, ckpt_dir = joinpath(OUT, "originzc"), label = "originzc",
    checkpoint_interval_s = 1000.0, verbose = false))

lp("="^100)
if isempty(FAILURES)
    println("ALL PHASE 8 TRANSFORMED-A DEFAULT SMOKE TESTS PASSED")
else
    println("FAILURES (", length(FAILURES), "): "); for f in FAILURES; println("  - ", f); end
    exit(1)
end
