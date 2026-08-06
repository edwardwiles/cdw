# Task section 7: short real public-driver outer smoke for CM+ZC at the INTENDED scientific spec
# (K_mean=3, K_pair=3, meanzc_profiled_level=2, two-family CM block), D20/W=20,000, real production
# manifest (sigma=3, sobol_randomized, Brazil-Korea gravity exclusion, exclude_row). Short budget
# (maxtime_real=180s) -- a smoke, not a campaign, per the task's own "do not launch a full
# campaign" instruction.
const _D4E = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_hessian_subblock_profiling.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "shared_a_gradient.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "hcz_reordered_candidate_2026-08-01.jl",
          "cm_meanzc_lookup_production.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Dates
lp(xs...) = (println(xs...); flush(stdout))

W = 20_000; L = 50
probs = cm_equal_grid_probs(L)
GRAV = default_gravity_exclude_cells_brazil_korea()
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :sobol_randomized, draw_seed = 20260719,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true, gravity_exclude_cells = GRAV, σHat = 3.0)
pe = build_pivot_elimination(ctx)
theta_cm = cm_fixed_theta(ctx); xy_cm = precompute_cm_aspace_xy(ctx)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
xf_from_w_econ(w_econ) = x_free_from_w(vcat(w_econ[1], cm_z_from_a(w_econ[2:end], theta_cm, xy_cm, pe)))
w0_a = cm_w0_from_calibration(ctx, pe, :powered_aspace)
xf0 = xf_from_w_econ(w0_a)

K_mean = 3; K_pair = 3
# nu-vector guess, matching diag_cmzc_k3_intendedspec_2026-08-06.jl's own construction (level 2
# profiled to the autarky-consistent value; levels 1,3 use the diag_meanzc_verify4 mean-of-raw-Z
# convention). Built via a throwaway context just to reach pcx.aug.Zraw_all -- the real w0 is what
# the driver's own cb_F!/cb_G! will use at every subsequent call via meanzc_profiled_level below.
pcx_probe = build_cm_meanzc_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = :anchored,
    include_truncated_moment = true, meanzc_basis = :direct, probs = probs, moment_representation = :operator)
nu_profiled_2 = meanzc_profiled_nu_value(xf0, ctx)
nu_guess = [sum(pcx_probe.aug.Zraw_all[k]) / length(pcx_probe.aug.Zraw_all[k]) for k in 1:K_mean]
nu_guess[2] = nu_profiled_2
w0 = vcat(w0_a, log.(nu_guess))
lp("[cmzc-k3-smoke] w0 built, nu_guess=", nu_guess)

OUT = joinpath(_D4E, "..", "..", "..", "..", "repo_scratch", "cm-extensions-gradient-and-production-final-2026-08-06", "cmzc_k3_smoke_w20k")
lp("[cmzc-k3-smoke] resolved OUT=", abspath(OUT))
isdir(OUT) || mkpath(OUT)   # NOTE: do not rm/wipe -- resume script (smoke_cmzc_k3_resume_2026-08-06.jl) points at this same dir

t0 = time()
result = run_cm_upper_checkpointed(w0;
    W = W, delta = 1.0, draw_design = :sobol_randomized, draw_seed = 20260719, L = L, contrasts = :anchored, probs = probs,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true, gravity_exclude_cells = GRAV, σHat = 3.0,
    cm_extension = :cm_plus_moments, meanzc_K_mean = K_mean, meanzc_K_pair = K_pair,
    meanzc_profiled_level = 2, include_truncated_moment = true,
    ckpt_dir = OUT, run_id = "cmzc_k3_smoke", label = "cmzc_k3_smoke",
    checkpoint_interval_s = 60.0, maxtime_real = 900.0, verbose = true)
wall = time() - t0

lp("="^100)
@printf("RESULT cmzc_k3_smoke: wall=%.1fs knitro_status=%s n_eval=%d n_grad=%d kappa=%s\n",
    wall, string(result.knitro_status), result.n_eval, result.n_grad, string(result.kappa))
lp("="^100)
