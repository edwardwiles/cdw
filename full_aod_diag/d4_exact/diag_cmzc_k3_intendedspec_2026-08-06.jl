# Task section 7 quick feasibility gate: CM+ZC at the INTENDED scientific spec (K_mean=3,
# K_pair=3, k=(sigma-1)=2 collinearity handled via meanzc_profiled_level=2, the nu-profiling fix
# merged this branch at commit 648d043/0671d34/e22976d -- confirmed `git merge-base --is-ancestor`
# TRUE against this branch's HEAD before writing this script). Direct pcx-level inner-solve check
# (archC_meanzc_verified_state), not the full outer KNITRO driver -- answers "is K=3 scientifically
# reachable at all" cheaply before attempting any outer smoke.
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
using Printf
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
w0 = cm_w0_from_calibration(ctx, pe, :powered_aspace)
xf0 = xf_from_w_econ(w0)

lp("="^100); lp("=== CM+ZC intended-spec (K_mean=3, K_pair=3, meanzc_profiled_level=2) D20/W=20,000/L=50 ==="); lp("="^100)
K_mean = 3; K_pair = 3
pcx = build_cm_meanzc_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = :anchored,
    include_truncated_moment = true, meanzc_basis = :direct, probs = probs, moment_representation = :operator)
lp("[cmzc-k3] context built: n_families=", pcx.cctx.n_families, " Pow!==nothing=", pcx.cctx.Pow !== nothing,
   " inner_fg_backend=", pcx.cctx.inner_fg_backend)

# Build the K_mean=3 nu-vector guess: level 2 (k=sigma-1=2) uses the profiled autarky-consistent
# value (meanzc_profiled_nu_value, cm_checkpoint.jl); the other two levels use the SAME
# uninformed-but-consistent convention diag_meanzc_verify4 established (mean of the raw Z draws for
# that power level) -- pcx.aug.Zraw_all[k] holds the k-th power level's raw Z draws.
nu_profiled_2 = meanzc_profiled_nu_value(xf0, ctx)
nu_guess = [sum(pcx.aug.Zraw_all[k]) / length(pcx.aug.Zraw_all[k]) for k in 1:K_mean]
nu_guess[2] = nu_profiled_2
lp("[cmzc-k3] nu_guess (level1,2,3) = ", nu_guess, " (level 2 profiled=", nu_profiled_2, ")")

try
    global base, verify = archC_meanzc_verified_state(xf0, nu_guess, pcx.ctx_cm, pcx.cctx)
    lp("[cmzc-k3] CALIBRATION inner solve: inner_status=", verify.inner_status, " Delta_dual=", verify.Delta_dual,
       " feasible=", verify.Delta_dual < 1.0, " primal_dual_gap=", verify.primal_dual_gap,
       " max_abs_moment_kkt_resid=", verify.max_abs_moment_kkt_resid)
    lp("[cmzc-k3] RESULT: scientific_spec=REACHABLE (calibration inner solve converged, verified)")
catch e
    lp("[cmzc-k3] CALIBRATION inner solve FAILED: ", sprint(showerror, e)[1:min(end, 300)])
    lp("[cmzc-k3] RESULT: scientific_spec=BLOCKED (calibration inner solve did not converge)")
end
lp("=== DONE ===")
