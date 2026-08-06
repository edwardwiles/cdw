# Trace every factor inside gamma_component_analytic (lfix_factorized.jl) for both CM+ZC and
# common-Frechet, at the identical calibration point, to find exactly which one is zero for
# Frechet specifically (the analytic d(Delta)/d(gp) for Frechet came back bit-exact 0.0 while a
# central-FD check gave a real, nonzero 0.46 -- user-flagged as implausible, correctly).
const _D4E = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_hessian_subblock_profiling.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
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
xf_from_w_econ(w_econ) = x_free_from_w(vcat(w_econ[1], cm_z_from_a(w_econ[2:end], theta_cm, xy_cm, pe)), pe)
cplus_pool = build_grad_workspace_pool(size(ctx.obj.U, 1))
cplus_ws = build_lfix_factorized_workspace(ctx.D, ctx.D_dest, size(ctx.obj.U, 1))

lp("="^90); lp("=== Frechet: trace gamma_component_analytic's own factors ==="); lp("="^90)
pcx_f = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = :anchored, probs = probs,
    cm_hessian_backend = :structured, threaded_bins = true, include_truncated_moment = false,
    moment_representation = :operator)
w0_frechet = cm_w0_from_calibration(ctx, pe, :powered_aspace)
xf_f = xf_from_w_econ(w0_frechet)
base_f = archC_frechet_base_state(xf_f, pcx_f.ctx_cm, pcx_f.cctx, pcx_f.aug.level_targets)
lp("base_f.ζstar=", base_f.ζstar, " length(base_f.λstar)=", length(base_f.λstar),
   " extrema(λstar)=", extrema(base_f.λstar), " length(base_f.m_star)=", length(base_f.m_star),
   " extrema(m_star)=", extrema(base_f.m_star))
cache_f0 = build_lfix_base_cache_C!(cplus_ws, xf_f, pcx_f.ctx_cm, base_f)
lp("cache_f0.λ_cf=", cache_f0.λ_cf, " cache_f0.wPrime_bi=", cache_f0.wPrime_bi, " cache_f0.LPrime_bi=", cache_f0.LPrime_bi,
   " cache_f0.gammafac=", cache_f0.gammafac, " cache_f0.σ=", cache_f0.σ, " cache_f0.oci=", cache_f0.oci)
mean_mSW_f = sum(base_f.m_star .* cache_f0.SW) / cache_f0.W
lp("mean_mSW_f=", mean_mSW_f, " wPrime_bi_gdp=", wPrime_bi_gdp(cache_f0.wPrime_bi, cache_f0.LPrime_bi))
g1_f = gamma_component_analytic(cache_f0, base_f, w0_frechet[1])
lp("g1_f (should match gfull[1])=", g1_f)

lp("="^90); lp("=== CM+ZC control: trace the SAME factors ==="); lp("="^90)
K_mean = 1; K_pair = 1
pcx = build_cm_meanzc_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = :anchored,
    include_truncated_moment = true, meanzc_basis = :direct, probs = probs, moment_representation = :operator)
nu1_guess = sum(pcx.aug.Zraw_all[1]) / length(pcx.aug.Zraw_all[1])
xf_c = xf_from_w_econ(w0_frechet)   # SAME xf (economic point independent of family)
base_c = archC_meanzc_base_state(xf_c, [nu1_guess], pcx.ctx_cm, pcx.cctx)
lp("base_c.ζstar=", base_c.ζstar, " length(base_c.λstar)=", length(base_c.λstar),
   " length(base_c.m_star)=", length(base_c.m_star), " extrema(m_star)=", extrema(base_c.m_star))
cache_c0 = build_lfix_base_cache_C!(cplus_ws, xf_c, pcx.ctx_cm, base_c)
lp("cache_c0.λ_cf=", cache_c0.λ_cf, " cache_c0.wPrime_bi=", cache_c0.wPrime_bi, " cache_c0.LPrime_bi=", cache_c0.LPrime_bi,
   " cache_c0.gammafac=", cache_c0.gammafac, " cache_c0.oci=", cache_c0.oci)
mean_mSW_c = sum(base_c.m_star .* cache_c0.SW) / cache_c0.W
lp("mean_mSW_c=", mean_mSW_c, " wPrime_bi_gdp=", wPrime_bi_gdp(cache_c0.wPrime_bi, cache_c0.LPrime_bi))
g1_c = gamma_component_analytic(cache_c0, base_c, w0_frechet[1])
lp("g1_c=", g1_c)
lp("=== DONE ===")
