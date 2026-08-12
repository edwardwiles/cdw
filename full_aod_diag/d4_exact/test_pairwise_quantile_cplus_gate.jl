# ================================================================================================
# GATE: this family's Backend C+ (factorized) outer gradient against its DENSE reference.
#
# `pairwise_quantile_cplus.jl` recomputes the economic half of the outer gradient through the
# factorized price representation (never materializing the W x D x Ddest tensors) instead of the
# dense `build_lfix_base_cache` path. The restriction half is untouched by the change. So:
#
#   1. the two gradients must agree -- the SAME quantity by two representations;
#   2. the RESTRICTION tail must be bit-identical (nothing about it changed at all);
#   3. the q0 fold must still be exact against the independently recomputed r, in BOTH paths --
#      that check caught a real bug once and is the reason the fold exists at all;
#   4. C+ must not be slower, which is the entire point.
#
# Tolerance on (1): the two are different representations of the same arithmetic, not a
# reassociation of the same one, so bit-identity is NOT claimed. `c23_cplus_gate.jl` -- this repo's
# own established C+ gate -- holds its directional check to atol/rtol 1e-6; the same standard is
# used here, and the MEASURED agreement is printed so a regression is visible even inside tolerance.
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=16 julia --project=. \
#     full_aod_diag/d4_exact/test_pairwise_quantile_cplus_gate.jl [W] [L] [cutoff_source]
# ================================================================================================
_D4E = joinpath(@__DIR__)
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "compressed_factual_buffer_reuse.jl", "cm_screen_bridge.jl", "gradient_workspace.jl",
          "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl",
          "cm_outer_driver.jl", "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl",
          "cm_meanzc_production.jl", "cm_meanzc_cplus.jl", "incumbent_logic.jl", "cm_checkpoint.jl",
          "cm_originzc_target_layout.jl", "zc_restriction_operator_ragged.jl", "cm_aspace_coordinate.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_config.jl",
          "cm_originzc_checkpoint.jl", "direction_bounds.jl", "cm_frechet_hessian.jl", "cm_frechet_level.jl",
          "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl", "country_resolve.jl",
          "cross_delta_cache.jl", "compressed_moments.jl", "canonical_price_precompute_workspace.jl",
          "hard_score_b_cache.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "lfix_buffer_reuse.jl", "lfix_base_workspace_pooled.jl", "lfix_kbplus_workspace.jl",
          "bandwidth_cache_policy.jl", "fast_range_screen.jl", "dual_bank.jl", "negative_cache.jl",
          "dual_bank_ab_harness.jl", "reusable_context.jl", "organic_failure_capture.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl", "cm_originzc_cross_cplus.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl", "cm_meanzc_cross_production.jl", "cm_meanzc_cross_cplus.jl",
          "multistart_seed_generator.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "operator_psi_bundle.jl",
          "cm_callback_health.jl", "lfix_base_workspace.jl", "shared_a_gradient.jl", "operator_verification.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_verification.jl",
          "pairwise_quantile_production.jl", "pairwise_quantile_mass_gradient.jl",
          "pairwise_quantile_outer_production.jl", "pairwise_quantile_cplus.jl"]
    include(joinpath(_D4E, f))
end
using LinearAlgebra, Printf, SpecialFunctions

lp(xs...) = (println(xs...); flush(stdout))
ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool, detail::AbstractString = "")
    global ALL_PASS[] &= cond
    lp(cond ? "PASS  " : "FAIL  ", name, isempty(detail) ? "" : "  (" * detail * ")")
end

const W_G    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
const L_G    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5
const CUTSRC = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :frechet_theoretical
const GRAV   = default_gravity_exclude_cells_brazil_korea()

lp("="^96)
lp("PAIRWISE-QUANTILE Backend C+ GATE: W=", W_G, " L=", L_G, " cutoff_source=:", CUTSRC)
lp("="^96)

ctx_raw = d20_real_setup_design(W = W_G, δ = 50.0, find_smallest = true, draw_design = :sobol_randomized,
    draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = GRAV, σHat = 3.0, inner_lower_limit = -10.0)
ctx = attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)
layout = PairwiseQuantileMassLayout(ctx.D, L_G)
pcx = build_pairwise_quantile_production_context(ctx, layout;
    cutoff_source = CUTSRC, min_bin_count = max(10, W_G ÷ (2 * L_G^2)))
ctx_cm = pcx.ctx_cm
geo = build_aspace_geometry(ctx); pe = geo.pe
w_cal = cm_w0_from_calibration(ctx, pe, :powered_aspace)
xf = x_free_from_w(vcat(w_cal[1], cm_z_from_a(w_cal[2:end], cm_fixed_theta(ctx),
    precompute_cm_aspace_xy(ctx), pe)), pe)
mass0 = uniform_mass_raw(layout)

lp("solving once (both paths reuse this base/verify, so only the GRADIENT differs) ...")
base, verify = archPQ_verified_state(xf, mass0, ctx_cm)
lp("Delta_dual = ", verify.Delta_dual, "  class = ", classify_inner_result(verify))
check("inner solve VerifiedSolved", classify_inner_result(verify) == VerifiedSolved)

econ_ws = get_or_build_econ_a_grad_ws(W_G)
pool, wsC = pairwise_quantile_cplus_workspaces(ctx_cm)
D2_econ = ctx.D * ctx.D_dest
kw = (threaded = true, h_mode = :cached)

# warm both paths (JIT), then measure
pairwise_quantile_production_gradient(xf, mass0, pcx, ctx, pe; base = base, verify = verify,
    econ_ws = econ_ws, bandwidth_cache = Dict{Int,Float64}(), kw...)
pairwise_quantile_production_gradient_cplus(xf, mass0, pcx, ctx, pe, pool, wsC; base = base,
    verify = verify, bandwidth_cache = Dict{Int,Float64}(), kw...)

bwc_d = Dict{Int,Float64}(); bwc_c = Dict{Int,Float64}()
t_dense = @elapsed g_dense, _ = pairwise_quantile_production_gradient(xf, mass0, pcx, ctx, pe;
    base = base, verify = verify, econ_ws = econ_ws, bandwidth_cache = bwc_d, kw...)
t_cplus = @elapsed g_cplus, _ = pairwise_quantile_production_gradient_cplus(xf, mass0, pcx, ctx, pe,
    pool, wsC; base = base, verify = verify, bandwidth_cache = bwc_c, kw...)

lp("")
check("both gradients have the same length", length(g_dense) == length(g_cplus) == D2_econ + n_raw(layout))
gd_e = g_dense[1:D2_econ];        gc_e = g_cplus[1:D2_econ]
gd_r = g_dense[D2_econ+1:end];    gc_r = g_cplus[D2_econ+1:end]
maxabs = maximum(abs, gd_e .- gc_e)
relerr = maxabs / max(maximum(abs, gd_e), eps())
@printf("economic block: max|dense - cplus| = %.3e   relative = %.3e\n", maxabs, relerr)
check("economic block agrees to the repo's own C+ standard (c23_cplus_gate: atol/rtol 1e-6)",
      isapprox(gd_e, gc_e; atol = 1e-6, rtol = 1e-6), "max|diff|=" * string(maxabs))
check("RESTRICTION tail is BIT-IDENTICAL (nothing about it changed)", gd_r == gc_r,
      "max|diff|=" * string(maximum(abs, gd_r .- gc_r)))

# q0 exactness must survive the port -- in BOTH paths.
cache_d = build_lfix_base_cache_pairwise_quantile(xf, ctx_cm, base; verify = verify, econ_ws = econ_ws)
cache_c = build_lfix_base_cache_pairwise_quantile_C!(wsC, xf, ctx_cm, base; verify = verify)
ed = maximum(abs, cache_d.q0 .- verify.r_current)
ec = maximum(abs, cache_c.q0 .- verify.r_current)
@printf("q0 vs independently recomputed r:  dense %.3e   cplus %.3e\n", ed, ec)
check("q0 fold exact in the DENSE path", ed <= 1e-8, string(ed))
check("q0 fold exact in the C+ path", ec <= 1e-8, string(ec))

@printf("\nwall: dense %.3f s   cplus %.3f s   (speedup %.2fx)\n", t_dense, t_cplus, t_dense / t_cplus)
check("C+ is not slower than the dense path", t_cplus <= t_dense,
      @sprintf("%.3f vs %.3f", t_cplus, t_dense))

lp("")
lp(ALL_PASS[] ? "ALL PAIRWISE-QUANTILE C+ GATE CHECKS PASSED" : "SOME C+ GATE CHECKS FAILED")
exit(ALL_PASS[] ? 0 : 1)
