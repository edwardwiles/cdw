# ================================================================================================
# DIAGNOSTIC: are the two routes to "the calibration point" the same point?
#
# Two of this family's D=20 runs reported Delta* at "the calibration point" and disagreed in the 4th
# significant figure (0.028738400455 vs 0.028782523524 at W=100,000). They differed in TWO ways at
# once -- `delta` (1.0 vs 50.0) AND how theta was constructed:
#
#   route A:  x_free = ctx.theta0_up[ctx.free_idx]                       (direct slice)
#   route B:  w = cm_w0_from_calibration(ctx, pe, :powered_aspace)
#             x_free = x_free_from_w(vcat(w[1], cm_z_from_a(w[2:end], ...)), pe)   (a-space round trip)
#
# This file holds `delta` FIXED and varies only the construction, so the two candidate explanations
# are separated instead of confounded.
#
# WHY IT IS WORTH A DEDICATED RUN. This repo's own standing warning is that two "calibration" points
# built differently must be RECONSTRUCTED AND DIFFED, never assumed close -- a past session assumed
# exactly that and was wrong by 27 log points (memory
# `feedback-gravity-elimination-zero-is-not-calibration`). A 4th-significant-figure gap is small
# enough to wave away and that is precisely why it should not be.
#
# Reports the coordinate-space difference FIRST (which is the real question) and only then Delta*,
# so the answer does not depend on the inner solve at all if the points already differ.
#
# Usage: julia --project=. -t N full_aod_diag/d4_exact/diag_cmpq_calibration_point_routes.jl <W> <delta>
# ================================================================================================

const D4X = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl", "three_way_derivatives.jl",
          "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "compressed_factual_buffer_reuse.jl", "cm_screen_bridge.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_outer_driver.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "incumbent_logic.jl", "cm_checkpoint.jl", "cm_originzc_target_layout.jl",
          "zc_restriction_operator_ragged.jl", "cm_aspace_coordinate.jl", "cm_originzc_moments.jl",
          "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_config.jl",
          "cm_originzc_checkpoint.jl", "direction_bounds.jl", "cm_frechet_hessian.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl",
          "country_resolve.jl", "cross_delta_cache.jl", "compressed_moments.jl",
          "canonical_price_precompute_workspace.jl", "hard_score_b_cache.jl", "structured_moment_build.jl",
          "compressed_cc_inner.jl", "compressed_live.jl", "lfix_buffer_reuse.jl",
          "lfix_base_workspace_pooled.jl", "lfix_kbplus_workspace.jl", "bandwidth_cache_policy.jl",
          "fast_range_screen.jl", "dual_bank.jl", "negative_cache.jl", "dual_bank_ab_harness.jl",
          "reusable_context.jl", "organic_failure_capture.jl", "knitro_status.jl",
          "knitro_version_check.jl", "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl", "cm_originzc_cross_cplus.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl", "cm_meanzc_cross_production.jl", "cm_meanzc_cross_cplus.jl",
          "multistart_seed_generator.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_production.jl",
          "cm_pairwise_quantile_config.jl", "cm_pairwise_quantile_moments.jl",
          "cm_pairwise_quantile_hessian.jl", "cm_pairwise_quantile_hessian_assembly.jl",
          "cm_pairwise_quantile_lookup_kernels.jl", "cm_pairwise_quantile_production.jl",
          "cm_pairwise_quantile_verification.jl", "cm_pairwise_quantile_outer_production.jl",
          "cm_pairwise_quantile_cplus.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra

const W_ARG = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
const DELTA = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0
lp(xs...) = (println(xs...); flush(stdout))

lp("="^96)
lp("CM+PQ calibration-point ROUTE COMPARISON:  W=", W_ARG, "  delta=", DELTA, " (HELD FIXED)")
lp("="^96)

const GRAV = default_gravity_exclude_cells_brazil_korea()
ctx_raw = d20_real_setup_design(W = W_ARG, δ = DELTA, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = GRAV, σHat = 3.0,
    inner_lower_limit = -10.0, inner_loop_opt = joinpath(dirname(D4X), "ek_inner_cmpq.opt"))
ctx = attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)

# ---- route A: the direct slice ---------------------------------------------------------------
xfA = ctx.θ0_up[ctx.free_idx]
# ---- route B: the a-space round trip the campaign chain uses -----------------------------------
geo = build_aspace_geometry(ctx); pe = geo.pe
wB = cm_w0_from_calibration(ctx, pe, :powered_aspace)
xfB = x_free_from_w(vcat(wB[1], cm_z_from_a(wB[2:end], cm_fixed_theta(ctx),
    precompute_cm_aspace_xy(ctx), pe)), pe)

lp("\n--- 1. THE COORDINATES THEMSELVES (this is the real question) ---")
@printf("length: A=%d  B=%d\n", length(xfA), length(xfB))
if length(xfA) == length(xfB)
    d = abs.(xfA .- xfB)
    rel = d ./ max.(abs.(xfA), 1e-300)
    @printf("max |A - B|        = %.6e   at coordinate %d\n", maximum(d), argmax(d))
    @printf("max relative |A-B| = %.6e   at coordinate %d\n", maximum(rel), argmax(rel))
    @printf("bit-identical?      %s\n", all(xfA .=== xfB) ? "YES" : "NO")
    @printf("gp (coord 1):       A=%.17g   B=%.17g\n", xfA[1], xfB[1])
    # In LOG terms, which is how A_od differences are meaningful in this codebase.
    lg = abs.(log.(max.(abs.(xfA), 1e-300)) .- log.(max.(abs.(xfB), 1e-300)))
    @printf("max |log A - log B| = %.6e  (log-points)\n", maximum(lg))
end
flush(stdout)

# ---- 2. and only then, Delta* at each --------------------------------------------------------
cfg = CMPairwiseQuantileConfig(L = 5, cm_grid_size = 50, cm_moment_families = 2,
                               contrasts = :orthonormal, min_bin_count = 1, mass_start = :uniform)
pcx = build_cm_pairwise_quantile_production_context(ctx, cfg;
    inner_opt = joinpath(dirname(D4X), "ek_inner_cmpq.opt"))
mass0 = cmpq_uniform_mass_raw(5)

lp("\n--- 2. Delta* at each route, SAME context, SAME delta, SAME masses ---")
results = Any[]
for (nm, xf) in (("A direct theta0_up slice", xfA), ("B a-space round trip", xfB))
    t = @elapsed b, v = archCMPQ_verified_state(xf, mass0, pcx.ctx_cm)
    @printf("  %-26s Delta* = %.15g   n_fg=%d  n_hess=%d  status=%d  class=%s  (%.1fs)\n",
            nm, v.Delta_dual, v.n_fg, v.n_hess, v.inner_status, string(classify_inner_result(v)), t)
    flush(stdout)
    push!(results, (nm = nm, D = v.Delta_dual))
end
dA, dB = results[1].D, results[2].D
@printf("\n  |Delta*_A - Delta*_B| = %.6e   relative = %.6e\n", abs(dA - dB), abs(dA - dB) / abs(dA))

lp("\n--- VERDICT ---")
# Threshold, not bit-identity. Two routes through different floating-point arithmetic will not be
# bit-identical and it would be wrong to call that "a different point": what matters is whether the
# difference is at round-trip scale or at economic scale. This codebase's own reference for "an
# economically different A_od point" is LOG-POINTS (a past confusion was 27 of them), so that is the
# scale the verdict is stated on.
maxlog = length(xfA) == length(xfB) ?
    maximum(abs.(log.(max.(abs.(xfA), 1e-300)) .- log.(max.(abs.(xfB), 1e-300)))) : Inf
relD = abs(dA - dB) / abs(dA)
if maxlog < 1e-10 && relD < 1e-10
    lp("  SAME POINT to floating point: max |log A - log B| = ", maxlog, " log-points, and Delta*")
    lp("  agrees to ", relD, " relative. The two routes are interchangeable, so neither the")
    lp("  construction NOR anything downstream of it explains a larger Delta* discrepancy between")
    lp("  two runs -- look for a setting that differed in the PROBLEM (gravity mask, sigma,")
    lp("  destination_sample, draw design/seed, W).")
else
    lp("  GENUINELY DIFFERENT: max |log A - log B| = ", maxlog, " log-points, Delta* relative ",
       relD, ". These are not the same point and their Delta* must not be compared.")
end
