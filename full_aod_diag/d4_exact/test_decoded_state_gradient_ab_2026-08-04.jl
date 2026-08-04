# task §9 (profiled-outer-ab-readiness-2026-08-04): decoded-state outer-gradient A/B.
# Cross-formulation comparison at a MATCHED decoded economic state (same calibration A/gp),
# for a single shared economic direction, family=flexible_cm (representative of the non-ZC
# shared-engine families; ZC eta/A/joint directions are a genuine follow-up, not attempted here
# given session time constraints -- see the interim closeout for exactly what remains).
#
# "Raw gradient vectors of different dimensions/bases are not directly comparable" (task's own
# words) -- REDUCED's analytic gradient lives in r_free-space (dim n_retained-1), FULL's lives in
# z_nonpivot-space (dim D*Ddest-1, confirmed by direct read: composite_gradient_at_fast's own
# `w0 = vcat(x_free0[1], pivot_reduce(z0,pe))` means its FD loop operates on z_nonpivot directly,
# NOT a-space -- the :powered_aspace transform is a SEPARATE layer, outer_coordinate_layout.jl's
# own gradient_transform_unified, applied outside this function). This script maps a SINGLE shared
# economic direction `dz` (a D x Ddest log-A tangent vector, gravity-feasible AND REDUCED-anchor-
# feasible, i.e. valid in BOTH formulations' own tangent subspaces since REDUCED's is a strict
# subset of FULL's) into each formulation's own coordinate tangent space via each side's OWN
# already-derived affine Jacobian (REDUCED: pivot_expand_on_retained/decode_relative_A, chosen
# construction: DIFFERENCE the affine map at the point vs at zero, which cancels the additive
# offset and gives the pure linear/tangent part with NO new derivation; FULL: dz_nonpivot IS the
# FD-loop's own native coordinate directly, no transform needed), then compares the resulting
# directional derivatives.
#
# Ground truth: `fixed_dual_L` (three_way_derivatives.jl) -- a full-rebuild, dense-G fixed-dual
# functional. Used ONCE, at ONE point, for ONE direction (not swept across coordinates) --
# deliberately bounded scope, per this repo's own hard lesson (docs/audits/profiled-functional-
# readiness-closeout-2026-08-03/CONTINUATION_2026-08-04.md's own "First version... materializing
# dense G... exactly the pattern feedback-no-dense-reduced-ever-anywhere exists to prevent" episode
# -- that was a 361-coordinate SWEEP; this is a single directional probe, a materially different
# and bounded cost, but still logged explicitly as dense-G usage below, not hidden).
D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "no_dense_g_counters.jl", "economic_operator.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl",
          "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl",
          "blas_thread_policy.jl", "knitro_outer_algorithm.jl", "production_backend_manifest.jl",
          "incumbent_logic.jl", "cm_hessian_subblock_profiling.jl", "production_bundle_api.jl", "country_resolve.jl",
          "cm_exact_cache_production.jl", "cm_checkpoint.jl", "draw_design.jl",
          "operator_hessian_weights.jl", "operator_psi_bundle.jl", "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl",
          "gravity_pivot_on_retained_2026-07-31.jl", "outer_coordinate_layout_profiled_2026-07-31.jl",
          "reduced_operator_verification_2026-08-01.jl",
          "recover_full_a_2026-07-31.jl",
          "profiled_reduced_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_frechet_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_originzc_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl",
          "profiled_outer_gradient_layout_contract_2026-08-01.jl",
          "profiled_stable_layout_digest_2026-08-01.jl",
          "profiled_operator_bundle_2026-08-01.jl",
          "profiled_outer_evaluator_2026-08-01.jl",
          "profiled_lfix_incremental_2026-08-01.jl",
          "profiled_shared_economic_gradient_engine_2026-08-01.jl",
          "profiled_family_adapters_2026-08-01.jl",
          "profiled_restriction_contrib0_operators_2026-08-01.jl",
          "profiled_restricted_family_adapters_2026-08-02.jl",
          "profiled_matched_gradient_instrumentation_2026-08-04.jl"]
    include(joinpath(D4X, f))
end

lp(xs...) = (println(xs...); flush(stdout))
const W_VAL = parse(Int, get(ENV, "GATE_W", "20000"))

ctx = d20_real_setup_design(; W = W_VAL, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = [(3, 14)], σHat = 3.0)
D = ctx.D
korea_idx, brazil_idx = 14, 3

# ---- REDUCED (flexible_CM) ----
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
θ0 = ctx.θ0_up
z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*ctx.D_dest], D, ctx.D_dest))
gauge = build_anchor_gauge(z_calib, spec)
pe_r = build_pivot_elimination_on_retained(ctx, spec, gauge)
gp0 = θ0[3+D]
w0 = reduce_to_w_profiled(gp0, z_calib, pe_r)
θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
aug = build_cm_augmented_obj_archB(ctx, CS; L = 50, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
cctx = build_cm_bin_ctx(ctx, aug; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = true)
fctx = build_flexcm_family_ctx(ctx, spec, pe_r, layout, cctx)
ev = evaluate_profiled_flexcm_point(w0, fctx)
g_r, meta_r = shared_family_outer_gradient(w0, ctx, fctx, ev; threaded = false)
lp("REDUCED gradient computed, length=", length(g_r))

# ---- FULL (unrestricted) ----
pe_f = build_pivot_elimination(ctx)
x_free0 = ctx.θ0_up[ctx.free_idx]
base = solve_base_state(x_free0, ctx)
cacheF = build_lfix_base_cache(x_free0, ctx, base)
g_F, metaF = composite_gradient_at_fast(x_free0, ctx, pe_f; cache = cacheF, base = base, threaded = false,
    h_mode = :adaptive, validate_frac = 0.0)
lp("FULL gradient computed, length=", length(g_F))

# ---- shared economic direction: REDUCED coordinate k_free (1-indexed within r_free) ----
const K_FREE = 2
n_free = length(pe_r.other_pos)
dr_free = zeros(n_free); dr_free[K_FREE] = 1.0

dr = pivot_expand_on_retained(dr_free, pe_r) .- pivot_expand_on_retained(zeros(n_free), pe_r)
dz = decode_relative_A(dr, spec, gauge) .- decode_relative_A(zeros(length(dr)), spec, gauge)

c = gravity_linear_coeffs(ctx)
gravity_resid = sum(c .* dz)
lp("gravity residual c'*dz = ", gravity_resid, " (should be ~0, gravity-feasible direction)")

# REDUCED directional derivative: dr_free = e_{K_FREE}, so g_r . dr_free is just that ONE component
# (index 1+K_FREE in g_r's own [gp; r_free] layout).
dd_reduced = g_r[1 + K_FREE]

# FULL directional derivative: dz restricted to FULL's own z_nonpivot ordering, dotted with FULL's
# OWN z-space gradient (g_F[2:end], native z_nonpivot coordinates -- no a-space transform needed,
# confirmed by direct read above).
dz_nonpivot_full = vec(dz)[pe_f.other_idx]
dd_full = dot(g_F[2:end], dz_nonpivot_full)

lp("")
lp("REDUCED directional derivative (analytic, r_free basis) = ", dd_reduced)
lp("FULL directional derivative     (analytic, z_nonpivot basis, chain-rule-mapped dz) = ", dd_full)
rel_err_cross = abs(dd_reduced - dd_full) / max(abs(dd_reduced), abs(dd_full), 1e-12)
lp("cross-formulation relative error = ", rel_err_cross)

# ---- shared FD ground truth (fixed_dual_L, dense-G, ONE point ONE direction -- bounded scope) ----
t_step = meta_r.h_used[1 + K_FREE]   # reuse REDUCED's own already-selected, non-tied-region bandwidth
Aod_lvl0 = exp.(z_calib)
x_free_at(t) = vcat(gp0, vec(exp.(z_calib .+ t .* dz)))
Kp = fixed_dual_L(x_free_at(t_step), ctx, base)
Km = fixed_dual_L(x_free_at(-t_step), ctx, base)
fd_ground_truth = (Kp - Km) / (2 * t_step)
lp("")
lp("shared FD ground truth (fixed_dual_L, t=", t_step, ") = ", fd_ground_truth)
lp("REDUCED vs FD rel_err = ", abs(dd_reduced - fd_ground_truth) / max(abs(fd_ground_truth), 1e-12))
lp("FULL     vs FD rel_err = ", abs(dd_full - fd_ground_truth) / max(abs(fd_ground_truth), 1e-12))

sign_agree = sign(dd_reduced) == sign(dd_full) == sign(fd_ground_truth)
lp("")
lp("sign agreement (REDUCED, FULL, FD all same sign) = ", sign_agree)

gate = (abs(gravity_resid) < 1e-8) && (rel_err_cross < 0.01) &&
       (abs(dd_reduced - fd_ground_truth) / max(abs(fd_ground_truth), 1e-12) < 0.05) &&
       (abs(dd_full - fd_ground_truth) / max(abs(fd_ground_truth), 1e-12) < 0.05) && sign_agree
lp("")
lp("DECODED_STATE_GRADIENT_AB_flexible_cm_calibration: ", gate ? "PASS" : "FAIL")
