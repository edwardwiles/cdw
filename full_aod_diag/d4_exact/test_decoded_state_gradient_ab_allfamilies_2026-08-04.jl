# Task §9 (profiled-outer-ab-completion-2026-08-04): decoded-state outer-gradient A/B, extended to
# ALL FIVE REDUCED families (the prior session's own evidence, task §8 there, covered only
# flexible_cm x calibration x one direction). Real D4 KNITRO context, no mocks.
#
# SCOPE (honestly bounded, see docs/audits/.../SECTION9_DECODED_STATE_GRADIENT_AB.md for the full
# accounting): this file covers the A-BLOCK analytic-vs-central-fixed-dual-FD comparison (the
# "ordinary A movement"/"gravity-pivot-coupled A movement" direction categories -- EVERY REDUCED A
# coordinate is gravity-pivot-coupled by construction, see profiled_affected_cells) at 2 points x 3
# representative coordinates x all 5 families, using the SAME shared_family_outer_gradient/
# profiled_lfix_incremental_at machinery every existing gate already trusts, with bandwidth MATCHED
# to profiled_select_bandwidth's own adaptive choice (task's own standing FD-bandwidth pitfall,
# confirmed live in section 6 of this same continuation -- a naive fixed h gives a false ~1500%
# gap). The gp-only direction and eta_nu-only/joint-A+eta_nu directions for origin_ZC/CM_plus_ZC
# are NOT re-derived here -- both already have real, independently-verified evidence elsewhere
# (gp: test_flexcm_frechet_outer_gradient_d20_w100k_2026-08-04.jl, 11 coords incl. gp,
# max_rel_err=1.11e-10/1.03e-10; eta_nu: profiled_zc_free_eta's own D4/D20w20k/D20w100k gates,
# ~1e-10 to ~2.5e-12) -- referenced, not duplicated.
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
          "cm_exact_cache_production.jl", "cm_checkpoint.jl",
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
          "profiled_originzc_family_adapter_2026-08-02.jl",
          "profiled_cmzc_family_adapter_2026-08-02.jl",
          "profiled_zc_lane_point_evaluators_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Random, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))
results = Dict{Tuple{Symbol,String,Int},Bool}()
summary_rows = NamedTuple[]

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(3 => 1))
θ0 = ctx.θ0_up
z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))
gauge = build_anchor_gauge(z_calib, spec)
pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
gp0 = θ0[3+D]
w_calib = reduce_to_w_profiled(gp0, z_calib, pe)
n_free = length(w_calib) - 1

θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)

Random.seed!(20260804)
w_p1 = copy(w_calib); w_p1[2:end] .+= 0.01 .* randn(n_free)

"A-block analytic-vs-matched-bandwidth-FD check at `w0`, `n_coords` representative coordinates, using the family's own `ev`/`fctx`."
function check_ablock!(family::Symbol, label::String, w0::Vector{Float64}, fctx, ev, n_coords::Int)
    g_native, _ = shared_family_outer_gradient(w0, ctx, fctx, ev; threaded = false)
    cache = build_shared_profiled_lfix_cache(w0, fctx, ctx, ev)
    coords = collect(2:min(1 + n_coords, length(w0)))
    max_rel = 0.0
    for k in coords
        h, _m, _sel = profiled_select_bandwidth(cache, ctx, spec, pe, w0, k)
        Lp = profiled_lfix_incremental_at(cache, ctx, spec, pe, w0, k, w0[k] + h)
        Lm = profiled_lfix_incremental_at(cache, ctx, spec, pe, w0, k, w0[k] - h)
        fd = (Lp - Lm) / (2h)
        rel = abs(fd - g_native[k]) / max(abs(fd), 1e-8)
        max_rel = max(max_rel, rel)
        push!(summary_rows, (family = family, point = label, coord = k, h = h, analytic = g_native[k], fd = fd, rel_err = rel))
    end
    pass = max_rel < 1e-4
    results[(family, label, n_coords)] = pass
    lp("  [", family, " @ ", label, "] A-block coords ", coords, ": max_rel_err=", max_rel, "  ", pass ? "PASS" : "FAIL")
    return pass
end

# ---------------------------------------------------------------------------
# unrestricted
# ---------------------------------------------------------------------------
lp("\n" * "="^80); lp("unrestricted"); lp("="^80)
ev_u0 = evaluate_profiled_point(w_calib, ctx, spec, pe)
ufctx = build_unrestricted_family_ctx(ctx, spec, pe, ev_u0)
check_ablock!(:unrestricted, "P0_calibration", w_calib, ufctx, ev_u0, 3)
ev_u1 = evaluate_profiled_point(w_p1, ctx, spec, pe)
check_ablock!(:unrestricted, "P1_perturbed", w_p1, ufctx, ev_u1, 3)

# ---------------------------------------------------------------------------
# flexible_cm
# ---------------------------------------------------------------------------
lp("\n" * "="^80); lp("flexible_cm"); lp("="^80)
aug_cm = build_cm_augmented_obj_archB(ctx, CS; L = 10, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
cctx_cm = build_cm_bin_ctx(ctx, aug_cm; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = false)
fctx_cm = build_flexcm_family_ctx(ctx, spec, pe, layout, cctx_cm)
ev_cm0 = evaluate_profiled_flexcm_point(w_calib, fctx_cm)
check_ablock!(:flexible_cm, "P0_calibration", w_calib, fctx_cm, ev_cm0, 3)
ev_cm1 = evaluate_profiled_flexcm_point(w_p1, fctx_cm)
check_ablock!(:flexible_cm, "P1_perturbed", w_p1, fctx_cm, ev_cm1, 3)

# ---------------------------------------------------------------------------
# common_frechet
# ---------------------------------------------------------------------------
lp("\n" * "="^80); lp("common_frechet"); lp("="^80)
aug_f = build_cm_frechet_augmented_obj_archB(ctx, CS; L = 10, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
cctx_f = build_cm_bin_ctx(ctx, aug_f; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = false,
    core_hessian_backend = :dense_reference, cm_cross_hessian_backend = :winner_bin)
fctx_f = build_frechet_family_ctx(ctx, spec, pe, layout, cctx_f, aug_f.level_targets)
ev_f0 = evaluate_profiled_frechet_point(w_calib, fctx_f)
check_ablock!(:common_frechet, "P0_calibration", w_calib, fctx_f, ev_f0, 3)
ev_f1 = evaluate_profiled_frechet_point(w_p1, fctx_f)
check_ablock!(:common_frechet, "P1_perturbed", w_p1, fctx_f, ev_f1, 3)

# ---------------------------------------------------------------------------
# origin_ZC (fixed-nu evaluator -- A-block only, eta_nu direction already covered elsewhere)
# ---------------------------------------------------------------------------
lp("\n" * "="^80); lp("origin_ZC"); lp("="^80)
layout_o = OriginByPowerLayout(D, 1, 0)
νvec0 = fill(1.0, D)
aug_oz = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
octx_oz = build_originzc_core_hess_ctx(aug_oz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
fctx_oz = build_originzc_family_ctx(ctx, spec, pe, layout, aug_oz)
pes_oz = OriginZCPointEvalState(octx_oz, νvec0)
ev_oz0 = evaluate_profiled_originzc_point(w_calib, fctx_oz, pes_oz)
check_ablock!(:origin_zc, "P0_calibration", w_calib, fctx_oz, ev_oz0, 3)
ev_oz1 = evaluate_profiled_originzc_point(w_p1, fctx_oz, pes_oz)
check_ablock!(:origin_zc, "P1_perturbed", w_p1, fctx_oz, ev_oz1, 3)

# ---------------------------------------------------------------------------
# cm_meanzc (fixed-nu evaluator -- A-block only)
# ---------------------------------------------------------------------------
lp("\n" * "="^80); lp("cm_meanzc"); lp("="^80)
K_MEAN, K_PAIR = 1, 0
νvec0_z = fill(1.0, max(K_MEAN, 1))
aug_z = build_cm_meanzc_augmented_obj(ctx, CS; L = 10, K_mean = K_MEAN, K_pair = K_PAIR, base_obj = reduced_obj0, profiled_layout = layout)
cctx_z = build_cm_meanzc_bin_ctx(ctx, aug_z; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = false, inner_fg_backend = :dense_reference, profiled_layout = layout)
bins_u32 = cctx_z.Bidx isa Matrix{UInt32} ? cctx_z.Bidx : Matrix{UInt32}(cctx_z.Bidx)
fctx_z = build_cmzc_family_ctx(ctx, spec, pe, layout, aug_z, cctx_z, bins_u32)
pes_z = CMZCPointEvalState(cctx_z, νvec0_z)
ev_z0 = evaluate_profiled_cmzc_point(w_calib, fctx_z, pes_z)
check_ablock!(:cm_meanzc, "P0_calibration", w_calib, fctx_z, ev_z0, 3)
ev_z1 = evaluate_profiled_cmzc_point(w_p1, fctx_z, pes_z)
check_ablock!(:cm_meanzc, "P1_perturbed", w_p1, fctx_z, ev_z1, 3)

lp("\n" * "="^80)
lp("ALL_RESULTS: ", results)
all_pass = all(values(results))
lp("DECODED_STATE_GRADIENT_AB_ALLFAMILIES (D4, A-block, 2 points x 3 coords x 5 families): ", all_pass ? "PASS" : "FAIL")
lp("worst rows (top 5 by rel_err):")
sorted_rows = sort(summary_rows, by = r -> -r.rel_err)
for r in sorted_rows[1:min(5, length(sorted_rows))]
    lp("  ", r)
end
all_pass || error("test_decoded_state_gradient_ab_allfamilies_2026-08-04: one or more checks failed -- see ALL_RESULTS above")
