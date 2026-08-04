# profiled-outer-production-readiness-2026-08-03 (task 2), section 7 gate: formalizes
# REDUCED's existing (already-real, already-used) coordinate system as a named mode
# (:profiled_pivot_anchor_relative, FamilyRegistry.jl) and verifies, at D4, three of the four
# properties task section 7 requires: exact decode/encode round trip, same reconstructed full
# logA, and same gravity residual (both endpoints exactly gravity-feasible, by construction of
# gravity-pivot elimination). The fourth (correct analytic chain-rule gradient through this
# coordinate transform) is NOT verified here -- that needs the outer-gradient engine
# (shared_family_outer_gradient) exercised against finite differences, a section 8 concern, not
# duplicated in this file.
const D4X = @__DIR__
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
          "profiled_originzc_family_adapter_2026-08-02.jl",
          "profiled_cmzc_family_adapter_2026-08-02.jl",
          "profiled_zc_lane_point_evaluators_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
spec = build_anchor_spec_from_ctx(ctx)
θ0 = ctx.θ0_up
z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))
gauge = build_anchor_gauge(z_calib, spec)
pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
gp0 = θ0[3+D]

println("="^90); println(":profiled_pivot_anchor_relative round-trip gate, D4"); println("="^90)

# --- Property 1: exact decode/encode round trip ---
w = reduce_to_w_profiled(gp0, z_calib, pe)
check("outer_dim_profiled(pe) matches length(w)", length(w) == outer_dim_profiled(pe))
dec = decode_outer_profiled(w, ctx, pe)
check("round trip: decoded gp == original gp0 (exact, no transform on gp)", dec.gp == gp0)
z_err = maximum(abs.(dec.z_full .- z_calib))
check("round trip: decoded z_full matches z_calib to 1e-9 (encode -> reduce -> decode)", z_err < 1e-9)
@printf("round-trip max|z_full - z_calib| = %.3e\n", z_err)

# --- Property 2: same reconstructed full logA (Aod_levels = exp(z_full)) ---
Aod_calib = vec(exp.(z_calib))
Aod_err = maximum(abs.(dec.Aod_levels .- Aod_calib) ./ max.(abs.(Aod_calib), 1e-12))
check("round trip: decoded Aod_levels matches exp(z_calib) to 1e-8 relative", Aod_err < 1e-8)
@printf("round-trip max relative |Aod_levels - exp(z_calib)| = %.3e\n", Aod_err)

# --- Property 3: same gravity residual -- both endpoints exactly gravity-feasible by
# construction of gravity-pivot elimination (build_pivot_elimination's own docstring: "pivot
# solved so g_gravity(z)==0 exactly"; build_pivot_elimination_on_retained is the same mechanism
# layered on the anchor-relative-gauge-fixed retained coordinates). ---
g_calib = gravity_from_logz(z_calib, ctx)
g_decoded = gravity_from_logz(dec.z_full, ctx)
check("gravity residual at z_calib is ~0 (gravity-feasible calibration point)", abs(g_calib) < 1e-6)
check("gravity residual at decoded z_full is ~0 (gravity-pivot elimination invariant preserved through round trip)", abs(g_decoded) < 1e-6)
check("gravity residual matches between z_calib and decoded z_full", abs(g_calib - g_decoded) < 1e-6)
@printf("gravity residual: calib=%.3e decoded=%.3e\n", g_calib, g_decoded)

# --- A second, independent perturbed point (not just the calibration point) -- confirms the
# round trip and gravity-feasibility invariant hold generically, not only at one special point. ---
rng_w = copy(w); rng_w[2:end] .+= 0.01 .* sin.(1:length(rng_w)-1)   # deterministic, not random -- reproducible
dec2 = decode_outer_profiled(rng_w, ctx, pe)
w2 = reduce_to_w_profiled(dec2.gp, dec2.z_full, pe)
check("round trip at a perturbed (non-calibration) point: re-encoding the decode reproduces w to 1e-9", maximum(abs.(w2 .- rng_w)) < 1e-9)
check("gravity residual at the perturbed point is still ~0", abs(gravity_from_logz(dec2.z_full, ctx)) < 1e-6)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
exit(ALL_PASS[] ? 0 : 1)
