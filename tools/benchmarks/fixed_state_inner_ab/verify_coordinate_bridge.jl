# Derisking check (no KNITRO): confirms full_coordinate_bridge.jl's `full_w0_from_state` really
# does land in each family's REAL production A_coordinate_mode by round-tripping through
# `full_decode_w0` and comparing against the REDUCED-decoded state it started from -- for all 5
# families, at the real calibration point, real D20.

const D4X = joinpath(dirname(dirname(dirname(@__DIR__))), "full_aod_diag", "d4_exact")

for f in ["context_real_d20.jl", "draw_design.jl",
          "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
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
          "c10_d20_production_driver.jl", "flexible_theta.jl", "flexible_theta_aspace_production.jl",
          "outer_coordinate_layout.jl", "c10_d20_production_driver_unified.jl",
          "cm_aspace_coordinate.jl"]
    isdefined(Main, :d4_exact_setup) && f == "context.jl" && continue
    include(joinpath(D4X, f))
end

include(joinpath(@__DIR__, "frozen_manifest.jl"))
using .FixedStateInnerABFrozenManifest
include(joinpath(@__DIR__, "full_coordinate_bridge.jl"))

using Printf

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

const CANONICAL_FAMILIES = (:unrestricted, :flexible_cm, :common_frechet, :origin_zc, :cm_meanzc)

sci = mode_a_scientific_manifest()
ctx = d20_real_setup_design(W = sci.W, δ = 1.0, find_smallest = true,
    draw_design = sci.draw_design, draw_seed = sci.draw_seed,
    destination_sample = sci.destination_sample,
    exclude_diagonal_gravity = sci.exclude_diagonal_gravity,
    gravity_exclude_cells = sci.gravity_exclude_cells, σHat = sci.sigma)

D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
θ0 = ctx.θ0_up
z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))
gp0 = θ0[3+D]
Aod0 = vec(exp.(z_calib))

println("="^90)
println("FULL coordinate bridge round-trip, real D20 calibration point, all 5 families")
println("="^90)

for family in CANONICAL_FAMILIES
    w0 = full_w0_from_state(family, gp0, z_calib, ctx)
    gp_back, Aod_back = full_decode_w0(family, w0, ctx)
    gp_err = abs(gp_back - gp0)
    A_rel_err = maximum(abs.(Aod_back .- Aod0) ./ max.(abs.(Aod0), 1e-12))
    check("$family (mode=$(CM_FAMILY_A_COORDINATE_MODE[family])): gp round-trips to 1e-9", gp_err < 1e-9)
    check("$family: full A round-trips to 1e-6 relative", A_rel_err < 1e-6)
    @printf("  %s: gp_err=%.3e  A_rel_err=%.3e  len(w0)=%d\n", family, gp_err, A_rel_err, length(w0))
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
exit(ALL_PASS[] ? 0 : 1)
