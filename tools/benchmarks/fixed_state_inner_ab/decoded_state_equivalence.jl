# Task "fixed-state FULL-vs-REDUCED inner A/B", step 3: decoded-state equivalence.
#
# Real D20 (not the pre-existing D4-only gate, test_profiled_coordinate_mode_roundtrip_2026-08-03.jl,
# which this script's checks are modeled on -- same primitives, same properties, D20/real-manifest
# scale instead of D4 synthetic). Builds ONE canonical decoded FULL economic state (calibration
# point: full A_od, gp, from ctx.θ0_up, via the frozen manifest's own real production fields),
# encodes it into REDUCED's native :profiled_pivot_anchor_relative coordinates
# (`reduce_to_w_profiled`), recovers the full FULL A matrix back out
# (`decode_outer_profiled`), and compares against the original FULL state -- NOT two raw vectors
# in different coordinate systems, but the same reconstructed economic object (full log A, full A,
# gravity residual, gp) both ways.
#
# Read-only with respect to gravity_elimination.jl / outer_coordinate_layout_profiled_2026-07-31.jl
# / relative_a_coordinate_2026-07-31.jl / gravity_pivot_on_retained_2026-07-31.jl / context_real_d20.jl
# / draw_design.jl -- only `include`d, never edited.

const D4X = joinpath(dirname(dirname(dirname(@__DIR__))), "full_aod_diag", "d4_exact")
const REPO_ROOT = dirname(dirname(dirname(@__DIR__)))

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
          "recover_full_a_2026-07-31.jl"]
    include(joinpath(D4X, f))
end

include(joinpath(@__DIR__, "frozen_manifest.jl"))
using .FixedStateInnerABFrozenManifest

include(joinpath(REPO_ROOT, "scientific_manifest", "RunManifest.jl"))
using .RunManifestMod: digest_economic_state

using Printf, TOML

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

"""
    run_decoded_state_equivalence(; mode::Symbol) -> NamedTuple

`mode` in `(:mode_a, :mode_b)` -- selects the frozen ScientificManifest (W=20_000/threads=1 vs
W=100_000/production threads). Builds ctx via `d20_real_setup_design` with EVERY frozen-manifest
field passed explicitly (no function default relied upon), extracts the real calibration point
from `ctx.θ0_up`, round-trips it through REDUCED's native pivot-relative coordinates, and checks
the properties task section 3 requires. Returns a NamedTuple recording every comparison plus a
`decoded_state_digest` shared reference for both arms to cite.
"""
function run_decoded_state_equivalence(; mode::Symbol)
    sci = mode === :mode_a ? mode_a_scientific_manifest() :
          mode === :mode_b ? mode_b_scientific_manifest() :
          error("run_decoded_state_equivalence: mode must be :mode_a or :mode_b, got :$mode")

    println("="^90)
    println("Decoded-state equivalence (FULL calibration <-> REDUCED :profiled_pivot_anchor_relative), ",
             "real D20, mode=$mode, W=$(sci.W)")
    println("="^90)

    ctx = d20_real_setup_design(W = sci.W, δ = 1.0, find_smallest = true,
        draw_design = sci.draw_design, draw_seed = sci.draw_seed,
        destination_sample = sci.destination_sample,
        exclude_diagonal_gravity = sci.exclude_diagonal_gravity,
        gravity_exclude_cells = sci.gravity_exclude_cells,
        σHat = sci.sigma)

    D = ctx.D
    Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    θ0 = ctx.θ0_up
    z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))
    gp0 = θ0[3+D]

    spec = build_anchor_spec_from_ctx(ctx)
    gauge = build_anchor_gauge(z_calib, spec)
    pe = build_pivot_elimination_on_retained(ctx, spec, gauge)

    # FULL -> REDUCED encode
    w = reduce_to_w_profiled(gp0, z_calib, pe)
    check("outer_dim_profiled(pe) matches length(w)", length(w) == outer_dim_profiled(pe))

    # REDUCED -> FULL recover
    dec = decode_outer_profiled(w, ctx, pe)

    check("gp: decoded == original (exact)", dec.gp == gp0)
    z_abs_err = maximum(abs.(dec.z_full .- z_calib))
    z_rel_err = maximum(abs.(dec.z_full .- z_calib) ./ max.(abs.(z_calib), 1e-12))
    check("full log A: decoded matches original to 1e-9 absolute", z_abs_err < 1e-9)
    @printf("  max|logA_decoded - logA_original| = %.3e ; max relative = %.3e\n", z_abs_err, z_rel_err)

    Aod_calib = vec(exp.(z_calib))
    Aod_rel_err = maximum(abs.(dec.Aod_levels .- Aod_calib) ./ max.(abs.(Aod_calib), 1e-12))
    check("full A: decoded matches original to 1e-8 relative", Aod_rel_err < 1e-8)
    @printf("  max relative |A_decoded - A_original| = %.3e (real A spans ~%.1f orders of magnitude here)\n",
        Aod_rel_err, log10(maximum(Aod_calib) / max(minimum(Aod_calib), 1e-300)))

    g_calib = gravity_from_logz(z_calib, ctx)
    g_decoded = gravity_from_logz(dec.z_full, ctx)
    check("gravity residual at original calibration point ~0", abs(g_calib) < 1e-6)
    check("gravity residual at decoded point ~0", abs(g_decoded) < 1e-6)
    check("gravity residual matches between original and decoded", abs(g_calib - g_decoded) < 1e-6)
    @printf("  gravity residual: original=%.3e decoded=%.3e\n", g_calib, g_decoded)

    # ONE digest, computed from the canonical original FULL state -- this is the shared reference
    # both arms' RunManifest.initial_state_digest must cite (task section 3: "Record a
    # decoded-state digest shared by both arms"). NOT re-derived from `dec` and compared for exact
    # equality: `digest_economic_state` is an exact byte-level digest by design (its own docstring:
    # "two states that print identically digest identically, full stop"), and the round trip above
    # already legitimately introduces ~1e-15 floating-point noise (log/exp/encode/decode) -- an
    # exact-digest comparison would fail on that noise even though every real numerical check above
    # passes at machine precision. Re-deriving a second digest from `dec` and requiring bit-exact
    # equality would be over-strict, not a real equivalence requirement task section 3 asks for.
    decoded_state_digest = digest_economic_state(Float64[gp0], z_calib, Float64[])

    return (mode = mode, W = sci.W, D = D, Ddest = Ddest,
        gp0 = gp0, z_abs_err = z_abs_err, z_rel_err = z_rel_err, Aod_rel_err = Aod_rel_err,
        g_calib = g_calib, g_decoded = g_decoded,
        decoded_state_digest = decoded_state_digest, all_pass = ALL_PASS[])
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = length(ARGS) >= 1 ? Symbol(ARGS[1]) : error("usage: julia decoded_state_equivalence.jl <mode_a|mode_b>")
    result = run_decoded_state_equivalence(mode = mode)
    println()
    println(result.all_pass ? "ALL PASS" : "SOME FAILURES")
    println("decoded_state_digest = ", result.decoded_state_digest)
    exit(result.all_pass ? 0 : 1)
end
