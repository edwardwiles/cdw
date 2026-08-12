# Wiring/plumbing gate for the CM+ZC-CROSS + OZC-CROSS campaign integration (2026-08-09).
# Solver-free by design: everything here is config resolution, checkpoint schema round-tripping,
# and reproducibility-digest behavior -- the parts that are cheap to get wrong and expensive to
# discover 40 minutes into a real campaign. The scientific correctness of the families themselves is
# gated separately (smoke_cmzc_cross_d4 / verify_cmzc_cross_gradient_d4 / verify_cmzc_cross_cplus_ab_d4).
#
# Usage: julia --project=. -t 2 full_aod_diag/d4_exact/test_cmzc_cross_wiring_2026-08-09.jl
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "compressed_moments.jl", "structured_moment_build.jl",
          "compressed_cc_inner.jl", "compressed_live.jl", "compressed_factual_buffer_reuse.jl",
          "core_exact_hessian.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_threaded.jl", "winner_pair_cross_hessian.jl",
          "no_dense_g_counters.jl", "zc_restriction_operator.jl", "zc_restriction_operator_ragged.jl",
          "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "hcz_reordered_candidate_2026-08-01.jl",
          "hez_drawmajor_candidate_2026-08-01.jl", "hez_drawmajor_v2_candidate_2026-08-01.jl",
          "operator_hessian_weights.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "shared_a_gradient.jl", "lfix_factorized.jl",
          "lfix_factorized_workspace.jl", "lfix_base_workspace_pooled.jl", "lfix_cm_cplus.jl",
          "nested_quantile_grids.jl", "cm_aspace_coordinate.jl", "draw_design.jl", "autarky_cf.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl",
          "cm_meanzc_cplus.jl", "cm_meanzc_lookup_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl",
          "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl",
          "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl",
          "cm_originzc_cross_production.jl", "cm_originzc_cross_cplus.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl",
          "cm_meanzc_cross_production.jl", "cm_meanzc_cross_cplus.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl", "production_backend_manifest.jl",
          # The reproducible multistart-seed campaign (the OTHER thing called "five family").
          # Included last: it hard-checks that every dependency above is already defined, which
          # doubles as a load-order gate for the four CROSS symbols added to that check on
          # 2026-08-09.
          "country_resolve.jl", "fast_range_screen.jl",
          "multistart_seed_generator.jl"]
    include(joinpath(D4X, f))
end
using Printf, Serialization, SHA, Random, LinearAlgebra

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS
    ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end
function check_throws(name::AbstractString, f)
    threw = false
    try; f(); catch; threw = true; end
    check(name, threw)
end

println("="^96)
println("1. CMMeanZCConfig / meanzc_make_layout")
println("="^96)

const _CMCFG = CMConfig(cm_moment_families = 2)   # CMConfig requires cm_moment_families (no default, repo rule); the real driver supplies it the same way
cfg_diag  = CMMeanZCConfig(cm = _CMCFG, cm_extension = :cm_plus_moments, meanzc_K_mean = 3, meanzc_K_pair = 3)
cfg_cross = CMMeanZCConfig(cm = _CMCFG, cm_extension = :cm_plus_moments, meanzc_K_mean = 3, meanzc_K_pair = 3,
                            meanzc_target_layout = :shared_by_power_cross)
check("default meanzc_target_layout is :shared_by_power (zero behavior change for existing callers)",
      cfg_diag.meanzc_target_layout === :shared_by_power)
check("meanzc_make_layout(:shared_by_power) -> SharedByPowerLayout",
      meanzc_make_layout(cfg_diag, 20) isa SharedByPowerLayout)
check("meanzc_make_layout(:shared_by_power_cross) -> SharedByPowerCrossLayout",
      meanzc_make_layout(cfg_cross, 20) isa SharedByPowerCrossLayout)
check("cross layout n_eta == K_mean (NO new outer parameters)", n_eta(meanzc_make_layout(cfg_cross, 20)) == 3)
check("cross layout n_eta == diagonal layout n_eta", n_eta(meanzc_make_layout(cfg_cross, 20)) == n_eta(meanzc_make_layout(cfg_diag, 20)))
check("cross layout target_index ignores origin (shared) -> aml.dense_omit_idx == kstar",
      target_index(meanzc_make_layout(cfg_cross, 20), 7, 2) == 2)
check_throws("invalid meanzc_target_layout is rejected",
      () -> _meanzc_validate(CMMeanZCConfig(cm = _CMCFG, cm_extension = :cm_plus_moments, meanzc_K_mean = 2, meanzc_K_pair = 2,
                                             meanzc_target_layout = :not_a_layout)))
check_throws("cross layout with K_pair=0 is rejected (cross grid IS the pair block)",
      () -> _meanzc_validate(CMMeanZCConfig(cm = _CMCFG, cm_extension = :cm_plus_moments, meanzc_K_mean = 2, meanzc_K_pair = 0,
                                             meanzc_target_layout = :shared_by_power_cross)))

println()
println("="^96)
println("2. pair_targets dispatch: cross reduces to diagonal exactly at k1==k2")
println("="^96)
let D = 5, nu = [1.3, 2.7, 6.1]
    ldiag  = SharedByPowerLayout(3, 3)
    lcross = SharedByPowerCrossLayout(3, 3)
    levels = cross_pair_level_index(3)
    ok = true
    for (klin, (k1, k2)) in enumerate(levels)
        tc = pair_targets(lcross, nu, klin, D)
        expected = fill(nu[k1] * nu[k2], div(D * (D - 1), 2))
        ok &= (tc == expected)
        if k1 == k2
            ok &= (tc == pair_targets(ldiag, nu, k1, D))   # exact equality, not a tolerance
        end
    end
    check("cross pair_targets == nu_k1*nu_k2 everywhere, and == diagonal pair_targets when k1==k2", ok)
    check("cross grid has K_pair^2 = 9 level pairs", length(levels) == 9)
    check("nu_kstar (kstar=2) enters 2*K_pair-1 = 5 of them",
          count(kk -> kk[1] == 2 || kk[2] == 2, levels) == 5)
end

println()
println("="^96)
println("3. Checkpoint schema 11 round-trip + schema-10 upgrade")
println("="^96)
let tmpdir = mktempdir()
    mk(layout) = CMCheckpointV11(CM_CHECKPOINT_SCHEMA, "rid", "lbl", :cm_upper, true, 1.0, 100, 1, :sobol_randomized,
        "cu", "ct", 50, [0.5], :orthonormal, :nested, :cumulative, :structured, :cplus,
        :cm_plus_moments, 3, 3, :direct, MEANZC_MOMENT_LAYOUT_VERSION,
        0.97, [1.0, 2.0], [0.1, 0.2, 0.3], zeros(2, 2), [0.0], Dict{Int,Float64}(), nothing,
        1, 1, 1.0, 1.0, :new_best, "13.0.1", :exclude_row, 3, 19, :common_flexible, :powered_aspace,
        :cdf_plus_truncated_power_1msigma, 2, 1, "chk", layout)

    p = joinpath(tmpdir, "cross.jls")
    save_cm_checkpoint(p, mk(:shared_by_power_cross))
    r = load_cm_checkpoint(p)
    check("schema-11 checkpoint round-trips", r isa CMCheckpointV11 && r.schema == CM_CHECKPOINT_SCHEMA)
    check("meanzc_target_layout survives the round-trip", r.meanzc_target_layout === :shared_by_power_cross)
    check("every pre-existing field still round-trips (spot check)",
          r.meanzc_K_mean == 3 && r.cm_extension === :cm_plus_moments && r.destination_sample === :exclude_row &&
          r.A_coordinate_mode === :powered_aspace && r.cm_feature_family_count == 2)

    # A genuine schema-10 file (written by the PREVIOUS struct) must auto-upgrade to :shared_by_power.
    old = CMCheckpointV10(10, "rid", "lbl", :cm_upper, true, 1.0, 100, 1, :sobol_randomized,
        "cu", "ct", 50, [0.5], :orthonormal, :nested, :cumulative, :structured, :cplus,
        :cm_plus_moments, 3, 3, :direct, MEANZC_MOMENT_LAYOUT_VERSION,
        0.97, [1.0, 2.0], [0.1, 0.2, 0.3], zeros(2, 2), [0.0], Dict{Int,Float64}(), nothing,
        1, 1, 1.0, 1.0, :new_best, "13.0.1", :exclude_row, 3, 19, :common_flexible, :powered_aspace,
        :cdf_plus_truncated_power_1msigma, 2, 1, "chk")
    p10 = joinpath(tmpdir, "old10.jls")
    save_cm_checkpoint(p10, old)
    r10 = load_cm_checkpoint(p10)
    check("schema-10 file auto-upgrades to V11", r10 isa CMCheckpointV11)
    check("upgraded schema-10 file gets :shared_by_power (correct by provenance -- cross did not exist then)",
          r10.meanzc_target_layout === :shared_by_power)
    check("upgraded schema-10 file reports schema == CM_CHECKPOINT_SCHEMA", r10.schema == CM_CHECKPOINT_SCHEMA)
    check("upgraded schema-10 file preserves its payload", r10.g == 0.97 && r10.eta_nu == [0.1, 0.2, 0.3])
    rm(tmpdir; recursive = true, force = true)
end

println()
println("="^96)
println("4. Backend manifest family symbols")
println("="^96)
check("resolve_origin_zc_manifest default is :origin_zc",
      resolve_origin_zc_manifest(; octx = nothing, blas_threads = nothing).family === :origin_zc)
check("resolve_origin_zc_manifest(:origin_by_power_cross) is :origin_zc_cross",
      resolve_origin_zc_manifest(; octx = nothing, blas_threads = nothing,
                                   power_target_layout = :origin_by_power_cross).family === :origin_zc_cross)

println()
println("="^96)
println("5. Reproducibility digest: diagonal vs cross MUST differ (the collision the handover flagged)")
println("="^96)
let
    # A minimal stand-in ctx: compute_manifest_digest reads only these scalar/collection fields.
    fake_ctx = (σ = 3.0, bi = 7, μHat = 1 / 3, destination_sample = :exclude_row,
                exclude_diagonal_gravity = true, gravity_exclude_cells = Set(["BRA_KOR"]),
                draw_design = :sobol_randomized, draw_seed = 20260719,
                obj = (lower_limit = -10.0,))
    s_diag  = cm_zc_family_spec(:ZC_K3; K_mean = 3, K_pair = 3, L = 50)
    s_cross = cm_zc_cross_family_spec(:ZC_K3; K_mean = 3, K_pair = 3, L = 50)   # SAME id, same K, same L
    d_diag  = family_spec_descriptor(s_diag)
    d_cross = family_spec_descriptor(s_cross)
    println("    diagonal descriptor: ", d_diag)
    println("    cross    descriptor: ", d_cross)
    check("descriptors differ even at IDENTICAL id/K_mean/K_pair/L/contrasts/basis", d_diag != d_cross)
    g_diag  = compute_manifest_digest(fake_ctx, [s_diag], 100_000)
    g_cross = compute_manifest_digest(fake_ctx, [s_cross], 100_000)
    println("    diagonal digest: ", g_diag)
    println("    cross    digest: ", g_cross)
    check("manifest digests differ between a diagonal and a cross spec", g_diag != g_cross)

    o_diag  = origin_zc_family_spec(:OZ_K3; K_mean = 3, K_pair = 3)
    o_cross = origin_zc_cross_family_spec(:OZ_K3; K_mean = 3, K_pair = 3)
    check("origin-ZC diagonal vs cross digests also differ",
          compute_manifest_digest(fake_ctx, [o_diag], 100_000) != compute_manifest_digest(fake_ctx, [o_cross], 100_000))

    # The guard itself: two specs that ARE indistinguishable must be refused, not silently digested.
    check_throws("assert_distinct_family_descriptors refuses two identical specs",
                 () -> compute_manifest_digest(fake_ctx, [s_diag, cm_zc_family_spec(:ZC_K3; K_mean = 3, K_pair = 3, L = 50)], 100_000))

    # The five-family presets must each be internally distinct (this is what the guard protects).
    check("production_five_family_seed_specs digests cleanly (all 5 descriptors distinct)",
          (compute_manifest_digest(fake_ctx, production_five_family_seed_specs(fake_ctx), 100_000); true))
    check("production_five_family_cross_seed_specs digests cleanly (all 5 descriptors distinct)",
          (compute_manifest_digest(fake_ctx, production_five_family_cross_seed_specs(fake_ctx), 100_000); true))
    check("the two five-family presets have DIFFERENT overall digests",
          compute_manifest_digest(fake_ctx, production_five_family_seed_specs(fake_ctx), 100_000) !=
          compute_manifest_digest(fake_ctx, production_five_family_cross_seed_specs(fake_ctx), 100_000))

    check_throws("cm_zc_cross_family_spec refuses K_pair=1 (degenerate: cross grid == diagonal)",
                 () -> cm_zc_cross_family_spec(:X; K_mean = 3, K_pair = 1, L = 50))
    check_throws("origin_zc_cross_family_spec refuses K_pair=1",
                 () -> origin_zc_cross_family_spec(:X; K_mean = 3, K_pair = 1))
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILED")
exit(ALL_PASS[] ? 0 : 1)
