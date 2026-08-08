# Bounded tests for multistart_seed_generator.jl (task reproducible-multistart-generator-2026-08-08,
# task section 23). Mirrors this repo's existing test-file convention: colocated with production
# source, self-`include`s its own dependency list, `using Test`, runnable standalone:
#   JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#     full_aod_diag/d4_exact/test_multistart_seed_generator.jl
#
# Section A tests are pure-logic (no ctx / no KNITRO). Section B builds ONE real D20 context
# (moderate W, production draw_design=:sobol_randomized -- NOT :pseudorandom, which is
# diagnostic-only and, confirmed live during this task's own development, can trip the exact
# pairwise screen at small W on pure sample-size grounds) and reuses it across every test that
# needs a real ctx, to pay the KNITRO/package-compile + context-build cost once.

_D4E = joinpath(@__DIR__)
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl", "lfix_cm_aware.jl",
          "cm_hessian_architectures.jl", "cm_production_bundle.jl", "compressed_factual_buffer_reuse.jl", "cm_screen_bridge.jl", "gradient_workspace.jl",
          "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl",
          "cm_outer_driver.jl", "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl",
          "cm_meanzc_cplus.jl", "cm_checkpoint.jl", "cm_originzc_target_layout.jl", "zc_restriction_operator_ragged.jl", "cm_aspace_coordinate.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_config.jl",
          "cm_originzc_checkpoint.jl", "direction_bounds.jl", "cm_frechet_hessian.jl", "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl", "country_resolve.jl",
          "multistart_seed_generator.jl"]
    include(joinpath(_D4E, f))
end
using Test, Random, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))

# ============================================================================
# Section A: pure-logic tests, no ctx / no KNITRO.
# ============================================================================

@testset "A_scale: empirical RMS never exceeds A_scale (task section 4/23)" begin
    rng = Random.Xoshiro(1)
    A_scale = 0.07
    n_A = 379
    worst = 0.0
    for _ in 1:200
        delta_a, radius = draw_A_perturbation(rng, n_A, A_scale)
        rms = sqrt(sum(abs2, delta_a) / n_A)
        @test rms <= A_scale + 1e-9
        @test isapprox(rms, radius; atol = 1e-9)
        worst = max(worst, rms)
    end
    @test worst > 0.0   # not degenerate (radius draws are not all ~0)
    # :fixed_radius always saturates at exactly A_scale
    delta_a2, radius2 = draw_A_perturbation(rng, n_A, A_scale; radius_mode = :fixed_radius)
    @test isapprox(radius2, A_scale; atol = 1e-12)
    @test isapprox(sqrt(sum(abs2, delta_a2) / n_A), A_scale; atol = 1e-9)
end

@testset "gp_scale: fraction always in [0, gp_scale], candidate always between calibration and target" begin
    rng = Random.Xoshiro(2)
    gp_cal, gp_target = 0.6, 0.3   # e.g. an :upper-direction interval, gp_target < gp_cal
    gp_scale = 0.4
    for _ in 1:200
        gp_c, frac = draw_gp_perturbation(rng, gp_cal, gp_target, gp_scale)
        @test 0.0 <= frac <= gp_scale
        lo, hi = minmax(gp_cal, gp_target)
        @test lo <= gp_c <= hi
    end
    @test_throws ErrorException draw_gp_perturbation(rng, gp_cal, gp_target, 0.0)
    @test_throws ErrorException draw_gp_perturbation(rng, gp_cal, gp_target, 1.5)
end

@testset "Attempt-based RNG: deterministic, attempt-id-keyed, not one shared mutable stream" begin
    seed = UInt64(0xABCDEF)
    digest = "manifest_digest_stub"
    k1a = attempt_rng_seed(seed, 7, digest)
    k1b = attempt_rng_seed(seed, 7, digest)
    @test k1a == k1b   # pure function of (seed, attempt_id, digest)
    k2 = attempt_rng_seed(seed, 8, digest)
    @test k1a != k2   # different attempt_id -> different key (whp)
    k3 = attempt_rng_seed(UInt64(0x111111), 7, digest)
    @test k1a != k3   # different master seed -> different key (whp)
    k4 = attempt_rng_seed(seed, 7, "different_digest")
    @test k1a != k4   # different manifest digest -> different key (whp)

    # Drawing from two independently-constructed RNGs at the SAME (seed,attempt_id,digest) gives
    # bit-identical draws -- this is the actual reproducibility contract (task section 7), not
    # just equal integer keys.
    rngA = attempt_rng(seed, 3, digest)
    rngB = attempt_rng(seed, 3, digest)
    @test rand(rngA, 10) == rand(rngB, 10)
end

@testset "seed_distance: zero for identical vectors, symmetric, standardized-combined formula" begin
    w1 = vcat(0.5, randn(Random.Xoshiro(4), 20))
    w2 = copy(w1)
    d = seed_distance(w1, w2)
    @test d.rms_A_distance == 0.0
    @test d.gp_distance == 0.0
    @test d.combined_standardized_distance == 0.0
    w3 = vcat(0.5 + 0.01, w1[2:end] .+ 0.02)
    d2 = seed_distance(w1, w3)
    d2b = seed_distance(w3, w1)
    @test isapprox(d2.combined_standardized_distance, d2b.combined_standardized_distance; atol = 1e-14)
    @test isapprox(d2.combined_standardized_distance, sqrt(d2.rms_A_distance^2 + d2.gp_distance^2); atol = 1e-12)
end

@testset "resolve_cm_probs: matches production nested-grid convention at validated L, not equal-spacing" begin
    p50 = resolve_cm_probs(50)
    @test length(p50) == 50
    @test issorted(p50)
    @test p50 == nested_grid_sequence([10, 20, 50])[50]
    equal50 = collect(range(1 / 50, 49 / 50, length = 50))
    @test p50 != equal50   # confirms this is genuinely the nested grid, not the equal-spacing fallback
    p37 = resolve_cm_probs(37)   # outside the validated {10,20,50} family -> documented equal-spacing fallback
    @test p37 == collect(range(1 / 37, 36 / 37, length = 37))
end

@testset "FamilySeedSpec constructors + production_five_family_seed_specs shape" begin
    s1 = origin_zc_family_spec(:U_MEAN3; K_mean = 3, K_pair = 0)
    @test s1.kind == :origin_zc && s1.K_mean == 3 && s1.K_pair == 0
    s2 = cm_zc_family_spec(:CM_MEAN3; K_mean = 3, K_pair = 0, L = 50)
    @test s2.kind == :cm_zc && s2.L == 50 && s2.probs !== nothing && length(s2.probs) == 50
    s3 = common_frechet_family_spec(:COMMON_FRECHET; L = 50)
    @test s3.kind == :common_frechet
end

@testset "json_scalar / csv-safe encoding: escapes and round-trips the shapes this file emits" begin
    @test json_scalar(nothing) == "null"
    @test json_scalar(true) == "true"
    @test json_scalar(3) == "3"
    @test json_scalar(3.5) == "3.5"
    @test json_scalar(:U_MEAN3) == "\"U_MEAN3\""
    @test json_scalar("a\"b\\c") == "\"a\\\"b\\\\c\""
    @test json_scalar([:a, :b]) == "[\"a\",\"b\"]"
end

# ============================================================================
# Section B: one real D20 ctx, reused across every test below that needs it.
# ============================================================================

lp("="^100)
lp("Section B: building ONE real D20 context (W=8000, sobol_randomized -- functional testing only, ")
lp("NOT the production W>=20000 scale; see the separate D20 W=20000 release smoke for scientific validation).")
lp("="^100)

const GRAV_TEST = default_gravity_exclude_cells_brazil_korea()
_ctx_test_raw = d20_real_setup_design(W = 8000, δ = 1.0, find_smallest = true, draw_design = :sobol_randomized,
    draw_seed = 20260808, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = GRAV_TEST, σHat = 3.0, inner_lower_limit = -10.0)
# Required once per outer-solve process, before any real family evaluation (see
# generate_multistart_seeds's own call to this) -- attached here too since several tests below
# call build_family/qualify_economic_point/evaluate_attempt directly, bypassing
# generate_multistart_seeds's own attach call.
const CTX_TEST = attach_compressed_factual_workspace(_ctx_test_raw, _ctx_test_raw.D, _ctx_test_raw.D_dest, _ctx_test_raw.W)
lp("ctx built: D=", CTX_TEST.D, " W=", CTX_TEST.W, " bi=", CTX_TEST.bi, " sigma=", CTX_TEST.σ)

@testset "gp directional bounds match ctx.bounds exactly, not a re-derived formula" begin
    gp_cal, gp_up_target = gp_calibration_and_target(CTX_TEST, :upper)
    @test gp_cal == CTX_TEST.θ0_up[3 + CTX_TEST.D]
    @test gp_up_target == CTX_TEST.bounds.γp_lo
    gp_cal2, gp_lo_target = gp_calibration_and_target(CTX_TEST, :lower)
    @test gp_cal2 == gp_cal
    @test gp_lo_target == CTX_TEST.bounds.γp_hi
    @test gp_up_target < gp_cal < gp_lo_target || gp_up_target <= gp_cal <= gp_lo_target
    @test_throws ErrorException gp_calibration_and_target(CTX_TEST, :sideways)
end

@testset "Gravity reconstruction: decode_w_econ round-trips exactly through the existing pivot machinery" begin
    geo = build_aspace_geometry(CTX_TEST)
    w_cal = cm_w0_from_calibration(CTX_TEST, geo.pe, :powered_aspace)
    x_free_cal = decode_w_econ(geo, w_cal)
    x_free_ref = CTX_TEST.θ0_up[CTX_TEST.free_idx]
    @test norm(x_free_cal .- x_free_ref) / max(1.0, norm(x_free_ref)) < 1e-8

    # A perturbed candidate's decode must be an EXACT pivot_expand reconstruction: re-reducing it
    # must give back exactly the a-space entries we perturbed (no manual pivot patching anywhere).
    rng = Random.Xoshiro(99)
    delta_a, _ = draw_A_perturbation(rng, geo.n_A, 0.03)
    w_pert = vcat(w_cal[1], w_cal[2:end] .+ delta_a)
    x_free_pert = decode_w_econ(geo, w_pert)
    logA_full_pert = log.(reshape(x_free_pert[2:end], CTX_TEST.D, CTX_TEST.D_dest))
    z_reduced_back = pivot_reduce(logA_full_pert, geo.pe)
    a_reduced_back = cm_a_from_z(z_reduced_back, geo.theta, geo.xy, geo.pe)
    @test norm(a_reduced_back .- w_pert[2:end]) / max(1.0, norm(w_pert[2:end])) < 1e-8
    @test precheck_candidate(x_free_pert)   # finite, positive levels, gp in (0,1]
end

@testset "Nu lift is deterministic, independent of RNG state, and correctly length-reduced by the focal omission" begin
    spec_oz = origin_zc_family_spec(:U_MEAN3; K_mean = 3, K_pair = 0)
    kstar = focal_kstar(CTX_TEST)
    @test kstar == Int(round(CTX_TEST.σ)) - 1

    Random.seed!(1)   # perturb the GLOBAL RNG state deliberately -- must not affect build_family at all
    fb1 = build_family(CTX_TEST, spec_oz)
    Random.seed!(999999)
    fb2 = build_family(CTX_TEST, spec_oz)
    @test fb1.nu0_active == fb2.nu0_active   # bit-identical regardless of global RNG state
    layout = OriginByPowerLayout(CTX_TEST.D, spec_oz.K_mean, spec_oz.K_pair)
    n_dense = n_eta(layout)
    expected_len = (1 <= kstar <= spec_oz.K_mean) ? n_dense - 1 : n_dense
    @test length(fb1.nu0_active) == expected_len
    @test fb1.aml !== nothing   # sigma=3 => kstar=2 <= K_mean=3, profiling SHOULD be active here

    spec_cz = cm_zc_family_spec(:CM_MEAN3; K_mean = 3, K_pair = 0, L = 50)
    fb3 = build_family(CTX_TEST, spec_cz)
    fb4 = build_family(CTX_TEST, spec_cz)
    @test fb3.nu0_active == fb4.nu0_active
end

@testset "qualify_economic_point / all-family intersection: cheap-first ordering short-circuits, no later families evaluated" begin
    geo = build_aspace_geometry(CTX_TEST)
    w_cal = cm_w0_from_calibration(CTX_TEST, geo.pe, :powered_aspace)
    x_free_cal = decode_w_econ(geo, w_cal)

    impossible = origin_zc_family_spec(:IMPOSSIBLE; K_mean = 3, K_pair = 0)
    never_reached = cm_zc_family_spec(:NEVER_REACHED; K_mean = 3, K_pair = 0, L = 50)
    fams, results, reason = qualify_economic_point(CTX_TEST, [impossible, never_reached], x_free_cal, -1.0e10; eval_id = 1)
    @test fams == [:IMPOSSIBLE]              # NEVER_REACHED never even attempted
    @test length(results) <= 1
    @test reason !== nothing
    lp("family-intersection short-circuit reason: ", reason, " (expect Delta_above_max_IMPOSSIBLE or verification_failure_IMPOSSIBLE or inner_failure_IMPOSSIBLE)")
end

@testset "Attempt cap: exits at max_attempts, returns fewer than M, does not loop forever" begin
    out = mktempdir()
    specs = [origin_zc_family_spec(:IMPOSSIBLE; K_mean = 3, K_pair = 0)]
    # delta_max must be > 0 (validated by generate_multistart_seeds); an absurdly tiny positive
    # threshold is practically unreachable (task CLAUDE.md: real Delta* is either small-finite,
    # roughly 2-10, or unbounded -- never near 1e-9) so this still guarantees rejection.
    res = generate_multistart_seeds(CTX_TEST; M = 5, direction = :upper, delta_max = 1.0e-9,
        family_specs = specs, W = 8000, rng_seed = UInt64(2026080801), A_scale = 0.02, gp_scale = 0.5,
        max_attempts = 3, include_calibration = false, min_seed_distance = 0.0, output_dir = out)
    @test res.n_attempted == 3
    @test res.n_accepted < 5
    @test res.stop_reason == :attempt_limit
    @test length(res.ledger) == 3
    @test isfile(joinpath(out, "attempts.csv"))
    @test isfile(joinpath(out, "attempts.jsonl"))
end

@testset "Reproducibility: identical inputs -> bit-identical candidate economic vectors/digests; different seed -> different" begin
    out1 = mktempdir(); out2 = mktempdir(); out3 = mktempdir()
    specs = [origin_zc_family_spec(:IMPOSSIBLE; K_mean = 3, K_pair = 0)]
    common = (M = 4, direction = :upper, delta_max = 1.0e-9, family_specs = specs, W = 8000,
              A_scale = 0.03, gp_scale = 0.4, max_attempts = 4, include_calibration = false, min_seed_distance = 0.0)
    res1 = generate_multistart_seeds(CTX_TEST; common..., rng_seed = UInt64(42), output_dir = out1)
    res2 = generate_multistart_seeds(CTX_TEST; common..., rng_seed = UInt64(42), output_dir = out2)
    res3 = generate_multistart_seeds(CTX_TEST; common..., rng_seed = UInt64(43), output_dir = out3)
    @test [r.economic_digest for r in res1.ledger] == [r.economic_digest for r in res2.ledger]
    @test [r.gp for r in res1.ledger] == [r.gp for r in res2.ledger]
    @test res1.manifest_digest == res2.manifest_digest
    @test [r.economic_digest for r in res1.ledger] != [r.economic_digest for r in res3.ledger]

    # Simulated-reordered-completion: evaluate_attempt is a PURE function of attempt_id (task
    # section 20) -- shuffling evaluation order and re-sorting by attempt_id before the sequential
    # accept/diversity pass must reproduce res1 exactly.
    geo = build_aspace_geometry(CTX_TEST)
    w_cal = cm_w0_from_calibration(CTX_TEST, geo.pe, :powered_aspace)
    gp_cal, gp_target = gp_calibration_and_target(CTX_TEST, :upper)
    manifest_digest = compute_manifest_digest(CTX_TEST, specs, 8000)
    shuffled_ids = shuffle(Random.Xoshiro(7), collect(1:4))
    out_of_order = Dict(i => evaluate_attempt(CTX_TEST, geo, w_cal, gp_cal, gp_target, specs, i, 1,
        UInt64(42), manifest_digest, 0.03, 0.4, -1.0e10) for i in shuffled_ids)
    in_order_digests = [out_of_order[i].economic_digest for i in 1:4]
    @test in_order_digests == [r.economic_digest for r in res1.ledger]
end

lp("="^100)
lp("ALL multistart_seed_generator.jl TESTS DONE")
lp("="^100)
