# Checkpoint-schema gates for the origin-specific-ZC integration (2026-07-23):
# CMCheckpointV5's schema-4 auto-upgrade (distribution_restriction=:unrestricted,
# origin_K_mean=origin_K_pair=0, power_target_layout inferred from the old
# cm_extension, origin_D=0) and config-level resolve/validate refusal logic.
# Pure Julia struct/logic checks -- no KNITRO solve, matching
# test_cm_meanzc_checkpoint_schema.jl's own convention.
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
include(joinpath(@__DIR__, "cm_originzc_target_layout.jl"))
include(joinpath(@__DIR__, "cm_originzc_config.jl"))
include(joinpath(@__DIR__, "cm_originzc_checkpoint.jl"))
using Test, Serialization

tmpdir = mktempdir()

println("="^100)
println("Checkpoint schema: V4 (cm_only, no origin-ZC) auto-upgrades to V5")
println("="^100)
@testset "schema-4 -> schema-5 upgrade, cm_only case" begin
    ck4 = CMCheckpointV4(4, "old_run", "old_label", :cm_upper, true, 1.0, 100, 1, :sobol_randomized,
        "csum_u", "csum_t", 10, [0.1, 0.5, 0.9], :orthonormal, :equal, :cumulative, :structured, :cplus,
        :cm_only, 0, 0, :direct, MEANZC_MOMENT_LAYOUT_VERSION,
        0.5, [0.1, 0.2], Float64[], zeros(2, 2), Float64[], Dict{Int,Float64}(), nothing, 5, 3, 1.0, 2.0, :new_best, "13.0.1")
    path4 = joinpath(tmpdir, "schema4_cmonly_test.jls")
    serialize(path4, ck4)
    loaded = load_cm_checkpoint_v5(path4)
    @test loaded isa CMCheckpointV5
    @test loaded.schema == 4
    @test loaded.distribution_restriction == :unrestricted
    @test loaded.origin_K_mean == 0
    @test loaded.origin_K_pair == 0
    @test loaded.power_target_layout == :none
    @test loaded.origin_D == 0
end
println()

println("="^100)
println("Checkpoint schema: V4 (meanzc shared-nu arm) auto-upgrades to V5 with power_target_layout=:shared_by_power")
println("="^100)
@testset "schema-4 -> schema-5 upgrade, meanzc case" begin
    ck4 = CMCheckpointV4(4, "old_run2", "old_label2", :cm_upper, true, 1.0, 100, 1, :sobol_randomized,
        "csum_u2", "csum_t2", 10, [0.1, 0.5, 0.9], :orthonormal, :equal, :cumulative, :structured, :cplus,
        :cm_plus_equal_means_zero_covariance, 1, 1, :direct, MEANZC_MOMENT_LAYOUT_VERSION,
        0.5, [0.1, 0.2], [0.038], zeros(2, 2), Float64[], Dict{Int,Float64}(), nothing, 5, 3, 1.0, 2.0, :new_best, "13.0.1")
    path4 = joinpath(tmpdir, "schema4_meanzc_test.jls")
    serialize(path4, ck4)
    loaded = load_cm_checkpoint_v5(path4)
    @test loaded isa CMCheckpointV5
    @test loaded.distribution_restriction == :unrestricted   # this WAS a CM-family run, not origin-family
    @test loaded.power_target_layout == :shared_by_power      # CORRECT inference: every V4 meanzc arm shared one nu_k
    @test loaded.cm_extension == :cm_plus_equal_means_zero_covariance
    @test loaded.eta_nu == [0.038]
end
println()

println("="^100)
println("Checkpoint schema: origin-ZC CMCheckpointV5 round-trips through save/load unchanged")
println("="^100)
@testset "origin-ZC CMCheckpointV5 round-trip" begin
    D = 20; K_mean = 2; K_pair = 2
    eta0 = randn(K_mean * D)
    ck5 = CMCheckpointV5(5, "originzc_run", "originzc_label", :cm_upper, true, 1.0, 80000, 20260719,
        :sobol_randomized, "csum_u5", "csum_t5", 0, Float64[], :anchored, :equal, :cumulative, :dense_reference, :cplus,
        :cm_only, 0, 0, :direct, 0,
        :origin_specific_moments_zero_covariance, K_mean, K_pair, :origin_by_power, D, ORIGINZC_MOMENT_LAYOUT_VERSION,
        0.7, randn(23), eta0, zeros(20, 20), Float64[], Dict{Int,Float64}(),
        nothing, 12, 8, 100.0, 3500.0, :new_best, "13.0.1")
    path5 = joinpath(tmpdir, "originzc_v5_test.jls")
    save_cm_checkpoint(path5, ck5)
    loaded = load_cm_checkpoint_v5(path5)
    @test loaded.schema == 5
    @test loaded.distribution_restriction == :origin_specific_moments_zero_covariance
    @test loaded.origin_K_mean == K_mean
    @test loaded.origin_K_pair == K_pair
    @test loaded.power_target_layout == :origin_by_power
    @test loaded.origin_D == D
    @test loaded.eta_nu == eta0
end
println()

println("="^100)
println("Checkpoint schema: mixing cm_extension and distribution_restriction is refused")
println("="^100)
@testset "save_cm_checkpoint refuses a checkpoint with BOTH families active" begin
    D = 4
    ck5_bad = CMCheckpointV5(5, "bad_run", "bad_label", :cm_upper, true, 1.0, 100, 1, :sobol_randomized,
        "csum_u", "csum_t", 10, [0.1, 0.5, 0.9], :orthonormal, :equal, :cumulative, :structured, :cplus,
        :cm_plus_equal_means, 1, 0, :direct, MEANZC_MOMENT_LAYOUT_VERSION,   # cm_extension ACTIVE
        :origin_specific_moments, 1, 0, :origin_by_power, D, ORIGINZC_MOMENT_LAYOUT_VERSION,   # AND distribution_restriction ACTIVE
        0.5, [0.1], randn(4), zeros(2, 2), Float64[], Dict{Int,Float64}(), nothing, 5, 3, 1.0, 2.0, :new_best, "13.0.1")
    path_bad = joinpath(tmpdir, "mixed_families_test.jls")
    @test_throws ErrorException save_cm_checkpoint(path_bad, ck5_bad)
end
println()

println("="^100)
println("Config-level refusal: distribution_restriction/K/layout inconsistencies")
println("="^100)
@testset "OriginZCConfig refuses inconsistent configs" begin
    @test_throws ErrorException originzc_resolve_K(OriginZCConfig(distribution_restriction = :unrestricted, K_mean = 1))
    @test_throws ErrorException originzc_resolve_K(OriginZCConfig(distribution_restriction = :origin_specific_moments, K_mean = 0))
    @test_throws ErrorException originzc_resolve_K(OriginZCConfig(distribution_restriction = :origin_specific_moments, K_mean = 1, K_pair = 1))
    @test_throws ErrorException originzc_resolve_K(OriginZCConfig(distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = 1, K_pair = 0))
    @test_throws ErrorException originzc_resolve_K(OriginZCConfig(distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = 1, K_pair = 2))
    @test originzc_resolve_K(OriginZCConfig(distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = 2, K_pair = 2)) == (2, 2)
    # :anchored has no origin-by-power analog -- must be refused
    @test_throws ErrorException originzc_make_layout(
        OriginZCConfig(distribution_restriction = :origin_specific_moments, K_mean = 1, power_target_layout = :origin_by_power, meanzc_basis = :anchored), 4)
    # :anchored IS fine for :shared_by_power (delegates to existing meanzc validation)
    @test originzc_make_layout(
        OriginZCConfig(distribution_restriction = :origin_specific_moments, K_mean = 1, power_target_layout = :shared_by_power, meanzc_basis = :anchored), 4) isa SharedByPowerLayout
end
println()
println("ALL ORIGIN-ZC CHECKPOINT SCHEMA TESTS PASSED")
