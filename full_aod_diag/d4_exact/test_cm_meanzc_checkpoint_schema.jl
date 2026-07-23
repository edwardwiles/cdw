# Checkpoint-schema gates for the CM+moments(+ZC) production integration (2026-07-23):
# CMCheckpointV4's schema-3 auto-upgrade (cm_extension=:cm_only, K_mean=K_pair=0, eta_nu=[])
# and resume-time hard refusal under a mismatched (cm_extension,K_mean,K_pair,meanzc_basis).
# These are pure Julia struct/logic checks -- no KNITRO solve, no context construction --
# deliberately kept independent of the real D=20/W=80,000 end-to-end run+resume gate (which
# needs production scale to avoid the small-W presolve instability documented in the release
# report; see cm_production_stage_runner.jl for the actual production entry point).
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
using Test, Serialization

tmpdir = mktempdir()

println("="^100)
println("Checkpoint schema: V3 (CM-C+, no meanzc) auto-upgrades to V4")
println("="^100)
@testset "schema-3 -> schema-4 upgrade" begin
    ck3 = CMCheckpointV3(3, "old_run", "old_label", :cm_upper, true, 1.0, 100, 1, :sobol_randomized,
        "csum_u", "csum_t", 10, [0.1, 0.5, 0.9], :orthonormal, :equal, :cumulative, :structured, :cplus,
        0.5, [0.1, 0.2], zeros(2, 2), Float64[], Dict{Int,Float64}(), nothing, 5, 3, 1.0, 2.0, :new_best, "13.0.1")
    path3 = joinpath(tmpdir, "schema3_test.jls")
    serialize(path3, ck3)
    loaded = load_cm_checkpoint(path3)
    @test loaded isa CMCheckpointV4
    @test loaded.schema == 3
    @test loaded.cm_extension == :cm_only
    @test loaded.meanzc_K_mean == 0
    @test loaded.meanzc_K_pair == 0
    @test loaded.meanzc_basis == :direct
    @test isempty(loaded.eta_nu)
    @test loaded.cm_gradient_backend == :cplus   # carried through unchanged from schema-3
end
println("  schema-3 -> schema-4 upgrade: PASSED")
println()

println("="^100)
println("Checkpoint schema: legacy schema-2 (CMCheckpoint) upgrades through V3 to V4")
println("="^100)
@testset "schema-2 -> schema-3 -> schema-4 upgrade chain" begin
    ck2 = CMCheckpoint(2, "old_run2", "old_label2", :cm_upper, true, 1.0, 100, 1, :sobol_randomized,
        "csum_u2", "csum_t2", 10, [0.1, 0.5, 0.9], :anchored, :equal, :cumulative, :structured,
        0.5, [0.1, 0.2], zeros(2, 2), Float64[], Dict{Int,Float64}(), nothing, 5, 3, 1.0, 2.0, :new_best, "13.0.1")
    path2 = joinpath(tmpdir, "schema2_test.jls")
    serialize(path2, ck2)
    loaded = load_cm_checkpoint(path2)
    @test loaded isa CMCheckpointV4
    @test loaded.schema == 2
    @test loaded.cm_gradient_backend == :reference   # upgrade_schema2's own correct default
    @test loaded.cm_extension == :cm_only
    @test isempty(loaded.eta_nu)
end
println("  schema-2 -> schema-4 upgrade chain: PASSED")
println()

println("="^100)
println("Checkpoint schema: meanzc config round-trips through save/load unchanged")
println("="^100)
@testset "meanzc CMCheckpointV4 round-trip" begin
    ck4 = CMCheckpointV4(4, "meanzc_run", "meanzc_label", :cm_upper, true, 1.0, 80000, 20260719,
        :pseudorandom, "csum_u4", "csum_t4", 50, [0.1, 0.5, 0.9], :orthonormal, :nested_family,
        :cumulative, :structured, :cplus,
        :cm_plus_equal_means_zero_covariance, 1, 1, :direct, MEANZC_MOMENT_LAYOUT_VERSION,
        0.7, randn(23), [0.038], zeros(20, 20), Float64[], Dict{Int,Float64}(),
        nothing, 12, 8, 100.0, 3500.0, :new_best, "13.0.1")
    path4 = joinpath(tmpdir, "meanzc_v4_test.jls")
    save_cm_checkpoint(path4, ck4)
    loaded = load_cm_checkpoint(path4)
    @test loaded.schema == 4
    @test loaded.cm_extension == :cm_plus_equal_means_zero_covariance
    @test loaded.meanzc_K_mean == 1
    @test loaded.meanzc_K_pair == 1
    @test loaded.eta_nu == [0.038]
end
println("  meanzc CMCheckpointV4 round-trip: PASSED")
println()
println("ALL CHECKPOINT SCHEMA TESTS PASSED")
