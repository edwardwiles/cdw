# ============================================================================
# Regression tests for D20Checkpoint schema-3 (c10_d20_production_driver.jl):
# round-trip save/load, the knitro_version additive field, the schema
# fail-fast on mismatch, and guard_checkpoint_path's design/checksum guard.
# Backfilled for the final-production-merge consolidation (brief §6): no
# dedicated checkpoint-schema test existed prior to this file. Loads the full
# production driver (needs KNITRO to load, but builds no real D=20 context --
# no inner/outer solve happens in this file).
# ============================================================================
using Test, Serialization

include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))

function dummy_checkpoint(; schema::Int = CHECKPOINT_SCHEMA, knitro_version::String = LOADED_KNITRO_RELEASE)
    D = 3
    return D20Checkpoint(schema, "test_run", "test_label", :upper, true, 2.0, 80000, 20260719,
        0.91, [0.1, -0.2, 0.05], zeros(D, D), [0.0, 0.0], Dict{Int,Float64}(1 => 0.02),
        nothing, 5, 3, 12.3, :iteration, (pairwise = 0, witness = 0, winner = 0, envelope = 0,
        winning_range = 0, safety_net = 0, passed = 5),
        0.01, -3.2, 1e-8, 1e-7, "test solver-state note",
        :pseudorandom, "checksum_u_abc", "checksum_t_def", knitro_version)
end

@testset "D20Checkpoint schema-3 round-trip" begin
    mktempdir() do dir
        path = joinpath(dir, "ckpt.jls")
        ckpt = dummy_checkpoint()

        @testset "save/load preserves every field" begin
            save_checkpoint(path, ckpt)
            loaded = load_checkpoint(path)
            for f in fieldnames(D20Checkpoint)
                @test getfield(loaded, f) == getfield(ckpt, f)
            end
        end

        @testset "knitro_version is the schema-3 additive field, matches the active release" begin
            loaded = load_checkpoint(path)
            @test loaded.schema == 3
            @test loaded.knitro_version == LOADED_KNITRO_RELEASE
            @test loaded.knitro_version isa String
            @test !isempty(loaded.knitro_version)
        end

        @testset "atomic write: no .tmp file left behind after a successful save" begin
            @test !isfile(path * ".tmp")
        end

        @testset "load_checkpoint hard-errors on schema mismatch (pre-schema-3 checkpoint)" begin
            stale = dummy_checkpoint(schema = 2)
            stale_path = joinpath(dir, "stale.jls")
            serialize(stale_path, stale)   # raw serialize, bypassing save_checkpoint's schema, to simulate a real old file
            @test_throws ErrorException load_checkpoint(stale_path)
        end
    end

    @testset "guard_checkpoint_path" begin
        mktempdir() do dir
            path = joinpath(dir, "ckpt.jls")
            ckpt = dummy_checkpoint()
            save_checkpoint(path, ckpt)

            @testset "same draw_design/checksums: no error (safe overwrite)" begin
                @test guard_checkpoint_path(path, :pseudorandom, "checksum_u_abc", "checksum_t_def") === nothing
            end

            @testset "different draw_design: hard error, refuses to overwrite" begin
                @test_throws ErrorException guard_checkpoint_path(path, :halton_scrambled, "checksum_u_abc", "checksum_t_def")
            end

            @testset "different checksum, same design: hard error" begin
                @test_throws ErrorException guard_checkpoint_path(path, :pseudorandom, "DIFFERENT", "checksum_t_def")
            end

            @testset "no file yet at path: no error (nothing to guard against)" begin
                fresh_path = joinpath(dir, "does_not_exist.jls")
                @test guard_checkpoint_path(fresh_path, :pseudorandom, "checksum_u_abc", "checksum_t_def") === nothing
            end
        end
    end
end

println("All checkpoint-schema tests passed.")
