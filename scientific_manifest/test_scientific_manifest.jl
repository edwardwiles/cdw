# Tests for ScientificManifest -- TOML round-trip, validation, and the production config file.
# Standalone: does not depend on anything in full_aod_diag/, src/melitz/, or scripts/melitz_*.
# Run with: julia --project=. scientific_manifest/test_scientific_manifest.jl

using Test

include(joinpath(@__DIR__, "ScientificManifest.jl"))
using .ScientificManifestMod

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const PROD_CONFIG = joinpath(REPO_ROOT, "configs", "fullA_production_2026-08-03.toml")

@testset "ScientificManifest" begin

    @testset "TOML round trip" begin
        m = ScientificManifest(
            dataset_version = "test_dataset", dataset_checksum = "deadbeef",
            country_order = ["fra", "bra", "kor", "row"], focal_country = "fra",
            sigma = 3.0, exclude_diagonal_gravity = true,
            gravity_exclude_cells = [(2, 3)],
            destination_sample = :exclude_row, draw_design = :sobol_randomized,
            draw_seed = 42, W = 1000, L = 10, K_mean = 1, K_pair = 1,
            julia_threads = 4, blas_threads = 2,
            inner_opt_checksum = "abc123", outer_opt_checksum = "def456")

        mktempdir() do d
            path = joinpath(d, "test_manifest.toml")
            write_manifest_toml(path, m)
            m2 = read_manifest_toml(path)

            @test m2.schema_version == m.schema_version
            @test m2.dataset_version == m.dataset_version
            @test m2.dataset_checksum == m.dataset_checksum
            @test m2.country_order == m.country_order
            @test m2.focal_country == m.focal_country
            @test m2.sigma == m.sigma
            @test m2.exclude_diagonal_gravity == m.exclude_diagonal_gravity
            @test m2.gravity_exclude_cells == m.gravity_exclude_cells
            @test m2.destination_sample == m.destination_sample
            @test m2.draw_design == m.draw_design
            @test m2.draw_seed == m.draw_seed
            @test m2.W == m.W
            @test m2.L == m.L
            @test m2.K_mean == m.K_mean
            @test m2.K_pair == m.K_pair
            @test m2.julia_threads == m.julia_threads
            @test m2.blas_threads == m.blas_threads
            @test m2.inner_opt_checksum == m.inner_opt_checksum
            @test m2.outer_opt_checksum == m.outer_opt_checksum
        end
    end

    @testset "from_toml_dict errors on missing required key" begin
        d = Dict{String,Any}("schema_version" => 1, "dataset_version" => "x")
        @test_throws ErrorException from_toml_dict(d)
    end

    @testset "validate_manifest: internal consistency checks" begin
        base = ScientificManifest(
            dataset_version = "d", dataset_checksum = "c",
            country_order = ["fra", "bra", "kor", "row"], focal_country = "fra",
            sigma = 3.0, exclude_diagonal_gravity = true,
            gravity_exclude_cells = [(2, 3)],
            destination_sample = :exclude_row, draw_design = :sobol_randomized,
            draw_seed = 1, W = 100, L = 10, K_mean = 1, K_pair = 1,
            julia_threads = 1, blas_threads = 1,
            inner_opt_checksum = "i", outer_opt_checksum = "o")
        @test isempty(validate_manifest(base))

        bad_focal = ScientificManifest(
            dataset_version = "d", dataset_checksum = "c",
            country_order = ["fra", "bra", "kor", "row"], focal_country = "NOT_A_COUNTRY",
            sigma = 3.0, exclude_diagonal_gravity = false, gravity_exclude_cells = Tuple{Int,Int}[],
            destination_sample = :exclude_row, draw_design = :sobol_randomized,
            draw_seed = 1, W = 100, L = 10, K_mean = 1, K_pair = 1,
            julia_threads = 1, blas_threads = 1, inner_opt_checksum = "i", outer_opt_checksum = "o")
        problems = validate_manifest(bad_focal)
        @test any(occursin("focal_country", p) for p in problems)

        bad_sample = ScientificManifest(
            dataset_version = "d", dataset_checksum = "c",
            country_order = ["fra", "bra", "kor", "row"], focal_country = "fra",
            sigma = 3.0, exclude_diagonal_gravity = false, gravity_exclude_cells = [(2, 3)],
            destination_sample = :all_legacy, draw_design = :sobol_randomized,
            draw_seed = 1, W = 100, L = 10, K_mean = 1, K_pair = 1,
            julia_threads = 1, blas_threads = 1, inner_opt_checksum = "i", outer_opt_checksum = "o")
        problems2 = validate_manifest(bad_sample)
        @test any(occursin("gravity_exclude_cells", p) for p in problems2)
    end

    @testset "sha256_of_directory is deterministic and order-independent" begin
        mktempdir() do d
            mkpath(joinpath(d, "sub"))
            write(joinpath(d, "a.txt"), "hello")
            write(joinpath(d, "sub", "b.txt"), "world")
            h1 = sha256_of_directory(d)
            h2 = sha256_of_directory(d)
            @test h1 == h2
        end
    end

    if isfile(PROD_CONFIG)
        @testset "production config file: loads, validates against real data" begin
            m = read_manifest_toml(PROD_CONFIG)
            @test m.focal_country == "fra"
            @test m.sigma == 3.0
            @test m.destination_sample == :exclude_row
            @test m.gravity_exclude_cells == [(3, 14)]

            data_dir = joinpath(REPO_ROOT, "real_data", "noah_D20")
            inner_opt = joinpath(REPO_ROOT, "full_aod_diag", "ek_inner.opt")
            outer_opt = joinpath(REPO_ROOT, "full_aod_diag", "csw_outer_25.opt")
            if isdir(data_dir) && isfile(inner_opt) && isfile(outer_opt)
                problems = validate_manifest(m; data_dir = data_dir,
                    opt_dir_inner = inner_opt, opt_dir_outer = outer_opt)
                @test isempty(problems)
            end
        end
    end

end
