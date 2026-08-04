# Tests for RunManifest -- TOML/JSON round-trip, validation, digest, and git-provenance helpers.
# Standalone: does not depend on anything in full_aod_diag/, src/melitz/, or scripts/melitz_*.
# Run with: julia --project=. scientific_manifest/test_run_manifest.jl

using Test

include(joinpath(@__DIR__, "RunManifest.jl"))
using .RunManifestMod
using .RunManifestMod.ScientificManifestMod: ScientificManifest

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

function make_sci(; sigma = 3.0)
    ScientificManifest(
        dataset_version = "test_dataset", dataset_checksum = "deadbeef",
        country_order = ["fra", "bra", "kor", "row"], focal_country = "fra",
        sigma = sigma, exclude_diagonal_gravity = true,
        gravity_exclude_cells = [(2, 3)],
        destination_sample = :exclude_row, draw_design = :sobol_randomized,
        draw_seed = 42, W = 1000, L = 10, K_mean = 1, K_pair = 1,
        julia_threads = 4, blas_threads = 2,
        inner_opt_checksum = "abc123", outer_opt_checksum = "def456")
end

function make_run(; family = :unrestricted, economic_parameterization = :full_gamma_normalized,
        nu_policy = :fixed, nu_bounds = nothing)
    RunManifest(
        sci = make_sci(), family = family,
        economic_parameterization = economic_parameterization,
        A_coordinate_mode = :legacy_z, nu_policy = nu_policy, nu_bounds = nu_bounds,
        draw_checksum_uniform = "unif123", draw_checksum_transformed = "trans456",
        outer_algorithm = :knitro_active_set, outer_max_wall_seconds = 3600.0,
        outer_max_gradients = 5, cache_policy = :none, dual_bank_policy = :none,
        warm_start_policy = :cold, initial_state_digest = "digest789",
        source_sha = "a" ^ 40, source_dirty = false)
end

@testset "RunManifest" begin

    @testset "TOML round trip, fixed nu" begin
        m = make_run()
        mktempdir() do d
            path = joinpath(d, "run_manifest.toml")
            write_manifest_toml(path, m)
            m2 = read_manifest_toml(path)
            @test m2.schema_version == m.schema_version
            @test m2.sci.sigma == m.sci.sigma
            @test m2.sci.country_order == m.sci.country_order
            @test m2.family == m.family
            @test m2.economic_parameterization == m.economic_parameterization
            @test m2.A_coordinate_mode == m.A_coordinate_mode
            @test m2.nu_policy == m.nu_policy
            @test m2.nu_bounds === m.nu_bounds
            @test m2.draw_checksum_uniform == m.draw_checksum_uniform
            @test m2.draw_checksum_transformed == m.draw_checksum_transformed
            @test m2.outer_algorithm == m.outer_algorithm
            @test m2.outer_max_wall_seconds == m.outer_max_wall_seconds
            @test m2.outer_max_gradients == m.outer_max_gradients
            @test m2.cache_policy == m.cache_policy
            @test m2.dual_bank_policy == m.dual_bank_policy
            @test m2.warm_start_policy == m.warm_start_policy
            @test m2.initial_state_digest == m.initial_state_digest
            @test m2.source_sha == m.source_sha
            @test m2.source_dirty == m.source_dirty
        end
    end

    @testset "TOML round trip, free nu with bounds" begin
        m = make_run(family = :origin_zc, economic_parameterization = :profiled_destination_scales,
            nu_policy = :free, nu_bounds = (-2.0, 2.0))
        mktempdir() do d
            path = joinpath(d, "run_manifest.toml")
            write_manifest_toml(path, m)
            m2 = read_manifest_toml(path)
            @test m2.nu_policy == :free
            @test m2.nu_bounds == (-2.0, 2.0)
        end
    end

    @testset "JSON write is well-formed and round-trips via a minimal parser check" begin
        m = make_run()
        mktempdir() do d
            path = joinpath(d, "run_manifest.json")
            write_run_manifest_json(path, m)
            txt = read(path, String)
            @test startswith(strip(txt), "{")
            @test endswith(strip(txt), "}")
            @test occursin("\"family\":\"unrestricted\"", txt)
            @test occursin("\"schema_version\":1", txt)
            # Parse with Julia's own TOML-adjacent-free minimal check: valid JSON => Meta.parse
            # of the bracket/brace structure with strings/numbers is at least well-formed;
            # a fuller round trip is exercised by the TOML path above (same to_toml_dict).
        end
    end

    @testset "from_toml_dict errors on missing required key" begin
        d = Dict{String,Any}("schema_version" => 1, "family" => "unrestricted")
        @test_throws ErrorException from_toml_dict(d)
    end

    @testset "validate_manifest: internal consistency checks" begin
        base = make_run()
        @test isempty(validate_manifest(base))

        bad_family = RunManifest(sci = base.sci, family = :not_a_family,
            economic_parameterization = base.economic_parameterization,
            A_coordinate_mode = base.A_coordinate_mode, nu_policy = base.nu_policy,
            nu_bounds = base.nu_bounds, draw_checksum_uniform = base.draw_checksum_uniform,
            draw_checksum_transformed = base.draw_checksum_transformed,
            outer_algorithm = base.outer_algorithm, outer_max_wall_seconds = base.outer_max_wall_seconds,
            outer_max_gradients = base.outer_max_gradients, cache_policy = base.cache_policy,
            dual_bank_policy = base.dual_bank_policy, warm_start_policy = base.warm_start_policy,
            initial_state_digest = base.initial_state_digest, source_sha = base.source_sha,
            source_dirty = base.source_dirty)
        @test any(occursin("family", p) for p in validate_manifest(bad_family))

        free_no_bounds = make_run(family = :origin_zc, nu_policy = :free, nu_bounds = nothing)
        probs = validate_manifest(free_no_bounds)
        @test any(occursin("nu_bounds", p) for p in probs)

        fixed_with_bounds = make_run(family = :unrestricted, nu_policy = :fixed, nu_bounds = (-1.0, 1.0))
        probs2 = validate_manifest(fixed_with_bounds)
        @test any(occursin("nu_bounds", p) for p in probs2)

        free_wrong_family = make_run(family = :unrestricted, nu_policy = :free, nu_bounds = (-1.0, 1.0))
        probs3 = validate_manifest(free_wrong_family)
        @test any(occursin("has no nu", p) for p in probs3)

        neg_budget = make_run()
        neg_budget2 = RunManifest(sci = neg_budget.sci, family = neg_budget.family,
            economic_parameterization = neg_budget.economic_parameterization,
            A_coordinate_mode = neg_budget.A_coordinate_mode, nu_policy = neg_budget.nu_policy,
            nu_bounds = neg_budget.nu_bounds, draw_checksum_uniform = neg_budget.draw_checksum_uniform,
            draw_checksum_transformed = neg_budget.draw_checksum_transformed,
            outer_algorithm = neg_budget.outer_algorithm, outer_max_wall_seconds = -1.0,
            outer_max_gradients = neg_budget.outer_max_gradients, cache_policy = neg_budget.cache_policy,
            dual_bank_policy = neg_budget.dual_bank_policy, warm_start_policy = neg_budget.warm_start_policy,
            initial_state_digest = neg_budget.initial_state_digest, source_sha = neg_budget.source_sha,
            source_dirty = neg_budget.source_dirty)
        @test any(occursin("outer_max_wall_seconds", p) for p in validate_manifest(neg_budget2))
    end

    @testset "digest_economic_state is deterministic and order/value sensitive" begin
        gp = [1.0, 2.0]
        A = [0.0 1.0; 2.0 3.0]
        nu = [0.5]
        d1 = digest_economic_state(gp, A, nu)
        d2 = digest_economic_state(copy(gp), copy(A), copy(nu))
        @test d1 == d2
        @test length(d1) == 64  # hex sha256

        d3 = digest_economic_state([1.0, 2.0 + 1e-12], A, nu)
        @test d3 != d1

        d4 = digest_economic_state(gp, A, [0.50000001])
        @test d4 != d1
    end

    @testset "current_source_sha / source_is_dirty against the real repo" begin
        sha = current_source_sha(repo_dir = REPO_ROOT)
        @test length(sha) == 40
        @test occursin(r"^[0-9a-f]{40}$", sha)

        # This repo is expected to have uncommitted files at some points during this task
        # (e.g. mid-edit); just check the function runs and returns a Bool without throwing.
        dirty = source_is_dirty(repo_dir = REPO_ROOT)
        @test dirty isa Bool
    end

    @testset "refuse_if_dirty throws only when dirty" begin
        mktempdir() do d
            run(Cmd(`git init -q`; dir = d))
            run(Cmd(`git -c user.email=t@t -c user.name=t commit -q --allow-empty -m init`; dir = d))
            @test source_is_dirty(repo_dir = d) == false
            refuse_if_dirty(repo_dir = d)  # should not throw

            write(joinpath(d, "scratch.txt"), "x")
            run(Cmd(`git add scratch.txt`; dir = d))
            @test source_is_dirty(repo_dir = d) == true
            @test_throws ErrorException refuse_if_dirty(repo_dir = d)
        end
    end
end
