# Tests for ABComparability -- the hard A/B gate (task profiled-outer-production-readiness-2026-08-03
# section 10). Standalone: no full_aod_diag/KNITRO dependency.
# Run with: julia --project=. scientific_manifest/test_ab_comparability.jl

using Test

include(joinpath(@__DIR__, "ABComparability.jl"))
using .ABComparabilityMod
using .ABComparabilityMod.RunManifestMod
using .ABComparabilityMod.RunManifestMod.ScientificManifestMod: ScientificManifest

function make_sci()
    ScientificManifest(
        dataset_version = "d20_real", dataset_checksum = "deadbeef",
        country_order = ["fra", "bra", "kor", "row"], focal_country = "fra",
        sigma = 3.0, exclude_diagonal_gravity = true, gravity_exclude_cells = [(2, 3)],
        destination_sample = :exclude_row, draw_design = :sobol_randomized,
        draw_seed = 20260719, W = 100_000, L = 10, K_mean = 1, K_pair = 0,
        julia_threads = 8, blas_threads = 1,
        inner_opt_checksum = "innerabc", outer_opt_checksum = "outerdef")
end

function make_manifest(; economic_parameterization, family = :origin_zc,
        A_coordinate_mode = economic_parameterization == :full_gamma_normalized ? :legacy_z : :profiled_pivot_anchor_relative,
        source_sha = "a"^40, source_dirty = false, initial_state_digest = "state123",
        sci = make_sci())
    RunManifest(sci = sci, family = family, economic_parameterization = economic_parameterization,
        A_coordinate_mode = A_coordinate_mode, nu_policy = :fixed, nu_bounds = nothing,
        draw_checksum_uniform = "u1", draw_checksum_transformed = "t1",
        outer_algorithm = :knitro_direct_sr1, outer_max_wall_seconds = 600.0, outer_max_gradients = 5,
        cache_policy = :exact_cache, dual_bank_policy = :recording_only, warm_start_policy = :cold,
        verification_policy = :inner_status_only, initial_state_digest = initial_state_digest,
        source_sha = source_sha, source_dirty = source_dirty)
end

function make_sci_with(base::ScientificManifest; kwargs...)
    ScientificManifest(dataset_version = base.dataset_version, dataset_checksum = base.dataset_checksum,
        country_order = base.country_order, focal_country = base.focal_country, sigma = base.sigma,
        exclude_diagonal_gravity = base.exclude_diagonal_gravity, gravity_exclude_cells = base.gravity_exclude_cells,
        destination_sample = base.destination_sample, draw_design = base.draw_design, draw_seed = base.draw_seed,
        W = base.W, L = base.L, K_mean = base.K_mean, K_pair = base.K_pair,
        julia_threads = base.julia_threads, blas_threads = base.blas_threads,
        inner_opt_checksum = base.inner_opt_checksum, outer_opt_checksum = base.outer_opt_checksum; kwargs...)
end

@testset "ABComparability" begin

    @testset "two genuinely comparable manifests pass (same A_coordinate_mode, isolating the other fields)" begin
        full = make_manifest(economic_parameterization = :full_gamma_normalized, A_coordinate_mode = :legacy_z)
        reduced = make_manifest(economic_parameterization = :profiled_destination_scales, A_coordinate_mode = :legacy_z)
        r = ab_comparable(full, reduced)
        @test r.comparable
        @test isempty(r.problems)
        @test r.decoded_state_equivalent
        @test !r.coordinate_mode_differs
    end

    @testset "A_coordinate_mode legitimately differs between formulations by construction -- must be declared" begin
        # make_manifest's own defaults set full->:legacy_z, reduced->:profiled_pivot_anchor_relative,
        # which ARE different symbols -- re-verify that specific pairing is flagged unless allowed.
        full = make_manifest(economic_parameterization = :full_gamma_normalized, A_coordinate_mode = :legacy_z)
        reduced = make_manifest(economic_parameterization = :profiled_destination_scales, A_coordinate_mode = :profiled_pivot_anchor_relative)
        r_default = ab_comparable(full, reduced)
        @test !r_default.comparable
        @test r_default.coordinate_mode_differs
        @test any(occursin("A_coordinate_mode", p) for p in r_default.problems)

        r_no_label = ab_comparable(full, reduced; allow_coordinate_mode_diff = true)
        @test !r_no_label.comparable
        @test any(occursin("label", p) for p in r_no_label.problems)

        r_labeled = ab_comparable(full, reduced; allow_coordinate_mode_diff = true, coordinate_mode_experiment_label = "coord-mode-experiment-2026-08-03")
        @test r_labeled.comparable
        @test r_labeled.coordinate_mode_experiment_label == "coord-mode-experiment-2026-08-03"
    end

    @testset "arms passed in the wrong order are refused" begin
        full = make_manifest(economic_parameterization = :full_gamma_normalized)
        reduced = make_manifest(economic_parameterization = :profiled_destination_scales)
        r = ab_comparable(reduced, full)   # swapped
        @test !r.comparable
        @test any(occursin("wrong order", p) for p in r.problems)
    end

    @testset "family mismatch is refused" begin
        full = make_manifest(economic_parameterization = :full_gamma_normalized, family = :unrestricted)
        reduced = make_manifest(economic_parameterization = :profiled_destination_scales, family = :origin_zc)
        r = ab_comparable(full, reduced)
        @test !r.comparable
        @test any(occursin("family differs", p) for p in r.problems)
    end

    @testset "every sci field mismatch is independently caught" begin
        full = make_manifest(economic_parameterization = :full_gamma_normalized)
        base_sci = make_sci()
        mismatches = Dict(
            :sigma => make_sci_with(base_sci, sigma = 2.5),
            :W => make_sci_with(base_sci, W = 20_000),
            :draw_seed => make_sci_with(base_sci, draw_seed = 1),
            :destination_sample => make_sci_with(base_sci, destination_sample = :all_legacy),
        )
        for (field, bad_sci) in mismatches
            reduced = make_manifest(economic_parameterization = :profiled_destination_scales, sci = bad_sci)
            r = ab_comparable(full, reduced)
            @test !r.comparable
        end
    end

    @testset "different initial_state_digest is refused (not the same decoded economic point)" begin
        full = make_manifest(economic_parameterization = :full_gamma_normalized, initial_state_digest = "stateA")
        reduced = make_manifest(economic_parameterization = :profiled_destination_scales, initial_state_digest = "stateB")
        r = ab_comparable(full, reduced)
        @test !r.comparable
        @test !r.decoded_state_equivalent
    end

    @testset "different source_sha is refused" begin
        full = make_manifest(economic_parameterization = :full_gamma_normalized, source_sha = "a"^40)
        reduced = make_manifest(economic_parameterization = :profiled_destination_scales, source_sha = "b"^40)
        r = ab_comparable(full, reduced)
        @test !r.comparable
        @test any(occursin("source_sha", p) for p in r.problems)
    end

    @testset "dirty worktree on either arm is refused" begin
        full = make_manifest(economic_parameterization = :full_gamma_normalized, source_dirty = true)
        reduced = make_manifest(economic_parameterization = :profiled_destination_scales)
        r = ab_comparable(full, reduced)
        @test !r.comparable
        @test any(occursin("dirty", p) for p in r.problems)
    end

    @testset "manifest_digest is deterministic and value-sensitive" begin
        m1 = make_manifest(economic_parameterization = :full_gamma_normalized)
        m2 = make_manifest(economic_parameterization = :full_gamma_normalized)
        @test manifest_digest(m1) == manifest_digest(m2)
        m3 = make_manifest(economic_parameterization = :full_gamma_normalized, initial_state_digest = "different")
        @test manifest_digest(m1) != manifest_digest(m3)
    end

    @testset "require_ab_comparable throws with the reasons, and returns the result on success" begin
        full = make_manifest(economic_parameterization = :full_gamma_normalized, A_coordinate_mode = :legacy_z)
        reduced_bad = make_manifest(economic_parameterization = :profiled_destination_scales, family = :cm_meanzc, A_coordinate_mode = :legacy_z)
        @test_throws ErrorException require_ab_comparable(full, reduced_bad)

        reduced_good = make_manifest(economic_parameterization = :profiled_destination_scales, A_coordinate_mode = :legacy_z)
        r = require_ab_comparable(full, reduced_good)
        @test r.comparable
        @test !isempty(r.full_manifest_digest)
        @test !isempty(r.reduced_manifest_digest)
    end
end
