# Tests for FamilyRegistry -- canonicalization mapping and registry completeness/self-consistency.
# Standalone: does not depend on anything in full_aod_diag/, src/melitz/, or scripts/melitz_*.
# Run with: julia --project=. scientific_manifest/test_family_registry.jl

using Test

include(joinpath(@__DIR__, "FamilyRegistry.jl"))
using .FamilyRegistryMod

@testset "FamilyRegistry" begin

    @testset "canonicalization is a real bijection over the 5 families" begin
        @test length(CANONICAL_FAMILIES) == 5
        @test length(REDUCED_FAMILY_KIND_TO_CANONICAL) == 5
        @test Set(values(REDUCED_FAMILY_KIND_TO_CANONICAL)) == Set(CANONICAL_FAMILIES)
        for fam in CANONICAL_FAMILIES
            kind = reduced_kind_of_canonical_family(fam)
            @test canonical_family_of_reduced_kind(kind) == fam
        end
    end

    @testset "known mismatches are exactly the documented ones" begin
        @test canonical_family_of_reduced_kind(:unrestricted) == :unrestricted   # same spelling
        @test canonical_family_of_reduced_kind(:flexible_CM) == :flexible_cm     # case differs
        @test canonical_family_of_reduced_kind(:common_Frechet) == :common_frechet  # case differs
        @test canonical_family_of_reduced_kind(:ZC_only) == :origin_zc           # different name
        @test canonical_family_of_reduced_kind(:CM_plus_ZC) == :cm_meanzc        # different name
    end

    @testset "canonicalization throws on unrecognized symbols, does not guess" begin
        @test_throws ErrorException canonical_family_of_reduced_kind(:not_a_real_family)
        @test_throws ErrorException reduced_kind_of_canonical_family(:not_a_real_family)
        # Case sensitivity is real and load-bearing -- lowercase :flexible_cm is the CANONICAL
        # name, not a valid REDUCED family_kind() value, so this must throw, not silently match.
        @test_throws ErrorException canonical_family_of_reduced_kind(:flexible_cm)
    end

    @testset "registry has exactly the 10 (family, formulation) cells, none missing" begin
        for fam in CANONICAL_FAMILIES, form in (:full_gamma_normalized, :profiled_destination_scales)
            cap = capability(fam, form)
            @test cap.family == fam
            @test cap.formulation == form
        end
        @test_throws ErrorException capability(:unrestricted, :not_a_real_formulation)
        @test_throws ErrorException capability(:not_a_real_family, :full_gamma_normalized)
    end

    @testset "FULL rows are all checkpoint-resumable and production-ready; REDUCED rows reflect real per-family evidence" begin
        for fam in CANONICAL_FAMILIES
            full_cap = capability(fam, :full_gamma_normalized)
            @test full_cap.checkpoint_resume == true
            @test full_cap.production_ready == true
            @test full_cap.checkpoint_function !== nothing
        end
        # 2026-08-04: ALL FIVE REDUCED rows now flip production_ready=true -- each row's own notes
        # had named a specific, precise blocker, and all five are now closed with real evidence:
        # unrestricted (D20/W80000 outer-gradient gp-coordinate discrepancy, RESOLVED via
        # investigate_gp_fd_bandwidth_2026-08-04.jl -- an unchecked-solver-status FD artifact, not
        # a gradient bug), flexible_cm/common_frechet (D20/W100000 native outer-gradient gate,
        # test_flexcm_frechet_outer_gradient_d20_w100k_2026-08-04.jl, ~1e-10, zero dense G), and
        # origin_zc/cm_meanzc (free-nu wired into a genuine new production driver,
        # profiled_zc_free_nu_production_driver_2026-08-04.jl / run_profiled_upper_constrained_
        # free_nu -- ADDITIVE, does not touch run_profiled_upper_constrained itself -- real D4
        # AND D20/W=20,000 KNITRO solves confirm eta_nu genuinely moves and checkpoint/resume
        # round-trips it correctly). free_nu_supported flips true for origin_zc/cm_meanzc
        # specifically (the only 2 REDUCED families with a nu parameter at all) on that evidence;
        # the other 3 REDUCED families have no nu parameter and free_nu_supported stays false for
        # them, not because of missing work.
        for fam in CANONICAL_FAMILIES
            @test capability(fam, :profiled_destination_scales).production_ready == true
        end
        for fam in (:origin_zc, :cm_meanzc)
            @test capability(fam, :profiled_destination_scales).free_nu_supported == true
        end
        for fam in (:unrestricted, :flexible_cm, :common_frechet)
            @test capability(fam, :profiled_destination_scales).free_nu_supported == false
        end
    end

    @testset "production_ready_families reflects the registry, not a hardcoded list" begin
        @test Set(production_ready_families(:full_gamma_normalized)) == Set(CANONICAL_FAMILIES)
        @test Set(production_ready_families(:profiled_destination_scales)) == Set(CANONICAL_FAMILIES)
    end

    @testset "only origin_zc/cm_meanzc are free-nu-capable on the FULL side (the other 3 families have no nu at all)" begin
        for fam in (:origin_zc, :cm_meanzc)
            @test capability(fam, :full_gamma_normalized).free_nu_supported == true
        end
        for fam in (:unrestricted, :flexible_cm, :common_frechet)
            @test capability(fam, :full_gamma_normalized).free_nu_supported == false
        end
    end
end
