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

            red_cap = capability(fam, :profiled_destination_scales)
            @test red_cap.free_nu_supported == false   # confirmed zero free-nu implementations wired into the production driver anywhere
        end
        # 2026-08-04: unrestricted/flexible_cm/common_frechet's REDUCED rows flipped
        # production_ready=true -- each row's own notes had named a specific outer-gradient FD
        # gate as its sole stated blocker; those gates now PASS for all three
        # (test_profiled_outer_gradient_gate_D20_W80000_2026-08-01.jl re-run + its gp-coordinate
        # discrepancy resolved via investigate_gp_fd_bandwidth_2026-08-04.jl for unrestricted;
        # test_flexcm_frechet_outer_gradient_d20_w100k_2026-08-04.jl for flexible_cm/
        # common_frechet, real D20/W=100,000, 11 representative coords incl. gp, max_rel_err
        # ~1e-10, zero dense G confirmed via NO_DENSE_G_COUNTERS). origin_zc/cm_meanzc each still
        # have a different, still-open, precisely-documented blocker (free-nu not wired into the
        # production driver; eta-generation cache/checkpoint wiring not done -- see
        # docs/audits/profiled-functional-readiness-closeout-2026-08-03/CONTINUATION_2026-08-04.md's
        # own final verdict block) -- this is not a uniform "section 12 bar" any more, it is
        # per-family, evidence-driven, and must stay that way rather than reverting to a blanket
        # assumption.
        for fam in (:unrestricted, :flexible_cm, :common_frechet)
            @test capability(fam, :profiled_destination_scales).production_ready == true
        end
        for fam in (:origin_zc, :cm_meanzc)
            @test capability(fam, :profiled_destination_scales).production_ready == false
        end
    end

    @testset "production_ready_families reflects the registry, not a hardcoded list" begin
        @test Set(production_ready_families(:full_gamma_normalized)) == Set(CANONICAL_FAMILIES)
        @test Set(production_ready_families(:profiled_destination_scales)) == Set([:unrestricted, :flexible_cm, :common_frechet])
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
