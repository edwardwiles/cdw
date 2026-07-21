# ============================================================================
# Regression test wrapping c13_validate_interval_native_archC.jl (an existing,
# already-validated print-based script) in a real @testset. Confirms the
# interval-native Architecture C CM Hessian (cm_hessian_architecture_interval.jl)
# matches the trusted dense interval-basis reference to machine precision,
# across L in {10,20,50} and both contrast schemes, at calibration and an
# unrestricted candidate point. Backfilled for the final-production-merge
# consolidation (brief §6): reuses the original script's logic verbatim rather
# than reimplementing it, since it was already the trusted validation for this
# exact equivalence.
# ============================================================================
using Test

include(joinpath(@__DIR__, "c13_validate_interval_native_archC.jl"))   # defines `results`, `worst`

@testset "CM interval-native vs dense-interval-reference Hessian equivalence" begin
    @test worst < 1e-8
    @testset "L=$(r.L) contrasts=$(r.contrasts) point=$(r.point)" for r in results
        @test r.maxdiff_hess < 1e-8
    end
end

println("All CM interval-equivalence tests passed.")
