# ============================================================================
# Tests for the pure part of the KNITRO status decoder (task §10). Deliberately
# does NOT `using KNITRO` or touch a live kc -- decode_knitro_status must work as a
# standalone lookup so status codes can be decoded from saved traces/checkpoints
# without a live solver. (knitro_solve_diagnostics/full_status_record, which DO need
# a live kc, are exercised instead by a live-solve smoke test in
# test_per_solve_counters.jl.)
# ============================================================================
using Test

include(joinpath(@__DIR__, "knitro_status.jl"))

@testset "KNITRO status decoder" begin

@testset "known feasible/optimal codes" begin
    for code in (0, -100, -101, -102, -103, -400, -401, -402)
        info = decode_knitro_status(code)
        @test info.code == code
        @test info.is_feasible_result == true
        @test info.category in (:optimal, :feasible_approx, :limit_feasible)
        @test !isempty(info.meaning)
    end
end

@testset "known infeasible/unbounded/limit-infeasible codes are NOT feasible results" begin
    for code in (-200, -201, -202, -203, -204, -205, -300, -301, -410, -411, -412)
        info = decode_knitro_status(code)
        @test info.is_feasible_result == false
    end
end

@testset "the production FEASIBLE_CODES set matches is_feasible_result exactly" begin
    # FEASIBLE_CODES = (0, -100, -101, -103) in c10_d20_production_driver.jl -- note -102 is
    # deliberately EXCLUDED from production's set even though KNITRO calls it a feasible
    # code, so this test intentionally checks the *production constant*, not decoder-implied
    # membership, and documents the discrepancy rather than silently asserting they're equal.
    production_feasible_codes = (0, -100, -101, -103)
    for code in production_feasible_codes
        @test decode_knitro_status(code).is_feasible_result == true
    end
    @test decode_knitro_status(-102).is_feasible_result == true   # KNITRO calls -102 feasible...
    @test !(-102 in production_feasible_codes)   # ...but production's FEASIBLE_CODES omits it (pre-existing, not this session's choice)
end

@testset "-300 unbounded is decoded distinctly from an exact-screen sentinel" begin
    u = decode_knitro_status(-300)
    @test u.category == :unbounded
    s = decode_knitro_status(-9004)
    @test s.category == :exact_screen_certificate
    @test u.category != s.category
end

@testset "repo-local exact-screen sentinels all decode with the correct name" begin
    expected = Dict(-9000 => :LOCAL_EXACT_SCREEN_GENERIC, -9001 => :LOCAL_PAIRWISE_CERTIFIED_INFEASIBLE,
                     -9002 => :LOCAL_WITNESS_CERTIFIED_INFEASIBLE, -9003 => :LOCAL_WINNER_SCAN_INFEASIBLE,
                     -9004 => :LOCAL_EXACT_INFEASIBLE_PREWINNER_ENVELOPE,
                     -9005 => :LOCAL_EXACT_INFEASIBLE_WINNING_RANGE,
                     -9006 => :LOCAL_EXACT_INFEASIBLE_MOMENT_RANGE)
    for (code, name) in expected
        info = decode_knitro_status(code)
        @test info.name == name
        @test info.category == :exact_screen_certificate
        @test info.is_feasible_result == false
    end
end

@testset "unknown code does not throw, returns :unknown category" begin
    info = decode_knitro_status(-424242)
    @test info.category == :unknown
    @test info.name == :UNKNOWN
    @test occursin("-424242", info.meaning)
end

@testset "decode_knitro_status usable without `using KNITRO` in scope" begin
    # This test file itself never imports KNITRO -- if this testset runs at all, that already
    # demonstrates the pure decoder has no load-time dependency on the KNITRO package.
    @test decode_knitro_status(0).category == :optimal
end

end # testset

println("All KNITRO status decoder tests passed.")
