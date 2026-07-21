# ============================================================================
# Regression tests for parallelism_guards.jl's runtime mutual-exclusion invariant
# (inner-KNITRO-solve phase vs coordinate-probe-pool phase must never overlap).
# Pure Julia, no KNITRO, no context -- exercises the Ref-based state machine
# directly. Backfilled for the final-production-merge consolidation (brief §6):
# no dedicated test existed for this guard prior to this file.
# ============================================================================
using Test

include(joinpath(@__DIR__, "parallelism_guards.jl"))

@testset "parallelism guards" begin
    @testset "normal sequencing: no violation" begin
        guard_reset!()
        guard_enter_inner_solve!()
        guard_exit_inner_solve!()
        guard_enter_coord_pool!()
        guard_exit_coord_pool!()
        @test GUARD_VIOLATIONS[] == 0
        @test !INNER_SOLVE_ACTIVE[]
        @test !COORD_POOL_ACTIVE[]
    end

    @testset "coord pool active while inner solve tries to start -> error + violation counted" begin
        guard_reset!()
        guard_enter_coord_pool!()
        @test_throws ErrorException guard_enter_inner_solve!()
        @test GUARD_VIOLATIONS[] == 1
        guard_exit_coord_pool!()
    end

    @testset "inner solve active while coord pool tries to start -> error + violation counted" begin
        guard_reset!()
        guard_enter_inner_solve!()
        @test_throws ErrorException guard_enter_coord_pool!()
        @test GUARD_VIOLATIONS[] == 1
        guard_exit_inner_solve!()
    end

    @testset "exit is idempotent-safe after a caught violation (try/finally pattern)" begin
        guard_reset!()
        guard_enter_inner_solve!()
        try
            guard_enter_coord_pool!()
        catch
        finally
            guard_exit_inner_solve!()
        end
        @test !INNER_SOLVE_ACTIVE[]
        @test !COORD_POOL_ACTIVE[]
    end

    @testset "GUARD_ENABLED=false is a complete no-op (zero overhead path)" begin
        guard_reset!()
        GUARD_ENABLED[] = false
        try
            guard_enter_inner_solve!()
            guard_enter_coord_pool!()   # would normally error -- must NOT here
            @test GUARD_VIOLATIONS[] == 0
            @test !INNER_SOLVE_ACTIVE[]   # never set when disabled
            @test !COORD_POOL_ACTIVE[]
        finally
            GUARD_ENABLED[] = true
        end
    end

    guard_reset!()
end

println("All parallelism-guard tests passed.")
