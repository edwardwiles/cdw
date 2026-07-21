# ============================================================================
# Regression tests for the staged-delta-continuation incumbent bug
# (docs/fullA_driver_delta5_diagnostics_handoff.md §3-4). Deterministic, no KNITRO,
# no real D=20 context -- exercises the pure helpers in incumbent_logic.jl directly
# plus a toy KNITRO-shaped callback loop that reproduces the exact
# seed-then-conditionally-update pattern used by run_profile_checkpointed /
# run_polish_checkpointed, so these tests would FAIL under the pre-fix behavior
# (`best = Ref(resumed !== nothing ? resumed.best_feasible : nothing)`, no seeding
# from the cold-verified start point).
# ============================================================================
using Test

include(joinpath(@__DIR__, "incumbent_logic.jl"))

# ----------------------------------------------------------------------------
# A toy stand-in for the real cb_F! callback loop, isolating exactly the
# seed/compare/update pattern under test (no ctx, no KNITRO, no economics).
# `trial_gps` simulates the sequence of gp values KNITRO's own callback would
# see; any entry `nothing` simulates a rejected (infeasible/thrown) trial that
# never reaches the is_new_best comparison at all.
# ----------------------------------------------------------------------------
function toy_polish_run(; resumed_best, seed_gp::Float64, seed_feasible::Bool,
        trial_gps::Vector{Union{Nothing,Float64}}, find_smallest::Bool)
    seed_cand = (gp = seed_gp, w = [seed_gp], Delta = 0.0, gravity = 0.0, kkt = 0.0,
                 inner_status = 0, t_elapsed = 0.0, n_eval = 0)
    best = Ref{Any}(seed_incumbent(resumed_best, seed_feasible, seed_cand))
    for g in trial_gps
        g === nothing && continue   # rejected trial: cb_F! throws before touching best
        if is_better_polish(g, best[] === nothing ? nothing : best[].gp, find_smallest)
            best[] = (gp = g, w = [g], Delta = 0.0, gravity = 0.0, kkt = 0.0,
                      inner_status = 0, t_elapsed = 0.0, n_eval = 0)
        end
    end
    return best[]
end

@testset "incumbent seeding" begin

@testset "increasing budget: fresh stage initializes at least as well as its feasible start" begin
    # Simulates a stage-i+1 call seeded from stage-i's genuinely feasible best point, where
    # KNITRO finds nothing better before the wall-time budget runs out (trial_gps empty --
    # the exact failure mode diagnosed in §3: ctx rebuild eats the whole per-stage budget).
    b = toy_polish_run(resumed_best = nothing, seed_gp = 0.0806, seed_feasible = true,
                        trial_gps = Union{Nothing,Float64}[], find_smallest = false)
    @test b !== nothing
    @test b.gp == 0.0806   # NOT nothing / NOT worse -- this is what the pre-fix code violated
end

@testset "zero-iteration termination returns the supplied feasible start, not nothing" begin
    b = toy_polish_run(resumed_best = nothing, seed_gp = 0.05, seed_feasible = true,
                        trial_gps = Union{Nothing,Float64}[], find_smallest = true)
    @test b !== nothing
    @test b.gp == 0.05
end

@testset "rejected final trial does not overwrite a prior real incumbent" begin
    # A real accepted improvement, then every subsequent trial is a rejection (nothing).
    b = toy_polish_run(resumed_best = nothing, seed_gp = 0.02, seed_feasible = true,
                        trial_gps = Union{Nothing,Float64}[0.05, nothing, nothing],
                        find_smallest = false)
    @test b.gp == 0.05   # the one real improvement survives
end

@testset "upper direction (find_smallest=false) prefers larger gp" begin
    @test is_better_polish(0.09, 0.08, false) == true
    @test is_better_polish(0.07, 0.08, false) == false
    @test is_better_polish(0.08, nothing, false) == true
end

@testset "lower direction (find_smallest=true) prefers smaller gp" begin
    @test is_better_polish(0.07, 0.08, true) == true
    @test is_better_polish(0.09, 0.08, true) == false
    @test is_better_polish(0.08, nothing, true) == true
end

@testset "profile direction always minimizes Delta_dual" begin
    @test is_better_profile(0.20, 0.25) == true
    @test is_better_profile(0.30, 0.25) == false
    @test is_better_profile(0.25, nothing) == true
end

@testset "checkpoint/resume retains the prior segment's incumbent" begin
    resumed_best = (gp = 0.0806, w = [0.0806], Delta = 0.1, gravity = 0.0, kkt = 0.0,
                     inner_status = 0, t_elapsed = 10.0, n_eval = 42)
    # Even a WORSE cold-verified seed at the resumed delta must not override the checkpoint.
    b = toy_polish_run(resumed_best = resumed_best, seed_gp = 0.01, seed_feasible = true,
                        trial_gps = Union{Nothing,Float64}[], find_smallest = false)
    @test b === resumed_best
    @test b.n_eval == 42   # provenance (not just the value) is preserved, not reconstructed
end

@testset "infeasible seed correctly yields no incumbent (matches pre-fix behavior for this one case)" begin
    b = toy_polish_run(resumed_best = nothing, seed_gp = 0.03, seed_feasible = false,
                        trial_gps = Union{Nothing,Float64}[], find_smallest = false)
    @test b === nothing
end

@testset "later worse trial after seeding never regresses the incumbent" begin
    # Reproduces the task's reported pathology directly: a feasible start at kappa-equivalent
    # gp=0.0806 must never be replaced by a later, worse-but-still-"first-found" trial.
    b = toy_polish_run(resumed_best = nothing, seed_gp = 0.0806, seed_feasible = true,
                        trial_gps = Union{Nothing,Float64}[0.02], find_smallest = false)
    @test b.gp == 0.0806   # 0.02 is worse (smaller, under find_smallest=false) -- must be rejected
end

end # testset

println("All incumbent-seeding regression tests passed.")
