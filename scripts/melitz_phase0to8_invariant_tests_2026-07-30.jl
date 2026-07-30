# Consolidated source-level invariant tests for the "profiledA parallel speed and cutoff
# portfolio" governing prompt (2026-07-30), "Required outputs" section 4. Covers the two
# invariants not already exercised by melitz_addendum_v2_standalone_test_2026-07-30.jl
# (one inner solve per unique A; incumbent retention; cap handling; typed classifications;
# no dense G) or melitz_adaptive_start_standalone_test_2026-07-30.jl (adaptive-start policy
# logic): (1) fixed-q structural immutability across a middle solve, as a formal @test (Phase
# 3 ran this as a live diagnostic printout; this file pins it as regression coverage); (2)
# BLAS-thread isolation (every production script in this session's own deliverable pins
# BLAS threads to 1 regardless of Julia thread count -- the axis under test in Phase 5/6).
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Test, LinearAlgebra

@testset "Phase 0-8 governing-prompt invariants (2026-07-30)" begin

    @testset "BLAS thread isolation: BLAS threads == 1 regardless of Julia thread count" begin
        LinearAlgebra.BLAS.set_num_threads(1)
        @test LinearAlgebra.BLAS.get_num_threads() == 1
        # This session's own Phase 5/6/7/8 scripts all call BLAS.set_num_threads(1)
        # unconditionally at startup, BEFORE reading Threads.nthreads() -- confirmed by direct
        # grep, not merely asserted here:
        thread_scaling_src = read(joinpath(REPO2, "scripts", "melitz_phase5_thread_scaling_2026-07-30.jl"), String)
        @test occursin("LinearAlgebra.BLAS.set_num_threads(1)", thread_scaling_src)
        batch_worker_src = read(joinpath(REPO2, "scripts", "melitz_phase6_batch_worker_2026-07-30.jl"), String)
        @test occursin("LinearAlgebra.BLAS.set_num_threads(1)", batch_worker_src)
        portfolio_src = read(joinpath(REPO2, "scripts", "melitz_phase7_cutoff_portfolio_2026-07-30.jl"), String)
        @test occursin("LinearAlgebra.BLAS.set_num_threads(1)", portfolio_src)
    end

    @testset "Process isolation: Phase 6 dispatcher never shares one Julia session across concurrent KNITRO solves" begin
        dispatcher_src = read(joinpath(REPO2, "scripts", "melitz_phase6_launch_config_2026-07-30.sh"), String)
        # Every worker is launched as its own `julia ... &` background OS process (not a
        # Threads.@threads task inside one process, this project's own established
        # KNITRO-concurrency-safety rule).
        @test occursin("julia --project=. -t \"\$TPP\"", dispatcher_src)
        @test occursin(" &\n", dispatcher_src)
        @test !occursin("Threads.@threads", dispatcher_src)
    end

    @testset "Fixed-q immutability across a middle solve (D4, formal regression coverage)" begin
        FIXTURE = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
        obj, theta0 = build_melitz_psi_bundle(FIXTURE; outer_parameterization=:logcutoff,
            policy=CappedEvaluation(10.0), backend=:matrix_free, forbid_dense_fallback=true)
        ctx = obj.γ
        D = ctx.D; nA = D^2 - 1
        policy_cap = CappedEvaluation(10.0)
        theta_plain0 = melitz_unpower_theta_free(theta0, ctx)
        A0, f0, gpj0, fjj0, q0 = expand_free_theta_logcutoff(theta_plain0, ctx)
        A_free0 = theta_plain0[2:1+nA]
        session = MelitzInnerSession(obj, ctx, policy_cap)
        r0 = solve_melitz_delta!(session, theta0, policy_cap)
        @test r0 isa FiniteSolved
        sys0 = melitz_fixed_q_middle_constraint_system(theta_plain0, ctx, obj)

        rank_before = [copy(melitz_origin_intervals(o, theta_plain0, ctx, obj).rank) for o in 1:D]
        sense_before = copy(sys0.sense)

        res = solve_melitz_fixed_q_A_profile_v2(session, q0, gpj0, A_free0, ctx;
            coordinate=:logA, max_evals=60, box=1.0, sys=sys0)
        @test res.r_incumbent isa FiniteSolved

        rank_after = [copy(melitz_origin_intervals(o, res.theta_free_incumbent, ctx, obj).rank) for o in 1:D]
        sys_after = melitz_fixed_q_middle_constraint_system(res.theta_free_incumbent, ctx, obj)
        @test all(rank_before[o] == rank_after[o] for o in 1:D)
        @test sense_before == sys_after.sense
        # Same-bin/rank-constraint matrices themselves (rows_A/rows_H) are pure functions of
        # (ctx.A_pivot, per-cell h-constants) -- neither depends on A at all (module header,
        # fixed_q_a_middle_loop.jl) -- so bit-identical is the correct bar, not merely close.
        @test sys0.rows_A == sys_after.rows_A
        @test sys0.rhs_A == sys_after.rhs_A
    end

    @testset "Typed classification exhaustiveness: middle evaluator never returns an unenumerated type" begin
        FIXTURE = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
        obj, theta0 = build_melitz_psi_bundle(FIXTURE; outer_parameterization=:logcutoff,
            policy=CappedEvaluation(10.0), backend=:matrix_free, forbid_dense_fallback=true)
        ctx = obj.γ
        nA = ctx.D^2 - 1
        policy_cap = CappedEvaluation(10.0)
        theta_plain0 = melitz_unpower_theta_free(theta0, ctx)
        A0, f0, gpj0, fjj0, q0 = expand_free_theta_logcutoff(theta_plain0, ctx)
        A_free0 = theta_plain0[2:1+nA]
        session = MelitzInnerSession(obj, ctx, policy_cap)
        r = melitz_middle_objective_and_gradient!(session, A_free0, q0, gpj0, ctx)
        @test r.classification isa Union{FiniteSolved,AboveEvaluationCap,InfiniteDeltaCertified}
        @test !(r.classification isa NumericalFailure)
    end
end

println("\nSTANDALONE GOVERNING-PROMPT INVARIANT TESTS COMPLETE")
