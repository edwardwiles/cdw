# Standalone isolated correctness test for melitz_middle_two_start_adaptive!
# (src/melitz/fixed_q_a_middle_loop.jl, "ADDENDUM 2026-07-30 PART 2" -- Phase 2 of the
# "profiledA parallel speed and cutoff portfolio" governing prompt, 2026-07-30). Run in
# isolation, same convention as melitz_addendum_v2_standalone_test_2026-07-30.jl, to avoid the
# pre-existing, unrelated mul_G! SIGSEGV disclosed in the full 8000-line suite.
#
# Design note: this codebase's own moment/gradient construction is threaded
# (Threads.@threads, src/melitz/CLAUDE.md), so floating-point summation order -- and hence
# KNITRO's own terminal nStatus on a run that lands close to a convergence-tolerance boundary --
# is not guaranteed bit-identical run-to-run even from an identical seed/start (confirmed live
# during this test's own development: the SAME MersenneTwister(11) perturbation intermittently
# terminated at nStatus=0 in one run and a KKT-adjacent-but-distinct code in another). Tests
# below therefore assert the ROBUST structural invariants (never-worse-than-input;
# periodic_safeguard_due always forces the fallback; a genuine pivot change always forces the
# fallback) rather than pinning an exact `trigger_reason` symbol under a stochastic start, except
# where the start is deliberately UNPERTURBED (deterministic, confirmed nStatus=0 across repeats).
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Test, Random, LinearAlgebra

FIXTURE = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)

@testset "Adaptive second-start policy (Phase 2, 2026-07-30)" begin
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

    @testset "Deterministic (unperturbed) anchor start twice: reproducible baseline decision + never worse than input" begin
        # A_free0 (raw, deterministic, no randomness) reliably converges cleanly at this
        # fixture -- run twice to confirm the decision is reproducible for a fixed input.
        res_a = melitz_middle_two_start_adaptive!(session, q0, gpj0, A_free0, A_free0, ctx;
            delta_budget=100.0, max_evals=60, box=1.0, sys=sys0)
        res_b = melitz_middle_two_start_adaptive!(session, q0, gpj0, A_free0, A_free0, ctx;
            delta_budget=100.0, max_evals=60, box=1.0, sys=sys0)
        @test res_a.r_incumbent isa FiniteSolved
        @test res_a.trigger_reason == res_b.trigger_reason
        @test isapprox(res_a.Delta, res_b.Delta; atol=1e-8)
        @test res_a.Delta <= res_a.r_continuation.Delta_start_verified + 1e-6
        println("unperturbed-anchor trigger=", res_a.trigger_reason, " ran_compensated=", res_a.ran_compensated,
                " Delta=", res_a.Delta, " Delta_start_verified=", res_a.r_continuation.Delta_start_verified)
    end

    @testset "Improvement/near-budget gate both closed by construction -> fallback always runs (deterministic)" begin
        # Rather than relying on a live re-optimization to happen to land exactly at an
        # exhausted local optimum (empirically unreliable -- a second KNITRO run from a reported
        # incumbent can still find further real progress, since max_evals/box bound the FIRST
        # run's own thoroughness), construct the negligible-improvement/far-from-budget case
        # DETERMINISTICALLY via the policy's own thresholds: material_improvement_frac=0.9999
        # (requires near-total elimination of Delta to count as "material") and delta_budget
        # effectively unreachable (near_budget_frac=0.0 with a huge budget) together make the
        # accept-alone OR-gate structurally unsatisfiable regardless of what Delta is actually
        # found -- using the deterministic, reliably-nStatus=0 unperturbed anchor start.
        impossible_gate = MelitzAdaptiveStartPolicy(; material_improvement_frac=0.9999, near_budget_frac=0.0)
        res = melitz_middle_two_start_adaptive!(session, q0, gpj0, A_free0, A_free0, ctx;
            delta_budget=1.0e6, max_evals=60, box=1.0, sys=sys0, policy=impossible_gate)
        @test res.ran_compensated == true
        @test res.trigger_reason == :negligible_improvement
        @test res.Delta <= res.r_continuation.Delta_start_verified + 1e-6
    end

    @testset "Perturbed start: never-worse-than-input holds regardless of which trigger fires" begin
        rng = MersenneTwister(11)
        A_pert = melitz_project_start_to_middle_constraints(
            A_free0 .+ 0.02 .* randn(rng, nA), sys0, ctx)
        res = melitz_middle_two_start_adaptive!(session, q0, gpj0, A_pert, A_free0, ctx;
            delta_budget=100.0, max_evals=60, box=1.0, sys=sys0,
            policy=MelitzAdaptiveStartPolicy(; material_improvement_frac=0.001))
        @test res.r_incumbent isa FiniteSolved
        println("perturbed-start trigger=", res.trigger_reason, " ran_compensated=", res.ran_compensated,
                " Delta=", res.Delta)
        @test res.Delta <= res.r_continuation.Delta_start_verified + 1e-6
        if res.ran_compensated
            @test res.r_compensated !== nothing
        end
    end

    @testset "Never returns worse than the strictly better of the two individual v2 results" begin
        rng2 = MersenneTwister(202)
        A_pert2 = melitz_project_start_to_middle_constraints(
            A_free0 .+ 0.05 .* randn(rng2, nA), sys0, ctx)
        A_comp2 = melitz_project_start_to_middle_constraints(
            A_free0 .+ 0.05 .* randn(MersenneTwister(303), nA), sys0, ctx)
        res = melitz_middle_two_start_adaptive!(session, q0, gpj0, A_pert2, A_comp2, ctx;
            delta_budget=100.0, max_evals=60, box=1.0, sys=sys0,
            periodic_safeguard_due=true)   # force both starts to run
        @test res.ran_compensated == true
        @test res.r_compensated !== nothing
        cont_ok = res.r_continuation.r_incumbent isa FiniteSolved
        comp_ok = res.r_compensated.r_incumbent isa FiniteSolved
        if cont_ok && comp_ok
            @test res.Delta <= min(res.r_continuation.Delta_incumbent, res.r_compensated.Delta_incumbent) + 1e-9
        end
    end

    @testset "periodic_safeguard_due is a hard guarantee: always forces the compensated fallback" begin
        # Structural test, not a stochastic-convergence one: regardless of what nStatus the
        # continuation start happens to land on, periodic_safeguard_due=true must ALWAYS force
        # ran_compensated=true (the trigger cascade checks it before ever reaching
        # :accepted_continuation_alone) -- verified across several independent random starts.
        for seed in (11, 42, 77, 123)
            rngk = MersenneTwister(seed)
            A_k = melitz_project_start_to_middle_constraints(
                A_free0 .+ 0.02 .* randn(rngk, nA), sys0, ctx)
            res = melitz_middle_two_start_adaptive!(session, q0, gpj0, A_k, A_free0, ctx;
                delta_budget=100.0, max_evals=60, box=1.0, sys=sys0,
                policy=MelitzAdaptiveStartPolicy(; material_improvement_frac=0.001), periodic_safeguard_due=true)
            @test res.ran_compensated == true
            @test res.trigger_reason != :accepted_continuation_alone
        end
    end

    @testset "pivot_changed trigger fires on a genuine sense-vector flip (synthetic sys/prev_sys)" begin
        sys_flipped = MelitzFixedQMiddleConstraintSystem(sys0.D, sys0.n, sys0.rows_A, sys0.rhs_A,
            sys0.rows_H, sys0.rhs_H, ifelse.(sys0.sense .== :eq, :ge, :eq), sys0.kind, sys0.o_cell, sys0.d_lo, sys0.d_hi)
        res = melitz_middle_two_start_adaptive!(session, q0, gpj0, A_free0, A_free0, ctx;
            delta_budget=100.0, max_evals=60, box=1.0, sys=sys0, prev_sys=sys_flipped)
        @test res.ran_compensated == true
        # A_free0 is the deterministic unperturbed start (confirmed nStatus=0 in the first
        # testset above), so no earlier cascade check (finite/no-worse/kkt/cap) can fire first --
        # pivot_changed is the only remaining reason this can run the fallback.
        @test res.trigger_reason == :pivot_changed
    end

    @testset "sys/prev_sys identical (no pivot change): the pivot_changed gate does not spuriously fire" begin
        res = melitz_middle_two_start_adaptive!(session, q0, gpj0, A_free0, A_free0, ctx;
            delta_budget=100.0, max_evals=60, box=1.0, sys=sys0, prev_sys=sys0)
        @test res.trigger_reason != :pivot_changed
    end

    melitz_update_operator_at_theta!(obj.op, theta0, ctx)
end

println("\nSTANDALONE ADAPTIVE-START POLICY TEST RUN COMPLETE")
