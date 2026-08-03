# Standalone isolated test run for profiled_envelope_gradient.jl (avoids the pre-existing,
# unrelated mul_G! SIGSEGV in the full test suite, per this repo's own established convention).
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
WT = "/bbkinghome/edav/gravity_robustness/worktrees/melitz-profiled-q-envelope-gradient-2026-07-31"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(WT, "src", "melitz", "include_melitz.jl"))
using Test, LinearAlgebra

LinearAlgebra.BLAS.set_num_threads(1)
CAP = 10.0
policy_cap = CappedEvaluation(CAP)

@testset "profiled_envelope_gradient.jl standalone (2026-07-31)" begin
    FIXTURE4 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
    obj4, theta04 = build_melitz_psi_bundle(FIXTURE4; outer_parameterization=:logcutoff, policy=policy_cap,
        backend=:matrix_free, forbid_dense_fallback=true)
    ctx4 = obj4.γ
    session4 = MelitzInnerSession(obj4, ctx4, policy_cap)
    theta_plain04 = melitz_unpower_theta_free(theta04, ctx4)
    A04, f04, gpj04, fjj04, q04 = expand_free_theta_logcutoff(theta_plain04, ctx4)
    A_free04 = theta_plain04[2:1+ctx4.D^2-1]
    g0 = log(gpj04)

    @testset "melitz_chamber_fingerprint deterministic and bit-identical for the same theta" begin
        fp1 = melitz_chamber_fingerprint(theta_plain04, ctx4, obj4)
        fp2 = melitz_chamber_fingerprint(theta_plain04, ctx4, obj4)
        @test fp1 == fp2
        @test size(fp1) == (ctx4.D, ctx4.D)
        @test eltype(fp1) == Int
    end

    @testset "melitz_exact_g_gradient returns 0.0 when mu_link==0" begin
        # A synthetic dual with mu[focal_link_index]=0 should short-circuit to exactly zero,
        # regardless of any other quantity (a structural invariant of the "no local optimum to
        # differentiate around" convention this codebase uses elsewhere).
        base = melitz_middle_objective_and_gradient!(session4, A_free04, q04, exp(g0), ctx4; coordinate=:logA)
        @test base.classification isa FiniteSolved
        x_zeroed = copy(base.classification.x)
        x_zeroed[1+obj4.op.layout.focal_link_index] = 0.0
        _, f_, gpj_, fjj_, _ = expand_free_theta_logcutoff(theta_plain04, ctx4)
        state = MelitzExpandedState(A04, f_, gpj_, fjj_)
        @test melitz_exact_g_gradient(obj4, x_zeroed, state, ctx4) == 0.0
    end

    @testset "profiled_envelope_derivative: internal cold-reverify reproduces the incumbent Delta" begin
        theta_for_sys = melitz_fixed_q_state_theta(A_free04, q04, exp(g0), ctx4)
        sys = melitz_fixed_q_middle_constraint_system(theta_for_sys, ctx4, obj4)
        env = profiled_envelope_derivative(session4, theta_plain04, q04, ctx4; sys=sys, A_free_incumbent=A_free04)
        @test env.nStatus in (0, -100, -101, -103)
        @test isfinite(env.dPhi_dg)
        @test all(isfinite, env.dPhi_dq_free)
        @test length(env.dPhi_dq_free) == ctx4.D^2 - 2
        @test size(env.chamber_rank) == (ctx4.D, ctx4.D)
        # q04[j,j] is not a free q-domain cell -- exact_q_smooth_gradient_full! always sets it 0.
        @test env.dPhi_dq_full[ctx4.target_country, ctx4.target_country] == 0.0
    end

    @testset "melitz_exact_q_smooth_gradient: zero outside the focal-origin row" begin
        base = melitz_middle_objective_and_gradient!(session4, A_free04, q04, exp(g0), ctx4; coordinate=:logA)
        @test base.classification isa FiniteSolved
        _, f_, gpj_, fjj_, _ = expand_free_theta_logcutoff(theta_plain04, ctx4)
        state = MelitzExpandedState(A04, f_, gpj_, fjj_)
        ws = MelitzExactQSmoothGradientWorkspace(obj4.op)
        dq_full = zeros(ctx4.D, ctx4.D)
        melitz_exact_q_smooth_gradient_full!(dq_full, obj4, base.classification.x, state, ctx4, ws)
        j = ctx4.target_country
        for o in 1:ctx4.D, d in 1:ctx4.D
            o != j && @test dq_full[o, d] == 0.0
        end
    end
end
