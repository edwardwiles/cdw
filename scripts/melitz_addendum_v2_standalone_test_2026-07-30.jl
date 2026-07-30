# Standalone isolated run of the new "Mandatory addendum 2026-07-30 (repaired v2 middle-loop
# driver)" testset appended to test/melitz/runtests.jl -- run in isolation to avoid the
# pre-existing, unrelated mul_G! SIGSEGV two immediately-prior sessions already disclosed in
# the full 8000-line suite (same convention the original fixed_q_a_middle_loop testset used).
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Test, Random, LinearAlgebra

FIXTURE = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
const KNITRO_AVAILABLE = true

@testset "Mandatory addendum 2026-07-30 (repaired v2 middle-loop driver)" begin
    v2_obj, v2_theta0 = build_melitz_psi_bundle(FIXTURE; outer_parameterization=:logcutoff,
        policy=CappedEvaluation(10.0), backend=:matrix_free, forbid_dense_fallback=true)
    v2_ctx = v2_obj.γ
    v2_D = v2_ctx.D
    v2_nA = v2_D^2 - 1
    v2_policy = CappedEvaluation(10.0)
    v2_theta_plain0 = melitz_unpower_theta_free(v2_theta0, v2_ctx)
    v2_A0, v2_f0, v2_gpj0, v2_fjj0, v2_q0 = expand_free_theta_logcutoff(v2_theta_plain0, v2_ctx)
    v2_A_free0 = v2_theta_plain0[2:1+v2_nA]
    v2_session = MelitzInnerSession(v2_obj, v2_ctx, v2_policy)
    v2_r0 = solve_melitz_delta!(v2_session, v2_theta0, v2_policy)
    @test v2_r0 isa FiniteSolved
    v2_sys = melitz_fixed_q_middle_constraint_system(v2_theta_plain0, v2_ctx, v2_obj)

    @testset "Cached evaluator: exact repeat point is a cache hit, no second inner solve" begin
        ec = MelitzExactPointCache(); bc = MelitzMiddleBadPointCache(); st = MelitzMiddleCacheStats()
        r_a = melitz_middle_objective_and_gradient_cached!(v2_session, v2_A_free0, v2_q0, v2_gpj0, v2_ctx, ec, bc, st; coordinate=:logA)
        r_b = melitz_middle_objective_and_gradient_cached!(v2_session, v2_A_free0, v2_q0, v2_gpj0, v2_ctx, ec, bc, st; coordinate=:logA)
        @test r_a.cache_hit == false
        @test r_b.cache_hit == true
        @test isapprox(r_a.Delta, r_b.Delta; atol=1e-10)
        @test isapprox(r_a.grad_free, r_b.grad_free; atol=1e-8)
        @test st.n_actual_inner_solves == 1
        @test st.n_cache_hits == 1
        @test length(st.seen_keys) == 1
    end

    if KNITRO_AVAILABLE
        @testset "solve_melitz_fixed_q_A_profile_v2: strict incumbent retention, dedup, no dense G" begin
            before_dense = MELITZ_DENSE_G_MATERIALIZATIONS[]
            v2_session.obj.use_cached_x = false; v2_session.obj.x .= NaN
            res = solve_melitz_fixed_q_A_profile_v2(v2_session, v2_q0, v2_gpj0, v2_A_free0, v2_ctx;
                coordinate=:logA, max_evals=60, box=1.0, sys=v2_sys)
            @test res.r_incumbent isa FiniteSolved
            @test res.Delta_incumbent <= res.Delta_start_verified + 1e-6
            @test res.unique_inner_solves <= res.unique_A_points
            Af, ff, gpjf, fjjf = melitz_expand_theta(res.theta_free_incumbent, v2_ctx)
            zhatf = melitz_baseline_cutoff(Af, ff, v2_ctx.w, v2_ctx.tau, v2_ctx.expenditure, v2_ctx.sigma)
            @test maximum(abs.(log.(zhatf) .- v2_q0)) < 1e-6
            @test abs(dot(v2_ctx.c_full, vec(log.(Af)))) < 1e-6
            @test abs(dot(v2_ctx.c_full, vec(log.(ff)))) < 1e-6
            resid_final = melitz_middle_constraint_residuals(v2_sys, res.A_free_incumbent; coordinate=:logA)
            @test all(resid_final[v2_sys.sense .== :ge] .>= -1e-5)
            @test MELITZ_DENSE_G_MATERIALIZATIONS[] == before_dense
            println("v2 primary test: Delta_incumbent=", res.Delta_incumbent, " Delta_start_verified=", res.Delta_start_verified,
                    " unique_A=", res.unique_A_points, " unique_solves=", res.unique_inner_solves, " cache_hits=", res.cache_hits)
        end

        @testset "solve_melitz_fixed_q_A_profile_v2: never worse than a genuinely bad/capped start" begin
            rng_bad = MersenneTwister(77)
            A_bad = melitz_project_start_to_middle_constraints(
                v2_A_free0 .+ 0.3 .* randn(rng_bad, v2_nA), v2_sys, v2_ctx)
            v2_session.obj.use_cached_x = false; v2_session.obj.x .= NaN
            res_bad = solve_melitz_fixed_q_A_profile_v2(v2_session, v2_q0, v2_gpj0, A_bad, v2_ctx;
                coordinate=:logA, max_evals=60, box=1.0, sys=v2_sys)
            if isfinite(res_bad.Delta_start_verified)
                @test res_bad.Delta_incumbent <= res_bad.Delta_start_verified + 1e-6
            end
            println("v2 bad-start test: Delta_incumbent=", res_bad.Delta_incumbent, " Delta_start_verified=", res_bad.Delta_start_verified)
        end
    end

    melitz_update_operator_at_theta!(v2_obj.op, v2_theta0, v2_ctx)
end

println("\nSTANDALONE ADDENDUM v2 TEST RUN COMPLETE")
