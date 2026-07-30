# Standalone isolated regression test for the O(W*D) -> O(W+D) focal-link reformation wired
# into src/melitz/moment_operator.jl's melitz_update_moment_operator! (2026-07-30, follow-up
# to the profiledA_parallel_speed_and_cutoff_portfolio governing prompt's own Phase 4 audit).
# Run in isolation, same convention as the other standalone tests, to avoid the pre-existing,
# unrelated mul_G! SIGSEGV disclosed in the full 8000-line suite.
#
# Keeps a VERBATIM copy of the ORIGINAL O(W*D) per-draw melitz_firm loop (the code this
# session replaced) as an independent reference -- never called from production, only from
# this test -- so a future accidental regression in the fast path has something exact to be
# checked against.
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Test, Random, LinearAlgebra

"""
    melitz_focal_link_ell_reference!(ell_ref, op, p, eq, cf)

VERBATIM copy of the pre-2026-07-30 `melitz_update_moment_operator!`'s own O(W*D) focal-link
loop -- an independent, deliberately-unoptimized reference, never used in production.
"""
function melitz_focal_link_ell_reference!(ell_ref::Vector{Float64}, op, p, eq, cf)
    D = op.D
    W = op.W
    sigma = p.sigma
    sorted_ctx = op.sorted_ctx
    j = p.target_country
    z_orig = sorted_ctx.z_original
    fill!(ell_ref, 0.0)
    @inbounds for d in 1:D
        for w in 1:W
            z = z_orig[w, j]
            firm = melitz_firm(p.w[j], p.tau[j, d], p.A[j, d], p.f[j, d], sigma,
                                eq.expenditure[d], 1.0, z)
            ell_ref[w] += firm.realized_operating_profit
        end
    end
    price_power_autarky = p.gamma_prime_target
    @inbounds for w in 1:W
        z_j = z_orig[w, j]
        firm_autarky = melitz_firm(cf.w_prime, 1.0, p.A[j, j], p.f[j, j], sigma,
                                    cf.expenditure_prime, price_power_autarky, z_j)
        ell_ref[w] = ell_ref[w] / p.w[j] - firm_autarky.realized_operating_profit / cf.w_prime
    end
    return ell_ref
end

@testset "Focal-link O(W*D)->O(W+D) reformation regression (2026-07-30)" begin
    FIXTURE = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
    obj, theta0 = build_melitz_psi_bundle(FIXTURE; outer_parameterization=:logcutoff,
        policy=CappedEvaluation(10.0), backend=:matrix_free, forbid_dense_fallback=true)
    ctx = obj.γ
    D = ctx.D

    function build_state(theta)
        A, f, gamma_prime_j, f_jj = melitz_expand_theta(theta, ctx)
        primitives = MelitzPrimitives(D, ctx.sigma, ctx.theta_star, ctx.target_country,
                                       ctx.tau, ctx.w, A, f, gamma_prime_j)
        cutoff = melitz_baseline_cutoff(A, f, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma)
        eq = MelitzEquilibrium(ctx.expenditure, ones(Float64, D), cutoff, ctx.X_data)
        expenditure_prime = ctx.w_prime * ctx.L[ctx.target_country]
        cf = MelitzCounterfactual(ctx.target_country, ctx.w_prime, expenditure_prime, 1.0, expenditure_prime)
        return primitives, eq, cf
    end

    rng = MersenneTwister(99)
    @testset "op.ell agrees with the reference to machine precision at random points" begin
        for trial in 1:8
            theta = theta0 .+ 0.05 .* randn(rng, length(theta0))
            p, eq, cf = build_state(theta)
            melitz_update_moment_operator!(obj.op, p, eq, cf; X_data=ctx.X_data)
            ell_fast = copy(obj.op.ell)
            ell_ref = zeros(length(ell_fast))
            melitz_focal_link_ell_reference!(ell_ref, obj.op, p, eq, cf)
            relerr = maximum(abs.(ell_fast .- ell_ref)) / maximum(abs.(ell_ref))
            @test relerr < 1e-9
        end
    end

    @testset "Full inner solve still reaches FiniteSolved with the fast focal-link path" begin
        session = MelitzInnerSession(obj, ctx, CappedEvaluation(10.0))
        r = solve_melitz_delta!(session, theta0, CappedEvaluation(10.0))
        @test r isa FiniteSolved
    end

    melitz_update_operator_at_theta!(obj.op, theta0, ctx)
end

println("\nSTANDALONE FOCAL-LINK REGRESSION TEST COMPLETE")
