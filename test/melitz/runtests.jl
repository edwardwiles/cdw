# Formal test suite for the full-D Melitz Christensen-Connault benchmark, ACTIVE
# minimal-moment / LFD-recovery closure (superseding the f_entry-primitive / N-derived /
# N'=N closure -- see docs/melitz_delta_star.md Section 16 for the full list of superseded
# claims). Run with: julia --project=. test/melitz/runtests.jl
#
# Testset organization:
#   1. Pareto draws (Halton + pseudorandom), firm-level calcs -- UNCHANGED from the
#      superseded branch (firm_quantities.jl was reused verbatim).
#   2. Gravity coefficient vector + pivot machinery (NEW).
#   3. Origin-scale N_o=1 normalization invariance (NEW, addendum Sec 3).
#   4. Fixture construction: gravity-exactness, feasibility, D^2+1 moment layout (NEW).
#   5. Active moment system structure + perturbation sensitivity (D^2+1, NEW).
#   6. Exact-sample correction solve (fstar_solver.jl, NEW).
#   7. CC inner minimum-divergence loop + LFD recovery (real KNITRO, NEW).
#   8. Ex-post omitted-equilibrium-equation checks under the recovered LFD (NEW).
#   9. Nearby gravity-feasible perturbations (NEW).
#  10. Legacy Pareto-fixed-N closure -- ARCHIVED diagnostics only (pareto_* functions),
#      never exercised via the active path; kept so the addendum Sec 15 cross-check has
#      something to cross-check against.

using Test
using Random
using Statistics: std, mean
using LinearAlgebra: dot, norm

const MELITZ_DIR = joinpath(@__DIR__, "..", "..", "src", "melitz")
include(joinpath(dirname(dirname(@__DIR__)), "misc", "doubleDiff.jl"))
include(joinpath(MELITZ_DIR, "types.jl"))
include(joinpath(MELITZ_DIR, "pareto.jl"))
include(joinpath(MELITZ_DIR, "firm_quantities.jl"))
include(joinpath(MELITZ_DIR, "equilibrium.jl"))
include(joinpath(MELITZ_DIR, "moments.jl"))
include(joinpath(MELITZ_DIR, "delta_star.jl"))
include(joinpath(MELITZ_DIR, "fake_data.jl"))
include(joinpath(MELITZ_DIR, "fstar_solver.jl"))
include(joinpath(MELITZ_DIR, "fstar_direct.jl"))
include(joinpath(MELITZ_DIR, "gradient_lab.jl"))
include(joinpath(MELITZ_DIR, "outer_solve.jl"))
include(joinpath(MELITZ_DIR, "finite_delta_outer.jl"))

# ============================================================================
# 1. Pareto draws
# ============================================================================
@testset "Pareto draws" begin
    theta_star, sigma = 6.8, 2.5

    @testset "Halton (default): support >= 1, deterministic given seed" begin
        z1 = pareto_draws(5000, 4, theta_star; seed=1)
        @test all(z1 .>= 1.0)
        z2 = pareto_draws(5000, 4, theta_star; seed=1)
        @test z1 == z2
        z3 = pareto_draws(5000, 4, theta_star; seed=2)
        @test z1 != z3
    end

    @testset "pseudorandom mode: support >= 1, deterministic, differs from Halton" begin
        zp1 = pareto_draws(5000, 4, theta_star; seed=1, mode=:pseudorandom)
        @test all(zp1 .>= 1.0)
        zp2 = pareto_draws(5000, 4, theta_star; seed=1, mode=:pseudorandom)
        @test zp1 == zp2
        zh = pareto_draws(5000, 4, theta_star; seed=1, mode=:halton)
        @test zp1 != zh
    end

    @testset "unknown mode errors" begin
        @test_throws ArgumentError pareto_draws(10, 4, theta_star; seed=1, mode=:bogus)
    end

    @testset "sample moments converge to known Pareto moments (Halton)" begin
        z = pareto_draws(200_000, 4, theta_star; seed=7)
        theory_mean = theta_star / (theta_star - 1)
        @test isapprox(mean(z), theory_mean; rtol=0.01)
    end

    @testset "theta_star > sigma-1 enforced" begin
        @test_throws ArgumentError MelitzPrimitives(2, sigma, 1.2, 1, [1.0 1.2; 1.3 1.0],
            ones(2), ones(2, 2), ones(2, 2) * 0.1, 0.8) # theta_star=1.2 < sigma-1=1.5
    end

    @testset "analytical tail moment vs numerical integration" begin
        for (zhat, s, th) in [(1.0, 2.5, 6.8), (1.5, 2.5, 6.8), (2.0, 3.0, 5.0), (1.0, 1.5, 2.0)]
            a = pareto_tail_power_mean(zhat, s, th)
            n = pareto_tail_power_mean_numeric(zhat, s, th)
            @test isapprox(a, n; rtol=1e-6)
        end
    end
end

# ============================================================================
# 2. Firm-quantity tests (unchanged logic, reused verbatim from firm_quantities.jl)
# ============================================================================
@testset "Firm-level Melitz calculations" begin
    w_o, tau_od, A_od, f_od, sigma = 1.2, 1.3, 0.8, 0.05, 2.5
    expenditure_d, price_power_d = 10.0, 1.0

    @testset "higher productivity lowers marginal cost and price" begin
        @test marginal_cost(w_o, tau_od, A_od, 2.0) < marginal_cost(w_o, tau_od, A_od, 1.0)
        @test melitz_price(w_o, tau_od, A_od, sigma, 2.0) < melitz_price(w_o, tau_od, A_od, sigma, 1.0)
    end

    @testset "higher productivity raises unconstrained revenue" begin
        p1 = melitz_price(w_o, tau_od, A_od, sigma, 1.0)
        p2 = melitz_price(w_o, tau_od, A_od, sigma, 2.0)
        r1 = unconstrained_revenue(p1, sigma, expenditure_d, price_power_d)
        r2 = unconstrained_revenue(p2, sigma, expenditure_d, price_power_d)
        @test r2 > r1
    end

    @testset "increasing A lowers marginal cost" begin
        @test marginal_cost(w_o, tau_od, 2.0, 1.0) < marginal_cost(w_o, tau_od, 0.5, 1.0)
    end

    C_od = melitz_C(w_o, tau_od, A_od, sigma, expenditure_d)
    zhat = melitz_cutoff(w_o, f_od, sigma, C_od)

    @testset "firm at cutoff has zero operating profit" begin
        res = melitz_firm(w_o, tau_od, A_od, f_od, sigma, expenditure_d, price_power_d, zhat)
        @test abs(res.operating_profit) < 1e-9
    end

    @testset "just below cutoff does not sell" begin
        res = melitz_firm(w_o, tau_od, A_od, f_od, sigma, expenditure_d, price_power_d, zhat * 0.999)
        @test res.active == false
        @test res.realized_revenue == 0.0
        @test res.realized_operating_profit == 0.0
    end

    @testset "just above cutoff sells" begin
        res = melitz_firm(w_o, tau_od, A_od, f_od, sigma, expenditure_d, price_power_d, zhat * 1.001)
        @test res.active == true
        @test res.realized_revenue > 0.0
        @test res.realized_operating_profit > 0.0
    end
end

# ============================================================================
# 3. Gravity coefficient vector + pivot machinery
# ============================================================================
@testset "Gravity coefficient vector and pivot machinery" begin
    D = 4
    rng = MersenneTwister(99)
    tau = ones(Float64, D, D)
    for o in 1:D, d in 1:D
        o == d && continue
        tau[o, d] = exp(0.1 + 0.4 * rand(rng))
    end
    c = gravity_coefficient_vector(D, tau)
    T = withinTransform(tau)

    @testset "c reproduces sum(T.*withinTransform(X)) == dot(c, vec(log X)) for random X" begin
        for _ in 1:5
            X = exp.(0.5 .* randn(rng, D, D))
            lhs = sum(T .* withinTransform(X))
            rhs = dot(c, vec(log.(X)))
            @test isapprox(lhs, rhs; rtol=1e-8, atol=1e-10)
        end
    end

    @testset "homogeneous: c is nonzero (D>=3), residual at X=ones is exactly 0" begin
        @test any(!=(0.0), c)
        @test sum(T .* withinTransform(ones(D, D))) == 0.0
    end

    @testset "Gate A5: c == vec(withinTransform(tau)) exactly (self-adjointness of the two-way FE projection)" begin
        @test isapprox(c, vec(T); rtol=1e-8, atol=1e-10)
    end

    @testset "Gate A5: withinTransform matches the true OLS origin+destination-FE gravity coefficient; doubleDiff does not" begin
        # Regression test for the Gate A5 finding: production/fullA-exact's own canonical
        # gravity moment (moments/newGravityMoment!.jl, UoModel==1 branch -- universal in
        # every production run config in this repo) is `withinTransform`-based, verified
        # (via this repo's own pre-existing gravity_check.jl, independently re-run) to
        # match an explicit OLS-with-two-way-FE-dummies regression coefficient to machine
        # precision, while the ORIGINAL `doubleDiff`-based choice here does not (off by up
        # to ~2.3 in the coefficient, occasionally even the wrong SIGN).
        function ols_fe_coef(x, y)
            Dl = size(x, 1); n = Dl * Dl
            xv = vec(x); yv = vec(y)
            Xcols = [ones(n), xv]
            for o in 2:Dl
                push!(Xcols, [(((i - 1) % Dl) + 1) == o ? 1.0 : 0.0 for i in 1:n])
            end
            for d in 2:Dl
                push!(Xcols, [(((i - 1) ÷ Dl) + 1) == d ? 1.0 : 0.0 for i in 1:n])
            end
            return (hcat(Xcols...) \ yv)[2]
        end
        rng2 = MersenneTwister(123)
        max_within_err = 0.0
        max_dd_err = 0.0
        for _ in 1:200
            lntau = randn(rng2, D, D)
            lnA = randn(rng2, D, D)
            b_fe = ols_fe_coef(lntau, lnA)
            Wt, Wa = withinTransform(exp.(lntau)), withinTransform(exp.(lnA))
            b_within = sum(Wt .* Wa) / sum(Wt .* Wt)
            Dt, Da = doubleDiff(exp.(lntau)), doubleDiff(exp.(lnA))
            b_dd = sum(Dt .* Da) / sum(Dt .* Dt)
            max_within_err = max(max_within_err, abs(b_within - b_fe))
            max_dd_err = max(max_dd_err, abs(b_dd - b_fe))
        end
        @test max_within_err < 1e-8
        @test max_dd_err > 0.1 # NOT a good estimator of the FE coefficient -- documents why it was replaced
    end

    @testset "pivot_expand/pivot_reduce round-trip and satisfy the affine constraint" begin
        gp = build_gravity_pivot(c, 0.0)
        z_free = randn(rng, D^2 - 1)
        z_full = pivot_expand(z_free, gp)
        @test isapprox(dot(c, z_full), 0.0; atol=1e-8)
        @test pivot_reduce(z_full, gp) == z_free
    end

    @testset "pivot honors avoid= (excludes a candidate cell from pivot selection)" begin
        gp_free = build_gravity_pivot(c, 0.0)
        gp_avoid = build_gravity_pivot(c, 0.0; avoid=gp_free.pivot)
        @test gp_avoid.pivot != gp_free.pivot
    end

    @testset "project_to_gravity_manifold satisfies the constraint and is minimum-norm" begin
        z_raw = randn(rng, D^2)
        g0 = 0.37
        z_proj = project_to_gravity_manifold(z_raw, c, g0)
        @test isapprox(dot(c, z_proj) + g0, 0.0; atol=1e-8)
        # minimum-norm: the correction (z_proj - z_raw) is parallel to c
        correction = z_proj .- z_raw
        @test isapprox(abs(dot(correction, c)), norm(correction) * norm(c); rtol=1e-8)
    end

    # ========================================================================
    # 2026-07-22 session Section 1.1: off-diagonal, mutually-distinct outer gravity
    # pivots. Regression tests for the fix -- before this session, the active outer
    # expansion (delta_star.jl's build_gravity_pivots/reduce_to_free_theta/
    # expand_free_theta) excluded ONLY the A-pivot cell when choosing the f-pivot, never
    # restricting either pivot to off-diagonal cells. Under Gate A5's withinTransform
    # coefficient vector (diagonal-dominated, see f_pivot_domestic_avoid_indices), the
    # UNRESTRICTED A-pivot lands on a domestic cell, entangling that country's own
    # cutoffs with the gravity-feasibility construction.
    # ========================================================================
    @testset "Section 1.1: off-diagonal, mutually-distinct A/f gravity pivots" begin
        target_country = 1
        for seed in (7, 29, 99, 123)
            rng3 = MersenneTwister(seed)
            tau3 = ones(Float64, D, D)
            for o in 1:D, d in 1:D
                o == d && continue
                tau3[o, d] = exp(0.05 + 0.4 * rand(rng3))
            end
            A_pivot_od, f_pivot_od = gravity_pivot_cells(tau3, target_country)

            @testset "seed=$seed: o != d for both pivots" begin
                @test A_pivot_od[1] != A_pivot_od[2]
                @test f_pivot_od[1] != f_pivot_od[2]
            end
            @testset "seed=$seed: A pivot != f pivot" begin
                @test A_pivot_od != f_pivot_od
            end
            @testset "seed=$seed: neither pivot is (j,j)" begin
                @test A_pivot_od != (target_country, target_country)
                @test f_pivot_od != (target_country, target_country)
            end
        end
    end

    @testset "Section 1.1: reduce(expand(theta))==theta and expand(reduce(p))==p round-trip through the FIXED off-diagonal pivots" begin
        data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=1_000)
        p = data.primitives
        moment_layout = MelitzMomentLayout(D)
        c_full3, A_pivot3 = build_gravity_pivots(p.tau, p.target_country)
        outer_layout3 = melitz_outer_layout(D, p.target_country)
        ctx3 = (D=D, sigma=p.sigma, theta_star=p.theta_star, target_country=p.target_country,
            tau=p.tau, w=p.w, w_prime=data.counterfactual.w_prime, L=data.L,
            expenditure=data.equilibrium.expenditure, cutoff=data.equilibrium.cutoff,
            moment_layout=moment_layout, X_data=data.equilibrium.trade_flow, c_full=c_full3,
            A_pivot=A_pivot3, jj_lin=outer_layout3.jj_lin, f_free_lin=outer_layout3.f_free_lin)

        theta_free = reduce_to_free_theta(p, ctx3)

        @testset "expand(reduce(p)) reconstructs p exactly" begin
            A_rt, f_rt, gamma_rt, f_jj_rt = expand_free_theta(theta_free, ctx3)
            @test isapprox(A_rt, p.A; rtol=1e-10)
            @test isapprox(f_rt, p.f; rtol=1e-10)
            @test isapprox(gamma_rt, p.gamma_prime_target; rtol=1e-10)
            @test isapprox(f_jj_rt, p.f[p.target_country, p.target_country]; rtol=1e-10)
        end

        @testset "reduce(expand(theta))==theta" begin
            A_e, f_e, gamma_e, _ = expand_free_theta(theta_free, ctx3)
            p_e = MelitzPrimitives(D, p.sigma, p.theta_star, p.target_country, p.tau, p.w, A_e, f_e, gamma_e)
            theta_rt = reduce_to_free_theta(p_e, ctx3)
            @test isapprox(theta_rt, theta_free; rtol=1e-10, atol=1e-10)
        end

        @testset "both gravity residuals at machine precision at the round-tripped point" begin
            A_rt, f_rt, gamma_rt, _ = expand_free_theta(theta_free, ctx3)
            p_rt = MelitzPrimitives(D, p.sigma, p.theta_star, p.target_country, p.tau, p.w, A_rt, f_rt, gamma_rt)
            res_A, res_f = gravity_residuals(p_rt)
            @test abs(res_A) < 1e-8
            @test abs(res_f) < 1e-8
        end
    end

    @testset "Section 1.1: pivot_conditioning_diagnostics (conditioning-only comparison)" begin
        diag = pivot_conditioning_diagnostics(tau, 1)
        # Under Gate A5's withinTransform coefficient vector, the unrestricted max-|c|
        # pivot is expected to land on a diagonal cell at this D=4 fixture (the
        # diagonal-dominance finding motivating this whole fix) -- not asserted as a
        # hard requirement (a different random tau could avoid it), only reported.
        @test diag.restricted_pivot_od[1] != diag.restricted_pivot_od[2]
        @test isfinite(diag.unrestricted_max_leverage)
        @test isfinite(diag.restricted_max_leverage)
        @test diag.orthonormal_leverage == 1.0
    end
end

# ============================================================================
# 4. Origin-scale N_o=1 normalization invariance (addendum Section 3)
# ============================================================================
@testset "normalize_baseline_entrant_mass invariance" begin
    D = 4
    sigma, theta_star = 2.5, 6.8
    rng = MersenneTwister(7)
    N_old = 0.4 .+ 1.5 .* rand(rng, D)
    w = 0.7 .+ 0.9 .* rand(rng, D)
    tau = ones(D, D)
    for o in 1:D, d in 1:D
        o == d && continue
        tau[o, d] = exp(0.1 + 0.4 * rand(rng))
    end
    A = exp.(randn(rng, D, D))
    f = exp.(randn(rng, D, D) .- 1.0)
    f_entry = 0.1 .+ 0.3 .* rand(rng, D)
    N_prime_old = [0.6]

    zhat_old = zeros(D, D)
    X_old = zeros(D, D)
    expenditure = 5.0 .+ 5.0 .* rand(rng, D)
    for o in 1:D, d in 1:D
        C_od = melitz_C(w[o], tau[o, d], A[o, d], sigma, expenditure[d])
        zhat_old[o, d] = melitz_cutoff(w[o], f[o, d], sigma, C_od)
        X_old[o, d] = N_old[o] * C_od * pareto_tail_power_mean(zhat_old[o, d], sigma, theta_star)
    end

    A_new, f_new, f_entry_new, N_prime_new = normalize_baseline_entrant_mass(
        N_old, A, f, f_entry, sigma; N_prime=N_prime_old)

    @testset "N'_new == 1 exactly (self-consistency of the closed-form transform)" begin
        N_check = [pareto_entrant_mass_from_labor(1.0, f_entry_new[o], sigma, theta_star) for o in 1:D]
        # N_o_new = (sigma-1)/(sigma*theta_star) * L_o/f_entry_new_o; solving the identity
        # backwards confirms f_entry_new is exactly N_old .* f_entry_old (docs formula)
        @test isapprox(f_entry_new, N_old .* f_entry, rtol=1e-12)
    end

    @testset "trade flows and cutoffs unchanged after normalization" begin
        zhat_new = zeros(D, D)
        X_new = zeros(D, D)
        for o in 1:D, d in 1:D
            C_od = melitz_C(w[o], tau[o, d], A_new[o, d], sigma, expenditure[d])
            zhat_new[o, d] = melitz_cutoff(w[o], f_new[o, d], sigma, C_od)
            X_new[o, d] = 1.0 * C_od * pareto_tail_power_mean(zhat_new[o, d], sigma, theta_star)
        end
        @test isapprox(zhat_new, zhat_old; rtol=1e-10)
        @test isapprox(X_new, X_old; rtol=1e-10)
    end

    @testset "gravity restrictions unchanged" begin
        p_old = MelitzPrimitives(D, sigma, theta_star, 1, tau, w, A, f, 0.8)
        p_new = MelitzPrimitives(D, sigma, theta_star, 1, tau, w, A_new, f_new, 0.8)
        @test isapprox(gravity_residuals(p_old)[1], gravity_residuals(p_new)[1]; atol=1e-8)
        @test isapprox(gravity_residuals(p_old)[2], gravity_residuals(p_new)[2]; atol=1e-8)
    end

    @testset "counterfactual mass rescales by 1/N_o" begin
        @test isapprox(N_prime_new[1], N_prime_old[1] / N_old[1]; rtol=1e-12)
    end
end

# ============================================================================
# 5. Fixture construction (fake_data.jl): gravity-exactness, feasibility, N_o=1
# ============================================================================
const FIXTURE = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
    target_country=1, seed=29, W=20_000)
const LAYOUT = MelitzMomentLayout(FIXTURE.primitives.D)

@testset "Fixture construction (active N_o=1, searched gamma_prime_j closure)" begin
    p, eq, cf = FIXTURE.primitives, FIXTURE.equilibrium, FIXTURE.counterfactual

    @testset "both gravity restrictions exact to machine precision" begin
        rA, rf = gravity_residuals(p)
        @test abs(rA) < 1e-8
        @test abs(rf) < 1e-8
    end

    @testset "cutoff feasibility: zhat>=1 everywhere, export>=domestic" begin
        @test all(eq.cutoff .>= 1.0)
        D = p.D
        for o in 1:D, d in 1:D
            d == o && continue
            @test eq.cutoff[o, d] >= eq.cutoff[o, o]
        end
    end

    @testset "focal autarky cutoff is exactly 1 by construction" begin
        @test isapprox(cf.cutoff_prime, 1.0; atol=1e-12)
    end

    @testset "gamma_prime_target is a searched primitives field, f[j,j] excluded from any packed f_entry" begin
        @test p.gamma_prime_target > 0
        @test !isapprox(p.gamma_prime_target, 1.0; atol=1e-6) # nontrivial, not accidentally normalized away
    end

    @testset "N'[target] not imposed -- NaN until recovered from the LFD" begin
        @test isnan(cf.entrant_mass_prime)
    end

    @testset "heterogeneity: A, f genuinely vary, never trivially uniform" begin
        @test std(log.(p.A)) > 0.01
        @test std(log.(p.f)) > 0.01
    end
end

@testset "Autarky price-power fix: economic zero-profit residual (not just cutoff_prime==1)" begin
    # `counterfactual.cutoff_prime == 1` alone would have passed even with the bug (it is a
    # NORMALIZATION choice, set before f[j,j] is derived -- it says nothing about whether the
    # productivity-one firm's REALIZED autarky profit is actually zero at that cutoff). The
    # bug was autarky revenue evaluated at price_power_d=1.0 instead of gamma_prime_j, which
    # is only caught by evaluating the firm's own zero-profit condition directly.
    sigma = 2.5
    for (gamma_prime, A_jj, expenditure_prime) in
        [(0.7561231654704139, 3.4, 1.9), (0.35, 0.8, 5.2), (1.6, 12.0, 0.4), (0.95, 1.0, 1.0)]
        w_prime_j = 1.0
        f_jj = derive_fjj_from_autarky_cutoff(gamma_prime, w_prime_j, 1.0, A_jj, expenditure_prime, sigma)

        @testset "gamma_prime=$gamma_prime, A_jj=$A_jj, expenditure_prime=$expenditure_prime" begin
            # Correct: price_power_d = gamma_prime_j -- x'_jj(1)/sigma - w'_j*f_jj == 0 exactly.
            firm_correct = melitz_firm(w_prime_j, 1.0, A_jj, f_jj, sigma, expenditure_prime, gamma_prime, 1.0)
            @test isapprox(firm_correct.operating_profit, 0.0; atol=1e-10)

            # Buggy: price_power_d = 1.0 -- residual is EXACTLY (gamma_prime-1)*w'_j*f_jj, not 0
            # (unless gamma_prime happens to be 1). This is the algebraic signature that
            # caught the bug live (docs Section 12: gamma_prime_target=0.7561..., residual
            # -1.6959... == (0.7561...-1)*6.9541...).
            firm_buggy = melitz_firm(w_prime_j, 1.0, A_jj, f_jj, sigma, expenditure_prime, 1.0, 1.0)
            expected_buggy_residual = (gamma_prime - 1) * w_prime_j * f_jj
            @test isapprox(firm_buggy.operating_profit, expected_buggy_residual; rtol=1e-10)
        end
    end
end

@testset "Full-D indexing (no rest-of-world aggregation)" begin
    p = FIXTURE.primitives
    D = p.D

    @testset "exactly D^2+1 moments, D^2 trade cells each exactly once + 1 link moment" begin
        @test LAYOUT.num_moments == D^2 + 1
        all_trade_cols = vec(LAYOUT.trade_index)
        @test length(unique(all_trade_cols)) == D^2
        @test Set(all_trade_cols) == Set(1:D^2)
        @test LAYOUT.focal_link_index == D^2 + 1
    end

    @testset "no origin or destination cell silently skipped" begin
        for o in 1:D, d in 1:D
            @test LAYOUT.trade_index[o, d] != 0
        end
    end
end

# ============================================================================
# 6. Active moment system structure + perturbation sensitivity
# ============================================================================
@testset "melitz_moments! structure and sensitivity" begin
    p, eq, cf = FIXTURE.primitives, FIXTURE.equilibrium, FIXTURE.counterfactual
    D = p.D
    Wtest = 500
    z = FIXTURE.z_draws[1:Wtest, :]
    K = zeros(Wtest)
    G1 = zeros(Wtest, LAYOUT.num_moments)
    melitz_moments!(K, G1, p, eq, cf, z, LAYOUT)

    @testset "changing a single A[o,d] (o != target) affects only trade[o,d] (and possibly the link col if o==target)" begin
        o, d = 2, 3
        p2_A = copy(p.A)
        p2_A[o, d] *= 5.0
        p2 = MelitzPrimitives(D, p.sigma, p.theta_star, p.target_country, p.tau, p.w, p2_A, p.f, p.gamma_prime_target)
        G2 = zeros(Wtest, LAYOUT.num_moments)
        melitz_moments!(K, G2, p2, eq, cf, z, LAYOUT)
        changed_cols = [c for c in 1:LAYOUT.num_moments if !isapprox(G1[:, c], G2[:, c]; atol=1e-10)]
        expected = o == p.target_country ? Set([LAYOUT.trade_index[o, d], LAYOUT.focal_link_index]) : Set([LAYOUT.trade_index[o, d]])
        @test Set(changed_cols) == expected
    end

    @testset "K equals gamma_prime_target - 1 (documented placeholder, unconsumed by inner_loop)" begin
        @test all(isapprox.(K, p.gamma_prime_target - 1; atol=1e-12))
    end

    @testset "focal_link_index changes when target country's f changes" begin
        j = p.target_country
        p2_f = copy(p.f)
        p2_f[j, 2] *= (j == 1 ? 2.0 : 1.0) # perturb a non-domestic focal export cost
        j == 1 || (p2_f[j, 1] *= 2.0)
        p2 = MelitzPrimitives(D, p.sigma, p.theta_star, j, p.tau, p.w, p.A, p2_f, p.gamma_prime_target)
        G2 = zeros(Wtest, LAYOUT.num_moments)
        melitz_moments!(K, G2, p2, eq, cf, z, LAYOUT)
        @test !isapprox(G1[:, LAYOUT.focal_link_index], G2[:, LAYOUT.focal_link_index]; atol=1e-10)
    end
end

@testset "min_active_draw_count and cell_participation_diagnostics" begin
    p, eq = FIXTURE.primitives, FIXTURE.equilibrium
    mc, worst = min_active_draw_count(p, eq, FIXTURE.z_draws)
    @test mc >= 0
    @test worst[1] >= 1 && worst[2] >= 1

    diag = cell_participation_diagnostics(p, eq, FIXTURE.z_draws)
    @test size(diag.reference_probability) == (p.D, p.D)
    @test all(0.0 .<= diag.reference_probability .<= 1.0)
end

# ============================================================================
# 7. Population-Pareto construction identities (addendum Sections 2, 6, 8) -- REPLACES the
# superseded exact-sample-correction benchmark. `fstar_solver.jl`/`solve_fstar` is ARCHIVED
# per addendum Section 1 ("do not force exact-sample feasibility") -- kept in the repo as an
# optional debugging utility only, no longer exercised by the active/default suite. A single
# smoke test below confirms it still runs without crashing (structural-only, no economic
# claim), separated from the OPTIONAL STRESS TESTS section.
# ============================================================================
@testset "Population-Pareto construction: exact population-level identities" begin
    p, eq, cf, L = FIXTURE.primitives, FIXTURE.equilibrium, FIXTURE.counterfactual, FIXTURE.L
    j = p.target_country

    @testset "baseline factor-market clearing (iceberg closure, no tariff-revenue term)" begin
        @test maximum(abs.(p.w .* L .- vec(sum(eq.trade_flow, dims=2)))) < 1e-8
    end

    @testset "baseline price-index normalization gamma_d==1 holds exactly (not just tautologically)" begin
        @test maximum(abs.(eq.price_power .- 1.0)) < 1e-8
    end

    @testset "population focal free-entry link residual vanishes at the SOLVED gamma_prime_target" begin
        f_jj = p.f[j, j]
        @test abs(population_focal_link_residual(p, eq.trade_flow, eq.cutoff, f_jj, cf)) < 1e-8
    end

    @testset "autarky zero-profit at z=1 holds using the GE-final A[j,j] (not the pre-rescale draw)" begin
        # regression test for a real bug: the GE solve's per-destination gamma_d==1
        # rescaling also rescales column j of A (country j is itself a destination), so
        # f[j,j] must be derived self-consistently against the FINAL A[j,j], not the
        # pre-rescale value -- found live as a ~0.28 nonzero z=1 profit (main prompt
        # Section 1.2's own unit test only covers the algebraic formula in isolation, not
        # this GE-interaction bug).
        firm_at_one = melitz_firm(cf.w_prime, 1.0, p.A[j, j], p.f[j, j], p.sigma, cf.expenditure_prime,
            p.gamma_prime_target, 1.0)
        @test isapprox(firm_at_one.operating_profit, 0.0; atol=1e-10)
    end

    @testset "W-convergence: equal-weight raw moment residuals shrink with W (addendum Section 3)" begin
        residuals_at_W = Dict{Int,Float64}()
        link_at_W = Dict{Int,Float64}()
        for W in (5_000, 20_000, 80_000)
            data_w = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=W)
            pw, eqw, cfw = data_w.primitives, data_w.equilibrium, data_w.counterfactual
            residuals_at_W[W] = maximum(abs.(trade_flow_residuals(pw, eqw, data_w.z_draws) ./ eqw.expenditure'))
            link_at_W[W] = abs(focal_link_residual(pw, eqw, cfw, data_w.z_draws))
        end
        @test residuals_at_W[80_000] < residuals_at_W[20_000] < residuals_at_W[5_000]
        # link residual is small at every W (it is not merely declining -- it is ~0 in
        # population by construction, so even W=5,000 should already be small, contrast
        # the uncalibrated-gamma_prime_target regression this guards: that bug left it
        # pinned around -0.3 to -0.36 REGARDLESS of W)
        @test all(v -> v < 0.01, values(link_at_W))
    end
end

if get(ENV, "MELITZ_RUN_SLOW_TESTS", "false") == "true"
    @testset "solve_fstar: archived debugging utility (structural smoke test only)" begin
        # NOT part of the active benchmark (addendum Section 1) -- this only confirms the
        # archived utility still runs without crashing on the current fixture/data shapes, no
        # economic claim about its output is made or required.
        # Gated behind MELITZ_RUN_SLOW_TESTS: this smoke test's own bisection-based solve
        # takes ~20 minutes wall time post Gate A5's seed/transform change (docs Section
        # 14.7) -- too slow for ordinary development test runs of an ARCHIVED, inactive
        # utility. Run with MELITZ_RUN_SLOW_TESTS=true julia --project=. test/melitz/runtests.jl
        # to exercise it.
        medium_fixture = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=20_000)
        result = solve_fstar(medium_fixture; sweeps=2)
        @test result isa MelitzFStarResult
    end
else
    @info "Skipping solve_fstar smoke test (~20min, archived utility) -- set MELITZ_RUN_SLOW_TESTS=true to run it"
end

# ============================================================================
# 7-8. CC inner minimum-divergence loop + LFD recovery + ex-post checks (real KNITRO)
# ============================================================================
const KNITRO_AVAILABLE = try
    include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "include_cc_algo.jl"))
    @eval using .CounterfactualSensitivity
    true
catch e
    @warn "Skipping CC inner-loop testsets: cc_algo/KNITRO not available in this environment" exception = e
    false
end

if KNITRO_AVAILABLE
    @testset "CC inner minimum-divergence loop + LFD recovery (real KNITRO)" begin
        # Directly on the RAW population-Pareto fixture -- NO exact-sample correction
        # (addendum Section 1: solve_fstar is archived, not part of the active benchmark).
        # W=80,000 matches this repo's own established threshold for reliably avoiding
        # zero-active-draw cells at D=4 (min active count ~1,393 here, comfortably above).
        # This REPLACES the previously-documented nStatus=-102/Delta=1e10 failure: that
        # failure was caused by the exact-sample-correction fixture being internally
        # inconsistent (main prompt's own diagnosis, since confirmed: the autarky
        # price-power bug PLUS the A[j,j]/f[j,j] GE self-consistency bug below), not a
        # genuine finite-support infeasibility of the CC inner problem.
        big_fixture = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=80_000)

        lfd, obj = run_melitz_inner_delta(big_fixture;
            inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"))

        @testset "KNITRO reaches a genuinely bounded optimum (nStatus==0), not the previous -102" begin
            @test lfd.nStatus == 0
        end

        @testset "Delta(theta*) is small and positive (population target, not exact-sample zero)" begin
            @test 0 < lfd.Delta < 1e-4
        end

        @testset "LFD reconstruction is a valid probability distribution" begin
            @test lfd.lfd_ok
            @test isapprox(sum(lfd.weights), 1.0; atol=1e-8)
            @test all(w -> w >= 0.0, lfd.weights)
            @test abs(lfd.normalization_residual) < 1e-6
        end

        @testset "LFD zeroes every imposed moment" begin
            @test maximum(abs.(lfd.moment_residuals)) < 1e-6
        end

        @testset "Gate A4: primal-dual diagnostics stored explicitly, scale-aware gap gate" begin
            @test lfd.dual_divergence == lfd.Delta
            @test isapprox(lfd.primal_divergence, lfd.dual_divergence; atol=1e-6)
            @test lfd.primal_dual_gap == abs(lfd.primal_divergence - lfd.dual_divergence)
            gap_tol = max(1e-10, 1e-6 * max(1.0, abs(lfd.primal_divergence), abs(lfd.dual_divergence)))
            @test lfd.primal_dual_gap <= gap_tol
            @test lfd.maximum_weighted_moment_residual == maximum(abs.(lfd.moment_residuals))
            @test lfd.probability_normalization_residual == lfd.normalization_residual
            @test isfinite(lfd.kkt_opt_error) && lfd.kkt_opt_error < 1e-6
            @test isfinite(lfd.kkt_feas_error) && lfd.kkt_feas_error < 1e-6
        end

        @testset "Gate A3: every omitted-equilibrium-equation residual, tight explicit tolerances" begin
            # Classification (main prompt Section A3), NOT merely a finiteness check:
            #   IMPLIED    -- a linear combination of already-imposed moments (must hold
            #                 whenever moment_tol does; bounded by D*moment_tol, not
            #                 independently informative).
            #   DEFINITIONAL -- the residual of a quantity DEFINED to make it zero (e.g.
            #                 f_entry := E[Pi]/w); confirms the recovery formula was
            #                 implemented correctly, not a property of the LFD itself.
            #   INDEPENDENT -- a genuine cross-check tying together two formulas/blocks
            #                 that are not algebraically forced to agree.
            p, eq, cf = big_fixture.primitives, big_fixture.equilibrium, big_fixture.counterfactual
            j = p.target_country
            check = check_profiled_melitz_equilibrium(p, eq, cf, big_fixture.z_draws, lfd.weights)
            for fn in fieldnames(typeof(check))
                val = getfield(check, fn)
                @test all(isfinite, val isa AbstractArray ? val : [val])
            end

            @testset "IMPLIED by imposed trade-share moments (9.1)" begin
                # residual_gamma_baseline[d] == sum_o moment_residual[trade_index[o,d]]
                # exactly (verified algebraically in equilibrium.jl's docstring); bounded
                # by D * moment_tol=1e-6, so 1e-8 is a genuine (if redundant) check here.
                @test maximum(abs.(check.residual_gamma_baseline)) <= 1e-8
            end

            @testset "DEFINITIONAL profiling identities (9.2/9.3, 7.1)" begin
                @test maximum(abs.(check.residual_free_entry_baseline)) <= 1e-10
                @test abs(check.residual_free_entry_autarky) <= 1e-10
            end

            @testset "IMPLIED by the focal free-entry LINK moment (ties baseline/autarky f_entry)" begin
                @test abs(check.f_entry_recovered[j] - check.f_entry_autarky_recovered) <= 1e-8
            end

            @testset "INDEPENDENT cross-checks (9.4/9.5, the genuinely informative tests)" begin
                @test abs(check.residual_market_clearing_autarky) <= 1e-8
                @test abs(check.residual_gamma_autarky) <= 1e-8
                @test check.N_prime_diff_rel <= 1e-8
                # Main prompt Section A2: the recovered autarky entrant mass N'[j] should
                # itself be close to 1 under the N[j]==1 normalization (NOT imposed
                # anywhere in the construction -- a real, nontrivial equilibrium property
                # of this closure, confirmed to converge to 1 as W grows, see the
                # W-convergence testset below).
                @test isapprox(check.N_prime_market_clearing, 1.0; atol=1e-3)
            end

            @testset "DEFINITIONAL: autarky cutoff-at-one construction (9.6)" begin
                @test abs(check.residual_autarky_cutoff) <= 1e-10
            end

            @testset "Feasibility inequalities (9.9) and gravity (9.8, exact by pivot construction)" begin
                @test check.min_cutoff_minus_one >= -1e-6
                @test check.min_export_minus_domestic >= -1e-6
                @test abs(check.gravity_residual_A) < 1e-8
                @test abs(check.gravity_residual_f) < 1e-8
            end
        end

        @testset "Gate A1/A2: gains from trade, wage-ratio formula vs ACR/Chaney (population, no LFD needed)" begin
            GT_model = melitz_gains_from_trade(big_fixture.primitives, big_fixture.counterfactual)
            lambda_jj, GT_ACR = acr_gains_from_trade(big_fixture.primitives, big_fixture.equilibrium)
            @test isapprox(GT_model, GT_ACR; atol=1e-8)
            # Regression test for the bug this Gate fixed: omitting the wage ratio changes
            # not just the MAGNITUDE but can flip the SIGN of the reported gains from trade
            # whenever w[target_country] != 1 (verified live at this exact fixture).
            j = big_fixture.primitives.target_country
            w_j = big_fixture.primitives.w[j]
            if !isapprox(w_j, 1.0; atol=1e-6)
                GT_naive_wrong = 1 - big_fixture.primitives.gamma_prime_target^(1 / (big_fixture.primitives.sigma - 1))
                @test !isapprox(GT_model, GT_naive_wrong; atol=1e-4)
            end
        end
    end

    # ========================================================================
    # 2026-07-22 session Section 2: evaluate_melitz_delta + MelitzDeltaEvalCache
    # ========================================================================
    @testset "Section 2: evaluate_melitz_delta authoritative evaluator + cache" begin
        small_fixture = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=20_000)
        obj, theta0 = build_melitz_psi_bundle(small_fixture;
            inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"))
        ctx = obj.γ
        cache = MelitzDeltaEvalCache()

        @testset "at the benchmark point: verified, matches the documented W=20,000 benchmark Delta" begin
            r = evaluate_melitz_delta(theta0, ctx, obj; cache=cache)
            @test r.verified
            @test r.feasible
            @test r.nStatus == 0
            # session prompt's own stated benchmark: Delta(theta_Fstar) = 7.5545e-6 at
            # W=20,000, seed=29 -- exact match confirms melitz_outer_state's freshly
            # computed cutoff reproduces the SAME inner solve as the pre-fix path at this
            # (benchmark) point (only DISPLACED points should ever differ).
            @test isapprox(r.Delta, 7.5545e-6; rtol=1e-3)
            @test r.G !== nothing
            @test size(r.G) == (20_000, ctx.moment_layout.num_moments)
            @test r.equilibrium_check !== nothing
            @test r.state_time >= 0 && r.inner_time > 0
        end

        @testset "cache hit returns the identical stored object, no recomputation" begin
            r1 = evaluate_melitz_delta(theta0, ctx, obj; cache=cache)
            misses_before = cache.misses
            r2 = evaluate_melitz_delta(theta0, ctx, obj; cache=cache)
            @test cache.misses == misses_before
            @test cache.hits >= 1
            @test r1 === r2
        end

        @testset "store_G=false omits the moment matrix but keeps everything else" begin
            r = evaluate_melitz_delta(theta0, ctx, obj; store_G=false)
            @test r.G === nothing
            @test isfinite(r.Delta)
        end

        @testset "a grossly infeasible/failed point is returned but NEVER cached" begin
            cache2 = MelitzDeltaEvalCache()
            rng = MersenneTwister(3)
            theta_bad = theta0 .+ 3.0 .* randn(rng, length(theta0))
            r_bad = evaluate_melitz_delta(theta_bad, ctx, obj; cache=cache2, cold=true)
            @test !r_bad.verified
            @test length(cache2.store) == 0
            # re-evaluating the SAME good point afterward still caches normally (the bad
            # point does not corrupt the cache or obj's warm-start state going forward)
            r_good = evaluate_melitz_delta(theta0, ctx, obj; cache=cache2, cold=true)
            @test r_good.verified
            @test length(cache2.store) == 1
        end
    end

    # ========================================================================
    # 2026-07-22 session Section 3: direct F* feasibility solve (fstar_direct.jl)
    # ========================================================================
    @testset "Section 3: solve_fstar_direct (direct F* feasibility solve)" begin
        small_fixture = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=2_000)
        obj3, theta_pop = build_melitz_psi_bundle(small_fixture;
            inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"))
        ctx3 = obj3.γ

        m_init = fstar_equal_weight_moments(theta_pop, ctx3, obj3)
        @test length(m_init) == ctx3.moment_layout.num_moments

        res = solve_fstar_direct(theta_pop, ctx3, obj3; rho=1e-4, iterations=5, time_limit=30.0)
        @testset "structural: result type, moment vector lengths, non-negative distance" begin
            @test res isa MelitzFStarDirectResult
            @test length(res.m_final) == ctx3.moment_layout.num_moments
            @test res.theta_distance_from_population >= 0
        end
        @testset "cutoff feasibility is preserved (small rho, short run, starts feasible)" begin
            @test res.eval.feasible
        end
        @testset "cold-verified inner solve at the F*-direct result is a real KNITRO call" begin
            @test res.eval.nStatus in (0, -100, -101, -103, -400, -500, -502)  # any real status, not a stub
            @test isfinite(res.eval.Delta)
        end
        @testset "moments did not get WORSE than the population starting point" begin
            # a real (even if not fully converged) improvement step should not increase
            # the worst-case moment residual
            @test res.max_abs_moment_final <= res.max_abs_moment_initial + 1e-8
        end
    end

    # ========================================================================
    # 2026-07-22 session Section 5: gradient laboratory (gradient_lab.jl) -- structural +
    # the core Section 6 Q1 check (analytic/ForwardDiff envelope agrees with a small-h
    # fixed-dual secant when no draw switches), at reduced scale (small W for speed).
    # ========================================================================
    @testset "Section 5: gradient laboratory (Methods A/B/C), structural + zero-switch agreement" begin
        small_fixture = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=2_000)
        obj5, theta0 = build_melitz_psi_bundle(small_fixture;
            inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"))
        ctx5 = obj5.γ

        r_base = evaluate_melitz_delta(theta0, ctx5, obj5; cold=true, store_G=false)
        @test r_base.nStatus == 0
        x_base = copy(r_base.dual_x)
        mask = base_active_mask(theta0, ctx5, obj5)

        @testset "base_active_mask has the right shape and is not degenerate" begin
            @test mask isa MelitzActiveSetSnapshot
            @test size(mask.baseline) == (ctx5.D, ctx5.D, size(obj5.U, 1))
            @test 0 < sum(mask.baseline) < length(mask.baseline)
            @test length(mask.autarky) == size(obj5.U, 1)
            # NOT asserted to be a nontrivial mix (unlike the baseline mask): the focal
            # autarky cutoff can legitimately put ~100% of draws in the active region at
            # this small-W fixture (a low autarky cutoff is a valid economic state, not a
            # bug) -- only check it's a well-formed, non-empty Bool vector.
            @test 0 <= sum(mask.autarky) <= length(mask.autarky)
        end

        rng5 = MersenneTwister(11)
        v = randn(rng5, length(theta0)); v ./= norm(v)

        @testset "Method B at a small bandwidth with zero switches agrees with Method C (exact) to a few %" begin
            h_small = 1e-6
            nsw_p, _, auk_p = count_switches(mask, theta0 .+ h_small .* v, ctx5, obj5)
            nsw_m, _, auk_m = count_switches(mask, theta0 .- h_small .* v, ctx5, obj5)
            @test nsw_p == 0 && nsw_m == 0  # confirms this h genuinely has no switches
            @test auk_p == 0 && auk_m == 0  # ...INCLUDING the separately-tracked autarky decision
            rb = method_b_fixed_dual_secant(theta0, v, h_small, x_base, ctx5, obj5)
            rc = method_c_forwarddiff_envelope(theta0, v, h_small, x_base, ctx5, obj5)
            # rtol=5%, not 1%: this is a finite-h (h=1e-6) SECANT approximation to Method
            # C's exact derivative, at a small-W (2,000) fixture -- a few percent residual
            # discrepancy is expected numerical behavior, not a correctness bug (the
            # governing session's own reduced battery, W=20,000, real KNITRO, found
            # agreement ranging from 0.002% to ~5% across directions at confirmed
            # zero-switch bandwidths, docs/melitz_delta_star.md Section 15.7).
            @test isapprox(rb.deriv, rc.deriv; rtol=5e-2)
        end

        @testset "Method A (reoptimized FD) returns a finite derivative at a moderate bandwidth" begin
            ra = method_a_reoptimized_fd(theta0, v, 1e-4, ctx5, obj5; cold=true)
            @test isfinite(ra.deriv)
        end

        @testset "count_switches is zero against itself (mask compared to its own base point)" begin
            nsw0, _, auk0 = count_switches(mask, theta0, ctx5, obj5)
            @test nsw0 == 0
            @test auk0 == 0
        end

        @testset "2026-07-23 regression: gamma direction at a confirmed zero-switch (incl. autarky) bandwidth agrees B vs C tightly" begin
            # Section 1.1 bug fix regression test: before the fix, the OLD Method C reused
            # the baseline (j,j) mask as a (wrong) proxy for the autarky decision, so a
            # `gamma` perturbation -- which shifts price_power_autarky and hence ONLY the
            # autarky cutoff, leaving every baseline decision unchanged -- could show
            # "zero switches" under the OLD (baseline-only) diagnostic while the autarky
            # decision had, in fact, moved, and B/C would disagree sharply as a result
            # (docs/melitz_delta_star.md Section 15.7's own documented `gamma`
            # counterexample). At a small enough h that BOTH baseline and autarky are
            # confirmed unchanged, the fixed Methods B/C must now agree tightly.
            gamma_idx = 1  # theta_free's own layout: g = log gamma_prime[j] is coordinate 1
            v_gamma = zeros(length(theta0)); v_gamma[gamma_idx] = 1.0
            h_g = 1e-7
            nsw_gp, _, auk_gp = count_switches(mask, theta0 .+ h_g .* v_gamma, ctx5, obj5)
            nsw_gm, _, auk_gm = count_switches(mask, theta0 .- h_g .* v_gamma, ctx5, obj5)
            @test nsw_gp == 0 && nsw_gm == 0 && auk_gp == 0 && auk_gm == 0
            rb_g = method_b_fixed_dual_secant(theta0, v_gamma, h_g, x_base, ctx5, obj5)
            rc_g = method_c_forwarddiff_envelope(theta0, v_gamma, h_g, x_base, ctx5, obj5)
            @test isapprox(rb_g.deriv, rc_g.deriv; rtol=5e-2)
        end

        @testset "Section 1.5: Method D (hand-derived) agrees with Method C at zero switches" begin
            for (label, vv) in (("ordinary_A_v", v), ("gamma", (u = zeros(length(theta0)); u[1] = 1.0; u)))
                h_small = 1e-6
                nsw_p, _, auk_p = count_switches(mask, theta0 .+ h_small .* vv, ctx5, obj5)
                nsw_m, _, auk_m = count_switches(mask, theta0 .- h_small .* vv, ctx5, obj5)
                if nsw_p == 0 && nsw_m == 0 && auk_p == 0 && auk_m == 0
                    rc = method_c_forwarddiff_envelope(theta0, vv, h_small, x_base, ctx5, obj5)
                    rd = method_d_hand_derived(theta0, vv, h_small, x_base, ctx5, obj5)
                    @test isapprox(rc.deriv, rd.deriv; rtol=1e-6, atol=1e-10)
                end
            end
        end

        @testset "Section 1.6: f_high_switch_direction is genuinely distinct from ordinary_f" begin
            nA = ctx5.D^2 - 1
            ordinary_f_idx = 2 + nA
            v_ord = zeros(length(theta0)); v_ord[ordinary_f_idx] = 1.0
            h_probe = 1e-3
            nsw_ord, _, auk_ord = count_switches(mask, theta0 .+ h_probe .* v_ord, ctx5, obj5)

            v_hs, hs_idx, hs_switches_probe = f_high_switch_direction(theta0, ctx5, obj5; test_h=h_probe)
            nsw_hs, _, auk_hs = count_switches(mask, theta0 .+ h_probe .* v_hs, ctx5, obj5)

            @test v_hs != v_ord  # regression: no longer accidentally the same coordinate/direction
            @test (nsw_hs + auk_hs) >= (nsw_ord + auk_ord)  # at least as many switches as ordinary_f
        end
    end

    # ========================================================================
    # Infrastructure-only regression test (formerly "Section 4 nested Delta-star outer
    # solve"): 2026-07-23 governing correction relabels this as a software smoke test,
    # NOT an economic minimum-divergence result -- see outer_solve.jl's file-level note.
    # ========================================================================
    @testset "Infrastructure smoke test: run_minimum_divergence_outer_smoke_test (nested outer solve, cold-verified)" begin
        small_fixture = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=2_000)
        obj4, theta0 = build_melitz_psi_bundle(small_fixture;
            inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"))
        ctx4 = obj4.γ

        res = run_minimum_divergence_outer_smoke_test(theta0, ctx4, obj4; iterations=3, time_limit=40.0)
        @testset "structural: result type, cold-verified incumbent is real" begin
            @test res isa MinimumDivergenceSmokeTestResult
            @test res.cold_verified isa MelitzDeltaEvalResult
            @test res.n_inner_solves > 0
        end
        @testset "smoke test's own Delta(theta_final) <= Delta(theta_init) (regression check only, not an economic finding)" begin
            @test res.cold_verified.Delta <= res.Delta_init + 1e-9
        end
        @testset "cold-verified incumbent passes the Gate A ex-post equilibrium checks" begin
            if res.cold_verified.verified
                @test res.cold_verified.equilibrium_check !== nothing
                c = res.cold_verified.equilibrium_check
                @test maximum(abs.(c.residual_gamma_baseline)) < 1e-3
                @test abs(c.gravity_residual_A) < 1e-8
                @test abs(c.gravity_residual_f) < 1e-8
            end
        end
    end

    # ========================================================================
    # Section 3 (2026-07-23 governing-correction session): the direct finite-delta
    # bound problem via KNITRO with explicit nonlinear constraints. Regression coverage
    # only (a short, loose run) -- the real economic campaigns are Section 4, run
    # separately, not as part of the automated test suite.
    # ========================================================================
    @testset "Section 3: solve_melitz_finite_delta_bound (finite-delta KNITRO outer NLP)" begin
        small_fixture = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=2_000)
        obj3fd, theta0_3fd = build_melitz_psi_bundle(small_fixture;
            inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"))
        ctx3fd = obj3fd.γ
        r0 = evaluate_melitz_delta(theta0_3fd, ctx3fd, obj3fd; cold=true, store_G=false)
        @test r0.nStatus == 0

        # A loose delta (comfortably above the population Delta) and a short run --
        # this is a regression/plumbing check (does the combined callback run without
        # crashing, does it respect the corrected constraint direction, does the
        # verified-success gate work), not a converged economic result.
        delta_loose = max(r0.Delta * 5, 1e-3)
        res3 = solve_melitz_finite_delta_bound(ctx3fd, obj3fd, theta0_3fd; delta=delta_loose,
            direction=:upper, gradient_backend=:B, h=1e-4, theta_box=0.5,
            inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"))

        @testset "structural: result type, ran without a KNITRO callback crash" begin
            @test res3 isa MelitzFiniteDeltaOuterResult
            @test res3.nStatus != -500  # the combined-callback crash this session's fix targets
            @test res3.inner_solve_count > 0
        end
        @testset "terminal point's own cutoff feasibility is internally consistent" begin
            @test res3.terminal_eval.feasible == (res3.terminal_eval.min_slack >= 0)
        end
        @testset "IF a cold-verified incumbent was found, it respects the delta budget" begin
            if res3.cold_verified_incumbent !== nothing
                @test res3.cold_verified_incumbent.Delta <= delta_loose + 1e-6
                @test res3.cold_verified_incumbent.verified
            end
        end
    end
end

# ============================================================================
# 9. Nearby gravity-feasible perturbations (main prompt Section 12)
# ============================================================================
@testset "Nearby gravity-feasible perturbations (pivot-based, always exact)" begin
    p = FIXTURE.primitives
    D, j = p.D, p.target_country
    c_full, A_pivot = build_gravity_pivots(p.tau, j)
    outer_layout = melitz_outer_layout(D, j)
    moment_layout = MelitzMomentLayout(D)
    ctx = (D=D, sigma=p.sigma, theta_star=p.theta_star, target_country=j, tau=p.tau, w=p.w,
           w_prime=FIXTURE.counterfactual.w_prime, L=FIXTURE.L, expenditure=FIXTURE.equilibrium.expenditure,
           cutoff=FIXTURE.equilibrium.cutoff, moment_layout=moment_layout, X_data=FIXTURE.equilibrium.trade_flow,
           c_full=c_full, A_pivot=A_pivot, jj_lin=outer_layout.jj_lin, f_free_lin=outer_layout.f_free_lin)
    theta0 = reduce_to_free_theta(p, ctx)

    rng = MersenneTwister(2024)
    magnitudes = [1e-4, 1e-3, 1e-2, 1e-2, 1e-3]
    n_feasible = 0
    for (i, mag) in enumerate(magnitudes)
        perturbation = mag .* randn(rng, length(theta0))
        i == 1 && (perturbation[1] = mag) # ensure at least one perturbation touches gamma_prime_j directly
        theta_pert = theta0 .+ perturbation
        A_p, f_p, gamma_p, f_jj_p = expand_free_theta(theta_pert, ctx)

        @testset "perturbation $i (magnitude $mag): gravity restrictions remain exact" begin
            p_pert = MelitzPrimitives(D, p.sigma, p.theta_star, j, p.tau, p.w, A_p, f_p, gamma_p)
            rA, rf = gravity_residuals(p_pert)
            @test abs(rA) < 1e-8
            @test abs(rf) < 1e-8
        end

        zhat_pert = [melitz_cutoff(p.w[o], f_p[o, d], p.sigma,
                                    melitz_C(p.w[o], p.tau[o, d], A_p[o, d], p.sigma, ctx.expenditure[d]))
                     for o in 1:D, d in 1:D]
        feasible = all(zhat_pert .>= 1.0) &&
                   all(zhat_pert[o, d] >= zhat_pert[o, o] for o in 1:D, d in 1:D if d != o)
        feasible && (n_feasible += 1)
        # A perturbed point may legitimately be infeasible -- report, don't force feasibility
        # by loosening the cutoff restrictions.
    end
    @testset "at least one small perturbation is cutoff-feasible" begin
        @test n_feasible >= 1
    end
end

# ============================================================================
# 2026-07-22 session Section 1.2/1.3: melitz_outer_state (never reads a benchmark cutoff
# at a displaced point) and the deterministic cutoff-constraint Jacobian (exact via
# ForwardDiff -- legitimate here since the cutoff formula is smooth, unlike Delta(theta)'s
# hard participation gate).
# ============================================================================
@testset "Section 1.2: melitz_outer_state never reuses a stale benchmark cutoff" begin
    p = FIXTURE.primitives
    D, j = p.D, p.target_country
    c_full, A_pivot = build_gravity_pivots(p.tau, j)
    outer_layout = melitz_outer_layout(D, j)
    moment_layout = MelitzMomentLayout(D)
    ctx = (D=D, sigma=p.sigma, theta_star=p.theta_star, target_country=j, tau=p.tau, w=p.w,
           w_prime=FIXTURE.counterfactual.w_prime, L=FIXTURE.L, expenditure=FIXTURE.equilibrium.expenditure,
           benchmark_cutoff=FIXTURE.equilibrium.cutoff, moment_layout=moment_layout,
           X_data=FIXTURE.equilibrium.trade_flow, c_full=c_full, A_pivot=A_pivot,
           jj_lin=outer_layout.jj_lin, f_free_lin=outer_layout.f_free_lin,
           inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"),
           outer_loop_opt=joinpath(dirname(dirname(@__DIR__)), "ek_outer_loop_options.opt"))
    theta0 = reduce_to_free_theta(p, ctx)

    @testset "at the benchmark point, the fresh cutoff matches the fixture's own cutoff" begin
        st = melitz_outer_state(theta0, ctx)
        @test isapprox(st.cutoff, FIXTURE.equilibrium.cutoff; rtol=1e-8, atol=1e-10)
        @test st.feasible
        @test st.min_slack > 0
    end

    @testset "at a displaced point, the fresh cutoff DIFFERS from the stale benchmark cutoff" begin
        rng = MersenneTwister(555)
        theta_pert = theta0 .+ 0.05 .* randn(rng, length(theta0))
        st = melitz_outer_state(theta_pert, ctx)
        # regression test for the bug this fixes: a stale-cutoff implementation would
        # silently return ctx.benchmark_cutoff unchanged here, which is essentially never
        # what a genuinely displaced (A,f) implies.
        @test !isapprox(st.cutoff, ctx.benchmark_cutoff; rtol=1e-6)
        @test isapprox(st.cutoff, melitz_baseline_cutoff(st.A, st.f, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma); rtol=1e-10)
    end

    @testset "melitz_moments_adapter! also never touches ctx.benchmark_cutoff (construction-level check)" begin
        # the adapter builds its own eq internally; confirm it does not error/read a
        # cutoff field that isn't there by constructing a ctx WITHOUT benchmark_cutoff at
        # all and confirming moments! still runs (proves benchmark_cutoff is load-bearing
        # nowhere in the active moments path).
        ctx_no_bench = (D=D, sigma=p.sigma, theta_star=p.theta_star, target_country=j, tau=p.tau,
            w=p.w, w_prime=ctx.w_prime, L=ctx.L, expenditure=ctx.expenditure,
            moment_layout=moment_layout, X_data=ctx.X_data, c_full=c_full, A_pivot=A_pivot,
            jj_lin=outer_layout.jj_lin, f_free_lin=outer_layout.f_free_lin)
        obj_stub = (γ=ctx_no_bench,)
        K = zeros(100); G = zeros(100, moment_layout.num_moments)
        U = FIXTURE.z_draws[1:100, :]
        melitz_moments_adapter!(K, G, theta0, U, obj_stub)
        @test all(isfinite, G)
    end
end

@testset "Section 1.3: deterministic cutoff constraints and their exact Jacobian" begin
    p = FIXTURE.primitives
    D, j = p.D, p.target_country
    c_full, A_pivot = build_gravity_pivots(p.tau, j)
    outer_layout = melitz_outer_layout(D, j)
    moment_layout = MelitzMomentLayout(D)
    ctx = (D=D, sigma=p.sigma, theta_star=p.theta_star, target_country=j, tau=p.tau, w=p.w,
           w_prime=FIXTURE.counterfactual.w_prime, L=FIXTURE.L, expenditure=FIXTURE.equilibrium.expenditure,
           benchmark_cutoff=FIXTURE.equilibrium.cutoff, moment_layout=moment_layout,
           X_data=FIXTURE.equilibrium.trade_flow, c_full=c_full, A_pivot=A_pivot,
           jj_lin=outer_layout.jj_lin, f_free_lin=outer_layout.f_free_lin,
           inner_loop_opt="unused", outer_loop_opt="unused")
    theta0 = reduce_to_free_theta(p, ctx)

    @testset "g_domestic/g_export have the right lengths and are feasible at theta0" begin
        g_d, g_e = melitz_cutoff_constraints_at(theta0, ctx)
        @test length(g_d) == D
        @test length(g_e) == D * (D - 1)
        @test all(>=(0), g_d)
        @test all(>=(0), g_e)
    end

    @testset "Jacobian matches central finite differences at theta0 and at a perturbed point" begin
        rng = MersenneTwister(1)
        for theta_test in (theta0, theta0 .+ 0.02 .* randn(rng, length(theta0)))
            Jd, Je = melitz_cutoff_constraint_jacobian(theta_test, ctx)
            @test size(Jd) == (D, length(theta0))
            @test size(Je) == (D * (D - 1), length(theta0))

            h = 1e-6
            n = length(theta_test)
            Jd_fd = zeros(size(Jd))
            Je_fd = zeros(size(Je))
            for k in 1:n
                tp = copy(theta_test); tp[k] += h
                tm = copy(theta_test); tm[k] -= h
                gdp, gep = melitz_cutoff_constraints_at(tp, ctx)
                gdm, gem = melitz_cutoff_constraints_at(tm, ctx)
                Jd_fd[:, k] = (gdp .- gdm) ./ (2h)
                Je_fd[:, k] = (gep .- gem) ./ (2h)
            end
            @test isapprox(Jd, Jd_fd; rtol=1e-4, atol=1e-6)
            @test isapprox(Je, Je_fd; rtol=1e-4, atol=1e-6)
        end
    end
end

# ============================================================================
# 10. Legacy Pareto-fixed-N closure -- archived diagnostics only (addendum Sec 15)
# ============================================================================
@testset "Legacy Pareto-fixed-N closure (archived, pareto_* diagnostics only)" begin
    p = FIXTURE.primitives
    D = p.D
    L = FIXTURE.L
    f_entry_diag = recover_entry_costs_from_lfd(p, FIXTURE.equilibrium, FIXTURE.z_draws, fill(1.0 / size(FIXTURE.z_draws, 1), size(FIXTURE.z_draws, 1)))

    @testset "pareto_entrant_mass_from_labor / pareto_entry_cost_from_free_entry round-trip" begin
        # closed-form Pareto identity: N_o = (sigma-1)/(sigma*theta_star) * L_o/f_entry_o,
        # cross-checked against the general-F equal-weight recovery above (approximately
        # consistent at the Pareto benchmark since the fixture is Pareto-generated).
        N_pareto = [pareto_entrant_mass_from_labor(L[o], max(f_entry_diag[o], 1e-6), p.sigma, p.theta_star) for o in 1:D]
        @test all(isfinite, N_pareto)
        @test all(>(0), N_pareto)
    end

    @testset "pareto_autarky_fixed_cost_fixed_mass / pareto_target_cutoff_fixed_mass are pure closed forms" begin
        f_jj_legacy = pareto_autarky_fixed_cost_fixed_mass(0.2, p.sigma, p.theta_star)
        @test f_jj_legacy > 0
        zhat_legacy = pareto_target_cutoff_fixed_mass(5.0, 1.0, 1.2, p.theta_star)
        @test zhat_legacy > 0
    end

    @testset "these functions are never called from the active moment/delta_star path" begin
        # documentation-as-test: grep the active files for the demoted names.
        active_files = [joinpath(MELITZ_DIR, f) for f in
            ("moments.jl", "delta_star.jl", "fake_data.jl", "fstar_solver.jl")]
        demoted = ["entrant_mass_from_labor(", "autarky_fixed_cost(",
                   "target_baseline_cutoff_for_autarky(", "solve_autarky_counterfactual("]
        for file in active_files, name in demoted
            content = read(file, String)
            @test !occursin(name, content)
        end
    end
end

println("\n" * "="^70)
println("Melitz Delta-star test suite (active minimal-moment closure) complete.")
println("="^70)
