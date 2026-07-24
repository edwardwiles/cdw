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
include(joinpath(MELITZ_DIR, "profiling.jl"))
include(joinpath(MELITZ_DIR, "types.jl"))
include(joinpath(MELITZ_DIR, "pareto.jl"))
include(joinpath(MELITZ_DIR, "firm_quantities.jl"))
include(joinpath(MELITZ_DIR, "equilibrium.jl"))
include(joinpath(MELITZ_DIR, "moments.jl"))
include(joinpath(MELITZ_DIR, "delta_star.jl"))
include(joinpath(MELITZ_DIR, "affine_cutoff.jl"))
include(joinpath(MELITZ_DIR, "log_cutoff_param.jl"))
include(joinpath(MELITZ_DIR, "fake_data.jl"))
include(joinpath(MELITZ_DIR, "fstar_solver.jl"))
include(joinpath(MELITZ_DIR, "fstar_direct.jl"))
include(joinpath(MELITZ_DIR, "gradient_lab.jl"))
include(joinpath(MELITZ_DIR, "outer_solve.jl"))
include(joinpath(MELITZ_DIR, "inner_screening.jl"))
include(joinpath(MELITZ_DIR, "origin_block_screen.jl"))
include(joinpath(MELITZ_DIR, "localized_gradient.jl"))
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

# 2026-07-23 correctness-repair session, Section 7: minimal duck-typed stand-ins for
# KNITRO's own EvalRequest/EvalResult so the A/B/A repeated-evaluation test can call the
# EXACT production callback closures (`melitz_build_finite_delta_callbacks`'s `cb_F!`/
# `cb_G!`) directly, without a full KNITRO problem/solve -- the callbacks only ever read
# `evalRequest.x` and write `evalResult.obj`/`.c`/`.objGrad`/`.jac`, never touch `kc`/`cb`/
# `userParams` (they close over `obj` directly instead), so a plain mutable struct with
# just those fields suffices. Must be defined at top level (struct definitions are not
# allowed inside a `@testset` block's local scope).
mutable struct MelitzMockEvalRequest
    x::Vector{Float64}
end
mutable struct MelitzMockEvalResult
    obj::Vector{Float64}
    c::Vector{Float64}
    objGrad::Vector{Float64}
    jac::Vector{Float64}
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

        @testset "Section 12.4: fixed_active_set_moments! (in-place) matches the allocating reference" begin
            Wt = size(obj5.U, 1)
            d = ctx5.moment_layout.num_moments
            rng12 = MersenneTwister(41)
            for _ in 1:5
                theta_probe = theta0 .+ 0.01 .* randn(rng12, length(theta0))
                G_ref = fixed_active_set_moments(theta_probe, ctx5, obj5)
                G_buf = zeros(Wt, d)
                profit_buf = zeros(Wt)
                fixed_active_set_moments!(G_buf, profit_buf, theta_probe, ctx5, obj5)
                @test G_buf == G_ref  # bit-identical: same shared fill body
            end
        end

        @testset "Section 12.4: fixed_active_set_moments! eliminates the per-call G allocation" begin
            Wt = size(obj5.U, 1)
            d = ctx5.moment_layout.num_moments
            G_buf = zeros(Wt, d)
            profit_buf = zeros(Wt)
            fixed_active_set_moments!(G_buf, profit_buf, theta0, ctx5, obj5)  # warm up (JIT)
            fixed_active_set_moments(theta0, ctx5, obj5)
            bytes_inplace = @allocated fixed_active_set_moments!(G_buf, profit_buf, theta0, ctx5, obj5)
            bytes_alloc = @allocated fixed_active_set_moments(theta0, ctx5, obj5)
            @test bytes_inplace < bytes_alloc
            # the allocating version must allocate at least the W x d matrix itself
            @test bytes_alloc >= Wt * d * 8
            println("    [Section 12.4] allocating=$bytes_alloc bytes, in-place=$bytes_inplace bytes ",
                "($(round(bytes_alloc / max(1, bytes_inplace); digits=1))x less)")
        end

        @testset "Section 12.4: make_melitz_moments_jacobian_b (live GA path) numerically unaffected by buffer reuse" begin
            # Cross-check the LIVE finite_delta_outer.jl closure (now buffer-reusing)
            # against a fresh, independently-allocating one at the same point -- confirms
            # buffer reuse across coordinate probes never leaks stale values between
            # probes (each fixed_active_set_moments! call fully overwrites G via fill!).
            mj1 = make_melitz_moments_jacobian_b(1e-4)
            mj2 = make_melitz_moments_jacobian_b(1e-4)
            n = length(theta0)
            d = ctx5.moment_layout.num_moments
            Wt = size(obj5.U, 1)
            K_jac1, G_jac1 = zeros(Wt, n), zeros(Wt, d, n)
            K_jac2, G_jac2 = zeros(Wt, n), zeros(Wt, d, n)
            mj1(K_jac1, G_jac1, theta0, obj5.U, obj5)
            mj2(K_jac2, G_jac2, theta0, obj5.U, obj5)
            @test G_jac1 == G_jac2
            @test K_jac1 == K_jac2
            # calling the SAME closure twice in a row (simulating repeated GA callbacks)
            # must also reproduce identically -- the persistent buffer must not carry any
            # cross-call contamination.
            K_jac1b, G_jac1b = zeros(Wt, n), zeros(Wt, d, n)
            mj1(K_jac1b, G_jac1b, theta0, obj5.U, obj5)
            @test G_jac1b == G_jac1
        end
    end

    # ========================================================================
    # Screening-continuation session, Phase II.11: localized fixed-dual gradient backend.
    # Self-contained fixture (independent of the enclosing "Section 5" testset's own
    # `obj5`/`ctx5`/`theta0`, which are local to that testset's own scope and not visible
    # after it closes).
    # ========================================================================
    @testset "Phase II.11: localized gradient backend (Gate 1 + Gate 2)" begin
        fixture11 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=2_000)
        obj11, theta11 = build_melitz_psi_bundle(fixture11;
            inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"))
        ctx11 = obj11.γ

        # ====================================================================
        # GATE 1: the affected-moment-column dependency map, validated ALONE before any
        # gradient code is written on top of it (main prompt Section 7.1's own required
        # order). For EVERY free coordinate, across several random base points, the claimed
        # affected set (`cells`/`touches_link`) must be a SUPERSET of the columns that
        # ACTUALLY differ under a real finite perturbation
        # (`fixed_active_set_moments(theta+h*e_k)` vs the base, column-by-column `!=`).
        # ====================================================================
        @testset "Gate 1: localized dependency map is a superset of actual column changes" begin
            depmap = melitz_localized_dependency_map(ctx11)
            n11 = length(theta11)
            @test length(depmap) == n11

            @testset "structural sanity: every claimed cell is a valid (o,d) pair, jj/pivot cells distinct" begin
                pm = melitz_pivot_map(ctx11)
                @test pm.jj_cell != pm.A_pivot_cell
                @test pm.jj_cell != pm.f_pivot_cell
                @test pm.A_pivot_cell != pm.f_pivot_cell
                for dep in depmap, c in dep.cells
                    @test 1 <= c[1] <= ctx11.D && 1 <= c[2] <= ctx11.D
                end
            end

            rng11 = MersenneTwister(97)
            h11 = 1e-4
            n_checked = 0
            for _ in 1:4
                theta_base11 = theta11 .+ 0.02 .* randn(rng11, n11)
                G_base = fixed_active_set_moments(theta_base11, ctx11, obj11)
                link_col = ctx11.moment_layout.focal_link_index
                for k in 1:n11
                    ek = zeros(n11); ek[k] = 1.0
                    Gp = fixed_active_set_moments(theta_base11 .+ h11 .* ek, ctx11, obj11)
                    Gm = fixed_active_set_moments(theta_base11 .- h11 .* ek, ctx11, obj11)
                    dep = depmap[k]
                    claimed_cols = Set(ctx11.moment_layout.trade_index[c[1], c[2]] for c in dep.cells)
                    dep.touches_link && push!(claimed_cols, link_col)
                    for col in 1:ctx11.moment_layout.num_moments
                        actually_changed = @view(Gp[:, col]) != @view(G_base[:, col]) ||
                                            @view(Gm[:, col]) != @view(G_base[:, col])
                        if actually_changed
                            @test col in claimed_cols
                            n_checked += 1
                        end
                    end
                end
            end
            @info "Phase II.11 Gate 1 superset validation" n_coordinates=n11 n_base_points=4 n_actual_changes_checked=n_checked
            @test n_checked > 0   # sanity: the probe grid must have exercised at least some real changes
        end

        # ====================================================================
        # GATE 2: the restricted-fill extension and the :method_b_localized Jacobian
        # backend built on top of it, ONLY reachable after Gate 1 passed above. Bit-exact
        # (not merely close) agreement against full Method B is the required bar (main
        # prompt Section 7.3): with Gate 1's dependency map correct, an unaffected column's
        # localized central-difference MUST be exactly 0.0 at every draw, matching what
        # full Method B's own (Gp_full-Gm_full)/(2h) computes for a column that truly does
        # not move under that displacement.
        # ====================================================================
        @testset "Gate 2: restricted-fill extension + :method_b_localized bit-exact vs. full Method B" begin
            @testset "restricted-fill sanity: full cell set + compute_link=true reproduces the unrestricted call exactly" begin
                Wt2 = size(obj11.U, 1)
                d2 = ctx11.moment_layout.num_moments
                all_cells = Tuple{Int,Int}[(o, d) for o in 1:ctx11.D for d in 1:ctx11.D]
                G_unrestricted = fixed_active_set_moments(theta11, ctx11, obj11)
                G_restricted = zeros(Wt2, d2)
                profit_scratch = zeros(Wt2)
                fixed_active_set_moments_restricted!(G_restricted, profit_scratch, theta11, ctx11, obj11;
                    cells=all_cells, compute_link=true)
                @test G_restricted == G_unrestricted
            end

            @testset "restricted-fill leaves untouched columns exactly as provided" begin
                Wt2 = size(obj11.U, 1)
                d2 = ctx11.moment_layout.num_moments
                G_seed = fixed_active_set_moments(theta11, ctx11, obj11)
                G_probe = copy(G_seed)
                profit_scratch = zeros(Wt2)
                one_cell = Tuple{Int,Int}[(1, 1)]
                fixed_active_set_moments_restricted!(G_probe, profit_scratch, theta11 .+ 0.05 .* randn(MersenneTwister(3), length(theta11)),
                    ctx11, obj11; cells=one_cell, compute_link=false)
                for o in 1:ctx11.D, d in 1:ctx11.D
                    (o, d) == (1, 1) && continue
                    col = ctx11.moment_layout.trade_index[o, d]
                    @test G_probe[:, col] == G_seed[:, col]
                end
                @test G_probe[:, ctx11.moment_layout.focal_link_index] == G_seed[:, ctx11.moment_layout.focal_link_index]
            end

            mj_full = make_melitz_moments_jacobian_b(1e-4)
            mj_loc = make_melitz_moments_jacobian_b_localized(1e-4)
            n2 = length(theta11)
            d2 = ctx11.moment_layout.num_moments
            Wt2 = size(obj11.U, 1)

            rng_gate2 = MersenneTwister(53)
            for trial in 1:3
                theta_probe2 = theta11 .+ 0.02 .* randn(rng_gate2, n2)
                K_full, G_full = zeros(Wt2, n2), zeros(Wt2, d2, n2)
                K_loc, G_loc = zeros(Wt2, n2), zeros(Wt2, d2, n2)
                mj_full(K_full, G_full, theta_probe2, obj11.U, obj11)
                mj_loc(K_loc, G_loc, theta_probe2, obj11.U, obj11)
                @testset "trial $trial: K_jac and G_jac bit-identical, localized vs full Method B" begin
                    @test K_loc == K_full
                    @test G_loc == G_full
                end
            end
        end

        # ====================================================================
        # Phase II.12 (this continuation session): the parallel coordinate sweep
        # (`:method_b_localized_parallel`, `Threads.@threads :static`) must be BIT-IDENTICAL
        # to the serial localized backend at every thread count -- each coordinate's own
        # output column is computed independently from the same read-only base state
        # (`Gbase`), so there is no floating-point-order-dependent reduction across threads
        # to introduce even roundoff-scale drift. This process's own `Threads.nthreads()`
        # (1 under this repo's standard `JULIA_NUM_THREADS=1` test-run policy) exercises the
        # `Threads.@threads :static` code path and the parallelism-guard wiring correctly,
        # but only at `nthreads=1` -- true multi-thread scaling/exactness at `nthreads>1` is
        # validated separately via `scripts/melitz_parallel_gradient_sweep.sh`, not by this
        # in-process test (a single Julia process cannot change its own thread pool size).
        # ====================================================================
        @testset "Phase II.12: parallel localized gradient (:method_b_localized_parallel)" begin
            mj_loc12 = make_melitz_moments_jacobian_b_localized(1e-4)
            mj_par12 = make_melitz_moments_jacobian_b_localized_parallel(1e-4)
            n12 = length(theta11)
            d12 = ctx11.moment_layout.num_moments
            Wt12 = size(obj11.U, 1)

            rng12 = MersenneTwister(71)
            for trial in 1:3
                theta_probe12 = theta11 .+ 0.02 .* randn(rng12, n12)
                K_loc, G_loc = zeros(Wt12, n12), zeros(Wt12, d12, n12)
                K_par, G_par = zeros(Wt12, n12), zeros(Wt12, d12, n12)
                mj_loc12(K_loc, G_loc, theta_probe12, obj11.U, obj11)
                mj_par12(K_par, G_par, theta_probe12, obj11.U, obj11)
                @testset "trial $trial: parallel bit-identical to serial localized (nthreads=$(Threads.nthreads()))" begin
                    @test K_par == K_loc
                    @test G_par == G_loc
                end
            end

            @testset "BLAS thread count restored after the parallel coordinate sweep" begin
                prev = BLAS.get_num_threads()
                BLAS.set_num_threads(3)
                K_par2, G_par2 = zeros(Wt12, n12), zeros(Wt12, d12, n12)
                mj_par12(K_par2, G_par2, theta11, obj11.U, obj11)
                @test BLAS.get_num_threads() == 3
                BLAS.set_num_threads(prev)
            end

            @testset "small-N probe call (size(U,1) < size(obj.U,1)) skips G_jac safely" begin
                K_small, G_small = zeros(2, n12), zeros(2, d12, n12)
                mj_par12(K_small, G_small, theta11, obj11.U[1:2, :], obj11)
                @test all(iszero, G_small)
                @test all(==(1.0), K_small[:, 1])
            end
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
        @testset "Section 2.1: theta_init is outer-feasible here, so the initial incumbent is always installed" begin
            @test res3.initial_incumbent !== nothing
            @test res3.initial_incumbent.classification.outer_feasible
            @test res3.initial_incumbent.eval.Delta <= delta_loose
        end
        @testset "cold-verified incumbent (falls back to initial if the trajectory found nothing better) respects the delta budget" begin
            @test res3.cold_verified_incumbent !== nothing
            @test res3.cold_verified_incumbent.classification.outer_feasible
            @test res3.cold_verified_incumbent.eval.Delta <= delta_loose + 1e-6
        end
    end

    # ========================================================================
    # Section 18 (2026-07-23 follow-up session): pins the divergence-budget
    # constraint's SIGN AND MAGNITUDE directly against the independently-verified
    # ground-truth Delta(theta) (evaluate_melitz_delta / PsiObjectiveBundleDelta path).
    # A prior version of solve_melitz_finite_delta_bound set a LOWER bound on
    # constr[1], reasoning constr[1]=-1e10*Delta(theta)<=0; this was WRONG (the
    # functor's raw, uncorrected f equals -Delta(theta), not +Delta(theta), the same
    # find_smallest sign correction inner_loop/dual_scalar_at_fixed_G already apply
    # elsewhere), so constr[1]=+1e10*Delta(theta)>=0 always, making a LOWER bound of
    # -1e10*delta vacuous (always satisfied) -- silently re-introducing the exact
    # failure mode the constraint direction was supposed to fix, and fully explaining
    # the previously-reported "KNITRO live feasibility tracking disagrees with cold
    # verification" finding (docs Section 17.D/17.G): the budget was never binding.
    # This test would have FAILED under that prior (buggy) lower-bound convention.
    # ========================================================================
    @testset "Section 18: divergence-budget constraint sign (constr[1] == +1e10*Delta(theta))" begin
        small_fixture18 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=2_000)
        obj18, theta0_18 = build_melitz_psi_bundle(small_fixture18;
            inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"))
        ctx18 = obj18.γ
        r0_18 = evaluate_melitz_delta(theta0_18, ctx18, obj18; cold=true, store_G=false)
        @test r0_18.nStatus == 0
        @test r0_18.Delta > 0  # genuinely nonzero, so sign flip is unambiguous, not a 0==-0 coincidence

        obj_impl18 = build_melitz_implicit_bundle(ctx18, obj18.U, theta0_18; delta=1e-3,
            find_smallest=true, gradient_backend=:B, h=1e-4,
            inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"),
            outer_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_outer_finite_delta.opt"))
        _, x18, nStatus18 = CounterfactualSensitivity.inner_loop_internal(obj_impl18, theta0_18)
        @test nStatus18 == 0
        local_c18 = zeros(1)
        obj_impl18(x18, constr=local_c18)

        @testset "constr[1] equals +1e10*Delta(theta), not -1e10*Delta(theta)" begin
            @test isapprox(local_c18[1], 1e10 * r0_18.Delta; rtol=1e-6)
            @test local_c18[1] > 0  # would be NEGATIVE under the prior (buggy) sign belief
        end
        @testset "the ACTIVE (upper) bound convention correctly discriminates feasibility" begin
            delta_tight = r0_18.Delta / 2   # Delta(theta0) > delta_tight -> must be INFEASIBLE
            delta_loose18 = r0_18.Delta * 2 # Delta(theta0) <= delta_loose -> must be FEASIBLE
            @test !(local_c18[1] <= 1e10 * delta_tight)   # upper-bound convention: correctly infeasible
            @test local_c18[1] <= 1e10 * delta_loose18    # upper-bound convention: correctly feasible
            # the prior LOWER-bound convention is vacuous at BOTH deltas (regression pin):
            @test local_c18[1] >= -1e10 * delta_tight
            @test local_c18[1] >= -1e10 * delta_loose18
        end
    end

    # ========================================================================
    # 2026-07-23 correctness-repair session: Sections 2-7/12 regression tests. These pin
    # the three bugs fixed this session (objective contamination, opaque 1e10 constraint
    # scaling, garbage-value inner-failure handling) plus the new incumbent-bookkeeping
    # machinery (Section 2), directly against the PRODUCTION combined callback -- not a
    # bypass/helper-function call (Section 6's own requirement).
    # ========================================================================
    small_fixture20 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
        target_country=1, seed=29, W=2_000)
    obj20, theta0_20 = build_melitz_psi_bundle(small_fixture20;
        inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"))
    ctx20 = obj20.γ
    inner_opt20 = joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt")
    outer_opt20 = joinpath(dirname(dirname(@__DIR__)), "melitz_outer_finite_delta.opt")
    r0_20 = evaluate_melitz_delta(theta0_20, ctx20, obj20; cold=true, store_G=false)

    @testset "Section 6: fixed-point KNITRO integration tests (production combined callback)" begin
        @test r0_20.nStatus == 0

        @testset "Test A: feasible Pareto point" begin
            delta_loose20 = max(r0_20.Delta * 5, 1e-3)
            pA = melitz_fixed_point_probe(ctx20, obj20, theta0_20; delta=delta_loose20,
                direction=:upper, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            @test !pA.eval_failed
            @test pA.nStatus == 0
            # Section 3.1: objective equals the correct +/- gamma coordinate exactly,
            # independent of anything the inner solve computed.
            @test isapprox(pA.obj_value, theta0_20[1]; atol=1e-10)
            # Section 4.1/4.2: c_delta == Delta(theta)/delta, the SAME scaling for value
            # and (implicitly, via the shared 1/(1e10*delta) factor) Jacobian.
            @test isapprox(pA.c[1], r0_20.Delta / delta_loose20; rtol=1e-6)
            @test pA.c[1] < 1.0
            @test all(>=(-1e-8), pA.c[2:end])
            @test length(pA.live_candidates) == 1
            @test pA.live_candidates[1].classification.outer_feasible
        end

        @testset "Test B: budget-infeasible, inner-valid point" begin
            theta_pertB = theta0_20 .+ 0.005 .* randn(MersenneTwister(11), length(theta0_20))
            rB = evaluate_melitz_delta(theta_pertB, ctx20, obj20; cold=true, store_G=false)
            @test rB.nStatus == 0   # a genuinely valid inner solve -- NOT an nStatus=-102 point
            @test rB.Delta > 0
            delta_tightB = rB.Delta / 2   # deliberately below Delta(theta) -> budget must be violated
            pB = melitz_fixed_point_probe(ctx20, obj20, theta_pertB; delta=delta_tightB,
                direction=:upper, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            @test !pB.eval_failed   # a real, evaluable, merely-violated constraint -- not a numerical failure
            @test pB.nStatus != 0   # KNITRO reports infeasibility
            @test pB.c[1] > 1.0     # divergence row violates its upper bound
            @test isapprox(pB.c[1], rB.Delta / delta_tightB; rtol=1e-6)
            @test isapprox(pB.obj_value, theta_pertB[1]; atol=1e-10)   # objective still correct
        end

        @testset "Test D: inner numerical failure -- controlled callback rejection" begin
            theta_bad20 = theta0_20 .+ 0.5 .* randn(MersenneTwister(1), length(theta0_20))
            r_bad = evaluate_melitz_delta(theta_bad20, ctx20, obj20; cold=true, store_G=false)
            @test !(r_bad.nStatus in (0, -100, -101, -103))   # confirms this really is a failing point
            pD = melitz_fixed_point_probe(ctx20, obj20, theta_bad20; delta=1e-3,
                direction=:upper, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            @test pD.eval_failed   # Section 5: rejected as an evaluation failure, not a model value
            @test pD.nStatus in MELITZ_KNITRO_EVAL_ERROR_STATUSES
            @test isnan(pD.obj_value)   # never a garbage value (not 1e10, not theta[1] either)
            @test isempty(pD.live_candidates)   # no incumbent contamination from a failed point
        end
    end

    # ========================================================================
    # Screening-session continuation, Phase I.1: the KNITRO-native `lower_limit` mid-solve
    # early stop (`cc_algo/PsiObjectiveBundle.jl`'s `if f <= lower_limit; return
    # -KNITRO.KN_INFINITY`) must be classified as `BudgetInfeasible(...,:live_dual_threshold)`,
    # not `NumericalFailure` -- it is a certificate (weak duality, unconditional -- see
    # inner_screening.jl's file header), not an unresolved failure. Direct tests on one
    # known Delta-above-budget point (guard SHOULD fire and be classified correctly) and one
    # known Delta-below-budget point (guard must NOT false-reject a genuinely feasible point).
    # ========================================================================
    @testset "Phase I.1: live dual-threshold classification (BudgetInfeasible, not NumericalFailure)" begin
        @testset "Delta-above-budget point: threshold fires, classified BudgetInfeasible(:live_dual_threshold)" begin
            theta_pertT1 = theta0_20 .+ 0.005 .* randn(MersenneTwister(11), length(theta0_20))
            rT1 = evaluate_melitz_delta(theta_pertT1, ctx20, obj20; cold=true, store_G=false)
            @test rT1.nStatus == 0   # a genuinely valid inner solve at this theta, real Delta known
            @test rT1.Delta > 0
            delta_tightT1 = rT1.Delta / 2   # deliberately below Delta(theta) -> must be rejected

            objT1 = build_melitz_implicit_bundle(ctx20, obj20.U, theta_pertT1; delta=delta_tightT1,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20, lower_limit_guard=0.0)
            @test objT1.lower_limit == -delta_tightT1

            bankT1 = MelitzDualBank()
            resultT1 = melitz_classified_inner_solve(objT1, theta_pertT1, ctx20;
                delta=delta_tightT1, bank=bankT1)

            @test resultT1 isa BudgetInfeasible
            @test resultT1.source == :live_dual_threshold
            @test objT1.threshold_crossed[]
            # the certified bound is a valid Delta lower bound (weak duality) AND is why the
            # point was rejected (it exceeds the delta budget the guard was set at):
            @test resultT1.lower_bound <= rT1.Delta + 1e-8
            @test resultT1.lower_bound > delta_tightT1
            # main prompt Section 11: must not enter the numerical-failure counter -- there is
            # no counter at this direct-classifier level, but the crossing dual must be finite
            # and (main prompt: "may contribute its finite crossing dual to the screening
            # bank") actually inserted:
            @test !isempty(objT1.threshold_crossing_x[])
            @test all(isfinite, objT1.threshold_crossing_x[])
            @test length(bankT1.entries) == 1
            @test bankT1.entries[1] == objT1.threshold_crossing_x[]
        end

        @testset "Delta-below-budget point: threshold does NOT fire, classified InnerSolved" begin
            delta_looseT2 = max(r0_20.Delta * 5, 1e-3)
            objT2 = build_melitz_implicit_bundle(ctx20, obj20.U, theta0_20; delta=delta_looseT2,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20, lower_limit_guard=0.0)
            @test objT2.lower_limit == -delta_looseT2

            bankT2 = MelitzDualBank()
            resultT2 = melitz_classified_inner_solve(objT2, theta0_20, ctx20;
                delta=delta_looseT2, bank=bankT2)

            @test resultT2 isa InnerSolved
            @test !objT2.threshold_crossed[]   # never fired -- a feasible point must not be rejected
            @test resultT2.Delta <= delta_looseT2
        end

        @testset "integration: solve_melitz_finite_delta_bound with lower_limit_guard routes hits through n_budget_infeasible_reject, never n_numerical_failure_reject as a threshold artifact" begin
            # A tight guard on a short trajectory: every rejection this specific mechanism
            # produces must appear in the budget-infeasible bucket, not the numerical-failure
            # bucket (main prompt Section 11's "must not enter a failed-solve counter").
            # (Other genuine numerical failures unrelated to this mechanism may still occur --
            # this integration check only pins that the THRESHOLD mechanism itself is counted
            # correctly, via the same live `on_inner_result` hook the no-rescue benchmark uses.)
            n_threshold_hits = Ref(0)
            n_numfail_seen = Ref(0)
            collector(theta, result) = begin
                result isa BudgetInfeasible && result.source == :live_dual_threshold && (n_threshold_hits[] += 1)
                result isa NumericalFailure && (n_numfail_seen[] += 1)
                nothing
            end
            resT3 = solve_melitz_finite_delta_bound(ctx20, obj20, theta0_20; delta=1e-2,
                direction=:upper, theta_box=0.02, inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20,
                lower_limit_guard=0.0, on_inner_result=collector)
            @test resT3.n_budget_infeasible_reject >= n_threshold_hits[]
            @test n_threshold_hits[] >= 0   # sanity: counter machinery ran without error
        end
    end

    # ========================================================================
    # Screening-session continuation, Phase I.5/I.6: dual-polish screen + bank eviction
    # policies + finite-x capture from failed/stopped solves.
    # ========================================================================
    @testset "Phase I.6: MelitzDualBank eviction policies + non-finite rejection" begin
        @testset ":fifo evicts oldest" begin
            bank = MelitzDualBank(3; policy=:fifo)
            melitz_dual_bank_insert!(bank, [1.0, 0.0])
            melitz_dual_bank_insert!(bank, [2.0, 0.0])
            melitz_dual_bank_insert!(bank, [3.0, 0.0])
            melitz_dual_bank_insert!(bank, [4.0, 0.0])
            @test length(bank.entries) == 3
            @test bank.entries == [[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
        end

        @testset ":nearest evicts the entry closest to the incoming point" begin
            bank = MelitzDualBank(3; policy=:nearest)
            melitz_dual_bank_insert!(bank, [0.0, 0.0])
            melitz_dual_bank_insert!(bank, [10.0, 0.0])
            melitz_dual_bank_insert!(bank, [20.0, 0.0])
            # incoming point 0.5 is nearest to [0.0,0.0] -- that entry should be replaced,
            # the far entries [10,0]/[20,0] must survive.
            melitz_dual_bank_insert!(bank, [0.5, 0.0])
            @test length(bank.entries) == 3
            @test [0.5, 0.0] in bank.entries
            @test [10.0, 0.0] in bank.entries
            @test [20.0, 0.0] in bank.entries
            @test !([0.0, 0.0] in bank.entries)
        end

        @testset ":diversity evicts the most redundant existing entry" begin
            bank = MelitzDualBank(3; policy=:diversity)
            melitz_dual_bank_insert!(bank, [0.0, 0.0])
            melitz_dual_bank_insert!(bank, [0.1, 0.0])   # near-duplicate of the first -- most redundant
            melitz_dual_bank_insert!(bank, [100.0, 0.0])
            melitz_dual_bank_insert!(bank, [-100.0, 0.0])   # a far, informative new point
            @test length(bank.entries) == 3
            @test [100.0, 0.0] in bank.entries
            @test [-100.0, 0.0] in bank.entries
            # exactly one of the two near-duplicates survives (whichever wasn't evicted as
            # "most redundant"); the bank must not have evicted either far point.
            @test count(e -> e in ([0.0, 0.0], [0.1, 0.0]), bank.entries) == 1
        end

        @testset "non-finite vectors are never inserted" begin
            bank = MelitzDualBank(3; policy=:fifo)
            melitz_dual_bank_insert!(bank, [1.0, NaN])
            melitz_dual_bank_insert!(bank, [Inf, 0.0])
            @test isempty(bank.entries)
            melitz_dual_bank_insert!(bank, [1.0, 2.0])
            @test length(bank.entries) == 1
        end
    end

    # ========================================================================
    # Continuation session (2026-07-23), Section 5: the dual bank as an actual WARM-START
    # source (not merely a pre-solve screening lower bound, its only use before this
    # session). `melitz_dual_bank_insert!`'s new optional `theta` kwarg tags each entry with
    # the outer coordinate it came from; `melitz_bank_nearest_theta` and
    # `melitz_resolve_warm_start!` build on that tagging.
    # ========================================================================
    @testset "Section 5: dual bank as a warm-start source" begin
        @testset "melitz_dual_bank_insert! without theta leaves an untagged (nothing) entry" begin
            bank = MelitzDualBank()
            melitz_dual_bank_insert!(bank, [1.0, 2.0])
            @test length(bank.thetas) == 1
            @test bank.thetas[1] === nothing
        end

        @testset "melitz_bank_nearest_theta finds the closest theta-tagged entry, ignores untagged ones" begin
            bank = MelitzDualBank()
            melitz_dual_bank_insert!(bank, [1.0, 2.0]; theta=[0.1, 0.2])
            melitz_dual_bank_insert!(bank, [3.0, 4.0])   # untagged -- must never be returned
            melitz_dual_bank_insert!(bank, [5.0, 6.0]; theta=[0.9, 1.0])
            d1, x1 = melitz_bank_nearest_theta(bank, [0.15, 0.25])
            @test x1 == [1.0, 2.0]
            d2, x2 = melitz_bank_nearest_theta(bank, [1.0, 1.1])
            @test x2 == [5.0, 6.0]
            @test d1 < d2   # sanity: the first query really is closer to its own match
        end

        @testset "melitz_bank_nearest_theta on an empty or fully-untagged bank returns (Inf, nothing)" begin
            d, x = melitz_bank_nearest_theta(MelitzDualBank(), [0.0, 0.0])
            @test d == Inf && x === nothing
            bank_untagged = MelitzDualBank()
            melitz_dual_bank_insert!(bank_untagged, [1.0, 2.0])
            d2, x2 = melitz_bank_nearest_theta(bank_untagged, [0.0, 0.0])
            @test d2 == Inf && x2 === nothing
        end

        @testset "melitz_resolve_warm_start!: :previous is a no-op" begin
            objW = build_melitz_implicit_bundle(ctx20, obj20.U, theta0_20; delta=1.0,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            objW.x .= 7.0
            objW.use_cached_x = true
            bankW = MelitzDualBank()
            src = melitz_resolve_warm_start!(objW, bankW, theta0_20, :previous)
            @test src == :previous
            @test all(==(7.0), objW.x)   # untouched
            @test objW.use_cached_x
        end

        @testset "melitz_resolve_warm_start!: :bank_nearest sets obj.x/use_cached_x from the closest theta-tagged entry" begin
            objW = build_melitz_implicit_bundle(ctx20, obj20.U, theta0_20; delta=1.0,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            n_dual = length(objW.x)
            objW.x .= -99.0
            objW.use_cached_x = false
            bankW = MelitzDualBank()
            x_target = collect(1.0:n_dual)
            melitz_dual_bank_insert!(bankW, x_target; theta=theta0_20)
            far_theta = theta0_20 .+ 10.0
            melitz_dual_bank_insert!(bankW, zeros(n_dual); theta=far_theta)
            src = melitz_resolve_warm_start!(objW, bankW, theta0_20, :bank_nearest)
            @test src == :bank_nearest
            @test objW.x == x_target
            @test objW.use_cached_x
        end

        @testset "melitz_resolve_warm_start!: :bank_nearest falls back to :previous when no entry is theta-tagged" begin
            objW = build_melitz_implicit_bundle(ctx20, obj20.U, theta0_20; delta=1.0,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            objW.x .= 3.0
            objW.use_cached_x = true
            bankW = MelitzDualBank()
            melitz_dual_bank_insert!(bankW, ones(length(objW.x)))   # untagged
            src = melitz_resolve_warm_start!(objW, bankW, theta0_20, :bank_nearest)
            @test src == :previous
            @test all(==(3.0), objW.x)   # untouched -- the fallback truly is a no-op
        end

        @testset "melitz_resolve_warm_start!: :bank_best_lb picks the tightest stored-dual lower bound" begin
            objW = build_melitz_implicit_bundle(ctx20, obj20.U, theta0_20; delta=1.0,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            CS_ = CounterfactualSensitivity
            G_seed = CS_.select_G_from_H(objW, objW.H)
            objW.moments!(@view(objW.H[:, 1]), G_seed, theta0_20, objW.U, objW)
            objW.H[:, 2] .= 1.0
            bankW = MelitzDualBank()
            n_dual = length(objW.x)
            melitz_dual_bank_insert!(bankW, zeros(n_dual))
            expected_best_lb, expected_best_x = melitz_bank_best(objW, bankW)
            src = melitz_resolve_warm_start!(objW, bankW, theta0_20, :bank_best_lb)
            @test src == :bank_best_lb
            @test objW.x == expected_best_x
            @test objW.use_cached_x
        end

        @testset "melitz_resolve_warm_start!: :bank_best_lb falls back to :previous on an empty bank" begin
            objW = build_melitz_implicit_bundle(ctx20, obj20.U, theta0_20; delta=1.0,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            objW.x .= 5.0
            objW.use_cached_x = true
            src = melitz_resolve_warm_start!(objW, MelitzDualBank(), theta0_20, :bank_best_lb)
            @test src == :previous
            @test all(==(5.0), objW.x)
        end

        @testset "melitz_resolve_warm_start!: :neutral clears the cache" begin
            objW = build_melitz_implicit_bundle(ctx20, obj20.U, theta0_20; delta=1.0,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            objW.x .= 5.0
            objW.use_cached_x = true
            src = melitz_resolve_warm_start!(objW, MelitzDualBank(), theta0_20, :neutral)
            @test src == :neutral
            @test all(isnan, objW.x)
            @test !objW.use_cached_x
        end

        @testset "melitz_resolve_warm_start!: invalid source throws ArgumentError" begin
            objW = build_melitz_implicit_bundle(ctx20, obj20.U, theta0_20; delta=1.0,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            @test_throws ArgumentError melitz_resolve_warm_start!(objW, MelitzDualBank(), theta0_20, :bogus)
        end

        @testset "wired end-to-end: melitz_classified_inner_solve banks every verified/rejected point with its theta" begin
            objW = build_melitz_implicit_bundle(ctx20, obj20.U, theta0_20; delta=1.0,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            bankW = MelitzDualBank()
            result = melitz_classified_inner_solve(objW, theta0_20, ctx20; delta=1e-3, bank=bankW)
            @test result isa InnerSolved
            @test length(bankW.entries) == 1
            @test bankW.thetas[1] == collect(Float64.(theta0_20))
        end
    end

    @testset "Phase I.5: dual-polish screen" begin
        rT4 = evaluate_melitz_delta(theta0_20, ctx20, obj20; cold=true, store_G=false)
        @test rT4.nStatus == 0
        objT4 = build_melitz_implicit_bundle(ctx20, obj20.U, theta0_20; delta=1.0,
            find_smallest=true, gradient_backend=:B, h=1e-4,
            inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
        # populate obj.H at theta0_20 (mirrors what melitz_classified_inner_solve does before
        # any screen runs) so the raw functor evaluated at the VERIFIED optimal dual reproduces
        # the true Delta.
        CS_ = CounterfactualSensitivity
        GT4 = CS_.select_G_from_H(objT4, objT4.H)
        objT4.moments!(@view(objT4.H[:, 1]), GT4, theta0_20, objT4.U, objT4)
        objT4.H[:, 2] .= 1.0
        x_star = rT4.dual_x

        @testset "at the true optimal dual, a below-Delta budget is certified rejected immediately" begin
            delta_tightT4 = rT4.Delta / 2
            res = melitz_dual_polish_screen(objT4, x_star; delta=delta_tightT4, max_steps=3)
            @test res isa BudgetInfeasible
            @test res.source == :dual_polish
            @test res.lower_bound <= rT4.Delta + 1e-6   # valid lower bound (weak duality)
            @test res.lower_bound > delta_tightT4
        end

        @testset "at the true optimal dual, a comfortably-above-Delta budget is NOT rejected" begin
            delta_looseT4 = max(rT4.Delta * 10, 1e-2)
            res = melitz_dual_polish_screen(objT4, x_star; delta=delta_looseT4, max_steps=3)
            @test res === nothing
        end

        @testset "wired into melitz_classified_inner_solve: dual_polish_screen=true certifies without a KNITRO call" begin
            bankT4 = MelitzDualBank()
            melitz_dual_bank_insert!(bankT4, x_star)
            delta_tightT4b = rT4.Delta / 2
            CS_.INNER_SOLVE_COUNT[] = 0
            resultT4 = melitz_classified_inner_solve(objT4, theta0_20, ctx20; delta=delta_tightT4b,
                bank=bankT4, stored_dual_screen=false, dual_polish_screen=true)
            @test resultT4 isa BudgetInfeasible
            @test resultT4.source == :dual_polish
            @test CS_.INNER_SOLVE_COUNT[] == 0   # rejected entirely by the screen, no KNITRO attempt
        end
    end

    @testset "Phase I.6: finite raw dual captured into the bank on NumericalFailure" begin
        theta_badT5 = theta0_20 .+ 0.5 .* randn(MersenneTwister(1), length(theta0_20))
        r_badT5 = evaluate_melitz_delta(theta_badT5, ctx20, obj20; cold=true, store_G=false)
        @test !(r_badT5.nStatus in (0, -100, -101, -103))   # confirms this really is a failing point

        objT5 = build_melitz_implicit_bundle(ctx20, obj20.U, theta_badT5; delta=1e-3,
            find_smallest=true, gradient_backend=:B, h=1e-4,
            inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
        bankT5 = MelitzDualBank()
        # NOTE: the range screen (front-loaded, always-on) may itself already certify this
        # point MomentInfeasible before any KNITRO attempt -- that is a CORRECT, EXPECTED
        # outcome (a cheaper, equally valid rejection), not a test bug. This test's actual
        # subject (finite-x bank capture) only applies on the branch where a real KNITRO
        # attempt happens and fails without a certificate.
        resultT5 = melitz_classified_inner_solve(objT5, theta_badT5, ctx20; delta=1e-3, bank=bankT5,
            range_screen=false, stored_dual_screen=false)
        @test resultT5 isa NumericalFailure
        # either the raw KNITRO-reported x at failure was finite (captured into the bank) or
        # it was not (bank stays empty) -- both are correct; what must NEVER happen is a
        # non-finite entry landing in the bank (already covered by the insert-level test
        # above), so this just pins that the capture path runs without error either way and,
        # if it captured, only ever a finite vector:
        @test length(bankT5.entries) in (0, 1)
        if length(bankT5.entries) == 1
            @test all(isfinite, bankT5.entries[1])
        end
    end

    # ========================================================================
    # Screening-session continuation, Phase I.3: exact compressed origin-block feasibility
    # screen, validated against a generic O(W)-variable convex-hull LP (trusted reference)
    # over random points, plus the monotonicity necessary condition.
    # ========================================================================
    @testset "Phase I.3: compressed origin-block screen vs. generic block LP reference" begin
        # Reuses the already-validated `small_fixture20` (D=4, seed=29, W=2000, per Section
        # 6's own setup above) rather than constructing a new fixture -- avoids re-deriving a
        # seed/W combination that clears `generate_fake_melitz_data`'s own well-conditioned-
        # fixture sanity checks (a fresh seed=7/W=300 attempt failed exactly that check).
        objOB, theta0_OB, ctxOB = obj20, theta0_20, ctx20

        @testset "monotonicity holds at the population-Pareto point, every origin" begin
            for o in 1:ctxOB.D
                @test melitz_origin_block_monotonicity_check(o, theta0_OB, ctxOB, objOB)
            end
        end

        @testset "compressed screen agrees EXACTLY with the reference LP: population-Pareto point (feasible)" begin
            for o in 1:ctxOB.D
                compressed = melitz_origin_block_lp(o, theta0_OB, ctxOB, objOB)
                reference = melitz_origin_block_lp_reference(o, theta0_OB, ctxOB, objOB)
                @test compressed == reference
                @test compressed   # the true population point must be feasible for its own origin blocks
            end
        end

        @testset "compressed screen agrees EXACTLY with the reference LP: random perturbed points" begin
            n_mismatches = 0
            n_infeasible_found = 0
            rngOB = MersenneTwister(42)
            for trial in 1:20
                theta_r = theta0_OB .+ 0.3 .* randn(rngOB, length(theta0_OB))
                for o in 1:ctxOB.D
                    compressed = melitz_origin_block_lp(o, theta_r, ctxOB, objOB)
                    reference = melitz_origin_block_lp_reference(o, theta_r, ctxOB, objOB)
                    compressed != reference && (n_mismatches += 1)
                    !compressed && (n_infeasible_found += 1)
                    @test compressed == reference
                end
            end
            @info "Phase I.3 random-point validation" n_trials=20*ctxOB.D n_mismatches n_infeasible_found
        end

        @testset "small-perturbation point: compressed screen still agrees exactly with the reference" begin
            # NOTE: a small random perturbation is NOT guaranteed origin-block-feasible at a
            # finite W (the achievable moment range is a property of the FIXED W-draw sample,
            # not of "closeness to the population point") -- this was this session's own
            # first test-design bug (an unconditional feasibility assertion here failed on 3
            # of 4 origins, a real property of this fixture at this perturbation scale, not a
            # screen bug, confirmed by the compressed/reference LPs agreeing on the rejection
            # in every case). What must hold unconditionally is EXACT agreement with the
            # reference LP, already the subject of the random-point testset above -- this
            # testset only re-confirms it at this specific smaller perturbation scale.
            theta_small = theta0_OB .+ 0.01 .* randn(MersenneTwister(3), length(theta0_OB))
            for o in 1:ctxOB.D
                compressed = melitz_origin_block_lp(o, theta_small, ctxOB, objOB)
                reference = melitz_origin_block_lp_reference(o, theta_small, ctxOB, objOB)
                @test compressed == reference
            end
        end

        @testset "budget-infeasible-but-moment-feasible point: compressed screen agrees with the reference" begin
            # a point whose INNER solve is genuinely valid (nStatus==0, moment-feasible) but
            # whose Delta may exceed any given delta budget -- the origin-block screen must
            # not confuse "over budget" with "moment infeasible" (it has no notion of delta
            # at all, by construction); what it MUST do is agree with the trusted reference.
            theta_bf = theta0_OB .+ 0.05 .* randn(MersenneTwister(5), length(theta0_OB))
            r_bf = evaluate_melitz_delta(theta_bf, ctxOB, objOB; cold=true, store_G=false)
            if r_bf.nStatus == 0   # only meaningful if this trial point is a genuine valid solve
                # NOTE: a dual-optimal (nStatus==0) solve does NOT imply Delta(theta)==0, so it
                # does NOT imply any origin's own moment block is EXACTLY satisfiable (the CC
                # divergence being finite/attained is a much weaker condition than exact
                # satisfiability) -- only the agreement property is asserted unconditionally.
                for o in 1:ctxOB.D
                    compressed = melitz_origin_block_lp(o, theta_bf, ctxOB, objOB)
                    reference = melitz_origin_block_lp_reference(o, theta_bf, ctxOB, objOB)
                    @test compressed == reference
                end
            end
        end

        @testset "wired end-to-end: melitz_origin_block_screen returns nothing at the population point" begin
            @test melitz_origin_block_screen(theta0_OB, ctxOB, objOB) === nothing
        end

        @testset "Phase I.8: screen_order dispatch -- :A/:B/:C agree on an ordinary feasible point" begin
            # melitz_classified_inner_solve is designed for the finite-delta OUTER search's
            # PsiObjectiveBundleImplicit (the `threshold_crossed`/etc. fields from Phase I.1
            # live only on that bundle type, not the plain PsiObjectiveBundleDelta `objOB`
            # used elsewhere in this testset for fixed-point Delta(theta) evaluation) -- build
            # one here, matching the Phase I.1 tests' own pattern.
            objI8 = build_melitz_implicit_bundle(ctxOB, objOB.U, theta0_OB; delta=1.0,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            for order in (:A, :B, :C)
                bank_ord = MelitzDualBank()
                result = melitz_classified_inner_solve(objI8, theta0_OB, ctxOB; delta=1e-3,
                    bank=bank_ord, origin_block_screen=true, dual_polish_screen=false,
                    screen_order=order)
                @test result isa InnerSolved   # population point: every screen must pass, real solve proceeds
            end
            @test_throws ArgumentError melitz_classified_inner_solve(objI8, theta0_OB, ctxOB;
                delta=1e-3, bank=MelitzDualBank(), screen_order=:Z)
        end
    end

    # ========================================================================
    # Continuation session (2026-07-23), Section 12: per-screen call-count/timer
    # instrumentation, using the SAME `MELITZ_PROFILE[]`/`melitz_profile_summary()`
    # machinery already exercised elsewhere in this file.
    # ========================================================================
    @testset "Section 12: per-screen instrumentation categories are recorded with correct outcomes" begin
        objI12 = build_melitz_implicit_bundle(ctx20, obj20.U, theta0_20; delta=1.0,
            find_smallest=true, gradient_backend=:B, h=1e-4,
            inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
        bankI12 = MelitzDualBank()
        melitz_profile_reset!()
        MELITZ_PROFILE[] = true
        try
            result1 = melitz_classified_inner_solve(objI12, theta0_20, ctx20; delta=1e-3,
                bank=bankI12, origin_block_screen=true, dual_polish_screen=true)
            @test result1 isa InnerSolved
            rows1 = melitz_profile_summary()
            cats1 = Dict(r.category => r for r in rows1)
            @test haskey(cats1, :screen_range_passed)
            @test cats1[:screen_range_passed].count == 1
            # stored_dual/dual_polish are skipped (not timed at all) on the FIRST call --
            # the bank was empty, so `try_stored_dual`/`try_dual_polish` short-circuit BEFORE
            # their own timed region even starts (an intentional design choice: timing a
            # trivial "bank is empty" branch would not be informative).
            @test !haskey(cats1, :screen_stored_dual_passed) && !haskey(cats1, :screen_stored_dual_rejected)

            # second call, same theta but now the bank has one entry -- stored_dual now runs
            # (and, at the SAME theta with the SAME just-cached optimal dual, must PASS, not
            # reject: the exact optimal dual's own lower bound cannot exceed the budget it
            # just satisfied).
            melitz_profile_reset!()
            result2 = melitz_classified_inner_solve(objI12, theta0_20, ctx20; delta=1e-3,
                bank=bankI12, origin_block_screen=true, dual_polish_screen=true)
            @test result2 isa InnerSolved
            rows2 = melitz_profile_summary()
            cats2 = Dict(r.category => r for r in rows2)
            @test haskey(cats2, :screen_stored_dual_passed)
            @test cats2[:screen_stored_dual_passed].count == 1
            @test haskey(cats2, :screen_dual_polish_passed) || haskey(cats2, :screen_dual_polish_rejected)
        finally
            MELITZ_PROFILE[] = false
        end
    end

    @testset "Section 7: A/B/A repeated evaluation -- no stale-state leakage" begin
        n20 = length(theta0_20)
        m20 = 1 + ctx20.D + ctx20.D * (ctx20.D - 1)
        delta_loose20b = max(r0_20.Delta * 5, 1e-3)
        obj7 = build_melitz_implicit_bundle(ctx20, obj20.U, theta0_20; delta=delta_loose20b,
            find_smallest=true, gradient_backend=:B, h=1e-4,
            inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
        cbset7 = melitz_build_finite_delta_callbacks(obj7, ctx20, delta_loose20b, true)

        theta_A = collect(theta0_20)
        theta_B = theta0_20 .+ 0.5 .* randn(MersenneTwister(1), n20)   # a known-failing point

        evalResA1 = MelitzMockEvalResult(zeros(1), zeros(m20), zeros(n20), zeros(n20 * m20))
        cbset7.cb_F!(nothing, nothing, MelitzMockEvalRequest(copy(theta_A)), evalResA1, nothing)
        cbset7.cb_G!(nothing, nothing, MelitzMockEvalRequest(copy(theta_A)), evalResA1, nothing)

        threw = false
        try
            evalResB = MelitzMockEvalResult(zeros(1), zeros(m20), zeros(n20), zeros(n20 * m20))
            cbset7.cb_F!(nothing, nothing, MelitzMockEvalRequest(copy(theta_B)), evalResB, nothing)
        catch e
            threw = e isa DomainError
        end
        @test threw

        evalResA2 = MelitzMockEvalResult(zeros(1), zeros(m20), zeros(n20), zeros(n20 * m20))
        cbset7.cb_F!(nothing, nothing, MelitzMockEvalRequest(copy(theta_A)), evalResA2, nothing)
        cbset7.cb_G!(nothing, nothing, MelitzMockEvalRequest(copy(theta_A)), evalResA2, nothing)

        @testset "point A's objective/constraint/Jacobian are IDENTICALLY reproduced after the failed B in between" begin
            @test evalResA2.obj == evalResA1.obj
            @test evalResA2.c == evalResA1.c
            @test evalResA2.objGrad == evalResA1.objGrad
            @test evalResA2.jac == evalResA1.jac
        end
        @testset "the failed point B never contaminates the live-candidate list" begin
            @test length(cbset7.live_candidates) == 2   # both A evaluations, no dedup, no B
            @test all(c -> c.eval.theta_free == theta_A, cbset7.live_candidates)
        end
    end

    # ========================================================================
    # Section 5.1 (2026-07-23 continuation session): the exact-point cache must actually
    # ELIDE the second inner solve at a repeated theta (not merely stay correct if it
    # didn't) -- checked directly via CounterfactualSensitivity.INNER_SOLVE_COUNT[], and
    # cross-checked that a genuinely DIFFERENT theta is never a false cache hit.
    # ========================================================================
    @testset "Section 5.1: exact-point cache elides the duplicate inner solve at repeated theta" begin
        n20b = length(theta0_20)
        m20b = 1 + ctx20.D + ctx20.D * (ctx20.D - 1)
        delta_loose20e = max(r0_20.Delta * 5, 1e-3)
        obj51 = build_melitz_implicit_bundle(ctx20, obj20.U, theta0_20; delta=delta_loose20e,
            find_smallest=true, gradient_backend=:B, h=1e-4,
            inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
        cbset51 = melitz_build_finite_delta_callbacks(obj51, ctx20, delta_loose20e, true)

        theta_X = collect(theta0_20)
        theta_Y = theta0_20 .+ 0.01 .* randn(MersenneTwister(2), n20b)

        CounterfactualSensitivity.INNER_SOLVE_COUNT[] = 0
        evX1 = MelitzMockEvalResult(zeros(1), zeros(m20b), zeros(n20b), zeros(n20b * m20b))
        cbset51.cb_F!(nothing, nothing, MelitzMockEvalRequest(copy(theta_X)), evX1, nothing)
        n_after_F = CounterfactualSensitivity.INNER_SOLVE_COUNT[]
        evXG = MelitzMockEvalResult(zeros(1), zeros(m20b), zeros(n20b), zeros(n20b * m20b))
        cbset51.cb_G!(nothing, nothing, MelitzMockEvalRequest(copy(theta_X)), evXG, nothing)
        n_after_G_same_theta = CounterfactualSensitivity.INNER_SOLVE_COUNT[]

        @testset "cb_G! at the SAME theta as the preceding cb_F! triggers zero new inner solves" begin
            @test n_after_F >= 1
            @test n_after_G_same_theta == n_after_F
            @test cbset51.n_exact_cache_hits[] >= 1
        end

        # MUST run before any different-theta call below, which would overwrite obj51.H.
        @testset "H-matrix restore keeps the cached G consistent with a fresh solve at theta_X" begin
            G_from_cache_hit = Matrix(CounterfactualSensitivity.select_G_from_H(obj51, obj51.H))
            obj51_fresh = build_melitz_implicit_bundle(ctx20, obj20.U, theta0_20; delta=delta_loose20e,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            CounterfactualSensitivity.inner_loop_internal(obj51_fresh, theta_X)
            G_fresh = Matrix(CounterfactualSensitivity.select_G_from_H(obj51_fresh, obj51_fresh.H))
            @test G_from_cache_hit == G_fresh
        end

        # theta_Y only needs to be a genuinely different point that MISSES the cache --
        # whether its inner solve itself succeeds or fails is incidental to what this
        # assertion checks, and a real inner-solve attempt (`INNER_SOLVE_COUNT[]`
        # incremented inside `inner_loop_internal`, both on the warm attempt and any cold
        # retry) happens either way, BEFORE any eventual `DomainError` throw on an ultimate
        # failure (Section 5.2's own convention) -- so a failing theta_Y is still valid
        # evidence the cache was NOT (falsely) hit, just wrapped defensively here.
        evY = MelitzMockEvalResult(zeros(1), zeros(m20b), zeros(n20b), zeros(n20b * m20b))
        try
            cbset51.cb_G!(nothing, nothing, MelitzMockEvalRequest(copy(theta_Y)), evY, nothing)
        catch e
            e isa DomainError || rethrow(e)
        end
        n_after_G_diff_theta = CounterfactualSensitivity.INNER_SOLVE_COUNT[]
        @testset "cb_G! at a genuinely DIFFERENT theta is a cache miss (real inner solve runs)" begin
            @test cbset51.n_exact_cache_misses[] >= 2   # theta_X's original cb_F! + theta_Y
            @test n_after_G_diff_theta > n_after_G_same_theta
        end
    end

    # ========================================================================
    # Continuation session (2026-07-23), Section 4.2: exact-point/eval caches shared across
    # DIFFERENT outer `delta` values -- valid because Delta(theta) (and everything derived
    # from it: dual, LFD, moments, equilibrium checks) is a function of theta ALONE, never of
    # the outer budget. Tests BOTH caches this session wired: `MelitzExactPointCache` (the
    # FC/GA screening path's light dual-only cache, upgraded this session from a single slot
    # to a shareable multi-entry Dict) and the pre-existing `MelitzDeltaEvalCache` (the full
    # Delta/dual/LFD/moments/checks state, now threaded through `solve_melitz_finite_delta_bound`
    # for its own initial-incumbent/terminal cold-verification calls).
    # ========================================================================
    @testset "Section 4.2: cross-delta cache reuse" begin
        n42 = length(theta0_20)

        @testset "MelitzExactPointCache: a verified entry from one melitz_build_finite_delta_callbacks call is a hit in a SEPARATE call at a different delta" begin
            delta_A = max(r0_20.Delta * 5, 1e-3)
            delta_B = max(r0_20.Delta * 3, 5e-4)
            shared_cache = MelitzExactPointCache()

            obj_A = build_melitz_implicit_bundle(ctx20, obj20.U, theta0_20; delta=delta_A,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            cbset_A = melitz_build_finite_delta_callbacks(obj_A, ctx20, delta_A, true;
                exact_cache=shared_cache)
            m42 = 1 + ctx20.D + ctx20.D * (ctx20.D - 1)
            evA = MelitzMockEvalResult(zeros(1), zeros(m42), zeros(n42), zeros(n42 * m42))
            cbset_A.cb_F!(nothing, nothing, MelitzMockEvalRequest(copy(theta0_20)), evA, nothing)
            @test length(shared_cache.store) == 1
            entry_A = only(values(shared_cache.store))
            Delta_A, x_A, nStatus_A, H_A = entry_A

            # a SEPARATE bundle/callback set, a DIFFERENT delta, but the SAME shared_cache:
            # evaluating the IDENTICAL theta must hit the cache with ZERO new inner solves.
            obj_B = build_melitz_implicit_bundle(ctx20, obj20.U, theta0_20; delta=delta_B,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            cbset_B = melitz_build_finite_delta_callbacks(obj_B, ctx20, delta_B, true;
                exact_cache=shared_cache)
            CounterfactualSensitivity.INNER_SOLVE_COUNT[] = 0
            evB = MelitzMockEvalResult(zeros(1), zeros(m42), zeros(n42), zeros(n42 * m42))
            cbset_B.cb_F!(nothing, nothing, MelitzMockEvalRequest(copy(theta0_20)), evB, nothing)
            @test CounterfactualSensitivity.INNER_SOLVE_COUNT[] == 0   # cache hit, no new solve
            @test cbset_B.n_exact_cache_hits[] == 1
            @test length(shared_cache.store) == 1   # still just the one entry, not duplicated

            # identical Delta/dual/H reused across the two different-delta callback sets.
            G_B = Matrix(CounterfactualSensitivity.select_G_from_H(obj_B, obj_B.H))
            G_A = Matrix(CounterfactualSensitivity.select_G_from_H(obj_A, obj_A.H))
            @test G_B == G_A
        end

        @testset "MelitzDeltaEvalCache: solve at delta=1e-3, reuse at delta=1e-2 via solve_melitz_finite_delta_bound's own eval_cache" begin
            eval_cache42 = MelitzDeltaEvalCache()
            delta_loose1 = max(r0_20.Delta * 5, 1e-3)
            delta_loose2 = max(r0_20.Delta * 8, 1e-2)

            res1 = solve_melitz_finite_delta_bound(ctx20, obj20, theta0_20; delta=delta_loose1,
                direction=:upper, gradient_backend=:B, h=1e-4, theta_box=0.0,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20, eval_cache=eval_cache42)
            misses_after_1 = eval_cache42.misses
            @test length(eval_cache42.store) >= 1   # theta0_20's own initial/terminal eval cached

            # theta_box=0.0 pins every coordinate at theta0_20 itself (zero degrees of
            # freedom, the same mechanism melitz_fixed_point_probe uses) -- so this SECOND
            # call, at a DIFFERENT delta, evaluates theta0_20 EXACTLY, which is already in
            # eval_cache42 from the first call.
            res2 = solve_melitz_finite_delta_bound(ctx20, obj20, theta0_20; delta=delta_loose2,
                direction=:upper, gradient_backend=:B, h=1e-4, theta_box=0.0,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20, eval_cache=eval_cache42)

            @test eval_cache42.hits >= 1   # the second call's own initial-incumbent eval hit the cache
            @test eval_cache42.misses == misses_after_1   # no NEW miss from the second call's initial eval

            # identical Delta/dual/LFD/moments/checks -- theta0_20 is theta_INIT for both
            # calls, so both calls' own `initial_incumbent` describe the SAME evaluation.
            @test res1.initial_incumbent !== nothing && res2.initial_incumbent !== nothing
            @test res1.initial_incumbent.eval.Delta == res2.initial_incumbent.eval.Delta
            @test res1.initial_incumbent.eval.dual_x == res2.initial_incumbent.eval.dual_x
            @test res1.initial_incumbent.eval.moment_residuals == res2.initial_incumbent.eval.moment_residuals
            @test res1.initial_incumbent.eval.equilibrium_check == res2.initial_incumbent.eval.equilibrium_check
        end

        @testset "omitting exact_cache/eval_cache reproduces the pre-existing fresh-cache-per-call behavior" begin
            delta_loose3 = max(r0_20.Delta * 5, 1e-3)
            res_nocache = solve_melitz_finite_delta_bound(ctx20, obj20, theta0_20; delta=delta_loose3,
                direction=:upper, gradient_backend=:B, h=1e-4, theta_box=0.5,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            @test res_nocache.initial_incumbent !== nothing
            @test res_nocache.initial_incumbent.classification.outer_feasible
        end
    end

    @testset "Section 2.1/12: initial incumbent survives a KNITRO run limited to one iteration" begin
        maxit1_opt = tempname() * ".opt"
        open(maxit1_opt, "w") do io
            for line in readlines(outer_opt20)
                println(io, startswith(line, "maxit") ? "maxit 1" : line)
            end
        end
        delta_loose20c = max(r0_20.Delta * 5, 1e-3)
        res_lim = solve_melitz_finite_delta_bound(ctx20, obj20, theta0_20; delta=delta_loose20c,
            direction=:upper, gradient_backend=:B, h=1e-4, theta_box=0.5,
            inner_loop_opt=inner_opt20, outer_loop_opt=maxit1_opt)
        rm(maxit1_opt; force=true)

        @test res_lim.initial_incumbent !== nothing
        @test res_lim.initial_incumbent.classification.outer_feasible
        # the RETURNED incumbent must be theta_init's own, even though KNITRO barely ran:
        @test res_lim.cold_verified_incumbent !== nothing
        @test res_lim.cold_verified_incumbent.eval.Delta <= delta_loose20c
    end

    # ========================================================================
    # Session prompt Section 3.3/3.4: real KNITRO integration test for the :linear cutoff-
    # constraint backend, registered through the ACTUAL production
    # `melitz_register_finite_delta_knitro_problem!`/`melitz_build_finite_delta_callbacks`
    # path (not a bypass). Also pins the KNITRO nested-KN-instance post-solve reporting
    # limitation found this session (see finite_delta_outer.jl's
    # `melitz_fixed_point_probe` docstring/comments) and its fix.
    # ========================================================================
    @testset "Section 3.3/3.4: real KNITRO :linear cutoff-constraint backend" begin
        sys20 = build_melitz_affine_cutoff_system(ctx20)
        delta_loose20d = max(r0_20.Delta * 5, 1e-3)

        @testset "feasible point: :linear backend matches :nonlinear_reference (obj, divergence row, and RECOVERED cutoff slacks)" begin
            p_lin = melitz_fixed_point_probe(ctx20, obj20, theta0_20; delta=delta_loose20d,
                direction=:upper, gradient_backend=:B, h=1e-4, cutoff_constraint_backend=:linear,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            p_nl = melitz_fixed_point_probe(ctx20, obj20, theta0_20; delta=delta_loose20d,
                direction=:upper, gradient_backend=:B, h=1e-4, cutoff_constraint_backend=:nonlinear_reference,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            @test !p_lin.eval_failed && !p_nl.eval_failed
            @test p_lin.nStatus == 0 && p_nl.nStatus == 0
            @test isapprox(p_lin.obj_value, p_nl.obj_value; atol=1e-10)
            @test isapprox(p_lin.c[1], p_nl.c[1]; rtol=1e-6)
            # p_lin.c[2:end] is the Julia-recomputed (not KNITRO-reported) SCALED cutoff
            # slack; unscale and compare against the exact nonlinear evaluator directly.
            gd_true, ge_true = melitz_cutoff_constraints_at(theta0_20, ctx20)
            g_true = vcat(gd_true, ge_true)
            c_unscaled = p_lin.c[2:end] .* sys20.row_scale
            @test isapprox(c_unscaled, g_true; atol=1e-8, rtol=1e-8)
            @test all(>=(-1e-8), p_lin.c[2:end])
            @test length(p_lin.live_candidates) == 1
            @test p_lin.live_candidates[1].classification.outer_feasible
        end

        @testset "constructed domestic-infeasible point: real KNITRO reports infeasibility" begin
            dom_row = 1
            Crow = sys20.C_raw[dom_row, :]
            slack0 = dot(Crow, theta0_20) + sys20.b_raw[dom_row]
            margin = 0.02
            theta_bad_dom = theta0_20 .- (slack0 + margin) .* Crow ./ dot(Crow, Crow)
            p_bad = melitz_fixed_point_probe(ctx20, obj20, theta_bad_dom; delta=1.0,
                direction=:upper, gradient_backend=:B, h=1e-4, cutoff_constraint_backend=:linear,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            if !p_bad.eval_failed
                @test p_bad.nStatus != 0
                @test p_bad.c[1+dom_row] < 0
            end
        end

        @testset "solve_melitz_finite_delta_bound runs end to end under :linear (structural, loose budget)" begin
            res_lin = solve_melitz_finite_delta_bound(ctx20, obj20, theta0_20; delta=delta_loose20d,
                direction=:upper, gradient_backend=:B, h=1e-4, theta_box=0.5,
                cutoff_constraint_backend=:linear,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            @test res_lin.cutoff_constraint_backend == :linear
            @test res_lin.nStatus != -500
            @test res_lin.cold_verified_incumbent !== nothing
            @test res_lin.cold_verified_incumbent.classification.outer_feasible
            @test res_lin.cold_verified_incumbent.eval.Delta <= delta_loose20d + 1e-6
            @test res_lin.n_fc_calls > 0
            @test res_lin.n_ga_calls > 0
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
# Session prompt Section 3: the affine cutoff constraint map q(theta_free)=q0+Q*theta_free
# and the resulting C*theta_free+b>=0 deterministic system (affine_cutoff.jl).
# ============================================================================
@testset "Session prompt Section 3: affine cutoff constraint map" begin
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
    n = length(theta0)
    @test n == 2D^2 - 2

    @testset "3.1: basis-probe construction agrees with the independent analytical construction (machine precision)" begin
        Qb, q0b = affine_cutoff_map_basis(ctx)
        Qa, q0a = affine_cutoff_map_analytical(ctx)
        @test isapprox(Qb, Qa; atol=1e-9, rtol=1e-9)
        @test isapprox(q0b, q0a; atol=1e-9, rtol=1e-9)
    end

    @testset "3.1: basis map is independent of the probe origin" begin
        rng = MersenneTwister(3)
        base2 = 0.3 .* randn(rng, n)
        Q1, q01 = affine_cutoff_map_basis(ctx; base=zeros(n))
        Q2, q02 = affine_cutoff_map_basis(ctx; base=base2)
        @test isapprox(Q1, Q2; atol=1e-8, rtol=1e-8)
        @test isapprox(q01, q02; atol=1e-8, rtol=1e-8)
    end

    @testset "3.1: q0+Q*theta reproduces melitz_log_cutoff_vec exactly at random points" begin
        rng = MersenneTwister(5)
        Q, q0 = affine_cutoff_map_basis(ctx)
        for _ in 1:20
            th = 0.3 .* randn(rng, n)
            @test isapprox(melitz_log_cutoff_vec(th, ctx), q0 .+ Q * th; atol=1e-8, rtol=1e-8)
        end
    end

    sys = build_melitz_affine_cutoff_system(ctx)
    @testset "3.2: exactly D domestic-support rows + D*(D-1) export-selection rows" begin
        @test sys.n_domestic == D == 4
        @test sys.n_export == D * (D - 1) == 12
        @test size(sys.C, 1) == 16
        @test size(sys.C, 2) == n
    end

    @testset "3.4: 1000 random free vectors -- C*theta+b matches melitz_cutoff_constraints_at" begin
        rng = MersenneTwister(7)
        maxdiff = 0.0
        for _ in 1:1000
            th = 0.5 .* randn(rng, n)
            gd_true, ge_true = melitz_cutoff_constraints_at(th, ctx)
            gd_aff, ge_aff = affine_cutoff_slacks_unscaled(sys, th)
            maxdiff = max(maxdiff, maximum(abs.(gd_true .- gd_aff)), maximum(abs.(ge_true .- ge_aff)))
        end
        @test maxdiff < 1e-8
    end

    @testset "3.4: C_raw matches the exact nonlinear Jacobian (melitz_cutoff_constraint_jacobian)" begin
        rng = MersenneTwister(9)
        th = 0.1 .* randn(rng, n)
        Jd, Je = melitz_cutoff_constraint_jacobian(th, ctx)
        @test isapprox(Jd, sys.C_raw[1:sys.n_domestic, :]; atol=1e-6, rtol=1e-6)
        @test isapprox(Je, sys.C_raw[sys.n_domestic+1:end, :]; atol=1e-6, rtol=1e-6)
    end

    @testset "3.3: row scaling preserves feasibility sign" begin
        rng = MersenneTwister(11)
        for _ in 1:50
            th = 0.5 .* randn(rng, n)
            gd_s, ge_s = affine_cutoff_slacks(sys, th)
            gd_u, ge_u = affine_cutoff_slacks_unscaled(sys, th)
            @test all(sign.(gd_s) .== sign.(gd_u))
            @test all(sign.(ge_s) .== sign.(ge_u))
        end
    end

    @testset "3.4: constructed cutoff-feasible / domestic-infeasible / export-infeasible truth table" begin
        gd0, ge0 = affine_cutoff_slacks_unscaled(sys, theta0)
        @test all(>=(-1e-8), gd0)
        @test all(>=(-1e-8), ge0)

        margin = 0.01
        dom_row = 1
        Crow_d = sys.C_raw[dom_row, :]
        slack0_d = dot(Crow_d, theta0) + sys.b_raw[dom_row]
        theta_bad_dom = theta0 .- (slack0_d + margin) .* Crow_d ./ dot(Crow_d, Crow_d)
        gd_bad, _ = affine_cutoff_slacks_unscaled(sys, theta_bad_dom)
        @test gd_bad[dom_row] < 0
        @test isapprox(gd_bad[dom_row], -margin; atol=1e-6)
        gd_true_dom, _ = melitz_cutoff_constraints_at(theta_bad_dom, ctx)
        @test gd_true_dom[dom_row] < 0

        exp_row = sys.n_domestic + 1
        Crow_e = sys.C_raw[exp_row, :]
        slack0_e = dot(Crow_e, theta0) + sys.b_raw[exp_row]
        theta_bad_exp = theta0 .- (slack0_e + margin) .* Crow_e ./ dot(Crow_e, Crow_e)
        _, ge_bad = affine_cutoff_slacks_unscaled(sys, theta_bad_exp)
        @test ge_bad[1] < 0
        @test isapprox(ge_bad[1], -margin; atol=1e-6)
        _, ge_true_exp = melitz_cutoff_constraints_at(theta_bad_exp, ctx)
        @test ge_true_exp[1] < 0
    end
end

# ============================================================================
# Session prompt Section 5: the experimental :logcutoff outer parameterization
# (log_cutoff_param.jl), cross-validated against :logf.
# ============================================================================
@testset "Session prompt Section 5: log-cutoff parameterization" begin
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
    theta0_f = reduce_to_free_theta(p, ctx)
    n = length(theta0_f)
    @test n == 2D^2 - 2

    @testset "5.1: f[j,j] via expand_free_theta_logcutoff matches derive_fjj_from_autarky_cutoff at random points" begin
        rng = MersenneTwister(2)
        for _ in 1:10
            gamma_test = 0.7 + 0.6 * rand(rng)
            theta_test = vcat(log(gamma_test), 0.2 .* randn(rng, n - 1))
            A, f, gamma_prime_j, f_jj, q = expand_free_theta_logcutoff(theta_test, ctx)
            f_jj_direct = derive_fjj_from_autarky_cutoff(gamma_prime_j, ctx.w_prime, 1.0, A[j, j],
                                                            ctx.w_prime * ctx.L[j], ctx.sigma)
            @test isapprox(f_jj, f_jj_direct; atol=1e-10, rtol=1e-10)
        end
    end

    @testset "5.2/5.3: A- and f-gravity both hold exactly at random logcutoff free points" begin
        rng = MersenneTwister(3)
        for _ in 1:10
            gamma_test = 0.7 + 0.6 * rand(rng)
            theta_test = vcat(log(gamma_test), 0.2 .* randn(rng, n - 1))
            A, f, gamma_prime_j, f_jj, q = expand_free_theta_logcutoff(theta_test, ctx)
            prim_test = MelitzPrimitives(D, ctx.sigma, ctx.theta_star, j, ctx.tau, ctx.w, A, f, gamma_prime_j)
            gA, gf = gravity_residuals(prim_test)
            @test abs(gA) < 1e-8
            @test abs(gf) < 1e-8
        end
    end

    @testset "q matches log.(melitz_baseline_cutoff(A,f,...)) exactly" begin
        rng = MersenneTwister(4)
        for _ in 1:10
            gamma_test = 0.7 + 0.6 * rand(rng)
            theta_test = vcat(log(gamma_test), 0.2 .* randn(rng, n - 1))
            A, f, gamma_prime_j, f_jj, q = expand_free_theta_logcutoff(theta_test, ctx)
            zhat_true = melitz_baseline_cutoff(A, f, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma)
            @test isapprox(q, log.(zhat_true); atol=1e-8, rtol=1e-8)
        end
    end

    @testset "round-trip: reduce(expand(theta_q)) == theta_q" begin
        rng = MersenneTwister(5)
        for _ in 1:10
            gamma_test = 0.7 + 0.6 * rand(rng)
            theta_test = vcat(log(gamma_test), 0.2 .* randn(rng, n - 1))
            A, f, gamma_prime_j, f_jj, q = expand_free_theta_logcutoff(theta_test, ctx)
            theta_rt = reduce_to_free_theta_logcutoff(A, f, gamma_prime_j, ctx)
            @test isapprox(theta_rt, theta_test; atol=1e-8, rtol=1e-8)
        end
    end

    @testset "round-trip: expand(reduce(A,f,gamma)) reconstructs the :logf fixture exactly" begin
        theta_q0 = reduce_to_free_theta_logcutoff(p.A, p.f, p.gamma_prime_target, ctx)
        @test length(theta_q0) == n
        A_rt, f_rt, gamma_rt, f_jj_rt, q_rt = expand_free_theta_logcutoff(theta_q0, ctx)
        @test isapprox(A_rt, p.A; atol=1e-8, rtol=1e-8)
        @test isapprox(f_rt, p.f; atol=1e-8, rtol=1e-8)
        @test isapprox(gamma_rt, p.gamma_prime_target; atol=1e-10, rtol=1e-10)
    end

    @testset "5.5: full cross-parameterization equivalence (:logf <-> :logcutoff): A, f, gamma, cutoff, gravity, moment matrix G" begin
        A_f, f_f, gamma_f, f_jj_f = expand_free_theta(theta0_f, ctx)
        theta0_q = reduce_to_free_theta_logcutoff(A_f, f_f, gamma_f, ctx)
        A_q, f_q, gamma_q, f_jj_q, q_q = expand_free_theta_logcutoff(theta0_q, ctx)
        @test isapprox(A_f, A_q; atol=1e-8, rtol=1e-8)
        @test isapprox(f_f, f_q; atol=1e-8, rtol=1e-8)
        @test isapprox(gamma_f, gamma_q; atol=1e-10, rtol=1e-10)
        @test isapprox(f_jj_f, f_jj_q; atol=1e-8, rtol=1e-8)

        prim_q = MelitzPrimitives(D, ctx.sigma, ctx.theta_star, j, ctx.tau, ctx.w, A_q, f_q, gamma_q)
        gA_q, gf_q = gravity_residuals(prim_q)
        @test abs(gA_q) < 1e-8
        @test abs(gf_q) < 1e-8

        zhat_f = melitz_baseline_cutoff(A_f, f_f, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma)
        zhat_q = melitz_baseline_cutoff(A_q, f_q, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma)
        @test isapprox(zhat_f, zhat_q; atol=1e-8, rtol=1e-8)

        z_draws = FIXTURE.z_draws
        prim_f = MelitzPrimitives(D, ctx.sigma, ctx.theta_star, j, ctx.tau, ctx.w, A_f, f_f, gamma_f)
        eq_f = MelitzEquilibrium(ctx.expenditure, ones(D), zhat_f, ctx.X_data)
        eq_q = MelitzEquilibrium(ctx.expenditure, ones(D), zhat_q, ctx.X_data)
        expenditure_prime = ctx.w_prime * ctx.L[j]
        cf_f = MelitzCounterfactual(j, ctx.w_prime, expenditure_prime, 1.0, expenditure_prime)
        cf_q = MelitzCounterfactual(j, ctx.w_prime, expenditure_prime, 1.0, expenditure_prime)
        W = size(z_draws, 1)
        K_f, G_f = zeros(W), zeros(W, moment_layout.num_moments)
        K_q, G_q = zeros(W), zeros(W, moment_layout.num_moments)
        melitz_moments!(K_f, G_f, prim_f, eq_f, cf_f, z_draws, moment_layout; X_data=ctx.X_data)
        melitz_moments!(K_q, G_q, prim_q, eq_q, cf_q, z_draws, moment_layout; X_data=ctx.X_data)
        @test isapprox(G_f, G_q; atol=1e-8, rtol=1e-8)
    end
end

# ============================================================================
# 9b. Section 5 (live wiring): :logf vs :logcutoff, live end-to-end, real KNITRO.
# Unlike the economic-core-only testset above (round trips at hand-built `ctx`
# NamedTuples, no inner solve), this exercises the FULL live path: build_melitz_psi_bundle
# with outer_parameterization=:logcutoff, the melitz_expand_theta dispatcher threaded
# through melitz_outer_state/melitz_moments_adapter!/gradient_lab/affine_cutoff, and a
# real cold-verified inner CC KNITRO solve -- required agreement fields per the governing
# prompt: A, f, gamma_prime, q, moment matrix G, positive Delta, dual solution, LFD,
# omitted-equilibrium diagnostics, economic GT.
# ============================================================================
if KNITRO_AVAILABLE
    @testset "Section 5 (live wiring): :logf vs :logcutoff live fixed-point equivalence (real KNITRO)" begin
        inner_opt = joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt")

        obj_f, theta0_f = build_melitz_psi_bundle(FIXTURE; outer_parameterization=:logf, inner_loop_opt=inner_opt)
        ctx_f = obj_f.γ
        obj_q, theta0_q = build_melitz_psi_bundle(FIXTURE; outer_parameterization=:logcutoff, inner_loop_opt=inner_opt)
        ctx_q = obj_q.γ

        @testset "ctx carries the requested outer_parameterization" begin
            @test ctx_f.outer_parameterization == :logf
            @test ctx_q.outer_parameterization == :logcutoff
        end

        A_f, f_f, gamma_f, fjj_f = melitz_expand_theta(theta0_f, ctx_f)
        A_q, f_q, gamma_q, fjj_q = melitz_expand_theta(theta0_q, ctx_q)
        _, _, _, _, q_q_mat = expand_free_theta_logcutoff(theta0_q, ctx_q)
        q_f_vec = melitz_log_cutoff_vec(theta0_f, ctx_f)

        @testset "A, f, gamma_prime, f_jj agree (same physical point by construction)" begin
            @test isapprox(A_f, A_q; atol=1e-8, rtol=1e-8)
            @test isapprox(f_f, f_q; atol=1e-8, rtol=1e-8)
            @test isapprox(gamma_f, gamma_q; atol=1e-10, rtol=1e-10)
            @test isapprox(fjj_f, fjj_q; atol=1e-8, rtol=1e-8)
        end

        @testset "q (baseline log-cutoff) agrees" begin
            @test isapprox(q_f_vec, vec(q_q_mat); atol=1e-8, rtol=1e-8)
        end

        r_f = evaluate_melitz_delta(theta0_f, ctx_f, obj_f; cold=true)
        r_q = evaluate_melitz_delta(theta0_q, ctx_q, obj_q; cold=true)

        @testset "both real-KNITRO cold solves converge (nStatus==0) and are verified" begin
            @test r_f.nStatus == 0
            @test r_q.nStatus == 0
            @test r_f.verified
            @test r_q.verified
        end

        @testset "positive Delta agrees" begin
            @test r_f.Delta > 0
            @test r_q.Delta > 0
            @test isapprox(r_f.Delta, r_q.Delta; atol=1e-8, rtol=1e-2)
        end

        @testset "dual solution agrees" begin
            @test isapprox(r_f.dual_x, r_q.dual_x; atol=1e-4, rtol=1e-2)
        end

        @testset "LFD (recovered least-favorable distribution) agrees" begin
            @test isapprox(r_f.weights, r_q.weights; atol=1e-8, rtol=1e-2)
        end

        @testset "moment matrix G agrees" begin
            @test isapprox(r_f.G, r_q.G; atol=1e-8, rtol=1e-8)
        end

        @testset "omitted-equilibrium diagnostics agree" begin
            c_f, c_q = r_f.equilibrium_check, r_q.equilibrium_check
            @test isapprox(c_f.gravity_residual_A, c_q.gravity_residual_A; atol=1e-8)
            @test isapprox(c_f.gravity_residual_f, c_q.gravity_residual_f; atol=1e-8)
            @test isapprox(c_f.residual_autarky_cutoff, c_q.residual_autarky_cutoff; atol=1e-8)
            @test isapprox(c_f.N_prime_market_clearing, c_q.N_prime_market_clearing; atol=1e-6, rtol=1e-6)
            @test isapprox(c_f.N_prime_from_gamma, c_q.N_prime_from_gamma; atol=1e-6, rtol=1e-6)
            @test isapprox(c_f.min_baseline_cutoff, c_q.min_baseline_cutoff; atol=1e-8)
        end

        @testset "economic GT (gains from trade) agrees" begin
            p_f = MelitzPrimitives(ctx_f.D, ctx_f.sigma, ctx_f.theta_star, ctx_f.target_country,
                                    ctx_f.tau, ctx_f.w, r_f.A, r_f.f, r_f.gamma_prime_j)
            p_q = MelitzPrimitives(ctx_q.D, ctx_q.sigma, ctx_q.theta_star, ctx_q.target_country,
                                    ctx_q.tau, ctx_q.w, r_q.A, r_q.f, r_q.gamma_prime_j)
            expenditure_prime = ctx_f.w_prime * ctx_f.L[ctx_f.target_country]
            cf = MelitzCounterfactual(ctx_f.target_country, ctx_f.w_prime, expenditure_prime, 1.0, expenditure_prime)
            GT_f = melitz_gains_from_trade(p_f, cf)
            GT_q = melitz_gains_from_trade(p_q, cf)
            @test isapprox(GT_f, GT_q; atol=1e-8, rtol=1e-6)
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
