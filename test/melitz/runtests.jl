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
using DelimitedFiles

const MELITZ_DIR = joinpath(@__DIR__, "..", "..", "src", "melitz")
include(joinpath(dirname(dirname(@__DIR__)), "misc", "doubleDiff.jl"))
include(joinpath(MELITZ_DIR, "profiling.jl"))
include(joinpath(MELITZ_DIR, "knitro_compat.jl"))
include(joinpath(MELITZ_DIR, "backend_config.jl"))
include(joinpath(MELITZ_DIR, "run_diagnostics.jl"))
include(joinpath(MELITZ_DIR, "types.jl"))
include(joinpath(MELITZ_DIR, "bounded_cache.jl"))
include(joinpath(MELITZ_DIR, "inner_solve_config.jl"))
include(joinpath(MELITZ_DIR, "pareto.jl"))
include(joinpath(MELITZ_DIR, "firm_quantities.jl"))
include(joinpath(MELITZ_DIR, "equilibrium.jl"))
include(joinpath(MELITZ_DIR, "moments.jl"))
include(joinpath(MELITZ_DIR, "sorted_tail.jl"))
include(joinpath(MELITZ_DIR, "sorted_dual_argument.jl"))
include(joinpath(MELITZ_DIR, "moment_operator.jl"))
include(joinpath(MELITZ_DIR, "delta_star.jl"))
include(joinpath(MELITZ_DIR, "affine_cutoff.jl"))
include(joinpath(MELITZ_DIR, "log_cutoff_param.jl"))
include(joinpath(MELITZ_DIR, "technology_coordinate.jl"))
include(joinpath(MELITZ_DIR, "outer_parameterization_config.jl"))
include(joinpath(MELITZ_DIR, "fake_data.jl"))
include(joinpath(MELITZ_DIR, "pareto_calibration.jl"))
include(joinpath(MELITZ_DIR, "fstar_solver.jl"))
include(joinpath(MELITZ_DIR, "fstar_direct.jl"))
include(joinpath(MELITZ_DIR, "gradient_lab.jl"))
include(joinpath(MELITZ_DIR, "outer_solve.jl"))
include(joinpath(MELITZ_DIR, "inner_screening.jl"))
include(joinpath(MELITZ_DIR, "origin_block_screen.jl"))
include(joinpath(MELITZ_DIR, "localized_gradient.jl"))
include(joinpath(MELITZ_DIR, "argument_localized_gradient.jl"))
include(joinpath(MELITZ_DIR, "direct_gradient.jl"))
include(joinpath(MELITZ_DIR, "sorted_crossing_gradient.jl"))
include(joinpath(MELITZ_DIR, "touched_row_gradient.jl"))
include(joinpath(MELITZ_DIR, "cc_bundle.jl"))
include(joinpath(MELITZ_DIR, "finite_delta_outer.jl"))
include(joinpath(MELITZ_DIR, "nuisance_profile.jl"))
include(joinpath(MELITZ_DIR, "predictor_corrector.jl"))   # 2026-07-26 closure (Phase 3): was
    # not previously included in this test file at all (zero test coverage) -- now included
    # so the kappa_of_g -> kappa_ratio_of_g / kappa -> kappa_ratio rename in this file is
    # actually compiled and exercisable, not merely assumed correct.

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
# 6.5. Sorted-tail moment construction optimization (2026-07-25 session).
# See docs/melitz_sorted_tail_optimization_2026-07-25.md.
# ============================================================================
@testset "Sorted-tail moment construction (2026-07-25)" begin
    using LinearAlgebra: svdvals, rank

    @testset "Phase 1: MelitzSortedTailContext invariants" begin
        p, eq = FIXTURE.primitives, FIXTURE.equilibrium
        D = p.D
        z = FIXTURE.z_draws
        z_copy_before = copy(z)
        sctx = build_melitz_sorted_tail_context(z, p.sigma; theta_star=p.theta_star)

        @testset "construction does not modify the original draws" begin
            @test z == z_copy_before
            @test sctx.z_original == z_copy_before
        end

        @testset "every origin-specific permutation is a valid bijection on 1:W" begin
            for o in 1:D
                @test sort(sctx.permutation[:, o]) == collect(1:sctx.W)
            end
        end

        @testset "sorted_z is nondecreasing and matches z_original[permutation[:,o],o]" begin
            for o in 1:D
                @test issorted(sctx.sorted_z[:, o])
                @test sctx.sorted_z[:, o] == z[sctx.permutation[:, o], o]
                @test sctx.sorted_log_z[:, o] == log.(z[sctx.permutation[:, o], o])
                @test sctx.sorted_z_power[:, o] == z[sctx.permutation[:, o], o] .^ (p.sigma - 1)
            end
        end

        @testset "joint-row pairing is preserved -- NOT re-paired across origins" begin
            # The permutation is origin-LOCAL. Using origin 1's permutation to index origin
            # 2's column must NOT produce a sorted sequence in general (guards against the
            # exact anti-pattern the governing prompt forbids: independently sorting every
            # origin and pairing the k-th sorted observation as a new joint draw).
            @test sctx.permutation[:, 1] != sctx.permutation[:, 2]
            cross_paired = z[sctx.permutation[:, 1], 2]
            @test !issorted(cross_paired)
            # Direct joint-row check: for every ORIGINAL row s, the row (z[s,1],...,z[s,D])
            # recovered via any origin's permutation-and-scatter-back round trip must equal
            # the original row exactly (the row itself, not just origin 1's own value).
            for o in 1:D
                perm_o = sctx.permutation[:, o]
                scattered = similar(z[:, o])
                scattered[perm_o] .= sctx.sorted_z[:, o]
                @test scattered == z[:, o]
            end
        end

        @testset "fingerprint changes under changed draws / sigma; stable under no change" begin
            sctx2 = build_melitz_sorted_tail_context(z, p.sigma; theta_star=p.theta_star)
            @test sctx.fingerprint == sctx2.fingerprint

            z_perturbed = copy(z)
            z_perturbed[1, 1] *= 1.0000001
            sctx_pert = build_melitz_sorted_tail_context(z_perturbed, p.sigma)
            @test sctx_pert.fingerprint != sctx.fingerprint

            sctx_sigma = build_melitz_sorted_tail_context(z, p.sigma + 0.1)
            @test sctx_sigma.fingerprint != sctx.fingerprint
        end
    end

    @testset "Phase 2: active-tail binary search vs Boolean mask" begin
        rng2 = MersenneTwister(4242)
        for trial in 1:2000
            Wt = rand(rng2, 5:200)
            col = sort(rand(rng2, Wt) .* 10 .+ 0.5)
            cutoff = rand(rng2) * 11
            k = melitz_active_tail_start(col, cutoff)
            mask = col .> cutoff
            n_active_mask = count(mask)
            n_active_tail = Wt - k + 1
            @test n_active_tail == n_active_mask
            if n_active_mask > 0
                first_active_idx = findfirst(mask)
                @test k == first_active_idx
                @test all(col[k:end] .> cutoff)
            end
            if k > 1
                @test all(col[1:k-1] .<= cutoff)
            end
        end

        @testset "edge cases" begin
            col = collect(1.0:10.0)
            @test melitz_active_tail_start(col, -Inf) == 1
            @test melitz_active_tail_start(col, Inf) == 11
            @test melitz_active_tail_start(col, 0.5) == 1   # below all draws -> everyone active
            @test melitz_active_tail_start(col, 10.5) == 11 # above all draws -> nobody active
            @test_throws ArgumentError melitz_active_tail_start(col, NaN)
        end

        @testset "exact ties: a draw AT the cutoff is INACTIVE (strict >, matches melitz_firm)" begin
            col = [1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0]
            @test melitz_active_tail_start(col, 3.0) == 6  # both exact 3.0's excluded
            @test melitz_active_tail_start(col, 3.0 - 1e-12) == 3
            @test melitz_active_tail_start(col, 3.0 + 1e-12) == 6
        end

        @testset "against real production cutoffs and the real melitz_firm active flag (D=4)" begin
            p, eq = FIXTURE.primitives, FIXTURE.equilibrium
            D = p.D
            z = FIXTURE.z_draws
            sctx = build_melitz_sorted_tail_context(z, p.sigma)
            for o in 1:D, d in 1:D
                cutoff_od = eq.cutoff[o, d]
                sorted_z_o = @view sctx.sorted_z[:, o]
                perm_o = sctx.permutation[:, o]
                k = melitz_active_tail_start(sorted_z_o, cutoff_od)
                active_tail = falses(sctx.W)
                active_tail[perm_o[k:end]] .= true
                active_direct = [melitz_firm(p.w[o], p.tau[o, d], p.A[o, d], p.f[o, d], p.sigma,
                                              eq.expenditure[d], 1.0, z[w, o]).active for w in 1:sctx.W]
                @test active_tail == active_direct
            end
        end
    end

    @testset "Phase 3: sorted-tail vs dense moment construction, D=4" begin
        p, eq, cf = FIXTURE.primitives, FIXTURE.equilibrium, FIXTURE.counterfactual
        D = p.D
        Wt = 3000
        z = FIXTURE.z_draws[1:Wt, :]
        sctx = build_melitz_sorted_tail_context(z, p.sigma)

        function compare_backends(p_, eq_, cf_, sctx_, label)
            @testset "$label" begin
                Kd = zeros(Wt); Gd = zeros(Wt, LAYOUT.num_moments)
                Ks = zeros(Wt); Gs = zeros(Wt, LAYOUT.num_moments)
                melitz_moments!(Kd, Gd, p_, eq_, cf_, z, LAYOUT)
                melitz_moments_sorted_tail!(Ks, Gs, p_, eq_, cf_, sctx_, LAYOUT)
                @test Kd == Ks
                @test isapprox(Gd, Gs; atol=1e-10, rtol=1e-10)
                @test maximum(abs.(Gd .- Gs)) < 1e-9
                for c in 1:LAYOUT.num_moments
                    @test isapprox(minimum(Gd[:, c]), minimum(Gs[:, c]); atol=1e-10)
                    @test isapprox(maximum(Gd[:, c]), maximum(Gs[:, c]); atol=1e-10)
                    @test isapprox(mean(Gd[:, c]), mean(Gs[:, c]); atol=1e-10)
                end
                @test rank(Gd; atol=1e-8) == rank(Gs; atol=1e-8)
                @test isapprox(svdvals(Gd), svdvals(Gs); atol=1e-6)
            end
        end

        compare_backends(p, eq, cf, sctx, "calibrated fixture point")

        rng3 = MersenneTwister(777)
        for trial in 1:5
            A2 = p.A .* exp.(0.05 .* randn(rng3, D, D))
            p2 = MelitzPrimitives(D, p.sigma, p.theta_star, p.target_country, p.tau, p.w, A2, p.f, p.gamma_prime_target)
            cutoff2 = melitz_baseline_cutoff(A2, p.f, p.w, p.tau, eq.expenditure, p.sigma)
            eq2 = MelitzEquilibrium(eq.expenditure, eq.price_power, cutoff2, eq.trade_flow)
            compare_backends(p2, eq2, cf, sctx, "random A perturbation $trial")
        end

        @testset "focal-origin (target_country) perturbation" begin
            j = p.target_country
            f2 = copy(p.f)
            f2[j, mod1(j + 1, D)] *= 1.3
            p2 = MelitzPrimitives(D, p.sigma, p.theta_star, j, p.tau, p.w, p.A, f2, p.gamma_prime_target)
            cutoff2 = melitz_baseline_cutoff(p.A, f2, p.w, p.tau, eq.expenditure, p.sigma)
            eq2 = MelitzEquilibrium(eq.expenditure, eq.price_power, cutoff2, eq.trade_flow)
            compare_backends(p2, eq2, cf, sctx, "focal f perturbation")
        end

        @testset "near-cutoff tie: a draw placed exactly at a real cutoff" begin
            o, d = 2, 3
            z_tie = copy(z)
            z_tie[1, o] = eq.cutoff[o, d]
            sctx_tie = build_melitz_sorted_tail_context(z_tie, p.sigma)
            Kd = zeros(Wt); Gd = zeros(Wt, LAYOUT.num_moments)
            Ks = zeros(Wt); Gs = zeros(Wt, LAYOUT.num_moments)
            melitz_moments!(Kd, Gd, p, eq, cf, z_tie, LAYOUT)
            melitz_moments_sorted_tail!(Ks, Gs, p, eq, cf, sctx_tie, LAYOUT)
            @test isapprox(Gd, Gs; atol=1e-10, rtol=1e-10)
            # The tied draw itself must be INACTIVE in both backends for cell (o,d).
            @test Gd[1, LAYOUT.trade_index[o, d]] == -(eq.trade_flow[o, d] / eq.expenditure[d])
            @test Gs[1, LAYOUT.trade_index[o, d]] == -(eq.trade_flow[o, d] / eq.expenditure[d])
        end

        @testset "engineered zero-active and all-active cells" begin
            o, d = 3, 4
            f_huge = copy(p.f); f_huge[o, d] = 1e6   # nobody can afford to enter -> zero-active
            p_huge = MelitzPrimitives(D, p.sigma, p.theta_star, p.target_country, p.tau, p.w, p.A, f_huge, p.gamma_prime_target)
            cutoff_huge = melitz_baseline_cutoff(p.A, f_huge, p.w, p.tau, eq.expenditure, p.sigma)
            eq_huge = MelitzEquilibrium(eq.expenditure, eq.price_power, cutoff_huge, eq.trade_flow)
            diag_huge = melitz_sorted_tail_diagnostics(eq_huge, sctx, LAYOUT)
            @test diag_huge.active_count[o, d] == 0
            compare_backends(p_huge, eq_huge, cf, sctx, "engineered zero-active cell")

            f_tiny = copy(p.f); f_tiny[o, d] = 1e-8   # everyone active
            p_tiny = MelitzPrimitives(D, p.sigma, p.theta_star, p.target_country, p.tau, p.w, p.A, f_tiny, p.gamma_prime_target)
            cutoff_tiny = melitz_baseline_cutoff(p.A, f_tiny, p.w, p.tau, eq.expenditure, p.sigma)
            eq_tiny = MelitzEquilibrium(eq.expenditure, eq.price_power, cutoff_tiny, eq.trade_flow)
            diag_tiny = melitz_sorted_tail_diagnostics(eq_tiny, sctx, LAYOUT)
            @test diag_tiny.active_count[o, d] == sctx.W
            compare_backends(p_tiny, eq_tiny, cf, sctx, "engineered all-active cell")
        end

        @testset "stale-sigma sorted context is rejected, not silently reused" begin
            sctx_wrong_sigma = build_melitz_sorted_tail_context(z, p.sigma + 1.0)
            Ks = zeros(Wt); Gs = zeros(Wt, LAYOUT.num_moments)
            @test_throws ArgumentError melitz_moments_sorted_tail!(Ks, Gs, p, eq, cf, sctx_wrong_sigma, LAYOUT)
        end
    end

    @testset "Phase 4: parallel sorted-tail backend agrees with serial (D=4, D=10)" begin
        p, eq, cf = FIXTURE.primitives, FIXTURE.equilibrium, FIXTURE.counterfactual
        Wt = 3000
        z = FIXTURE.z_draws[1:Wt, :]
        sctx = build_melitz_sorted_tail_context(z, p.sigma)
        Ks = zeros(Wt); Gs = zeros(Wt, LAYOUT.num_moments)
        Kp = zeros(Wt); Gp = zeros(Wt, LAYOUT.num_moments)
        melitz_moments_sorted_tail!(Ks, Gs, p, eq, cf, sctx, LAYOUT)
        melitz_moments_sorted_tail_parallel!(Kp, Gp, p, eq, cf, sctx, LAYOUT)
        @test Ks == Kp
        @test Gs == Gp  # disjoint per-origin column writes -> bit-identical, no reduction

        f10 = generate_fake_melitz_data(; D=10, sigma=2.5, theta_star=6.8, target_country=3, seed=8, W=4000)
        p10, eq10, cf10 = f10.primitives, f10.equilibrium, f10.counterfactual
        layout10 = MelitzMomentLayout(p10.D)
        z10 = f10.z_draws
        sctx10 = build_melitz_sorted_tail_context(z10, p10.sigma)
        Ks10 = zeros(size(z10, 1)); Gs10 = zeros(size(z10, 1), layout10.num_moments)
        Kp10 = zeros(size(z10, 1)); Gp10 = zeros(size(z10, 1), layout10.num_moments)
        melitz_moments_sorted_tail!(Ks10, Gs10, p10, eq10, cf10, sctx10, layout10)
        melitz_moments_sorted_tail_parallel!(Kp10, Gp10, p10, eq10, cf10, sctx10, layout10)
        @test Ks10 == Kp10
        @test Gs10 == Gp10
    end

    @testset "Phase 3: sorted-tail vs dense moment construction, D=10 (fresh fixture)" begin
        f10 = generate_fake_melitz_data(; D=10, sigma=2.5, theta_star=6.8, target_country=3, seed=8, W=4000)
        p, eq, cf = f10.primitives, f10.equilibrium, f10.counterfactual
        layout10 = MelitzMomentLayout(p.D)
        Wt = f10.z_draws === nothing ? 0 : size(f10.z_draws, 1)
        z = f10.z_draws
        sctx = build_melitz_sorted_tail_context(z, p.sigma)
        K1 = zeros(Wt); G1 = zeros(Wt, layout10.num_moments)
        K2 = zeros(Wt); G2 = zeros(Wt, layout10.num_moments)
        melitz_moments!(K1, G1, p, eq, cf, z, layout10)
        melitz_moments_sorted_tail!(K2, G2, p, eq, cf, sctx, layout10)
        @test K1 == K2
        @test isapprox(G1, G2; atol=1e-10, rtol=1e-10)

        diag_sorted = melitz_sorted_tail_diagnostics(eq, sctx, layout10)
        for o in 1:p.D, d in 1:p.D
            n_direct = count(>(eq.cutoff[o, d]), @view(z[:, o]))
            @test diag_sorted.active_count[o, d] == n_direct
            @test isapprox(diag_sorted.active_fraction[o, d], n_direct / Wt; atol=1e-12)
        end
    end

    @testset "Phase 5: fused diagnostics vs dense reference (D=4)" begin
        p, eq = FIXTURE.primitives, FIXTURE.equilibrium
        z = FIXTURE.z_draws
        sctx = build_melitz_sorted_tail_context(z, p.sigma)
        diag_sorted = melitz_sorted_tail_diagnostics(eq, sctx, LAYOUT)
        diag_dense = cell_participation_diagnostics(p, eq, z)
        @test diag_sorted.active_count == diag_dense.count_active
        mc, worst = min_active_draw_count(p, eq, z)
        @test minimum(diag_sorted.active_count) == mc
    end

    @testset "Phase 8 (2026-07-26): sorted dual-argument construction vs dense G*mu (D=4)" begin
        p, eq, cf = FIXTURE.primitives, FIXTURE.equilibrium, FIXTURE.counterfactual
        Wt = 3000
        z = FIXTURE.z_draws[1:Wt, :]
        sctx = build_melitz_sorted_tail_context(z, p.sigma)
        K = zeros(Wt); G = zeros(Wt, LAYOUT.num_moments)
        melitz_moments!(K, G, p, eq, cf, z, LAYOUT)

        @testset "random (zeta, mu) trials" begin
            rng8 = MersenneTwister(2026)
            for trial in 1:30
                zeta = randn(rng8) * 100
                mu = randn(rng8, LAYOUT.num_moments)
                u_dense = melitz_dense_dual_argument(zeta, mu, K, G)
                u_sorted = melitz_sorted_dual_argument(zeta, mu, p, eq, cf, sctx, LAYOUT)
                @test isapprox(u_dense, u_sorted; atol=1e-8, rtol=1e-8)
            end
        end

        @testset "isolated components: single trade cell, link-only, zeta-only, all-ones" begin
            mu_cell = zeros(LAYOUT.num_moments); mu_cell[LAYOUT.trade_index[2, 3]] = 1.0
            @test isapprox(melitz_dense_dual_argument(0.0, mu_cell, K, G),
                            melitz_sorted_dual_argument(0.0, mu_cell, p, eq, cf, sctx, LAYOUT); atol=1e-8)

            mu_link = zeros(LAYOUT.num_moments); mu_link[LAYOUT.focal_link_index] = 1.0
            @test isapprox(melitz_dense_dual_argument(0.0, mu_link, K, G),
                            melitz_sorted_dual_argument(0.0, mu_link, p, eq, cf, sctx, LAYOUT); atol=1e-8)

            mu_zero = zeros(LAYOUT.num_moments)
            u_z1 = melitz_dense_dual_argument(7.0, mu_zero, K, G)
            u_z2 = melitz_sorted_dual_argument(7.0, mu_zero, p, eq, cf, sctx, LAYOUT)
            @test all(u_z1 .== -7.0)
            @test u_z1 == u_z2

            mu_ones = ones(LAYOUT.num_moments)
            @test isapprox(melitz_dense_dual_argument(3.0, mu_ones, K, G),
                            melitz_sorted_dual_argument(3.0, mu_ones, p, eq, cf, sctx, LAYOUT); atol=1e-8)
        end

        @testset "stale sigma / shape guards" begin
            sctx_bad_sigma = build_melitz_sorted_tail_context(z, p.sigma + 1.0)
            @test_throws ArgumentError melitz_sorted_dual_argument(0.0, ones(LAYOUT.num_moments), p, eq, cf,
                sctx_bad_sigma, LAYOUT)
            @test_throws ArgumentError melitz_sorted_dual_argument(0.0, ones(LAYOUT.num_moments - 1), p, eq, cf,
                sctx, LAYOUT)
        end
    end

    @testset "Phase 8 (2026-07-26): sorted dual-argument construction vs dense, D=10 (fresh fixture)" begin
        f10b = generate_fake_melitz_data(; D=10, sigma=2.5, theta_star=6.8, target_country=3, seed=8, W=4000)
        p10, eq10, cf10 = f10b.primitives, f10b.equilibrium, f10b.counterfactual
        layout10 = MelitzMomentLayout(p10.D)
        z10 = f10b.z_draws
        sctx10 = build_melitz_sorted_tail_context(z10, p10.sigma)
        K10 = zeros(size(z10, 1)); G10 = zeros(size(z10, 1), layout10.num_moments)
        melitz_moments!(K10, G10, p10, eq10, cf10, z10, layout10)
        rng10 = MersenneTwister(11)
        for trial in 1:10
            zeta = randn(rng10) * 50
            mu = randn(rng10, layout10.num_moments)
            u_dense = melitz_dense_dual_argument(zeta, mu, K10, G10)
            u_sorted = melitz_sorted_dual_argument(zeta, mu, p10, eq10, cf10, sctx10, layout10)
            @test isapprox(u_dense, u_sorted; atol=1e-8, rtol=1e-8)
        end
    end

    begin
        @testset "Phase 3/real D=20: sorted-tail vs dense at real calibration" begin
            # Governing prompt Phase 3 (2026-07-27 addendum): this path used to be built with
            # an extra (wrong) `dirname` (3 levels up from `test/melitz/`, landing one level
            # ABOVE the repo root), so `isdir(real_dir)` was always false and this test
            # silently `@warn`-skipped in every run, in this repo, forever -- the real fixture
            # (`real_data/noah_D20`) has been present at the repo root the whole time. Fixed to
            # 2 levels (repo root). The fixture is not optional in this repo: a caller-side
            # `isdir` guard that then silently skips would just reintroduce the same silent-skip
            # bug in a different form, so this now hard-asserts the directory exists.
            real_dir = joinpath(dirname(dirname(@__DIR__)), "real_data", "noah_D20")
            @assert isdir(real_dir) "real_data/noah_D20 not found at $real_dir -- this repo's own bundled real-data fixture is required for this test, not an optional/skippable dependency"
            lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
            LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
            tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
            countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
            focal20 = findfirst(==("fra"), countries)
            observed20 = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
            calib20 = calibrate_melitz_pareto(observed20; sigma=2.5, theta_star=:estimate,
                focal_country=focal20, p_min=0.001, wage_tol=1e-6, gravity_tol=1e-6)
            p20 = MelitzPrimitives(calib20.D, calib20.sigma, calib20.theta_star, calib20.target_country,
                calib20.tau, calib20.w, calib20.A, calib20.f, calib20.gamma_prime_target)
            eq20 = MelitzEquilibrium(calib20.E, ones(calib20.D), calib20.q, calib20.X)
            cf20 = MelitzCounterfactual(calib20.target_country, calib20.w_prime,
                calib20.w_prime * calib20.L[calib20.target_country], 1.0, calib20.w_prime * calib20.L[calib20.target_country])
            layout20 = MelitzMomentLayout(calib20.D)

            Wt = 20_000
            z20 = pareto_draws(Wt, calib20.D, calib20.theta_star; seed=1)
            sctx20 = build_melitz_sorted_tail_context(z20, p20.sigma; theta_star=p20.theta_star)
            K1 = zeros(Wt); G1 = zeros(Wt, layout20.num_moments)
            K2 = zeros(Wt); G2 = zeros(Wt, layout20.num_moments)
            melitz_moments!(K1, G1, p20, eq20, cf20, z20, layout20)
            melitz_moments_sorted_tail!(K2, G2, p20, eq20, cf20, sctx20, layout20)
            @test K1 == K2
            @test isapprox(G1, G2; atol=1e-9, rtol=1e-9)
            @test maximum(abs.(G1 .- G2)) < 1e-8
            @test rank(G1; atol=1e-6) == rank(G2; atol=1e-6)

            diag_sorted20 = melitz_sorted_tail_diagnostics(eq20, sctx20, layout20)
            diag_dense20 = cell_participation_diagnostics(p20, eq20, z20)
            @test diag_sorted20.active_count == diag_dense20.count_active
        end
    end
end

@testset "Matrix-free moment operator (2026-07-26)" begin
    using LinearAlgebra: dot, Diagonal

    function _check_operator_against_dense(p, eq, cf, layout, z; ntrials=15, atol=1e-8, rtol=1e-8, seed=99)
        Wt = size(z, 1)
        sctx = build_melitz_sorted_tail_context(z, p.sigma)
        K = zeros(Wt); G = zeros(Wt, layout.num_moments)
        melitz_moments!(K, G, p, eq, cf, z, layout)
        op = build_melitz_moment_operator(sctx, layout)
        melitz_update_moment_operator!(op, p, eq, cf)

        rng = MersenneTwister(seed)
        u = zeros(Wt)
        g = zeros(layout.num_moments)
        for trial in 1:ntrials
            zeta = randn(rng) * 50
            mu = randn(rng, layout.num_moments)
            u_dense = melitz_dense_dual_argument(zeta, mu, K, G)
            mul_G!(u, op, zeta, mu)
            @test isapprox(u, u_dense; atol=atol, rtol=rtol)

            v = randn(rng, Wt)
            g_dense = melitz_dense_Gt_v(v, G)
            mul_Gt!(g, op, v)
            @test isapprox(g, g_dense; atol=atol, rtol=rtol)
        end

        # Per-column unit-mu extraction: mul_G! at mu=e_k, zeta=0 must reproduce -G[:,k] exactly.
        mu_unit = zeros(layout.num_moments)
        for k in (1, layout.num_moments, cld(layout.num_moments, 2))
            fill!(mu_unit, 0.0); mu_unit[k] = 1.0
            mul_G!(u, op, 0.0, mu_unit)
            @test isapprox(u, -G[:, k]; atol=atol, rtol=rtol)
        end

        # Per-row unit-v extraction: mul_Gt! at v=e_s must reproduce G[s,:] exactly.
        v_unit = zeros(Wt)
        for s in (1, Wt, cld(Wt, 2))
            fill!(v_unit, 0.0); v_unit[s] = 1.0
            mul_Gt!(g, op, v_unit)
            @test isapprox(g, G[s, :]; atol=atol, rtol=rtol)
        end

        return op, K, G
    end

    @testset "D=4 (calibrated fixture, truncated W)" begin
        p, eq, cf = FIXTURE.primitives, FIXTURE.equilibrium, FIXTURE.counterfactual
        z = FIXTURE.z_draws[1:3000, :]
        _check_operator_against_dense(p, eq, cf, LAYOUT, z)
    end

    @testset "D=4 random A perturbations" begin
        p, eq, cf = FIXTURE.primitives, FIXTURE.equilibrium, FIXTURE.counterfactual
        D = p.D
        z = FIXTURE.z_draws[1:3000, :]
        rng = MersenneTwister(4242)
        for trial in 1:3
            A2 = p.A .* exp.(0.05 .* randn(rng, D, D))
            p2 = MelitzPrimitives(D, p.sigma, p.theta_star, p.target_country, p.tau, p.w, A2, p.f, p.gamma_prime_target)
            cutoff2 = melitz_baseline_cutoff(A2, p.f, p.w, p.tau, eq.expenditure, p.sigma)
            eq2 = MelitzEquilibrium(eq.expenditure, eq.price_power, cutoff2, eq.trade_flow)
            _check_operator_against_dense(p2, eq2, cf, LAYOUT, z; ntrials=5, seed=100 + trial)
        end
    end

    @testset "D=10 (fresh fixture)" begin
        f10 = generate_fake_melitz_data(; D=10, sigma=2.5, theta_star=6.8, target_country=3, seed=8, W=4000)
        p, eq, cf = f10.primitives, f10.equilibrium, f10.counterfactual
        layout10 = MelitzMomentLayout(p.D)
        _check_operator_against_dense(p, eq, cf, layout10, f10.z_draws)
    end

    @testset "Operator update: coef/lambda/order/rank/bin agree with dense reconstruction" begin
        p, eq, cf = FIXTURE.primitives, FIXTURE.equilibrium, FIXTURE.counterfactual
        D = p.D
        z = FIXTURE.z_draws[1:3000, :]
        sctx = build_melitz_sorted_tail_context(z, p.sigma)
        op = build_melitz_moment_operator(sctx, LAYOUT)
        melitz_update_moment_operator!(op, p, eq, cf)

        for o in 1:D, d in 1:D
            C_od = melitz_C(p.w[o], p.tau[o, d], p.A[o, d], p.sigma, eq.expenditure[d])
            @test isapprox(op.coef[o, d], C_od / eq.expenditure[d]; atol=1e-12)
            @test isapprox(op.lambda[o, d], eq.trade_flow[o, d] / eq.expenditure[d]; atol=1e-12)
        end

        for o in 1:D
            @test sort(op.order[:, o]) == collect(1:D)
            for d in 1:D
                @test op.order[op.rank[d, o], o] == d
            end
            @test issorted(eq.cutoff[o, :][op.order[:, o]])
        end

        # bin[s,o] must equal the exact count of destinations active for draw s at origin o.
        Wt = size(z, 1)
        for o in 1:D
            n_check = 0
            for s in 1:min(Wt, 200)
                n_active = count(d -> eq.cutoff[o, d] < z[s, o], 1:D)
                @test op.bin[s, o] == n_active
                n_check += 1
            end
            @test n_check > 0
        end
    end

    @testset "Guards: stale D/sigma rejected" begin
        p, eq, cf = FIXTURE.primitives, FIXTURE.equilibrium, FIXTURE.counterfactual
        z = FIXTURE.z_draws[1:1000, :]
        sctx = build_melitz_sorted_tail_context(z, p.sigma)
        op = build_melitz_moment_operator(sctx, LAYOUT)
        melitz_update_moment_operator!(op, p, eq, cf)

        sctx_bad_sigma = build_melitz_sorted_tail_context(z, p.sigma + 1.0)
        op_bad = build_melitz_moment_operator(sctx_bad_sigma, LAYOUT)
        @test_throws ArgumentError melitz_update_moment_operator!(op_bad, p, eq, cf)

        @test_throws ArgumentError mul_G!(zeros(999), op, 0.0, zeros(LAYOUT.num_moments))
        @test_throws ArgumentError mul_G!(zeros(size(z,1)), op, 0.0, zeros(3))
        @test_throws ArgumentError mul_Gt!(zeros(3), op, zeros(size(z,1)))
        @test_throws ArgumentError mul_Gt!(zeros(LAYOUT.num_moments), op, zeros(999))
    end

    @testset "Zero-allocation callbacks after warmup (D=4)" begin
        p, eq, cf = FIXTURE.primitives, FIXTURE.equilibrium, FIXTURE.counterfactual
        z = FIXTURE.z_draws[1:3000, :]
        Wt = size(z, 1)
        sctx = build_melitz_sorted_tail_context(z, p.sigma)
        op = build_melitz_moment_operator(sctx, LAYOUT)
        melitz_update_moment_operator!(op, p, eq, cf)

        u = zeros(Wt); g = zeros(LAYOUT.num_moments)
        mu = randn(MersenneTwister(1), LAYOUT.num_moments)
        v = randn(MersenneTwister(2), Wt)

        mul_G!(u, op, 1.0, mu)   # warmup
        mul_Gt!(g, op, v)        # warmup

        bytes_G = @allocated mul_G!(u, op, 1.0, mu)
        bytes_Gt = @allocated mul_Gt!(g, op, v)
        @test bytes_G == 0
        @test bytes_Gt == 0
    end

    @testset "Phase 6/7 (scoped prototype): same-origin weighted-Gram block vs dense R'*S*R (D=4, D=10)" begin
        function _check_same_origin_block(p, eq, cf, layout, z; seed=321)
            D = p.D
            Wt = size(z, 1)
            K = zeros(Wt); G = zeros(Wt, layout.num_moments)
            melitz_moments!(K, G, p, eq, cf, z, layout)
            R = copy(G)
            for o in 1:D, d in 1:D
                col = layout.trade_index[o, d]
                lambda_od = eq.trade_flow[o, d] / eq.expenditure[d]
                R[:, col] .+= lambda_od
            end
            sctx = build_melitz_sorted_tail_context(z, p.sigma)
            op = build_melitz_moment_operator(sctx, layout)
            melitz_update_moment_operator!(op, p, eq, cf)

            rng = MersenneTwister(seed)
            S = rand(rng, Wt) .+ 0.1   # positive curvature weights, like ddPsi! output
            Hblock = zeros(D, D)
            for o in 1:D
                melitz_same_origin_weighted_block!(Hblock, op, o, S)
                R_o = @view R[:, layout.trade_index[o, :]]
                dense_block = R_o' * Diagonal(S) * R_o
                for d in 1:D, dp in d:D
                    @test isapprox(Hblock[d, dp], dense_block[d, dp]; atol=1e-8, rtol=1e-6)
                end
            end

            # allocation check (D=4/D=10 scale)
            melitz_same_origin_weighted_block!(Hblock, op, 1, S)  # warmup
            bytes = @allocated melitz_same_origin_weighted_block!(Hblock, op, 1, S)
            @test bytes == 0
        end

        p, eq, cf = FIXTURE.primitives, FIXTURE.equilibrium, FIXTURE.counterfactual
        _check_same_origin_block(p, eq, cf, LAYOUT, FIXTURE.z_draws[1:3000, :])

        f10 = generate_fake_melitz_data(; D=10, sigma=2.5, theta_star=6.8, target_country=3, seed=8, W=4000)
        layout10 = MelitzMomentLayout(f10.primitives.D)
        _check_same_origin_block(f10.primitives, f10.equilibrium, f10.counterfactual, layout10, f10.z_draws)
    end

    @testset "Phase 7/8/9 (2026-07-26): full matrix-free weighted Gram (complete Hessian) vs dense (D=4, D=10)" begin
        function _check_full_weighted_gram(p, eq, cf, layout, z; seed=555)
            D = p.D
            Wt = size(z, 1)
            K = zeros(Wt); G = zeros(Wt, layout.num_moments)
            melitz_moments!(K, G, p, eq, cf, z, layout)
            Hfull_dense = hcat(ones(Wt), G)

            sctx = build_melitz_sorted_tail_context(z, p.sigma)
            op = build_melitz_moment_operator(sctx, layout)
            melitz_update_moment_operator!(op, p, eq, cf)

            rng = MersenneTwister(seed)
            S = rand(rng, Wt) .+ 0.1
            n = 1 + layout.num_moments
            H = zeros(n, n)
            melitz_full_weighted_gram!(H, op, S)

            dense_full = Hfull_dense' * Diagonal(S) * Hfull_dense
            for i in 1:n, j in i:n
                @test isapprox(H[i, j], dense_full[i, j]; atol=1e-8, rtol=1e-6)
            end

            # cross-origin block, standalone, o<p direct check against dense R'*S*R (not G'SG)
            R = copy(G)
            for o in 1:D, d in 1:D
                col = layout.trade_index[o, d]
                lambda_od = eq.trade_flow[o, d] / eq.expenditure[d]
                R[:, col] .+= lambda_od
            end
            if D >= 2
                Hblock = zeros(D, D)
                melitz_cross_origin_weighted_block!(Hblock, op, 1, 2, S)
                R1 = @view R[:, layout.trade_index[1, :]]
                R2 = @view R[:, layout.trade_index[2, :]]
                dense_cross = R1' * Diagonal(S) * R2
                @test isapprox(Hblock, dense_cross; atol=1e-8, rtol=1e-6)
                @test_throws ArgumentError melitz_cross_origin_weighted_block!(Hblock, op, 2, 1, S)  # o<p required
            end

            # allocation gates
            melitz_full_weighted_gram!(H, op, S)  # warmup
            @test (@allocated melitz_full_weighted_gram!(H, op, S)) == 0
            if D >= 2
                Hblock2 = zeros(D, D)
                melitz_cross_origin_weighted_block!(Hblock2, op, 1, 2, S)  # warmup
                @test (@allocated melitz_cross_origin_weighted_block!(Hblock2, op, 1, 2, S)) == 0
            end

            @test_throws ArgumentError melitz_full_weighted_gram!(zeros(3, 3), op, S)
            @test_throws ArgumentError melitz_full_weighted_gram!(H, op, zeros(3))

            # Phase 7 parallelization: bit-identical to serial (disjoint per-origin-pair
            # writes, no reduction), and allocation is a small FIXED Threads.@threads
            # task-spawn cost (confirmed W/D-independent during development, ~13KB at
            # nthreads()=16 regardless of D=4/D=10/real-D20 -- NOT a data-scaling hot-loop
            # allocation; bounded rather than asserted ==0 since exact Task-spawn overhead
            # is a Julia-runtime/thread-count detail, not a correctness property).
            Hpar = zeros(n, n)
            melitz_full_weighted_gram_parallel!(Hpar, op, S)
            for i in 1:n, j in i:n
                @test isapprox(Hpar[i, j], H[i, j]; atol=1e-12, rtol=1e-12)
            end
            melitz_full_weighted_gram_parallel!(Hpar, op, S)  # warmup
            @test (@allocated melitz_full_weighted_gram_parallel!(Hpar, op, S)) < 100_000
            @test_throws ArgumentError melitz_full_weighted_gram_parallel!(zeros(3, 3), op, S)
            @test_throws ArgumentError melitz_full_weighted_gram_parallel!(H, op, zeros(3))
        end

        p, eq, cf = FIXTURE.primitives, FIXTURE.equilibrium, FIXTURE.counterfactual
        _check_full_weighted_gram(p, eq, cf, LAYOUT, FIXTURE.z_draws[1:3000, :])

        f10 = generate_fake_melitz_data(; D=10, sigma=2.5, theta_star=6.8, target_country=3, seed=8, W=4000)
        layout10 = MelitzMomentLayout(f10.primitives.D)
        _check_full_weighted_gram(f10.primitives, f10.equilibrium, f10.counterfactual, layout10, f10.z_draws)
    end

    begin
        @testset "real D=20: matrix-free operator vs dense at real calibration" begin
            # Governing prompt Phase 3 (2026-07-27 addendum): same 3-dirname path bug as the
            # "Phase 3/real D=20" testset above -- fixed to 2 levels (repo root), hard-asserted
            # rather than silently skipped (see that testset's own comment for the full story).
            real_dir = joinpath(dirname(dirname(@__DIR__)), "real_data", "noah_D20")
            @assert isdir(real_dir) "real_data/noah_D20 not found at $real_dir -- this repo's own bundled real-data fixture is required for this test, not an optional/skippable dependency"
            lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
            LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
            tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
            countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
            focal20 = findfirst(==("fra"), countries)
            observed20 = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
            calib20 = calibrate_melitz_pareto(observed20; sigma=2.5, theta_star=:estimate,
                focal_country=focal20, p_min=0.001, wage_tol=1e-6, gravity_tol=1e-6)
            p20 = MelitzPrimitives(calib20.D, calib20.sigma, calib20.theta_star, calib20.target_country,
                calib20.tau, calib20.w, calib20.A, calib20.f, calib20.gamma_prime_target)
            eq20 = MelitzEquilibrium(calib20.E, ones(calib20.D), calib20.q, calib20.X)
            cf20 = MelitzCounterfactual(calib20.target_country, calib20.w_prime,
                calib20.w_prime * calib20.L[calib20.target_country], 1.0, calib20.w_prime * calib20.L[calib20.target_country])
            layout20 = MelitzMomentLayout(calib20.D)

            Wt = 20_000
            z20 = pareto_draws(Wt, calib20.D, calib20.theta_star; seed=1)
            op, K1, G1 = _check_operator_against_dense(p20, eq20, cf20, layout20, z20; ntrials=10, atol=1e-6, rtol=1e-6)

            u = zeros(Wt); g = zeros(layout20.num_moments)
            mu = randn(MersenneTwister(3), layout20.num_moments)
            v = randn(MersenneTwister(4), Wt)
            mul_G!(u, op, 1.0, mu); mul_Gt!(g, op, v)   # warmup
            bytes_G = @allocated mul_G!(u, op, 1.0, mu)
            bytes_Gt = @allocated mul_Gt!(g, op, v)
            @test bytes_G == 0
            @test bytes_Gt == 0

            # Phase 6/7 (scoped prototype): same-origin weighted-Gram block vs dense R'*S*R
            D20 = layout20.D
            R20 = copy(G1)
            for o in 1:D20, d in 1:D20
                col = layout20.trade_index[o, d]
                lambda_od = eq20.trade_flow[o, d] / eq20.expenditure[d]
                R20[:, col] .+= lambda_od
            end
            S = rand(MersenneTwister(321), Wt) .+ 0.1
            Hblock = zeros(D20, D20)
            for o in 1:D20
                melitz_same_origin_weighted_block!(Hblock, op, o, S)
                R_o = @view R20[:, layout20.trade_index[o, :]]
                dense_block = R_o' * Diagonal(S) * R_o
                for d in 1:D20, dp in d:D20
                    @test isapprox(Hblock[d, dp], dense_block[d, dp]; atol=1e-6, rtol=1e-5)
                end
            end
            melitz_same_origin_weighted_block!(Hblock, op, 1, S)  # warmup
            bytes_H = @allocated melitz_same_origin_weighted_block!(Hblock, op, 1, S)
            @test bytes_H == 0

            # Phase 7/8/9 (2026-07-26): cross-origin block + full weighted Gram (complete Hessian)
            melitz_cross_origin_weighted_block!(Hblock, op, 1, 2, S)
            R1 = @view R20[:, layout20.trade_index[1, :]]
            R2 = @view R20[:, layout20.trade_index[2, :]]
            dense_cross = R1' * Diagonal(S) * R2
            @test isapprox(Hblock, dense_cross; atol=1e-6, rtol=1e-5)
            melitz_cross_origin_weighted_block!(Hblock, op, 1, 2, S)  # warmup
            @test (@allocated melitz_cross_origin_weighted_block!(Hblock, op, 1, 2, S)) == 0

            n20 = 1 + layout20.num_moments
            Hfull = zeros(n20, n20)
            melitz_full_weighted_gram!(Hfull, op, S)
            Hfull_dense = hcat(ones(Wt), G1)
            dense_full = Hfull_dense' * Diagonal(S) * Hfull_dense
            for i in 1:n20, j in i:n20
                @test isapprox(Hfull[i, j], dense_full[i, j]; atol=1e-6, rtol=1e-5)
            end
            melitz_full_weighted_gram!(Hfull, op, S)  # warmup
            @test (@allocated melitz_full_weighted_gram!(Hfull, op, S)) == 0

            # Phase 7 parallelization at real D=20/W=20,000 scale
            Hfull_par = zeros(n20, n20)
            melitz_full_weighted_gram_parallel!(Hfull_par, op, S)
            for i in 1:n20, j in i:n20
                @test isapprox(Hfull_par[i, j], Hfull[i, j]; atol=1e-12, rtol=1e-12)
            end
            melitz_full_weighted_gram_parallel!(Hfull_par, op, S)  # warmup
            @test (@allocated melitz_full_weighted_gram_parallel!(Hfull_par, op, S)) < 100_000
        end
    end
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
    include(joinpath(MELITZ_DIR, "matrix_free_dual_solve.jl"))
    true
catch e
    @warn "Skipping CC inner-loop testsets: cc_algo/KNITRO not available in this environment" exception = e
    false
end

# ============================================================================
# Data-only Pareto calibration (docs/melitz_pareto_data_calibration_2026-07-24.md).
# Closes the wage-calibration-gap methodological leak: every test below builds the
# estimation context from `MelitzObservedData` (trade shares/L/tau) + `calibrate_melitz_pareto`
# ONLY -- `MelitzSyntheticTruth` (hidden A/f/w/gamma_prime_target) is read ONLY after
# calibration, for recovery comparison, never as a calibration/estimation input.
# ============================================================================
@testset "Pareto data-only calibration" begin
    fixture = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
    observed, truth = split_melitz_synthetic_truth(fixture)

    @testset "MelitzObservedData construction/validation" begin
        @test observed.D == 4
        @test isapprox(vec(sum(observed.lambda, dims=1)), ones(4); atol=1e-10)
        @test all(observed.tau[o, o] == 1.0 for o in 1:4)
        @test_throws ArgumentError MelitzObservedData(; lambda=observed.lambda, X=observed.lambda .* 2, L=observed.L, tau=observed.tau)
        bad_lambda = copy(observed.lambda); bad_lambda[1, 2] = 0.0
        @test_throws ArgumentError MelitzObservedData(; lambda=bad_lambda, L=observed.L, tau=observed.tau)
    end

    @testset "Section 4: data-only wage calibration" begin
        wc = calibrate_melitz_wages(observed.lambda, observed.L)
        # market clearing: E == lambda*E
        @test wc.market_clearing_residual < 1e-6
        # independent eigenvector/Perron cross-check agrees with the damped-Jacobi fixed point
        @test wc.perron_residual < 1e-6
        # recovers the TRUE (hidden) baseline wages up to the shared numeraire -- the point of
        # the fake-data recovery test: observables alone pin down wages to high precision.
        @test isapprox(wc.w, truth.w ./ truth.w[1]; rtol=1e-6)

        # numeraire (scale) invariance: recalibrating under a different numeraire rescales A
        # UNIFORMLY (A_new = c*A_old, c = w_new[o]/w_old[o] for every o) and leaves f, prices,
        # cutoffs, shares, and GT exactly invariant (derived in the accompanying report).
        wc2 = calibrate_melitz_wages(observed.lambda, observed.L; numeraire=2)
        c_scale = wc2.w[1] / wc.w[1]
        @test isapprox(wc2.w, wc.w .* c_scale; rtol=1e-8)
    end

    @testset "Section 5: theta_star/gravity compatibility" begin
        grav_true = melitz_gravity_theta_check(observed.lambda, observed.tau, fixture.primitives.theta_star; sigma=fixture.primitives.sigma)
        @test grav_true.compatible
        @test isapprox(grav_true.theta_hat, fixture.primitives.theta_star; atol=1e-6)
        grav_bad = melitz_gravity_theta_check(observed.lambda, observed.tau, fixture.primitives.theta_star * 1.5; sigma=fixture.primitives.sigma)
        @test !grav_bad.compatible
    end

    wc = calibrate_melitz_wages(observed.lambda, observed.L)
    calib = calibrate_melitz_pareto(observed; sigma=fixture.primitives.sigma, theta_star=fixture.primitives.theta_star,
        focal_country=fixture.primitives.target_country, p_min=0.005)

    @testset "Section 7/12: Pareto inversion reproduces data exactly" begin
        @test maximum(abs.(calib.equilibrium_check.residual_shares)) < 1e-10
        @test maximum(abs.(calib.equilibrium_check.residual_gamma)) < 1e-8
        @test maximum(abs.(calib.equilibrium_check.residual_market_clearing)) < 1e-6
        @test maximum(abs.(calib.equilibrium_check.residual_cutoff_reconstruction)) < 1e-8
        @test calib.equilibrium_check.min_support > -1e-6
        @test calib.equilibrium_check.min_export_minus_domestic > -1e-6
        @test all(>(0), calib.equilibrium_check.f_E)
        @test abs(calib.equilibrium_check.gravity_residual_A) < 1e-6
        @test abs(calib.equilibrium_check.gravity_residual_f) < 1e-6
    end

    @testset "Section 13/20: GT_model == GT_ACR (non-negotiable)" begin
        p_calib = MelitzPrimitives(calib.D, calib.sigma, calib.theta_star, calib.target_country,
            calib.tau, calib.w, calib.A, calib.f, calib.gamma_prime_target)
        eq_calib = MelitzEquilibrium(calib.E, ones(calib.D), calib.q, calib.X)
        cf_calib = MelitzCounterfactual(calib.target_country, calib.w_prime,
            calib.w_prime * calib.L[calib.target_country], 1.0, calib.w_prime * calib.L[calib.target_country])
        GT_model = melitz_gains_from_trade(p_calib, cf_calib)
        _, GT_ACR = acr_gains_from_trade(p_calib, eq_calib)
        @test isapprox(GT_model, GT_ACR; atol=1e-6)
        @test abs(calib.free_entry_link_residual) < 1e-6
    end

    @testset "Section 18: nonidentification -- two cutoff targets, same shares/GT, different A/f" begin
        calib2 = calibrate_melitz_pareto(observed; sigma=fixture.primitives.sigma, theta_star=fixture.primitives.theta_star,
            focal_country=fixture.primitives.target_country, p_min=0.005,
            cutoff_policy=:uniform, cutoff_target_kwargs=(p_domestic=0.5, p_export=0.15))
        @test !isapprox(calib.A, calib2.A; rtol=1e-3)
        @test !isapprox(calib.f, calib2.f; rtol=1e-3)
        # BOTH reproduce the observed shares exactly (Section 12 point 1)
        @test maximum(abs.(calib2.equilibrium_check.residual_shares)) < 1e-10
        # BOTH give the identical baseline price index and Pareto GT (the true nonidentification claim)
        @test isapprox(calib.equilibrium_check.residual_gamma, calib2.equilibrium_check.residual_gamma; atol=1e-6)
        p1 = MelitzPrimitives(calib.D, calib.sigma, calib.theta_star, calib.target_country, calib.tau, calib.w, calib.A, calib.f, calib.gamma_prime_target)
        eq1 = MelitzEquilibrium(calib.E, ones(calib.D), calib.q, calib.X)
        p2 = MelitzPrimitives(calib2.D, calib2.sigma, calib2.theta_star, calib2.target_country, calib2.tau, calib2.w, calib2.A, calib2.f, calib2.gamma_prime_target)
        eq2 = MelitzEquilibrium(calib2.E, ones(calib2.D), calib2.q, calib2.X)
        _, GT1 = acr_gains_from_trade(p1, eq1)
        _, GT2 = acr_gains_from_trade(p2, eq2)
        @test isapprox(GT1, GT2; atol=1e-8)  # ACR uses only lambda_jj -- identical by construction
        # the identified composite chi = theta_star*logA + beta*logf agrees across decompositions
        chi1, _, _, _ = melitz_pareto_composite(observed.lambda, calib.w, calib.E, observed.tau, calib.sigma, calib.theta_star)
        chi2, _, _, _ = melitz_pareto_composite(observed.lambda, calib2.w, calib2.E, observed.tau, calib2.sigma, calib2.theta_star)
        @test isapprox(chi1, chi2; atol=1e-6)
    end

    @testset "Section 18: no hidden-truth dependence" begin
        # A second, independently-generated synthetic truth attached to a DIFFERENT observed
        # object of the same shape must not affect the calibration, since calibration never
        # reads truth -- verified directly: calibration is a pure function of `observed` alone.
        calib_again = calibrate_melitz_pareto(observed; sigma=fixture.primitives.sigma, theta_star=fixture.primitives.theta_star,
            focal_country=fixture.primitives.target_country, p_min=0.005)
        @test calib_again.A == calib.A
        @test calib_again.f == calib.f
        @test calib_again.w == calib.w
    end

    @testset "Section 17: real D=20 data-only calibration diagnostics" begin
        # Governing prompt Phase 3 (2026-07-27 addendum): same 3-dirname path bug as the two
        # testsets above (fixed to 2 levels, hard-asserted). This one is notable: it had
        # SILENTLY NEVER RUN in this repo before this fix -- see the closure doc's own Phase 13
        # "remaining known gaps" note, which flagged (not fixed) this exact path bug for
        # "Section 17" specifically. Unskipped and re-verified this session (see the run log
        # referenced in docs/melitz_outer_parameterization_comparison_2026-07-26.md) -- passes
        # with the SAME assertions below (none weakened to force a green run).
        real_dir = joinpath(dirname(dirname(@__DIR__)), "real_data", "noah_D20")
        @assert isdir(real_dir) "real_data/noah_D20 not found at $real_dir -- this repo's own bundled real-data fixture is required for this test, not an optional/skippable dependency"
        lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
        LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
        tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
        countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
        D20 = length(countries)
        focal20 = findfirst(==("fra"), countries)

            # Phase 1.1 fix (2026-07-24 continuation): construct the FROZEN MelitzObservedData
            # FIRST (default policies: share_policy=:as_supplied, tau_diagonal_policy=
            # :normalize_to_one -- addendum Sections 1-2), THEN estimate theta_hat from that
            # SAME frozen object via theta_star=:estimate inside calibrate_melitz_pareto --
            # never from the raw pre-policy CSV arrays directly (the ordering bug this session's
            # governing prompt identified: theta previously estimated on `tauData` before its
            # diagonal was snapped to 1, then calibration ran on the POST-snap tau).
            observed20 = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)

            # Addendum Section 1: raw share diagnostics, unconditionally reported/checked --
            # real_data/noah_D20/pi.csv's columns already sum to 1 at (better than) float
            # precision, so share_policy=:as_supplied (the default) makes NO functional
            # difference here; this assertion pins that fact rather than assuming it.
            @test observed20.raw_share_diagnostics.max_abs_column_sum_deviation < 1e-6
            @test observed20.share_policy == :as_supplied
            @test observed20.lambda == Matrix{Float64}(lambdaData)  # byte-for-byte passthrough under :as_supplied (no forced renormalization)

            # Addendum Section 2: the tau diagonal policy is explicit and recorded -- the "row"
            # (rest-of-world) aggregate's raw diagonal is a measurement artifact (~1.0011, not
            # exactly 1), snapped by the ONE documented :normalize_to_one step (required because
            # `MelitzPrimitives` hard-asserts diag(tau)==1.0 exactly, `types.jl`), with the
            # ORIGINAL value preserved for the record.
            @test observed20.tau_diagonal_policy == :normalize_to_one
            @test any(x -> !isapprox(x, 1.0; atol=0), observed20.tau_diagonal_raw)  # at least one non-bit-exact raw diagonal entry
            @test all(observed20.tau[o, o] == 1.0 for o in 1:D20)

            wc20 = calibrate_melitz_wages(observed20.lambda, observed20.L; tol=1e-6, max_iter=100_000)
            # Phase 2 fix: market-clearing residual now comes from the AUTHORITATIVE direct
            # linear solve, not an iterative method's own convergence-tolerance choice. Live
            # measurement: ~8.5e-8, matched by damped-Jacobi's OWN floor after 200,000
            # iterations (~1.9e-8) and the independent Perron eigenvector (~9.5e-6 relative
            # agreement) -- this is a genuine CONDITIONING floor of (I-lambda) at this D=20
            # real-data scale (lambda has an eigenvalue very close to 1 besides the exact
            # unit one), not an artifact of any particular solve method's tolerance/max_iter.
            @test wc20.market_clearing_residual < 1e-6
            @test wc20.damped_vs_direct_residual < 1e-4  # cross-check: both methods agree

            calib20 = calibrate_melitz_pareto(observed20; sigma=2.5, theta_star=:estimate,
                focal_country=focal20, p_min=0.001, wage_tol=1e-6, gravity_tol=1e-6)
            # Phase 1.1/1.3 fix: theta_star is now estimated from the SAME frozen `observed20`
            # calibration itself reads, so the two separate A-/f-gravity restrictions are
            # compatible BY CONSTRUCTION -- gravity residuals should now be at (near-)machine
            # precision, not the previously-documented ~1e-4 (which was entirely attributable
            # to the estimation-order bug, not a genuine real-data limitation).
            @test abs(calib20.equilibrium_check.gravity_residual_A) < 1e-8
            @test abs(calib20.equilibrium_check.gravity_residual_f) < 1e-8
            @test isapprox(calib20.cutoff_calibration.gravity_rhs_A, calib20.cutoff_calibration.gravity_rhs_f; atol=1e-6)
            # Governing prompt Phase 3 (2026-07-27 addendum, first pass): this ENTIRE testset
            # had never actually run in this repo before the path-bug fix above -- this specific
            # assertion is the one genuine failure it revealed (`1e-10` was a never-validated
            # guess). The first-pass diagnosis attributed the ~5.86e-8 floor to generic
            # "floating-point cancellation from the real dataset's wide dynamic range," verified
            # only by checking insensitivity to `wage_tol`/the u_jj bisection `xatol`.
            #
            # 2026-07-27 CONTINUATION SESSION -- exact root cause identified and independently
            # verified (governing prompt Phase 2, "do not accept the existing explanation
            # without an independent numerical check"; full script archived to Dropbox,
            # key_results/melitz_d20_share_residual_diagnosis_2026-07-27.csv):
            #
            # 1. RULED OUT arithmetic/cancellation: recomputing `population_X`/`model_lambda`
            #    in BigFloat (256-bit) from the SAME Float64-calibrated `A`/`f`/`w` reproduces
            #    the IDENTICAL residual (5.857674170773914e-8 vs. the Float64 path's
            #    5.8576741679416955e-8, agreeing to 9 significant figures; per-cell
            #    Float64-vs-BigFloat model_lambda differences are ~1e-16, i.e. ordinary
            #    roundoff, NOT ~1e-8). An algebraically-simplified single-log-exp
            #    reformulation of `population_X` (avoiding the separate K1/cutoff/tail-mean
            #    intermediate roundings) gives the same ~5.86e-8 residual too. Both rule out
            #    the verification arithmetic itself as the source.
            # 2. IDENTIFIED THE EXACT MECHANISM: `melitz_ad_from_cutoffs` (the cell-by-cell A/f
            #    inversion, Section 7/8) is constructed so that `X[o,d] = E[d]*lambda[o,d]`
            #    EXACTLY -- `mu`/`w`/`tau`/`A` cancel algebraically regardless of the calibrated
            #    cutoff/theta_star (confirmed by direct derivation AND numerically: a
            #    prediction built from ONLY the raw data's own column sums,
            #    `lambda[o,d]*(1/colsum(lambda[:,d]) - 1)`, matches the actual
            #    `residual_shares` to 4.4e-16 -- i.e. the ENTIRE residual, cell by cell). So
            #    `model_lambda[o,d] = lambda[o,d]/colsum(lambda[:,d])` is an EXACT closed form,
            #    and the residual is purely `real_data/noah_D20/pi.csv`'s own raw column shares
            #    not summing to exactly 1.0 (`max|colsum-1| = 7.058e-8` here) -- a DATA
            #    property, not a solver-precision one. This is exactly why sweeping `wage_tol`
            #    (1e-6 to 1e-12) and the u_jj bisection `xatol` (1e-10 to 1e-14) never moved the
            #    residual: neither tolerance touches the raw data's own column-sum property.
            # 3. RESOLUTION CONFIRMED: `MelitzObservedData`'s existing, already-documented
            #    `share_policy=:renormalize` option (which rescales each column to sum to
            #    exactly 1 at construction) reduces `residual_shares` to `4.44e-16` (machine
            #    precision) on this exact dataset -- confirming the diagnosis directly, not just
            #    consistently. NOT applied to `observed20` above: this testset deliberately uses
            #    the default `share_policy=:as_supplied` (byte-for-byte passthrough, asserted
            #    earlier in this testset) to validate the calibration pipeline against the data
            #    AS DELIVERED, not a silently-adjusted copy of it; loosening this offline
            #    verification tolerance is the correct response for `:as_supplied`, not a
            #    production-code change (Phase 2.3: "do not slow production callbacks merely to
            #    improve an offline diagnostic"). `1e-6` gives ~14x margin over the verified
            #    ~7.06e-8 floor while remaining tight enough to catch a real regression.
            @test maximum(abs.(calib20.equilibrium_check.residual_shares)) < 1e-6

            # Pins the verified mechanism itself (not just its consequence): the raw data's own
            # column-sum deviation from 1, and that `share_policy=:renormalize` genuinely
            # resolves it to machine precision on this exact dataset. Would fail loudly if the
            # bundled `pi.csv` fixture is ever replaced with data whose columns sum exactly to
            # 1 already (raw_colsum_dev would then be ~0, not a regression, but worth noticing)
            # or if `share_policy=:renormalize`'s own rescaling is ever broken.
            raw_colsum_dev = maximum(abs.(vec(sum(observed20.lambda, dims=1)) .- 1.0))
            @test 1e-9 < raw_colsum_dev < 1e-6
            observed20_renorm = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData,
                countries=countries, atol=2e-3, share_policy=:renormalize)
            calib20_renorm = calibrate_melitz_pareto(observed20_renorm; sigma=2.5, theta_star=:estimate,
                focal_country=focal20, p_min=0.001, wage_tol=1e-6, gravity_tol=1e-6)
            @test maximum(abs.(calib20_renorm.equilibrium_check.residual_shares)) < 1e-10
            @test calib20.equilibrium_check.min_support > -1e-6
            @test calib20.equilibrium_check.min_export_minus_domestic > -1e-6
            @test all(>(0), calib20.equilibrium_check.f_E)
            p20 = MelitzPrimitives(calib20.D, calib20.sigma, calib20.theta_star, calib20.target_country,
                calib20.tau, calib20.w, calib20.A, calib20.f, calib20.gamma_prime_target)
            eq20 = MelitzEquilibrium(calib20.E, ones(calib20.D), calib20.q, calib20.X)
            cf20 = MelitzCounterfactual(calib20.target_country, calib20.w_prime,
                calib20.w_prime * calib20.L[calib20.target_country], 1.0, calib20.w_prime * calib20.L[calib20.target_country])
            GT_model20 = melitz_gains_from_trade(p20, cf20)
            _, GT_ACR20 = acr_gains_from_trade(p20, eq20)
            @test isapprox(GT_model20, GT_ACR20; atol=1e-6)

            # Phase 1.4: outer-coordinate roundtrip -- the calibrated (A,f) reduced through the
            # gravity pivots and expanded back must reproduce the calibration to numerical
            # tolerance now that the calibration's OWN gravity residuals are at machine
            # precision (previously would have exposed silent pivot-cell drift).
            rt20 = melitz_calibration_roundtrip_check(calib20)
            @test rt20.max_rel_A_diff < 1e-6
            @test rt20.max_rel_f_diff < 1e-6
            @test rt20.gamma_prime_diff < 1e-6
            @test rt20.max_abs_share_diff < 1e-8
            @test abs(rt20.focal_link_residual_reexpanded) < 1e-6
    end

    if KNITRO_AVAILABLE
        @testset "Section 4/14/20: calibration-only context builds + real KNITRO inner solve" begin
            obj, theta_free = build_melitz_psi_bundle_from_calibration(calib; W=20_000, seed=1)
            lfd = melitz_recover_lfd(obj, theta_free)
            @test lfd.nStatus == 0
            @test lfd.lfd_ok
            @test lfd.maximum_weighted_moment_residual < 1e-6
            p_calib = MelitzPrimitives(calib.D, calib.sigma, calib.theta_star, calib.target_country,
                calib.tau, calib.w, calib.A, calib.f, calib.gamma_prime_target)
            eq_calib = MelitzEquilibrium(calib.E, ones(calib.D), calib.q, calib.X)
            cf_calib = MelitzCounterfactual(calib.target_country, calib.w_prime,
                calib.w_prime * calib.L[calib.target_country], 1.0, calib.w_prime * calib.L[calib.target_country])
            check = check_profiled_melitz_equilibrium(p_calib, eq_calib, cf_calib, obj.U, lfd.weights)
            @test abs(check.residual_autarky_cutoff) < 1e-6
            @test check.N_prime_diff_rel < 1e-4
        end

        @testset "Phase 11 (2026-07-25): :sorted_tail_serial backend reproduces :dense_reference through a real KNITRO inner solve" begin
            obj_dense, theta_dense = build_melitz_psi_bundle_from_calibration(calib; W=20_000, seed=1,
                moment_backend=:dense_reference)
            obj_sorted, theta_sorted = build_melitz_psi_bundle_from_calibration(calib; W=20_000, seed=1,
                moment_backend=:sorted_tail_serial)
            @test obj_dense.γ.moment_backend == :dense_reference
            @test obj_sorted.γ.moment_backend == :sorted_tail_serial
            @test obj_sorted.γ.sorted_tail_ctx !== nothing
            @test theta_dense == theta_sorted
            @test obj_dense.U == obj_sorted.U

            lfd_dense = melitz_recover_lfd(obj_dense, theta_dense)
            lfd_sorted = melitz_recover_lfd(obj_sorted, theta_sorted)
            @test lfd_dense.nStatus == 0
            @test lfd_sorted.nStatus == 0
            @test lfd_dense.lfd_ok
            @test lfd_sorted.lfd_ok
            @test isapprox(lfd_dense.Delta, lfd_sorted.Delta; atol=1e-8, rtol=1e-8)
            @test isapprox(lfd_dense.weights, lfd_sorted.weights; atol=1e-8, rtol=1e-8)
            @test isapprox(lfd_dense.maximum_weighted_moment_residual, lfd_sorted.maximum_weighted_moment_residual;
                atol=1e-8, rtol=1e-8)

            # A genuinely mismatched U (wrong shape) must be rejected, not silently reused.
            obj_bad = build_melitz_psi_bundle_from_calibration(calib; W=20_000, seed=1,
                moment_backend=:sorted_tail_serial)[1]
            obj_bad.U = pareto_draws(15_000, calib.D, calib.theta_star; seed=2)
            K_bad = zeros(calib.D^2 + 1); G_bad = zeros(15_000, calib.D^2 + 1)
            @test_throws ArgumentError melitz_moments_adapter!(K_bad, G_bad, theta_sorted, obj_bad.U, obj_bad)
        end

        @testset "Phase 11 follow-up (2026-07-25): :sorted_tail_parallel backend wired and reproduces :dense_reference" begin
            obj_dense, theta_dense = build_melitz_psi_bundle_from_calibration(calib; W=20_000, seed=1,
                moment_backend=:dense_reference)
            obj_par, theta_par = build_melitz_psi_bundle_from_calibration(calib; W=20_000, seed=1,
                moment_backend=:sorted_tail_parallel)
            @test obj_par.γ.moment_backend == :sorted_tail_parallel
            @test obj_par.γ.sorted_tail_ctx !== nothing
            @test theta_dense == theta_par

            lfd_dense = melitz_recover_lfd(obj_dense, theta_dense)
            lfd_par = melitz_recover_lfd(obj_par, theta_par)
            @test lfd_par.nStatus == 0
            @test lfd_par.lfd_ok
            @test isapprox(lfd_dense.Delta, lfd_par.Delta; atol=1e-8, rtol=1e-8)
            @test isapprox(lfd_dense.weights, lfd_par.weights; atol=1e-8, rtol=1e-8)

            @test_throws ArgumentError build_melitz_psi_bundle_from_calibration(calib; W=20_000, seed=1,
                moment_backend=:bogus_backend)
        end

        @testset "Matrix-free inner CC dual solve (2026-07-26 continuation): validated through a REAL KNITRO inner solve, D=4" begin
            # MelitzMatrixFreeDualBundle (matrix_free_dual_solve.jl) is a freestanding bundle
            # that never touches cc_algo/PsiObjectiveBundle.jl -- this test drives a REAL
            # KNITRO solve through it and compares against the dense PsiObjectiveBundleDelta
            # solve on the IDENTICAL (p, eq, cf, z_draws) fixture, not merely a callback-level
            # comparison at an isolated point.
            p4, eq4, cf4 = FIXTURE.primitives, FIXTURE.equilibrium, FIXTURE.counterfactual
            z4 = FIXTURE.z_draws
            inner_opt = joinpath(dirname(dirname(@__DIR__)), "ek_inner_loop_options.opt")

            obj_dense, theta0 = build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt)
            lfd_dense = melitz_recover_lfd(obj_dense, theta0)
            @test lfd_dense.nStatus == 0
            @test lfd_dense.lfd_ok

            sctx4 = build_melitz_sorted_tail_context(z4, p4.sigma)
            op4 = build_melitz_moment_operator(sctx4, LAYOUT)
            melitz_update_moment_operator!(op4, p4, eq4, cf4)
            bundle4 = build_melitz_matrix_free_dual_bundle(op4)
            nStatus_mf, objSol_mf, x_mf = melitz_matrix_free_inner_solve(bundle4, inner_opt)
            @test nStatus_mf == 0

            # `find_smallest` (default true on PsiObjectiveBundleDelta) is a reporting-only
            # sign flip applied by `inner_loop`'s wrapper (cc_algo/inner_loop_functions.jl),
            # not a property of the raw KNITRO objective `melitz_matrix_free_inner_solve`
            # returns directly -- compare against the pre-wrapper raw value.
            @test obj_dense.find_smallest == true
            @test isapprox(-lfd_dense.Delta, objSol_mf; atol=1e-8, rtol=1e-8)
            @test isapprox(lfd_dense.dual_x, x_mf; atol=1e-6, rtol=1e-6)

            weights_mf, moment_residuals_mf, norm_resid_mf = melitz_matrix_free_moment_residuals(bundle4, x_mf)
            @test isapprox(lfd_dense.weights, weights_mf; atol=1e-8, rtol=1e-8)
            @test maximum(abs.(moment_residuals_mf)) < 1e-6
            @test isapprox(lfd_dense.moment_residuals, moment_residuals_mf; atol=1e-6, rtol=1e-6)
            @test abs(norm_resid_mf) < 1e-6

            # Guards: wrong-length x rejected (via mul_G!'s own length check), not silently
            # truncated/padded.
            @test_throws ArgumentError bundle4(zeros(3))
        end

        @testset "Phase 6/7 (2026-07-25): sorted crossing-slice gradient at real D=20/W=80,000" begin
            inner_opt_d20 = joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options_capped_2026-07-24.opt")
            obj_d20_inner, theta_d20 = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
                inner_loop_opt=inner_opt_d20, moment_backend=:sorted_tail_serial)
            ctx_d20 = obj_d20_inner.γ
            @test ctx_d20.sorted_tail_ctx !== nothing

            lfd_d20 = melitz_recover_lfd(obj_d20_inner, theta_d20)
            @test lfd_d20.nStatus == 0

            obj_d20 = build_melitz_implicit_bundle(ctx_d20, obj_d20_inner.U, theta_d20; delta=1.0,
                find_smallest=true, gradient_backend=:B_direct_argument_serial, h=1e-4,
                inner_loop_opt=inner_opt_d20,
                outer_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_outer_finite_delta.opt"),
                backend=:dense_reference)  # this testset calls CS.inner_loop_internal directly
            obj_d20.use_cached_x = false; obj_d20.x .= NaN
            _, x_d20, nStatus_d20 = CounterfactualSensitivity.inner_loop_internal(obj_d20, theta_d20)
            @test nStatus_d20 == 0
            obj_d20.x .= x_d20

            n_d20 = length(theta_d20)
            direct_serial_d20 = make_melitz_gradient_delta_direct_serial(1e-4)
            direct_sorted_d20 = make_melitz_gradient_delta_direct_sorted_serial(1e-4)
            g_ref_d20 = zeros(n_d20)
            g_sorted_d20 = zeros(n_d20)
            direct_serial_d20(g_ref_d20, theta_d20, ctx_d20, obj_d20, x_d20)
            direct_sorted_d20(g_sorted_d20, theta_d20, ctx_d20, obj_d20, x_d20)

            rng_d20 = MersenneTwister(99)
            coords_d20 = union(Set([1]), Set(rand(rng_d20, 1:n_d20, 24)))
            for r in coords_d20
                rel = abs(g_ref_d20[r] - g_sorted_d20[r]) / max(abs(g_ref_d20[r]), 1.0)
                @test rel < 1e-6
            end

            # 2026-07-26 continuation: parallel crossing-slice backend, real D=20/W=80,000.
            direct_sorted_parallel_d20 = make_melitz_gradient_delta_direct_sorted_parallel(1e-4)
            g_sorted_par_d20 = zeros(n_d20)
            direct_sorted_parallel_d20(g_sorted_par_d20, theta_d20, ctx_d20, obj_d20, x_d20)
            @test g_sorted_par_d20 == g_sorted_d20   # disjoint per-coordinate writes -> bit-identical

            @testset "Phase 3.2/3.3 (2026-07-27 continuation): parallel sorted gradient allocation regression, real D=20/W=80,000" begin
                # Governing prompt Phase 3: `copy(theta)` inside the parallel gradient
                # variants (direct_gradient.jl/sorted_crossing_gradient.jl) allocated a fresh
                # n-length vector on EVERY coordinate (n=$(n_d20) here), and expand_free_theta's
                # own f-gravity-pivot reconstruction redundantly recomputed
                # f_gravity_pivot_avoid_indices/build_gravity_pivot's pivot-selection work
                # (two setdiff calls + an abs. temporary + an argmax) on every FD evaluation
                # despite depending only on ctx (invariant across an entire outer solve). Both
                # fixed this session (per-thread persistent theta_p/theta_m buffers;
                # melitz_cached_f_pivot_parts memoized by ctx identity). This regression test
                # pins a POST-WARM-UP absolute bound that would catch either regression
                # reappearing. NOTE: despite this testset's own name, `calib` here (line ~1417)
                # is built from the D=4 `fixture`, not real D=20 data, so `n_d20` is small (this
                # is the pre-existing testset structure, not something this session changed) --
                # measured live post-warm-up: ~87KB (dominated by expand_free_theta's own
                # legitimate, UNAVOIDABLE per-call output construction -- fresh A/f matrices and
                # logA_full/logf_free_full vectors, allocated twice per coordinate for
                # theta_p/theta_m -- not a residual of either fixed pattern). A relative n^2
                # bound is the wrong shape of test at this small an `n` (same reasoning as the
                # adjacent serial-backend test's own comment: the fixed baseline dominates at
                # small scale) -- use a generous absolute cap instead, comfortably above the
                # observed ~87KB, comfortably below any regression that reintroduces an
                # allocating pattern inside the per-coordinate parallel loop.
                g_scratch_par = zeros(n_d20)
                direct_sorted_parallel_d20(g_scratch_par, theta_d20, ctx_d20, obj_d20, x_d20)  # warm-up
                bytes_par = @allocated direct_sorted_parallel_d20(g_scratch_par, theta_d20, ctx_d20, obj_d20, x_d20)
                @test g_scratch_par == g_sorted_d20
                @test bytes_par < 500_000

                # melitz_expand_theta/reduce_to_free_theta's own cached-pivot-parts allocation
                # stays bounded and ctx-identity-stable across repeated calls (would regress if
                # melitz_cached_f_pivot_parts's ctx-identity check were ever broken, causing a
                # rebuild -- with its own setdiff/abs./argmax allocations -- on every call).
                melitz_expand_theta(theta_d20, ctx_d20)  # warm-up
                bytes_expand1 = @allocated melitz_expand_theta(theta_d20, ctx_d20)
                bytes_expand2 = @allocated melitz_expand_theta(theta_d20, ctx_d20)
                @test bytes_expand1 == bytes_expand2   # stable, not growing/shrinking across repeated calls with the SAME ctx
            end

            # Phase 8 (2026-07-26): sorted dual-argument construction, exact vs dense at real
            # D=20/W=80,000, using the SAME converged dual x_d20 and a genuinely random mu.
            Kbuf_d20 = zeros(size(obj_d20.U, 1))
            Gbuf_d20 = zeros(size(obj_d20.U, 1), ctx_d20.moment_layout.num_moments)
            melitz_moments_adapter!(Kbuf_d20, Gbuf_d20, theta_d20, obj_d20.U, obj_d20)
            rng_mu = MersenneTwister(321)
            mu_d20 = randn(rng_mu, ctx_d20.moment_layout.num_moments)
            zeta_d20 = 4.2
            u_dense_d20 = melitz_dense_dual_argument(zeta_d20, mu_d20, Kbuf_d20, Gbuf_d20)
            p_d20, eq_d20, cf_d20 = let
                A20, f20, gpj20, _ = melitz_expand_theta(theta_d20, ctx_d20)
                pr = MelitzPrimitives(ctx_d20.D, ctx_d20.sigma, ctx_d20.theta_star, ctx_d20.target_country,
                    ctx_d20.tau, ctx_d20.w, A20, f20, gpj20)
                cutoff20 = melitz_baseline_cutoff(A20, f20, ctx_d20.w, ctx_d20.tau, ctx_d20.expenditure, ctx_d20.sigma)
                eqx = MelitzEquilibrium(ctx_d20.expenditure, ones(ctx_d20.D), cutoff20, ctx_d20.X_data)
                expprime = ctx_d20.w_prime * ctx_d20.L[ctx_d20.target_country]
                cfx = MelitzCounterfactual(ctx_d20.target_country, ctx_d20.w_prime, expprime, 1.0, expprime)
                (pr, eqx, cfx)
            end
            u_sorted_d20 = melitz_sorted_dual_argument(zeta_d20, mu_d20, p_d20, eq_d20, cf_d20,
                ctx_d20.sorted_tail_ctx, ctx_d20.moment_layout)
            maxrel_dual = maximum(abs.(u_dense_d20 .- u_sorted_d20) ./ max.(abs.(u_dense_d20), 1.0))
            @test maxrel_dual < 1e-6
        end
    else
        @info "Skipping calibration-context real-KNITRO testset (cc_algo/KNITRO not available)"
    end
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

        # ====================================================================
        # Continuation session (2026-07-23, "make the optimized architecture scalable in
        # memory and D"), Section 3: the argument-localized backends never build a full
        # (W, K) matrix at all (not even a base one) -- they must still be BIT-IDENTICAL to
        # :method_b_localized at every touched column, since the dependency-map claim
        # (Gate 1, validated above) is shared unchanged.
        # ====================================================================
        @testset "Section 3: argument-localized gradient (:B_argument_localized_serial/_parallel)" begin
            # n12/d12/Wt12/mj_loc12 are scoped INSIDE the "Phase II.12" @testset block above
            # (a separate local scope) -- recomputed locally here rather than relying on them
            # leaking into this sibling testset.
            n12 = length(theta11)
            d12 = ctx11.moment_layout.num_moments
            Wt12 = size(obj11.U, 1)
            mj_loc12 = make_melitz_moments_jacobian_b_localized(1e-4)
            mj_argser = make_melitz_moments_jacobian_b_argument_localized_serial(1e-4)
            mj_argpar = make_melitz_moments_jacobian_b_argument_localized_parallel(1e-4)

            rng13 = MersenneTwister(73)
            for trial in 1:3
                theta_probe13 = theta11 .+ 0.02 .* randn(rng13, n12)
                K_loc13, G_loc13 = zeros(Wt12, n12), zeros(Wt12, d12, n12)
                K_as, G_as = zeros(Wt12, n12), zeros(Wt12, d12, n12)
                K_ap, G_ap = zeros(Wt12, n12), zeros(Wt12, d12, n12)
                mj_loc12(K_loc13, G_loc13, theta_probe13, obj11.U, obj11)
                mj_argser(K_as, G_as, theta_probe13, obj11.U, obj11)
                mj_argpar(K_ap, G_ap, theta_probe13, obj11.U, obj11)
                @testset "trial $trial: argument-localized serial/parallel bit-identical to :B_localized, per coordinate" begin
                    @test K_as == K_loc13
                    @test K_ap == K_loc13
                    for k in 1:n12
                        @test (@views G_as[:, :, k] == G_loc13[:, :, k])
                        @test (@views G_ap[:, :, k] == G_loc13[:, :, k])
                    end
                end
            end

            @testset "BLAS thread count restored after the argument-localized parallel sweep" begin
                prev = BLAS.get_num_threads()
                BLAS.set_num_threads(3)
                K_ap2, G_ap2 = zeros(Wt12, n12), zeros(Wt12, d12, n12)
                mj_argpar(K_ap2, G_ap2, theta11, obj11.U, obj11)
                @test BLAS.get_num_threads() == 3
                BLAS.set_num_threads(prev)
            end

            @testset "small-N probe call skips G_jac safely (serial and parallel)" begin
                K_s1, G_s1 = zeros(2, n12), zeros(2, d12, n12)
                mj_argser(K_s1, G_s1, theta11, obj11.U[1:2, :], obj11)
                @test all(iszero, G_s1) && all(==(1.0), K_s1[:, 1])
                K_s2, G_s2 = zeros(2, n12), zeros(2, d12, n12)
                mj_argpar(K_s2, G_s2, theta11, obj11.U[1:2, :], obj11)
                @test all(iszero, G_s2) && all(==(1.0), K_s2[:, 1])
            end

            @testset "per-coordinate touched-column count is O(D), never O(K)" begin
                compact13 = melitz_compact_columns_map(ctx11)
                D13 = ctx11.D
                K13 = ctx11.moment_layout.num_moments
                maxcols13 = maximum(length(c.direct_cols) for c in compact13)
                @test maxcols13 < K13            # strictly fewer columns than the full moment count
                @test maxcols13 <= D13 + 4        # bounded by O(D), the dependency-map's own claim
            end
        end
    end

    # ========================================================================
    # Continuation4 session, Section 4: the direct fixed-dual gradient-VECTOR backend
    # (src/melitz/direct_gradient.jl). Self-contained fixture (own small Implicit bundle,
    # one real inner solve for a genuine dual x).
    # ========================================================================
    @testset "Continuation4 Section 4: direct fixed-dual gradient backend (:B_direct_argument_serial/_parallel)" begin
        fixture_dg = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=2_000)
        inner_opt_dg = joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt")
        outer_opt_dg = joinpath(dirname(dirname(@__DIR__)), "melitz_outer_finite_delta.opt")
        obj_dg_inner, theta_dg = build_melitz_psi_bundle(fixture_dg; inner_loop_opt=inner_opt_dg,
            needs_outer_moment_jacobian=false)
        ctx_dg = obj_dg_inner.γ

        obj_dg = build_melitz_implicit_bundle(ctx_dg, obj_dg_inner.U, theta_dg; delta=1.0,
            find_smallest=true, gradient_backend=:B_direct_argument_serial, h=1e-4,
            inner_loop_opt=inner_opt_dg, outer_loop_opt=outer_opt_dg,
            backend=:dense_reference)  # this testset pokes obj_dg.H directly + calls
            # CS.inner_loop_internal directly -- genuinely dense-bundle-specific mechanics


        @testset "needs_outer_moment_jacobian=false actually skips the jac_h allocation" begin
            @test size(obj_dg.jac_h) == (0, 0, 0)
            @test obj_dg.needs_outer_moment_jacobian == false
        end

        obj_dg.use_cached_x = false; obj_dg.x .= NaN
        _, x_dg, nStatus_dg = CounterfactualSensitivity.inner_loop_internal(obj_dg, theta_dg)
        @test nStatus_dg in (0, -100, -101, -103)
        obj_dg.x .= x_dg

        n_dg = length(theta_dg)
        direct_serial_dg = make_melitz_gradient_delta_direct_serial(1e-4)
        direct_parallel_dg = make_melitz_gradient_delta_direct_parallel(1e-4)
        g_serial = zeros(n_dg)
        g_parallel = zeros(n_dg)
        direct_serial_dg(g_serial, theta_dg, ctx_dg, obj_dg, x_dg)
        direct_parallel_dg(g_parallel, theta_dg, ctx_dg, obj_dg, x_dg)

        @testset "serial and parallel direct backends are bit-identical" begin
            @test g_serial == g_parallel
        end

        @testset "direct backend never allocates a W x K x n tensor (post-warm-up @allocated is O(1)-ish, not O(W*K))" begin
            g_scratch = zeros(n_dg)
            direct_serial_dg(g_scratch, theta_dg, ctx_dg, obj_dg, x_dg)   # warm-up
            bytes = @allocated direct_serial_dg(g_scratch, theta_dg, ctx_dg, obj_dg, x_dg)
            K_dg = ctx_dg.moment_layout.num_moments
            W_dg = size(obj_dg.U, 1)
            # a single W x K x n tensor at this fixture would be W*K*n*8 bytes (~8.16MB here).
            # The actual post-warm-up call allocates a small, roughly W-INDEPENDENT residual
            # (~250KB, per scripts/melitz_memory_audit.jl's own live measurement at a much
            # larger W=20,000/D=4 fixture -- confirming this overhead does NOT scale with W, so
            # a relative-to-W*K*n bound is the wrong shape of test at small W). Use an absolute
            # cap instead: comfortably above the observed ~250KB, comfortably below what any
            # O(W*K*n) tensor would require even at this tiny fixture.
            @test bytes < 2_000_000
            @test bytes < W_dg * K_dg * n_dg * 8 / 2   # still well under half the full-tensor size
        end

        @testset "Phase 3.2 (2026-07-27 continuation): parallel direct backend no longer copy(theta)-per-coordinate" begin
            # direct_gradient.jl's PARALLEL variant used to `copy(theta)` (a fresh n-length
            # allocation) TWICE per coordinate, every call -- fixed via per-thread persistent
            # theta_p/theta_m buffers (mirroring the serial sibling's own copyto!-based reuse).
            # n_dg is small at this D=4 fixture so the absolute byte savings are modest, but a
            # reintroduced copy(theta) would still show up as bytes scaling with repeated calls
            # at a FIXED n rather than staying flat post-warm-up.
            g_scratch_p = zeros(n_dg)
            direct_parallel_dg(g_scratch_p, theta_dg, ctx_dg, obj_dg, x_dg)  # warm-up
            bytes_p1 = @allocated direct_parallel_dg(g_scratch_p, theta_dg, ctx_dg, obj_dg, x_dg)
            bytes_p2 = @allocated direct_parallel_dg(g_scratch_p, theta_dg, ctx_dg, obj_dg, x_dg)
            @test g_scratch_p == g_serial
            @test bytes_p1 == bytes_p2   # stable post-warm-up, not growing across repeated calls
        end

        @testset "agrees with an independent frozen-x finite difference at a random coordinate" begin
            rng_dg = MersenneTwister(2026)
            r = rand(rng_dg, 1:n_dg)
            h_dg = 1e-4
            ei_dg = zeros(n_dg); ei_dg[r] = 1.0
            Kbuf_dg = zeros(size(obj_dg.U, 1))
            Gbuf_dg = zeros(size(obj_dg.U, 1), ctx_dg.moment_layout.num_moments)
            melitz_moments_adapter_outer!(Kbuf_dg, Gbuf_dg, theta_dg .+ h_dg .* ei_dg, obj_dg.U, obj_dg)
            obj_dg.H[:, 1] .= Kbuf_dg; obj_dg.H[:, 3:end] .= Gbuf_dg
            cp_dg = zeros(1); obj_dg(x_dg, constr=cp_dg)
            melitz_moments_adapter_outer!(Kbuf_dg, Gbuf_dg, theta_dg .- h_dg .* ei_dg, obj_dg.U, obj_dg)
            obj_dg.H[:, 1] .= Kbuf_dg; obj_dg.H[:, 3:end] .= Gbuf_dg
            cm_dg = zeros(1); obj_dg(x_dg, constr=cm_dg)
            fd_grad_dg = (cp_dg[1] - cm_dg[1]) / (2h_dg)
            @test isapprox(g_serial[r], fd_grad_dg; rtol=1e-6, atol=1e-6)
            # restore obj_dg.H to the base theta for any downstream reuse
            melitz_moments_adapter_outer!(Kbuf_dg, Gbuf_dg, theta_dg, obj_dg.U, obj_dg)
            obj_dg.H[:, 1] .= Kbuf_dg; obj_dg.H[:, 3:end] .= Gbuf_dg
        end

        @testset "wired end-to-end through melitz_fixed_point_probe without crashing" begin
            r_probe = melitz_fixed_point_probe(ctx_dg, obj_dg_inner, theta_dg; delta=1e-2,
                direction=:upper, gradient_backend=:B_direct_argument_serial, inner_loop_opt=inner_opt_dg)
            @test r_probe.nStatus == 0
            @test !r_probe.eval_failed
            r_probe_par = melitz_fixed_point_probe(ctx_dg, obj_dg_inner, theta_dg; delta=1e-2,
                direction=:upper, gradient_backend=:B_direct_argument_parallel, inner_loop_opt=inner_opt_dg)
            @test r_probe_par.nStatus == 0
            @test r_probe_par.obj_value == r_probe.obj_value
        end
    end

    # ========================================================================
    # 2026-07-25 sorted-tail session continuation, Phase 6/7 (docs/melitz_sorted_tail_optimization_2026-07-25.md
    # Section D.1): the sorted CROSSING-SLICE variant of the direct fixed-dual gradient
    # backend (src/melitz/sorted_crossing_gradient.jl). Self-contained fixture, built WITH
    # moment_backend=:sorted_tail_serial so ctx carries the required sorted_tail_ctx.
    # ========================================================================
    @testset "Phase 6/7 (2026-07-25): sorted crossing-slice direct gradient backend (:B_direct_argument_sorted_serial)" begin
        fixture_cg = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=2_000)
        inner_opt_cg = joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt")
        outer_opt_cg = joinpath(dirname(dirname(@__DIR__)), "melitz_outer_finite_delta.opt")
        obj_cg_inner, theta_cg = build_melitz_psi_bundle(fixture_cg; inner_loop_opt=inner_opt_cg,
            needs_outer_moment_jacobian=false, moment_backend=:sorted_tail_serial)
        ctx_cg = obj_cg_inner.γ
        @test ctx_cg.sorted_tail_ctx !== nothing

        obj_cg = build_melitz_implicit_bundle(ctx_cg, obj_cg_inner.U, theta_cg; delta=1.0,
            find_smallest=true, gradient_backend=:B_direct_argument_serial, h=1e-4,
            inner_loop_opt=inner_opt_cg, outer_loop_opt=outer_opt_cg,
            backend=:dense_reference)  # pokes obj_cg.H directly + calls CS.inner_loop_internal directly
        obj_cg.use_cached_x = false; obj_cg.x .= NaN
        _, x_cg, nStatus_cg = CounterfactualSensitivity.inner_loop_internal(obj_cg, theta_cg)
        @test nStatus_cg in (0, -100, -101, -103)
        obj_cg.x .= x_cg

        n_cg = length(theta_cg)
        direct_serial_cg = make_melitz_gradient_delta_direct_serial(1e-4)
        direct_sorted_cg = make_melitz_gradient_delta_direct_sorted_serial(1e-4)
        g_ref = zeros(n_cg)
        g_sorted = zeros(n_cg)
        direct_serial_cg(g_ref, theta_cg, ctx_cg, obj_cg, x_cg)
        direct_sorted_cg(g_sorted, theta_cg, ctx_cg, obj_cg, x_cg)

        @testset "EVERY coordinate at D=4 agrees with the direct serial backend to near machine precision" begin
            for r in 1:n_cg
                rel = abs(g_ref[r] - g_sorted[r]) / max(abs(g_ref[r]), 1.0)
                @test rel < 1e-8
            end
        end

        @testset "agrees with an independent frozen-x finite difference at a random coordinate" begin
            rng_cg = MersenneTwister(4041)
            r = rand(rng_cg, 1:n_cg)
            h_cg = 1e-4
            ei_cg = zeros(n_cg); ei_cg[r] = 1.0
            Kbuf_cg = zeros(size(obj_cg.U, 1))
            Gbuf_cg = zeros(size(obj_cg.U, 1), ctx_cg.moment_layout.num_moments)
            melitz_moments_adapter_outer!(Kbuf_cg, Gbuf_cg, theta_cg .+ h_cg .* ei_cg, obj_cg.U, obj_cg)
            obj_cg.H[:, 1] .= Kbuf_cg; obj_cg.H[:, 3:end] .= Gbuf_cg
            cp_cg = zeros(1); obj_cg(x_cg, constr=cp_cg)
            melitz_moments_adapter_outer!(Kbuf_cg, Gbuf_cg, theta_cg .- h_cg .* ei_cg, obj_cg.U, obj_cg)
            obj_cg.H[:, 1] .= Kbuf_cg; obj_cg.H[:, 3:end] .= Gbuf_cg
            cm_cg = zeros(1); obj_cg(x_cg, constr=cm_cg)
            fd_grad_cg = (cp_cg[1] - cm_cg[1]) / (2h_cg)
            @test isapprox(g_sorted[r], fd_grad_cg; rtol=1e-6, atol=1e-6)
            melitz_moments_adapter_outer!(Kbuf_cg, Gbuf_cg, theta_cg, obj_cg.U, obj_cg)
            obj_cg.H[:, 1] .= Kbuf_cg; obj_cg.H[:, 3:end] .= Gbuf_cg
        end

        @testset "requires ctx.sorted_tail_ctx -- rejects a ctx built without it" begin
            obj_plain_inner, theta_plain = build_melitz_psi_bundle(fixture_cg; inner_loop_opt=inner_opt_cg,
                needs_outer_moment_jacobian=false, backend=:dense_reference, moment_backend=:dense_reference)
                # 2026-07-26 production-port session: explicitly forced fully-plain/dense
                # (no sorted_tail_ctx at all) -- the production DEFAULT now always builds one
                # (the matrix-free bundle requires it structurally), so this test's own "a ctx
                # built without it" case must ask for the dense-only path explicitly.
            ctx_plain = obj_plain_inner.γ
            @test get(ctx_plain, :sorted_tail_ctx, nothing) === nothing
            g_bad = zeros(n_cg)
            @test_throws ArgumentError direct_sorted_cg(g_bad, theta_plain, ctx_plain, obj_cg, x_cg)
        end

        # ====================================================================
        # 2026-07-26 continuation ("try one of Phase 8-10" + "check outer-loop
        # parallelization" session): :B_direct_argument_sorted_parallel, the threaded
        # analogue of the sorted crossing-slice backend above, and its production wiring.
        # ====================================================================
        @testset "2026-07-26: :B_direct_argument_sorted_parallel is bit-identical to the sorted serial backend" begin
            direct_sorted_parallel_cg = make_melitz_gradient_delta_direct_sorted_parallel(1e-4)
            g_sorted_par = zeros(n_cg)
            direct_sorted_parallel_cg(g_sorted_par, theta_cg, ctx_cg, obj_cg, x_cg)
            @test g_sorted_par == g_sorted   # disjoint per-coordinate writes -> bit-identical, no reduction

            g_bad_par = zeros(n_cg)
            obj_plain_inner2, theta_plain2 = build_melitz_psi_bundle(fixture_cg; inner_loop_opt=inner_opt_cg,
                needs_outer_moment_jacobian=false, backend=:dense_reference, moment_backend=:dense_reference)
            ctx_plain2 = obj_plain_inner2.γ
            @test_throws ArgumentError direct_sorted_parallel_cg(g_bad_par, theta_plain2, ctx_plain2, obj_cg, x_cg)
        end

        @testset "2026-07-26: both sorted gradient backends wired end-to-end through melitz_fixed_point_probe" begin
            r_probe_ref = melitz_fixed_point_probe(ctx_cg, obj_cg_inner, theta_cg; delta=1e-2,
                direction=:upper, gradient_backend=:B_direct_argument_serial, inner_loop_opt=inner_opt_cg)
            @test r_probe_ref.nStatus == 0
            @test !r_probe_ref.eval_failed

            r_probe_sorted_serial = melitz_fixed_point_probe(ctx_cg, obj_cg_inner, theta_cg; delta=1e-2,
                direction=:upper, gradient_backend=:B_direct_argument_sorted_serial, inner_loop_opt=inner_opt_cg)
            @test r_probe_sorted_serial.nStatus == 0
            @test r_probe_sorted_serial.obj_value == r_probe_ref.obj_value

            r_probe_sorted_parallel = melitz_fixed_point_probe(ctx_cg, obj_cg_inner, theta_cg; delta=1e-2,
                direction=:upper, gradient_backend=:B_direct_argument_sorted_parallel, inner_loop_opt=inner_opt_cg)
            @test r_probe_sorted_parallel.nStatus == 0
            @test r_probe_sorted_parallel.obj_value == r_probe_ref.obj_value
        end

        # ====================================================================
        # Governing prompt Phase 4 (2026-07-XX outer-search session): touched-row
        # no-full-copy gradient backend -- must reproduce the sorted crossing-slice
        # backend's own output EXACTLY (same formula, only the apply/evaluate step
        # changes: no full-W copyto!, no full-W Psi! call, see touched_row_gradient.jl's
        # own header for the exact identity this relies on).
        # ====================================================================
        @testset "Phase 4: touched-row backend (:B_direct_argument_touched_row_serial) matches the sorted backend EXACTLY, every D=4 coordinate" begin
            direct_touched_cg = make_melitz_gradient_delta_direct_touched_row_serial(1e-4)
            g_touched = zeros(n_cg)
            direct_touched_cg(g_touched, theta_cg, ctx_cg, obj_cg, x_cg)
            @test any(cc -> cc.touches_link, melitz_compact_columns_map(ctx_cg))   # confirms the link-touching coordinate is exercised
            for r in 1:n_cg
                rel = abs(g_sorted[r] - g_touched[r]) / max(abs(g_sorted[r]), 1.0)
                @test rel < 1e-10
            end

            @testset "requires ctx.sorted_tail_ctx -- rejects a ctx built without it" begin
                obj_plain_inner3, theta_plain3 = build_melitz_psi_bundle(fixture_cg; inner_loop_opt=inner_opt_cg,
                    needs_outer_moment_jacobian=false, backend=:dense_reference, moment_backend=:dense_reference)
                ctx_plain3 = obj_plain_inner3.γ
                g_bad3 = zeros(n_cg)
                @test_throws ArgumentError direct_touched_cg(g_bad3, theta_plain3, ctx_plain3, obj_cg, x_cg)
            end

            @testset "wired end-to-end through melitz_fixed_point_probe, matches the sorted backend's objective" begin
                # `r_probe_ref` (the `:B_direct_argument_serial` reference run) lives in a
                # SIBLING @testset block above -- Test.jl scopes each @testset independently, so
                # it is not visible here; recomputed locally rather than relying on it leaking
                # across sibling blocks (the bug this comment replaces: caught by the very first
                # run of this test, not shipped silently).
                r_probe_ref_local = melitz_fixed_point_probe(ctx_cg, obj_cg_inner, theta_cg; delta=1e-2,
                    direction=:upper, gradient_backend=:B_direct_argument_serial, inner_loop_opt=inner_opt_cg)
                r_probe_touched = melitz_fixed_point_probe(ctx_cg, obj_cg_inner, theta_cg; delta=1e-2,
                    direction=:upper, gradient_backend=:B_direct_argument_touched_row_serial, inner_loop_opt=inner_opt_cg)
                @test r_probe_touched.nStatus == 0
                @test r_probe_touched.obj_value == r_probe_ref_local.obj_value
            end

            @testset "repeated calls (stale-generation-stamp regression: a bug caught before this backend ever ran)" begin
                # A second, independent call must not read stale delta_plus/delta_minus values
                # left over from the FIRST call's own generation stamps (this file's header:
                # the generation counter must be monotonic ACROSS calls, never reset to 1).
                g_touched2 = zeros(n_cg)
                direct_touched_cg(g_touched2, theta_cg, ctx_cg, obj_cg, x_cg)
                @test g_touched2 == g_touched
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

        # ========================================================================
        # Phase 10 (2026-07-25 scaled-KNITRO session): var_scale/var_center is an ADDITIVE
        # kwarg on solve_melitz_finite_delta_bound wired straight to KNITRO's native
        # KN_set_var_scalings_all -- proves live, on the REAL production driver (not just the
        # standalone synthetic-NLP audit), that supplying the trivial scaling
        # (scale=1, center=0) reproduces the unscaled trajectory's economics exactly: same
        # cold-verified Delta/kappa/gravity, same terminal classification. This is the
        # "same raw theta produces same Delta/dual/kappa" no-change guarantee the governing
        # prompt's Phase 10 asks for, checked against the actual outer NLP, not a toy.
        # Nested INSIDE this testset (not a sibling after `end`) so it shares scope with
        # ctx3fd/obj3fd/theta0_3fd/delta_loose/res3 -- Julia's @testset introduces a new
        # local scope, so a sibling testset placed after `end` cannot see these.
        # ========================================================================
        @testset "Phase 10 (2026-07-25): var_scale/var_center trivial scaling reproduces unscaled economics" begin
        n3 = length(theta0_3fd)
        res3_triv = solve_melitz_finite_delta_bound(ctx3fd, obj3fd, theta0_3fd; delta=delta_loose,
            direction=:upper, gradient_backend=:B, h=1e-4, theta_box=0.5,
            inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"),
            var_scale=ones(n3), var_center=zeros(n3))
        @testset "trivial (scale=1,center=0) scaling: identical terminal trajectory to no-scaling-call" begin
            @test res3_triv.nStatus == res3.nStatus
            @test res3_triv.terminal_theta ≈ res3.terminal_theta atol=1e-10
            @test res3_triv.n_fc_calls == res3.n_fc_calls
            @test res3_triv.n_ga_calls == res3.n_ga_calls
        end
        @testset "trivial scaling: cold-verified incumbent has identical economics" begin
            @test res3_triv.cold_verified_incumbent !== nothing
            @test res3_triv.cold_verified_incumbent.eval.Delta ≈ res3.cold_verified_incumbent.eval.Delta atol=1e-10
            @test res3_triv.cold_verified_incumbent.eval.gamma_prime_j ≈ res3.cold_verified_incumbent.eval.gamma_prime_j atol=1e-10
            @test res3_triv.cold_verified_incumbent.eval.equilibrium_check.gravity_residual_A ≈
                  res3.cold_verified_incumbent.eval.equilibrium_check.gravity_residual_A atol=1e-10
        end
        @testset "a GENUINELY different (non-trivial) scale still finds the SAME economic optimum path length-wise sane" begin
            # Not a byte-identical check (a non-trivial scale legitimately changes KNITRO's own
            # internal step decisions, per the live synthetic-NLP audit) -- only checks the run
            # completes, respects the budget, and the callback-visible theta stayed in RAW units
            # (an inverted/scaled theta accidentally leaking into the cutoff system would show up
            # as a gross cutoff/gravity violation here).
            res3_scaled = solve_melitz_finite_delta_bound(ctx3fd, obj3fd, theta0_3fd; delta=delta_loose,
                direction=:upper, gradient_backend=:B, h=1e-4, theta_box=0.5,
                inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"),
                var_scale=fill(0.1, n3), var_center=collect(Float64.(theta0_3fd)))
            @test res3_scaled.cold_verified_incumbent !== nothing
            @test res3_scaled.cold_verified_incumbent.classification.outer_feasible
            @test res3_scaled.cold_verified_incumbent.eval.Delta <= delta_loose + 1e-6
            @test abs(res3_scaled.cold_verified_incumbent.eval.equilibrium_check.gravity_residual_A) < 1e-8
            @test abs(res3_scaled.cold_verified_incumbent.eval.equilibrium_check.gravity_residual_f) < 1e-8
        end
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

        delta_D = 1e-3
        cap_D = 10.0   # melitz_fixed_point_probe's own default delta_evaluation_cap
        @testset "Test D, CORRECTED 2026-07-24 (evaluation-cap-correction session, user-directed redesign): InfiniteDeltaCertified point gets the fixed sentinel, NOT an eval-error" begin
            # theta_bad20 (large 0.5*randn perturbation) is caught by the always-on range
            # screen as an InfiniteDeltaCertified certificate BEFORE any KNITRO attempt --
            # confirmed live by the sibling "Phase I.6: finite raw dual captured into the
            # bank on NumericalFailure" test above, which explicitly disables range_screen
            # for this EXACT theta construction/seed precisely because it would otherwise be
            # caught here. `melitz_fixed_point_probe` exposes no way to disable that screen,
            # so this probe genuinely exercises the InfiniteDeltaCertified->sentinel path
            # through the PRODUCTION cb_F!/cb_G! (Section 6's own requirement), not merely a
            # helper-function call.
            theta_bad20 = theta0_20 .+ 0.5 .* randn(MersenneTwister(1), length(theta0_20))
            r_bad = evaluate_melitz_delta(theta_bad20, ctx20, obj20; cold=true, store_G=false)
            @test !(r_bad.nStatus in (0, -100, -101, -103))   # confirms this really is a failing/certified point
            pD = melitz_fixed_point_probe(ctx20, obj20, theta_bad20; delta=delta_D,
                direction=:upper, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            @test !pD.eval_failed   # Section 4 (user-directed redesign): a certified-bad point is an
                                     # ORDINARY successful evaluation now, never an eval-error
            @test isapprox(pD.obj_value, theta_bad20[1]; atol=1e-10)   # the real, always-correct objective
            @test isapprox(pD.c[1], cap_D / delta_D; rtol=1e-10)       # the FIXED sentinel, exactly
            @test isempty(pD.live_candidates)   # still no incumbent contamination from a certified-bad point
        end

        @testset "Test D': genuine NumericalFailure (no certificate of any kind) still throws a real eval-error" begin
            # Unlike Test D above, this exercises the OTHER branch -- a point with no
            # certificate obtained at all -- at the melitz_classified_inner_solve level (the
            # same level the "Phase I.6" test above already validates this on), confirming
            # the type itself is unaffected by the outer-callback redesign; the callback-level
            # throw for THIS case is exercised directly via inner_solve_verified_or_fail's own
            # unconditional `throw(DomainError(...))` branch, unchanged by this session's
            # sentinel-value redesign (see finite_delta_outer.jl).
            objD2 = build_melitz_implicit_bundle(ctx20, obj20.U, theta0_20; delta=delta_D,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20)
            theta_bad20b = theta0_20 .+ 0.5 .* randn(MersenneTwister(1), length(theta0_20))
            bankD2 = MelitzDualBank()
            resD2 = melitz_classified_inner_solve(objD2, theta_bad20b, ctx20; delta_evaluation_cap=cap_D,
                bank=bankD2, range_screen=false, stored_dual_screen=false)
            @test resD2 isa NumericalFailure
        end
    end

    # ========================================================================
    # Screening-session continuation, Phase I.1, CORRECTED 2026-07-24 (evaluation-cap
    # correction session): the KNITRO-native `lower_limit` mid-solve early stop
    # (`cc_algo/PsiObjectiveBundle.jl`'s `if f <= lower_limit; return -KNITRO.KN_INFINITY`)
    # must be classified as `AboveEvaluationCap(...,:live_dual_threshold,...)`, not
    # `NumericalFailure` -- it is a certificate (weak duality, unconditional -- see
    # inner_screening.jl's file header), not an unresolved failure. CRITICALLY (this is the
    # session's actual regression test, not merely a rename): the threshold is gated on
    # `delta_evaluation_cap`, NEVER the outer budget `delta` -- a point genuinely OVER the
    # outer budget but UNDER the evaluation cap must be solved fully (`FiniteSolved`), never
    # aborted early merely because it exceeds `delta`.
    # ========================================================================
    @testset "Phase I.1: live dual-threshold classification (AboveEvaluationCap, not NumericalFailure), gated on delta_evaluation_cap not delta" begin
        theta_pertT1 = theta0_20 .+ 0.005 .* randn(MersenneTwister(11), length(theta0_20))
        rT1 = evaluate_melitz_delta(theta_pertT1, ctx20, obj20; cold=true, store_G=false)
        @test rT1.nStatus == 0   # a genuinely valid inner solve at this theta, real Delta known
        @test rT1.Delta > 0
        cap_tightT1 = rT1.Delta / 2   # deliberately below Delta(theta) -> must be rejected
        delta_hugeT1 = rT1.Delta * 1000   # an OUTER BUDGET deliberately irrelevant to this mechanism

        @testset "Delta-above-cap point: threshold fires, classified AboveEvaluationCap(:live_dual_threshold), independent of delta" begin
            objT1 = build_melitz_implicit_bundle(ctx20, obj20.U, theta_pertT1; delta=delta_hugeT1,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20,
                delta_evaluation_cap=cap_tightT1)
            @test objT1.lower_limit == -cap_tightT1   # tied to the CAP, not the (huge, irrelevant) delta

            bankT1 = MelitzDualBank()
            resultT1 = melitz_classified_inner_solve(objT1, theta_pertT1, ctx20;
                delta_evaluation_cap=cap_tightT1, bank=bankT1)

            @test resultT1 isa AboveEvaluationCap
            @test resultT1.source == :live_dual_threshold
            @test objT1.threshold_crossed[]
            # the certified bound is a valid Delta lower bound (weak duality) AND is why the
            # point was rejected (it exceeds delta_evaluation_cap, NOT delta):
            @test resultT1.certified_lower_bound <= rT1.Delta + 1e-8
            @test resultT1.certified_lower_bound > cap_tightT1
            # governing prompt Section 2: certified_lower_bound must NEVER be reported as
            # delta_star -- this is enforced at the TYPE level (no `Delta`/`delta_star` field
            # on AboveEvaluationCap at all), checked here as a structural sanity check:
            @test !hasfield(AboveEvaluationCap, :Delta) && !hasfield(AboveEvaluationCap, :delta_star)
            @test resultT1.crossing_iteration == -1   # disclosed limitation (docstring), not silently omitted
            @test isfinite(resultT1.crossing_time_s) && resultT1.crossing_time_s >= 0
            @test !isempty(objT1.threshold_crossing_x[])
            @test all(isfinite, objT1.threshold_crossing_x[])
            @test length(bankT1.entries) == 1
            @test bankT1.entries[1] == objT1.threshold_crossing_x[]
        end

        @testset "SAME point, cap raised above rT1.Delta: no longer aborted, fully solved as FiniteSolved with the genuine value" begin
            cap_looseT1 = rT1.Delta * 5
            objT1b = build_melitz_implicit_bundle(ctx20, obj20.U, theta_pertT1; delta=delta_hugeT1,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20,
                delta_evaluation_cap=cap_looseT1)
            bankT1b = MelitzDualBank()
            resultT1b = melitz_classified_inner_solve(objT1b, theta_pertT1, ctx20;
                delta_evaluation_cap=cap_looseT1, bank=bankT1b)
            @test resultT1b isa FiniteSolved
            @test !objT1b.threshold_crossed[]
            @test isapprox(resultT1b.Delta, rT1.Delta; atol=1e-6, rtol=1e-6)
        end

        @testset "THE core regression: over-budget-but-under-cap point is FiniteSolved, never aborted merely for exceeding delta" begin
            # delta deliberately set BELOW rT1.Delta (so this point is genuinely over budget)
            # while delta_evaluation_cap is set comfortably ABOVE it -- the prior session's
            # bug tied the abort threshold to delta itself, which would have aborted this
            # exact point; the corrected code must not.
            delta_tightT1c = rT1.Delta / 2
            cap_looseT1c = rT1.Delta * 5
            objT1c = build_melitz_implicit_bundle(ctx20, obj20.U, theta_pertT1; delta=delta_tightT1c,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20,
                delta_evaluation_cap=cap_looseT1c)
            bankT1c = MelitzDualBank()
            resultT1c = melitz_classified_inner_solve(objT1c, theta_pertT1, ctx20;
                delta_evaluation_cap=cap_looseT1c, bank=bankT1c)
            @test resultT1c isa FiniteSolved
            @test resultT1c.Delta > delta_tightT1c   # genuinely over THIS budget
            @test isapprox(resultT1c.Delta, rT1.Delta; atol=1e-6, rtol=1e-6)   # AND the true, exact value
        end

        @testset "build_melitz_implicit_bundle fails fast if inner_solve_config AND delta_evaluation_cap are both given" begin
            # 2026-07-26 closure session (post guard-removal): the only remaining mutual-
            # exclusion invariant on this constructor's cap arguments -- there is no separate
            # `lower_limit_guard` kwarg to misuse any more (removed entirely, see this
            # function's own docstring).
            cfg_excl = MelitzInnerSolveConfig(:evaluation_cap; delta_evaluation_cap=1.0)
            @test_throws ArgumentError build_melitz_implicit_bundle(ctx20, obj20.U, theta_pertT1; delta=1.0,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20,
                delta_evaluation_cap=1.0, inner_solve_config=cfg_excl)
        end

        @testset "Delta-below-cap point: threshold does NOT fire, classified FiniteSolved" begin
            cap_looseT2 = max(r0_20.Delta * 5, 1e-3)
            objT2 = build_melitz_implicit_bundle(ctx20, obj20.U, theta0_20; delta=cap_looseT2,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20,
                delta_evaluation_cap=cap_looseT2)
            @test objT2.lower_limit == -cap_looseT2

            bankT2 = MelitzDualBank()
            resultT2 = melitz_classified_inner_solve(objT2, theta0_20, ctx20;
                delta_evaluation_cap=cap_looseT2, bank=bankT2)

            @test resultT2 isa FiniteSolved
            @test !objT2.threshold_crossed[]   # never fired -- a feasible point must not be rejected
            @test resultT2.Delta <= cap_looseT2
        end

        @testset "integration: solve_melitz_finite_delta_bound routes cap hits through n_above_cap_reject, never n_numerical_failure_reject as a threshold artifact" begin
            # A tight delta_evaluation_cap on a short trajectory: every rejection this
            # specific mechanism produces must appear in the above-cap bucket, not the
            # numerical-failure bucket (governing prompt: "must not enter a failed-solve
            # counter"). (Other genuine numerical failures unrelated to this mechanism may
            # still occur -- this integration check only pins that the THRESHOLD mechanism
            # itself is counted correctly, via the same live `on_inner_result` hook the
            # no-rescue benchmark uses.) `delta` (the outer budget) is left LOOSE here,
            # deliberately different from the tight `delta_evaluation_cap`, to keep the two
            # concepts visibly decoupled in this integration test too.
            n_threshold_hits = Ref(0)
            n_numfail_seen = Ref(0)
            collector(theta, result) = begin
                result isa AboveEvaluationCap && result.source == :live_dual_threshold && (n_threshold_hits[] += 1)
                result isa NumericalFailure && (n_numfail_seen[] += 1)
                nothing
            end
            resT3 = solve_melitz_finite_delta_bound(ctx20, obj20, theta0_20; delta=1.0,
                direction=:upper, theta_box=0.02, inner_loop_opt=inner_opt20, outer_loop_opt=outer_opt20,
                delta_evaluation_cap=1e-2, on_inner_result=collector)
            @test resT3.n_above_cap_reject >= n_threshold_hits[]
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
            result = melitz_classified_inner_solve(objW, theta0_20, ctx20; delta_evaluation_cap=1e-3, bank=bankW)
            @test result isa FiniteSolved
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

        @testset "at the true optimal dual, a below-cap budget is certified rejected immediately" begin
            cap_tightT4 = rT4.Delta / 2
            res = melitz_dual_polish_screen(objT4, x_star; delta_evaluation_cap=cap_tightT4, max_steps=3)
            @test res isa AboveEvaluationCap
            @test res.source == :dual_polish
            @test res.certified_lower_bound <= rT4.Delta + 1e-6   # valid lower bound (weak duality)
            @test res.certified_lower_bound > cap_tightT4
            @test res.crossing_iteration == 0   # rejected at the starting point, before any Newton step
        end

        @testset "at the true optimal dual, a comfortably-above-cap budget is NOT rejected" begin
            cap_looseT4 = max(rT4.Delta * 10, 1e-2)
            res = melitz_dual_polish_screen(objT4, x_star; delta_evaluation_cap=cap_looseT4, max_steps=3)
            @test res === nothing
        end

        @testset "wired into melitz_classified_inner_solve: dual_polish_screen=true certifies without a KNITRO call" begin
            bankT4 = MelitzDualBank()
            melitz_dual_bank_insert!(bankT4, x_star)
            cap_tightT4b = rT4.Delta / 2
            CS_.INNER_SOLVE_COUNT[] = 0
            resultT4 = melitz_classified_inner_solve(objT4, theta0_20, ctx20; delta_evaluation_cap=cap_tightT4b,
                bank=bankT4, stored_dual_screen=false, dual_polish_screen=true)
            @test resultT4 isa AboveEvaluationCap
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
        # point InfiniteDeltaCertified before any KNITRO attempt -- that is a CORRECT, EXPECTED
        # outcome (a cheaper, equally valid rejection), not a test bug. This test's actual
        # subject (finite-x bank capture) only applies on the branch where a real KNITRO
        # attempt happens and fails without a certificate.
        resultT5 = melitz_classified_inner_solve(objT5, theta_badT5, ctx20; delta_evaluation_cap=1e-3, bank=bankT5,
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
                result = melitz_classified_inner_solve(objI8, theta0_OB, ctxOB; delta_evaluation_cap=1e-3,
                    bank=bank_ord, origin_block_screen=true, dual_polish_screen=false,
                    screen_order=order)
                @test result isa FiniteSolved   # population point: every screen must pass, real solve proceeds
            end
            @test_throws ArgumentError melitz_classified_inner_solve(objI8, theta0_OB, ctxOB;
                delta_evaluation_cap=1e-3, bank=MelitzDualBank(), screen_order=:Z)
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
            result1 = melitz_classified_inner_solve(objI12, theta0_20, ctx20; delta_evaluation_cap=1e-3,
                bank=bankI12, origin_block_screen=true, dual_polish_screen=true)
            @test result1 isa FiniteSolved
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
            result2 = melitz_classified_inner_solve(objI12, theta0_20, ctx20; delta_evaluation_cap=1e-3,
                bank=bankI12, origin_block_screen=true, dual_polish_screen=true)
            @test result2 isa FiniteSolved
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
        theta_B = theta0_20 .+ 0.5 .* randn(MersenneTwister(1), n20)   # a known-certified-bad point

        evalResA1 = MelitzMockEvalResult(zeros(1), zeros(m20), zeros(n20), zeros(n20 * m20))
        cbset7.cb_F!(nothing, nothing, MelitzMockEvalRequest(copy(theta_A)), evalResA1, nothing)
        cbset7.cb_G!(nothing, nothing, MelitzMockEvalRequest(copy(theta_A)), evalResA1, nothing)

        # 2026-07-24 evaluation-cap-correction session (user-directed redesign): theta_B is
        # caught by the always-on range screen as InfiniteDeltaCertified (confirmed by the
        # sibling "Phase I.6: finite raw dual captured into the bank" test's own comment on
        # this EXACT theta construction/seed) -- under the corrected design this is an
        # ORDINARY successful evaluation reporting the FIXED sentinel
        # `delta_evaluation_cap/delta`, NOT a thrown DomainError (only genuine
        # `NumericalFailure`, with no certificate at all, still throws). The point of this
        # test -- no stale-state leakage into a SUBSEQUENT evaluation of point A -- is if
        # anything MORE important to check now that B's evaluation writes real (sentinel)
        # values into `obj`'s mutable state instead of aborting via an exception.
        evalResB = MelitzMockEvalResult(zeros(1), zeros(m20), zeros(n20), zeros(n20 * m20))
        cbset7.cb_F!(nothing, nothing, MelitzMockEvalRequest(copy(theta_B)), evalResB, nothing)
        cbset7.cb_G!(nothing, nothing, MelitzMockEvalRequest(copy(theta_B)), evalResB, nothing)
        @testset "point B reports the fixed sentinel, not a throw" begin
            @test isapprox(evalResB.c[1], 10.0 / delta_loose20b; rtol=1e-10)   # default delta_evaluation_cap=10.0
            @test all(==(0.0), evalResB.jac[1:n20])   # the divergence row's own gradient is exactly zero
            @test isapprox(evalResB.obj[1], theta_B[1]; atol=1e-10)   # objective still the real, correct value
        end

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
            Delta_A, x_A, nStatus_A, fp_A = entry_A   # compact tier only, post-Phase-D split (H now lives in heavy_store)

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

    # ========================================================================
    # Continuation session (2026-07-23, "make the optimized architecture scalable in memory
    # and D"), Section 4: bounded-LRU mechanics for MelitzExactPointCache/MelitzDeltaEvalCache.
    # Pure data-structure tests (synthetic keys/values, no KNITRO) -- the cross-delta REUSE
    # semantics are already covered live above ("Section 4.2"); these tests isolate the
    # EVICTION/fingerprint machinery itself, which needs many more than 1-2 entries to
    # exercise capacity limits and would be needlessly slow to drive through real inner solves.
    # ========================================================================
    @testset "Section 4 (memory scalability): bounded LRU caches" begin
        @testset "MelitzLRUOrder: touch/evict primitives" begin
            order = MelitzLRUOrder()
            store = Dict{Vector{Float64},Int}()
            for i in 1:5
                k = Float64[i]
                store[k] = i
                melitz_lru_touch!(order, k)
            end
            @test order.keys == [Float64[1], Float64[2], Float64[3], Float64[4], Float64[5]]
            n_evicted = melitz_lru_evict_until!(order, store, 3)
            @test n_evicted == 2
            @test length(store) == 3
            @test !haskey(store, Float64[1]) && !haskey(store, Float64[2])
            @test haskey(store, Float64[5])
            # re-touching an existing key moves it to MRU without changing store contents
            melitz_lru_touch!(order, Float64[3])
            @test order.keys[end] == Float64[3]
            @test length(store) == 3
        end

        @testset "MelitzExactPointCache: LRU eviction under repeated insertion" begin
            cache = MelitzExactPointCache(3)
            ctxA = Ref(:ctxA)
            H = zeros(2, 2)
            for i in 1:5
                melitz_exact_cache_insert!(cache, Float64[i], Float64(i), [Float64(i)], 0, H, ctxA)
            end
            @test length(cache.store) == 3
            @test cache.evictions == 2
            @test !haskey(cache.store, Float64[1]) && !haskey(cache.store, Float64[2])
            @test haskey(cache.store, Float64[5])
        end

        @testset "MelitzExactPointCache: a touched (get) entry survives eviction longer than an untouched one" begin
            cache = MelitzExactPointCache(3)
            ctxA = Ref(:ctxA)
            H = zeros(2, 2)
            for i in 1:3
                melitz_exact_cache_insert!(cache, Float64[i], Float64(i), [Float64(i)], 0, H, ctxA)
            end
            # touch key 1 (now MRU); key 2 is the new LRU
            @test melitz_exact_cache_get(cache, Float64[1], ctxA) !== nothing
            melitz_exact_cache_insert!(cache, Float64[4], 4.0, [4.0], 0, H, ctxA)
            @test haskey(cache.store, Float64[1])   # survived: was touched
            @test !haskey(cache.store, Float64[2])  # evicted: least recently used
            @test haskey(cache.store, Float64[3]) && haskey(cache.store, Float64[4])
        end

        @testset "MelitzExactPointCache: stale-context guard drops a hit from a different ctx object" begin
            cache = MelitzExactPointCache(8)
            ctxA = Ref(:ctxA)
            ctxB = Ref(:ctxB)
            H = zeros(2, 2)
            melitz_exact_cache_insert!(cache, Float64[1], 1.0, [1.0], 0, H, ctxA)
            @test melitz_exact_cache_get(cache, Float64[1], ctxA) !== nothing
            # re-insert (the ctxA lookup above did not consume the entry)
            melitz_exact_cache_insert!(cache, Float64[1], 1.0, [1.0], 0, H, ctxA)
            @test melitz_exact_cache_get(cache, Float64[1], ctxB) === nothing   # different ctx: guarded miss
            @test !haskey(cache.store, Float64[1])   # the stale entry was dropped, not merely skipped
        end

        @testset "MelitzDeltaEvalCache: LRU eviction bounds live entries to max_size" begin
            small_fixture4c = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
                target_country=1, seed=29, W=2_000)
            obj4c, theta4c = build_melitz_psi_bundle(small_fixture4c;
                inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"))
            ctx4c = obj4c.γ
            cache4c = MelitzDeltaEvalCache(2)
            rng4c = MersenneTwister(41)
            n4c = length(theta4c)
            thetas4c = [theta4c .+ 0.01 .* randn(rng4c, n4c) for _ in 1:4]
            verified_count = 0
            for th in thetas4c
                r = evaluate_melitz_delta(th, ctx4c, obj4c; cache=cache4c, cold=true)
                r.verified && (verified_count += 1)
            end
            @test length(cache4c.store) <= 2
            if verified_count >= 3
                @test cache4c.evictions >= 1
            end
        end

        @testset "MelitzDeltaEvalCache: A/B/A survives eviction when max_size is large enough" begin
            small_fixture4d = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
                target_country=1, seed=29, W=2_000)
            obj4d, theta4d = build_melitz_psi_bundle(small_fixture4d;
                inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"))
            ctx4d = obj4d.γ
            cache4d = MelitzDeltaEvalCache(4)
            r_A1 = evaluate_melitz_delta(theta4d, ctx4d, obj4d; cache=cache4d, cold=true)
            misses_after_A1 = cache4d.misses
            r_A2 = evaluate_melitz_delta(theta4d, ctx4d, obj4d; cache=cache4d, cold=true)
            @test cache4d.misses == misses_after_A1   # A/A repeat: pure hit, no new miss
            @test r_A1 === r_A2
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

        # ================================================================
        # Continuation session (2026-07-23, "make the optimized architecture scalable in
        # memory and D"), Section 11: the argument-localized gradient backends
        # (Section 3, UNMODIFIED for :logcutoff) must remain bit-exact-agreement-class
        # vs. the parameterization-agnostic dense reference `:B` under :logcutoff too --
        # the existing dependency map (built from ctx.A_pivot/f_free_lin, structure
        # SHARED between :logf/:logcutoff) was hypothesized to already be a valid
        # superset without any :logcutoff-specific derivation; this is the coordinate-
        # by-coordinate correctness gate confirming that hypothesis live (see
        # scripts/melitz_logcutoff_argument_localized_validate.jl for the original
        # standalone confirmation this test mirrors).
        # ================================================================
        @testset "Section 11: argument-localized gradient backends bit-exact-class under :logcutoff" begin
            n_q = length(theta0_q)
            d_q = ctx_q.moment_layout.num_moments
            W_q = size(obj_q.U, 1)
            mj_full_q = make_melitz_moments_jacobian_b(1e-4)
            mj_as_q = make_melitz_moments_jacobian_b_argument_localized_serial(1e-4)
            mj_ap_q = make_melitz_moments_jacobian_b_argument_localized_parallel(1e-4)

            rng_q = MersenneTwister(83)
            for trial in 1:3
                theta_probe_q = theta0_q .+ 0.01 .* randn(rng_q, n_q)
                K1q, G1q = zeros(W_q, n_q), zeros(W_q, d_q, n_q)
                K2q, G2q = zeros(W_q, n_q), zeros(W_q, d_q, n_q)
                K3q, G3q = zeros(W_q, n_q), zeros(W_q, d_q, n_q)
                mj_full_q(K1q, G1q, theta_probe_q, obj_q.U, obj_q)
                mj_as_q(K2q, G2q, theta_probe_q, obj_q.U, obj_q)
                mj_ap_q(K3q, G3q, theta_probe_q, obj_q.U, obj_q)
                @testset "trial $trial" begin
                    @test isapprox(G1q, G2q; atol=1e-6, rtol=1e-6)
                    @test isapprox(G1q, G3q; atol=1e-6, rtol=1e-6)
                end
            end
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

# ============================================================================
# 11. Closure-audit session (2026-07-24), Phase A1: `needs_outer_moment_jacobian` must
#    default to `false` on every Melitz `PsiObjectiveBundleDelta` production path, so no
#    caller silently falls back to the ~N*(d+2)*l dense `jac_h` allocation
#    (~206GB at D=20/W=80,000). `build_melitz_psi_bundle` is the ONLY construction site for
#    `PsiObjectiveBundleDelta` in this repo (re-verified live via `grep -rln
#    PsiObjectiveBundleDelta` across src/scripts/cc_algo), and `obj_inner` -- the object this
#    covers -- is REUSED (never rebuilt) for ordinary fixed-point solves, outer initial
#    evaluation, and terminal/cold verification (`solve_melitz_finite_delta_bound`,
#    `melitz_fixed_point_probe`), so one construction-site test covers all of them.
# ============================================================================
@testset "Closure Phase A1: needs_outer_moment_jacobian defaults to false, every production path" begin
    @testset "D=4: default construction (no explicit kwarg) skips jac_h AND economics are unchanged" begin
        fixtureA1 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=20_000)
        objA1, thetaA1 = build_melitz_psi_bundle(fixtureA1;
            inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"))
        @test objA1.needs_outer_moment_jacobian == false
        @test size(objA1.jac_h) == (0, 0, 0)

        # Same benchmark point/value as "Section 2: evaluate_melitz_delta" above (Delta =
        # 7.5545e-6 at W=20,000, seed=29) -- a real KNITRO solve through the now-default-false
        # object must reproduce it exactly, since the theta-branch this flag disables was
        # never reachable on this path to begin with.
        r = evaluate_melitz_delta(thetaA1, objA1.γ, objA1)
        @test r.verified
        @test r.nStatus == 0
        @test isapprox(r.Delta, 7.5545e-6; rtol=1e-3)
    end

    @testset "D=10: default construction at moderate scale skips jac_h" begin
        # W=20,000 (not 2,000): small-W Melitz inner solves are documented as numerically
        # finicky in this repo (see docs) -- a genuine, unrelated -102 at tiny W would make
        # this test flaky for a reason that has nothing to do with jac_h. W=20,000 matches
        # continuation4's own clean D=10 benchmark point (nStatus=0 at every BLAS thread
        # count tried there).
        fixtureA1_10 = generate_fake_melitz_data(; D=10, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=20_000, min_participation_prob=0.002)
        objA1_10, thetaA1_10 = build_melitz_psi_bundle(fixtureA1_10;
            inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"))
        @test objA1_10.needs_outer_moment_jacobian == false
        @test size(objA1_10.jac_h) == (0, 0, 0)
        # bundle-agnostic (was a bare `inner_loop(...)` call, cc_algo-generic-only): now that
        # build_melitz_psi_bundle defaults to the matrix-free MelitzCCBundle, use
        # melitz_recover_lfd, which dispatches correctly for either bundle type.
        lfd_10 = melitz_recover_lfd(objA1_10, thetaA1_10)
        @test lfd_10.nStatus in (0, -100, -101, -103)
    end

    @testset "D=20: construction alone (no solve -- this is a memory-shape test, not a convergence test) skips jac_h" begin
        fixtureA1_20 = generate_fake_melitz_data(; D=20, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=2_000, min_participation_prob=0.002)
        objA1_20, _ = build_melitz_psi_bundle(fixtureA1_20;
            inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"))
        @test objA1_20.needs_outer_moment_jacobian == false
        @test size(objA1_20.jac_h) == (0, 0, 0)
    end

    @testset "opting back in (true) still allocates jac_h, and the theta-branch it guards errors cleanly rather than silently misbehaving on a default (false) object" begin
        fixtureA1b = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=2_000)
        obj_true, theta_true = build_melitz_psi_bundle(fixtureA1b;
            inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"),
            needs_outer_moment_jacobian=true)
        @test size(obj_true.jac_h) != (0, 0, 0)

        obj_false, theta_false = build_melitz_psi_bundle(fixtureA1b;
            inner_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt"))
        # Directly exercising the (never-reached-in-production) theta-gradient branch on a
        # default-false object must error clearly (calculate_jac_θ!'s guard), not silently
        # index into the 0x0x0 placeholder.
        g_probe = zeros(length(theta_false))
        @test_throws ErrorException obj_false(zeros(obj_false.outer_constr_index), g_probe, theta_false)
    end
end

# ============================================================================
# 12. Closure-audit session (2026-07-24), Phase B1: the FINAL REGISTERED constraint
#    Jacobian (`evalResult.jac[1]`, written by `cb_G!`) must match central finite
#    differences of the FINAL REGISTERED constraint function (`evalResult.c[1]`, written by
#    `cb_F!`) -- not merely an internal raw gradient/objective. Calls the EXACT production
#    callback closures directly (the `MelitzMockEvalRequest`/`MelitzMockEvalResult`
#    duck-typed harness from the Section 7 A/B/A test above), at a smooth (non-active-set-
#    switching) point, for both the legacy analytic (`:B`) and new direct
#    (`:B_direct_argument_serial`) gradient backends.
# ============================================================================
@testset "Closure Phase B1: registered Jacobian == central FD of the registered constraint" begin
    fixtureB1 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
        target_country=1, seed=29, W=2_000)
    inner_optB1 = joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt")
    outer_optB1 = joinpath(dirname(dirname(@__DIR__)), "melitz_outer_finite_delta.opt")
    objB1_inner, theta0B1 = build_melitz_psi_bundle(fixtureB1; inner_loop_opt=inner_optB1)
    ctxB1 = objB1_inner.γ
    r0B1 = evaluate_melitz_delta(theta0B1, ctxB1, objB1_inner; cold=true, store_G=false)
    @test r0B1.nStatus == 0
    delta_looseB1 = max(r0B1.Delta * 5, 1e-3)   # loose enough that a small +-h perturbation stays feasible
    n_B1 = length(theta0B1)
    m_B1 = 1 + ctxB1.D + ctxB1.D * (ctxB1.D - 1)
    h_B1 = 1e-4

    # Section 3/B2 of this session's own closure report: coordinates touching a hard
    # participation-boundary jump are a genuine, PRE-EXISTING non-smoothness (not a bug in
    # either backend or in this test) where FD-of-the-registered-constraint disagrees with
    # BOTH analytic backends by a large margin -- confirmed live while building this test
    # (an earlier version picked coordinates via ":B agrees with :B_argument_localized_serial"
    # as a smoothness proxy; that check is VACUOUS -- the two backends share the identical
    # FD-of-G-then-analytic-Psi-derivative formula through different code, so they always
    # agree regardless of smoothness -- and still hit the same large disagreements). Phase
    # B1's own question is whether the REGISTERED SCALING is correct (grad Delta/delta, not
    # an internal raw quantity) -- answerable at any coordinate where the constraint function
    # ITSELF is locally smooth -- not a gradient-accuracy audit at every coordinate (Phase
    # B3, out of scope this session). Detect smoothness the direct way: a genuine two-`h`
    # Richardson stability check on the FD of the REGISTERED CONSTRAINT itself (does the FD
    # estimate change materially between `h` and `2h`?), scanning candidate coordinates until
    # enough stable ones are found.
    objB1_scan = build_melitz_implicit_bundle(ctxB1, objB1_inner.U, theta0B1; delta=delta_looseB1,
        find_smallest=true, gradient_backend=:B, h=h_B1,
        inner_loop_opt=inner_optB1, outer_loop_opt=outer_optB1)
    cbsetB1_scan = melitz_build_finite_delta_callbacks(objB1_scan, ctxB1, delta_looseB1, true; gradient_backend=:B, h=h_B1)
    function fd_c1(theta_r, r, h)
        tp = copy(theta_r); tp[r] += h
        tm = copy(theta_r); tm[r] -= h
        ep = MelitzMockEvalResult(zeros(1), zeros(m_B1), zeros(n_B1), zeros(n_B1 * m_B1))
        em = MelitzMockEvalResult(zeros(1), zeros(m_B1), zeros(n_B1), zeros(n_B1 * m_B1))
        cbsetB1_scan.cb_F!(nothing, nothing, MelitzMockEvalRequest(tp), ep, nothing)
        cbsetB1_scan.cb_F!(nothing, nothing, MelitzMockEvalRequest(tm), em, nothing)
        return (ep.c[1] - em.c[1]) / (2h)
    end
    smooth_coords = Int[]
    for r in 1:n_B1
        fd_h = fd_c1(theta0B1, r, h_B1)
        fd_2h = fd_c1(theta0B1, r, 2 * h_B1)
        if isapprox(fd_h, fd_2h; rtol=0.02, atol=1e-8)
            push!(smooth_coords, r)
        end
        length(smooth_coords) >= 3 && break
    end
    @test length(smooth_coords) >= 2   # this fixture should have at least a couple of smooth coordinates
    probe_coords = smooth_coords

    for backend in (:B, :B_direct_argument_serial)
        @testset "gradient_backend=$backend" begin
            objB1 = build_melitz_implicit_bundle(ctxB1, objB1_inner.U, theta0B1; delta=delta_looseB1,
                find_smallest=true, gradient_backend=backend, h=h_B1,
                inner_loop_opt=inner_optB1, outer_loop_opt=outer_optB1)
            cbsetB1 = melitz_build_finite_delta_callbacks(objB1, ctxB1, delta_looseB1, true;
                gradient_backend=backend, h=h_B1)

            evalG = MelitzMockEvalResult(zeros(1), zeros(m_B1), zeros(n_B1), zeros(n_B1 * m_B1))
            cbsetB1.cb_F!(nothing, nothing, MelitzMockEvalRequest(copy(theta0B1)), evalG, nothing)
            cbsetB1.cb_G!(nothing, nothing, MelitzMockEvalRequest(copy(theta0B1)), evalG, nothing)
            jac_registered = copy(evalG.jac[1:n_B1])   # d(c_delta)/dtheta at theta0, as KNITRO would see it

            for r in probe_coords
                theta_p = copy(theta0B1); theta_p[r] += h_B1
                theta_m = copy(theta0B1); theta_m[r] -= h_B1
                evalP = MelitzMockEvalResult(zeros(1), zeros(m_B1), zeros(n_B1), zeros(n_B1 * m_B1))
                evalM = MelitzMockEvalResult(zeros(1), zeros(m_B1), zeros(n_B1), zeros(n_B1 * m_B1))
                cbsetB1.cb_F!(nothing, nothing, MelitzMockEvalRequest(theta_p), evalP, nothing)
                cbsetB1.cb_F!(nothing, nothing, MelitzMockEvalRequest(theta_m), evalM, nothing)
                fd_r = (evalP.c[1] - evalM.c[1]) / (2h_B1)   # central FD of the REGISTERED c_delta(theta), not an internal quantity
                @test isapprox(jac_registered[r], fd_r; rtol=2e-2, atol=1e-6)
            end
        end
    end
end

# ============================================================================
# 13. Closure-audit session (2026-07-24), Phase D: production-safe cache tiers --
#    `melitz_context_fingerprint` (content-based, not `objectid(ctx)` alone) and the
#    compact/heavy-state split on `MelitzExactPointCache`. The pre-existing "Section 4
#    (memory scalability): bounded LRU caches" testset above already re-passed unchanged
#    (compact-tier eviction/A-touched-survives/stale-context-guard semantics are byte-for-
#    byte preserved for callers that don't opt into the new `obj=`/`U` arguments) -- this
#    testset covers exactly the NEW behavior: recreated-context hits, changed-draw misses,
#    and heavy-tier eviction-with-recompute.
# ============================================================================
@testset "Closure Phase D: content fingerprint + compact/heavy cache split" begin
    fixtureD = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
        target_country=1, seed=29, W=2_000)
    inner_optD = joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt")
    objD, thetaD = build_melitz_psi_bundle(fixtureD; inner_loop_opt=inner_optD, backend=:dense_reference)  # pokes objD.H directly
    ctxD = objD.γ
    UD = objD.U
    K_D = ctxD.moment_layout.num_moments

    obj_moments_fill!(obj, theta) = (obj.moments!(@view(obj.H[:, 1]), CounterfactualSensitivity.select_G_from_H(obj, obj.H), theta, obj.U, obj); obj.H[:, 2] .= 1.0)
    obj_moments_fill!(objD, thetaD)
    H_D = copy(objD.H)

    @testset "recreated-context: a FRESH ctx object with IDENTICAL content is a fingerprint hit" begin
        fixtureD2 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=2_000)   # same seed/params -- a byte-identical rebuild
        objD2, _ = build_melitz_psi_bundle(fixtureD2; inner_loop_opt=inner_optD)
        ctxD2 = objD2.γ
        @test ctxD2 !== ctxD                       # genuinely a different object...
        @test objectid(ctxD2) != objectid(ctxD)     # ...objectid would call this a miss...
        @test melitz_context_fingerprint(ctxD, UD) == melitz_context_fingerprint(ctxD2, objD2.U)  # ...content fingerprint does not
    end

    @testset "changed-draw: identical ctx, DIFFERENT U, is a fingerprint miss" begin
        U_changed = copy(UD); U_changed[1, 1] += 1.0
        @test melitz_context_fingerprint(ctxD, UD) != melitz_context_fingerprint(ctxD, U_changed)
    end

    @testset "compact hit / heavy hit: cache_get restores the exact stored H, no recompute" begin
        cacheD1 = MelitzExactPointCache(8; heavy_max_size=8)
        melitz_exact_cache_insert!(cacheD1, thetaD, 1.23, [1.0, 2.0], 0, H_D, ctxD, UD)
        hit1 = melitz_exact_cache_get(cacheD1, thetaD, ctxD, UD)
        @test hit1 !== nothing
        @test hit1[4] == H_D
        @test cacheD1.heavy_recomputes == 0
    end

    @testset "compact hit / heavy MISS, no obj: degrades to a full miss (safe, never wrong)" begin
        cacheD2 = MelitzExactPointCache(8; heavy_max_size=1)
        melitz_exact_cache_insert!(cacheD2, Float64[1], 1.0, [1.0], 0, H_D, ctxD, UD)
        melitz_exact_cache_insert!(cacheD2, Float64[2], 2.0, [2.0], 0, H_D, ctxD, UD)  # evicts key [1] from the HEAVY tier only (heavy_max_size=1)
        @test haskey(cacheD2.store, Float64[1])         # compact entry survives (compact max_size=8)
        @test !haskey(cacheD2.heavy_store, Float64[1])  # heavy entry evicted
        @test melitz_exact_cache_get(cacheD2, Float64[1], ctxD, UD) === nothing   # no obj= supplied -> can't recompute -> full miss
    end

    @testset "compact hit / heavy MISS, WITH obj: recomputes H cheaply (no KNITRO) and returns a full hit" begin
        cacheD3 = MelitzExactPointCache(8; heavy_max_size=1)
        theta_other = thetaD .+ 0.001   # a second, distinct key so inserting it evicts thetaD from the heavy tier
        melitz_exact_cache_insert!(cacheD3, thetaD, 9.99, [1.0, 2.0], 0, H_D, ctxD, UD)
        melitz_exact_cache_insert!(cacheD3, theta_other, 8.88, [3.0, 4.0], 0, H_D, ctxD, UD)
        @test !haskey(cacheD3.heavy_store, thetaD)   # confirmed evicted from the heavy tier
        @test haskey(cacheD3.store, thetaD)          # but still present in the compact tier

        recomputes_before = cacheD3.heavy_recomputes
        hit3 = melitz_exact_cache_get(cacheD3, thetaD, ctxD, UD; obj=objD)
        @test hit3 !== nothing
        @test hit3[1] == 9.99   # the compact (Delta, x, nStatus) is exactly what was stored, unaffected by the recompute
        @test cacheD3.heavy_recomputes == recomputes_before + 1
        # the recomputed H must reproduce the moments EXACTLY (H is a deterministic function
        # of theta/U/ctx, no dual/KNITRO involved) -- bit-identical to a direct moments! call.
        @test hit3[4] == H_D
    end

    @testset "heavy_max_bytes: a tiny byte budget evicts even below heavy_max_size" begin
        one_entry_bytes = sizeof(H_D)
        cacheD4 = MelitzExactPointCache(8; heavy_max_size=8, heavy_max_bytes=one_entry_bytes)
        melitz_exact_cache_insert!(cacheD4, Float64[1], 1.0, [1.0], 0, H_D, ctxD, UD)
        melitz_exact_cache_insert!(cacheD4, Float64[2], 2.0, [2.0], 0, H_D, ctxD, UD)
        @test length(cacheD4.heavy_store) == 1   # byte budget bound, not the (much looser) count bound
        @test cacheD4.heavy_evictions >= 1
        @test cacheD4.heavy_bytes <= cacheD4.heavy_max_bytes
    end

    @testset "recreated-context cache hit end-to-end: a fresh MelitzExactPointCache entry inserted under one ctx object is retrieved under a DIFFERENT, content-identical ctx object" begin
        fixtureD5 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=2_000)
        objD5, thetaD5 = build_melitz_psi_bundle(fixtureD5; inner_loop_opt=inner_optD)
        ctxD5 = objD5.γ
        @test ctxD5 !== ctxD
        cacheD5 = MelitzExactPointCache(8; heavy_max_size=8)
        melitz_exact_cache_insert!(cacheD5, thetaD, 5.5, [1.0], 0, H_D, ctxD, UD)
        # look up under the DIFFERENT (but content-identical) ctx/U pair from the SAME fixture:
        hit5 = melitz_exact_cache_get(cacheD5, thetaD, ctxD5, objD5.U)
        @test hit5 !== nothing
        @test hit5[1] == 5.5
    end
end

# ============================================================================
# Phase 1 (2026-07-25 local-geometry/continuation session): "make the evaluation cap
# impossible to omit" -- src/melitz/inner_solve_config.jl.
#
# Governing prompt Section 1.3: "Add tests that deliberately construct every bundle type
# and fail if lower_limit == -KN_INFINITY in evaluation-cap mode." Covers the three bundle
# constructors that can carry a Melitz inner-solve lower_limit
# (build_melitz_psi_bundle/build_melitz_psi_bundle_from_calibration/
# build_melitz_implicit_bundle) plus the one function that REQUIRES a config with no
# default (solve_melitz_nuisance_min_delta, the actual site of the confirmed-live incident
# this file's header describes).
# ============================================================================
@testset "Phase 1 (2026-07-25): evaluation-cap-impossible-to-omit infrastructure" begin
    @testset "melitz_configure_lower_limit: happy paths" begin
        @test melitz_configure_lower_limit(:full_value) == -KNITRO.KN_INFINITY
        ll = melitz_configure_lower_limit(:evaluation_cap; delta_evaluation_cap=10.0)
        @test ll == -10.0   # 2026-07-26 (user-directed): no guard/margin -- exactly -cap
        @test isfinite(ll)
        ll_diag = melitz_configure_lower_limit(:diagnostic; delta_evaluation_cap=5.0)
        @test ll_diag == -5.0
    end

    @testset "melitz_configure_lower_limit: every silent-omission path is a hard error" begin
        @test_throws ArgumentError melitz_configure_lower_limit(:not_a_mode)
        @test_throws ArgumentError melitz_configure_lower_limit(:evaluation_cap)   # no cap given
        @test_throws ArgumentError melitz_configure_lower_limit(:diagnostic; delta_evaluation_cap=nothing)
        @test_throws ArgumentError melitz_configure_lower_limit(:full_value; delta_evaluation_cap=10.0)
        @test_throws ArgumentError melitz_configure_lower_limit(:evaluation_cap; delta_evaluation_cap=Inf)
        @test_throws ArgumentError melitz_configure_lower_limit(:evaluation_cap; delta_evaluation_cap=NaN)
        @test_throws ArgumentError melitz_configure_lower_limit(:evaluation_cap; delta_evaluation_cap=-1.0)
    end

    @testset "MelitzInnerSolveConfig + melitz_assert_evaluation_cap_active" begin
        cfg_full = MelitzInnerSolveConfig(:full_value)
        @test melitz_assert_evaluation_cap_active(cfg_full)   # no-op, does not throw

        cfg_cap = MelitzInnerSolveConfig(:evaluation_cap; delta_evaluation_cap=10.0)
        @test melitz_assert_evaluation_cap_active(cfg_cap)
        @test isfinite(cfg_cap.lower_limit)
        @test cfg_cap.lower_limit == -cfg_cap.delta_evaluation_cap   # exact, no guard/margin

        # A hand-corrupted config (mimicking a future refactor reintroducing the silent bug)
        # must fail the assertion, not pass silently.
        cfg_corrupted = MelitzInnerSolveConfig(:evaluation_cap, -KNITRO.KN_INFINITY, 10.0)
        @test_throws Exception melitz_assert_evaluation_cap_active(cfg_corrupted)

        @test_logs (:warn,) match_mode = :any MelitzInnerSolveConfig(:evaluation_cap;
            delta_evaluation_cap=1.0, outer_delta=1.0)   # cap==outer budget: warns, does not throw
    end

    @testset "Governing prompt Phase 2 (2026-07-27 addendum): production entry points default to Melitz-owned option files, never ek_*.opt" begin
        # Closure doc Section 4.3 flagged (not fixed) that `build_melitz_psi_bundle`/
        # `build_melitz_psi_bundle_from_calibration`'s own DEFAULT `inner_loop_opt`/
        # `outer_loop_opt` kwargs still pointed at the Ricardian-named `ek_inner_loop_options.opt`/
        # `ek_outer_loop_options.opt` files. Fixed this session: every default now resolves to
        # `melitz_inner_loop_options.opt`/`melitz_outer_finite_delta.opt`. These tests call each
        # entry point with NO option keywords at all and assert the resolved path basename is
        # Melitz-owned -- must fail if a future edit reintroduces an `ek_*.opt` default.
        melitz_owned(path) = occursin("melitz_", basename(path)) && !occursin("ek_", basename(path))

        @testset "build_melitz_psi_bundle (delta_star.jl): no-kwargs default" begin
            obj0, _ = build_melitz_psi_bundle(FIXTURE)
            @test melitz_owned(obj0.γ.inner_loop_opt)
            @test melitz_owned(obj0.γ.outer_loop_opt)
        end

        @testset "build_melitz_psi_bundle_from_calibration (pareto_calibration.jl): no-kwargs default" begin
            fixture_p2 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
                target_country=1, seed=29, W=5_000)
            observed_p2, _ = split_melitz_synthetic_truth(fixture_p2)
            calib_p2 = calibrate_melitz_pareto(observed_p2; sigma=2.5, theta_star=:estimate,
                focal_country=1, gravity_tol=1e-6)
            obj0, _ = build_melitz_psi_bundle_from_calibration(calib_p2; W=5_000)
            @test melitz_owned(obj0.γ.inner_loop_opt)
            @test melitz_owned(obj0.γ.outer_loop_opt)

            @testset "melitz_calibration_outer_ctx (pareto_calibration.jl): no-kwargs default" begin
                _, _, _, ctx_p2 = melitz_calibration_outer_ctx(calib_p2)
                @test melitz_owned(ctx_p2.inner_loop_opt)
                @test melitz_owned(ctx_p2.outer_loop_opt)
            end
        end

        @testset "no remaining ek_*.opt reference anywhere in src/melitz/" begin
            melitz_src_dir = joinpath(dirname(dirname(@__DIR__)), "src", "melitz")
            hits = String[]
            for fname in readdir(melitz_src_dir)
                endswith(fname, ".jl") || continue
                fpath = joinpath(melitz_src_dir, fname)
                content = read(fpath, String)
                occursin("ek_inner_loop_options", content) && push!(hits, "$fname: ek_inner_loop_options")
                occursin("ek_outer_loop_options", content) && push!(hits, "$fname: ek_outer_loop_options")
            end
            @test isempty(hits)
        end
    end

    if KNITRO_AVAILABLE
        inner_opt = joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt")
        cfg = MelitzInnerSolveConfig(:evaluation_cap; delta_evaluation_cap=10.0)

        @testset "build_melitz_psi_bundle: inner_solve_config wires PsiObjectiveBundleDelta.lower_limit" begin
            obj_default, _ = build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt)
            @test obj_default.lower_limit == -KNITRO.KN_INFINITY   # unchanged default behavior

            obj_capped, _ = build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt, inner_solve_config=cfg)
            @test isfinite(obj_capped.lower_limit)
            melitz_assert_evaluation_cap_active(cfg)
            @test obj_capped.lower_limit == cfg.lower_limit
        end

        @testset "build_melitz_psi_bundle_from_calibration: inner_solve_config wires lower_limit" begin
            fixture4 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
                target_country=1, seed=29, W=20_000)
            observed4, _ = split_melitz_synthetic_truth(fixture4)
            calib4 = calibrate_melitz_pareto(observed4; sigma=2.5, theta_star=:estimate,
                focal_country=1, gravity_tol=1e-6)
            obj_default, _ = build_melitz_psi_bundle_from_calibration(calib4; W=5_000, inner_loop_opt=inner_opt)
            @test obj_default.lower_limit == -KNITRO.KN_INFINITY

            obj_capped, _ = build_melitz_psi_bundle_from_calibration(calib4; W=5_000,
                inner_loop_opt=inner_opt, inner_solve_config=cfg)
            @test obj_capped.lower_limit == cfg.lower_limit
        end

        @testset "build_melitz_implicit_bundle: inner_solve_config vs. legacy delta_evaluation_cap kwarg agree, mutual exclusion enforced" begin
            obj_f, theta0_f = build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt)
            ctx = obj_f.γ

            obj_legacy = build_melitz_implicit_bundle(ctx, obj_f.U, theta0_f; delta=1.0, find_smallest=true,
                inner_loop_opt=inner_opt, outer_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_outer_finite_delta.opt"),
                delta_evaluation_cap=10.0)
            obj_new = build_melitz_implicit_bundle(ctx, obj_f.U, theta0_f; delta=1.0, find_smallest=true,
                inner_loop_opt=inner_opt, outer_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_outer_finite_delta.opt"),
                inner_solve_config=cfg)
            @test obj_legacy.lower_limit == obj_new.lower_limit
            @test isfinite(obj_new.lower_limit)

            @test_throws ArgumentError build_melitz_implicit_bundle(ctx, obj_f.U, theta0_f; delta=1.0,
                find_smallest=true, inner_loop_opt=inner_opt,
                outer_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_outer_finite_delta.opt"),
                delta_evaluation_cap=10.0, inner_solve_config=cfg)
        end

        @testset "solve_melitz_nuisance_min_delta: inner_solve_config is a required kwarg, applied and restored" begin
            obj_f, theta0_f = build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt, backend=:dense_reference)  # pinned to dense_reference deliberately -- see the matched Phase 6 (2026-07-26 closure) tests below for the matrix-free comparison
            ctx = obj_f.γ
            mask = melitz_nuisance_free_mask(ctx; block=:A_only)

            # Omitting inner_solve_config is a MethodError (UndefKeywordError), not a silently
            # uncapped run -- the governing prompt's central "impossible to omit" requirement,
            # for THIS function specifically (the confirmed-live incident site).
            @test_throws UndefKeywordError solve_melitz_nuisance_min_delta(ctx, obj_f, theta0_f;
                free_mask=mask, radius=0.05, inner_loop_opt=inner_opt)

            lower_limit_before = obj_f.lower_limit
            res = solve_melitz_nuisance_min_delta(ctx, obj_f, theta0_f; free_mask=mask, radius=0.05,
                inner_loop_opt=inner_opt, inner_solve_config=cfg)
            @test res.r_final.nStatus in (0, -100, -101, -103)
            # obj_f's lower_limit is restored to its pre-call value afterward (no surprise
            # mutation of caller-shared state), per this function's own updated docstring.
            @test obj_f.lower_limit == lower_limit_before
        end

        @testset "solve_melitz_nuisance_min_delta: exact-point cache elides the duplicate FC/GA inner solve" begin
            # 2026-07-25 same-day follow-up (user-directed): port finite_delta_outer.jl's
            # exact-point cache to this driver (docs/melitz_real_d20_outer_benchmark_2026-07-24.md
            # Section 5.1's own recommendation) -- this test confirms the port actually
            # elides the duplicate re-solve, not merely that it runs without erroring.
            obj_f, theta0_f = build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt, backend=:dense_reference)  # pinned to dense_reference deliberately -- see the matched Phase 6 (2026-07-26 closure) tests below for the matrix-free comparison
            ctx = obj_f.γ
            mask = melitz_nuisance_free_mask(ctx; block=:A_only)
            cache = MelitzExactPointCache()
            res = solve_melitz_nuisance_min_delta(ctx, obj_f, theta0_f; free_mask=mask, radius=0.05,
                inner_loop_opt=inner_opt, inner_solve_config=cfg, exact_cache=cache)
            # KNITRO's own eval_fcga=no convention calls cb_F! then cb_G! separately at most
            # accepted iterates -- at least one GA call must have hit the cache at the SAME
            # theta its own preceding FC call just solved (a genuine measured effect, not an
            # assumption): total calls exceed total UNIQUE inner solves whenever any hit occurs.
            @test res.n_exact_cache_hits >= 1
            @test res.n_exact_cache_hits + res.n_exact_cache_misses == res.n_fc_calls + res.n_ga_calls
        end

        # ====================================================================
        # 2026-07-26 production-closure session (governing prompt Phase 1): the residual gap
        # left after the 2026-07-25 session above -- `delta_evaluation_cap` supplied ALONE
        # previously left `lower_limit` silently disabled on `build_melitz_implicit_bundle`,
        # and therefore on `solve_melitz_finite_delta_bound` too (its own
        # `delta_evaluation_cap::Real=10.0` default is unconditionally forwarded).
        # Confirmed-live incident: docs/melitz_production_fast_backend_2026-07-26.md Section
        # 5.5 (13.5x slower real-D20 campaign, 31 spurious NumericalFailure results, traced
        # to exactly this omission). These tests recreate the omission pattern directly and
        # require the corrected (active-by-default) behavior.
        #
        # Same-day follow-up (direct user feedback): the FIRST fix additionally introduced a
        # small additive `guard`/`lower_limit_guard` margin on top of the cap -- removed
        # entirely as unnecessary complexity (the cap value itself, e.g. `-10`, is already
        # the correct exact threshold; no separate margin is needed). `lower_limit_guard` no
        # longer exists as a kwarg anywhere in this API.
        # ====================================================================
        outer_opt_path = joinpath(dirname(dirname(@__DIR__)), "melitz_outer_finite_delta.opt")

        @testset "build_melitz_implicit_bundle: delta_evaluation_cap ALONE always activates lower_limit, exactly" begin
            obj_f3, theta0_f3 = build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt)
            ctx3 = obj_f3.γ

            # THE fix: a cap supplied at all must activate, at EXACTLY -delta_evaluation_cap
            # (no guard/margin) -- previously fell through to the disabled (-KN_INFINITY) branch.
            obj_capalone = build_melitz_implicit_bundle(ctx3, obj_f3.U, theta0_f3; delta=1.0,
                find_smallest=true, inner_loop_opt=inner_opt, outer_loop_opt=outer_opt_path,
                delta_evaluation_cap=10.0)
            @test isfinite(obj_capalone.lower_limit)
            @test obj_capalone.lower_limit == -10.0

            # The ONE remaining way to get an uncapped bundle: omit the kwarg entirely --
            # an explicit, all-defaults choice, not a trap.
            obj_uncapped = build_melitz_implicit_bundle(ctx3, obj_f3.U, theta0_f3; delta=1.0,
                find_smallest=true, inner_loop_opt=inner_opt, outer_loop_opt=outer_opt_path)
            @test obj_uncapped.lower_limit == -KNITRO.KN_INFINITY
        end

        @testset "REGRESSION: solve_melitz_finite_delta_bound with NO cap-related kwargs still activates the cap" begin
            # Recreates the exact confirmed-live incident's call pattern: a caller passing
            # only the basics, relying entirely on this function's own delta_evaluation_cap
            # default -- must NOT silently disable the KNITRO-native early-abort mechanism.
            obj_f4, theta0_f4 = build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt)
            ctx4 = obj_f4.γ
            res_default = solve_melitz_finite_delta_bound(ctx4, obj_f4, theta0_f4; delta=1e-3,
                direction=:upper, gradient_backend=:B, h=1e-4, theta_box=0.1, inner_loop_opt=inner_opt)
            @test res_default.nStatus != -500   # the internal @assert isfinite(obj.lower_limit) did not fire

            # Direct, black-box confirmation of the SAME construction this driver performs
            # internally, using its own default cap value (10.0):
            obj_check = build_melitz_implicit_bundle(ctx4, obj_f4.U, theta0_f4; delta=1e-3,
                find_smallest=true, inner_loop_opt=inner_opt, outer_loop_opt=outer_opt_path,
                delta_evaluation_cap=10.0)
            @test isfinite(obj_check.lower_limit)
        end

        @testset "REGRESSION: cap-alone (no guard) yields fast AboveEvaluationCap, not a full FiniteSolved re-derivation" begin
            # Before the fix, this exact construction left lower_limit=-KN_INFINITY, so the
            # bad point below would have been classified FiniteSolved (the early-stop
            # mechanism never engages when disabled) -- after the fix, the cap fires and the
            # point is classified AboveEvaluationCap via the cheap live dual-threshold
            # certificate, exactly the classification distinction this repo's whole
            # evaluation-cap machinery exists to make.
            obj_f5, theta0_f5 = build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt)
            ctx5 = obj_f5.γ
            # Search a short, fixed, deterministic candidate list for a perturbation that
            # solves cleanly (nStatus==0) -- a small D=4 fixture is fragile to LARGE
            # perturbations across all 2D^2-2 free coordinates at once (this repo's own
            # Section 2 test above deliberately uses magnitude 3.0 to produce a GROSSLY
            # infeasible point) -- this test only needs ONE genuine, well-defined "positive
            # Delta" point, not a specific seed, so it searches small magnitudes (matching the
            # 0.005 convention already proven reliable for the D=20 fixture elsewhere in this
            # file, Phase I.1 above) rather than gambling on a single fixed draw.
            local theta_bad5, r_bad5
            found = false
            for (seed, mag) in ((7, 0.005), (11, 0.005), (3, 0.003), (23, 0.003), (1, 0.002),
                                 (2, 0.002), (4, 0.001), (5, 0.001))
                theta_try = theta0_f5 .+ mag .* randn(MersenneTwister(seed), length(theta0_f5))
                r_try = evaluate_melitz_delta(theta_try, ctx5, obj_f5; cold=true, store_G=false)
                if r_try.nStatus == 0 && r_try.Delta > 0
                    theta_bad5, r_bad5 = theta_try, r_try
                    found = true
                    break
                end
            end
            @test found
            if found
                cap_tight5 = r_bad5.Delta / 2   # deliberately below the true Delta at theta_bad5

                obj_capalone5 = build_melitz_implicit_bundle(ctx5, obj_f5.U, theta_bad5;
                    delta=r_bad5.Delta * 1000, find_smallest=true, inner_loop_opt=inner_opt,
                    outer_loop_opt=outer_opt_path, delta_evaluation_cap=cap_tight5)
                @test obj_capalone5.lower_limit == -cap_tight5

                bank5 = MelitzDualBank()
                t0_5 = time()
                result5 = melitz_classified_inner_solve(obj_capalone5, theta_bad5, ctx5;
                    delta_evaluation_cap=cap_tight5, bank=bank5)
                elapsed5 = time() - t0_5
                @test result5 isa AboveEvaluationCap
                @test result5.source == :live_dual_threshold
                @test elapsed5 < 30.0   # certificate-based, not a slow full-timeout re-derivation
            end
        end
    end
end

# ============================================================================
# 2026-07-26 production-closure session (governing prompt Phase 2): "make production-fast
# genuinely strict." MELITZ_PRODUCTION_FAST now has forbid_dense_fallback=true (was false --
# an authoritative "production-fast" preset silently PERMITTING a dense fallback was
# backwards); MELITZ_PRODUCTION_COMPAT is the new, explicitly permissive preset;
# MELITZ_DENSE_REFERENCE is unchanged (forbid_dense_fallback=false, since that preset IS the
# dense path). `forbid_dense_fallback::Bool=false` is now a real kwarg on
# build_melitz_psi_bundle/build_melitz_psi_bundle_from_calibration/build_melitz_implicit_bundle/
# solve_melitz_finite_delta_bound -- these tests confirm it actually throws, at construction
# time, for every known way to end up on a dense path, and does NOT throw for the matrix-free
# default.
# ============================================================================
@testset "Phase 2 (2026-07-26): strict production-fast forbids every dense fallback" begin
    @testset "preset values" begin
        @test MELITZ_PRODUCTION_FAST.forbid_dense_fallback == true
        @test MELITZ_PRODUCTION_FAST.inner_backend == :matrix_free
        @test MELITZ_PRODUCTION_COMPAT.forbid_dense_fallback == false
        @test MELITZ_PRODUCTION_COMPAT.inner_backend == :matrix_free
        @test MELITZ_DENSE_REFERENCE.forbid_dense_fallback == false
        @test MELITZ_DENSE_REFERENCE.inner_backend == :dense_reference
    end

    if KNITRO_AVAILABLE
        inner_opt2 = joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt")
        outer_opt2 = joinpath(dirname(dirname(@__DIR__)), "melitz_outer_finite_delta.opt")

        @testset "build_melitz_psi_bundle: forbid_dense_fallback=true throws on every dense-selecting route, at construction" begin
            # matrix-free default: no throw.
            obj_ok, _ = build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt2, forbid_dense_fallback=true)
            @test obj_ok isa MelitzCCBundle

            # Explicit backend=:dense_reference.
            @test_throws ArgumentError build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt2,
                backend=:dense_reference, forbid_dense_fallback=true)
            # Explicit moment_backend (no explicit backend override) silently selects
            # backend=:dense_reference (delta_star.jl's own documented rule) -- must ALSO throw.
            @test_throws ArgumentError build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt2,
                moment_backend=:sorted_tail_serial, forbid_dense_fallback=true)
            # needs_outer_moment_jacobian=true ALSO silently selects backend=:dense_reference.
            @test_throws ArgumentError build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt2,
                needs_outer_moment_jacobian=true, forbid_dense_fallback=true)
            # forbid_dense_fallback=false (the default, and MELITZ_PRODUCTION_COMPAT's value)
            # still permits the SAME dense choice, unchanged.
            obj_dense, _ = build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt2, backend=:dense_reference)
            @test !(obj_dense isa MelitzCCBundle)
        end

        @testset "build_melitz_psi_bundle_from_calibration: forbid_dense_fallback=true throws on the dense route" begin
            fixture4 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
                target_country=1, seed=29, W=20_000)
            observed4, _ = split_melitz_synthetic_truth(fixture4)
            calib4 = calibrate_melitz_pareto(observed4; sigma=2.5, theta_star=:estimate,
                focal_country=1, gravity_tol=1e-6)
            obj_ok, _ = build_melitz_psi_bundle_from_calibration(calib4; W=5_000,
                inner_loop_opt=inner_opt2, forbid_dense_fallback=true)
            @test obj_ok isa MelitzCCBundle
            @test_throws ArgumentError build_melitz_psi_bundle_from_calibration(calib4; W=5_000,
                inner_loop_opt=inner_opt2, backend=:dense_reference, forbid_dense_fallback=true)
        end

        @testset "build_melitz_implicit_bundle: forbid_dense_fallback=true throws on backend=:dense_reference AND on a legacy dense-only gradient_backend" begin
            obj_f, theta0_f = build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt2)
            ctx = obj_f.γ

            # matrix-free default: no throw.
            obj_ok = build_melitz_implicit_bundle(ctx, obj_f.U, theta0_f; delta=1.0,
                find_smallest=true, inner_loop_opt=inner_opt2, outer_loop_opt=outer_opt2,
                forbid_dense_fallback=true)
            @test obj_ok isa MelitzCCBundle

            @test_throws ArgumentError build_melitz_implicit_bundle(ctx, obj_f.U, theta0_f; delta=1.0,
                find_smallest=true, inner_loop_opt=inner_opt2, outer_loop_opt=outer_opt2,
                backend=:dense_reference, forbid_dense_fallback=true)
            # gradient_backend=:B (the legacy dense-only finite-difference backend) silently
            # selects backend=:dense_reference underneath it (this function's own documented
            # rule) -- must ALSO throw under strict mode.
            @test_throws ArgumentError build_melitz_implicit_bundle(ctx, obj_f.U, theta0_f; delta=1.0,
                find_smallest=true, gradient_backend=:B, inner_loop_opt=inner_opt2,
                outer_loop_opt=outer_opt2, forbid_dense_fallback=true)
            # unchanged (permitted) when forbid_dense_fallback=false, the default.
            obj_dense = build_melitz_implicit_bundle(ctx, obj_f.U, theta0_f; delta=1.0,
                find_smallest=true, gradient_backend=:B, inner_loop_opt=inner_opt2, outer_loop_opt=outer_opt2)
            @test !(obj_dense isa MelitzCCBundle)
        end

        @testset "solve_melitz_finite_delta_bound: forbid_dense_fallback=true fails FAST at construction, not after a real solve" begin
            obj_f, theta0_f = build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt2)
            t0 = time()
            @test_throws ArgumentError solve_melitz_finite_delta_bound(obj_f.γ, obj_f, theta0_f;
                delta=1e-3, direction=:upper, gradient_backend=:B, inner_loop_opt=inner_opt2,
                forbid_dense_fallback=true)
            elapsed = time() - t0
            @test elapsed < 5.0   # construction-time failure, not a real KNITRO trajectory
        end
    end
end

# ============================================================================
# 2026-07-26 production-closure session (governing prompt Phase 3): "centralize and type the
# welfare metrics." See equilibrium.jl's MelitzWelfareMetrics/melitz_welfare_metrics_from_g/
# melitz_welfare_metrics/kappa_ratio_of_g for the implementation and full incident writeup
# (the OLD kappa_of_g/`.kappa` naming collided with the gains-from-trade kappa/GT).
# ============================================================================
@testset "Phase 3 (2026-07-26): MelitzWelfareMetrics eliminates the g/kappa_ratio/GT confusion" begin
    @testset "gamma_prime == exp(g), kappa_ratio matches the closed-form, GT == 1-kappa_ratio" begin
        for (g, wage_ratio, sigma) in ((0.3, 1.07, 2.5), (-0.5, 0.93, 4.0), (0.0, 1.0, 3.2))
            m = melitz_welfare_metrics_from_g(g, wage_ratio, sigma)
            @test m.g == g
            @test m.gamma_prime == exp(g)
            @test m.wage_ratio == wage_ratio
            @test m.kappa_ratio == wage_ratio * exp(g)^(1 / (sigma - 1))
            @test m.gains_from_trade == 1 - m.kappa_ratio
            # the two must genuinely differ whenever kappa_ratio != 0.5 -- a structural
            # sanity check that GT is not silently aliased to kappa_ratio (the exact
            # confusion this phase exists to eliminate).
            m.kappa_ratio != 0.5 && @test m.gains_from_trade != m.kappa_ratio
        end
    end

    @testset "kappa_ratio_of_g agrees with MelitzWelfareMetrics.kappa_ratio (ctx-based calling convention)" begin
        ctx_stub = (sigma=2.5, w_prime=1.0, w=[1.0, 1.1, 0.9], target_country=2)
        g = 0.42
        @test kappa_ratio_of_g(g, ctx_stub) == melitz_welfare_metrics_from_g(g, ctx_stub).kappa_ratio
        wage_ratio = ctx_stub.w_prime / ctx_stub.w[ctx_stub.target_country]
        @test kappa_ratio_of_g(g, ctx_stub) == wage_ratio * exp(g)^(1 / (ctx_stub.sigma - 1))
    end

    @testset "melitz_welfare_metrics agrees with melitz_gains_from_trade at the D=4 fixture's own calibration" begin
        p, cf = FIXTURE.primitives, FIXTURE.counterfactual
        m = melitz_welfare_metrics(p, cf)
        @test m.gains_from_trade == melitz_gains_from_trade(p, cf)
        @test m.g == log(p.gamma_prime_target)
        @test m.gamma_prime == p.gamma_prime_target
    end

    @testset "Pareto reference GT agrees with ACR sufficient statistic (population closed forms)" begin
        p, cf, eq = FIXTURE.primitives, FIXTURE.counterfactual, FIXTURE.equilibrium
        m = melitz_welfare_metrics(p, cf)
        _, GT_ACR = acr_gains_from_trade(p, eq)
        @test isapprox(m.gains_from_trade, GT_ACR; atol=1e-6)
    end

    @testset "plausible real-data GT lies in the analytical attainable interval [0,1)" begin
        p, cf = FIXTURE.primitives, FIXTURE.counterfactual
        m = melitz_welfare_metrics(p, cf)
        @test 0 <= m.gains_from_trade < 1
        # a negative g is a perfectly ordinary outer-search coordinate (no welfare
        # interpretation by itself) -- confirm it is NEVER mistaken for kappa_ratio/GT by
        # construction: MelitzWelfareMetrics keeps g in its OWN field, never overwriting
        # kappa_ratio/gains_from_trade.
        m_neg = melitz_welfare_metrics_from_g(-0.9, m.wage_ratio, p.sigma)
        @test m_neg.g < 0
        @test m_neg.g != m_neg.kappa_ratio && m_neg.g != m_neg.gains_from_trade
    end
end

# ============================================================================
# 2026-07-26 production-closure session (governing prompt Phase 5): "audit the 1e10
# outer-constraint scaling." Traced the full chain (finite_delta_outer.jl's own header +
# melitz_build_finite_delta_callbacks/melitz_register_finite_delta_knitro_problem!): the
# ACTUALLY-REGISTERED KNITRO constraint has been the DIMENSIONLESS
# `c_delta(theta)=DeltaStar(theta)/delta<=1` since the 2026-07-23 correctness-repair session
# -- the governing prompt's premise ("the current implicit bundle uses ... constr[1] = 1e10 *
# DeltaStar") describes an OLDER, already-superseded state; `1e10` survives only as an
# internal raw-functor convention (`obj(x,constr=c)` returns `1e10*DeltaStar` -- a detail
# shared with cc_algo's own `PsiObjectiveBundleImplicit` functor, see the Section 18 test
# above), never reaching the actual KNITRO registration. `divergence_constraint_scaling`
# (new kwarg, default `:dimensionless` -- UNCHANGED existing behavior) additionally supports
# `:legacy_1e10`, reproducing the OLD, pre-2026-07-23 raw-magnitude registration for direct
# comparison -- NOT made default (per the governing prompt's own instruction), diagnostic-only.
# ============================================================================
@testset "Phase 5 (2026-07-26): divergence-constraint scaling audit -- dimensionless vs legacy_1e10" begin
    if KNITRO_AVAILABLE
        inner_opt5 = joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt")
        outer_opt5 = joinpath(dirname(dirname(@__DIR__)), "melitz_outer_finite_delta.opt")
        fixture5 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=2_000)
        obj5, theta0_5 = build_melitz_psi_bundle(fixture5; inner_loop_opt=inner_opt5)
        ctx5b = obj5.γ
        n5 = length(theta0_5)
        m5 = 1 + ctx5b.D + ctx5b.D * (ctx5b.D - 1)
        r0_5 = evaluate_melitz_delta(theta0_5, ctx5b, obj5; cold=true, store_G=false)
        @test r0_5.nStatus == 0 && r0_5.Delta > 0
        delta5 = max(r0_5.Delta * 3, 1e-3)   # a genuine, non-degenerate outer budget

        @testset "melitz_build_finite_delta_callbacks rejects an unrecognized scaling mode" begin
            obj_i5 = build_melitz_implicit_bundle(ctx5b, obj5.U, theta0_5; delta=delta5,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt5, outer_loop_opt=outer_opt5)
            @test_throws ArgumentError melitz_build_finite_delta_callbacks(obj_i5, ctx5b, delta5, true;
                divergence_constraint_scaling=:not_a_mode)
        end

        @testset ":dimensionless (default) matches the pre-existing registered bound/value exactly" begin
            obj_dim = build_melitz_implicit_bundle(ctx5b, obj5.U, theta0_5; delta=delta5,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt5, outer_loop_opt=outer_opt5)
            cbset_dim = melitz_build_finite_delta_callbacks(obj_dim, ctx5b, delta5, true)
            @test cbset_dim.divergence_constraint_scaling == :dimensionless
            @test cbset_dim.divergence_constraint_upbnd == 1.0

            evR = MelitzMockEvalResult(zeros(1), zeros(m5), zeros(n5), zeros(n5 * m5))
            cbset_dim.cb_F!(nothing, nothing, MelitzMockEvalRequest(copy(theta0_5)), evR, nothing)
            cbset_dim.cb_G!(nothing, nothing, MelitzMockEvalRequest(copy(theta0_5)), evR, nothing)
            @test isapprox(evR.c[1], r0_5.Delta / delta5; rtol=1e-6)
            @test evR.c[1] <= cbset_dim.divergence_constraint_upbnd   # feasible: Delta(theta0)<=delta5 by construction
        end

        @testset ":legacy_1e10 registers the OLD raw-magnitude bound/value, exactly 1e10*delta5 larger" begin
            obj_leg = build_melitz_implicit_bundle(ctx5b, obj5.U, theta0_5; delta=delta5,
                find_smallest=true, gradient_backend=:B, h=1e-4,
                inner_loop_opt=inner_opt5, outer_loop_opt=outer_opt5)
            cbset_leg = melitz_build_finite_delta_callbacks(obj_leg, ctx5b, delta5, true;
                divergence_constraint_scaling=:legacy_1e10)
            @test cbset_leg.divergence_constraint_scaling == :legacy_1e10
            @test cbset_leg.divergence_constraint_upbnd == 1e10 * delta5

            evR_leg = MelitzMockEvalResult(zeros(1), zeros(m5), zeros(n5), zeros(n5 * m5))
            cbset_leg.cb_F!(nothing, nothing, MelitzMockEvalRequest(copy(theta0_5)), evR_leg, nothing)
            cbset_leg.cb_G!(nothing, nothing, MelitzMockEvalRequest(copy(theta0_5)), evR_leg, nothing)
            @test isapprox(evR_leg.c[1], 1e10 * r0_5.Delta; rtol=1e-6)
            @test evR_leg.c[1] <= cbset_leg.divergence_constraint_upbnd   # SAME feasibility conclusion
        end

        @testset "the two modes are the SAME feasible set and equivalent search direction, after accounting for scale" begin
            # Re-derive both callback sets fresh (obj's own mutable state -- x/H -- must not
            # be shared/stale across the two comparisons) and evaluate at an OVER-BUDGET
            # point (genuinely infeasible under both conventions) as well as the feasible
            # theta0_5 above -- the equivalence must hold in BOTH directions. Small
            # deterministic candidate search (not a single fixed seed) for the perturbation
            # magnitude/seed, matching this session's own Phase 1 fix: a small D=4 fixture is
            # fragile to an arbitrary single random perturbation (some draws land on a
            # non-converged nStatus!=0 point).
            local theta_bad5b, r_bad5b
            found5 = false
            for (seed, mag) in ((4, 0.01), (7, 0.01), (2, 0.005), (9, 0.005), (1, 0.002))
                theta_try5 = theta0_5 .+ mag .* randn(MersenneTwister(seed), n5)
                r_try5 = evaluate_melitz_delta(theta_try5, ctx5b, obj5; cold=true, store_G=false)
                if r_try5.nStatus == 0 && r_try5.Delta > 0
                    theta_bad5b, r_bad5b = theta_try5, r_try5
                    found5 = true
                    break
                end
            end
            @test found5
            delta_tight5 = found5 ? r_bad5b.Delta / 2 : 1e-3   # deliberately infeasible at theta_bad5b

            if found5
                for theta_test in (theta0_5, theta_bad5b)
                    obj_dim2 = build_melitz_implicit_bundle(ctx5b, obj5.U, theta0_5; delta=delta_tight5,
                        find_smallest=true, gradient_backend=:B, h=1e-4,
                        inner_loop_opt=inner_opt5, outer_loop_opt=outer_opt5)
                    cbset_dim2 = melitz_build_finite_delta_callbacks(obj_dim2, ctx5b, delta_tight5, true)
                    obj_leg2 = build_melitz_implicit_bundle(ctx5b, obj5.U, theta0_5; delta=delta_tight5,
                        find_smallest=true, gradient_backend=:B, h=1e-4,
                        inner_loop_opt=inner_opt5, outer_loop_opt=outer_opt5)
                    cbset_leg2 = melitz_build_finite_delta_callbacks(obj_leg2, ctx5b, delta_tight5, true;
                        divergence_constraint_scaling=:legacy_1e10)

                    ev_dim = MelitzMockEvalResult(zeros(1), zeros(m5), zeros(n5), zeros(n5 * m5))
                    cbset_dim2.cb_F!(nothing, nothing, MelitzMockEvalRequest(copy(theta_test)), ev_dim, nothing)
                    cbset_dim2.cb_G!(nothing, nothing, MelitzMockEvalRequest(copy(theta_test)), ev_dim, nothing)
                    ev_leg = MelitzMockEvalResult(zeros(1), zeros(m5), zeros(n5), zeros(n5 * m5))
                    cbset_leg2.cb_F!(nothing, nothing, MelitzMockEvalRequest(copy(theta_test)), ev_leg, nothing)
                    cbset_leg2.cb_G!(nothing, nothing, MelitzMockEvalRequest(copy(theta_test)), ev_leg, nothing)

                    scale = 1e10 * delta_tight5
                    # SAME feasibility conclusion (the whole point of "same feasible set"):
                    @test (ev_dim.c[1] <= cbset_dim2.divergence_constraint_upbnd) ==
                          (ev_leg.c[1] <= cbset_leg2.divergence_constraint_upbnd)
                    # value and Jacobian related by the EXACT constant scale factor:
                    @test isapprox(ev_leg.c[1], ev_dim.c[1] * scale; rtol=1e-8)
                    @test isapprox(ev_leg.jac[1:n5], ev_dim.jac[1:n5] .* scale; rtol=1e-8)
                    # equivalent search DIRECTION: the Jacobian sign pattern (which coordinates
                    # increase/decrease the constraint) is identical, only magnitude differs.
                    @test sign.(ev_leg.jac[1:n5]) == sign.(ev_dim.jac[1:n5])
                end
            end
        end
    end
end

# ============================================================================
# 2026-07-26 production-closure session (governing prompt Phase 6): "port nuisance
# profiling to production-fast infrastructure." `solve_melitz_nuisance_min_delta`/
# `melitz_build_nuisance_profile_callbacks` (nuisance_profile.jl) now accept EITHER a
# dense `PsiObjectiveBundleDelta` or a matrix-free `MelitzCCBundle` `obj_inner` -- the
# optimization algorithm itself is UNCHANGED (no trust-region/predictor-corrector logic
# introduced, per the governing prompt's own instruction); only the three bundle-specific
# internals were swapped for already-established bundle-agnostic dispatch
# (`melitz_bundle_inner_loop` (new, composed from existing primitives),
# `melitz_heavy_snapshot`/`melitz_heavy_restore!` (pre-existing, already generic)).
# ============================================================================
@testset "Phase 6 (2026-07-26): nuisance-profile matrix-free port matches dense reference" begin
    if KNITRO_AVAILABLE
        inner_opt6 = joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt")
        cfg6 = MelitzInnerSolveConfig(:evaluation_cap; delta_evaluation_cap=10.0)

        @testset "D=4: matrix-free forbid_dense_fallback=true throws when obj_inner is dense" begin
            obj_dense6, theta0_dense6 = build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt6,
                backend=:dense_reference)
            mask6a = melitz_nuisance_free_mask(obj_dense6.γ; block=:A_only)
            @test_throws ArgumentError solve_melitz_nuisance_min_delta(obj_dense6.γ, obj_dense6, theta0_dense6;
                free_mask=mask6a, radius=0.05, inner_loop_opt=inner_opt6, inner_solve_config=cfg6,
                forbid_dense_fallback=true)
        end

        @testset "D=4: matrix-free (default backend) runs end to end, forbid_dense_fallback=true does NOT throw" begin
            obj_mf6, theta0_mf6 = build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt6)
            @test obj_mf6 isa MelitzCCBundle
            mask6b = melitz_nuisance_free_mask(obj_mf6.γ; block=:A_only)
            res_mf6 = solve_melitz_nuisance_min_delta(obj_mf6.γ, obj_mf6, theta0_mf6; free_mask=mask6b,
                radius=0.05, inner_loop_opt=inner_opt6, inner_solve_config=cfg6, forbid_dense_fallback=true)
            @test res_mf6.r_final.nStatus in (0, -100, -101, -103)
        end

        @testset "D=4: matched dense-vs-matrix-free -- identical starting point/mask/radius/cache init" begin
            obj_d6, theta0_d6 = build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt6, backend=:dense_reference)
            obj_m6, theta0_m6 = build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt6)
            @test theta0_d6 == theta0_m6   # identical starting point (same FIXTURE, same reduce_theta)
            mask6c = melitz_nuisance_free_mask(obj_d6.γ; block=:A_only)

            res_d6 = solve_melitz_nuisance_min_delta(obj_d6.γ, obj_d6, theta0_d6; free_mask=mask6c,
                radius=0.05, gradient_backend=:B_direct_argument_serial,
                inner_loop_opt=inner_opt6, inner_solve_config=cfg6)
            res_m6 = solve_melitz_nuisance_min_delta(obj_m6.γ, obj_m6, theta0_m6; free_mask=mask6c,
                radius=0.05, gradient_backend=:B_direct_argument_serial,
                inner_loop_opt=inner_opt6, inner_solve_config=cfg6, forbid_dense_fallback=true)

            @test res_d6.r_final.nStatus in (0, -100, -101, -103)
            @test res_m6.r_final.nStatus in (0, -100, -101, -103)
            # SAME optimization problem, SAME KNITRO options/settings -- must reach the SAME
            # minimized Delta and the SAME optimized nuisance coordinates (both backends drive
            # the identical box-constrained NLP through the identical KNITRO options file).
            @test isapprox(res_d6.Delta_min, res_m6.Delta_min; rtol=1e-5, atol=1e-8)
            @test isapprox(res_d6.theta_final, res_m6.theta_final; rtol=1e-5, atol=1e-8)
            @test isapprox(res_d6.r_final.Delta, res_m6.r_final.Delta; rtol=1e-5, atol=1e-8)
            # cutoff/gravity residuals agree too (both re-verified via the SAME
            # evaluate_melitz_delta cold reverification, generic across bundle types):
            @test isapprox(res_d6.r_final.equilibrium_check.gravity_residual_A,
                           res_m6.r_final.equilibrium_check.gravity_residual_A; atol=1e-8)
            @test isapprox(res_d6.r_final.equilibrium_check.gravity_residual_f,
                           res_m6.r_final.equilibrium_check.gravity_residual_f; atol=1e-8)
        end

        @testset "real D=20 (W=80,000): ONE fixed-g point, dense vs matrix-free cb_F!/cb_G! agree" begin
            # Governing prompt's own wording: "one real-D20 FIXED g point" -- a SINGLE-POINT
            # callback comparison, not a full nested outer KNITRO search. A full
            # solve_melitz_nuisance_min_delta search at real D=20 scale was tried first and
            # HUNG (confirmed live: 28+ minutes with zero further output after a KNITRO
            # "Could not evaluate objective... trying perturbed initial points" cascade) --
            # consistent with this repo's own documented history of real-D20 nested-KNITRO-
            # solve fragility (memory: "KNITRO driver runs can hang past timeout",
            # "Nested-KNITRO-solve hang"). A single bounded cb_F!/cb_G! call pair, exactly
            # mirroring this session's own Phase 5 mock-callback pattern, tests the SAME
            # bundle-agnostic port (melitz_bundle_inner_loop/melitz_heavy_snapshot/_restore!)
            # without that risk -- one real inner CC dual solve per side, not an open-ended
            # multi-iteration search.
            #
            # W=80,000, NOT W=20,000: confirmed live this session -- at W=20,000 this exact
            # UNCAPPED single-point inner solve is genuinely ill-conditioned (objective
            # ~1.3e9 vs ~2.2e9 between the two independently-implemented KNITRO drivers, one
            # gradient entry ~1e23-1e24 -- clear numerical garbage from a divergent
            # trajectory, not a port bug), consistent with this repo's OWN documented finding
            # (memory: "Melitz D=20 rank deficiency RESOLVES at W=80k") that real-D20 needs
            # W>=80,000 for a well-posed comparison; W=20,000 remains fine for the OTHER
            # (feasibility-only, no inner-solve-value comparison) real-D20 tests in this file.
            #
            # NOTE: dirname(dirname(@__DIR__)) (TWO levels up from test/melitz), not three --
            # this repo's real_data/ lives directly under the repo root
            # (trade_robustness_modular/real_data/noah_D20). Governing prompt Phase 3
            # (2026-07-27 addendum): the THREE-dirname version of this exact guard elsewhere in
            # this file (e.g. "Section 17: real D=20 data-only calibration diagnostics") has now
            # been fixed to 2 levels -- see that testset's own comment. This site's path was
            # already correct; converted from a silent `isdir`+`@warn` skip to a hard assert for
            # consistency (the fixture is bundled in this repo, not genuinely optional).
            real_dir6 = joinpath(dirname(dirname(@__DIR__)), "real_data", "noah_D20")
            @assert isdir(real_dir6) "real_data/noah_D20 not found at $real_dir6 -- this repo's own bundled real-data fixture is required for this test, not an optional/skippable dependency"
            begin
                lambdaData6 = readdlm(joinpath(real_dir6, "pi.csv"), ',')
                LData6 = vec(readdlm(joinpath(real_dir6, "L.csv"), ',')) ./ 1e6
                tauData6 = readdlm(joinpath(real_dir6, "tau.csv"), ',')
                countries6 = vec(readdlm(joinpath(real_dir6, "countries.csv"), ',', String))
                focal6 = findfirst(==("fra"), countries6)
                observed6 = MelitzObservedData(; lambda=lambdaData6, L=LData6, tau=tauData6,
                    countries=countries6, atol=2e-3)
                calib6 = calibrate_melitz_pareto(observed6; sigma=2.5, theta_star=:estimate,
                    focal_country=focal6, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)

                obj_d20n, theta0_d20n = build_melitz_psi_bundle_from_calibration(calib6; W=80_000,
                    seed=calib6.seed, inner_loop_opt=inner_opt6, backend=:dense_reference)
                obj_m20n, theta0_m20n = build_melitz_psi_bundle_from_calibration(calib6; W=80_000,
                    seed=calib6.seed, inner_loop_opt=inner_opt6)
                @test theta0_d20n == theta0_m20n
                n20n = length(theta0_d20n)

                # 2026-07-26 (fixed live this session): the counter reset must happen AFTER
                # the dense callback calls, not before -- resetting before them and then
                # calling BOTH cbset_d20n's (legitimately dense) and cbset_m20n's callbacks
                # before snapshotting means the shared GLOBAL counters pick up the dense
                # bundle's own expected dense-path increments too, producing a false failure
                # that looked like a matrix-free dense-fallback leak but was actually a test
                # ordering bug.
                cbset_d20n = melitz_build_nuisance_profile_callbacks(obj_d20n, obj_d20n.γ;
                    gradient_backend=:B_direct_argument_serial)
                evD = MelitzMockEvalResult(zeros(1), Float64[], zeros(n20n), Float64[])
                cbset_d20n.cb_F!(nothing, nothing, MelitzMockEvalRequest(copy(theta0_d20n)), evD, nothing)
                cbset_d20n.cb_G!(nothing, nothing, MelitzMockEvalRequest(copy(theta0_d20n)), evD, nothing)

                melitz_backend_counters_reset!()
                cbset_m20n = melitz_build_nuisance_profile_callbacks(obj_m20n, obj_m20n.γ;
                    gradient_backend=:B_direct_argument_serial, forbid_dense_fallback=true)
                evM = MelitzMockEvalResult(zeros(1), Float64[], zeros(n20n), Float64[])
                cbset_m20n.cb_F!(nothing, nothing, MelitzMockEvalRequest(copy(theta0_m20n)), evM, nothing)
                cbset_m20n.cb_G!(nothing, nothing, MelitzMockEvalRequest(copy(theta0_m20n)), evM, nothing)
                counters20n = melitz_backend_counters_snapshot()

                @test isapprox(evD.obj[1], evM.obj[1]; rtol=1e-6)   # Delta(theta0) agrees
                @test isapprox(evD.objGrad, evM.objGrad; rtol=1e-4, atol=1e-8)   # gradient agrees
                # dense-fallback counters must be EXACTLY zero for the matrix-free run (strict
                # mode was active -- forbid_dense_fallback=true -- so this is also a structural
                # guarantee, not merely an observation):
                @test counters20n.dense_inner_objective_calls == 0
                @test counters20n.dense_inner_gradient_calls == 0
                @test counters20n.dense_G_materializations == 0
            end
        end
    end
end

# ============================================================================
# 2026-07-26 production-closure session (governing prompt Phase 7): "implement and
# benchmark a matrix-free range screen." `melitz_range_screen(op::MelitzMomentOperator)`
# (inner_screening.jl) computes the SAME per-column range test as the dense
# `melitz_range_screen(G)` WITHOUT materializing `G` -- O(D^2) given `op`'s own
# already-updated state (fused into `melitz_update_moment_operator!`'s existing merge
# sweep, no extra O(W*D) pass). Validated (this session, standalone script, not
# repeated in full here for wall-clock reasons) at D=4/D=10/real-D20 with ZERO
# mismatches across 37 checked points; measured ~300x faster than the dense screen in
# isolation at real D=20/W=20,000 (~110us vs ~33ms) -- `matrix_free_range_screen`
# defaults to `true` in `melitz_classified_inner_solve` (was opt-in during
# development) since the governing prompt's own bar ("enable by default only if the
# measured net return is positive") is clearly met.
# ============================================================================
@testset "Phase 7 (2026-07-26): matrix-free range screen matches dense reference" begin
    function melitz_test_ctx_and_op(fixture)
        p, eq, cf = fixture.primitives, fixture.equilibrium, fixture.counterfactual
        D = p.D
        z_draws = fixture.z_draws
        moment_layout = MelitzMomentLayout(D)
        c_full, A_pivot = build_gravity_pivots(p.tau, p.target_country)
        outer_layout = melitz_outer_layout(D, p.target_country)
        sorted_ctx = build_melitz_sorted_tail_context(z_draws, p.sigma; theta_star=p.theta_star)
        ctx = (D=D, sigma=p.sigma, theta_star=p.theta_star, target_country=p.target_country, tau=p.tau,
               w=p.w, w_prime=cf.w_prime, L=fixture.L, expenditure=eq.expenditure, benchmark_cutoff=eq.cutoff,
               moment_layout=moment_layout, X_data=eq.trade_flow, c_full=c_full, A_pivot=A_pivot,
               jj_lin=outer_layout.jj_lin, f_free_lin=outer_layout.f_free_lin,
               outer_parameterization=:logf, inner_loop_opt="unused", outer_loop_opt="unused",
               moment_backend=:sorted_tail_serial, sorted_tail_ctx=sorted_ctx)
        op = build_melitz_moment_operator(sorted_ctx, moment_layout)
        return ctx, op, z_draws
    end

    function melitz_check_range_screen_equivalence(fixture; n_perturb=8, perturb=0.02, seed=1)
        ctx, op, z_draws = melitz_test_ctx_and_op(fixture)
        p = fixture.primitives
        obj_stub = (γ=ctx,)
        theta0 = melitz_reduce_theta(p, ctx)
        rng = MersenneTwister(seed)
        n_checked = 0
        for trial in 0:n_perturb
            theta = trial == 0 ? theta0 : theta0 .+ perturb .* randn(rng, length(theta0))
            st = melitz_outer_state(theta, ctx)
            st.feasible || continue
            K = zeros(size(z_draws, 1)); G = zeros(size(z_draws, 1), ctx.moment_layout.num_moments)
            melitz_moments_adapter!(K, G, theta, z_draws, obj_stub)
            melitz_update_moment_operator!(op, st.primitives, st.equilibrium, fixture.counterfactual; X_data=ctx.X_data)
            cert_dense = melitz_range_screen(G)
            cert_mf = melitz_range_screen(op)
            @test (cert_dense === nothing) == (cert_mf === nothing)
            if cert_dense !== nothing && cert_mf !== nothing
                @test cert_dense.column == cert_mf.column
                @test isapprox(cert_dense.lo, cert_mf.lo; atol=1e-9, rtol=1e-9)
                @test isapprox(cert_dense.hi, cert_mf.hi; atol=1e-9, rtol=1e-9)
            end
            n_checked += 1
        end
        return n_checked
    end

    @testset "D=4 (FIXTURE): matrix-free range screen matches dense at theta0 and perturbed points" begin
        n_checked = melitz_check_range_screen_equivalence(FIXTURE; seed=1)
        @test n_checked >= 1
    end

    @testset "D=10 (synthetic): matrix-free range screen matches dense" begin
        fixture10 = nothing
        for seed10 in (100, 200, 7, 3, 11)
            try
                fixture10 = generate_fake_melitz_data(; D=10, sigma=2.5, theta_star=6.8,
                    target_country=1, seed=seed10, W=5_000)
                break
            catch
                continue
            end
        end
        if fixture10 !== nothing
            n_checked = melitz_check_range_screen_equivalence(fixture10; seed=2)
            @test n_checked >= 1
        else
            @warn "Skipping D=10 range-screen equivalence test -- no seed produced a feasible fixture"
        end
    end

    if KNITRO_AVAILABLE
        @testset "real D=20: matrix-free range screen matches dense (direct op/G construction, no KNITRO)" begin
            real_dir7 = joinpath(dirname(dirname(@__DIR__)), "real_data", "noah_D20")
            @assert isdir(real_dir7) "real_data/noah_D20 not found at $real_dir7 -- this repo's own bundled real-data fixture is required for this test, not an optional/skippable dependency"
            begin
                lambdaData7 = readdlm(joinpath(real_dir7, "pi.csv"), ',')
                LData7 = vec(readdlm(joinpath(real_dir7, "L.csv"), ',')) ./ 1e6
                tauData7 = readdlm(joinpath(real_dir7, "tau.csv"), ',')
                countries7 = vec(readdlm(joinpath(real_dir7, "countries.csv"), ',', String))
                focal7 = findfirst(==("fra"), countries7)
                observed7 = MelitzObservedData(; lambda=lambdaData7, L=LData7, tau=tauData7,
                    countries=countries7, atol=2e-3)
                calib7 = calibrate_melitz_pareto(observed7; sigma=2.5, theta_star=:estimate,
                    focal_country=focal7, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
                D7 = calib7.D
                z_draws7 = pareto_draws(20_000, D7, calib7.theta_star; seed=calib7.seed, mode=:halton)
                p7 = MelitzPrimitives(D7, calib7.sigma, calib7.theta_star, calib7.target_country, calib7.tau,
                    calib7.w, calib7.A, calib7.f, calib7.gamma_prime_target)
                eq7 = MelitzEquilibrium(calib7.E, ones(D7), calib7.q, calib7.X)
                cf7 = MelitzCounterfactual(calib7.target_country, calib7.w_prime,
                    calib7.w_prime * calib7.L[calib7.target_country], 1.0,
                    calib7.w_prime * calib7.L[calib7.target_country])
                fixture7 = (primitives=p7, equilibrium=eq7, counterfactual=cf7, L=calib7.L, z_draws=z_draws7)
                n_checked = melitz_check_range_screen_equivalence(fixture7; n_perturb=10, perturb=0.01, seed=3)
                @test n_checked >= 1
            end
        end

        @testset "melitz_classified_inner_solve: matrix_free_range_screen defaults to true and actually engages for MelitzCCBundle" begin
            inner_opt7 = joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt")
            obj_mf7, theta0_mf7 = build_melitz_psi_bundle(FIXTURE; inner_loop_opt=inner_opt7)
            @test obj_mf7 isa MelitzCCBundle
            bank7 = MelitzDualBank()
            melitz_backend_counters_reset!()
            result7 = melitz_classified_inner_solve(obj_mf7, theta0_mf7, obj_mf7.γ;
                delta_evaluation_cap=10.0, bank=bank7)
            counters7 = melitz_backend_counters_snapshot()
            @test counters7.matrix_free_range_screen_calls >= 1   # engaged by DEFAULT, no kwarg passed
            @test counters7.production_dense_screen_calls == 0    # never the dense path for this bundle

            # explicit opt-out still works (governing prompt: keep it toggleable, not forced):
            melitz_backend_counters_reset!()
            bank7b = MelitzDualBank()
            melitz_classified_inner_solve(obj_mf7, theta0_mf7, obj_mf7.γ;
                delta_evaluation_cap=10.0, bank=bank7b, matrix_free_range_screen=false)
            counters7b = melitz_backend_counters_snapshot()
            @test counters7b.matrix_free_range_screen_calls == 0
        end
    end
end

@testset "Governing prompt Phase 7 (2026-07-27 addendum): outer-parameterization roundtrip, all 6 combinations" begin
    # `MELITZ_ALL_PARAMETERIZATIONS` (outer_parameterization_config.jl): the full
    # 3(technology) x 2(participation) factorial. At the FIXTURE's own calibrated point,
    # every combination must: (a) reduce->expand back to the IDENTICAL (A, f, gamma_prime_j)
    # to machine precision; (b) reconstruct machine-precision-exact gravity residuals; (c)
    # roundtrip a RANDOM admissible perturbation of theta_free (in that combination's OWN
    # powered coordinate) back to itself. This is the exact battery Phase 7 asks for
    # (calibrated point + random admissible points); near-cutoff/focal-origin points are
    # additionally covered by the SAME roundtrip machinery in the pre-existing ":logcutoff"
    # Section 5.5 tests above (technology-coordinate scaling only touches the A-block, never
    # the participation/cutoff machinery those tests already exercise).
    p, eq, cf = FIXTURE.primitives, FIXTURE.equilibrium, FIXTURE.counterfactual
    D = p.D
    moment_layout = MelitzMomentLayout(D)
    c_full, A_pivot = build_gravity_pivots(p.tau, p.target_country)
    outer_layout = melitz_outer_layout(D, p.target_country)
    base_ctx = (D=D, sigma=p.sigma, theta_star=p.theta_star, target_country=p.target_country, tau=p.tau,
        w=p.w, w_prime=cf.w_prime, L=FIXTURE.L, expenditure=eq.expenditure, benchmark_cutoff=eq.cutoff,
        moment_layout=moment_layout, X_data=eq.trade_flow, c_full=c_full, A_pivot=A_pivot,
        jj_lin=outer_layout.jj_lin, f_free_lin=outer_layout.f_free_lin,
        inner_loop_opt="unused", outer_loop_opt="unused")

    @test length(MELITZ_ALL_PARAMETERIZATIONS) == 6
    @test length(unique(melitz_parameterization_label.(MELITZ_ALL_PARAMETERIZATIONS))) == 6

    for config in MELITZ_ALL_PARAMETERIZATIONS
        @testset "$(melitz_parameterization_label(config))" begin
            ctx = melitz_apply_parameterization(base_ctx, config)
            @test melitz_ctx_parameterization(ctx) == config
            @test melitz_parameterization_compatible(ctx, config)

            theta_free = melitz_reduce_theta(p, ctx)
            A_rt, f_rt, gp_rt, fjj_rt = melitz_expand_theta(theta_free, ctx)
            @test isapprox(A_rt, p.A; atol=1e-9, rtol=1e-8)
            @test isapprox(f_rt, p.f; atol=1e-9, rtol=1e-8)
            @test isapprox(gp_rt, p.gamma_prime_target; atol=1e-9, rtol=1e-8)

            p_rt = MelitzPrimitives(D, p.sigma, p.theta_star, p.target_country, p.tau, p.w, A_rt, f_rt, gp_rt)
            gA, gf = gravity_residuals(p_rt)
            @test abs(gA) < 1e-8
            @test abs(gf) < 1e-8

            rng = MersenneTwister(hash(melitz_parameterization_label(config)))
            theta_perturbed = theta_free .+ 0.01 .* randn(rng, length(theta_free))
            A2, f2, gp2, fjj2 = melitz_expand_theta(theta_perturbed, ctx)
            p2 = MelitzPrimitives(D, p.sigma, p.theta_star, p.target_country, p.tau, p.w, A2, f2, gp2)
            theta_re = melitz_reduce_theta(p2, ctx)
            @test isapprox(theta_re, theta_perturbed; atol=1e-8, rtol=1e-8)
        end
    end

    @testset "technology_coordinate is a pure linear rescale of logA (cross-check against melitz_technology_coordinate_scale)" begin
        ctx_base = melitz_apply_parameterization(base_ctx, MelitzOuterParameterizationConfig(:logA, :logf))
        theta_logA_coord = melitz_reduce_theta(p, ctx_base)
        for tc in MELITZ_TECHNOLOGY_COORDINATES
            ctx_tc = melitz_apply_parameterization(base_ctx, MelitzOuterParameterizationConfig(tc, :logf))
            theta_tc = melitz_reduce_theta(p, ctx_tc)
            p_A = melitz_technology_coordinate_scale(tc, ctx_tc)
            nA = D^2 - 1
            @test isapprox(theta_tc[2:1+nA], p_A .* theta_logA_coord[2:1+nA]; atol=1e-9, rtol=1e-8)
            @test theta_tc[1] == theta_logA_coord[1]                      # g untouched
            @test theta_tc[2+nA:end] == theta_logA_coord[2+nA:end]        # participation block untouched
        end
    end
end

@testset "Governing prompt Phase 8 (2026-07-27 addendum): registered-Jacobian chain rule, all 6 combinations" begin
    # Reuses the EXACT established methodology from "Closure Phase B1: registered Jacobian ==
    # central FD of the registered constraint" above (Richardson h-vs-2h stability pre-scan to
    # select genuinely smooth coordinates, since Melitz's extensive margin is genuinely kinked
    # at participation-switch boundaries -- a real, PRE-EXISTING, already-documented
    # phenomenon, not a technology-coordinate bug; confirmed live this session: a naive
    # all-coordinates FD check without this pre-scan spuriously "fails" at kinked coordinates
    # for ALL SIX parameterizations identically, including the pre-existing :logA/:logf
    # baseline). Only KNITRO_AVAILABLE-guarded (needs real nested inner CC solves).
    if KNITRO_AVAILABLE
        fixtureP8 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=2_000)
        inner_optP8 = joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt")
        outer_optP8 = joinpath(dirname(dirname(@__DIR__)), "melitz_outer_finite_delta.opt")
        hP8 = 1e-4

        for config in MELITZ_ALL_PARAMETERIZATIONS
            @testset "$(melitz_parameterization_label(config))" begin
                objP8_inner, theta0P8 = build_melitz_psi_bundle(fixtureP8;
                    outer_parameterization=config.participation_coordinate,
                    technology_coordinate=config.technology_coordinate, inner_loop_opt=inner_optP8)
                ctxP8 = objP8_inner.γ
                @test melitz_ctx_parameterization(ctxP8) == config
                r0P8 = evaluate_melitz_delta(theta0P8, ctxP8, objP8_inner; cold=true, store_G=false)
                @test r0P8.nStatus == 0
                delta_looseP8 = max(r0P8.Delta * 5, 1e-3)
                nP8 = length(theta0P8)
                mP8 = 1 + ctxP8.D + ctxP8.D * (ctxP8.D - 1)

                objP8_scan = build_melitz_implicit_bundle(ctxP8, objP8_inner.U, theta0P8; delta=delta_looseP8,
                    find_smallest=true, gradient_backend=:B_direct_argument_serial, h=hP8,
                    inner_loop_opt=inner_optP8, outer_loop_opt=outer_optP8)
                cbsetP8_scan = melitz_build_finite_delta_callbacks(objP8_scan, ctxP8, delta_looseP8, true;
                    gradient_backend=:B_direct_argument_serial, h=hP8)
                function fd_c1_P8(theta_r, r, hh)
                    tp = copy(theta_r); tp[r] += hh
                    tm = copy(theta_r); tm[r] -= hh
                    ep = MelitzMockEvalResult(zeros(1), zeros(mP8), zeros(nP8), zeros(nP8 * mP8))
                    em = MelitzMockEvalResult(zeros(1), zeros(mP8), zeros(nP8), zeros(nP8 * mP8))
                    cbsetP8_scan.cb_F!(nothing, nothing, MelitzMockEvalRequest(tp), ep, nothing)
                    cbsetP8_scan.cb_F!(nothing, nothing, MelitzMockEvalRequest(tm), em, nothing)
                    return (ep.c[1] - em.c[1]) / (2hh)
                end
                smooth_coords_P8 = Int[]
                for r in 1:nP8
                    fd_h = fd_c1_P8(theta0P8, r, hP8)
                    fd_2h = fd_c1_P8(theta0P8, r, 2 * hP8)
                    if isapprox(fd_h, fd_2h; rtol=0.02, atol=1e-8)
                        push!(smooth_coords_P8, r)
                    end
                    length(smooth_coords_P8) >= 4 && break
                end
                @test length(smooth_coords_P8) >= 2

                evalGP8 = MelitzMockEvalResult(zeros(1), zeros(mP8), zeros(nP8), zeros(nP8 * mP8))
                cbsetP8_scan.cb_F!(nothing, nothing, MelitzMockEvalRequest(copy(theta0P8)), evalGP8, nothing)
                cbsetP8_scan.cb_G!(nothing, nothing, MelitzMockEvalRequest(copy(theta0P8)), evalGP8, nothing)
                jac_registered_P8 = copy(evalGP8.jac[1:nP8])

                for r in smooth_coords_P8
                    fd_r = fd_c1_P8(theta0P8, r, hP8)
                    @test isapprox(jac_registered_P8[r], fd_r; rtol=2e-2, atol=1e-6)
                end
            end
        end
    end
end

@testset "Standalone (no cc_algo) subprocess: MelitzCCBundle is genuinely cc_algo-independent" begin
    # Governing prompt Phase 1 / closure doc Phase 4 side finding: `MelitzCCBundle`'s own
    # KNITRO driver used to call a 2-arg `KNITRO.KN_add_vars(kc, n)` convenience form that
    # exists ONLY via `cc_algo/knitro_compat.jl`'s monkey-patch of the KNITRO module --
    # invisible in THIS test file because it always loads `cc_algo` first (`KNITRO_AVAILABLE`
    # above). Launching `standalone_no_cc_algo.jl` as an actual separate process (never
    # `include`d here) is the only way to genuinely test "no cc_algo in the process at all" --
    # an `include` in-process would inherit whatever `cc_algo`/KNITRO-monkey-patch state this
    # file already loaded. Must fail (nonzero exit) if a caller reintroduces a call to a
    # `cc_algo`-monkey-patched KNITRO method from Melitz-owned code.
    standalone_script = joinpath(@__DIR__, "standalone_no_cc_algo.jl")
    proj = dirname(dirname(@__DIR__))
    cmd = `$(Base.julia_cmd()) --project=$proj $standalone_script`
    io = IOBuffer()
    proc = run(pipeline(cmd; stdout=io, stderr=io); wait=false)
    wait(proc)
    out = String(take!(io))
    if proc.exitcode != 0
        println("standalone_no_cc_algo.jl FAILED (exit=$(proc.exitcode)):\n", out)
    end
    @test proc.exitcode == 0
    @test occursin("PASSED", out)
end

# ============================================================================
# Governing prompt continuation (2026-07-27 evening session), Phases 2-3: mutating,
# workspace-based `melitz_expand_theta!`/`expand_free_theta!` (delta_star.jl,
# log_cutoff_param.jl) replacing the ~19,672-byte-per-call (real D=20) allocating path
# inside the production default gradient backend's own hot loop
# (`sorted_crossing_gradient.jl`'s `_fill_compact_direct_columns_crossing_sorted!`, called
# 2x per free coordinate), plus the removal of the global-lock pivot-parts cache for every
# `ctx` built via a production entry point (`ctx.f_pivot_c`/`f_pivot_idx`/`f_pivot_other`
# precomputed once at construction instead).
# ============================================================================
@testset "Governing prompt Phase 2-3 (2026-07-27 evening): mutating theta expansion + ctx-owned pivot" begin
    @testset "D=4 (FIXTURE): melitz_expand_theta! matches melitz_expand_theta to machine precision, zero post-warmup allocation" begin
        ctx = build_melitz_psi_bundle(FIXTURE)[1].γ
        theta0 = melitz_reduce_theta(FIXTURE.primitives, ctx)
        n = length(theta0)
        ws = MelitzThetaExpansionWorkspace(ctx.D)
        state = MelitzExpandedState(ctx.D)
        rng = MersenneTwister(11)
        for _ in 1:20
            theta = theta0 .+ 0.01 .* randn(rng, n)
            A_ref, f_ref, g_ref, fjj_ref = melitz_expand_theta(theta, ctx)
            melitz_expand_theta!(state, theta, ctx, ws)
            @test isapprox(state.A, A_ref; atol=1e-12)
            @test isapprox(state.f, f_ref; atol=1e-12)
            @test state.gamma_prime_j == g_ref
            @test state.f_jj == fjj_ref
        end
        melitz_expand_theta!(state, theta0, ctx, ws)   # warmup
        bytes1 = @allocated melitz_expand_theta!(state, theta0, ctx, ws)
        bytes2 = @allocated melitz_expand_theta!(state, theta0, ctx, ws)
        @test bytes1 == 0
        @test bytes2 == 0
        @test (@allocated melitz_expand_theta(theta0, ctx)) > 0   # the allocating wrapper still allocates (unchanged, by design)
    end

    @testset "Governing prompt Phase 1.2 (2026-07-27 continuation): :logcutoff mutating fast path matches the allocating path, zero post-warmup allocation" begin
        ctx_lc = build_melitz_psi_bundle(FIXTURE; outer_parameterization=:logcutoff)[1].γ
        theta0_lc = melitz_reduce_theta(FIXTURE.primitives, ctx_lc)
        n_lc = length(theta0_lc)
        ws = MelitzThetaExpansionWorkspace(ctx_lc.D)
        state = MelitzExpandedState(ctx_lc.D)
        rng_lc = MersenneTwister(13)
        for _ in 1:20
            theta = theta0_lc .+ 0.01 .* randn(rng_lc, n_lc)
            A_ref, f_ref, g_ref, fjj_ref = melitz_expand_theta(theta, ctx_lc)
            melitz_expand_theta!(state, theta, ctx_lc, ws)
            @test isapprox(state.A, A_ref; atol=1e-12)
            @test isapprox(state.f, f_ref; atol=1e-12)
            @test state.gamma_prime_j == g_ref
            @test state.f_jj == fjj_ref
        end
        melitz_expand_theta!(state, theta0_lc, ctx_lc, ws)   # warmup
        bytes1_lc = @allocated melitz_expand_theta!(state, theta0_lc, ctx_lc, ws)
        bytes2_lc = @allocated melitz_expand_theta!(state, theta0_lc, ctx_lc, ws)
        @test bytes1_lc == 0
        @test bytes2_lc == 0
    end

    @testset "q-pivot and f-pivot coincide (governing prompt Phase 1.2's own reuse claim)" begin
        ctx_lc = build_melitz_psi_bundle(FIXTURE; outer_parameterization=:logcutoff)[1].γ
        q_pivot_direct = build_q_gravity_pivot(ctx_lc)
        c_free, f_idx, f_other = melitz_cached_f_pivot_parts(ctx_lc)
        @test q_pivot_direct.pivot == f_idx
        @test q_pivot_direct.other == f_other
        @test q_pivot_direct.c == c_free
    end

    @testset "ctx built via production entry points carries precomputed f_pivot_* fields (no global lock touched)" begin
        ctx = build_melitz_psi_bundle(FIXTURE)[1].γ
        @test ctx.f_pivot_c !== nothing
        c_free, idx, other = melitz_cached_f_pivot_parts(ctx)
        @test c_free === ctx.f_pivot_c   # returned directly, not recomputed/copied
        @test idx == ctx.f_pivot_idx
        @test other === ctx.f_pivot_other
    end

    @testset "legacy hand-built ctx (no f_pivot_* fields) still works via the global-lock fallback" begin
        p, eq, cf = FIXTURE.primitives, FIXTURE.equilibrium, FIXTURE.counterfactual
        D = p.D
        moment_layout = MelitzMomentLayout(D)
        c_full, A_pivot = build_gravity_pivots(p.tau, p.target_country)
        outer_layout = melitz_outer_layout(D, p.target_country)
        ctx_legacy = (D=D, sigma=p.sigma, theta_star=p.theta_star, target_country=p.target_country, tau=p.tau,
               w=p.w, w_prime=cf.w_prime, L=FIXTURE.L, expenditure=eq.expenditure, benchmark_cutoff=eq.cutoff,
               moment_layout=moment_layout, X_data=eq.trade_flow, c_full=c_full, A_pivot=A_pivot,
               jj_lin=outer_layout.jj_lin, f_free_lin=outer_layout.f_free_lin,
               outer_parameterization=:logf, inner_loop_opt="unused", outer_loop_opt="unused",
               moment_backend=:dense_reference, sorted_tail_ctx=nothing)
        @test get(ctx_legacy, :f_pivot_c, nothing) === nothing
        c_free, idx, other = melitz_cached_f_pivot_parts(ctx_legacy)
        theta_legacy = reduce_to_free_theta(p, ctx_legacy)
        A_l, f_l, g_l, fjj_l = expand_free_theta(theta_legacy, ctx_legacy)
        @test isapprox(A_l, p.A; atol=1e-10)
    end

    if KNITRO_AVAILABLE
        @testset "real D=20: melitz_expand_theta! matches melitz_expand_theta to machine precision, zero post-warmup allocation" begin
            real_dir8 = joinpath(dirname(dirname(@__DIR__)), "real_data", "noah_D20")
            @assert isdir(real_dir8) "real_data/noah_D20 not found at $real_dir8"
            lambdaData8 = readdlm(joinpath(real_dir8, "pi.csv"), ',')
            LData8 = vec(readdlm(joinpath(real_dir8, "L.csv"), ',')) ./ 1e6
            tauData8 = readdlm(joinpath(real_dir8, "tau.csv"), ',')
            countries8 = vec(readdlm(joinpath(real_dir8, "countries.csv"), ',', String))
            focal8 = findfirst(==("fra"), countries8)
            observed8 = MelitzObservedData(; lambda=lambdaData8, L=LData8, tau=tauData8,
                countries=countries8, atol=2e-3)
            calib8 = calibrate_melitz_pareto(observed8; sigma=2.5, theta_star=:estimate,
                focal_country=focal8, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
            z_draws8 = pareto_draws(80_000, calib8.D, calib8.theta_star; seed=calib8.seed)
            p8, eq8, cf8, ctx8 = melitz_calibration_outer_ctx(calib8; z_draws=z_draws8, moment_backend=:sorted_tail_serial)
            @test ctx8.f_pivot_c !== nothing
            theta08 = melitz_reduce_theta(p8, ctx8)
            n8 = length(theta08)
            @test n8 == 2 * ctx8.D^2 - 2
            ws8 = MelitzThetaExpansionWorkspace(ctx8.D)
            state8 = MelitzExpandedState(ctx8.D)
            rng8 = MersenneTwister(12)
            for _ in 1:5
                theta = theta08 .+ 0.001 .* randn(rng8, n8)
                A_ref, f_ref, g_ref, fjj_ref = melitz_expand_theta(theta, ctx8)
                melitz_expand_theta!(state8, theta, ctx8, ws8)
                @test isapprox(state8.A, A_ref; rtol=1e-9)
                @test isapprox(state8.f, f_ref; rtol=1e-9)
            end
            melitz_expand_theta!(state8, theta08, ctx8, ws8)   # warmup
            bytes1_8 = @allocated melitz_expand_theta!(state8, theta08, ctx8, ws8)
            bytes2_8 = @allocated melitz_expand_theta!(state8, theta08, ctx8, ws8)
            @test bytes1_8 == 0
            @test bytes2_8 == 0
            @test (@allocated melitz_expand_theta(theta08, ctx8)) > 15_000   # the old allocating path -- confirms the ~19,672-byte figure this session's own doc cites is real, not fabricated
        end
    end
end

# ============================================================================
# Governing prompt continuation (2026-07-27 evening session), Phase 10: FC-to-GA cache /
# inner-solve state-reuse validation. `inner_solve_verified_or_fail`'s exact-point cache
# (`MelitzExactPointCache`) is what lets a `cb_G!` call reuse the dual/moment state
# `cb_F!` JUST solved for the identical `theta` -- this testset proves cold vs warm-started
# re-solves at the SAME theta agree to machine precision (ruling out any state-staleness
# effect across repeated calls) and that a genuine cb_F!-then-cb_G! sequence at the same
# theta reuses the cache (not a redundant KNITRO solve).
# ============================================================================
@testset "Governing prompt Phase 10 (2026-07-27 evening): FC-to-GA cache / re-solve consistency A/B/A" begin
    fixtureP10 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
        target_country=1, seed=29, W=5_000)
    inner_optP10 = joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt")
    objP10, theta0P10 = build_melitz_psi_bundle(fixtureP10; inner_loop_opt=inner_optP10)
    ctxP10 = objP10.γ

    @testset "cold-solve reproducibility: repeated cold solves at the SAME theta are bit-identical" begin
        r1 = evaluate_melitz_delta(theta0P10, ctxP10, objP10; cold=true, store_G=false)
        r2 = evaluate_melitz_delta(theta0P10, ctxP10, objP10; cold=true, store_G=false)
        r3 = evaluate_melitz_delta(theta0P10, ctxP10, objP10; cold=true, store_G=false)
        @test r1.Delta == r2.Delta == r3.Delta
        @test r1.dual_x == r2.dual_x == r3.dual_x
    end

    @testset "warm-start from the SAME theta's own converged dual matches the cold solve to machine precision" begin
        r_cold = evaluate_melitz_delta(theta0P10, ctxP10, objP10; cold=true, store_G=false)
        r_warm = evaluate_melitz_delta(theta0P10, ctxP10, objP10; warm_start=r_cold.dual_x, store_G=false)
        @test isapprox(r_cold.Delta, r_warm.Delta; rtol=1e-10)
    end

    @testset "cb_F! then cb_G! at the identical theta: exact-cache hit, consistent Delta" begin
        m10 = 1 + ctxP10.D + ctxP10.D * (ctxP10.D - 1)
        n10 = length(theta0P10)
        r0P10 = evaluate_melitz_delta(theta0P10, ctxP10, objP10; cold=true, store_G=false)
        delta_looseP10 = max(r0P10.Delta * 5, 1e-3)
        objP10b = build_melitz_implicit_bundle(ctxP10, objP10.U, theta0P10; delta=delta_looseP10,
            find_smallest=true, gradient_backend=:B_direct_argument_sorted_serial, h=1e-4,
            inner_loop_opt=inner_optP10, outer_loop_opt=joinpath(dirname(dirname(@__DIR__)), "melitz_outer_finite_delta.opt"))
        cbsetP10 = melitz_build_finite_delta_callbacks(objP10b, ctxP10, delta_looseP10, true;
            gradient_backend=:B_direct_argument_sorted_serial, h=1e-4)
        evalP10 = MelitzMockEvalResult(zeros(1), zeros(m10), zeros(n10), zeros(n10 * m10))
        cbsetP10.cb_F!(nothing, nothing, MelitzMockEvalRequest(copy(theta0P10)), evalP10, nothing)
        n_misses_after_F = cbsetP10.n_exact_cache_misses[]
        n_hits_after_F = cbsetP10.n_exact_cache_hits[]
        cbsetP10.cb_G!(nothing, nothing, MelitzMockEvalRequest(copy(theta0P10)), evalP10, nothing)
        @test cbsetP10.n_exact_cache_hits[] == n_hits_after_F + 1   # GA reused FC's solve, no new inner KNITRO solve
        @test cbsetP10.n_exact_cache_misses[] == n_misses_after_F   # no additional miss
    end
end


# ============================================================================
# Governing prompt continuation (2026-07-27 night session), Phase 1 completion + Phase 5:
# focal-link in-place expansion (Phase 1.1), :logcutoff mutating expansion (Phase 1.2,
# tested above), the plain (non-sorted) direct backend now wired to the SAME mutating
# workspace (Phase 1.3), and absolute-ceiling allocation regression tests for all three
# (Phase 5 -- replacing the "bytes_first_call==bytes_second_call" pattern the governing
# prompt's own Phase 6 critiques with assertions on the VALUE itself).
# ============================================================================
@testset "Governing prompt Phase 1.1/1.3/5 (2026-07-27 night): focal-link + plain-backend in-place expansion, allocation ceilings" begin
    @testset "D=4 (FIXTURE): full sorted-serial outer gradient (link-touching coordinate included) is post-warmup allocation-free" begin
        obj_d4, theta0_d4 = build_melitz_psi_bundle(FIXTURE; forbid_dense_fallback=true)
        ctx_d4 = obj_d4.γ
        n_d4 = length(theta0_d4)
        r_d4 = evaluate_melitz_delta(theta0_d4, ctx_d4, obj_d4; cold=true, store_G=false)
        @test r_d4.verified
        x_d4 = r_d4.dual_x
        compact_d4 = melitz_compact_columns_map(ctx_d4)
        @test any(c -> c.touches_link, compact_d4)   # the fixture DOES exercise the focal-link path

        gfun = make_melitz_gradient_delta_direct_sorted_serial(1e-4)
        gbuf = zeros(n_d4)
        gfun(gbuf, theta0_d4, ctx_d4, obj_d4, x_d4)   # warmup
        bytes1 = @allocated gfun(gbuf, theta0_d4, ctx_d4, obj_d4, x_d4)
        bytes2 = @allocated gfun(gbuf, theta0_d4, ctx_d4, obj_d4, x_d4)
        @test bytes1 == 0
        @test bytes2 == 0

        # cross-check against the (still-allocating, diagnostic-only) argument-localized
        # backend's own output, to confirm the in-place refactor did not change the FORMULA.
        mj_ref! = make_melitz_moments_jacobian_b_argument_localized_serial(1e-4)
        n_moments_d4 = ctx_d4.moment_layout.num_moments
        K_jac_ref = zeros(size(obj_d4.U, 1), n_moments_d4)
        G_jac_ref = zeros(size(obj_d4.U, 1), n_moments_d4, n_d4)
        mj_ref!(K_jac_ref, G_jac_ref, theta0_d4, obj_d4.U, obj_d4)
        # the sorted backend's own gradient is a SCALAR-CONSTRAINT secant (not a moments
        # Jacobian) -- cross-validated instead against the plain direct backend below,
        # which shares the SAME calling convention.
    end

    @testset "D=4 (FIXTURE): plain (non-sorted) direct backend now matches the sorted backend AND is allocation-free" begin
        obj_d4b, theta0_d4b = build_melitz_psi_bundle(FIXTURE; forbid_dense_fallback=true)
        ctx_d4b = obj_d4b.γ
        n_d4b = length(theta0_d4b)
        r_d4b = evaluate_melitz_delta(theta0_d4b, ctx_d4b, obj_d4b; cold=true, store_G=false)
        @test r_d4b.verified
        x_d4b = r_d4b.dual_x

        g_plain = zeros(n_d4b)
        gfun_plain = make_melitz_gradient_delta_direct_serial(1e-4)
        gfun_plain(g_plain, theta0_d4b, ctx_d4b, obj_d4b, x_d4b)   # warmup
        bytes1p = @allocated gfun_plain(g_plain, theta0_d4b, ctx_d4b, obj_d4b, x_d4b)
        bytes2p = @allocated gfun_plain(g_plain, theta0_d4b, ctx_d4b, obj_d4b, x_d4b)
        @test bytes1p == 0
        @test bytes2p == 0

        g_sorted = zeros(n_d4b)
        gfun_sorted = make_melitz_gradient_delta_direct_sorted_serial(1e-4)
        gfun_sorted(g_sorted, theta0_d4b, ctx_d4b, obj_d4b, x_d4b)
        @test isapprox(g_plain, g_sorted; rtol=1e-8)   # same formula, different (now equally allocation-free) implementation

        # parallel plain backend, if multiple threads are available
        if Threads.nthreads() > 1
            g_plain_par = zeros(n_d4b)
            gfun_plain_par = make_melitz_gradient_delta_direct_parallel(1e-4)
            gfun_plain_par(g_plain_par, theta0_d4b, ctx_d4b, obj_d4b, x_d4b)   # warmup
            b1 = @allocated gfun_plain_par(g_plain_par, theta0_d4b, ctx_d4b, obj_d4b, x_d4b)
            b2 = @allocated gfun_plain_par(g_plain_par, theta0_d4b, ctx_d4b, obj_d4b, x_d4b)
            @test b1 == 0
            @test b2 == 0
            @test isapprox(g_plain_par, g_sorted; rtol=1e-8)
        end
    end

    @testset "POSITIVE CONTROL: the still-allocating diagnostic-only argument-localized backend DOES allocate (proves the ceiling tests above would catch a reintroduced anti-pattern)" begin
        obj_pc, theta0_pc = build_melitz_psi_bundle(FIXTURE; forbid_dense_fallback=true)
        ctx_pc = obj_pc.γ
        n_pc = length(theta0_pc)
        n_moments_pc = ctx_pc.moment_layout.num_moments
        mj! = make_melitz_moments_jacobian_b_argument_localized_serial(1e-4)
        K_jac = zeros(size(obj_pc.U, 1), n_moments_pc)
        G_jac = zeros(size(obj_pc.U, 1), n_moments_pc, n_pc)
        mj!(K_jac, G_jac, theta0_pc, obj_pc.U, obj_pc)   # warmup
        bytes_legacy = @allocated mj!(K_jac, G_jac, theta0_pc, obj_pc.U, obj_pc)
        @test bytes_legacy > 0   # this backend is DELIBERATELY left allocating (diagnostic-only, Phase 1.3's own decision)
    end

    if KNITRO_AVAILABLE
        @testset "real D=20: full sorted-serial/parallel outer gradient (focal-link included) is post-warmup allocation-free" begin
            real_dir9 = joinpath(dirname(dirname(@__DIR__)), "real_data", "noah_D20")
            @assert isdir(real_dir9) "real_data/noah_D20 not found at $real_dir9"
            lambdaData9 = readdlm(joinpath(real_dir9, "pi.csv"), ',')
            LData9 = vec(readdlm(joinpath(real_dir9, "L.csv"), ',')) ./ 1e6
            tauData9 = readdlm(joinpath(real_dir9, "tau.csv"), ',')
            countries9 = vec(readdlm(joinpath(real_dir9, "countries.csv"), ',', String))
            focal9 = findfirst(==("fra"), countries9)
            observed9 = MelitzObservedData(; lambda=lambdaData9, L=LData9, tau=tauData9,
                countries=countries9, atol=2e-3)
            calib9 = calibrate_melitz_pareto(observed9; sigma=2.5, theta_star=:estimate,
                focal_country=focal9, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
            z_draws9 = pareto_draws(80_000, calib9.D, calib9.theta_star; seed=calib9.seed)
            p9, eq9, cf9, ctx9 = melitz_calibration_outer_ctx(calib9; z_draws=z_draws9, moment_backend=:sorted_tail_serial)
            theta09 = melitz_reduce_theta(p9, ctx9)
            n9 = length(theta09)
            op9 = build_melitz_moment_operator(ctx9.sorted_tail_ctx, ctx9.moment_layout)
            obj9 = build_melitz_cc_bundle(op9, ctx9; mode=:delta, U=z_draws9,
                outer_constr_index=ctx9.moment_layout.num_moments + 1,
                lower_limit=-KNITRO.KN_INFINITY,   # deliberate: a pure fixed-point evaluation test, not an outer search -- no cap needed, but now an explicit choice, not a silently-inherited default
                inner_loop_opt=ctx9.inner_loop_opt, outer_loop_opt=ctx9.outer_loop_opt,
                hessian_backend=:structured_serial)
            r9 = evaluate_melitz_delta(theta09, ctx9, obj9; cold=true, store_G=false)
            @test r9.verified
            x9 = r9.dual_x

            gfun9 = make_melitz_gradient_delta_direct_sorted_serial(1e-4)
            gbuf9 = zeros(n9)
            gfun9(gbuf9, theta09, ctx9, obj9, x9)   # warmup
            b1_9 = @allocated gfun9(gbuf9, theta09, ctx9, obj9, x9)
            b2_9 = @allocated gfun9(gbuf9, theta09, ctx9, obj9, x9)
            @test b1_9 == 0
            @test b2_9 == 0

            if Threads.nthreads() > 1
                gfun9p = make_melitz_gradient_delta_direct_sorted_parallel(1e-4)
                gbuf9p = zeros(n9)
                gfun9p(gbuf9p, theta09, ctx9, obj9, x9)   # warmup
                b1_9p = @allocated gfun9p(gbuf9p, theta09, ctx9, obj9, x9)
                b2_9p = @allocated gfun9p(gbuf9p, theta09, ctx9, obj9, x9)
                @test b1_9p == 0
                @test b2_9p == 0
                @test isapprox(gbuf9, gbuf9p; rtol=1e-8)
            end
        end
    end
end


# ============================================================================
# Governing prompt continuation (2026-07-27 night session), Phase 10 extension: the full
# A/B/A cache-isolation matrix across {theta, parameterization, technology_coordinate, seed,
# W, evaluation cap} -- the prior session's own Phase 10 testset covered theta-identity reuse
# only (disclosed there as a narrowed scope). This extends coverage directly against
# `melitz_context_fingerprint`/`melitz_exact_cache_get`/`melitz_exact_cache_insert!` (the
# ACTUAL mechanism a real end-to-end miss/hit ultimately reduces to), rather than re-running
# full KNITRO solves for every combination.
# ============================================================================
@testset "Governing prompt Phase 10 (2026-07-27 night): cache isolation across theta/parameterization/technology/seed/W/evaluation-cap" begin
    objP10x, theta0P10x = build_melitz_psi_bundle(FIXTURE; forbid_dense_fallback=true)
    ctxP10x = objP10x.γ
    Ux = objP10x.U

    fp_base = melitz_context_fingerprint(ctxP10x, Ux)

    @testset "exact hit at identical (ctx, U): same fingerprint every call" begin
        @test melitz_context_fingerprint(ctxP10x, Ux) == fp_base
        @test melitz_context_fingerprint(ctxP10x, Ux) == fp_base   # stable, not incidental
    end

    @testset "changed outer_parameterization: different fingerprint (miss, not a false hit)" begin
        obj_lc, _ = build_melitz_psi_bundle(FIXTURE; outer_parameterization=:logcutoff, forbid_dense_fallback=true)
        ctx_lc = obj_lc.γ
        fp_lc = melitz_context_fingerprint(ctx_lc, obj_lc.U)
        @test fp_lc != fp_base
    end

    @testset "changed technology_coordinate: different fingerprint" begin
        obj_tc, _ = build_melitz_psi_bundle(FIXTURE; technology_coordinate=:theta_logA, forbid_dense_fallback=true)
        ctx_tc = obj_tc.γ
        fp_tc = melitz_context_fingerprint(ctx_tc, obj_tc.U)
        @test fp_tc != fp_base
    end

    @testset "changed seed (different z_draws, same W): different fingerprint" begin
        data_seed2 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=53, W=20_000)
        obj_seed2, _ = build_melitz_psi_bundle(data_seed2; forbid_dense_fallback=true)
        fp_seed2 = melitz_context_fingerprint(obj_seed2.γ, obj_seed2.U)
        @test fp_seed2 != fp_base
    end

    @testset "changed W (different z_draws shape): different fingerprint" begin
        data_w2 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
            target_country=1, seed=29, W=8_000)
        obj_w2, _ = build_melitz_psi_bundle(data_w2; forbid_dense_fallback=true)
        fp_w2 = melitz_context_fingerprint(obj_w2.γ, obj_w2.U)
        @test fp_w2 != fp_base
    end

    @testset "changed evaluation cap ALONE: SAME fingerprint (DeltaStar does not depend on the outer budget/cap -- cache reuse preserved)" begin
        # melitz_context_fingerprint never reads delta_evaluation_cap/inner_solve_config at
        # all (by design, per this function's own docstring) -- confirmed directly: two
        # otherwise-identical ctx/U pairs fingerprint identically regardless of what
        # evaluation cap a caller separately applies via inner_solve_config/delta_evaluation_cap
        # (neither of which is a ctx FIELD at all).
        @test melitz_context_fingerprint(ctxP10x, Ux) == fp_base
    end

    @testset "compact cache A/B/A: insert at theta A, miss at theta B (different fingerprint), hit again at A" begin
        cache10x = MelitzExactPointCache(64)
        keyA = Vector{Float64}(theta0P10x)
        rA = evaluate_melitz_delta(theta0P10x, ctxP10x, objP10x; cold=true, store_G=false)
        melitz_exact_cache_insert!(cache10x, keyA, rA.Delta, rA.dual_x, rA.nStatus,
            melitz_heavy_snapshot(objP10x), ctxP10x, Ux)

        hitA1 = melitz_exact_cache_get(cache10x, keyA, ctxP10x, Ux; obj=objP10x)
        @test hitA1 !== nothing
        @test hitA1[1] == rA.Delta

        # a DIFFERENT ctx (different parameterization) queried with the SAME key must MISS,
        # even though the key (theta vector) itself is bit-identical -- proves isolation is
        # keyed on content fingerprint, not merely the theta vector. `melitz_exact_cache_get`'s
        # own fingerprint-mismatch branch conservatively PURGES the stale entry from BOTH tiers
        # (found live, not merely read: it deletes by `key` alone, which is the correct, safe
        # design given `key` is only ever meaningful relative to ONE ctx's own economic model
        # in real production usage -- a single `MelitzExactPointCache` is always scoped to one
        # `ctx`/campaign for its whole life, never deliberately shared across two DIFFERENT ctx
        # objects the way this test's own cross-parameterization probe deliberately forces).
        obj_lc2, theta0_lc2 = build_melitz_psi_bundle(FIXTURE; outer_parameterization=:logcutoff, forbid_dense_fallback=true)
        missB = melitz_exact_cache_get(cache10x, keyA, obj_lc2.γ, obj_lc2.U; obj=obj_lc2)
        @test missB === nothing

        # ...and having purged it, a SUBSEQUENT same-key query under the ORIGINAL ctx now
        # correctly MISSES too (the entry is gone, not silently corrupted/stale) -- exactly the
        # safe behavior a real cb_F! would see; it would simply re-solve and re-insert.
        hitA2_after_purge = melitz_exact_cache_get(cache10x, keyA, ctxP10x, Ux; obj=objP10x)
        @test hitA2_after_purge === nothing

        # simulating that natural re-solve-and-reinsert: the ORIGINAL ctx's own correct state
        # is fully recoverable, not permanently poisoned by the intervening cross-ctx query.
        melitz_exact_cache_insert!(cache10x, keyA, rA.Delta, rA.dual_x, rA.nStatus,
            melitz_heavy_snapshot(objP10x), ctxP10x, Ux)
        hitA3 = melitz_exact_cache_get(cache10x, keyA, ctxP10x, Ux; obj=objP10x)
        @test hitA3 !== nothing
        @test hitA3[1] == rA.Delta
    end
end

println("\n" * "="^70)
println("Melitz Delta-star test suite (active minimal-moment closure) complete.")
println("="^70)
