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
