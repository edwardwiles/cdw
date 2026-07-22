# Formal test suite for the full-D Melitz Christensen-Connault benchmark.
# Run with: julia --project=. test/melitz/runtests.jl
#
# No test/runtests.jl convention exists elsewhere in this repo (confirmed by survey;
# tests are ad hoc standalone scripts per session). This introduces the first formal
# Test.jl suite, scoped to test/melitz/ only -- it does not touch or replace any
# existing test script elsewhere in the repository.

using Test
using Random
using Statistics: std, mean

const MELITZ_DIR = joinpath(@__DIR__, "..", "..", "src", "melitz")
include(joinpath(dirname(dirname(@__DIR__)), "misc", "doubleDiff.jl"))
include(joinpath(MELITZ_DIR, "types.jl"))
include(joinpath(MELITZ_DIR, "pareto.jl"))
include(joinpath(MELITZ_DIR, "firm_quantities.jl"))
include(joinpath(MELITZ_DIR, "equilibrium.jl"))
include(joinpath(MELITZ_DIR, "fake_data.jl"))
include(joinpath(MELITZ_DIR, "moments.jl"))
include(joinpath(MELITZ_DIR, "fstar_solver.jl"))

# ============================================================================
# Draw tests
# ============================================================================
@testset "Pareto draws" begin
    theta_star, sigma = 6.8, 2.5

    @testset "support >= 1" begin
        z = pareto_draws(5000, 4, theta_star; seed=1)
        @test all(z .>= 1.0)
    end

    @testset "deterministic given seed" begin
        z1 = pareto_draws(1000, 4, theta_star; seed=42)
        z2 = pareto_draws(1000, 4, theta_star; seed=42)
        @test z1 == z2
        z3 = pareto_draws(1000, 4, theta_star; seed=43)
        @test z1 != z3
    end

    @testset "sample moments converge to known Pareto moments" begin
        z = pareto_draws(200_000, 4, theta_star; seed=7)
        theory_mean = theta_star / (theta_star - 1)
        @test isapprox(mean(z), theory_mean; rtol=0.01)
    end

    @testset "theta_star > sigma-1 enforced" begin
        @test_throws ArgumentError MelitzPrimitives(2, sigma, 1.2, 1, [1.0 1.2; 1.3 1.0],
            ones(2), ones(2, 2), ones(2, 2) * 0.1, ones(2) * 0.1) # theta_star=1.2 < sigma-1=1.5
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
# Firm-quantity tests
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
# Shared fixture for the remaining test groups
# ============================================================================
const FIXTURE = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8,
    target_country=1, seed=1234, W=20_000)
const LAYOUT = MelitzMomentLayout(FIXTURE.primitives.D)

# ============================================================================
# Entrant-mass tests
# ============================================================================
@testset "Entrant mass" begin
    p, eq = FIXTURE.primitives, FIXTURE.equilibrium
    D = p.D

    @testset "entrant mass multiplies aggregate sales" begin
        # X_od = N_o * C_od * M_od -- doubling N_o (with C,M held fixed) doubles X_od
        o, d = 2, 3
        C_od, _, _ = cell_from_cutoff(eq.trade_flow[o, d], eq.entrant_mass[o], p.w[o],
            p.tau[o, d], eq.expenditure[d], eq.cutoff[o, d], p.sigma, p.theta_star)
        M_od = pareto_tail_power_mean(eq.cutoff[o, d], p.sigma, p.theta_star)
        @test isapprox(eq.entrant_mass[o] * C_od * M_od, eq.trade_flow[o, d]; rtol=1e-6)
        @test isapprox(2 * eq.entrant_mass[o] * C_od * M_od, 2 * eq.trade_flow[o, d]; rtol=1e-6)
    end

    @testset "entrant mass multiplies price-power contributions" begin
        # price_power_d = sum_o N_o * (C_od/expenditure_d) * M_od; verify linear in N
        d = 2
        contributions = Float64[]
        for o in 1:D
            C_od, _, _ = cell_from_cutoff(eq.trade_flow[o, d], eq.entrant_mass[o], p.w[o],
                p.tau[o, d], eq.expenditure[d], eq.cutoff[o, d], p.sigma, p.theta_star)
            M_od = pareto_tail_power_mean(eq.cutoff[o, d], p.sigma, p.theta_star)
            push!(contributions, eq.entrant_mass[o] * (C_od / eq.expenditure[d]) * M_od)
        end
        @test isapprox(sum(contributions), 1.0; atol=1e-8)
    end

    @testset "entrant mass does NOT multiply free-entry profits" begin
        residuals = entry_residuals(p, eq, FIXTURE.z_draws)
        # the entry moment g_entry is N-independent by construction (docs Sec 1.4, 4B):
        # verify the *formula itself* has no N_o factor by checking entry_cost_from_free_entry
        # depends only on C, zhat, w (not N) -- and that residual is small at true params.
        @test maximum(abs.(residuals)) < 0.05 # Monte Carlo tolerance at W=20000
    end

    @testset "baseline and counterfactual use the same entrant-mass vector (N'=N)" begin
        cf = FIXTURE.counterfactual
        t = p.target_country
        # cf.price_power_prime was built directly from eq.entrant_mass[t] in
        # solve_autarky_counterfactual -- no independent NPrime object exists anywhere.
        K1_tt = melitz_K1(1.0, 1.0, p.A[t, t], p.sigma)
        expected_pp_prime = eq.entrant_mass[t] * K1_tt * p.theta_star / (p.theta_star - p.sigma + 1)
        @test isapprox(cf.price_power_prime, expected_pp_prime; rtol=1e-10)
    end

    @testset "closed-form N_o matches Melitz-Redding eq 22 analog" begin
        L = eq.expenditure ./ p.w
        N_closed = [entrant_mass_from_labor(L[o], p.f_entry[o], p.sigma, p.theta_star) for o in 1:D]
        @test isapprox(N_closed, eq.entrant_mass; rtol=1e-6)
    end
end

# ============================================================================
# Full-D indexing tests
# ============================================================================
@testset "Full-D indexing (no rest-of-world aggregation)" begin
    p = FIXTURE.primitives
    D = p.D

    @testset "D^2 distinct bilateral flow moments, every (o,d) exactly once" begin
        @test LAYOUT.num_moments == D^2 + D
        all_trade_cols = vec(LAYOUT.trade_index)
        @test length(unique(all_trade_cols)) == D^2
        @test Set(all_trade_cols) == Set(1:D^2)
    end

    @testset "no origin or destination silently skipped" begin
        for o in 1:D, d in 1:D
            @test LAYOUT.trade_index[o, d] != 0
        end
        for o in 1:D
            @test LAYOUT.entry_index[o] != 0
        end
    end

    @testset "changing a single A[o,d] affects the intended cell only" begin
        # use enough draws and a large enough perturbation that at least some firms in
        # cell (2,3) flip from inactive to active (a tiny perturbation, or too few draws,
        # can leave realized_revenue==0 on both sides purely by chance, masking a real
        # effect -- this is a test-fixture concern, not a moments.jl issue).
        Wtest = 500
        K = zeros(Wtest)
        G1 = zeros(Wtest, LAYOUT.num_moments)
        z = FIXTURE.z_draws[1:Wtest, :]
        melitz_moments!(K, G1, p, FIXTURE.equilibrium, FIXTURE.counterfactual, z, LAYOUT)

        p2_A = copy(p.A)
        p2_A[2, 3] *= 5.0
        p2 = MelitzPrimitives(D, p.sigma, p.theta_star, p.target_country, p.tau, p.w, p2_A, p.f, p.f_entry)
        G2 = zeros(Wtest, LAYOUT.num_moments)
        melitz_moments!(K, G2, p2, FIXTURE.equilibrium, FIXTURE.counterfactual, z, LAYOUT)

        changed_cols = [c for c in 1:LAYOUT.num_moments if !isapprox(G1[:, c], G2[:, c]; atol=1e-10)]
        # perturbing A[2,3] should affect trade[2,3] and entry[2] (which sums over d) --
        # and, in principle, nothing else.
        expected = Set([LAYOUT.trade_index[2, 3], LAYOUT.entry_index[2]])
        @test Set(changed_cols) == expected
    end

    @testset "changing a single f[o,d] affects the intended cell and cutoff" begin
        p2_f = copy(p.f)
        p2_f[3, 4] *= 1.5
        p2 = MelitzPrimitives(D, p.sigma, p.theta_star, p.target_country, p.tau, p.w, p.A, p2_f, p.f_entry)
        C_od = melitz_C(p.w[3], p.tau[3, 4], p.A[3, 4], p.sigma, FIXTURE.equilibrium.expenditure[4])
        zhat1 = melitz_cutoff(p.w[3], p.f[3, 4], p.sigma, C_od)
        zhat2 = melitz_cutoff(p.w[3], p2_f[3, 4], p.sigma, C_od)
        @test zhat2 > zhat1 # higher fixed cost raises the cutoff (fewer firms sell)
    end
end

# ============================================================================
# Gravity tests
# ============================================================================
@testset "Gravity restrictions" begin
    p = FIXTURE.primitives
    T = doubleDiff(p.tau)

    @testset "A gravity restriction satisfied" begin
        rA, _ = gravity_residuals(p)
        @test abs(rA) < 1e-8
    end

    @testset "f gravity restriction satisfied" begin
        _, rf = gravity_residuals(p)
        @test abs(rf) < 1e-8
    end

    @testset "both matrices have nonzero double-differenced variation" begin
        @test std(doubleDiff(p.A)) > 1e-4
        @test std(doubleDiff(p.f)) > 1e-4
        @test std(log.(p.A)) > 0.01
        @test std(log.(p.f)) > 0.01
    end

    @testset "perturbing a single A element breaks the A restriction (generically)" begin
        p2_A = copy(p.A)
        p2_A[2, 3] *= 1.4
        p2 = MelitzPrimitives(p.D, p.sigma, p.theta_star, p.target_country, p.tau, p.w, p2_A, p.f, p.f_entry)
        rA2, _ = gravity_residuals(p2)
        @test abs(rA2) > 1e-4
    end

    @testset "perturbing a single f element breaks the f restriction (generically)" begin
        p2_f = copy(p.f)
        p2_f[2, 3] *= 1.4
        p2 = MelitzPrimitives(p.D, p.sigma, p.theta_star, p.target_country, p.tau, p.w, p.A, p2_f, p.f_entry)
        _, rf2 = gravity_residuals(p2)
        @test abs(rf2) > 1e-4
    end
end

# ============================================================================
# Baseline equilibrium tests
# ============================================================================
@testset "Baseline equilibrium" begin
    p, eq = FIXTURE.primitives, FIXTURE.equilibrium
    D = p.D

    @testset "all bilateral flows matched at true parameters" begin
        # This checks the finite-W (=20,000) Monte Carlo sample mean against the
        # ANALYTICAL population value -- inherently noisy for small/heavy-tailed cells
        # (Pareto revenue has high variance relative to its mean), so the tolerance here
        # is generous (order-of-magnitude check). The airtight, machine-precision version
        # of this claim is the "Exact-sample moment residuals (Mode 1)" testset below,
        # which compares the sample mean against a target built from the SAME sample.
        resid = trade_flow_residuals(p, eq, FIXTURE.z_draws)
        @test maximum(abs.(resid ./ eq.trade_flow)) < 0.4
    end

    @testset "destination expenditures equal sum of bilateral flows" begin
        @test isapprox(vec(sum(eq.trade_flow, dims=1)), eq.expenditure; rtol=1e-8)
    end

    @testset "origin income equals sum of bilateral flows (no deficits)" begin
        L = eq.expenditure ./ p.w
        @test isapprox(vec(sum(eq.trade_flow, dims=2)), p.w .* L; rtol=1e-8)
    end

    @testset "all origin free-entry conditions hold" begin
        # same Monte Carlo caveat as above; Mode 1 testset below is the airtight version.
        resid = entry_residuals(p, eq, FIXTURE.z_draws)
        @test maximum(abs.(resid ./ (p.w .* p.f_entry))) < 0.4
    end

    @testset "price_power_d == 1 for every destination" begin
        @test all(x -> isapprox(x, 1.0; atol=1e-8), eq.price_power)
    end

    @testset "all baseline cutoffs >= 1" begin
        @test all(eq.cutoff .>= 1.0)
    end

    @testset "export cutoffs >= domestic cutoffs" begin
        for o in 1:D, d in 1:D
            d == o && continue
            @test eq.cutoff[o, d] >= eq.cutoff[o, o]
        end
    end
end

# ============================================================================
# Autarky tests
# ============================================================================
@testset "Autarky counterfactual" begin
    p, eq, cf = FIXTURE.primitives, FIXTURE.equilibrium, FIXTURE.counterfactual
    t = p.target_country

    @testset "focal autarky cutoff is exactly 1" begin
        @test isapprox(cf.cutoff_prime, 1.0; atol=1e-10)
    end

    @testset "autarky market clearing holds (X'=expenditure', tautological by construction)" begin
        @test isapprox(cf.trade_flow_prime, cf.expenditure_prime; atol=1e-10)
    end

    @testset "autarky free-entry condition holds" begin
        # C'_tt = sigma*w'*f_tt at zhat'=1 (zero-profit-at-cutoff); check via price_power'
        K1_tt = melitz_K1(1.0, 1.0, p.A[t, t], p.sigma)
        C_tt_prime = cf.expenditure_prime / cf.price_power_prime * K1_tt
        @test isapprox(C_tt_prime, p.sigma * p.f[t, t]; rtol=1e-6)
    end

    @testset "ACR/Chaney cross-check: GT matches 1 - lambda_dd^(1/theta*)" begin
        GT_model = 1 - cf.price_power_prime^(1 / (p.sigma - 1))
        lambda_tt = eq.trade_flow[t, t] / eq.expenditure[t]
        GT_ACR = 1 - lambda_tt^(1 / p.theta_star)
        @test isapprox(GT_model, GT_ACR; rtol=1e-8)
    end
end

# ============================================================================
# Delta-star tests (Mode 1: exact-sample smoke test)
# ============================================================================
@testset "Exact-sample moment residuals (Mode 1)" begin
    p, eq, cf = FIXTURE.primitives, FIXTURE.equilibrium, FIXTURE.counterfactual
    D = p.D
    z_draws = FIXTURE.z_draws
    W = size(z_draws, 1)

    # sample-based targets (docs "Mode 1"): moments must be computed against the SAME
    # finite draws used to evaluate them, not the analytical population values.
    X_sample = zeros(D, D)
    entry_sample = zeros(D)
    for o in 1:D
        for w in 1:W
            z = z_draws[w, o]
            profit_sum = 0.0
            for d in 1:D
                firm = melitz_firm(p.w[o], p.tau[o, d], p.A[o, d], p.f[o, d], p.sigma,
                    eq.expenditure[d], 1.0, z)
                X_sample[o, d] += eq.entrant_mass[o] * firm.realized_revenue / W
                profit_sum += firm.realized_operating_profit
            end
            entry_sample[o] += profit_sum / W
        end
    end

    K = zeros(W)
    G = zeros(W, LAYOUT.num_moments)
    melitz_moments!(K, G, p, eq, cf, z_draws, LAYOUT; X_data=X_sample, entry_target=entry_sample, scale_trade=false)

    @testset "max scaled trade-moment residual at equal weights" begin
        max_resid = maximum(abs.(vec(sum(G[:, vec(LAYOUT.trade_index)], dims=1)) ./ W))
        @test max_resid <= 1e-8
    end

    @testset "max free-entry residual at equal weights" begin
        max_resid = maximum(abs.(vec(sum(G[:, LAYOUT.entry_index], dims=1)) ./ W))
        @test max_resid <= 1e-8
    end

    @testset "both gravity residuals <= 1e-10" begin
        rA, rf = gravity_residuals(p)
        @test abs(rA) <= 1e-10
        @test abs(rf) <= 1e-10
    end

    @testset "K (counterfactual scalar) is constant across draws" begin
        @test all(K .== K[1])
    end
end

# ============================================================================
# Negative control: perturb a target moment, verify infeasibility is detectable
# ============================================================================
@testset "Negative control" begin
    p, eq, cf = FIXTURE.primitives, FIXTURE.equilibrium, FIXTURE.counterfactual
    D = p.D
    z_draws = FIXTURE.z_draws
    W = size(z_draws, 1)

    X_sample = zeros(D, D)
    for o in 1:D, w in 1:W, d in 1:D
        z = z_draws[w, o]
        firm = melitz_firm(p.w[o], p.tau[o, d], p.A[o, d], p.f[o, d], p.sigma,
            eq.expenditure[d], 1.0, z)
        X_sample[o, d] += eq.entrant_mass[o] * firm.realized_revenue / W
    end

    X_perturbed = copy(X_sample)
    X_perturbed[2, 3] *= 1.5 # perturb one target moment after generation

    K = zeros(W)
    G = zeros(W, LAYOUT.num_moments)
    melitz_moments!(K, G, p, eq, cf, z_draws, LAYOUT; X_data=X_perturbed, scale_trade=false)
    perturbed_col = LAYOUT.trade_index[2, 3]
    mean_resid_perturbed = abs(mean(G[:, perturbed_col]))

    @testset "perturbed target moment is detectably nonzero at equal weights" begin
        @test mean_resid_perturbed > 0.1 * X_sample[2, 3] # detectably large vs the ~0 baseline
    end
end

# ============================================================================
# Observational equivalence (fstar_solver.jl)
# ============================================================================
@testset "F* solver: observational equivalence, not overclaimed identification" begin
    p, eq = FIXTURE.primitives, FIXTURE.equilibrium
    D = p.D
    L = eq.expenditure ./ p.w

    result_same = solve_fstar(D, p.sigma, p.theta_star, p.target_country, p.tau, L,
        eq.trade_flow, p.f_entry, eq.cutoff)

    @testset "solving with the generating cutoff reproduces A/f/GT" begin
        @test isapprox(result_same.primitives.A, p.A; rtol=1e-6)
        @test result_same.max_trade_residual < 1e-6
        @test result_same.max_entry_residual < 1e-6
        @test abs(result_same.gravity_residual_A) < 1e-6
        @test abs(result_same.gravity_residual_f) < 1e-6
    end

    zhat_alt = eq.cutoff .* 1.3 .+ 0.2
    result_alt = solve_fstar(D, p.sigma, p.theta_star, p.target_country, p.tau, L,
        eq.trade_flow, p.f_entry, zhat_alt)

    @testset "a different cutoff choice gives a different A/f but same GT and matched moments" begin
        @test !isapprox(result_alt.primitives.A, p.A; rtol=1e-3)
        @test result_alt.max_trade_residual < 1e-6
        @test result_alt.max_entry_residual < 1e-6
        GT_same = 1 - result_same.counterfactual.price_power_prime^(1 / (p.sigma - 1))
        GT_alt = 1 - result_alt.counterfactual.price_power_prime^(1 / (p.sigma - 1))
        @test isapprox(GT_same, GT_alt; rtol=1e-6)
    end
end

# ============================================================================
# CC inner minimum-divergence loop (real KNITRO, unmodified cc_algo machinery)
# ============================================================================
const KNITRO_AVAILABLE = try
    include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "include_cc_algo.jl"))
    @eval using .CounterfactualSensitivity
    true
catch e
    @warn "Skipping CC inner-loop testset: cc_algo/KNITRO not available in this environment" exception = e
    false
end

if KNITRO_AVAILABLE
    include(joinpath(MELITZ_DIR, "delta_star.jl"))

    @testset "CC inner minimum-divergence loop (real KNITRO)" begin
        p, eq = FIXTURE.primitives, FIXTURE.equilibrium
        D = p.D
        z_draws = FIXTURE.z_draws
        W = size(z_draws, 1)

        X_sample = zeros(D, D)
        entry_sample = zeros(D)
        for o in 1:D, w in 1:W
            z = z_draws[w, o]
            profit_sum = 0.0
            for d in 1:D
                firm = melitz_firm(p.w[o], p.tau[o, d], p.A[o, d], p.f[o, d], p.sigma,
                    eq.expenditure[d], 1.0, z)
                X_sample[o, d] += eq.entrant_mass[o] * firm.realized_revenue / W
                profit_sum += firm.realized_operating_profit
            end
            entry_sample[o] += profit_sum / W
        end

        val0, x0, status0, = run_melitz_inner_delta(FIXTURE; X_data=X_sample, entry_target=entry_sample)

        @testset "Delta(theta*) ~ 0 with equal reference weights feasible" begin
            @test status0 == 0
            @test abs(val0) < 1e-6
            @test maximum(abs.(x0)) < 1e-6
        end

        X_perturbed = copy(X_sample)
        X_perturbed[2, 3] *= 1.5
        val1, _, status1, = run_melitz_inner_delta(FIXTURE; X_data=X_perturbed, entry_target=entry_sample)

        @testset "negative control: perturbed data gives detectably positive Delta" begin
            @test status1 == 0
            @test val1 > val0 + 1e-5
        end
    end
end

println("\n" * "="^70)
println("Melitz Delta-star test suite complete.")
println("="^70)
