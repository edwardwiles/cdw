#!/usr/bin/env julia
# Reproducible end-to-end runner for the D=4 Melitz Delta-star benchmark.
# See docs/melitz_delta_star.md for the full derivation and design rationale.
#
# Usage:
#   julia --project=. scripts/run_melitz_delta_star_fake.jl --D 4 --seed 29 --draws 80000
#
# Requires KNITRO (licensed on demand.mit.edu) for the Delta(theta*) step; source
# .knitro_env.sh first. All other steps (fixture construction, moment residuals, ACR
# cross-check) run without KNITRO.
#
# Gate A rewrite (docs/melitz_delta_star.md Section 14): the previous version of this
# script predated the critical-bugfix/population-Pareto reconstruction and referenced
# fields removed from MelitzPrimitives/MelitzEquilibrium/MelitzCounterfactual
# (`f_entry`, `entrant_mass`, `price_power_prime`) -- it would not even run against the
# current types. Also fixed: the reported "gains from trade" omitted the baseline/autarky
# WAGE RATIO (`GT = 1 - gamma_prime^(1/(sigma-1))`), which is only valid when
# `w_prime_j/w_j == 1`; that does NOT hold here (`w[target_country]` is a genuine
# general-equilibrium output, not renormalized) -- see `melitz_gains_from_trade`'s
# docstring.

using Printf

const REPO_ROOT = dirname(dirname(@__FILE__))
const MDIR = joinpath(REPO_ROOT, "src", "melitz")

include(joinpath(REPO_ROOT, "misc", "doubleDiff.jl"))
include(joinpath(MDIR, "include_melitz.jl"))

# ---- CLI parsing (no extra dependency; a handful of flags) ----
function parse_args(argv)
    D = 4
    seed = 29
    draws = 80_000
    sigma = 2.5
    theta_star = 6.8
    target_country = 1
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--D"
            D = parse(Int, argv[i+1]); i += 2
        elseif a == "--seed"
            seed = parse(Int, argv[i+1]); i += 2
        elseif a == "--draws"
            draws = parse(Int, argv[i+1]); i += 2
        elseif a == "--sigma"
            sigma = parse(Float64, argv[i+1]); i += 2
        elseif a == "--theta_star"
            theta_star = parse(Float64, argv[i+1]); i += 2
        elseif a == "--target_country"
            target_country = parse(Int, argv[i+1]); i += 2
        else
            error("unrecognized argument: $a")
        end
    end
    return (D=D, seed=seed, draws=draws, sigma=sigma, theta_star=theta_star, target_country=target_country)
end

args = parse_args(ARGS)

println("="^78)
println("Full-D Melitz Delta-star benchmark (D=$(args.D), sigma=$(args.sigma), theta*=$(args.theta_star))")
println("seed=$(args.seed), draws (W)=$(args.draws), target_country=$(args.target_country)")
println("="^78)

# ---- 1. Construct the synthetic economy (population-Pareto GE, no KNITRO needed) ----
data = generate_fake_melitz_data(; D=args.D, sigma=args.sigma, theta_star=args.theta_star,
                                  target_country=args.target_country, seed=args.seed, W=args.draws)
p, eq, cf = data.primitives, data.equilibrium, data.counterfactual
D = p.D
j = p.target_country

println("\n--- Normalizations ---")
println("Pareto lower support = 1")
println("Baseline wage w[$(j)] = ", p.w[j], "  (genuine GE output, NOT renormalized to 1)")
println("Autarky wage numeraire w'[$(j)] = ", cf.w_prime)
println("Baseline gamma_d == 1 for every destination d (universal normalization)")
println("Baseline entrant mass N_o == 1 for every origin o (universal normalization)")
println("Autarky domestic cutoff zhat'[$(j),$(j)] = ", cf.cutoff_prime, " (Pareto lower support, by construction)")

println("\n--- True parameter summary ---")
@printf("A range: [%.4g, %.4g]   f range: [%.4g, %.4g]\n", extrema(p.A)..., extrema(p.f)...)
@printf("gamma_prime_target (SOLVED, not chosen) = %.6f\n", p.gamma_prime_target)
println("w = ", round.(p.w; sigdigits=4))
println("baseline cutoffs (zhat) range: ", round.(extrema(eq.cutoff); sigdigits=4))

# ---- 2. Gravity restrictions + factor-market/price-index identities (exact, closed form) ----
gravity_A, gravity_f = gravity_residuals(p)
println("\n--- Gravity restrictions (Cov(withinTransform(log tau), withinTransform(.))) ---")
@printf("A gravity residual = %.4g\n", gravity_A)
@printf("f gravity residual = %.4g\n", gravity_f)

# ---- 3. Gains from trade + ACR/Chaney cross-check (Gate A1/A2) ----
GT_model = melitz_gains_from_trade(p, cf)
lambda_jj, GT_ACR = acr_gains_from_trade(p, eq)
GT_naive_wrong = 1 - p.gamma_prime_target^(1 / (p.sigma - 1)) # kept ONLY to show the bug's magnitude

println("\n--- Gains from trade (Gate A1/A2) ---")
@printf("GT_model (wage-ratio, CORRECT)        = %.6g\n", GT_model)
@printf("GT_ACR   (1 - lambda_jj^(1/theta*))   = %.6g   (lambda_jj=%.6g)\n", GT_ACR, lambda_jj)
@printf("|GT_model - GT_ACR|                   = %.4g\n", abs(GT_model - GT_ACR))
@printf("[for reference only] naive formula omitting the wage ratio = %.6g -- WRONG whenever w[j] != 1\n", GT_naive_wrong)

min_active, worst_cell = min_active_draw_count(p, eq, data.z_draws)
println("\nmin active-draw count across all cells = $min_active (at cell $worst_cell)")
if min_active < 10
    @warn "min_active_draw_count < 10 -- the Delta(theta*) KNITRO solve below may fail to " *
          "converge (dual variables can diverge when a cell has too few participating firms " *
          "in the sample). Increase --draws."
end

# ---- 4. Delta(theta*) via the real CC/KNITRO inner loop + LFD recovery + ex-post checks ----
println("\n--- Delta(theta*) via the real CC minimum-divergence inner loop (KNITRO) ---")
knitro_ok = try
    include(joinpath(REPO_ROOT, "cc_algo", "include_cc_algo.jl"))
    @eval using .CounterfactualSensitivity
    true
catch e
    println("KNITRO/cc_algo not available in this environment (", sprint(showerror, e), ")")
    println("Source .knitro_env.sh and re-run to get Delta(theta*).")
    false
end

if knitro_ok
    lfd, obj = run_melitz_inner_delta(data;
        inner_loop_opt=joinpath(REPO_ROOT, "melitz_inner_loop_options.opt"))

    @printf("Delta(theta*) = %.6e\n", lfd.Delta)
    println("KNITRO status = ", lfd.nStatus, (lfd.nStatus == 0 ? " (optimal)" : ""))
    println("lfd_ok = ", lfd.lfd_ok)
    @printf("primal_divergence=%.4e  dual_divergence=%.4e  primal_dual_gap=%.4e\n",
        lfd.primal_divergence, lfd.dual_divergence, lfd.primal_dual_gap)
    @printf("kkt_opt_error=%.4e  kkt_feas_error=%.4e\n", lfd.kkt_opt_error, lfd.kkt_feas_error)
    @printf("max_weighted_moment_residual=%.4e  probability_normalization_residual=%.4e\n",
        lfd.maximum_weighted_moment_residual, lfd.probability_normalization_residual)

    if lfd.lfd_ok
        check = check_profiled_melitz_equilibrium(p, eq, cf, data.z_draws, lfd.weights)
        println("\n--- Ex-post omitted-equilibrium-equation checks (Gate A3), under the recovered LFD ---")
        @printf("max|residual_gamma_baseline| (implied)          = %.4e\n", maximum(abs.(check.residual_gamma_baseline)))
        @printf("max|residual_free_entry_baseline| (definitional) = %.4e\n", maximum(abs.(check.residual_free_entry_baseline)))
        @printf("residual_free_entry_autarky (definitional)       = %.4e\n", check.residual_free_entry_autarky)
        @printf("|f_entry[j]-f_entry_autarky| (implied by link moment) = %.4e\n",
            abs(check.f_entry_recovered[j] - check.f_entry_autarky_recovered))
        @printf("residual_market_clearing_autarky (independent)   = %.4e\n", check.residual_market_clearing_autarky)
        @printf("residual_gamma_autarky (independent, KEY check)  = %.4e\n", check.residual_gamma_autarky)
        @printf("N_prime_market_clearing = %.8f   N_prime_from_gamma = %.8f   rel diff = %.4e\n",
            check.N_prime_market_clearing, check.N_prime_from_gamma, check.N_prime_diff_rel)
        @printf("|N_prime[j] - N[j]| == |N_prime_market_clearing - 1|  = %.4e\n", abs(check.N_prime_market_clearing - 1.0))
        @printf("residual_autarky_cutoff (definitional)           = %.4e\n", check.residual_autarky_cutoff)
        @printf("min_cutoff_minus_one = %.4e   min_export_minus_domestic = %.4e\n",
            check.min_cutoff_minus_one, check.min_export_minus_domestic)
        @printf("gravity_residual_A=%.4e  gravity_residual_f=%.4e\n", check.gravity_residual_A, check.gravity_residual_f)
    end
end

println("\n" * "="^78)
println("Done. See docs/melitz_delta_star.md for the full derivation and result discussion.")
println("="^78)
