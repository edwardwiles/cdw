#!/usr/bin/env julia
# Reproducible end-to-end runner for the D=4 Melitz Delta-star benchmark.
# See docs/melitz_delta_star.md for the full derivation and design rationale.
#
# Usage:
#   julia --project=. scripts/run_melitz_delta_star_fake.jl --D 4 --seed 1234 --draws 80000
#
# Requires KNITRO (licensed on demand.mit.edu) for the Delta(theta*) step; source
# .knitro_env.sh first. All other steps (fixture construction, moment residuals, ACR
# cross-check) run without KNITRO.

using Printf

const REPO_ROOT = dirname(dirname(@__FILE__))
const MDIR = joinpath(REPO_ROOT, "src", "melitz")

include(joinpath(REPO_ROOT, "misc", "doubleDiff.jl"))
include(joinpath(MDIR, "types.jl"))
include(joinpath(MDIR, "pareto.jl"))
include(joinpath(MDIR, "firm_quantities.jl"))
include(joinpath(MDIR, "equilibrium.jl"))
include(joinpath(MDIR, "fake_data.jl"))
include(joinpath(MDIR, "moments.jl"))
include(joinpath(MDIR, "fstar_solver.jl"))

# ---- CLI parsing (no extra dependency; a handful of flags) ----
function parse_args(argv)
    D = 4
    seed = 1234
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

# ---- 1. Construct the synthetic D=4 economy (closed-form F* solve, no KNITRO) ----
data = generate_fake_melitz_data(; D=args.D, sigma=args.sigma, theta_star=args.theta_star,
                                  target_country=args.target_country, seed=args.seed, W=args.draws)
p, eq, cf = data.primitives, data.equilibrium, data.counterfactual
D = p.D

println("\n--- Normalizations ---")
println("Pareto lower support = 1 (F* primitive)")
println("Wage numeraire: w[$(args.target_country)] = ", p.w[args.target_country])
println("Autarky wage numeraire: w'[target] = ", cf.w_prime)
println("price_power_d == 1 for every baseline destination d (by construction)")
println("Autarky cutoff zhat'[target,target] = ", cf.cutoff_prime, " (Pareto lower support)")

println("\n--- True parameter summary ---")
@printf("A range: [%.4g, %.4g]   f range: [%.4g, %.4g]\n", extrema(p.A)..., extrema(p.f)...)
println("f_entry = ", round.(p.f_entry; sigdigits=4))
println("w = ", round.(p.w; sigdigits=4))
println("entrant_mass (N) = ", round.(eq.entrant_mass; sigdigits=4))
println("baseline cutoffs (zhat) range: ", round.(extrema(eq.cutoff); sigdigits=4))

# ---- 2. Verify baseline/autarky moment residuals ----
trade_resid = trade_flow_residuals(p, eq, data.z_draws)
entry_resid = entry_residuals(p, eq, data.z_draws)
gravity_A, gravity_f = gravity_residuals(p)

println("\n--- Moment residuals (Monte Carlo, W=$(args.draws) draws, vs. closed-form population target) ---")
@printf("max |bilateral-flow residual| (economic units) = %.4g\n", maximum(abs.(trade_resid)))
@printf("max |free-entry residual| (economic units)     = %.4g\n", maximum(abs.(entry_resid)))
@printf("A gravity residual <DDlogtau,DDlogA>            = %.4g\n", gravity_A)
@printf("f gravity residual <DDlogtau,DDlogf>             = %.4g\n", gravity_f)

min_active, worst_cell = min_active_draw_count(p, eq, data.z_draws)
println("min active-draw count across all cells = $min_active (at cell $worst_cell)")
if min_active < 10
    @warn "min_active_draw_count < 10 -- the Delta(theta*) KNITRO solve below may fail to " *
          "converge (dual variables can diverge when a cell has too few participating firms " *
          "in the sample). Increase --draws."
end

# ---- 3. Gains from trade + ACR/Chaney cross-check ----
GT_model = 1 - cf.price_power_prime^(1 / (p.sigma - 1))
lambda_tt = eq.trade_flow[p.target_country, p.target_country] / eq.expenditure[p.target_country]
GT_ACR = 1 - lambda_tt^(1 / p.theta_star)

println("\n--- Gains from trade ---")
@printf("GT (model, price-power ratio) = %.6g\n", GT_model)
@printf("GT (ACR/Chaney, 1-lambda_dd^(1/theta*)) = %.6g\n", GT_ACR)
@printf("|GT_model - GT_ACR| = %.4g\n", abs(GT_model - GT_ACR))

# ---- 4. Delta(theta*) via the real CC/KNITRO inner loop ----
println("\n--- Delta(theta*) via the real CC minimum-divergence inner loop (KNITRO) ---")
knitro_ok = try
    include(joinpath(REPO_ROOT, "cc_algo", "include_cc_algo.jl"))
    @eval using .CounterfactualSensitivity
    include(joinpath(MDIR, "delta_star.jl"))
    true
catch e
    println("KNITRO/cc_algo not available in this environment (", sprint(showerror, e), ")")
    println("Source .knitro_env.sh and re-run to get Delta(theta*).")
    false
end

if knitro_ok
    val, x, nStatus = inner_loop(build_melitz_psi_bundle(data;
        inner_loop_opt=joinpath(REPO_ROOT, "melitz_inner_loop_options.opt"))...)
    @printf("Delta(theta*) = %.6g\n", val)
    println("KNITRO status = ", nStatus, (nStatus == 0 ? " (optimal)" : ""))
    @printf("max |optimal dual variable| = %.4g\n", maximum(abs.(x)))
end

println("\n" * "="^78)
println("Done. See docs/melitz_delta_star.md for the full derivation and result discussion.")
println("="^78)
