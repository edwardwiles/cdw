# Session driver: reproduces the Section 1 finite-delta incumbent campaign
# (docs/melitz_delta_star.md Section 20.F) at the frozen commit, and reports GAINS FROM
# TRADE using the CORRECT wage-ratio formula (equilibrium.jl's `melitz_gains_from_trade`,
# cross-checked against `acr_gains_from_trade`) rather than the naive
# `1 - gamma_prime^(1/(sigma-1))` formula -- main prompt Section 2.
#
# Usage: julia --project=. scripts/melitz_finite_delta_campaign.jl [deltas...]
# Default: D=4, W=20000, seed=29, theta_box=0.10, h=1e-4 (Backend B), maxit from
# melitz_outer_finite_delta.opt, deltas = [1e-2, 1e-3], both directions.

using Printf

const ROOT = dirname(dirname(@__DIR__)) == "/" ? dirname(@__DIR__) : dirname(@__DIR__)
const MELITZ_DIR = joinpath(@__DIR__, "..", "src", "melitz")
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
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
include(joinpath(MELITZ_DIR, "gradient_lab.jl"))
include(joinpath(MELITZ_DIR, "outer_solve.jl"))
include(joinpath(MELITZ_DIR, "inner_screening.jl"))
include(joinpath(MELITZ_DIR, "origin_block_screen.jl"))
include(joinpath(MELITZ_DIR, "finite_delta_outer.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

function gt_report(label, p::MelitzPrimitives, cf::MelitzCounterfactual, eq::MelitzEquilibrium)
    GT_wage_ratio = melitz_gains_from_trade(p, cf)
    lambda_jj, GT_acr = acr_gains_from_trade(p, eq)
    @printf("  %-28s gamma_prime=%.6f  GT(wage-ratio)=%.6f  GT(ACR)=%.6f  |diff|=%.3e  lambda_jj=%.6f\n",
        label, p.gamma_prime_target, GT_wage_ratio, GT_acr, abs(GT_wage_ratio - GT_acr), lambda_jj)
    return GT_wage_ratio, GT_acr
end

function primitives_at(theta_free, ctx)
    A, f, gamma_prime_j, f_jj = expand_free_theta(theta_free, ctx)
    return MelitzPrimitives(ctx.D, ctx.sigma, ctx.theta_star, ctx.target_country, ctx.tau,
                             ctx.w, A, f, gamma_prime_j)
end

function run_campaign(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000,
                       deltas=[1e-2, 1e-3], theta_box=0.10, h=1e-4, gradient_backend=:B,
                       cutoff_constraint_backend=:nonlinear_reference)
    println("="^100)
    println("Melitz finite-delta campaign reproduction -- Section 1 checkpoint")
    println("="^100)
    @printf("D=%d sigma=%.3f theta_star=%.3f target_country=%d seed=%d W=%d theta_box=%.3f h=%.1e backend=%s cutoff_backend=%s\n",
        D, sigma, theta_star, target_country, seed, W, theta_box, h, gradient_backend, cutoff_constraint_backend)

    data = generate_fake_melitz_data(; D=D, sigma=sigma, theta_star=theta_star,
        target_country=target_country, seed=seed, W=W)
    inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")
    outer_opt = joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt")

    obj, theta0 = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt)
    ctx = obj.γ

    r0 = evaluate_melitz_delta(theta0, ctx, obj; cold=true)
    println()
    println("Population-Pareto starting point:")
    @printf("  Delta_pop=%.6e  nStatus=%d  lfd_ok=%s  min_slack=%.4f  gravity(A/f)=%.2e/%.2e\n",
        r0.Delta, r0.nStatus, r0.lfd_ok, r0.min_slack,
        r0.equilibrium_check.gravity_residual_A, r0.equilibrium_check.gravity_residual_f)
    p0 = primitives_at(theta0, ctx)
    eq0 = MelitzEquilibrium(ctx.expenditure, ones(D), r0.cutoff, ctx.X_data)
    cf0 = MelitzCounterfactual(target_country, ctx.w_prime, ctx.w_prime * ctx.L[target_country], 1.0,
                                ctx.w_prime * ctx.L[target_country])
    gt_report("population-Pareto start", p0, cf0, eq0)

    results = NamedTuple[]
    for delta in deltas, direction in (:upper, :lower)
        println()
        println("-"^100)
        @printf("delta=%.1e  direction=%s\n", delta, direction)
        t0 = time()
        res = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=delta, direction=direction,
            gradient_backend=gradient_backend, h=h, theta_box=theta_box,
            inner_loop_opt=inner_opt, outer_loop_opt=outer_opt)
        wall = time() - t0

        inc = res.cold_verified_incumbent
        @printf("  terminal nStatus=%d  wall=%.1fs  inner_solves=%d (infeas=%d, eval_failures=%d)\n",
            res.nStatus, wall, res.inner_solve_count, res.inner_infeas_count, res.inner_eval_failures)
        if inc === nothing
            println("  ** NO cold-verified incumbent (fell back to nothing -- should not happen if theta0 outer-feasible) **")
        else
            e = inc.eval
            @printf("  cold-verified incumbent: outer_feasible=%s  Delta=%.6e (budget slack=%.3e)  min_cutoff_slack=%.4f  gravity(A/f)=%.2e/%.2e\n",
                inc.classification.outer_feasible, e.Delta, delta - e.Delta, e.min_slack,
                e.equilibrium_check.gravity_residual_A, e.equilibrium_check.gravity_residual_f)
            p_inc = primitives_at(e.theta_free, ctx)
            eq_inc = MelitzEquilibrium(ctx.expenditure, ones(D), e.cutoff, ctx.X_data)
            cf_inc = MelitzCounterfactual(target_country, ctx.w_prime, ctx.w_prime * ctx.L[target_country], 1.0,
                                           ctx.w_prime * ctx.L[target_country])
            GT_wr, GT_acr = gt_report("  incumbent", p_inc, cf_inc, eq_inc)
            push!(results, (delta=delta, direction=direction, nStatus=res.nStatus, wall=wall,
                inner_solve_count=res.inner_solve_count, inner_infeas_count=res.inner_infeas_count,
                inner_eval_failures=res.inner_eval_failures, Delta=e.Delta,
                gamma_prime=p_inc.gamma_prime_target, GT_wage_ratio=GT_wr, GT_acr=GT_acr,
                outer_feasible=inc.classification.outer_feasible, min_slack=e.min_slack,
                terminal_outer_feasible=res.terminal_classification.outer_feasible))
        end
    end

    println()
    println("="^100)
    println("Summary (cold-verified incumbents, CORRECT GT formula)")
    println("="^100)
    @printf("%-8s %-8s %10s %8s %14s %12s %12s %10s %8s\n",
        "delta", "dir", "nStatus", "wall(s)", "Delta", "gamma_prime", "GT(correct)", "feasible", "slack")
    for r in results
        @printf("%-8.1e %-8s %10d %8.1f %14.6e %12.6f %12.6f %10s %8.4f\n",
            r.delta, r.direction, r.nStatus, r.wall, r.Delta, r.gamma_prime, r.GT_wage_ratio,
            r.outer_feasible, r.min_slack)
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_campaign()
end
