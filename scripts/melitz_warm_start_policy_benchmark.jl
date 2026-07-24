# Continuation session (2026-07-23, "make the optimized architecture scalable in memory and
# D") Section 6: live before/after comparison of the four `warm_start_source` policies
# (`:previous`, `:bank_nearest`, `:bank_best_lb`, `:neutral`) -- the MECHANISM
# (`melitz_resolve_warm_start!`) has been live and tested since the prior continuation
# session's own addendum; this script is the live campaign measuring its actual production
# impact, not attempted in either prior session.
#
# Matched post-JIT trajectories: SAME fixture (D=4/W=20,000/seed=29), SAME delta/direction
# cells, SAME gradient_backend/screens/maxit -- varies ONLY warm_start_source, so any
# difference in wall time/inner-solve counts is attributable to the warm-start policy alone.
#
# Reported per policy: full trajectory wall, n_inner_solved/n_budget_infeasible_reject/
# n_numerical_failure_reject/n_moment_infeasible_reject, successful-inner-solve wall
# (mean/median/total via inner_solve_warm_success), live-threshold-crossing count/cost
# (inner_solve_budget_infeasible), and the economic incumbent (cold-verified Delta/gamma_prime)
# -- the last confirms the warm-start policy changes only HOW the optimum is reached, never
# WHAT it is. KNITRO-internal per-solve ITERATION counts are not separately available from
# this codebase's existing instrumentation (melitz_profile_summary times wall-clock, not
# KN_get_number_iters per call) -- wall-clock/solve-count are used as the comparison
# metrics instead, a disclosed scope limitation, not an oversight.
#
# Usage: julia --project=. scripts/melitz_warm_start_policy_benchmark.jl

using Printf
include(joinpath(@__DIR__, "melitz_finite_delta_campaign.jl"))

function run_policy(policy::Symbol; D=4, W=20_000, seed=29, deltas=[1e-2], directions=(:upper, :lower),
                     theta_box=0.10, cutoff_constraint_backend=:linear)
    data = generate_fake_melitz_data(; D=D, sigma=2.5, theta_star=6.8, target_country=1, seed=seed, W=W)
    inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options_budgetcheck.opt")
    outer_opt = joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt")

    melitz_profile_reset!()
    MELITZ_PROFILE[] = true

    rows = NamedTuple[]
    t_total0 = time()
    for delta in deltas, direction in directions
        obj, theta0 = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt)
        ctx = obj.γ
        t0 = time()
        res = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=delta, direction=direction,
            theta_box=theta_box, inner_loop_opt=inner_opt, outer_loop_opt=outer_opt,
            cutoff_constraint_backend=cutoff_constraint_backend,
            lower_limit_guard=0.0, origin_block_screen=true, dual_polish_screen=true,
            gradient_backend=:B_localized, warm_start_source=policy)
        wall = time() - t0
        inc = res.cold_verified_incumbent
        push!(rows, (delta=delta, direction=direction, wall=wall, nStatus=res.nStatus,
            n_inner_solved=res.n_inner_solved, n_moment_infeasible_reject=res.n_moment_infeasible_reject,
            n_budget_infeasible_reject=res.n_budget_infeasible_reject,
            n_numerical_failure_reject=res.n_numerical_failure_reject,
            outer_feasible=inc === nothing ? false : inc.classification.outer_feasible,
            Delta=inc === nothing ? NaN : inc.eval.Delta,
            gamma_prime=inc === nothing ? NaN : inc.eval.gamma_prime_j))
    end
    total_wall = time() - t_total0
    MELITZ_PROFILE[] = false

    summary = melitz_profile_summary()
    warm_success = filter(r -> r.category == :inner_solve_warm_success, summary)
    budget_infeas = filter(r -> r.category == :inner_solve_budget_infeasible, summary)

    return (; policy, rows, total_wall, warm_success, budget_infeas)
end

if abspath(PROGRAM_FILE) == @__FILE__
    policies = [:previous, :bank_nearest, :bank_best_lb, :neutral]
    results = Dict{Symbol,Any}()
    for pol in policies
        println("="^100)
        println("Policy: ", pol)
        println("="^100)
        r = run_policy(pol)
        results[pol] = r
        for row in r.rows
            @printf("  delta=%.1e dir=%-6s wall=%7.3fs nStatus=%5d inner_solved=%3d budget_infeas=%3d numerical_fail=%3d  Delta=%.6e gamma_prime=%.6f\n",
                row.delta, row.direction, row.wall, row.nStatus, row.n_inner_solved,
                row.n_budget_infeasible_reject, row.n_numerical_failure_reject, row.Delta, row.gamma_prime)
        end
        @printf("  TOTAL wall = %.3fs\n", r.total_wall)
        if !isempty(r.warm_success)
            ws = r.warm_success[1]
            @printf("  inner_solve_warm_success: count=%d total_s=%.4f mean_ms=%.3f median_ms=%.3f p90_ms=%.3f\n",
                ws.count, ws.total_s, ws.mean_ms, ws.median_ms, ws.p90_ms)
        end
        if !isempty(r.budget_infeas)
            bi = r.budget_infeas[1]
            @printf("  inner_solve_budget_infeasible (live-threshold crossings): count=%d total_s=%.4f mean_ms=%.3f\n",
                bi.count, bi.total_s, bi.mean_ms)
        end
    end

    println("\n", "="^100)
    println("SUMMARY (total wall by policy)")
    println("="^100)
    for pol in policies
        @printf("  %-14s total_wall=%.3fs\n", pol, results[pol].total_wall)
    end

    println("\nEconomic incumbent consistency check (Delta/gamma_prime per delta/direction, across policies):")
    for delta in [1e-2], direction in (:upper, :lower)
        vals = [(pol, only(filter(r -> r.delta == delta && r.direction == direction, results[pol].rows))) for pol in policies]
        @printf("  delta=%.1e dir=%s:\n", delta, direction)
        for (pol, row) in vals
            @printf("    %-14s Delta=%.8e gamma_prime=%.8f\n", pol, row.Delta, row.gamma_prime)
        end
    end
end
