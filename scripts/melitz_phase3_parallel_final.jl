# Continuation session (2026-07-23), Section 3 closing benchmark: IDENTICAL to
# scripts/melitz_phase2_final.jl (the recorded Phase I+II final campaign) EXCEPT
# `gradient_backend=:B_localized_parallel` instead of `:B_localized`, to connect this
# session's kernel-level parallel-gradient speedup (1.7x-5.1x at D=4's 30 coordinates,
# scripts/melitz_parallel_gradient_sweep.sh) to a real end-to-end trajectory wall-clock
# result. Every cold-verified incumbent must be IDENTICAL to melitz_phase2_final.jl's own
# recorded values (the parallel backend is bit-exact vs the serial one -- it can only change
# wall-clock cost, never the outer search's own trajectory or answer).
#
# Usage: JULIA_NUM_THREADS=<n> julia --project=. scripts/melitz_phase3_parallel_final.jl

using Printf
include(joinpath(@__DIR__, "melitz_finite_delta_campaign.jl"))

function run_phase3_parallel_final(; D=4, W=20_000, seed=29, deltas=[1e-3, 1e-2], directions=(:upper, :lower),
                                     theta_box=0.10, cutoff_constraint_backend=:linear)
    data = generate_fake_melitz_data(; D=D, sigma=2.5, theta_star=6.8, target_country=1, seed=seed, W=W)
    inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options_budgetcheck.opt")
    outer_opt = joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt")

    println("Threads.nthreads()=", Threads.nthreads(), "  Threads.maxthreadid()=", Threads.maxthreadid())

    rows = NamedTuple[]
    for delta in deltas, direction in directions
        obj, theta0 = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt)
        ctx = obj.γ
        t0 = time()
        res = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=delta, direction=direction,
            theta_box=theta_box, inner_loop_opt=inner_opt, outer_loop_opt=outer_opt,
            cutoff_constraint_backend=cutoff_constraint_backend,
            lower_limit_guard=0.0, origin_block_screen=true, dual_polish_screen=true,
            gradient_backend=:B_localized_parallel)
        wall = time() - t0
        @printf("delta=%.1e dir=%-6s wall=%7.2fs nStatus=%5d inner_solved=%3d moment_infeas=%3d budget_infeas=%3d numerical_fail=%3d fc=%3d ga=%3d\n",
            delta, direction, wall, res.nStatus, res.n_inner_solved, res.n_moment_infeasible_reject,
            res.n_budget_infeasible_reject, res.n_numerical_failure_reject, res.n_fc_calls, res.n_ga_calls)
        inc = res.cold_verified_incumbent
        if inc !== nothing
            @printf("  cold-verified: outer_feasible=%s Delta=%.6e gamma_prime=%.6f\n",
                inc.classification.outer_feasible, inc.eval.Delta, inc.eval.gamma_prime_j)
        end
        push!(rows, (delta=delta, direction=direction, wall=wall, nStatus=res.nStatus,
            n_inner_solved=res.n_inner_solved, n_moment_infeasible_reject=res.n_moment_infeasible_reject,
            n_budget_infeasible_reject=res.n_budget_infeasible_reject,
            n_numerical_failure_reject=res.n_numerical_failure_reject,
            outer_feasible=inc === nothing ? false : inc.classification.outer_feasible,
            Delta=inc === nothing ? NaN : inc.eval.Delta,
            gamma_prime=inc === nothing ? NaN : inc.eval.gamma_prime_j))
    end

    println("\n", "="^100)
    total = sum(r.wall for r in rows)
    @printf("TOTAL wall (all %d delta x direction cells, threads=%d) = %.2fs\n", length(rows), Threads.nthreads(), total)
    for delta in deltas
        drows = filter(r -> r.delta == delta, rows)
        @printf("  delta=%.1e subtotal = %.2fs\n", delta, sum(r.wall for r in drows))
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_phase3_parallel_final()
end
