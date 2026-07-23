# Continuation session (2026-07-23, "continue directly from the Melitz screening/localized-
# gradient checkpoint"), Section 2: re-profile the CURRENT optimized production stack (range+
# stored-dual+origin-block+dual-polish screens, lower_limit_guard=0.0, maxit=250, :logf/:linear,
# gradient_backend=:B_localized) now that the bottleneck ordering has changed since the prior
# session's own profiling campaign (which predates the localized-gradient win).
#
# Uses the EXISTING exception-safe profiling instrumentation (src/melitz/profiling.jl,
# @melitz_profile / melitz_record_seconds_outcome!, already wired into cb_F!/cb_G! in
# finite_delta_outer.jl by the prior continuation session) -- no new instrumentation added
# this session. Reports, per delta/direction cell: the full melitz_profile_report() category
# breakdown plus the Section 3.2 residual (total outer KNITRO wall minus complete FC+GA
# callback wall = true KNITRO-C/API overhead).
#
# Usage: julia --project=. scripts/melitz_continuation2_reprofile.jl

using Printf
include(joinpath(@__DIR__, "melitz_finite_delta_campaign.jl"))

function run_reprofile(; D=4, W=20_000, seed=29, deltas=[1e-3, 1e-2], directions=(:upper, :lower),
                         theta_box=0.10, cutoff_constraint_backend=:linear,
                         gradient_backend=:B_localized)
    data = generate_fake_melitz_data(; D=D, sigma=2.5, theta_star=6.8, target_country=1, seed=seed, W=W)
    inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options_budgetcheck.opt")   # maxit=250
    outer_opt = joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt")

    all_rows = NamedTuple[]
    for delta in deltas, direction in directions
        obj, theta0 = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt)
        ctx = obj.γ
        melitz_profile_reset!()
        MELITZ_PROFILE[] = true
        t0 = time()
        res = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=delta, direction=direction,
            theta_box=theta_box, inner_loop_opt=inner_opt, outer_loop_opt=outer_opt,
            cutoff_constraint_backend=cutoff_constraint_backend,
            lower_limit_guard=0.0, origin_block_screen=true, dual_polish_screen=true,
            gradient_backend=gradient_backend)
        wall = time() - t0
        MELITZ_PROFILE[] = false

        println("\n", "#"^100)
        @printf("delta=%.1e direction=%-6s wall=%.2fs nStatus=%d inner_solved=%d moment_infeas=%d budget_infeas=%d numerical_fail=%d fc=%d ga=%d\n",
            delta, direction, wall, res.nStatus, res.n_inner_solved, res.n_moment_infeasible_reject,
            res.n_budget_infeasible_reject, res.n_numerical_failure_reject, res.n_fc_calls, res.n_ga_calls)
        inc = res.cold_verified_incumbent
        if inc !== nothing
            @printf("cold-verified: outer_feasible=%s Delta=%.6e gamma_prime=%.6f\n",
                inc.classification.outer_feasible, inc.eval.Delta, inc.eval.gamma_prime_j)
        end
        rows = melitz_profile_report(; trajectory_total_s=wall)
        for r in rows
            push!(all_rows, merge((delta=delta, direction=direction), r))
        end
    end
    return all_rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_reprofile()
end
