# Variant of melitz_finite_delta_campaign.jl for this session's before/after comparison:
# lets the caller choose the inner-loop options file (full maxit=10000 vs. the new
# maxit=250 budget-check file) and whether to enable the lower_limit early-stop guard,
# without touching the production campaign script.
#
# Usage: julia --project=. scripts/melitz_screening_campaign_variant.jl

using Printf
include(joinpath(@__DIR__, "melitz_finite_delta_campaign.jl"))

function run_campaign_variant(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000,
                                deltas=[1e-2], theta_box=0.10, h=1e-4, gradient_backend=:B,
                                cutoff_constraint_backend=:nonlinear_reference,
                                inner_opt_name="melitz_inner_loop_options.opt",
                                lower_limit_guard=nothing)
    println("="^100)
    @printf("Melitz screening campaign variant: inner_opt=%s lower_limit_guard=%s\n", inner_opt_name, string(lower_limit_guard))
    println("="^100)

    data = generate_fake_melitz_data(; D=D, sigma=sigma, theta_star=theta_star,
        target_country=target_country, seed=seed, W=W)
    inner_opt = joinpath(dirname(@__DIR__), inner_opt_name)
    outer_opt = joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt")

    obj, theta0 = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt)
    ctx = obj.γ

    results = NamedTuple[]
    for delta in deltas, direction in (:upper, :lower)
        t0 = time()
        res = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=delta, direction=direction,
            gradient_backend=gradient_backend, h=h, theta_box=theta_box,
            inner_loop_opt=inner_opt, outer_loop_opt=outer_opt, lower_limit_guard=lower_limit_guard)
        wall = time() - t0
        @printf("delta=%.1e dir=%-6s wall=%6.1fs nStatus=%5d inner_solved=%3d moment_infeas=%3d budget_infeas=%3d numerical_fail=%3d fc_calls=%3d ga_calls=%3d\n",
            delta, direction, wall, res.nStatus, res.n_inner_solved, res.n_moment_infeasible_reject,
            res.n_budget_infeasible_reject, res.n_numerical_failure_reject,
            res.n_fc_calls, res.n_ga_calls)
        inc = res.cold_verified_incumbent
        if inc !== nothing
            @printf("  cold-verified: outer_feasible=%s Delta=%.6e gamma_prime=%.6f\n",
                inc.classification.outer_feasible, inc.eval.Delta, inc.eval.gamma_prime_j)
        end
        push!(results, (delta=delta, direction=direction, wall=wall, nStatus=res.nStatus,
            n_inner_solved=res.n_inner_solved, n_moment_infeasible_reject=res.n_moment_infeasible_reject,
            n_budget_infeasible_reject=res.n_budget_infeasible_reject,
            n_numerical_failure_reject=res.n_numerical_failure_reject,
            n_fc_calls=res.n_fc_calls, n_ga_calls=res.n_ga_calls,
            outer_feasible=inc === nothing ? false : inc.classification.outer_feasible,
            Delta=inc === nothing ? NaN : inc.eval.Delta))
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    inner_opt_name = length(ARGS) >= 1 ? ARGS[1] : "melitz_inner_loop_options.opt"
    guard = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : nothing
    run_campaign_variant(; inner_opt_name=inner_opt_name, lower_limit_guard=guard)
end
