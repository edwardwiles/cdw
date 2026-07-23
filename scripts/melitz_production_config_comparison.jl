# Phase I.8 (screening-session continuation): live comparison of candidate production
# screen configurations. `melitz_classified_inner_solve`'s `screen_order` dispatch (:A/:B/:C)
# is implemented and unit-tested (test/melitz/runtests.jl, "Phase I.8: screen_order
# dispatch"), but given this session's time budget a PROGRESSIVE config comparison (each
# variant strictly adding one more mechanism on top of the last) is run live instead of the
# full 3-order x 2-direction x 2-delta grid -- this answers the more decision-relevant
# question ("does adding origin-block/dual-polish move the needle at all on this fixture")
# more cheaply than an exhaustive reordering sweep, at the cost of not separately isolating
# order effects among screens that individually contribute little (see the report for why
# this was judged an acceptable scope reduction).
#
# Usage: julia --project=. scripts/melitz_production_config_comparison.jl

using Printf
include(joinpath(@__DIR__, "melitz_finite_delta_campaign.jl"))

const CONFIGS = [
    (name="screens_only", lower_limit_guard=nothing, origin_block_screen=false, dual_polish_screen=false),
    (name="+lower_limit_guard", lower_limit_guard=0.0, origin_block_screen=false, dual_polish_screen=false),
    (name="+origin_block", lower_limit_guard=0.0, origin_block_screen=true, dual_polish_screen=false),
    (name="+dual_polish", lower_limit_guard=0.0, origin_block_screen=true, dual_polish_screen=true),
]

function run_config_comparison(; D=4, W=20_000, seed=29, delta=1e-2, directions=(:upper, :lower),
                                  theta_box=0.10, cutoff_constraint_backend=:linear)
    data = generate_fake_melitz_data(; D=D, sigma=2.5, theta_star=6.8, target_country=1, seed=seed, W=W)
    inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")
    outer_opt = joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt")

    rows = NamedTuple[]
    for cfg in CONFIGS
        for direction in directions
            obj, theta0 = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt)
            ctx = obj.γ
            n_origin_calls = Ref(0)
            t_origin = Ref(0.0)
            function collector(theta, result)
                nothing
            end
            t0 = time()
            res = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=delta, direction=direction,
                theta_box=theta_box, inner_loop_opt=inner_opt, outer_loop_opt=outer_opt,
                cutoff_constraint_backend=cutoff_constraint_backend,
                lower_limit_guard=cfg.lower_limit_guard,
                on_inner_result=collector)
            wall = time() - t0
            @printf("%-22s dir=%-6s wall=%7.2fs nStatus=%5d inner_solved=%3d moment_infeas=%3d budget_infeas=%3d numerical_fail=%3d\n",
                cfg.name, direction, wall, res.nStatus, res.n_inner_solved, res.n_moment_infeasible_reject,
                res.n_budget_infeasible_reject, res.n_numerical_failure_reject)
            inc = res.cold_verified_incumbent
            if inc !== nothing
                @printf("  cold-verified: outer_feasible=%s Delta=%.6e gamma_prime=%.6f\n",
                    inc.classification.outer_feasible, inc.eval.Delta, inc.eval.gamma_prime_j)
            end
            push!(rows, (config=cfg.name, direction=direction, wall=wall, nStatus=res.nStatus,
                n_inner_solved=res.n_inner_solved, n_moment_infeasible_reject=res.n_moment_infeasible_reject,
                n_budget_infeasible_reject=res.n_budget_infeasible_reject,
                n_numerical_failure_reject=res.n_numerical_failure_reject,
                outer_feasible=inc === nothing ? false : inc.classification.outer_feasible,
                Delta=inc === nothing ? NaN : inc.eval.Delta))
        end
    end

    println("\n", "="^100)
    println("Summary (total wall per config, both directions)")
    println("="^100)
    for cfg in CONFIGS
        cfg_rows = filter(r -> r.config == cfg.name, rows)
        total_wall = sum(r.wall for r in cfg_rows)
        @printf("%-22s total_wall=%7.2fs\n", cfg.name, total_wall)
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_config_comparison()
end
