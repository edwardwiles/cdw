# Session prompt Section 4: benchmark cutoff_constraint_backend=:linear vs
# :nonlinear_reference at MATCHED settings (same theta_init, delta, direction, gradient
# backend, theta_box, maxit, inner tolerances) -- reports objective/Delta path via the
# cold-verified incumbent, terminal status, FC/GA callback counts, inner-solve counts, and
# wall time for the COMPLETE trajectory (not an isolated cutoff-evaluation microbenchmark,
# per the governing prompt's own instruction).
#
# Usage: julia --project=. scripts/melitz_cutoff_backend_benchmark.jl

using Printf

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
include(joinpath(MELITZ_DIR, "fake_data.jl"))
include(joinpath(MELITZ_DIR, "gradient_lab.jl"))
include(joinpath(MELITZ_DIR, "outer_solve.jl"))
include(joinpath(MELITZ_DIR, "inner_screening.jl"))
include(joinpath(MELITZ_DIR, "origin_block_screen.jl"))
include(joinpath(MELITZ_DIR, "localized_gradient.jl"))
include(joinpath(MELITZ_DIR, "finite_delta_outer.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

function run_benchmark(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000,
                        deltas=[1e-2], directions=[:upper, :lower], theta_box=0.10, h=1e-4,
                        gradient_backend=:B)
    data = generate_fake_melitz_data(; D=D, sigma=sigma, theta_star=theta_star,
        target_country=target_country, seed=seed, W=W)
    inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")
    outer_opt = joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt")

    obj, theta0 = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt)
    ctx = obj.γ
    r0 = evaluate_melitz_delta(theta0, ctx, obj; cold=true, store_G=false)
    @printf("Population-Pareto start: Delta=%.6e nStatus=%d gamma_prime=%.6f\n",
        r0.Delta, r0.nStatus, r0.gamma_prime_j)

    rows = NamedTuple[]
    for delta in deltas, direction in directions, backend in (:nonlinear_reference, :linear)
        println()
        println("="^100)
        @printf("delta=%.1e direction=%s backend=%s\n", delta, direction, backend)
        t0 = time()
        res = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=delta, direction=direction,
            gradient_backend=gradient_backend, h=h, theta_box=theta_box,
            cutoff_constraint_backend=backend,
            inner_loop_opt=inner_opt, outer_loop_opt=outer_opt)
        wall = time() - t0
        inc = res.cold_verified_incumbent
        @printf("  wall=%.1fs  nStatus=%d  FC_calls=%d  GA_calls=%d  inner_solves=%d (infeas=%d, eval_fail=%d)\n",
            wall, res.nStatus, res.n_fc_calls, res.n_ga_calls, res.inner_solve_count,
            res.inner_infeas_count, res.inner_eval_failures)
        if inc !== nothing
            @printf("  cold-verified: Delta=%.6e  gamma_prime=%.6f  feasible=%s  min_slack=%.4f\n",
                inc.eval.Delta, inc.eval.gamma_prime_j, inc.classification.outer_feasible, inc.eval.min_slack)
        else
            println("  ** no cold-verified incumbent **")
        end
        push!(rows, (delta=delta, direction=direction, backend=backend, wall=wall,
            nStatus=res.nStatus, fc_calls=res.n_fc_calls, ga_calls=res.n_ga_calls,
            inner_solves=res.inner_solve_count, inner_infeas=res.inner_infeas_count,
            inner_eval_failures=res.inner_eval_failures,
            Delta=inc === nothing ? NaN : inc.eval.Delta,
            gamma_prime=inc === nothing ? NaN : inc.eval.gamma_prime_j,
            outer_feasible=inc === nothing ? false : inc.classification.outer_feasible))
    end

    println()
    println("="^100)
    println("Section 4 benchmark summary")
    println("="^100)
    @printf("%-8s %-6s %-18s %8s %8s %8s %8s %8s %8s %14s %12s\n",
        "delta", "dir", "backend", "wall(s)", "nStatus", "FC", "GA", "inner", "infeas", "Delta", "gamma_prime")
    for r in rows
        @printf("%-8.1e %-6s %-18s %8.1f %8d %8d %8d %8d %8d %14.6e %12.6f\n",
            r.delta, r.direction, r.backend, r.wall, r.nStatus, r.fc_calls, r.ga_calls,
            r.inner_solves, r.inner_infeas, r.Delta, r.gamma_prime)
    end

    # matched-pair comparison
    println()
    println("Matched-pair deltas (linear vs nonlinear_reference, same delta/direction):")
    for delta in deltas, direction in directions
        rnl = only(filter(r -> r.delta == delta && r.direction == direction && r.backend == :nonlinear_reference, rows))
        rlin = only(filter(r -> r.delta == delta && r.direction == direction && r.backend == :linear, rows))
        speedup = rnl.wall / rlin.wall
        @printf("  delta=%.1e dir=%s: wall %.1fs -> %.1fs (%.2fx), Delta match=%s, gamma_prime match=%s\n",
            delta, direction, rnl.wall, rlin.wall, speedup,
            isapprox(rnl.Delta, rlin.Delta; rtol=0.05), isapprox(rnl.gamma_prime, rlin.gamma_prime; rtol=0.05))
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
