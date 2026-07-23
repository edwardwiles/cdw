# Session driver for the governing prompt's Sections 7-10:
#   - Section 7/8: profile representative fixed points and a couple of short outer
#     trajectories using the new `MELITZ_PROFILE`/`@melitz_profile` instrumentation
#     (profiling.jl), and quantify the upper-vs-lower wall-time asymmetry.
#   - Section 9: restricted nuisance-coordinate searches (G / GA / GF / GQ / GAF / GAQ)
#     at a matched delta/direction/W, via a per-coordinate `theta_box` vector (0.0 pins a
#     coordinate at its theta0 value -- the same "lo==up" mechanism
#     `melitz_fixed_point_probe` already uses for a full-point probe).
#   - Section 10: matched live :logf+:linear vs :logcutoff+:linear comparison at
#     D=4, W=20,000, seed=29, delta in {1e-3,1e-2}, both directions, with economically
#     CALIBRATED (not raw-identical) coordinate boxes -- see BOX_Q_SCALE below.
#
# Usage: julia --project=. scripts/melitz_profile_and_compare.jl

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
include(joinpath(MELITZ_DIR, "log_cutoff_param.jl"))
include(joinpath(MELITZ_DIR, "fake_data.jl"))
include(joinpath(MELITZ_DIR, "gradient_lab.jl"))
include(joinpath(MELITZ_DIR, "outer_solve.jl"))
include(joinpath(MELITZ_DIR, "inner_screening.jl"))
include(joinpath(MELITZ_DIR, "origin_block_screen.jl"))
include(joinpath(MELITZ_DIR, "localized_gradient.jl"))
include(joinpath(MELITZ_DIR, "finite_delta_outer.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const D = 4
const SIGMA = 2.5
const THETA_STAR = 6.8
const TARGET = 1
const SEED = 29
const W = 20_000
const INNER_OPT = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")
const OUTER_OPT = joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt")
# Section 10: q's elasticity w.r.t. log f is (sigma-1) (melitz_log_f_from_q), so a raw
# theta_box on q of the SAME numeric size as :logf's f-box would induce (sigma-1)x LARGER
# f-swings, not a matched comparison. Scale q's box down by 1/(sigma-1) to induce a
# comparable distribution of Delta(log f).
const BOX_Q_SCALE = 1 / (SIGMA - 1)

function nA_nF(D)
    nA = D^2 - 1
    nF = D^2 - 2
    return nA, nF
end

"Box vector for a restricted search: box_gamma at [1], box_A over A_free, box_f over f/q_free_free, zeros elsewhere for coordinates in `active`."
function restricted_box(n::Int, D::Int; active::Symbol, box_gamma::Real=0.10, box_A::Real=0.10, box_f::Real=0.10)
    nA, nF = nA_nF(D)
    box = zeros(n)
    active in (:G, :GA, :GF, :GQ, :GAF, :GAQ) || error("unknown active set $active")
    if active in (:G, :GA, :GF, :GQ, :GAF, :GAQ)
        box[1] = box_gamma
    end
    if active in (:GA, :GAF, :GAQ)
        box[2:1+nA] .= box_A
    end
    if active in (:GF, :GQ, :GAF, :GAQ)
        box[2+nA:end] .= box_f
    end
    return box
end

function coord_movement(theta0::Vector{Float64}, theta1::Vector{Float64}, D::Int)
    nA, nF = nA_nF(D)
    d = theta1 .- theta0
    dgamma = abs(d[1])
    dA = @view d[2:1+nA]
    df = @view d[2+nA:end]
    return (dgamma=dgamma, dA_l2=norm(dA), dA_linf=maximum(abs.(dA)),
            df_l2=norm(df), df_linf=maximum(abs.(df)))
end

using LinearAlgebra: norm

function main()
    println("="^100)
    println("Melitz Sections 7-10 live driver: profiling, restricted search, :logf vs :logcutoff")
    println("="^100)
    @printf("D=%d sigma=%.2f theta_star=%.2f target=%d seed=%d W=%d\n", D, SIGMA, THETA_STAR, TARGET, SEED, W)

    data = generate_fake_melitz_data(; D=D, sigma=SIGMA, theta_star=THETA_STAR,
        target_country=TARGET, seed=SEED, W=W)

    obj_f, theta0_f = build_melitz_psi_bundle(data; outer_parameterization=:logf, inner_loop_opt=INNER_OPT)
    ctx_f = obj_f.γ
    obj_q, theta0_q = build_melitz_psi_bundle(data; outer_parameterization=:logcutoff, inner_loop_opt=INNER_OPT)
    ctx_q = obj_q.γ
    n = length(theta0_f)

    # ------------------------------------------------------------------------
    # Section 7/8.1: profile the population-Pareto fixed point under both
    # parameterizations (a single real KNITRO fixed-point probe each).
    # ------------------------------------------------------------------------
    println()
    println("#"^100)
    println("Section 7/8: fixed-point profiling -- population-Pareto point")
    println("#"^100)
    for (label, ctx, obj, theta0) in (("logf", ctx_f, obj_f, theta0_f), ("logcutoff", ctx_q, obj_q, theta0_q))
        melitz_profile_reset!()
        MELITZ_PROFILE[] = true
        t0 = time()
        probe = melitz_fixed_point_probe(ctx, obj, theta0; delta=1e-2, direction=:upper,
            inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT)
        wall = time() - t0
        MELITZ_PROFILE[] = false
        @printf("\n[%s] population-Pareto fixed-point probe: nStatus=%d eval_failed=%s wall=%.3fs\n",
            label, probe.nStatus, probe.eval_failed, wall)
        melitz_profile_report(; trajectory_total_s=wall)
    end

    # ------------------------------------------------------------------------
    # Section 8: two short full outer trajectories (upper vs lower) at delta=1e-2,
    # :logf + :linear (the trusted production baseline), to quantify the asymmetry.
    # ------------------------------------------------------------------------
    println()
    println("#"^100)
    println("Section 8: upper vs lower asymmetry -- short full trajectories, delta=1e-2, :logf+:linear")
    println("#"^100)
    asym_rows = NamedTuple[]
    for direction in (:upper, :lower)
        melitz_profile_reset!()
        MELITZ_PROFILE[] = true
        t0 = time()
        res = solve_melitz_finite_delta_bound(ctx_f, obj_f, theta0_f; delta=1e-2, direction=direction,
            cutoff_constraint_backend=:linear, theta_box=0.10,
            inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT)
        wall = time() - t0
        MELITZ_PROFILE[] = false
        @printf("\n[%s] wall=%.2fs nStatus=%d inner_solves=%d (infeas=%d, eval_fail=%d) FC=%d GA=%d\n",
            direction, wall, res.nStatus, res.inner_solve_count, res.inner_infeas_count,
            res.inner_eval_failures, res.n_fc_calls, res.n_ga_calls)
        melitz_profile_report(; trajectory_total_s=wall)
        push!(asym_rows, (direction=direction, wall=wall, inner_solves=res.inner_solve_count,
            infeas=res.inner_infeas_count, fc=res.n_fc_calls, ga=res.n_ga_calls,
            per_inner_solve=wall / max(1, res.inner_solve_count)))
    end
    println("\nAsymmetry summary:")
    for r in asym_rows
        @printf("  %-6s wall=%.2fs inner_solves=%d infeas=%d wall/inner_solve=%.4fs\n",
            r.direction, r.wall, r.inner_solves, r.infeas, r.per_inner_solve)
    end
    if length(asym_rows) == 2
        ratio = asym_rows[2].per_inner_solve / asym_rows[1].per_inner_solve
        @printf("  lower/upper wall-per-inner-solve ratio = %.3fx\n", ratio)
    end

    # ------------------------------------------------------------------------
    # Section 9: restricted nuisance-coordinate searches at delta=1e-2, direction=:upper.
    # ------------------------------------------------------------------------
    println()
    println("#"^100)
    println("Section 9: restricted nuisance-coordinate searches, delta=1e-2, direction=upper")
    println("#"^100)
    restricted_rows = NamedTuple[]
    for active in (:G, :GA, :GF, :GAF)
        box = restricted_box(n, D; active=active, box_gamma=0.10, box_A=0.10, box_f=0.10)
        t0 = time()
        res = solve_melitz_finite_delta_bound(ctx_f, obj_f, theta0_f; delta=1e-2, direction=:upper,
            cutoff_constraint_backend=:linear, theta_box=box,
            inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT)
        wall = time() - t0
        inc = res.cold_verified_incumbent
        if inc === nothing
            @printf("[%s:logf] ** no cold-verified incumbent ** wall=%.2fs\n", active, wall)
            push!(restricted_rows, (active=active, param=:logf, wall=wall, gamma=NaN, Delta=NaN, feasible=false))
        else
            e = inc.eval
            mv = coord_movement(theta0_f, e.theta_free, D)
            @printf("[%-4s:logf] wall=%5.2fs gamma_prime=%.6f Delta=%.3e feasible=%s dA_l2=%.4f df_l2=%.4f\n",
                active, wall, e.gamma_prime_j, e.Delta, inc.classification.outer_feasible, mv.dA_l2, mv.df_l2)
            push!(restricted_rows, (active=active, param=:logf, wall=wall, gamma=e.gamma_prime_j,
                Delta=e.Delta, feasible=inc.classification.outer_feasible))
        end
    end
    for active in (:G, :GA, :GQ, :GAQ)
        box = restricted_box(n, D; active=active, box_gamma=0.10, box_A=0.10, box_f=0.10 * BOX_Q_SCALE)
        t0 = time()
        res = solve_melitz_finite_delta_bound(ctx_q, obj_q, theta0_q; delta=1e-2, direction=:upper,
            cutoff_constraint_backend=:linear, theta_box=box,
            inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT)
        wall = time() - t0
        inc = res.cold_verified_incumbent
        if inc === nothing
            @printf("[%s:logcutoff] ** no cold-verified incumbent ** wall=%.2fs\n", active, wall)
            push!(restricted_rows, (active=active, param=:logcutoff, wall=wall, gamma=NaN, Delta=NaN, feasible=false))
        else
            e = inc.eval
            @printf("[%-4s:logcutoff] wall=%5.2fs gamma_prime=%.6f Delta=%.3e feasible=%s\n",
                active, wall, e.gamma_prime_j, e.Delta, inc.classification.outer_feasible)
            push!(restricted_rows, (active=active, param=:logcutoff, wall=wall, gamma=e.gamma_prime_j,
                Delta=e.Delta, feasible=inc.classification.outer_feasible))
        end
    end
    println("\nSection 9 summary (upper direction, delta=1e-2):")
    @printf("%-8s %-10s %8s %14s %14s %10s\n", "active", "param", "wall(s)", "gamma_prime", "Delta", "feasible")
    for r in restricted_rows
        @printf("%-8s %-10s %8.2f %14.6f %14.3e %10s\n", r.active, r.param, r.wall, r.gamma, r.Delta, r.feasible)
    end

    # ------------------------------------------------------------------------
    # Section 10: matched :logf+:linear vs :logcutoff+:linear at delta in {1e-3,1e-2},
    # both directions.
    # ------------------------------------------------------------------------
    println()
    println("#"^100)
    println("Section 10: matched :logf vs :logcutoff comparison, delta in {1e-3,1e-2}, both directions")
    println("#"^100)
    cmp_rows = NamedTuple[]
    for delta in (1e-3, 1e-2), direction in (:upper, :lower)
        for (label, ctx, obj, theta0, box) in (
            ("logf", ctx_f, obj_f, theta0_f, 0.10),
            ("logcutoff", ctx_q, obj_q, theta0_q, restricted_box(n, D; active=:GAQ, box_gamma=0.10, box_A=0.10, box_f=0.10 * BOX_Q_SCALE)))
            melitz_profile_reset!()
            MELITZ_PROFILE[] = true
            t0 = time()
            res = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=delta, direction=direction,
                cutoff_constraint_backend=:linear, theta_box=box,
                inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT)
            wall = time() - t0
            MELITZ_PROFILE[] = false
            inc = res.cold_verified_incumbent
            row = if inc === nothing
                (delta=delta, direction=direction, param=label, wall=wall, nStatus=res.nStatus,
                 fc=res.n_fc_calls, ga=res.n_ga_calls, inner=res.inner_solve_count,
                 infeas=res.inner_infeas_count, Delta=NaN, gamma=NaN, min_slack=NaN, feasible=false)
            else
                e = inc.eval
                (delta=delta, direction=direction, param=label, wall=wall, nStatus=res.nStatus,
                 fc=res.n_fc_calls, ga=res.n_ga_calls, inner=res.inner_solve_count,
                 infeas=res.inner_infeas_count, Delta=e.Delta, gamma=e.gamma_prime_j,
                 min_slack=e.min_slack, feasible=inc.classification.outer_feasible)
            end
            push!(cmp_rows, row)
            @printf("[delta=%.0e dir=%-5s %-10s] wall=%6.2fs nStatus=%4d FC=%3d GA=%3d inner=%4d infeas=%3d Delta=%.3e gamma=%.6f feasible=%s\n",
                delta, direction, label, wall, res.nStatus, res.n_fc_calls, res.n_ga_calls,
                res.inner_solve_count, res.inner_infeas_count, row.Delta, row.gamma, row.feasible)
            println("  profiler breakdown:")
            melitz_profile_report(; trajectory_total_s=wall)
        end
    end

    println()
    println("="^100)
    println("Section 10 summary table")
    println("="^100)
    @printf("%-8s %-6s %-10s %8s %8s %6s %6s %6s %6s %14s %12s %10s\n",
        "delta", "dir", "param", "wall(s)", "nStatus", "FC", "GA", "inner", "infeas", "Delta", "gamma", "feasible")
    for r in cmp_rows
        @printf("%-8.1e %-6s %-10s %8.2f %8d %6d %6d %6d %6d %14.3e %12.6f %10s\n",
            r.delta, r.direction, r.param, r.wall, r.nStatus, r.fc, r.ga, r.inner, r.infeas,
            r.Delta, r.gamma, r.feasible)
    end
    return (asym=asym_rows, restricted=restricted_rows, comparison=cmp_rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
