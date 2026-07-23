# Phase I.2 (screening-session continuation): archive the residual NumericalFailure points
# from a real trajectory (with the current production screens live -- range + stored-dual,
# no lower_limit_guard, no dual polish) and run each in KNITRO-native diagnostic mode
# (outlev=6: full per-iteration Objective/FeasError/OptError/Step/CGits trace) across a
# {lower_limit in (none, -delta)} x {maxit in (25,50,100,250,10000)} grid, to answer
# concretely (not inferred from nStatus=-102 alone):
#
#   - Does the lower bound (-f) cross delta in the first few iterations?
#   - Does KNITRO then spend extra iterations trying to establish unboundedness?
#   - Does the raw dual objective tend to -Inf, or approach a finite limit with exploding
#     multipliers (||x|| growing without the objective itself diverging)?
#   - How many ms/iterations before the point is already certified unusable?
#
# Given this session's time budget, the grid is run on a representative subset (the LOWER
# direction, historically the more expensive/asymmetric one per
# docs/melitz_optimization_report_2026-07-23.md Section C) rather than the full
# 4-points x 2-directions x 2-lower_limit x 5-maxit = 80-cell grid -- reported honestly, see
# the session report's Section B.
#
# Usage: julia --project=. scripts/melitz_residual_failure_diagnostics.jl

using Printf, Random
include(joinpath(@__DIR__, "melitz_no_rescue_benchmark.jl"))   # brings in collect_numerical_failures, CS, etc.

const DIAG_DIR = dirname(@__DIR__)
const MAXIT_GRID = [25, 50, 100, 250, 10000]
const MAXIT_OPT_FILES = Dict(m => joinpath(DIAG_DIR, "melitz_inner_loop_options_diagnostic_maxit$(m).opt") for m in MAXIT_GRID)

"""
    diagnostic_probe(obj, theta, delta; maxit, use_lower_limit, log_path)

One diagnostic inner solve: builds a FRESH bundle sharing `obj`'s ctx/draws, sets
`lower_limit = use_lower_limit ? -delta : -KNITRO.KN_INFINITY`, swaps in the maxit/outlev=6
option file for `maxit`, redirects KNITRO's native iteration log to `log_path` (`outmode
file`, via a per-call option override), runs ONE inner solve at `theta`, and returns
`(nStatus, elapsed_s, log_path)`.
"""
function diagnostic_probe(ctx, U, theta::AbstractVector, delta::Real; maxit::Int,
                            use_lower_limit::Bool, log_path::AbstractString,
                            gradient_backend::Symbol=:B, h::Real=1e-4)
    outer_opt = joinpath(DIAG_DIR, "melitz_outer_finite_delta.opt")
    obj = build_melitz_implicit_bundle(ctx, U, theta; delta=delta, find_smallest=true,
        gradient_backend=gradient_backend, h=h,
        inner_loop_opt=MAXIT_OPT_FILES[maxit], outer_loop_opt=outer_opt,
        lower_limit_guard=use_lower_limit ? 0.0 : nothing)
    G_now = CS.select_G_from_H(obj, obj.H)
    obj.moments!(@view(obj.H[:, 1]), G_now, theta, obj.U, obj)
    obj.H[:, 2] .= 1.0

    # Capture KNITRO's native (outmode=screen, outlev=6) per-iteration stdout directly via
    # Julia's own redirect_stdout -- more reliable across KNITRO versions than relying on its
    # own outmode=file/outname option handling (a first attempt at the latter silently wrote
    # no file; not further debugged given this session's time budget).
    t0 = time_ns()
    nStatus, objSol, x, lambda_ = open(log_path, "w") do io
        redirect_stdout(io) do
            CS.inner_loop_KNITRO(obj)
        end
    end
    elapsed = (time_ns() - t0) / 1e9
    return (nStatus=Int(nStatus), elapsed_s=elapsed, log_path=log_path,
            threshold_crossed=obj.threshold_crossed[],
            threshold_bound=obj.threshold_crossing_bound[])
end

"""
Parses a KNITRO outlev=6 per-iteration log: returns (n_iters, first_iter_lb_exceeds_delta,
final_objective, exploding_multipliers::Bool, exit_line).
`-f` (the raw dual objective's negation) is the Delta lower bound; a "first crossing"
iteration is the first row whose `-Objective > delta`.
"""
function parse_knitro_iter_log(log_path::AbstractString, delta::Real)
    isfile(log_path) || return (n_iters=0, first_crossing_iter=nothing, final_obj=NaN,
        max_abs_obj=NaN, exit_line="(no log file written)")
    lines = readlines(log_path)
    n_iters = 0
    first_crossing = nothing
    final_obj = NaN
    max_abs_obj = 0.0
    exit_line = ""
    for line in lines
        m = match(r"^\s*(\d+)\s+(-?[\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)?\s*", line)
        if m !== nothing && all(c -> isdigit(c) || c == ' ', collect(strip(line))[1:min(6, end)])
            iter = tryparse(Int, m.captures[1])
            obj_val = tryparse(Float64, m.captures[2])
            if iter !== nothing && obj_val !== nothing
                n_iters = max(n_iters, iter)
                final_obj = obj_val
                max_abs_obj = max(max_abs_obj, abs(obj_val))
                lb = -obj_val
                if first_crossing === nothing && lb > delta
                    first_crossing = iter
                end
            end
        end
        if startswith(strip(line), "EXIT:")
            exit_line = strip(line)
        end
    end
    return (n_iters=n_iters, first_crossing_iter=first_crossing, final_obj=final_obj,
            max_abs_obj=max_abs_obj, exit_line=exit_line)
end

function run_residual_failure_diagnostics(; D=4, W=20_000, seed=29, delta=1e-2,
                                            direction=:lower, max_points=4,
                                            out_dir=joinpath(DIAG_DIR, "results", "phase1_2_diagnostics"))
    mkpath(out_dir)
    println("="^100)
    @printf("Phase I.2: archiving residual NumericalFailure points, direction=%s delta=%.1e\n", direction, delta)
    println("="^100)

    obj0, ctx, failures = collect_numerical_failures(; D=D, W=W, seed=seed, delta=delta,
        direction=direction, max_points=max_points)
    if isempty(failures)
        println("No NumericalFailure points observed -- nothing to diagnose.")
        return NamedTuple[]
    end

    # Archive point metadata (theta, starting dual, delta, parameterization, moment scaling)
    # per main prompt Section 2.
    for (i, pt) in enumerate(failures)
        open(joinpath(out_dir, "point_$(direction)_$(i)_meta.txt"), "w") do io
            println(io, "direction=$direction delta=$delta D=$D W=$W seed=$seed")
            println(io, "parameterization=:logf gradient_backend=:B h=1e-4")
            println(io, "theta=", pt.theta)
            println(io, "x_warm(starting dual)=", pt.x_warm)
        end
    end

    rows = NamedTuple[]
    for (i, pt) in enumerate(failures)
        theta = pt.theta
        println("\n-- residual point $i / $(length(failures)), direction=$direction --")
        for use_ll in (false, true), maxit in MAXIT_GRID
            log_path = joinpath(out_dir, "point_$(direction)_$(i)_maxit$(maxit)_ll$(use_ll).log")
            r = diagnostic_probe(ctx, obj0.U, theta, delta; maxit=maxit, use_lower_limit=use_ll,
                log_path=log_path)
            trace = parse_knitro_iter_log(log_path, delta)
            @printf("  maxit=%-6d lower_limit=%-5s nStatus=%-5d t=%7.3fs iters=%-4d first_cross_iter=%-6s final_obj=%.4e exit=%s\n",
                maxit, use_ll, r.nStatus, r.elapsed_s, trace.n_iters,
                string(trace.first_crossing_iter), trace.final_obj, trace.exit_line)
            push!(rows, (point=i, direction=direction, maxit=maxit, use_lower_limit=use_ll,
                nStatus=r.nStatus, elapsed_s=r.elapsed_s, n_iters=trace.n_iters,
                first_crossing_iter=trace.first_crossing_iter, final_obj=trace.final_obj,
                max_abs_obj=trace.max_abs_obj, exit_line=trace.exit_line,
                threshold_crossed=r.threshold_crossed, threshold_bound=r.threshold_bound))
        end
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    direction = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :lower
    max_points = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
    rows = run_residual_failure_diagnostics(; direction=direction, max_points=max_points)
    println("\n", length(rows), " diagnostic cells recorded.")
end
