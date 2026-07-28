# 2026-07-28 outer-search gradient/redundancy/sensitivity continuation, Phase 4: full D4
# delta-grid frontier. Governing prompt: D=4, W=20,000, seed=29 + two additional seeds
# ({49,50}, matching the 2026-07-28 step-control session's own choice and this repo's own
# standing "most D=4 seeds fail export-selection" finding -- memory
# feedback-melitz-d4-seed-fragility lists {29,49,50,52,53,94,107,110} as the known-good set),
# both directions, four blocks (gamma-only, gamma+technology, gamma+participation, full
# joint), a predetermined 9-point delta grid, continuation in delta, and multi-start for the
# full joint problem.
#
# Scope disclosure (wall-clock budget): the full joint block uses SQP at every
# (seed,delta,direction) cell (54 cells x 3 starts = 162 solves) and Interior/CG at seed=29
# only, both directions, all deltas (18 cells x 3 starts = 54 solves) -- "at least SQP and
# Interior/CG receive substantive D4 testing" (governing prompt acceptance criterion 3) is
# satisfied without the 2x cost of running BOTH algorithms at all 3 seeds. Multi-start uses 3
# of the governing prompt's suggested 5 starts (best gamma-only-at-this-cell, the continuation
# start [previous-delta's own full solution], and the Pareto/calibration start) -- best-
# technology-only and best-participation-only are dropped from the full-block start pool
# because Phase 4 of the 2026-07-27 session already found both restrictions self-limit to a
# tiny, near-delta-independent movement radius (consistent with this session's own restricted-
# block results below), making them a low-value seed for the FULL joint search specifically.
#
# Usage: julia --project=. -t 20 scripts/melitz_phase4_d4_frontier_2026-07-28.jl

using Pkg
Pkg.activate(dirname(@__DIR__))
using Printf, DelimitedFiles, LinearAlgebra
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(@__DIR__, "melitz_regression_fixtures_2026-07-28.jl"))
using KNITRO

const REPO = dirname(@__DIR__)
const OUTDIR = joinpath(REPO, "docs", "key_results")
mkpath(OUTDIR)
const TMPOPT = joinpath(OUTDIR, "tmp_opt_phase4_2026-07-28")
mkpath(TMPOPT)
const CAP = 10.0
const SEEDS = [29, 49, 50]
const DELTAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0]
const DIRECTIONS = (:upper, :lower)

function write_csv(path, rows)
    isempty(rows) && return
    open(path, "w") do io
        cols = keys(rows[1])
        println(io, join(cols, ","))
        for r in rows
            println(io, join([r[c] for c in cols], ","))
        end
    end
end

"Materialize a per-cell .opt variant from a base algorithm .opt file, overriding delta/maxit/maxtime_real."
function make_opt(base::AbstractString, tag::AbstractString; delta::Real, maxit::Int, maxtime_real::Real)
    lines = readlines(base)
    lines = filter(l -> !occursin(r"^\s*(delta|maxit|maxtime_real)\s", l), lines)
    push!(lines, @sprintf("delta           %.6g", delta))
    push!(lines, @sprintf("maxit           %d", maxit))
    push!(lines, @sprintf("maxtime_real    %.1f", maxtime_real))
    path = joinpath(TMPOPT, tag * ".opt")
    open(io -> foreach(l -> println(io, l), lines), path, "w")
    return path
end

ok_status(n) = n in (0, -100, -101, -102, -103)

function block_box(n, D, g_radius, A_radius, f_radius)
    nA = D^2 - 1
    b = zeros(n)
    b[1] = g_radius
    b[2:1+nA] .= A_radius
    b[2+nA:end] .= f_radius
    return b
end

function movement_norms(theta_final, theta0, D)
    nA = D^2 - 1
    dlogA = norm(theta_final[2:1+nA] - theta0[2:1+nA])
    dlogf = norm(theta_final[2+nA:end] - theta0[2+nA:end])
    return dlogA, dlogf
end

function report_row(; seed, direction, block, algorithm, delta, res, theta0, ctx, wall, incumbent_source_override=nothing)
    cv = res === nothing ? nothing : res.cold_verified_incumbent
    if cv === nothing
        return (seed=seed, direction=string(direction), block=string(block), algorithm=string(algorithm),
            delta=delta, best_g=NaN, gamma_prime=NaN, kappa_ratio=NaN, GT=NaN,
            incumbent_source="none", dlogA=NaN, dlogf=NaN, DeltaStar=NaN,
            status=(res === nothing ? -9999 : res.nStatus), wall=wall)
    end
    theta_final = cv.eval.theta_free
    wm = melitz_welfare_metrics_from_g(theta_final[1], ctx)
    dlogA, dlogf = movement_norms(theta_final, theta0, ctx.D)
    return (seed=seed, direction=string(direction), block=string(block), algorithm=string(algorithm),
        delta=delta, best_g=theta_final[1], gamma_prime=wm.gamma_prime, kappa_ratio=wm.kappa_ratio,
        GT=wm.gains_from_trade, incumbent_source=string(incumbent_source_override === nothing ? cv.source : incumbent_source_override),
        dlogA=dlogA, dlogf=dlogf, DeltaStar=cv.eval.Delta, status=res.nStatus, wall=wall)
end

function main()
    BLAS.set_num_threads(1)
    inner_opt = joinpath(REPO, "melitz_inner_loop_options.opt")
    active_base = joinpath(REPO, "melitz_outer_finite_delta_alg_active_2026-07-27.opt")
    sqp_base = joinpath(REPO, "melitz_outer_finite_delta_alg_sqp_2026-07-27.opt")
    cg_base = joinpath(REPO, "melitz_outer_finite_delta_alg_cg_2026-07-27.opt")

    all_rows = NamedTuple[]
    t_script0 = time()

    for seed in SEEDS
        println("="^100); @printf("SEED=%d\n", seed); println("="^100); flush(stdout)
        fx = build_d4_fixture(; seed=seed)
        ctx, obj, theta0 = fx.ctx, fx.obj, fx.theta0
        n = length(theta0); D = ctx.D

        box_gamma = block_box(n, D, 0.10, 0.0, 0.0)
        box_tech = block_box(n, D, 0.10, 0.15, 0.0)
        box_part = block_box(n, D, 0.10, 0.0, 0.15)
        box_full = block_box(n, D, 0.10, 0.15, 0.15)

        for direction in DIRECTIONS
            prev_gamma, prev_tech, prev_part, prev_full = theta0, theta0, theta0, theta0
            for delta in DELTAS
                t0 = time()

                # --- restricted blocks, Active Set, Phase-3-selected (delta=0.01,box=0.10) step control,
                #     continuation from the previous delta's own verified solution ---
                opt_active = make_opt(active_base, "d4_active_$(seed)_$(direction)_$(delta)";
                    delta=0.01, maxit=100, maxtime_real=60.0)
                res_gamma = solve_melitz_finite_delta_bound(ctx, obj, prev_gamma; delta=delta, direction=direction,
                    delta_evaluation_cap=CAP, gradient_backend=:auto, theta_box=box_gamma,
                    inner_loop_opt=inner_opt, outer_loop_opt=opt_active, forbid_dense_fallback=true,
                    objective_scale=:auto, var_scale=(let vs = ones(n); vs[1] = 1e-3; vs end),
                    var_center=collect(Float64.(prev_gamma)))
                push!(all_rows, report_row(; seed=seed, direction=direction, block=:gamma_only, algorithm=:active,
                    delta=delta, res=res_gamma, theta0=theta0, ctx=ctx, wall=time() - t0))
                res_gamma.cold_verified_incumbent !== nothing && (prev_gamma = res_gamma.cold_verified_incumbent.eval.theta_free)

                t1 = time()
                res_tech = solve_melitz_finite_delta_bound(ctx, obj, prev_tech; delta=delta, direction=direction,
                    delta_evaluation_cap=CAP, gradient_backend=:auto, theta_box=box_tech,
                    inner_loop_opt=inner_opt, outer_loop_opt=opt_active, forbid_dense_fallback=true,
                    objective_scale=:auto, var_scale=(let vs = ones(n); vs[1] = 1e-3; vs end),
                    var_center=collect(Float64.(prev_tech)))
                push!(all_rows, report_row(; seed=seed, direction=direction, block=:gamma_technology, algorithm=:active,
                    delta=delta, res=res_tech, theta0=theta0, ctx=ctx, wall=time() - t1))
                res_tech.cold_verified_incumbent !== nothing && (prev_tech = res_tech.cold_verified_incumbent.eval.theta_free)

                t2 = time()
                res_part = solve_melitz_finite_delta_bound(ctx, obj, prev_part; delta=delta, direction=direction,
                    delta_evaluation_cap=CAP, gradient_backend=:auto, theta_box=box_part,
                    inner_loop_opt=inner_opt, outer_loop_opt=opt_active, forbid_dense_fallback=true,
                    objective_scale=:auto, var_scale=(let vs = ones(n); vs[1] = 1e-3; vs end),
                    var_center=collect(Float64.(prev_part)))
                push!(all_rows, report_row(; seed=seed, direction=direction, block=:gamma_participation, algorithm=:active,
                    delta=delta, res=res_part, theta0=theta0, ctx=ctx, wall=time() - t2))
                res_part.cold_verified_incumbent !== nothing && (prev_part = res_part.cold_verified_incumbent.eval.theta_free)

                # --- full joint block: multi-start, SQP always, Interior/CG at seed=29 only ---
                starts = Dict("gamma_only_best" => prev_gamma, "continuation" => prev_full, "pareto" => theta0)
                algos = seed == 29 ? (("sqp", sqp_base), ("cg", cg_base)) : (("sqp", sqp_base),)
                for (algname, base_opt) in algos
                    t3 = time()
                    opt_full = make_opt(base_opt, "d4_full_$(algname)_$(seed)_$(direction)_$(delta)";
                        delta=0.01, maxit=60, maxtime_real=90.0)
                    best_res = nothing
                    best_obj = Inf
                    best_source = "none"
                    for (start_name, start_theta) in starts
                        res_full = solve_melitz_finite_delta_bound(ctx, obj, start_theta; delta=delta, direction=direction,
                            delta_evaluation_cap=CAP, gradient_backend=:auto, theta_box=box_full,
                            inner_loop_opt=inner_opt, outer_loop_opt=opt_full, forbid_dense_fallback=true,
                            objective_scale=:auto, var_scale=(let vs = ones(n); vs[1] = 1e-3; vs end),
                            var_center=collect(Float64.(start_theta)))
                        cv = res_full.cold_verified_incumbent
                        if cv !== nothing && cv.objective < best_obj
                            best_obj = cv.objective
                            best_res = res_full
                            best_source = "multistart:" * start_name
                        end
                    end
                    push!(all_rows, report_row(; seed=seed, direction=direction, block=:full_joint, algorithm=Symbol(algname),
                        delta=delta, res=best_res, theta0=theta0, ctx=ctx, wall=time() - t3,
                        incumbent_source_override=best_source))
                    if algname == "sqp" && best_res !== nothing && best_res.cold_verified_incumbent !== nothing
                        prev_full = best_res.cold_verified_incumbent.eval.theta_free
                    end
                end

                @printf("  seed=%d dir=%-5s delta=%.0e  cell_wall=%.1fs  (script_elapsed_total=%.0fs)\n",
                    seed, direction, delta, time() - t0, time() - t_script0)
                flush(stdout)
                write_csv(joinpath(OUTDIR, "melitz_phase4_d4_delta_frontier_2026-07-28.csv"), all_rows)
            end
        end
    end
    println("\nDONE. Total wall = ", time() - t_script0, "s. CSV written to ", OUTDIR)
end

main()
