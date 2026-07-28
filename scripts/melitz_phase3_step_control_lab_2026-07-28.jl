# 2026-07-28 step-control/robustness session, Phase 3: gamma-only step-control laboratory.
#
# Uses the gamma-only restriction (A/f block of theta_box fixed at 0, exactly Phase 3 of the
# PRIOR 2026-07-28 gamma-profile session's own methodology) because the correct direction and
# approximate boundary are already known from the Phase 2 direct fixed-A/f profile
# (`docs/key_results/melitz_phase2_gamma_profile_{d4,realD20}_2026-07-28.csv`) -- this makes
# the sweep's own quality directly checkable via `classify_gamma_only_config`
# (`melitz_regression_fixtures_2026-07-28.jl`), not just "did KNITRO report success."
#
# KNITRO 13.0.1 API audit (re-confirmed directly against the installed
# /opt/shared_sw/knitro/13.0.1/include/knitro.h this session, not from memory): the ONLY
# trust-region-style lever this KNITRO version exposes beyond `algorithm` is `KN_PARAM_DELTA`
# ("delta" in an options file) -- no separate "maximum scaled step"/"line-search trial cap
# beyond LINESEARCH_MAXTRIALS" parameter exists (grepped explicitly for
# steplimit/trust/maxstep-style names; none found). The governing prompt's own two-stage
# design ("compare trust radii... then compare maximum steps for the best trust radius") is
# therefore implemented as: Stage 1 sweeps `delta` (KN_PARAM_DELTA) at a fixed, moderate
# `theta_box` (this repo's own established g_radius=0.3 at D4 / 0.15 at real D20); Stage 2
# sweeps `theta_box`'s own g-radius at the Stage-1-selected `delta` -- a disclosed, real-lever
# substitution for "maximum scaled step" (which does not exist as a distinct KNITRO 13.0.1
# option), not an invented option name.
#
# Per-trial logging: KNITRO's own internal per-iteration/per-line-search-trial state (raw
# ||Step||, predicted vs realized constraint change) is only available as OUTLEV console text
# in this codebase's own wiring, not as a structured callback -- `on_inner_result` (the one
# structured hook `solve_melitz_finite_delta_bound` exposes, `inner_screening.jl`) fires once
# per FC/GA-classified inner-solve outcome (which, empirically, corresponds closely to one
# KNITRO iteration/line-search trial each -- confirmed against console `n_fc`/`n_ga` counts in
# prior sessions), giving trial-indexed g/dg/classification/Delta/lower-bound logging. This is
# a disclosed, real, but coarser substitute for a genuine per-iteration internal KNITRO log,
# not the literal iteration counter KNITRO itself uses.
#
# Usage: julia --project=. -t 20 scripts/melitz_phase3_step_control_lab_2026-07-28.jl

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
const TMPOPT = joinpath(REPO, "docs", "key_results", "tmp_opt_2026-07-28")
mkpath(TMPOPT)
const S_G = 1e-4
const CAP = 10.0

const ALG_FILES = Dict(
    :active => joinpath(REPO, "melitz_outer_finite_delta_alg_active_2026-07-27.opt"),
    :sqp => joinpath(REPO, "melitz_outer_finite_delta_alg_sqp_2026-07-27.opt"),
    :interior_cg => joinpath(REPO, "melitz_outer_finite_delta_alg_cg_2026-07-27.opt"),
    :interior_direct => joinpath(REPO, "melitz_outer_finite_delta_alg_direct_2026-07-27.opt"),
)

"Copy a base algorithm .opt file with `delta <val>` appended (removing any pre-existing `delta` line first -- none of this repo's own algorithm .opt files set it, confirmed by grep, so this is always a pure addition)."
function opt_with_delta(base_opt::AbstractString, delta_val::Real)
    lines = readlines(base_opt)
    lines = filter(l -> !occursin(r"^\s*delta\s", l), lines)
    push!(lines, @sprintf("delta           %.4g", delta_val))
    path = joinpath(TMPOPT, splitext(basename(base_opt))[1] * "_delta$(delta_val).opt")
    open(path, "w") do io
        for l in lines
            println(io, l)
        end
    end
    return path
end

gamma_only_box(n, g_radius) = (b = zeros(n); b[1] = g_radius; b)

mutable struct TrialLogger
    rows::Vector{NamedTuple}
    g0::Float64
    counter::Int
end
TrialLogger(g0) = TrialLogger(NamedTuple[], g0, 0)
function (logger::TrialLogger)(theta, result)
    logger.counter += 1
    dg = theta[1] - logger.g0
    cls, Delta, lb = if result isa FiniteSolved
        ("FiniteSolved", result.Delta, NaN)
    elseif result isa AboveEvaluationCap
        ("AboveEvaluationCap", NaN, result.certified_lower_bound)
    elseif result isa InfiniteDeltaCertified
        ("InfiniteDeltaCertified", NaN, NaN)
    else
        ("NumericalFailure", NaN, NaN)
    end
    push!(logger.rows, (trial=logger.counter, g=theta[1], dg=dg, classification=cls, Delta=Delta, lower_bound=lb))
end

function run_config(label, ctx, obj, theta0, delta, direction, g_radius, inner_opt, outer_opt;
                     var_scale, gradient_backend=:B_direct_argument_sorted_serial)
    n = length(theta0)
    box = gamma_only_box(n, g_radius)
    vc = var_scale === nothing ? nothing : collect(Float64.(theta0))
    logger = TrialLogger(theta0[1])
    local res
    wall = @elapsed begin
        res = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=delta, direction=direction,
            delta_evaluation_cap=CAP, gradient_backend=gradient_backend, h=1e-4,
            theta_box=box, cutoff_constraint_backend=:linear,
            inner_loop_opt=inner_opt, outer_loop_opt=outer_opt,
            var_scale=var_scale, var_center=vc,
            backend=:matrix_free, forbid_dense_fallback=true,
            objective_scale=:auto, on_inner_result=logger)
    end
    cv = res.cold_verified_incumbent
    best_g = cv === nothing ? theta0[1] : cv.eval.theta_free[1]
    best_Delta = cv === nothing ? NaN : cv.eval.Delta
    exploded = maximum(abs.(r.g - theta0[1] for r in logger.rows); init=0.0) > 5 * g_radius
    @printf("  [%-40s] nStatus=%5d wall=%6.2fs n_fc=%3d n_ga=%3d n_cap=%3d n_trials=%3d  g0=%.6f -> best_g=%.6f  Delta=%s  exploded=%s\n",
        label, res.nStatus, wall, res.n_fc_calls, res.n_ga_calls, res.n_above_cap_reject, length(logger.rows),
        theta0[1], best_g, isnan(best_Delta) ? "NA" : @sprintf("%.4e", best_Delta), exploded)
    flush(stdout)
    return (config=label, nStatus=res.nStatus, wall=wall, n_fc=res.n_fc_calls, n_ga=res.n_ga_calls,
        n_above_cap=res.n_above_cap_reject, n_numerical_failure=res.n_numerical_failure_reject,
        n_trials=length(logger.rows), g0=theta0[1], best_g=best_g, best_Delta=best_Delta,
        exploded=exploded, terminal_g=res.terminal_eval.theta_free[1],
        objective_scale_resolved=res.objective_scale_resolved), logger.rows
end

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

function main()
    BLAS.set_num_threads(1)
    d4 = build_d4_fixture()
    inner_opt4 = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
    n4 = length(d4.theta0)
    vs4 = ones(n4); vs4[1] = S_G
    profile_d4 = load_gamma_profile_csv(joinpath(OUTDIR, "melitz_phase2_gamma_profile_d4_2026-07-28.csv"))

    summary_rows = NamedTuple[]
    trial_rows = NamedTuple[]

    println("="^100); println("D4 STAGE 1: initial trust radius sweep at delta=1e-2, upper, moderate box (g_radius=0.3)"); println("="^100)
    delta_candidates = [0.01, 0.05, 0.10]
    box_candidates = [0.10, 0.25, 0.50]
    best_per_alg = Dict{Symbol,Float64}()
    for alg in (:active, :sqp, :interior_cg)
        best_score = -Inf
        for dcand in delta_candidates
            opt = opt_with_delta(ALG_FILES[alg], dcand)
            r, trials = run_config("D4_stage1_$(alg)_delta$(dcand)", d4.ctx, d4.obj, d4.theta0, 1e-2, :upper, 0.3, inner_opt4, opt; var_scale=vs4)
            cls, detail = classify_gamma_only_config(r.g0, r.best_g, r.best_Delta, profile_d4)
            push!(summary_rows, merge(r, (stage=1, algorithm=alg, delta_param=dcand, box_radius=0.3, classification=cls, detail=detail)))
            append!(trial_rows, [merge(t, (config=r.config,)) for t in trials])
            score = r.exploded ? -1e9 : (cls == :passed ? abs(r.best_g - r.g0) : (cls == :stuck_at_pareto ? -1e6 : -1e3))
            if score > best_score
                best_score = score
                best_per_alg[alg] = dcand
            end
        end
    end
    println("Stage 1 selected delta per algorithm: ", best_per_alg)

    println("\n" * "="^100); println("D4 STAGE 2: box-radius sweep at Stage-1-selected delta, delta_budget=1e-2, upper"); println("="^100)
    best_box_per_alg = Dict{Symbol,Float64}()
    for alg in (:active, :sqp, :interior_cg)
        dcand = best_per_alg[alg]
        opt = opt_with_delta(ALG_FILES[alg], dcand)
        best_score = -Inf
        for bcand in box_candidates
            r, trials = run_config("D4_stage2_$(alg)_box$(bcand)", d4.ctx, d4.obj, d4.theta0, 1e-2, :upper, bcand, inner_opt4, opt; var_scale=vs4)
            cls, detail = classify_gamma_only_config(r.g0, r.best_g, r.best_Delta, profile_d4)
            push!(summary_rows, merge(r, (stage=2, algorithm=alg, delta_param=dcand, box_radius=bcand, classification=cls, detail=detail)))
            append!(trial_rows, [merge(t, (config=r.config,)) for t in trials])
            score = r.exploded ? -1e9 : (cls == :passed ? abs(r.best_g - r.g0) : (cls == :stuck_at_pareto ? -1e6 : -1e3))
            if score > best_score
                best_score = score
                best_box_per_alg[alg] = bcand
            end
        end
    end
    println("Stage 2 selected box radius per algorithm: ", best_box_per_alg)

    println("\n" * "="^100); println("D4 confirmation grid: selected (delta,box) per algorithm x delta_budget{1e-3,1e-2,1} x direction{upper,lower}"); println("="^100)
    for alg in (:active, :sqp, :interior_cg)
        opt = opt_with_delta(ALG_FILES[alg], best_per_alg[alg])
        for delta_budget in (1e-3, 1e-2, 1.0), direction in (:upper, :lower)
            r, trials = run_config("D4_confirm_$(alg)_delta$(delta_budget)_$(direction)", d4.ctx, d4.obj, d4.theta0,
                delta_budget, direction, best_box_per_alg[alg], inner_opt4, opt; var_scale=vs4)
            cls, detail = classify_gamma_only_config(r.g0, r.best_g, r.best_Delta, profile_d4)
            push!(summary_rows, merge(r, (stage=3, algorithm=alg, delta_param=best_per_alg[alg],
                box_radius=best_box_per_alg[alg], classification=cls, detail=detail)))
            append!(trial_rows, [merge(t, (config=r.config,)) for t in trials])
        end
    end

    println("\n" * "="^100); println("real-D20 STAGE 1: initial trust radius sweep at delta=1.0, upper, moderate box (g_radius=0.15)"); println("="^100)
    BLAS.set_num_threads(20)
    inner_opt20 = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
    d20 = build_realD20_fixture()
    n20 = length(d20.theta0)
    vs20 = ones(n20); vs20[1] = S_G
    profile_d20 = load_gamma_profile_csv(joinpath(OUTDIR, "melitz_phase2_gamma_profile_realD20_2026-07-28.csv"))

    best_per_alg20 = Dict{Symbol,Float64}()
    for alg in (:active, :sqp)
        best_score = -Inf
        for dcand in delta_candidates
            opt = opt_with_delta(ALG_FILES[alg], dcand)
            r, trials = run_config("D20_stage1_$(alg)_delta$(dcand)", d20.ctx, d20.obj, d20.theta0, 1.0, :upper, 0.15, inner_opt20, opt; var_scale=vs20)
            cls, detail = classify_gamma_only_config(r.g0, r.best_g, r.best_Delta, profile_d20)
            push!(summary_rows, merge(r, (stage=1, algorithm=alg, delta_param=dcand, box_radius=0.15, classification=cls, detail=detail)))
            append!(trial_rows, [merge(t, (config=r.config,)) for t in trials])
            score = r.exploded ? -1e9 : (cls == :passed ? abs(r.best_g - r.g0) : (cls == :stuck_at_pareto ? -1e6 : -1e3))
            if score > best_score
                best_score = score
                best_per_alg20[alg] = dcand
            end
        end
    end
    println("real-D20 Stage 1 selected delta per algorithm: ", best_per_alg20)

    println("\n" * "="^100); println("real-D20 STAGE 2: box-radius sweep at Stage-1-selected delta"); println("="^100)
    best_box_per_alg20 = Dict{Symbol,Float64}()
    for alg in (:active, :sqp)
        dcand = best_per_alg20[alg]
        opt = opt_with_delta(ALG_FILES[alg], dcand)
        best_score = -Inf
        for bcand in box_candidates
            r, trials = run_config("D20_stage2_$(alg)_box$(bcand)", d20.ctx, d20.obj, d20.theta0, 1.0, :upper, bcand, inner_opt20, opt; var_scale=vs20)
            cls, detail = classify_gamma_only_config(r.g0, r.best_g, r.best_Delta, profile_d20)
            push!(summary_rows, merge(r, (stage=2, algorithm=alg, delta_param=dcand, box_radius=bcand, classification=cls, detail=detail)))
            append!(trial_rows, [merge(t, (config=r.config,)) for t in trials])
            score = r.exploded ? -1e9 : (cls == :passed ? abs(r.best_g - r.g0) : (cls == :stuck_at_pareto ? -1e6 : -1e3))
            if score > best_score
                best_score = score
                best_box_per_alg20[alg] = bcand
            end
        end
    end
    println("real-D20 Stage 2 selected box radius per algorithm: ", best_box_per_alg20)

    println("\n" * "="^100); println("real-D20 short reference: Interior/Direct at default box/delta (documented runaway check)"); println("="^100)
    opt_direct = ALG_FILES[:interior_direct]
    r, trials = run_config("D20_interior_direct_reference", d20.ctx, d20.obj, d20.theta0, 1.0, :upper, 0.15, inner_opt20, opt_direct; var_scale=vs20)
    cls, detail = classify_gamma_only_config(r.g0, r.best_g, r.best_Delta, profile_d20)
    push!(summary_rows, merge(r, (stage=0, algorithm=:interior_direct, delta_param=1.0, box_radius=0.15, classification=cls, detail=detail)))
    append!(trial_rows, [merge(t, (config=r.config,)) for t in trials])
    BLAS.set_num_threads(1)

    write_csv(joinpath(OUTDIR, "melitz_phase3_step_control_summary_2026-07-28.csv"), summary_rows)
    write_csv(joinpath(OUTDIR, "melitz_phase3_step_control_trials_2026-07-28.csv"), trial_rows)
    println("\nDONE. Best D4 config per algorithm: delta=", best_per_alg, " box=", best_box_per_alg)
    println("Best real-D20 config per algorithm: delta=", best_per_alg20, " box=", best_box_per_alg20)
    println("CSVs written to ", OUTDIR)
end

main()
