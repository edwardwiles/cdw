# 2026-07-28 step-control/robustness session, Phases 4-5: restricted D4 searches as a
# verified incumbent pool (Phase 4), then longer joint D4 searches that can never be
# reported worse than that pool (Phase 5, via `external_incumbent`).
#
# Reads Phase 3's own step-control summary CSV to pick, per algorithm, the selected
# (delta trust-radius, box) pair -- falls back to a documented-safe default
# (delta=0.05, box=0.25, matching Phase 3's own Stage-1/2 candidate midpoints) if Phase 3's
# CSV is not yet present, so this script can also be run standalone/re-run without a strict
# ordering dependency (disclosed, not a silent guess).
#
# Usage: julia --project=. -t 20 scripts/melitz_phase4_5_d4_restricted_and_joint_2026-07-28.jl

using Pkg
Pkg.activate(dirname(@__DIR__))
using Printf, DelimitedFiles, LinearAlgebra, Statistics
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
const SEEDS_D4 = [29, 49, 50]   # 29 primary; 49/50 the "two additional seeds"

const ALG_FILES = Dict(
    :active => joinpath(REPO, "melitz_outer_finite_delta_alg_active_2026-07-27.opt"),
    :sqp => joinpath(REPO, "melitz_outer_finite_delta_alg_sqp_2026-07-27.opt"),
    :interior_cg => joinpath(REPO, "melitz_outer_finite_delta_alg_cg_2026-07-27.opt"),
)

function opt_with_delta_maxit(base_opt::AbstractString, delta_val::Real, maxit_val::Int)
    lines = readlines(base_opt)
    lines = filter(l -> !occursin(r"^\s*delta\s", l) && !occursin(r"^\s*maxit\s", l), lines)
    push!(lines, @sprintf("delta           %.4g", delta_val))
    push!(lines, @sprintf("maxit           %d", maxit_val))
    path = joinpath(TMPOPT, splitext(basename(base_opt))[1] * "_delta$(delta_val)_maxit$(maxit_val).opt")
    open(path, "w") do io
        for l in lines
            println(io, l)
        end
    end
    return path
end

"""
Pick, per algorithm, the (delta,box) pair Phase 3's own Stage-2 sweep selected for D4. Hardcoded
directly from Phase 3's own console summary (`Stage 1 selected delta per algorithm:
Dict(:sqp=>0.01,:active=>0.01,:interior_cg=>0.01)`, `Stage 2 selected box radius per algorithm:
Dict(:sqp=>0.1,:active=>0.1,:interior_cg=>0.1)`, `melitz_phase3_step_control_lab_2026-07-28.jl`
run this session) rather than re-parsed from
`melitz_phase3_step_control_summary_2026-07-28.csv` at runtime: that CSV's own `detail` column
(from `classify_gamma_only_config`) contains embedded, unescaped commas in its `:passed` message
(a real bug found and fixed this session, `melitz_regression_fixtures_2026-07-28.jl`, AFTER
Phase 3 had already produced its CSV) -- `readdlm` is not comma-quote-aware, so re-reading that
specific CSV throws `duplicate field name in NamedTuple: "" is not unique`. Re-running Phase 3
(a ~40-minute campaign) purely to regenerate a byte-identical CSV with the fix applied was not
worth the wall-clock given the exact selected values are already known with certainty from the
same run's own console output -- disclosed, not a silent guess.
"""
function phase3_selection(alg::Symbol)
    d4_selection = Dict(:active => (delta=0.01, box=0.10), :sqp => (delta=0.01, box=0.10), :interior_cg => (delta=0.01, box=0.10))
    return get(d4_selection, alg, (delta=0.05, box=0.25))
end

function block_box(n::Int, D::Int, g_radius::Real, A_radius::Real, f_radius::Real)
    nA = D^2 - 1
    box = zeros(n)
    box[1] = g_radius
    box[2:1+nA] .= A_radius
    box[2+nA:end] .= f_radius
    return box
end

function nstatus_reason(n::Int)
    n == 0 && return "optimal"
    n == -101 && return "feas_xtol"
    n == -102 && return "feas_no_improve"
    n == -103 && return "feas_ftol"
    n == -200 && return "infeasible"
    n == -400 && return "iter_limit_feas"
    n == -410 && return "iter_limit_infeas"
    return "other($n)"
end

function run_search(label, ctx, obj, theta0, delta, direction, box, opt_file, inner_opt;
                     var_scale, external_incumbent=nothing)
    n = length(theta0)
    vc = var_scale === nothing ? nothing : collect(Float64.(theta0))
    local res
    wall = @elapsed begin
        res = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=delta, direction=direction,
            delta_evaluation_cap=CAP, gradient_backend=:B_direct_argument_sorted_serial, h=1e-4,
            theta_box=box, cutoff_constraint_backend=:linear,
            inner_loop_opt=inner_opt, outer_loop_opt=opt_file,
            var_scale=var_scale, var_center=vc,
            backend=:matrix_free, forbid_dense_fallback=true,
            objective_scale=:auto, external_incumbent=external_incumbent)
    end
    cv = res.cold_verified_incumbent
    D = ctx.D
    nA = D^2 - 1
    dg = cv === nothing ? 0.0 : cv.eval.theta_free[1] - theta0[1]
    dlogA = cv === nothing ? 0.0 : norm(cv.eval.theta_free[2:1+nA] - theta0[2:1+nA])
    dlogf = cv === nothing ? 0.0 : norm(cv.eval.theta_free[2+nA:end] - theta0[2+nA:end])
    incumbent_source = cv === nothing ? "none" : string(cv.source)
    @printf("  [%-55s] nStatus=%5s(%s) wall=%5.2fs n_fc=%3d n_ga=%3d n_solved=%3d n_cap=%3d n_numfail=%3d  dg=%+.5f  Delta=%s  src=%s\n",
        label, res.nStatus, nstatus_reason(res.nStatus), wall, res.n_fc_calls, res.n_ga_calls,
        res.n_inner_solved, res.n_above_cap_reject, res.n_numerical_failure_reject, dg,
        cv === nothing ? "NA" : @sprintf("%.4e", cv.eval.Delta), incumbent_source)
    flush(stdout)
    return (config=label, nStatus=res.nStatus, terminal_reason=nstatus_reason(res.nStatus), wall=wall,
        n_fc=res.n_fc_calls, n_ga=res.n_ga_calls, n_inner_solved=res.n_inner_solved,
        n_above_cap=res.n_above_cap_reject, n_numerical_failure=res.n_numerical_failure_reject,
        dg=dg, dlogA=dlogA, dlogf=dlogf,
        Delta=(cv === nothing ? NaN : cv.eval.Delta), incumbent_source=incumbent_source,
        cold_verified_theta=(cv === nothing ? nothing : cv.eval.theta_free)), res
end

function main()
    BLAS.set_num_threads(1)
    phase4_rows = NamedTuple[]
    phase5_rows = NamedTuple[]
    incumbent_pool = Dict{Tuple{Int,Float64,Symbol},Vector{Any}}()   # (seed,delta,direction) -> list of (label,theta,Delta)

    println("="^100); println("PHASE 4: restricted D4 incumbent pool"); println("="^100)
    sel_active = phase3_selection(:active)
    println("Phase 3 selection reused: active=", sel_active)

    for seed in SEEDS_D4
        d4 = build_d4_fixture(; seed=seed)
        n = length(d4.theta0)
        D = d4.ctx.D
        inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
        vs = ones(n); vs[1] = S_G; vs[2:end] .= 1e-5
        opt_file = opt_with_delta_maxit(ALG_FILES[:active], sel_active.delta, 25)
        for delta_budget in (1e-3, 1e-2, 1.0), direction in (:upper, :lower)
            key = (seed, delta_budget, direction)
            incumbent_pool[key] = Any[]
            for (restrict_label, A_r, f_r) in (("gamma_only", 0.0, 0.0), ("gamma_technology", 0.15, 0.0), ("gamma_participation", 0.0, 0.15))
                box = block_box(n, D, sel_active.box, A_r, f_r)
                label = "D4_seed$(seed)_delta$(delta_budget)_$(direction)_$(restrict_label)"
                row, res = run_search(label, d4.ctx, d4.obj, d4.theta0, delta_budget, direction, box, opt_file, inner_opt; var_scale=vs)
                push!(phase4_rows, merge(row, (seed=seed, delta_budget=delta_budget, direction=direction, restriction=restrict_label)))
                if row.cold_verified_theta !== nothing
                    push!(incumbent_pool[key], (restrict_label, row.cold_verified_theta, row.Delta))
                end
            end
        end
    end
    write_csv(joinpath(OUTDIR, "melitz_phase4_d4_restricted_incumbents_2026-07-28.csv"), phase4_rows)

    println("\n" * "="^100); println("PHASE 5: longer D4 joint searches (SQP + Interior/CG, maxit>=100)"); println("="^100)
    sel_sqp = phase3_selection(:sqp)
    sel_cg = phase3_selection(:interior_cg)
    println("Phase 3 selections reused: sqp=", sel_sqp, " interior_cg=", sel_cg)
    long_algs = Dict(:sqp => (ALG_FILES[:sqp], sel_sqp), :interior_cg => (ALG_FILES[:interior_cg], sel_cg))

    for alg in (:sqp, :interior_cg)
        base_opt, sel = long_algs[alg]
        opt_file = opt_with_delta_maxit(base_opt, sel.delta, 120)
        for delta_budget in (1e-3, 1e-2, 1.0)
            seeds_here = delta_budget == 1.0 ? [29] : SEEDS_D4
            for seed in seeds_here, direction in (:upper, :lower)
                d4 = build_d4_fixture(; seed=seed)
                n = length(d4.theta0)
                D = d4.ctx.D
                inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
                vs = ones(n); vs[1] = S_G; vs[2:end] .= 1e-5
                box = block_box(n, D, sel.box, 0.15, 0.15)
                key = (seed, delta_budget, direction)
                pool = get(incumbent_pool, key, Any[])
                best_ext = isempty(pool) ? nothing : pool[argmin([p[3] for p in pool])][2]
                label = "D4_joint_$(alg)_seed$(seed)_delta$(delta_budget)_$(direction)"
                row, res = run_search(label, d4.ctx, d4.obj, d4.theta0, delta_budget, direction, box, opt_file, inner_opt;
                    var_scale=vs, external_incumbent=best_ext)
                snap = base_active_mask(d4.theta0, d4.ctx, d4.obj)
                nsw = row.cold_verified_theta === nothing ? 0 : count_switches(snap, row.cold_verified_theta, d4.ctx, d4.obj)[1]
                # local poll: perturb the best incumbent by a few small steps and reoptimize DeltaStar directly
                poll_deltas = Float64[]
                if row.cold_verified_theta !== nothing
                    for pert in (1e-4, -1e-4, 5e-4, -5e-4)
                        tp = copy(row.cold_verified_theta); tp[1] += pert
                        rp = melitz_classified_inner_solve(d4.obj, tp, d4.ctx; delta_evaluation_cap=50.0, bank=MelitzDualBank(4))
                        push!(poll_deltas, rp isa FiniteSolved ? rp.Delta : NaN)
                    end
                end
                poll_all_worse = row.Delta === NaN || all(pd -> isnan(pd) || pd >= row.Delta, poll_deltas)
                push!(phase5_rows, merge(row, (algorithm=alg, seed=seed, delta_budget=delta_budget, direction=direction,
                    external_incumbent_used=(best_ext !== nothing), n_switches=nsw,
                    local_poll_deltas=join(poll_deltas, ";"), local_poll_confirms_local_opt=poll_all_worse)))
            end
        end
    end
    write_csv(joinpath(OUTDIR, "melitz_phase5_d4_longer_joint_2026-07-28.csv"), phase5_rows)
    println("\nDONE. CSVs written to ", OUTDIR)
end

function write_csv(path, rows)
    isempty(rows) && return
    open(path, "w") do io
        cols = keys(rows[1])
        println(io, join(cols, ","))
        for r in rows
            vals = [r[c] === nothing ? "" : (r[c] isa AbstractVector ? "\"$(join(r[c], ";"))\"" : r[c]) for c in cols]
            println(io, join(vals, ","))
        end
    end
end

main()
