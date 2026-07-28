# 2026-07-28 step-control/robustness session, Phases 8-10: real-D20 nuisance slack from
# interior fixed-A/f profile points (Phase 8), joint-algorithm comparison started from an
# interior point rather than the Delta~=1 boundary (Phase 9), and KNITRO-native wall-clock
# decomposition via the existing MELITZ_PROFILE instrumentation (Phase 10, reusing
# `melitz_profile_summary`/`melitz_profile_report` -- no new instrumentation invented).
#
# Interior points: selected directly against the Phase 2 real-D20 fixed-A/f profile CSV
# (`melitz_phase2_gamma_profile_realD20_2026-07-28.csv`), log-linearly interpolating g for
# DeltaStar~=0.5 and ~=0.8 between the two bracketing SOLVED profile rows, then confirming
# (and, if needed, one-step-correcting) the interpolated g via a real direct inner solve --
# this is deliberately NOT the profile GRID itself (which must stay root-finder-free per
# Phase 2/6), just locating two reference points ON the already-solved corridor for Phase 8
# to start from, exactly as the governing prompt's own Phase 8 "for example" framing allows.
#
# Usage: julia --project=. -t 20 scripts/melitz_phase8_9_10_realD20_2026-07-28.jl

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

block_free_mask(n, D, free_g, free_A, free_f) = (nA = D^2 - 1; m = falses(n); m[1] = free_g; m[2:1+nA] .= free_A; m[2+nA:end] .= free_f; m)
block_box(n, D, g_radius, A_radius, f_radius) = (nA = D^2 - 1; b = zeros(n); b[1] = g_radius; b[2:1+nA] .= A_radius; b[2+nA:end] .= f_radius; b)
ok_status(n) = n in (0, -101, -102, -103)

"Log-linearly interpolate g at a target DeltaStar between two bracketing FiniteSolved profile rows, then confirm with a real inner solve (one-step secant correction if the direct solve disagrees materially)."
function locate_interior_point(profile_rows, target_delta, ctx, obj, bank)
    finite = filter(r -> r.classification == "FiniteSolved", profile_rows)
    gs = [r.g for r in finite]; ds = [r.DeltaStar for r in finite]
    order = sortperm(gs); gs, ds = gs[order], ds[order]
    k = findfirst(i -> ds[i] <= target_delta <= ds[i+1] || ds[i] >= target_delta >= ds[i+1], 1:length(ds)-1)
    @assert k !== nothing "target_delta=$target_delta not bracketed by the solved profile rows"
    t = (log(target_delta) - log(ds[k])) / (log(ds[k+1]) - log(ds[k]))
    g_guess = gs[k] + t * (gs[k+1] - gs[k])
    return g_guess
end

function solve_at_g(g, theta0, ctx, obj, bank; cap=50.0)
    theta = copy(theta0); theta[1] = g
    r = melitz_classified_inner_solve(obj, theta, ctx; delta_evaluation_cap=cap, bank=bank)
    return theta, r
end

function main()
    BLAS.set_num_threads(20)
    d20 = build_realD20_fixture()
    ctx, obj, theta0 = d20.ctx, d20.obj, d20.theta0
    n = length(theta0)
    D = ctx.D
    inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
    profile20 = load_gamma_profile_csv(joinpath(OUTDIR, "melitz_phase2_gamma_profile_realD20_2026-07-28.csv"))
    bank = MelitzDualBank(8)

    println("="^100); println("PHASE 8: locating two interior fixed-A/f points (target Delta~0.5, ~0.8)"); println("="^100)
    g05 = locate_interior_point(profile20, 0.5, ctx, obj, bank)
    theta_g05, r_g05 = solve_at_g(g05, theta0, ctx, obj, bank)
    @printf("  target Delta~0.5: g=%.6f -> %s\n", g05, r_g05 isa FiniteSolved ? (@sprintf("Delta=%.4e", r_g05.Delta)) : string(typeof(r_g05)))
    g08 = locate_interior_point(profile20, 0.8, ctx, obj, bank)
    theta_g08, r_g08 = solve_at_g(g08, theta0, ctx, obj, bank)
    @printf("  target Delta~0.8: g=%.6f -> %s\n", g08, r_g08 isa FiniteSolved ? (@sprintf("Delta=%.4e", r_g08.Delta)) : string(typeof(r_g08)))
    flush(stdout)

    interior_points = [("interior_g05", theta_g05, r_g05), ("interior_g08", theta_g08, r_g08)]
    inner_cfg = MelitzInnerSolveConfig(:evaluation_cap; delta_evaluation_cap=CAP)
    melitz_assert_evaluation_cap_active(inner_cfg)
    mask_A = block_free_mask(n, D, false, true, false)
    mask_f = block_free_mask(n, D, false, false, true)
    mask_full = block_free_mask(n, D, false, true, true)

    phase8_rows = NamedTuple[]
    best_interior_candidates = NamedTuple[]   # for Phase 9 seeding
    for (label, theta_g, r_fixed) in interior_points
        r_fixed isa FiniteSolved || (println("  SKIP $label: fixed point itself not FiniteSolved"); continue)
        Delta_fixed = r_fixed.Delta
        results = Dict{String,Any}()
        for (stage_label, mask, warm_from) in (("A_only", mask_A, nothing), ("f_only", mask_f, nothing))
            wall = @elapsed begin
                res = solve_melitz_nuisance_min_delta(ctx, obj, theta_g; free_mask=mask, radius=0.15,
                    gradient_backend=:B_direct_argument_parallel, h=1e-4, inner_loop_opt=inner_opt,
                    inner_solve_config=inner_cfg, forbid_dense_fallback=true)
            end
            ok = ok_status(res.nStatus) && res.r_final.verified
            results[stage_label] = (res=res, ok=ok, wall=wall)
            @printf("  [%s/%s] nStatus=%4d wall=%6.2fs n_fc=%3d Delta=%s ok=%s\n", label, stage_label,
                res.nStatus, wall, res.n_fc_calls, ok ? (@sprintf("%.4e", res.r_final.Delta)) : "UNVERIFIED", ok)
            flush(stdout)
        end
        cand = NamedTuple[]
        results["A_only"].ok && push!(cand, (theta=results["A_only"].res.theta_final, dual=results["A_only"].res.r_final.dual_x, Delta=results["A_only"].res.r_final.Delta))
        results["f_only"].ok && push!(cand, (theta=results["f_only"].res.theta_final, dual=results["f_only"].res.r_final.dual_x, Delta=results["f_only"].res.r_final.Delta))
        seed_theta = isempty(cand) ? theta_g : cand[argmin([c.Delta for c in cand])].theta
        seed_dual = isempty(cand) ? nothing : cand[argmin([c.Delta for c in cand])].dual
        wall_full = @elapsed begin
            res_full = solve_melitz_nuisance_min_delta(ctx, obj, seed_theta; free_mask=mask_full, radius=0.15,
                gradient_backend=:B_direct_argument_parallel, h=1e-4, inner_loop_opt=inner_opt,
                warm_start_x=seed_dual, inner_solve_config=inner_cfg, forbid_dense_fallback=true)
        end
        ok_full = ok_status(res_full.nStatus) && res_full.r_final.verified
        @printf("  [%s/full] nStatus=%4d wall=%6.2fs n_fc=%3d Delta=%s ok=%s\n", label,
            res_full.nStatus, wall_full, res_full.n_fc_calls, ok_full ? (@sprintf("%.4e", res_full.r_final.Delta)) : "UNVERIFIED", ok_full)
        flush(stdout)

        all_deltas = Dict{String,Float64}("fixed" => Delta_fixed)
        results["A_only"].ok && (all_deltas["A_only"] = results["A_only"].res.r_final.Delta)
        results["f_only"].ok && (all_deltas["f_only"] = results["f_only"].res.r_final.Delta)
        ok_full && (all_deltas["full"] = res_full.r_final.Delta)
        best_src = first(sort(collect(keys(all_deltas)), by=k -> all_deltas[k]))
        best_delta = all_deltas[best_src]
        best_theta = best_src == "fixed" ? theta_g :
                     best_src == "A_only" ? results["A_only"].res.theta_final :
                     best_src == "f_only" ? results["f_only"].res.theta_final : res_full.theta_final
        nA = D^2 - 1
        dlogA = norm(best_theta[2:1+nA] - theta_g[2:1+nA])
        dlogf = norm(best_theta[2+nA:end] - theta_g[2+nA:end])
        slack_ratio = Delta_fixed / max(best_delta, 1e-300)
        push!(phase8_rows, (point=label, g=theta_g[1], Delta_fixed=Delta_fixed, best_source=best_src,
            best_Delta=best_delta, slack_ratio=slack_ratio, dlogA=dlogA, dlogf=dlogf,
            A_only_Delta=get(all_deltas, "A_only", NaN), f_only_Delta=get(all_deltas, "f_only", NaN),
            full_Delta=get(all_deltas, "full", NaN), wall_A=results["A_only"].wall, wall_f=results["f_only"].wall,
            wall_full=wall_full, n_fc_full=res_full.n_fc_calls))
        push!(best_interior_candidates, (label=label, theta=best_theta, Delta=best_delta))
    end
    write_csv(joinpath(OUTDIR, "melitz_phase8_realD20_interior_nuisance_2026-07-28.csv"), phase8_rows)

    println("\n" * "="^100); println("PHASE 9/10: joint algorithm comparison from an interior nuisance-improved point"); println("="^100)
    isempty(best_interior_candidates) && error("no interior candidate available for Phase 9 -- Phase 8 must produce at least one")
    start_cand = best_interior_candidates[argmin([c.Delta for c in best_interior_candidates])]
    theta_start = start_cand.theta
    println("Phase 9 starting point: ", start_cand.label, " Delta=", start_cand.Delta)

    r_boundary = filter(r -> r.gamma_fraction == 0.65, profile20)
    external_boundary_theta = nothing
    if !isempty(r_boundary)
        tb = copy(theta0); tb[1] = r_boundary[1].g
        external_boundary_theta = tb
    end

    vs20 = ones(n); vs20[1] = S_G
    box = block_box(n, D, 0.10, 0.15, 0.15)
    delta_budget = 1.0
    MAXTIME = 240.0   # scoped down from the governing prompt's 300-600s given this session's overall wall-clock budget; disclosed.

    alg_files = Dict(:active => joinpath(REPO, "melitz_outer_finite_delta_alg_active_2026-07-27.opt"),
                      :sqp => joinpath(REPO, "melitz_outer_finite_delta_alg_sqp_2026-07-27.opt"))
    phase9_rows = NamedTuple[]
    timing_rows = NamedTuple[]
    for alg in (:active, :sqp)
        lines = readlines(alg_files[alg])
        lines = filter(l -> !occursin(r"^\s*(delta|maxit|maxtime_real)\s", l), lines)
        push!(lines, "delta           0.05")
        push!(lines, "maxit           2000")
        push!(lines, @sprintf("maxtime_real    %.1f", MAXTIME))
        opt_path = joinpath(TMPOPT, "phase9_$(alg)_2026-07-28.opt")
        open(io -> foreach(l -> println(io, l), lines), opt_path, "w")

        MELITZ_PROFILE[] = true
        melitz_profile_reset!()
        local res
        wall = @elapsed begin
            res = solve_melitz_finite_delta_bound(ctx, obj, theta_start; delta=delta_budget, direction=:upper,
                delta_evaluation_cap=CAP, gradient_backend=:B_direct_argument_sorted_serial, h=1e-4,
                theta_box=box, cutoff_constraint_backend=:linear,
                inner_loop_opt=inner_opt, outer_loop_opt=opt_path,
                var_scale=vs20, var_center=collect(Float64.(theta_start)),
                backend=:matrix_free, forbid_dense_fallback=true,
                objective_scale=:auto, external_incumbent=external_boundary_theta)
        end
        prof_rows = melitz_profile_summary()
        io = IOBuffer()
        melitz_profile_report(io; trajectory_total_s=wall)
        println("--- ", alg, " profile report ---")
        println(String(take!(io)))
        MELITZ_PROFILE[] = false

        cv = res.cold_verified_incumbent
        nA = D^2 - 1
        dg = cv === nothing ? 0.0 : cv.eval.theta_free[1] - theta_start[1]
        dlogA = cv === nothing ? 0.0 : norm(cv.eval.theta_free[2:1+nA] - theta_start[2:1+nA])
        dlogf = cv === nothing ? 0.0 : norm(cv.eval.theta_free[2+nA:end] - theta_start[2+nA:end])
        @printf("  [D20_phase9_%-6s] nStatus=%5d wall=%6.2fs n_fc=%3d n_ga=%3d n_solved=%3d n_cap=%3d n_numfail=%3d dg=%+.5f Delta=%s\n",
            alg, res.nStatus, wall, res.n_fc_calls, res.n_ga_calls, res.n_inner_solved, res.n_above_cap_reject,
            res.n_numerical_failure_reject, dg, cv === nothing ? "NA" : @sprintf("%.4e", cv.eval.Delta))
        flush(stdout)
        push!(phase9_rows, (algorithm=alg, nStatus=res.nStatus, wall=wall, n_fc=res.n_fc_calls, n_ga=res.n_ga_calls,
            n_inner_solved=res.n_inner_solved, n_above_cap=res.n_above_cap_reject,
            n_numerical_failure=res.n_numerical_failure_reject, dg=dg, dlogA=dlogA, dlogf=dlogf,
            best_kappa=(cv === nothing ? NaN : cv.eval.gamma_prime_j), best_Delta=(cv === nothing ? NaN : cv.eval.Delta),
            incumbent_source=(cv === nothing ? "none" : string(cv.source))))
        for r in prof_rows
            push!(timing_rows, merge(r, (algorithm=alg, trajectory_total_s=wall)))
        end
    end
    write_csv(joinpath(OUTDIR, "melitz_phase9_realD20_algorithm_comparison_2026-07-28.csv"), phase9_rows)
    write_csv(joinpath(OUTDIR, "melitz_phase10_realD20_timing_decomposition_2026-07-28.csv"), timing_rows)
    BLAS.set_num_threads(1)
    println("\nDONE. CSVs written to ", OUTDIR)
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

main()
