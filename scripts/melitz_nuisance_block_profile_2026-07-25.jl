# 2026-07-25 local-geometry/continuation session, Phase 8: low-dimensional nuisance-profile
# experiment at an INTERIOR point (Delta approx 0.7-0.9), not directly at the fragile
# Delta~=1 boundary -- governing prompt's own explicit instruction. Compares A-only,
# f/q-only, and alternating A-then-f block minimization; the 3D subspace/full-nuisance
# polish is DEFERRED (see report Section I) given this session's wall-clock budget.
#
# Every call below passes `inner_solve_config` explicitly (Phase 1's own new mandatory
# kwarg on solve_melitz_nuisance_min_delta) -- this script doubles as a live real-D20
# validation of that infrastructure, not just the block-profile question.
#
# Usage: julia --project=. -t 16 scripts/melitz_nuisance_block_profile_2026-07-25.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS, norm

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
const INNER_OPT = joinpath(dirname(@__DIR__), "melitz_inner_loop_options_capped_2026-07-24.opt")
const OUTER_OPT = joinpath(dirname(@__DIR__), "melitz_outer_nuisance_profile.opt")
const DELTA_EVALUATION_CAP = 10.0
const W = 80_000

function load_calibration()
    lambdaData = readdlm(joinpath(REAL_DIR, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(REAL_DIR, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(REAL_DIR, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(REAL_DIR, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    return calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate,
        focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
end

function main()
    println("="^100)
    println("Melitz nuisance block-profile experiment -- 2026-07-25 (interior point, reduced scope)")
    println("="^100)
    BLAS.set_num_threads(16)
    calib = load_calibration()
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ

    # Find an interior g with Delta in [0.7, 0.9] (governing prompt's explicit instruction --
    # NOT directly at the Delta~1 boundary). Reuses the already-validated gamma-only grid
    # from the Phase 6 fixed-A/f profile (docs/melitz_real_d20_outer_benchmark_2026-07-24.md):
    # g=-0.4688 gave Delta=0.2187 at the OLD (pre-theta_star-fix) calibration; recompute fresh
    # here rather than trust the old number (that session's own calibration bug fix changed
    # Delta values -- do not reuse stale figures).
    println("\n-- locating an interior starting g (target Delta in [0.7,0.9]) --")
    g_candidates = -0.418871:-0.01:-0.50
    theta_g = nothing
    Delta_g = NaN
    for g in g_candidates
        theta_try = copy(theta_calib); theta_try[1] = g
        r = evaluate_melitz_delta(theta_try, ctx, obj_inner; cold=true, store_G=false)
        @printf("  g=%9.5f  Delta=%12.6e  nStatus=%d\n", g, r.Delta, r.nStatus)
        flush(stdout)
        if r.nStatus == 0 && 0.7 <= r.Delta <= 0.9
            theta_g, Delta_g = theta_try, r.Delta
            break
        end
    end
    if theta_g === nothing
        println("No grid point landed in [0.7,0.9] on this coarse grid -- falling back to the closest finite point below 1.0.")
        best_g, best_Delta = nothing, Inf
        for g in g_candidates
            theta_try = copy(theta_calib); theta_try[1] = g
            r = evaluate_melitz_delta(theta_try, ctx, obj_inner; cold=true, store_G=false)
            if r.nStatus == 0 && r.Delta < 1.0 && abs(r.Delta - 0.8) < abs(best_Delta - 0.8)
                best_g, best_Delta = g, r.Delta
            end
        end
        @assert best_g !== nothing "no finite sub-1.0 point found on this grid"
        theta_g = copy(theta_calib); theta_g[1] = best_g
        Delta_g = best_Delta
    end
    @printf("\nStarting point: g=%.6f  Delta_fixed_af=%.6e\n", theta_g[1], Delta_g)
    flush(stdout)

    cfg = MelitzInnerSolveConfig(:evaluation_cap; delta_evaluation_cap=DELTA_EVALUATION_CAP, guard=1e-6)
    RADIUS = 0.02   # a conservative radius given this fixture's own documented conditioning fragility

    results = NamedTuple[]

    for block in (:A_only, :f_only)
        mask = melitz_nuisance_free_mask(ctx; block=block)
        @printf("\n-- block=%s  n_free=%d --\n", block, count(mask))
        flush(stdout)
        t0 = time()
        res = solve_melitz_nuisance_min_delta(ctx, obj_inner, theta_g; free_mask=mask, radius=RADIUS,
            inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT, inner_solve_config=cfg)
        wall = time() - t0
        @printf("  nStatus=%d  Delta_min(KNITRO)=%.6e  Delta_min(cold-verified)=%.6e  wall=%.2fs  n_fc=%d  n_ga=%d\n",
            res.nStatus, res.Delta_min, res.r_final.Delta, wall, res.n_fc_calls, res.n_ga_calls)
        @printf("  verified non-increase vs. Delta_fixed_af: %.6e <= %.6e ? %s\n",
            res.r_final.Delta, Delta_g, res.r_final.Delta <= Delta_g)
        push!(results, (block=block, Delta_min=res.r_final.Delta, wall=wall, theta_final=res.theta_final,
                          nStatus=res.r_final.nStatus))
        flush(stdout)
    end

    # Alternating A-then-f, starting from the BEST of the two single-block results above,
    # each step required to be a verified non-increase (governing prompt: "retain the best
    # verified state monotonically").
    println("\n-- alternating A-then-f (2 rounds), monotone-verified --")
    best_single = reduce((a, b) -> a.Delta_min <= b.Delta_min ? a : b, results)
    theta_alt = copy(best_single.theta_final)
    Delta_alt = best_single.Delta_min
    @printf("  seed from best single block (%s): Delta=%.6e\n", best_single.block, Delta_alt)
    for round in 1:2, block in (:A_only, :f_only)
        mask = melitz_nuisance_free_mask(ctx; block=block)
        t0 = time()
        res = solve_melitz_nuisance_min_delta(ctx, obj_inner, theta_alt; free_mask=mask, radius=RADIUS,
            inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT, inner_solve_config=cfg)
        wall = time() - t0
        improved = res.r_final.nStatus == 0 && res.r_final.Delta <= Delta_alt
        @printf("  round=%d block=%-6s Delta_candidate=%.6e  improved=%-5s  wall=%.2fs\n",
            round, block, res.r_final.Delta, improved, wall)
        if improved
            theta_alt, Delta_alt = res.theta_final, res.r_final.Delta
        end
        flush(stdout)
    end
    @printf("\nAlternating result: Delta=%.6e (vs. fixed-af starting Delta=%.6e, vs. best single block=%.6e)\n",
        Delta_alt, Delta_g, best_single.Delta_min)

    println("\n" * "="^100)
    println("Nuisance block-profile experiment complete. (3D-subspace / full-nuisance polish DEFERRED -- see report.)")
    println("="^100)
end

main()
