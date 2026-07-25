# 2026-07-24, continuation of the evaluation-cap-correction session (user-directed):
# Delta_profile(g) = min_eta Delta(g,eta) SWEPT across g from the raw Pareto CALIBRATION
# point (kappa = calibration value) toward the theoretical GT ceiling
# (g_ceiling = log((lambda_dd^(1/(sigma-1)))^(sigma-1) / (w_prime/w_target)^(sigma-1))),
# for a SINGLE nuisance block per run (env MELITZ_PROFILE_BLOCK = "f_only" [A held fixed,
# the user's first request] or "A_only" [f held fixed, the user's follow-up request]).
#
# Reuses src/melitz/nuisance_profile.jl's solve_melitz_nuisance_min_delta UNCHANGED (built in
# a prior session, confirmed correct at D=4 but left unfinished at real D=20 scale --
# docs/melitz_real_d20_outer_correction_2026-07-24.md Section 5.1: each outer FC/GA pair does
# TWO full real inner CC solves, no exact-point cache, so this is genuinely expensive; a hard
# per-point KNITRO maxtime_real=240s (melitz_outer_nuisance_profile_capped_2026-07-24.opt) AND
# a Julia-level total-sweep wall-clock budget (MELITZ_SWEEP_BUDGET_S, default 1500s) bound the
# total cost -- if the budget runs out mid-sweep, whatever points completed are reported
# honestly, not padded with invented values).
#
# Continuation (warm start of BOTH the nuisance coordinates and the dual) between consecutive
# g-steps, matching melitz_real_d20_profile_continuation_2026-07-24.jl's own Section 9 pattern.
#
# Usage: MELITZ_PROFILE_BLOCK=f_only julia --project=. -t 16 scripts/melitz_real_d20_gt_profile_sweep_2026-07-24.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS, norm

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
const INNER_OPT = get(ENV, "MELITZ_INNER_OPT_PROFILE", joinpath(dirname(@__DIR__), "melitz_inner_loop_options_capped_2026-07-24.opt"))
const OUTER_OPT_PROFILE = get(ENV, "MELITZ_OUTER_OPT_PROFILE", joinpath(dirname(@__DIR__), "melitz_outer_nuisance_profile_capped_2026-07-24.opt"))
const PROFILE_BLOCK = Symbol(get(ENV, "MELITZ_PROFILE_BLOCK", "f_only"))
const RADIUS = parse(Float64, get(ENV, "MELITZ_NUISANCE_RADIUS", "0.5"))
const N_STEPS = parse(Int, get(ENV, "MELITZ_PROFILE_N_STEPS", "6"))
const SWEEP_BUDGET_S = parse(Float64, get(ENV, "MELITZ_SWEEP_BUDGET_S", "1500"))
const W_PROFILE = parse(Int, get(ENV, "MELITZ_PROFILE_W", "80000"))

function load_calibration()
    lambdaData = readdlm(joinpath(REAL_DIR, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(REAL_DIR, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(REAL_DIR, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(REAL_DIR, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate,
        focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
    return calib, lambdaData, focal
end

kappa_of_g(g::Real, calib) = (calib.w_prime / calib.w[calib.target_country]) * exp(g)^(1 / (calib.sigma - 1))

function theoretical_g_ceiling(calib, lambdaData, focal)
    lambda_dd = lambdaData[focal, focal]
    kappa_min = lambda_dd^(1 / (calib.sigma - 1))
    g_ceiling = log((kappa_min / (calib.w_prime / calib.w[calib.target_country]))^(calib.sigma - 1))
    return g_ceiling, kappa_min
end

function main()
    calib, lambdaData, focal = load_calibration()
    g_ceiling, kappa_min = theoretical_g_ceiling(calib, lambdaData, focal)
    GT_ceiling = 1 - kappa_min
    @printf("theoretical g_ceiling = %.6f  (GT_ceiling = %.6f)\n", g_ceiling, GT_ceiling)
    @printf("PROFILE_BLOCK = %s   N_STEPS = %d   sweep budget = %.0fs   W = %d\n",
        PROFILE_BLOCK, N_STEPS, SWEEP_BUDGET_S, W_PROFILE)
    flush(stdout)

    BLAS.set_num_threads(16)
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=W_PROFILE, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ
    mask = melitz_nuisance_free_mask(ctx; block=PROFILE_BLOCK)
    @printf("free_mask: n_free=%d (block=%s)\n", count(mask), PROFILE_BLOCK)

    g_calib = theta_calib[1]
    kappa_calib = kappa_of_g(g_calib, calib)
    @printf("Starting point: RAW CALIBRATION, g=%.8f  kappa=%.8f  GT=%.8f\n", g_calib, kappa_calib, 1 - kappa_calib)

    r0 = evaluate_melitz_delta(theta_calib, ctx, obj_inner; cold=true, store_G=false)
    @printf("Delta at calibration (fixed A/f, unrestricted): %.6e  nStatus=%d\n\n", r0.Delta, r0.nStatus)
    flush(stdout)

    g_grid = collect(range(g_calib, g_ceiling; length=N_STEPS + 1))[2:end]   # skip g_calib itself (already have r0)
    trajectory = NamedTuple[(step=0, g=g_calib, kappa=kappa_calib, GT=1 - kappa_calib,
        Delta_min_knitro=r0.Delta, Delta_cold=r0.Delta, nStatus=0, wall=0.0, pct_to_ceiling=0.0)]

    theta_cur = copy(theta_calib)
    dual_x = copy(r0.dual_x)
    t_sweep0 = time()

    for (step, g) in enumerate(g_grid)
        if time() - t_sweep0 > SWEEP_BUDGET_S
            @printf("[step %2d] SWEEP BUDGET (%.0fs) EXHAUSTED -- stopping here, reporting partial results honestly\n", step, SWEEP_BUDGET_S)
            flush(stdout)
            break
        end
        theta_g = copy(theta_cur); theta_g[1] = g
        t0 = time()
        local res
        try
            res = solve_melitz_nuisance_min_delta(ctx, obj_inner, theta_g; free_mask=mask,
                radius=RADIUS, gradient_backend=:B_direct_argument_parallel, h=1e-4,
                inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT_PROFILE, warm_start_x=dual_x)
        catch e
            wall = time() - t0
            @printf("[step %2d] g=%.6f  EXCEPTION after %.2fs: %s -- stopping sweep here\n", step, g, wall, sprint(showerror, e))
            flush(stdout)
            break
        end
        wall = time() - t0
        kappa = kappa_of_g(g, calib)
        pct = 100 * (g_calib - g) / (g_calib - g_ceiling)
        @printf("[step %2d] g=%9.6f (%.1f%% to ceiling)  kappa=%.6f  GT=%.6f  Delta_min(KNITRO)=%.6e  Delta_min(cold)=%.6e  nStatus=%d  wall=%.2fs  verified=%s\n",
            step, g, pct, kappa, 1 - kappa, res.Delta_min, res.r_final.Delta, res.nStatus, wall, res.r_final.verified)
        flush(stdout)
        push!(trajectory, (step=step, g=g, kappa=kappa, GT=1 - kappa, Delta_min_knitro=res.Delta_min,
            Delta_cold=res.r_final.Delta, nStatus=res.nStatus, wall=wall, pct_to_ceiling=pct))

        dual_x = copy(res.r_final.dual_x)
        theta_cur = copy(res.theta_final)
        theta_cur[1] = g   # keep g EXACTLY pinned at the grid value for the next step's own start
    end

    println("\n" * "="^100)
    println("PROFILE SWEEP SUMMARY [block=$PROFILE_BLOCK]")
    println("="^100)
    @printf("%6s %10s %10s %10s %14s %14s %8s %8s\n", "step", "g", "kappa", "GT", "Delta(KNITRO)", "Delta(cold)", "nStatus", "wall(s)")
    for t in trajectory
        @printf("%6d %10.6f %10.6f %10.6f %14.6e %14.6e %8d %8.2f\n",
            t.step, t.g, t.kappa, t.GT, t.Delta_min_knitro, t.Delta_cold, t.nStatus, t.wall)
    end
    @printf("\ntotal sweep wall = %.2fs, steps completed = %d/%d\n", time() - t_sweep0, length(trajectory) - 1, N_STEPS)

    BLAS.set_num_threads(1)
    println("\nDONE.")
    return trajectory
end

main()
