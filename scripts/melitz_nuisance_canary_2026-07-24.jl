# 2026-07-24: minimal canary test before attempting another real GT-profile sweep --
# does a SINGLE solve_melitz_nuisance_min_delta call (:f_only block, A held fixed) even
# complete one outer iteration in reasonable time with a MUCH SMALLER radius (0.02 instead
# of the prior 0.5) and a much tighter outer maxit (8)/maxtime_real (60s)? The prior attempt
# (radius=0.5, maxit=40/maxtime_real=240) never printed past outer "Iter 0" in 35+ minutes --
# this tests whether a smaller, closer-to-calibration nuisance radius avoids whatever is
# driving that (most likely: many internal barrier-method line-search trial evaluations,
# each a full, uncached real inner CC solve, before KNITRO's own time check ever fires).
#
# Usage: julia --project=. -t 16 scripts/melitz_nuisance_canary_2026-07-24.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
const INNER_OPT = joinpath(dirname(@__DIR__), "melitz_inner_loop_options_capped_2026-07-24.opt")
const OUTER_OPT_CANARY = joinpath(dirname(@__DIR__), "melitz_outer_nuisance_profile_canary_2026-07-24.opt")
const RADIUS = parse(Float64, get(ENV, "MELITZ_CANARY_RADIUS", "0.02"))
const G_STEP = parse(Float64, get(ENV, "MELITZ_CANARY_G_STEP", "-0.02"))   # small step from calibration

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
    calib = load_calibration()
    @printf("radius=%.4f  g_step=%.4f  maxit=8  maxtime_real=60s\n", RADIUS, G_STEP)
    flush(stdout)

    BLAS.set_num_threads(16)
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ
    mask = melitz_nuisance_free_mask(ctx; block=:f_only)
    @printf("free_mask: n_free=%d (f_only, A held fixed)\n", count(mask))

    r0 = evaluate_melitz_delta(theta_calib, ctx, obj_inner; cold=true, store_G=false)
    @printf("calibration point: g=%.6f  Delta=%.6e  nStatus=%d\n", theta_calib[1], r0.Delta, r0.nStatus)
    flush(stdout)

    theta_g = copy(theta_calib); theta_g[1] += G_STEP
    @printf("canary point: g=%.6f (calibration %+ .4f)\n\n", theta_g[1], G_STEP)
    flush(stdout)

    t0 = time()
    res = solve_melitz_nuisance_min_delta(ctx, obj_inner, theta_g; free_mask=mask,
        radius=RADIUS, gradient_backend=:B_direct_argument_parallel, h=1e-4,
        inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT_CANARY, warm_start_x=r0.dual_x)
    wall = time() - t0

    @printf("\nCANARY RESULT: nStatus=%d  Delta_min(KNITRO)=%.6e  Delta_min(cold)=%.6e  wall=%.2fs  n_fc=%d  n_ga=%d  verified=%s\n",
        res.nStatus, res.Delta_min, res.r_final.Delta, wall, res.n_fc_calls, res.n_ga_calls, res.r_final.verified)

    BLAS.set_num_threads(1)
    println("\nDONE.")
    return res
end

main()
