# 2026-07-25 scaled-KNITRO session, Phase 5 (reduced pilot): a SHORT, matched comparison of
# the unscaled joint finite-delta KNITRO search against the SAME problem with KNITRO's
# native KN_set_var_scalings_all applied, using scale factors read directly off this repo's
# OWN already-measured local-geometry evidence
# (docs/melitz_real_d20_local_geometry_and_continuation_2026-07-25.md Section E):
#   r_g   (pure-g radius)              >= 1e-2   (not yet bracketed -- g is forgiving)
#   r_eta (pure nuisance-descent radius)  in [1e-5, 3e-5]  (A/f block)
# Candidate scales, per the governing prompt's own suggested starting range and this
# session's own audit (var_scale/var_center wired into solve_melitz_finite_delta_bound,
# src/melitz/finite_delta_outer.jl):
#   s_g  = 1e-4    (KNITRO's own scaled-Euclidean step norm ||dy|| ~ O(1) then maps to
#                   raw dg ~ O(1e-4), inside the pure-g forgiving range)
#   s_A = s_fq = 1e-5   (matches the measured r_eta breaking radius almost exactly)
# var_center = theta_init (the near-boundary starting point itself), so y=0 at the start.
#
# This is a DIAGNOSTIC pilot, not a tuned production search: ONE scale candidate, ONE
# algorithm (KNITRO's own auto default), a SHORT matched wall-clock budget for both arms.
# See the session report for what a fuller Phase 5 grid (algorithm x step-control x scale)
# would require and why it was not attempted at full scope this session.
#
# Usage: julia --project=. -t 16 scripts/melitz_real_d20_scaled_knitro_pilot_2026-07-25.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS, norm

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
const INNER_OPT = joinpath(dirname(@__DIR__), "melitz_inner_loop_options_capped_2026-07-24.opt")
const OUTER_OPT = joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt")
const OUTER_BUDGET_DELTA = 1.0
const DELTA_EVALUATION_CAP = 10.0
const CAMPAIGN_START_G = -0.497333          # Section D.1 near-boundary reference, Delta0~0.9653
const G_FIXED_REFERENCE = -0.49783321       # Phase 6 fixed-A/f verified boundary point
const KAPPA_FIXED_REFERENCE = 0.92939627
const W = 80_000
const CAMPAIGN_SECONDS = parse(Float64, get(ENV, "MELITZ_PILOT_SECONDS", "300"))
const S_G = parse(Float64, get(ENV, "MELITZ_PILOT_S_G", "1e-4"))
const S_A = parse(Float64, get(ENV, "MELITZ_PILOT_S_A", "1e-5"))
const S_FQ = parse(Float64, get(ENV, "MELITZ_PILOT_S_FQ", "1e-5"))

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

function block_scaled_theta_box(ctx; g_radius=0.05, A_radius=0.15, f_radius=0.15)
    D = ctx.D
    nA = D^2 - 1
    n = 2 * D^2 - 2
    nf = n - 1 - nA
    return vcat(g_radius, fill(A_radius, nA), fill(f_radius, nf))
end

function run_arm(label, ctx, obj_inner, theta_init, theta_fixed_af; var_scale=nothing, var_center=nothing)
    println("="^100)
    println("ARM: $label")
    println("="^100)
    flush(stdout)
    theta_box = block_scaled_theta_box(ctx)
    t0 = time()
    result = solve_melitz_finite_delta_bound(ctx, obj_inner, theta_init; delta=OUTER_BUDGET_DELTA,
        direction=:upper, gradient_backend=:B_direct_argument_parallel, h=1e-4, theta_box=theta_box,
        cutoff_constraint_backend=:linear, delta_evaluation_cap=DELTA_EVALUATION_CAP,
        lower_limit_guard=1e-6,
        inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT,
        external_incumbent=theta_fixed_af, var_scale=var_scale, var_center=var_center)
    wall = time() - t0
    @printf("[%s] wall=%.2fs  nStatus=%d  n_fc=%d  n_ga=%d  n_inner_solved=%d  n_infinite_reject=%d  n_above_cap=%d  n_numfail=%d\n",
        label, wall, result.nStatus, result.n_fc_calls, result.n_ga_calls, result.n_inner_solved,
        result.n_infinite_delta_reject, result.n_above_cap_reject, result.n_numerical_failure_reject)
    if result.cold_verified_incumbent !== nothing
        cv = result.cold_verified_incumbent
        @printf("[%s] cold_verified_incumbent: g=%.8f  Delta=%.6e  source=%s  objective=%.8f\n",
            label, cv.eval.theta_free[1], cv.eval.Delta, string(cv.source), cv.objective)
    else
        @printf("[%s] cold_verified_incumbent: NONE\n", label)
    end
    if result.best_live_incumbent !== nothing
        bl = result.best_live_incumbent
        @printf("[%s] best_live_incumbent: g=%.8f  Delta=%.6e\n", label, bl.eval.theta_free[1], bl.eval.Delta)
    end
    flush(stdout)
    return (label=label, wall=wall, result=result)
end

function main()
    println("Melitz real-D20 scaled-KNITRO pilot -- 2026-07-25")
    BLAS.set_num_threads(16)
    calib = load_calibration()
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ
    n = length(theta_calib)
    @printf("D=%d  n_theta=%d  W=%d  seed=1\n", ctx.D, n, W)

    theta_init = copy(theta_calib); theta_init[1] = CAMPAIGN_START_G
    r0 = evaluate_melitz_delta(theta_init, ctx, obj_inner; cold=true, store_G=false)
    @assert r0.nStatus == 0 "reference starting point must be a genuine converged solve"
    @printf("theta_init: g=%.6f  Delta0=%.6e  nStatus=%d\n", CAMPAIGN_START_G, r0.Delta, r0.nStatus)

    theta_fixed_af = copy(theta_calib); theta_fixed_af[1] = G_FIXED_REFERENCE

    D = ctx.D
    nA = D^2 - 1
    nf = n - 1 - nA
    var_scale = vcat(S_G, fill(S_A, nA), fill(S_FQ, nf))
    var_center = copy(theta_init)
    @printf("scale candidate: s_g=%.3e  s_A=%.3e  s_fq=%.3e  (n=%d = 1 + %d(A) + %d(f/q))\n",
        S_G, S_A, S_FQ, n, nA, nf)
    @printf("matched outer opt file for both arms: %s (maxit=25, no maxtime_real override -- see file)\n", OUTER_OPT)

    arm_unscaled = run_arm("UNSCALED (baseline)", ctx, obj_inner, theta_init, theta_fixed_af)
    arm_scaled = run_arm("SCALED (native KN_set_var_scalings_all)", ctx, obj_inner, theta_init, theta_fixed_af;
        var_scale=var_scale, var_center=var_center)

    println("="^100)
    println("SUMMARY")
    println("="^100)
    for arm in (arm_unscaled, arm_scaled)
        r = arm.result
        kappa_cv = r.cold_verified_incumbent === nothing ? NaN :
            (r.cold_verified_incumbent.source == :external ? KAPPA_FIXED_REFERENCE : NaN)
        @printf("%-45s wall=%7.2fs  n_fc=%4d  n_ga=%3d  n_inner_solved=%3d  n_above_cap=%4d  cv_source=%s\n",
            arm.label, arm.wall, r.n_fc_calls, r.n_ga_calls, r.n_inner_solved, r.n_above_cap_reject,
            r.cold_verified_incumbent === nothing ? "none" : string(r.cold_verified_incumbent.source))
    end
    flush(stdout)
end

main()
