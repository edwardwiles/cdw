# Quick follow-up to melitz_real_d20_w_sensitivity_and_gradient_stepsize_2026-07-24.jl:
# is the descent-direction step that broke the inner CC solve (eps=1e-4, ||dAf||~1e-4)
# actually violating the DETERMINISTIC cutoff constraints (a hard, expected economic
# boundary -- melitz_outer_state's own cheap, closed-form, no-Monte-Carlo min_slack check),
# or is it a nominally cutoff-feasible point where the CC dual solve itself just broke down?
# No KNITRO solve needed for this -- min_slack is a deterministic function of (A,f,w,tau,expenditure).
using Printf
using DelimitedFiles
using LinearAlgebra: BLAS, norm

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
const INNER_OPT = joinpath(dirname(@__DIR__), "melitz_inner_loop_options_capped_2026-07-24.opt")

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
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ

    theta_init = copy(theta_calib); theta_init[1] = -0.497333
    r0 = evaluate_melitz_delta(theta_init, ctx, obj_inner; cold=true, store_G=false)
    println("theta_init: min_slack=$(r0.min_slack)  feasible=$(r0.feasible)  Delta=$(r0.Delta)")

    obj = build_melitz_implicit_bundle(ctx, obj_inner.U, theta_init; delta=1.0,
        find_smallest=true, gradient_backend=:B_direct_argument_parallel, h=1e-4,
        inner_loop_opt=INNER_OPT, outer_loop_opt=joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt"))
    CS = CounterfactualSensitivity
    G_now = CS.select_G_from_H(obj, obj.H)
    obj.moments!(@view(obj.H[:, 1]), G_now, theta_init, obj.U, obj)
    obj.H[:, 2] .= 1.0
    x_star = r0.dual_x
    n = length(theta_init)
    grad = zeros(n)
    direct_gradient_fn = make_melitz_gradient_delta_direct_parallel(1e-4)
    direct_gradient_fn(grad, theta_init, ctx, obj, x_star)
    grad ./= 1e10
    d_desc = -grad ./ norm(grad)

    println("\nchecking min_slack (deterministic cutoff feasibility, NO KNITRO/Monte-Carlo) along the descent direction,")
    println("at step sizes that broke the inner CC solve (nStatus=-401 in the prior run):")
    for eps in (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
        theta_try = theta_init .+ eps .* d_desc
        state = melitz_outer_state(theta_try, ctx)
        @printf("  eps=%8.5f  min_slack=%14.8f  feasible=%-5s  (min_slack<0 means a HARD cutoff-constraint violation)\n",
            eps, state.min_slack, state.feasible)
    end

    println("\nsame check along the pure objective (g-only) direction, for comparison:")
    e1 = zeros(n); e1[1] = 1.0
    for eps in (0.0, 1e-3, 1e-2, 0.0039)
        theta_try = theta_init .- eps .* e1
        state = melitz_outer_state(theta_try, ctx)
        @printf("  eps=%8.5f  min_slack=%14.8f  feasible=%-5s\n", eps, state.min_slack, state.feasible)
    end

    println("\nDONE.")
end

main()
