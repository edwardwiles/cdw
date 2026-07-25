# 2026-07-25 local-geometry/continuation session -- Phases 0/3/4 (reduced, honestly scoped
# subset; see docs/melitz_real_d20_local_geometry_and_continuation_2026-07-25.md Section I
# for exactly what this covers vs. defers relative to the full governing-prompt grid).
#
# Reuses the EXACT tested pattern from scripts/melitz_real_d20_w_sensitivity_and_gradient_stepsize_2026-07-24.jl
# (same calibration loader, same starting point g=-0.497333, same
# :B_direct_argument_parallel gradient backend, same evaluate_melitz_delta cold-solve
# convention) rather than re-deriving it -- extended with:
#
#   Phase 0 reproduction: the near-boundary finite point, a pure-g over-budget point, one
#     nuisance step ~1e-5 (finite) and one ~1e-4 (fails/exceeds cap).
#   Phase 3: bandwidth sweep {1e-6,3e-6,1e-5,3e-5,1e-4,3e-4} on 3 sparse coordinate
#     directions (NOT the full 8-direction x 6-bandwidth grid -- see report Section I),
#     comparing the direct fixed-dual secant against a genuinely reoptimized central
#     difference, recording participation-switch counts.
#   Phase 4: direction families A (pure g), B (steepest nuisance descent, dg=0), and C
#     (minimum-norm first-order tangent correction) at several step sizes each, every trial
#     FULLY reoptimized (cold solve) -- NOT families D/E/F/G (see report Section I).
#
# Every gradient/Delta number here is computed live at run time, never hard-coded from the
# governing prompt's own illustrative figures.
#
# Usage: julia --project=. -t 16 scripts/melitz_local_geometry_lab_2026-07-25.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS, norm, dot
using Random

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
const INNER_OPT = joinpath(dirname(@__DIR__), "melitz_inner_loop_options_capped_2026-07-24.opt")
const OUTER_OPT = joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt")
const OUTER_BUDGET_DELTA = 1.0
const DELTA_EVALUATION_CAP = 10.0
const CAMPAIGN_START_G = -0.497333
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

"""Fully reoptimized cold solve + classification, honest terminology (finite/over-budget/
above-cap/numerical-failure -- never conflating "over budget" with "infeasible")."""
function classify_point(theta, ctx, obj_inner; delta=OUTER_BUDGET_DELTA, cap=DELTA_EVALUATION_CAP)
    r = evaluate_melitz_delta(theta, ctx, obj_inner; cold=true, store_G=false)
    if r.nStatus == 0
        finite = true
        over_budget = r.Delta > delta
        return (kind = over_budget ? :finite_over_budget : :finite_within_budget,
                Delta=r.Delta, nStatus=r.nStatus, r=r)
    else
        return (kind = :unresolved, Delta=NaN, nStatus=r.nStatus, r=r)
    end
end

function print_point(label, theta, res; dg=NaN, deta_norm=NaN, wall=NaN)
    @printf("  %-28s kind=%-20s Delta=%12s  nStatus=%5d  dg=%10s  ||deta||=%10s  wall=%6.2fs\n",
        label, string(res.kind), isnan(res.Delta) ? "NaN" : @sprintf("%.6e", res.Delta),
        res.nStatus, isnan(dg) ? "--" : @sprintf("%.3e", dg),
        isnan(deta_norm) ? "--" : @sprintf("%.3e", deta_norm), wall)
    flush(stdout)
end

function main()
    println("="^100)
    println("Melitz local-geometry lab -- 2026-07-25 (reduced, honest subset)")
    println("="^100)
    BLAS.set_num_threads(16)
    calib = load_calibration()
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ
    n = length(theta_calib)
    @printf("D=%d  n_theta=%d  W=%d  seed=1\n", ctx.D, n, W)

    theta0 = copy(theta_calib); theta0[1] = CAMPAIGN_START_G
    r0 = evaluate_melitz_delta(theta0, ctx, obj_inner; cold=true, store_G=false)
    @assert r0.nStatus == 0 "reference starting point must be a genuine converged solve"
    @printf("\nPHASE 0.1: reference near-boundary point  g=%.6f  Delta=%.6e  nStatus=%d\n",
        CAMPAIGN_START_G, r0.Delta, r0.nStatus)

    # ------------------------------------------------------------------------
    # Exact envelope-theorem gradient at theta0, via the SAME backend/dual the production
    # outer search itself uses -- recomputed here, never hard-coded.
    # ------------------------------------------------------------------------
    obj = build_melitz_implicit_bundle(ctx, obj_inner.U, theta0; delta=OUTER_BUDGET_DELTA,
        find_smallest=true, gradient_backend=:B_direct_argument_parallel, h=1e-4,
        inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT)
    CS = CounterfactualSensitivity
    G_now = CS.select_G_from_H(obj, obj.H)
    obj.moments!(@view(obj.H[:, 1]), G_now, theta0, obj.U, obj)
    obj.H[:, 2] .= 1.0
    x_star = r0.dual_x
    grad = zeros(n)
    direct_gradient_fn = make_melitz_gradient_delta_direct_parallel(1e-4)
    direct_gradient_fn(grad, theta0, ctx, obj, x_star)
    grad ./= 1e10
    q_g = grad[1]
    q_eta = @view grad[2:end]
    @printf("PHASE 0.1 gradient: q_g=%.6f  ||q_eta||=%.6f  ||grad||=%.6f  (%.4f%% of norm in eta)\n",
        q_g, norm(q_eta), norm(grad), 100 * norm(q_eta)^2 / norm(grad)^2)

    # ------------------------------------------------------------------------
    # PHASE 0.2: pure-g over-budget point (family A at a larger step).
    # ------------------------------------------------------------------------
    println("\nPHASE 0.2: pure-g sweep (A, f fixed)")
    e1 = zeros(n); e1[1] = 1.0
    for dg in (-0.002, -0.004, -0.006, -0.01)
        theta_try = theta0 .+ dg .* e1
        t0 = time(); res = classify_point(theta_try, ctx, obj_inner); wall = time() - t0
        print_point("pure_g dg=$dg", theta_try, res; dg=dg, wall=wall)
    end

    # ------------------------------------------------------------------------
    # PHASE 0.3: one nuisance-only step ~1e-5 (finite, expected) and one ~1e-4 (expected to
    # fail/exceed cap), along steepest descent of Delta itself, dg pinned to 0 (pure eta
    # move) -- reusing the already-validated steepest-descent direction but restricted to
    # the eta block only, re-normalized.
    # ------------------------------------------------------------------------
    println("\nPHASE 0.3: pure nuisance-descent step (dg=0, d_eta ∝ -q_eta)")
    d_eta_unit = -collect(q_eta) ./ norm(q_eta)
    for step_norm in (1e-5, 1e-4)
        theta_try = copy(theta0)
        theta_try[2:end] .+= step_norm .* d_eta_unit
        t0 = time(); res = classify_point(theta_try, ctx, obj_inner); wall = time() - t0
        print_point("nuisance_descent ||deta||=$step_norm", theta_try, res; deta_norm=step_norm, wall=wall)
    end

    # ------------------------------------------------------------------------
    # PHASE 3: bandwidth sweep on 3 sparse directions -- direct fixed-dual secant vs a
    # genuinely reoptimized central difference, participation-switch counts.
    # ------------------------------------------------------------------------
    println("\nPHASE 3: bandwidth sweep (direct fixed-dual secant vs. reoptimized secant)")
    rng = Random.MersenneTwister(7)
    dirs = Dict{String,Vector{Float64}}()
    dirs["g_e1"] = e1
    v = zeros(n); v[2] = 1.0; dirs["A_free_first"] = v            # first A-block free coordinate
    v = zeros(n); v[div(n, 2)] = 1.0; dirs["f_free_mid"] = v      # a mid-range f-block free coordinate

    function participation_switches(theta_a, theta_b, ctx, z_draws)
        D = ctx.D
        ca = melitz_outer_state(theta_a, ctx).cutoff
        cb = melitz_outer_state(theta_b, ctx).cutoff
        nsw = 0
        @inbounds for o in 1:D, d in 1:D
            active_a = @view(z_draws[:, o]) .> ca[o, d]
            active_b = @view(z_draws[:, o]) .> cb[o, d]
            nsw += count(active_a .!= active_b)
        end
        return nsw
    end

    for (name, v) in dirs
        @printf("\n  direction: %s\n", name)
        for h in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
            gplus = zeros(n); gminus = zeros(n)
            theta_p = theta0 .+ h .* v
            theta_m = theta0 .- h .* v
            direct_secant = dot(grad, v)   # exact fixed-dual directional derivative at theta0, bandwidth-independent by construction
            rp = evaluate_melitz_delta(theta_p, ctx, obj_inner; cold=true, store_G=false)
            rm = evaluate_melitz_delta(theta_m, ctx, obj_inner; cold=true, store_G=false)
            nsw = try
                participation_switches(theta_p, theta_m, ctx, obj_inner.U)
            catch
                -1
            end
            if rp.nStatus == 0 && rm.nStatus == 0
                reopt_secant = (rp.Delta - rm.Delta) / (2h)
                sign_match = sign(direct_secant) == sign(reopt_secant)
                relerr = abs(reopt_secant - direct_secant) / max(abs(direct_secant), 1e-12)
                @printf("    h=%8.1e  direct=%12.6e  reopt=%12.6e  sign_match=%-5s  relerr=%8.3f  switches=%4d\n",
                    h, direct_secant, reopt_secant, sign_match, relerr, nsw)
            else
                @printf("    h=%8.1e  direct=%12.6e  reopt=UNRESOLVED (nStatus+=%d,nStatus-=%d)  switches=%4d\n",
                    h, direct_secant, rp.nStatus, rm.nStatus, nsw)
            end
            flush(stdout)
        end
    end

    # ------------------------------------------------------------------------
    # PHASE 4: direction families A / B / C.
    # ------------------------------------------------------------------------
    println("\nPHASE 4A: pure-g direction, finer grid than Phase 0.2")
    for dg in (-1e-4, -3e-4, -1e-3, -3e-3, -1e-2)
        theta_try = theta0 .+ dg .* e1
        t0 = time(); res = classify_point(theta_try, ctx, obj_inner); wall = time() - t0
        pred = q_g * dg
        actual = res.kind == :unresolved ? NaN : res.Delta - r0.Delta
        print_point("A dg=$dg", theta_try, res; dg=dg, wall=wall)
        if !isnan(actual)
            @printf("      predicted dDelta=%.3e  actual dDelta=%.3e  pred_err=%.3e\n", pred, actual, actual - pred)
        end
    end

    println("\nPHASE 4B: steepest nuisance descent, dg=0, aggregate ||d_eta|| grid")
    for step_norm in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
        theta_try = copy(theta0)
        theta_try[2:end] .+= step_norm .* d_eta_unit
        t0 = time(); res = classify_point(theta_try, ctx, obj_inner); wall = time() - t0
        pred = dot(q_eta, step_norm .* d_eta_unit)
        actual = res.kind == :unresolved ? NaN : res.Delta - r0.Delta
        print_point("B ||deta||=$step_norm", theta_try, res; deta_norm=step_norm, wall=wall)
        if !isnan(actual)
            @printf("      predicted dDelta=%.3e  actual dDelta=%.3e  pred_err=%.3e\n", pred, actual, actual - pred)
        end
    end

    println("\nPHASE 4C: minimum-norm first-order tangent correction (the important family)")
    println("  d_eta = -(q_g*dg / dot(q_eta,q_eta)) * q_eta  =>  q_g*dg + dot(q_eta,d_eta) == 0 to first order")
    for dg in (-1e-5, -3e-5, -1e-4, -3e-4, -1e-3)
        d_eta = -(q_g * dg / dot(q_eta, q_eta)) .* collect(q_eta)
        theta_try = copy(theta0)
        theta_try[1] += dg
        theta_try[2:end] .+= d_eta
        orth_check = q_g * dg + dot(q_eta, d_eta)   # should be ~0
        t0 = time(); res = classify_point(theta_try, ctx, obj_inner); wall = time() - t0
        pred = q_g * dg + dot(q_eta, d_eta)   # == 0 by construction -- first-order-flat by design
        actual = res.kind == :unresolved ? NaN : res.Delta - r0.Delta
        print_point("C dg=$dg", theta_try, res; dg=dg, deta_norm=norm(d_eta), wall=wall)
        @printf("      orthogonality residual (should be ~0)=%.3e   actual dDelta=%s\n",
            orth_check, isnan(actual) ? "NaN" : @sprintf("%.3e", actual))
    end

    println("\n" * "="^100)
    println("Melitz local-geometry lab complete.")
    println("="^100)
end

main()
