# 2026-07-25, controlled A/B comparison (user-directed): reconstructs the ORIGINAL
# (pre-cache-port) nuisance-profile callback pair EXACTLY as it existed at the start of this
# session -- no exact-point cache, cb_G! always re-solves `inner_loop(obj_inner, theta)`
# from scratch even at the same theta cb_F! just solved -- and runs it under the SAME
# starting point/radius/cap as the cached version, LAUNCHED SIMULTANEOUSLY with
# scripts/melitz_nuisance_ab_cached_2026-07-25.jl so both experience identical machine
# contention. This isolates ONE variable (cache present vs. absent) rather than comparing
# across two runs made at different times under unknown, possibly different load.
#
# The ONLY difference from the current (fixed) src/melitz/nuisance_profile.jl is that this
# file's own cb_F!/cb_G! never check or populate an exact-point cache -- everything else
# (inner_solve_config/lower_limit cap, KNITRO option files, registration) is identical and
# reused verbatim from the loaded library.
#
# Usage: julia --project=. -t 16 scripts/melitz_nuisance_ab_uncached_2026-07-25.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS, norm

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity
using KNITRO

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
const INNER_OPT = joinpath(dirname(@__DIR__), "melitz_inner_loop_options_capped_2026-07-24.opt")
const OUTER_OPT = joinpath(dirname(@__DIR__), "melitz_outer_nuisance_profile.opt")
const DELTA_EVALUATION_CAP = 10.0
const W = 80_000
const G_START = -0.488871
const DIAG_TIMEOUT_S = parse(Float64, get(ENV, "MELITZ_DIAG_TIMEOUT_S", "600"))

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

# ============================================================================
# EXACT reconstruction of the pre-2026-07-25 melitz_build_nuisance_profile_callbacks --
# no exact_cache parameter, no cache lookup/insert, cb_G! always calls inner_loop fresh.
# ============================================================================
function melitz_build_nuisance_profile_callbacks_UNCACHED(obj_inner, ctx; gradient_backend::Symbol=:B_direct_argument_parallel,
                                                            h::Real=1e-4, on_eval=nothing, on_start=nothing)
    direct_gradient_fn = gradient_backend == :B_direct_argument_parallel ? make_melitz_gradient_delta_direct_parallel(h) :
                         make_melitz_gradient_delta_direct_serial(h)
    n_fc_calls = Ref(0)
    n_ga_calls = Ref(0)

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        theta = collect(evalRequest.x)
        n_fc_calls[] += 1
        on_start !== nothing && on_start(theta, n_fc_calls[], :fc)
        t0 = time_ns()
        val, x, nStatus = inner_loop(obj_inner, theta)
        elapsed_s = (time_ns() - t0) / 1e9
        on_eval !== nothing && on_eval(theta, val, nStatus, :fc, elapsed_s, false)
        accepted = nStatus in (0, -100, -101, -103)
        accepted || throw(DomainError(theta[1],
            "melitz nuisance-profile FC (UNCACHED replica): inner CC dual solve failed, nStatus=$nStatus"))
        evalResult.obj[1] = val
        return 0
    end

    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        theta = collect(evalRequest.x)
        n_ = length(theta)
        n_ga_calls[] += 1
        on_start !== nothing && on_start(theta, n_ga_calls[], :ga)
        t0 = time_ns()
        val, x, nStatus = inner_loop(obj_inner, theta)   # ALWAYS a fresh solve -- no cache, exactly the pre-fix behavior
        elapsed_s = (time_ns() - t0) / 1e9
        on_eval !== nothing && on_eval(theta, val, nStatus, :ga, elapsed_s, false)
        accepted = nStatus in (0, -100, -101, -103)
        accepted || throw(DomainError(theta[1],
            "melitz nuisance-profile GA (UNCACHED replica): inner CC dual solve failed, nStatus=$nStatus"))
        local_jac = zeros(n_)
        direct_gradient_fn(local_jac, theta, ctx, obj_inner, x)
        evalResult.objGrad .= local_jac ./ 1e10
        return 0
    end

    return (cb_F! = cb_F!, cb_G! = cb_G!, n_fc_calls = n_fc_calls, n_ga_calls = n_ga_calls)
end

function solve_melitz_nuisance_min_delta_UNCACHED(ctx, obj_inner, theta_start; free_mask, radius,
        inner_loop_opt, outer_loop_opt, inner_solve_config, on_eval=nothing, on_start=nothing)
    n = length(theta_start)
    theta0 = collect(Float64.(theta_start))
    cbset = melitz_build_nuisance_profile_callbacks_UNCACHED(obj_inner, ctx; on_eval=on_eval, on_start=on_start)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, outer_loop_opt)
    xIndices = KNITRO.KN_add_vars(kc, n)
    r = radius isa Real ? fill(Float64(radius), n) : collect(Float64.(radius))
    lo = copy(theta0); hi = copy(theta0)
    @inbounds for k in 1:n
        if free_mask[k]
            lo[k] -= r[k]; hi[k] += r[k]
        end
    end
    KNITRO.KN_set_var_lobnds_all(kc, lo)
    KNITRO.KN_set_var_upbnds_all(kc, hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, theta0)
    melitz_register_nuisance_profile_knitro_problem!(kc, ctx, cbset, xIndices, n)

    saved_ll = obj_inner.lower_limit
    obj_inner.lower_limit = inner_solve_config.lower_limit
    local nStatus, objVal
    try
        KNITRO.KN_solve(kc)
        nStatus, objVal, _, _ = KNITRO.KN_get_solution(kc)
    finally
        KNITRO.KN_free(kc)
        obj_inner.lower_limit = saved_ll
    end
    return (nStatus=nStatus, Delta_min=Float64(objVal), n_fc_calls=cbset.n_fc_calls[], n_ga_calls=cbset.n_ga_calls[])
end

function main()
    println("="^100)
    println("Melitz nuisance A/B test -- UNCACHED replica (pre-2026-07-25 behavior) -- bounded $(DIAG_TIMEOUT_S)s")
    println("="^100)
    BLAS.set_num_threads(16)
    calib = load_calibration()
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ
    theta_g = copy(theta_calib); theta_g[1] = G_START
    r0 = evaluate_melitz_delta(theta_g, ctx, obj_inner; cold=true, store_G=false)
    @printf("[UNCACHED] Starting point: g=%.6f  Delta=%.6e  nStatus=%d\n", G_START, r0.Delta, r0.nStatus)
    flush(stdout)

    cfg = MelitzInnerSolveConfig(:evaluation_cap; delta_evaluation_cap=DELTA_EVALUATION_CAP, guard=1e-6)
    mask = melitz_nuisance_free_mask(ctx; block=:A_only)
    t_sweep0 = time()

    function on_eval(theta, val, nStatus, kind, elapsed_s, cache_hit)
        t = time() - t_sweep0
        @printf("[UNCACHED][%6.1fs] %-3s Delta=%.6e nStatus=%d inner_solve_wall=%.2fs\n", t, kind, val, nStatus, elapsed_s)
        flush(stdout)
        time() - t_sweep0 > DIAG_TIMEOUT_S && error("MELITZ_DIAG_TIMEOUT")
    end

    try
        res = solve_melitz_nuisance_min_delta_UNCACHED(ctx, obj_inner, theta_g; free_mask=mask, radius=0.02,
            inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT, inner_solve_config=cfg, on_eval=on_eval)
        @printf("\n[UNCACHED] Completed within budget: nStatus=%d Delta_min=%.6e n_fc=%d n_ga=%d wall=%.1fs\n",
            res.nStatus, res.Delta_min, res.n_fc_calls, res.n_ga_calls, time() - t_sweep0)
    catch e
        if e isa ErrorException && occursin("MELITZ_DIAG_TIMEOUT", e.msg)
            println("\n[UNCACHED] (Stopped by this script's own bounded window, not KNITRO.)")
        else
            rethrow(e)
        end
    end
    println("="^100)
end

main()
