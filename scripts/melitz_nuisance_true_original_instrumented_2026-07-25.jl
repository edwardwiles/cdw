# 2026-07-25: BYTE-FOR-BYTE copy of scripts/melitz_nuisance_block_profile_2026-07-25.jl
# (the actual script that ran 2+ hours), with ONLY on_eval/on_start observation hooks added
# to the two solve_melitz_nuisance_min_delta call sites -- no other line changed, no logic
# changed, same double grid-scan preamble, same everything. Purpose: get real per-callback
# timing from the ACTUAL slow scenario, not a hand-reconstructed approximation (user
# correctly flagged that the earlier "replica" script was not a faithful reproduction).
#
# Usage: julia --project=. -t 16 scripts/melitz_nuisance_true_original_instrumented_2026-07-25.jl

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

function main()
    println("="^100)
    println("Melitz nuisance block-profile experiment -- INSTRUMENTED true-original replay")
    println("="^100)
    BLAS.set_num_threads(16)
    calib = load_calibration()
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ

    # ---- IDENTICAL to the original: same double grid-scan preamble ----
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
    RADIUS = 0.02

    # ---- ONLY ADDITION: instrumentation on the A_only call (the one that was slow) ----
    t_sweep0 = time()
    n_fc_seen = Ref(0)
    n_ga_seen = Ref(0)
    function on_start(theta, idx, kind)
        kind == :fc ? (n_fc_seen[] += 1) : (n_ga_seen[] += 1)
    end
    function on_eval(theta, val, nStatus, kind)
        t = time() - t_sweep0
        @printf("[INSTR][%7.1fs] %-3s (#fc=%d,#ga=%d)  Delta=%.6e  nStatus=%d\n",
            t, kind, n_fc_seen[], n_ga_seen[], val, nStatus)
        flush(stdout)
        t > DIAG_TIMEOUT_S && error("MELITZ_DIAG_TIMEOUT")
    end

    mask = melitz_nuisance_free_mask(ctx; block=:A_only)
    @printf("\n-- block=A_only  n_free=%d --\n", count(mask))
    flush(stdout)
    try
        res = solve_melitz_nuisance_min_delta(ctx, obj_inner, theta_g; free_mask=mask, radius=RADIUS,
            inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT, inner_solve_config=cfg,
            on_eval=on_eval, on_start=on_start)
        @printf("  nStatus=%d  Delta_min(KNITRO)=%.6e  Delta_min(cold-verified)=%.6e  n_fc=%d  n_ga=%d\n",
            res.nStatus, res.Delta_min, res.r_final.Delta, res.n_fc_calls, res.n_ga_calls)
    catch e
        if e isa ErrorException && occursin("MELITZ_DIAG_TIMEOUT", e.msg)
            println("\n(Stopped by this script's own bounded window.)")
        else
            rethrow(e)
        end
    end

    println("\n" * "="^100)
    println("Instrumented true-original replay complete.")
    println("="^100)
end

main()
