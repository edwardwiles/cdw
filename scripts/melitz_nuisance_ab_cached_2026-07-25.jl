# 2026-07-25, controlled A/B comparison (user-directed): the CACHED (current,
# post-2026-07-25-fix) counterpart to scripts/melitz_nuisance_ab_uncached_2026-07-25.jl --
# same starting point/radius/cap/W/seed, launched SIMULTANEOUSLY with that script so both
# experience identical machine contention. This isolates the cache as the ONE variable.
#
# Usage: julia --project=. -t 16 scripts/melitz_nuisance_ab_cached_2026-07-25.jl

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

function main()
    println("="^100)
    println("Melitz nuisance A/B test -- CACHED (current, fixed) -- bounded $(DIAG_TIMEOUT_S)s")
    println("="^100)
    BLAS.set_num_threads(16)
    calib = load_calibration()
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ
    theta_g = copy(theta_calib); theta_g[1] = G_START
    r0 = evaluate_melitz_delta(theta_g, ctx, obj_inner; cold=true, store_G=false)
    @printf("[CACHED] Starting point: g=%.6f  Delta=%.6e  nStatus=%d\n", G_START, r0.Delta, r0.nStatus)
    flush(stdout)

    cfg = MelitzInnerSolveConfig(:evaluation_cap; delta_evaluation_cap=DELTA_EVALUATION_CAP, guard=1e-6)
    mask = melitz_nuisance_free_mask(ctx; block=:A_only)
    cache = MelitzExactPointCache()
    t_sweep0 = time()

    function on_eval(theta, val, nStatus, kind, elapsed_s, cache_hit)
        t = time() - t_sweep0
        @printf("[CACHED][%6.1fs] %-3s %-9s Delta=%.6e nStatus=%d inner_solve_wall=%.4fs\n",
            t, kind, cache_hit ? "CACHE_HIT" : "cache_miss", val, nStatus, elapsed_s)
        flush(stdout)
        t > DIAG_TIMEOUT_S && error("MELITZ_DIAG_TIMEOUT")
    end

    local res
    try
        res = solve_melitz_nuisance_min_delta(ctx, obj_inner, theta_g; free_mask=mask, radius=0.02,
            inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT, inner_solve_config=cfg,
            exact_cache=cache, on_eval=on_eval)
        @printf("\n[CACHED] Completed within budget: nStatus=%d Delta_min=%.6e n_fc=%d n_ga=%d n_hits=%d n_misses=%d wall=%.1fs\n",
            res.nStatus, res.Delta_min, res.n_fc_calls, res.n_ga_calls, res.n_exact_cache_hits,
            res.n_exact_cache_misses, time() - t_sweep0)
    catch e
        if e isa ErrorException && occursin("MELITZ_DIAG_TIMEOUT", e.msg)
            println("\n[CACHED] (Stopped by this script's own bounded window, not KNITRO.)")
        else
            rethrow(e)
        end
    end
    println("="^100)
end

main()
