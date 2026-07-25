# 2026-07-25, same-day follow-up (user-directed): diagnose WHY the A_only nuisance block
# minimization (scripts/melitz_nuisance_block_profile_2026-07-25.jl) took 2+ hours for only
# 17 outer iterations, rather than speculating. Two things changed since that run:
#
#   1. src/melitz/nuisance_profile.jl now ports finite_delta_outer.jl's exact-point cache
#      (previously: cb_G! always re-solved from scratch at the SAME theta cb_F! just solved
#      -- a confirmed, but previously unmeasured, 2x multiplier on inner-solve calls).
#   2. `on_start`/`on_eval` now report REAL per-callback wall time and cache-hit status --
#      this script uses that to print a live, per-call timing trace, so the actual cost of
#      an individual inner CC dual solve at this fixture (399 free nuisance coordinates) is
#      MEASURED, not inferred from the outer iteration log alone (which was the mistake in
#      the original report -- the outer "Iter" count is not the same thing as per-callback
#      wall time).
#
# Bounded wall-clock (default 10 minutes, override via MELITZ_DIAG_TIMEOUT_S) -- this is a
# diagnostic probe, not a campaign: reports whatever it observes in that window, honestly,
# rather than running to convergence.
#
# Usage: julia --project=. -t 16 scripts/melitz_nuisance_cache_diagnosis_2026-07-25.jl
#    (wrap in `timeout <N> julia ...` at the OS level too, per this repo's own standing
#    practice for nuisance-profile runs -- KNITRO's own maxtime_real is checked only
#    between callback returns and cannot preempt a single slow callback.)

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
const G_START = -0.488871   # the SAME point the original (pre-cache) A_only run started from
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
    println("Melitz nuisance-profile cache diagnosis -- 2026-07-25 (bounded to $(DIAG_TIMEOUT_S)s)")
    println("="^100)
    BLAS.set_num_threads(16)
    calib = load_calibration()
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ
    theta_g = copy(theta_calib); theta_g[1] = G_START
    r0 = evaluate_melitz_delta(theta_g, ctx, obj_inner; cold=true, store_G=false)
    @printf("Starting point: g=%.6f  Delta=%.6e  nStatus=%d\n", G_START, r0.Delta, r0.nStatus)
    flush(stdout)

    cfg = MelitzInnerSolveConfig(:evaluation_cap; delta_evaluation_cap=DELTA_EVALUATION_CAP, guard=1e-6)
    mask = melitz_nuisance_free_mask(ctx; block=:A_only)
    cache = MelitzExactPointCache()

    t_sweep0 = time()
    n_events = Ref(0)
    hit_times = Float64[]
    miss_times = Float64[]

    function on_eval(theta, val, nStatus, kind, elapsed_s, cache_hit)
        n_events[] += 1
        t = time() - t_sweep0
        if cache_hit
            push!(hit_times, t)
            @printf("[%6.1fs] %-3s CACHE HIT   Delta=%.6e nStatus=%d  (elapsed=%.4fs)\n", t, kind, val, nStatus, elapsed_s)
        else
            push!(miss_times, elapsed_s)
            @printf("[%6.1fs] %-3s cache miss  Delta=%.6e nStatus=%d  inner_solve_wall=%.2fs\n", t, kind, val, nStatus, elapsed_s)
        end
        flush(stdout)
        if time() - t_sweep0 > DIAG_TIMEOUT_S
            error("MELITZ_DIAG_TIMEOUT: bounded diagnostic window ($(DIAG_TIMEOUT_S)s) elapsed -- stopping cleanly, not a crash")
        end
    end

    local res
    try
        res = solve_melitz_nuisance_min_delta(ctx, obj_inner, theta_g; free_mask=mask, radius=0.02,
            inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT, inner_solve_config=cfg,
            exact_cache=cache, on_eval=on_eval)
        @printf("\nCompleted within budget: nStatus=%d Delta_min=%.6e wall=%.1fs\n",
            res.nStatus, res.Delta_min, time() - t_sweep0)
    catch e
        if e isa ErrorException && occursin("MELITZ_DIAG_TIMEOUT", e.msg)
            println("\n(Stopped by this script's own bounded window, not KNITRO -- expected for a diagnostic probe.)")
        else
            rethrow(e)
        end
    end

    println("\n-- summary --")
    @printf("total FC+GA calls observed: %d   cache hits: %d   cache misses: %d\n",
        n_events[], length(hit_times), length(miss_times))
    if !isempty(miss_times)
        @printf("inner-solve wall time on a cache MISS: min=%.2fs  median=%.2fs  max=%.2fs  mean=%.2fs\n",
            minimum(miss_times), sort(miss_times)[cld(length(miss_times), 2)], maximum(miss_times),
            sum(miss_times) / length(miss_times))
    end
    @printf("cache hit rate: %.1f%%\n", 100 * length(hit_times) / max(n_events[], 1))

    println("\n" * "="^100)
    println("Cache diagnosis complete.")
    println("="^100)
end

main()
