# 2026-07-25: follow-up to melitz_nuisance_canary_instrumented_2026-07-24.jl's finding that
# the nuisance-profile stall's own ~90-93s-per-failed-retry pattern matches the INNER solve's
# `maxtime_real=90` wall-clock cap, NOT the KNITRO-native `lower_limit` objective-value
# bailout (`cc_algo/PsiObjectiveBundle.jl`'s `if f <= lower_limit; return -KN_INFINITY`).
#
# User's direct question: is `lower_limit` genuinely never firing here, and if so, is that
# because the inner solve converges (slowly) or because it diverges without `lower_limit`
# ever being SET high enough (in this codebase's sign convention, LOW enough) to catch it?
#
# Root cause identified: `build_melitz_psi_bundle_from_calibration` (src/melitz/pareto_calibration.jl)
# constructs `obj_inner::PsiObjectiveBundleDelta` WITHOUT ever passing `lower_limit` -- it
# silently stays at the struct's own default (`-KNITRO.KN_INFINITY`, i.e. PERMANENTLY
# DISABLED) for every inner solve `nuisance_profile.jl`/`evaluate_melitz_delta` ever makes.
# This is a DIFFERENT object from the outer `PsiObjectiveBundleImplicit` bundle
# (`build_melitz_implicit_bundle`, `finite_delta_outer.jl`) that this whole session's
# `delta_evaluation_cap`/`lower_limit_guard` correction already wired up correctly -- that
# fix never touched THIS object, because `nuisance_profile.jl` did not exist as an outer-
# search consumer of `obj_inner` in that part of the session.
#
# This script wires up `obj_inner.lower_limit = -(cap + guard)` directly (user's own
# preferred cap=10, matching `delta_evaluation_cap`'s production default elsewhere) and
# reruns the IDENTICAL canary (same starting point, radius, warm start) to see directly
# whether the previously-failing calls now abort FAST (proving the raw dual objective WAS
# diverging past the cap -- lower_limit is the right, and much cheaper, fix) or STILL take
# ~90s (proving `maxtime_real`, not divergence, is what is actually binding -- a genuinely
# slow-but-bounded inner problem, not one that blows up).
#
# Usage (bounded by an OS-level timeout, as before -- do not rely on KNITRO's own
# maxtime_real to preempt a single slow callback):
#   timeout 300 julia --project=. -t 16 scripts/melitz_nuisance_canary_lowerlimit_2026-07-25.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS, norm
using Serialization: serialize

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
const INNER_OPT = joinpath(dirname(@__DIR__), "melitz_inner_loop_options_capped_2026-07-24.opt")
const OUTER_OPT_CANARY = joinpath(dirname(@__DIR__), "melitz_outer_nuisance_profile_canary_2026-07-24.opt")
const RADIUS = parse(Float64, get(ENV, "MELITZ_CANARY_RADIUS", "0.02"))
const G_STEP = parse(Float64, get(ENV, "MELITZ_CANARY_G_STEP", "-0.02"))
const LOWER_LIMIT_CAP = parse(Float64, get(ENV, "MELITZ_LOWER_LIMIT_CAP", "10.0"))   # user's preferred cap
const LOWER_LIMIT_GUARD = 1e-6
const DUMP_PATH = joinpath(dirname(@__DIR__), "melitz_nuisance_canary_lowerlimit_tried_points_2026-07-25.jls")

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

mutable struct CallRecord
    call_index::Int
    kind::Symbol
    started_at::Float64
    finished_at::Float64
    theta::Vector{Float64}
    val::Float64
    nStatus::Int
end

function main()
    calib = load_calibration()
    @printf("radius=%.4f  g_step=%.4f  lower_limit_cap=%.4f (=> obj_inner.lower_limit=%.6f)\n",
        RADIUS, G_STEP, LOWER_LIMIT_CAP, -(LOWER_LIMIT_CAP + LOWER_LIMIT_GUARD))
    flush(stdout)

    BLAS.set_num_threads(16)
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ
    mask = melitz_nuisance_free_mask(ctx; block=:f_only)
    @printf("free_mask: n_free=%d (f_only, A held fixed)\n", count(mask))
    @printf("BEFORE fix: obj_inner.lower_limit = %.6e (== -KNITRO.KN_INFINITY, i.e. permanently disabled)\n", obj_inner.lower_limit)

    # THE FIX: wire up the SAME KNITRO-native early-stop mechanism this session's main
    # correction already validated for the OUTER search's own PsiObjectiveBundleImplicit --
    # here applied to the PsiObjectiveBundleDelta object nuisance_profile.jl actually uses,
    # which never had it set at all.
    obj_inner.lower_limit = -(LOWER_LIMIT_CAP + LOWER_LIMIT_GUARD)
    @printf("AFTER fix:  obj_inner.lower_limit = %.6f\n\n", obj_inner.lower_limit)
    flush(stdout)

    r0 = evaluate_melitz_delta(theta_calib, ctx, obj_inner; cold=true, store_G=false)
    @printf("calibration point: g=%.6f  Delta=%.6e  nStatus=%d\n", theta_calib[1], r0.Delta, r0.nStatus)
    flush(stdout)

    theta_g = copy(theta_calib); theta_g[1] += G_STEP
    @printf("canary point: g=%.6f (calibration %+.4f)\n\n", theta_g[1], G_STEP)
    flush(stdout)

    t_sweep0 = time()
    records = CallRecord[]

    function on_start(theta, call_index, kind)
        t = time() - t_sweep0
        pert = norm(theta .- theta_g)
        rec = CallRecord(call_index, kind, t, NaN, copy(theta), NaN, -999)
        push!(records, rec)
        @printf("[START] kind=%-3s idx=%3d  t=%8.2fs  ||theta-theta_g||=%.6e\n", kind, call_index, t, pert)
        flush(stdout)
        serialize(DUMP_PATH, records)
    end

    function on_eval(theta, val, nStatus, kind)
        t = time() - t_sweep0
        idxs = [i for (i, r) in enumerate(records) if r.kind == kind && isnan(r.finished_at)]
        if isempty(idxs)
            @printf("[EVAL, unmatched] kind=%-3s  t=%8.2fs  val=%.6e  nStatus=%d\n", kind, t, val, nStatus)
        else
            i = last(idxs)
            records[i].finished_at = t
            records[i].val = val
            records[i].nStatus = nStatus
            elapsed = t - records[i].started_at
            # threshold_crossed[]/threshold_crossing_bound[] (PsiObjectiveBundleDelta does NOT
            # carry these fields -- only PsiObjectiveBundleImplicit does, per this session's
            # own earlier `AboveEvaluationCap` work -- so we cannot read a "did lower_limit
            # fire" flag directly off obj_inner here; nStatus/val/elapsed are the only signal
            # available for THIS object type, which is exactly what we compare below).
            @printf("[ EVAL ] kind=%-3s idx=%3d  t=%8.2fs  elapsed=%7.2fs  val=%.6e  nStatus=%d\n",
                kind, records[i].call_index, t, elapsed, val, nStatus)
        end
        flush(stdout)
        serialize(DUMP_PATH, records)
    end

    result = nothing
    try
        result = solve_melitz_nuisance_min_delta(ctx, obj_inner, theta_g; free_mask=mask,
            radius=RADIUS, gradient_backend=:B_direct_argument_parallel, h=1e-4,
            inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT_CANARY, warm_start_x=r0.dual_x,
            on_start=on_start, on_eval=on_eval)
        @printf("\nCOMPLETED: nStatus=%d  Delta_min=%.6e  wall=%.2fs\n", result.nStatus, result.Delta_min, time() - t_sweep0)
    catch e
        @printf("\nEXCEPTION after %.2fs: %s\n", time() - t_sweep0, sprint(showerror, e))
    end

    println("\n" * "="^100)
    println("CALL RECORD SUMMARY")
    println("="^100)
    for r in records
        status_str = isnan(r.finished_at) ? "NEVER COMPLETED (hung / run ended before return)" :
            @sprintf("completed, elapsed=%.2fs, val=%.6e, nStatus=%d", r.finished_at - r.started_at, r.val, r.nStatus)
        @printf("idx=%3d kind=%-3s started_at=%8.2fs  %s\n", r.call_index, r.kind, r.started_at, status_str)
    end

    println("\nSerializing all tried points to disk...")
    serialize(DUMP_PATH, records)
    @printf("wrote %d call records to %s\n", length(records), DUMP_PATH)

    BLAS.set_num_threads(1)
    println("\nDONE.")
    return records
end

main()
