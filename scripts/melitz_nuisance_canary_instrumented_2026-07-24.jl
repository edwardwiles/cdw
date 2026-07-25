# 2026-07-24/25, FINAL diagnostic pass on the nuisance-profile stall (user-directed): records
# per-call timing (on_start fires BEFORE inner_loop, so a hanging call's own attempted theta
# is on record even if it never returns; on_eval fires AFTER, giving elapsed time for calls
# that DO complete) and the FULL theta vector for every attempted point, serialized to disk
# for later investigation. Bounded by an OS-level `timeout` wrapper (the shell command that
# launches this script), not relied upon from within Julia -- the prior attempt confirmed
# KNITRO's own maxtime_real is checked only BETWEEN callback returns, so it cannot preempt a
# single slow callback.
#
# Usage (with OS-level timeout, from the launching shell):
#   timeout 300 julia --project=. -t 16 scripts/melitz_nuisance_canary_instrumented_2026-07-24.jl

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
const DUMP_PATH = joinpath(dirname(@__DIR__), "melitz_nuisance_canary_tried_points_2026-07-24.jls")

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
    kind::Symbol           # :fc or :ga
    started_at::Float64    # time() at on_start
    finished_at::Float64   # time() at on_eval, NaN if never completed (call hung / run killed)
    theta::Vector{Float64}
    val::Float64
    nStatus::Int
end

function main()
    calib = load_calibration()
    @printf("radius=%.4f  g_step=%.4f  (canary outer opts: maxit=8, maxtime_real=60s)\n", RADIUS, G_STEP)
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
    @printf("canary point: g=%.6f (calibration %+.4f)\n\n", theta_g[1], G_STEP)
    flush(stdout)

    t_sweep0 = time()
    records = CallRecord[]
    open_calls = Dict{Tuple{Symbol,Int},Int}()   # (kind, call_index) -> index into records

    function on_start(theta, call_index, kind)
        t = time() - t_sweep0
        pert = norm(theta .- theta_g)
        rec = CallRecord(call_index, kind, t, NaN, copy(theta), NaN, -999)
        push!(records, rec)
        open_calls[(kind, call_index)] = length(records)
        @printf("[START] kind=%-3s idx=%3d  t=%8.2fs  ||theta-theta_g||=%.6e\n", kind, call_index, t, pert)
        flush(stdout)
        serialize(DUMP_PATH, records)   # see on_eval's matching comment: robust to an abrupt kill
    end

    function on_eval(theta, val, nStatus, kind)
        t = time() - t_sweep0
        call_index = kind == :fc ? -1 : -1   # placeholder; matched via most-recent open call of this kind below
        # Match to the most recently STARTED, not-yet-finished call of this kind (cb_F!/cb_G!
        # calls of the SAME kind never overlap -- KNITRO calls them strictly sequentially).
        idxs = [i for (i, r) in enumerate(records) if r.kind == kind && isnan(r.finished_at)]
        if isempty(idxs)
            @printf("[EVAL, unmatched] kind=%-3s  t=%8.2fs  val=%.6e  nStatus=%d\n", kind, t, val, nStatus)
        else
            i = last(idxs)
            records[i].finished_at = t
            records[i].val = val
            records[i].nStatus = nStatus
            elapsed = t - records[i].started_at
            @printf("[ EVAL ] kind=%-3s idx=%3d  t=%8.2fs  elapsed=%7.2fs  val=%.6e  nStatus=%d\n",
                kind, records[i].call_index, t, elapsed, val, nStatus)
        end
        flush(stdout)
        # Incremental dump after every completed call -- robust to an abrupt OS-level kill
        # (SIGTERM/SIGKILL) that would otherwise skip the end-of-script serialize entirely,
        # per the user's own explicit "record the actual tried points" request.
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

    println("\nSerializing all tried points (theta vectors) to disk for later investigation...")
    serialize(DUMP_PATH, records)
    @printf("wrote %d call records to %s\n", length(records), DUMP_PATH)

    BLAS.set_num_threads(1)
    println("\nDONE.")
    return records
end

main()
