# ============================================================================
# Shared config/helpers for the 4-method (LC/LU/GC/GU) head-to-head comparison.
# See HEAD_TO_HEAD_PROMPT.md for the full experimental design. Must be
# `include`d AFTER:
#   ENV["SKIP_BATCH_LOOP"] = "true"
#   ENV["FAKEDATA"]="3"; ENV["DVAL"]="20"; ENV["WVAL"]="80000"; ENV["PARALLEL_INVERSION"]="true"
#   include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
# which provides θr0, D, W, γ, U, σ, seq_gravcol, exact_inner_divergence_at, gp2kappa, etc.
# ============================================================================
using JLD2, Printf, Dates, LinearAlgebra

const H2H_DIR = @__DIR__
const SHARED_STARTS_PATH = joinpath(H2H_DIR, "shared_starts.jld2")

# Uniform per-solve wall-clock cap, user-approved 2026-07-15 (see HEAD_TO_HEAD_PROMPT.md
# "Fair-comparison requirements" -- same cap applied to every one of the 36 solves,
# regardless of method, so the comparison is not confounded by unequal compute budgets).
# LC/LU enforce this via KNITRO's own maxtime_real (full_aod_diag/csw_outer_90min.opt);
# GC/GU enforce it via BlackBoxOptim's own MaxTime kwarg (BBO_MAXTIME env, set to match
# by each driver). 9 solves x 90min, run as 4 concurrent method-jobs (NOT parallelized
# within a job -- see rationale in common.jl below) => total wall clock ~13.5h. Env-
# overridable ONLY for smoke-testing the pipeline mechanics with a short cap
# (H2H_CAP_SECONDS) -- the real overnight run relies on the 5400.0 default.
const CAP_SECONDS = parse(Float64, get(ENV, "H2H_CAP_SECONDS", "5400.0"))

# exact_inner_divergence_at's cold-started inner KNITRO dual solve can fail outright and
# return a huge sentinel value (observed: exactly 1e10) rather than signaling failure
# cleanly -- NOT caught by gravity_ok (gravity itself can converge fine even when the dual
# solve on top of it fails). Any delta_star at or above this threshold is a solve failure,
# not a genuine achieved divergence (multistart_screening_d20.jl never saw real divergence
# above ~1.5 even at relΔA~20). Matches the threshold already used in
# run_bbo_d20_profiled_deltastar.jl.
const MAX_SANE_DELTA_STAR = 50.0

# The 3 target points, EXACT saved best_feasible_gp/kappa values pulled directly from
# sequential_gravity/batch_out_realD20_W80000_fixeddualfdfull_scaled05/seq_upper_delta{0.1,1.0,2.0}.jld2
# (confirmed 2026-07-15 by loading the JLD2 files directly, matching HEAD_TO_HEAD_PROMPT.md's
# table exactly). For LC/GC these `delta` values are the divergence BUDGET given to the
# constrained search; for LU/GU, `gp`/`kappa` are the FIXED targets and `delta` is only a
# reference point (the budget the original LC run used to reach that gp).
const TARGETS = [
    (name = "T1", delta = 0.1, gp = 0.9728658401964491, kappa = 0.04481332038003094),
    (name = "T2", delta = 1.0, gp = 0.950259956422648,  kappa = 0.08151786294271901),
    (name = "T3", delta = 2.0, gp = 0.9440553861575977, kappa = 0.09149123004529891),
]

"""
    load_shared_starts()

Loads rand1/rand2 (the two log-normal A_od perturbations shared identically across all 4
methods) from shared_starts.jld2 -- generated ONCE by generate_shared_starts.jl. Asserts the
file's own recorded Acol_star matches THIS process's θr0[4:3+D] (catches an accidental
mismatch in D/W/data between when the shared file was generated and when a method driver
runs).
"""
function load_shared_starts()
    isfile(SHARED_STARTS_PATH) || error(
        "shared_starts.jld2 not found at $SHARED_STARTS_PATH -- run generate_shared_starts.jl first")
    d = JLD2.load(SHARED_STARTS_PATH)
    Acol_star_here = θr0[4:3+D]
    @assert isapprox(d["Acol_star"], Acol_star_here; rtol = 1e-10) "shared_starts.jld2's Acol_star does not match this run's θr0 -- regenerate (mismatched D/W/data setup)"
    (rand1 = Float64.(d["rand1"]), rand2 = Float64.(d["rand2"]))
end

"""
    load_done(path)

Returns the loaded Dict if `path` exists AND has `done=true`, else `nothing`. The
skip-if-done pattern used by every driver's per-solve loop (same convention as
run_profiled_production.jl's own batch loop) -- a killed/restarted process re-derives
everything (including "which start won its target") purely from these files, so no separate
resume-state file is needed.
"""
function load_done(path)
    isfile(path) || return nothing
    d = try
        JLD2.load(path)
    catch
        return nothing
    end
    get(d, "done", false) === true ? d : nothing
end

istop_delta(δstar) = !isfinite(δstar) || δstar >= MAX_SANE_DELTA_STAR
