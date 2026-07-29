# campaign_unrestricted_runner.jl -- five-family overnight campaign, unrestricted family
# (rewritten 2026-07-29 to fix a real bug: the original 2026-07-28 shakedown version of this
# script called `run_profile_checkpointed` (c10_d20_production_driver.jl), which HOLDS gp FIXED
# at the starting value and only minimizes Delta over the A-coordinates -- confirmed empirically
# by deserializing a live checkpoint mid-campaign: gp was bit-identical across every one of 139
# evaluations in one cell. That function is stage 1 ("profile") of a two-stage design; the
# campaign never called stage 2 ("polish", run_polish_checkpointed_unified), which is what
# actually jointly optimizes (gp, A) to find the extremal gp subject to Delta<=delta -- matching
# what the other 4 families' drivers do. This exact regression was already found and fixed in
# unrestricted_stage_runner.jl on 2026-07-26 (see docs/UNRESTRICTED_UNIFIED_DRIVER_RELEASE_2026-07-26.md)
# but that fix never made it into this campaign script. This rewrite follows
# unrestricted_stage_runner.jl's own calling convention for run_polish_checkpointed_unified.
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. -t <threads_per_family> \
#       campaign_unrestricted_runner.jl <direction> <manifest_json> <outroot> [maxtime_real=30.0] [deltas_csv] [starts_csv]
const _D4E = @__DIR__
for f in ["c10_d20_production_driver.jl", "flexible_theta.jl", "flexible_theta_aspace_production.jl",
          "outer_coordinate_layout.jl", "c10_d20_production_driver_unified.jl"]
    include(joinpath(_D4E, f))
end
include(joinpath(_D4E, "json_lite.jl"))
include(joinpath(_D4E, "campaign_cell_io.jl"))
using Printf, Dates

lp(xs...) = (println(xs...); flush(stdout))

const DIRECTION     = ARGS[1]
const MANIFEST_PATH = ARGS[2]
const OUTROOT       = ARGS[3]
const MAXTIME       = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 30.0
const DELTAS_OVERRIDE = length(ARGS) >= 5 && !isempty(ARGS[5]) ? parse.(Float64, split(ARGS[5], ",")) : nothing
const STARTS_OVERRIDE = length(ARGS) >= 6 && !isempty(ARGS[6]) ? parse.(Int, split(ARGS[6], ",")) : nothing

DIRECTION in ("upper", "lower") || error("campaign_unrestricted_runner: direction must be upper|lower, got $DIRECTION")
const FIND_SMALLEST = DIRECTION == "upper"
const FAMILY = "unrestricted"

const DELTAS = DELTAS_OVERRIDE === nothing ? [0.01, 0.1, 0.5, 1.0, 2.0] : DELTAS_OVERRIDE
const W = 100_000
const DRAW_DESIGN = :sobol_randomized
const DRAW_SEED = 20260719
# theta=fixed, A_coordinate_mode=powered_aspace, gp_coordinate_mode=raw -- matches the other 4
# families' own production defaults (confirmed live in their own smoke-test log lines) and the
# task's science config ("theta = fixed", "transformed-A production coordinates = true").
const LAYOUT = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :powered_aspace, gp_coordinate_mode = :raw)

lp("="^100)
lp("CAMPAIGN UNRESTRICTED RUNNER (UNIFIED DRIVER) -- ", Dates.now(), "  direction=", DIRECTION,
   " (find_smallest=", FIND_SMALLEST, ")  maxtime_real=", MAXTIME)
lp("public_driver = run_polish_checkpointed_unified (fixed 2026-07-29; was run_profile_checkpointed, a fixed-gp legacy stage that never optimized gp -- see file header)")
lp("layout = ", LAYOUT)
lp("continuation_enabled = false")
lp("prior_delta_state_loaded = false")
lp("prior_direction_state_loaded = false")
lp("cross_start_state_loaded = false")
lp("julia threads=", Threads.nthreads(), " BLAS threads=", BLAS.get_num_threads())
lp("="^100)

manifest = json_load(MANIFEST_PATH)
all_starts = manifest["starts"]
length(all_starts) == 5 || error("manifest at $MANIFEST_PATH does not have exactly 5 starts (has $(length(all_starts)))")
starts = STARTS_OVERRIDE === nothing ? all_starts : filter(s -> Int(s["index"]) in STARTS_OVERRIDE, all_starts)
isempty(starts) && error("STARTS_OVERRIDE=$STARTS_OVERRIDE matched no manifest starts")
lp(">> manifest loaded: ", length(starts), " starts")

for st in starts
    w_a = jf64(st["w_transformed_a"])
    h = string(hash(w_a), base = 16)
    h == st["checksum_w_hash"] || error("start $(st["index"]): recomputed checksum_w_hash mismatch -- refusing to run.")
end
lp(">> all start checksums re-verified against manifest.")
lp("-"^90)
lp("NO-H BUNDLE FACTS: unrestricted -- bundle_type=OperatorPsiBundle (production default);")
lp("  has_H_field=false, has_H_copy_field=false, has_moments!_field=false, has_K_field=false (renamed payoff)")
lp("  -- structural absence proven in TRUE_OPERATOR_NO_H_PREMERGE_GATE_2026-07-28.md, same production HEAD.")
lp("-"^90)

rows_outer = NamedTuple[]
rows_inner = NamedTuple[]

const OUTER_CSV = joinpath(OUTROOT, "unrestricted_$(DIRECTION)_outer_log.csv")
const INNER_CSV = joinpath(OUTROOT, "unrestricted_$(DIRECTION)_inner_log.csv")
const OUTER_HEADER = "family,direction,delta,start,checksum_w,elapsed_s,outer_status,outer_evals,n_inner_solves," *
                     "n_verified,n_infeasible_or_dashed,best_verified_gp,best_verified_Delta,budget_residual,error"
const INNER_HEADER = "family,direction,delta,start,eval_idx,t_s,gp,theta,Delta_star,feasible,classification_proxy,final_incumbent"
outer_row_line(r) = string(r.family, ",", r.direction, ",", r.delta, ",", r.start, ",", r.checksum_w, ",",
    r.elapsed_s, ",", r.outer_status, ",", r.outer_evals, ",", r.n_inner_solves, ",",
    r.n_verified, ",", r.n_infeasible_or_dashed, ",",
    r.best_verified_gp, ",", r.best_verified_Delta, ",", r.budget_residual, ",",
    "\"", replace(string(r.error), "\"" => "'"), "\"")
inner_row_line(r) = string(r.family, ",", r.direction, ",", r.delta, ",", r.start, ",", r.eval_idx, ",", r.t_s, ",",
    r.gp, ",", r.theta, ",", r.Delta_star, ",", r.feasible, ",", r.classification_proxy, ",", r.final_incumbent)

const MAX_CELL_ATTEMPTS = 3  # section 11: max 2 automatic retries (3 attempts total) before a cell is left FAILED

for delta in DELTAS, st in starts
    global rows_outer, rows_inner
    start_idx = Int(st["index"])
    w_a = jf64(st["w_transformed_a"])
    ckdir = joinpath(OUTROOT, FAMILY, DIRECTION, "delta_$(delta)", "start_$(start_idx)")
    label = "$(FAMILY)_$(DIRECTION)_d$(delta)_s$(start_idx)"

    if cell_already_done(ckdir)
        lp("[", label, "] SKIP -- already DONE (resume: not re-solved, not overwritten): ", ckdir)
        prior = read_cell_outer_status(ckdir)
        if prior !== nothing
            push!(rows_outer, (family = FAMILY, direction = DIRECTION, delta = delta, start = start_idx,
                checksum_w = st["checksum_w_hash"], elapsed_s = get(prior, "elapsed_s", missing),
                outer_status = get(prior, "outer_status", "RESUMED"), outer_evals = get(prior, "outer_evals", missing),
                n_inner_solves = get(prior, "n_inner_solves", missing), n_verified = get(prior, "n_verified", missing),
                n_infeasible_or_dashed = get(prior, "n_infeasible_or_dashed", missing),
                best_verified_gp = get(prior, "best_verified_gp", missing),
                best_verified_Delta = get(prior, "best_verified_Delta", missing),
                budget_residual = get(prior, "budget_residual", missing), error = ""))
        end
        continue
    end
    prior_attempts = cell_attempt_count(ckdir)
    if prior_attempts >= MAX_CELL_ATTEMPTS
        lp("[", label, "] SKIP -- exceeded max attempts (", prior_attempts, "/", MAX_CELL_ATTEMPTS,
           "); left FAILED, not retried further")
        continue
    end
    rm(ckdir; force = true, recursive = true); mkpath(ckdir)
    open(cell_attempt_file(ckdir), "w") do io; print(io, prior_attempts + 1); end
    lp("="^100)
    lp("[", label, "] family=unrestricted direction=", DIRECTION, " delta=", delta, " start=", start_idx,
       " checksum_w=", st["checksum_w_hash"], " ckdir=", ckdir)
    lp("outer_initial_start_id = ", start_idx)
    lp("outer_initial_coordinate_checksum = ", st["checksum_w_hash"])
    lp("outer_initialized_from_prior_solution = false")
    lp("outer_initialized_from_prior_delta_solution = false")
    lp("outer_initialized_from_prior_direction_solution = false")
    lp("outer_initialized_from_other_start_solution = false")
    lp("mutable_workspace_reuse = allowed (shared immutable ctx built fresh per cell, matching unrestricted_stage_runner.jl's own convention; no cross-cell state observed)")
    lp("inner_dual_warm_start_reuse = allowed_not_used (resume_from=nothing, use_dual_bank default per cell)")
    lp("exact_cache_reuse = allowed_when_key_matches_not_used (resume_from=nothing, fresh cache per cell)")

    t0 = time()
    errored = false; errmsg = ""
    result = nothing
    try
        # w_a = [gp; A_nonpivot_native] is ALREADY in run_polish_checkpointed_unified's own
        # :powered_aspace, pivot-reduced outer-vector format (confirmed empirically: decoding w_a
        # directly reproduces the known-good reference reconstruction's gravity_value to the exact
        # same 8.97947651906942e-18, bit-for-bit) -- use it directly, exactly like the other 4
        # families use w_a directly as their own w0. An earlier version of this fix incorrectly
        # ran w_a through pivot_expand + reduce_to_w_unified (treating w_a as raw pivot-reduced
        # log-A, which it is NOT -- that produced a different, wrong point: max|logA diff|=30.6 vs
        # true calibration, still gravity-feasible but economically nonsensical, hence "not
        # inner-feasible"). No transform needed; w_start = w_a.
        w_start = copy(w_a)

        result = run_polish_checkpointed_unified(label, FIND_SMALLEST, w_start;
            layout = LAYOUT, theta_lo = NaN, theta_hi = NaN,
            maxtime_real = MAXTIME, W_in = W, delta_in = delta, draw_seed_in = DRAW_SEED,
            draw_design_in = DRAW_DESIGN, ckpt_dir = ckdir, checkpoint_interval_s = 3600.0,
            resume_from = nothing, destination_sample = :exclude_row)
    catch e
        errored = true
        errmsg = sprint(showerror, e)
        lp("[", label, "] *** EXCEPTION *** ", typeof(e), ": ", errmsg[1:min(end, 800)])
    end
    wall = time() - t0
    cfg = Dict{String,Any}("family" => FAMILY, "direction" => DIRECTION, "delta" => delta, "start_id" => start_idx,
        "W" => W, "draw_design" => string(DRAW_DESIGN), "draw_seed" => DRAW_SEED,
        "maxtime_real" => MAXTIME, "checksum_w_hash" => st["checksum_w_hash"], "run_id" => label,
        "public_driver" => "run_polish_checkpointed_unified", "attempt" => prior_attempts + 1,
        "outer_initialized_from_prior_solution" => false, "outer_initialized_from_prior_delta_solution" => false,
        "outer_initialized_from_prior_direction_solution" => false, "outer_initialized_from_other_start_solution" => false)

    if errored
        orow = (family = FAMILY, direction = DIRECTION, delta = delta, start = start_idx,
            checksum_w = st["checksum_w_hash"], elapsed_s = round(wall, digits = 2), outer_status = "EXCEPTION",
            outer_evals = missing, n_inner_solves = missing, n_verified = missing,
            n_infeasible_or_dashed = missing, best_verified_gp = missing, best_verified_Delta = missing,
            budget_residual = missing, error = errmsg[1:min(end, 300)])
        push!(rows_outer, orow)
        write_cell_status!(ckdir, orow, NamedTuple[], nothing, cfg; failed = true)
        append_csv_row!(OUTER_CSV, OUTER_HEADER, outer_row_line(orow))
        continue
    end

    trace = result.trace
    n_verified = count(r -> r.feasible, trace)   # feasible here already implies is_verified_success at the
    # time it was recorded as a candidate for best_feasible (see run_polish_checkpointed_unified's own
    # is_new_best = feasible && is_verified_success(r) && ... gate) -- unlike the legacy driver's naive
    # fallback, best_feasible here is genuinely nothing unless a verified, delta-feasible point was found.
    n_infeasible = length(trace) - n_verified
    best = result.best_feasible
    orow = (family = FAMILY, direction = DIRECTION, delta = delta, start = start_idx,
        checksum_w = st["checksum_w_hash"], elapsed_s = round(wall, digits = 2),
        outer_status = string(result.knitro_status), outer_evals = result.n_eval,
        n_inner_solves = result.n_eval, n_verified = n_verified, n_infeasible_or_dashed = n_infeasible,
        best_verified_gp = best === nothing ? missing : best.gp,
        best_verified_Delta = best === nothing ? missing : best.Delta,
        budget_residual = best === nothing ? missing : (delta - best.Delta),
        error = "")
    push!(rows_outer, orow)

    cell_inner_rows = NamedTuple[]
    for r in trace
        irow = (family = FAMILY, direction = DIRECTION, delta = delta, start = start_idx,
            eval_idx = r.idx, t_s = round(r.t_elapsed, digits = 3), gp = r.gp, theta = r.theta, Delta_star = r.Delta_dual,
            feasible = r.feasible, classification_proxy = r.feasible ? "verified_feasible" : "infeasible_or_dashed",
            final_incumbent = (best !== nothing && r.idx == best.n_eval))
        push!(rows_inner, irow)
        push!(cell_inner_rows, irow)
    end

    best_nt = best === nothing ? nothing : (gp = best.gp, Delta = best.Delta, n_eval = best.n_eval)
    write_cell_status!(ckdir, orow, cell_inner_rows, best_nt, cfg; failed = false)
    append_csv_row!(OUTER_CSV, OUTER_HEADER, outer_row_line(orow))
    for irow in cell_inner_rows
        append_csv_row!(INNER_CSV, INNER_HEADER, inner_row_line(irow))
    end

    lp("[", label, "] DONE  wall=", round(wall, digits = 1), "s  knitro_status=", result.knitro_status,
       "  n_eval=", result.n_eval, "  verified=", n_verified, "/", length(trace),
       "  best_gp=", best === nothing ? "none" : @sprintf("%.10f", best.gp),
       "  best_Delta=", best === nothing ? "none" : @sprintf("%.6e", best.Delta))
end

lp(">> outer/inner CSVs already written incrementally, per cell, throughout this run: ", OUTER_CSV, " / ", INNER_CSV)
lp("="^100)
lp("STATE-REUSE ACCOUNTING (unrestricted/", DIRECTION, ", ", length(rows_outer), " cells):")
lp("  OUTER_CONTINUATION_CALLS = 0   (every cell calls run_polish_checkpointed_unified with a fresh w_start derived from the manifest and no resume_from; no code path in this runner can set it otherwise)")
lp("  INNER_WARM_START_REUSES = 0    (no dual bank/state shared across cells in this runner)")
lp("  CACHE_HITS_ACROSS_SOLVES = 0   (no exact_cache_override passed across cells)")
lp("CAMPAIGN UNRESTRICTED RUNNER COMPLETE -- direction=", DIRECTION,
   " cells=", length(rows_outer), " errors=", count(r -> r.outer_status == "EXCEPTION", rows_outer))
lp("="^100)
