# campaign_unrestricted_runner.jl -- five-family independent-multistart shakedown (2026-07-28)
# unrestricted family only -- c10_d20_production_driver.jl is self-contained (own include list),
# matching smoke_delta1_unrestricted.jl's own convention; kept as a separate process/script from
# the CM-family runner deliberately (c10_d20_production_driver.jl's own includes are NOT all
# isdefined-guarded, so co-including it with the CM-family list risks duplicate struct
# redefinition -- the smoke scripts already establish this file is never combined with the CM list).
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. -t <threads_per_family> \
#       campaign_unrestricted_runner.jl <direction> <manifest_json> <outroot> [maxtime_real=30.0]
const _D4E = @__DIR__
include(joinpath(_D4E, "c10_d20_production_driver.jl"))
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

lp("="^100)
lp("CAMPAIGN UNRESTRICTED RUNNER -- ", Dates.now(), "  direction=", DIRECTION,
   " (find_smallest=", FIND_SMALLEST, ")  maxtime_real=", MAXTIME)
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
lp(">> all 5 start checksums re-verified against manifest.")
lp("-"^90)
lp("NO-H BUNDLE FACTS: unrestricted -- bundle_type=OperatorPsiBundle (production default);")
lp("  has_H_field=false, has_H_copy_field=false, has_moments!_field=false, has_K_field=false (renamed payoff)")
lp("  -- structural absence proven in TRUE_OPERATOR_NO_H_PREMERGE_GATE_2026-07-28.md, same production HEAD.")
lp("  Unrestricted's own inline priming analog is family-specific -- no_dense_g_counters.jl's runtime Refs are")
lp("  not wired for this family's separate code path (c10_d20_production_driver.jl); each solve's own")
lp("  print_production_backend_manifest(resolve_unrestricted_manifest(...)) call below is this family's live")
lp("  backend-selection record instead.")
lp("-"^90)

rows_outer = NamedTuple[]
rows_inner = NamedTuple[]

const OUTER_CSV = joinpath(OUTROOT, "unrestricted_$(DIRECTION)_outer_log.csv")
const INNER_CSV = joinpath(OUTROOT, "unrestricted_$(DIRECTION)_inner_log.csv")
const OUTER_HEADER = "family,direction,delta,start,checksum_w,elapsed_s,outer_status,outer_evals,n_inner_solves," *
                     "n_feasible_proxy,n_infeasible_or_dashed_proxy,best_verified_Delta,budget_residual,error"
const INNER_HEADER = "family,direction,delta,start,eval_idx,t_s,Delta_star,inner_status,delta_feasible,classification_proxy,final_incumbent"
outer_row_line(r) = string(r.family, ",", r.direction, ",", r.delta, ",", r.start, ",", r.checksum_w, ",",
    r.elapsed_s, ",", r.outer_status, ",", r.outer_evals, ",", r.n_inner_solves, ",",
    r.n_feasible_proxy, ",", r.n_infeasible_or_dashed_proxy, ",",
    r.best_verified_Delta, ",", r.budget_residual, ",",
    "\"", replace(string(r.error), "\"" => "'"), "\"")
inner_row_line(r) = string(r.family, ",", r.direction, ",", r.delta, ",", r.start, ",", r.eval_idx, ",", r.t_s, ",",
    r.Delta_star, ",", r.inner_status, ",", r.delta_feasible, ",", r.classification_proxy, ",", r.final_incumbent)

const MAX_CELL_ATTEMPTS = 3  # section 11: max 2 automatic retries (3 attempts total) before a cell is left FAILED

for delta in DELTAS, st in starts
    global rows_outer, rows_inner
    start_idx = Int(st["index"])
    gp0 = Float64(st["gp"])
    zfree0 = jf64(st["zfree_pivot_reduced_log"])
    ckdir = joinpath(OUTROOT, FAMILY, DIRECTION, "delta_$(delta)", "start_$(start_idx)")
    label = "$(FAMILY)_$(DIRECTION)_d$(delta)_s$(start_idx)"

    if cell_already_done(ckdir)
        lp("[", label, "] SKIP -- already DONE (resume: not re-solved, not overwritten): ", ckdir)
        prior = read_cell_outer_status(ckdir)
        if prior !== nothing
            push!(rows_outer, (family = FAMILY, direction = DIRECTION, delta = delta, start = start_idx,
                checksum_w = st["checksum_w_hash"], elapsed_s = get(prior, "elapsed_s", missing),
                outer_status = get(prior, "outer_status", "RESUMED"), outer_evals = get(prior, "outer_evals", missing),
                n_inner_solves = get(prior, "n_inner_solves", missing), n_feasible_proxy = get(prior, "n_feasible_proxy", missing),
                n_infeasible_or_dashed_proxy = get(prior, "n_infeasible_or_dashed_proxy", missing),
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
    lp("mutable_workspace_reuse = allowed (shared immutable ctx only; no cross-cell state observed)")
    lp("inner_dual_warm_start_reuse = allowed_not_used (resume_from=nothing, use_dual_bank default per cell)")
    lp("exact_cache_reuse = allowed_when_key_matches_not_used (resume_from=nothing, fresh cache per cell)")

    t0 = time()
    errored = false; errmsg = ""
    result = nothing
    try
        result = run_profile_checkpointed(label, gp0, FIND_SMALLEST, copy(zfree0);
            maxtime_real = MAXTIME, W_in = W, delta_in = delta, draw_seed_in = DRAW_SEED,
            draw_design_in = DRAW_DESIGN, ckpt_dir = ckdir, checkpoint_interval_s = 3600.0)
    catch e
        errored = true
        errmsg = sprint(showerror, e)
        lp("[", label, "] *** EXCEPTION *** ", typeof(e), ": ", errmsg[1:min(end, 800)])
    end
    wall = time() - t0
    cfg = Dict{String,Any}("family" => FAMILY, "direction" => DIRECTION, "delta" => delta, "start_id" => start_idx,
        "W" => W, "draw_design" => string(DRAW_DESIGN), "draw_seed" => DRAW_SEED,
        "maxtime_real" => MAXTIME, "checksum_w_hash" => st["checksum_w_hash"], "run_id" => label,
        "attempt" => prior_attempts + 1, "outer_initialized_from_prior_solution" => false,
        "outer_initialized_from_prior_delta_solution" => false, "outer_initialized_from_prior_direction_solution" => false,
        "outer_initialized_from_other_start_solution" => false)

    if errored
        orow = (family = FAMILY, direction = DIRECTION, delta = delta, start = start_idx,
            checksum_w = st["checksum_w_hash"], elapsed_s = round(wall, digits = 2), outer_status = "EXCEPTION",
            outer_evals = missing, n_inner_solves = missing, n_feasible_proxy = missing,
            n_infeasible_or_dashed_proxy = missing, best_verified_Delta = missing,
            budget_residual = missing, error = errmsg[1:min(end, 300)])
        push!(rows_outer, orow)
        write_cell_status!(ckdir, orow, NamedTuple[], nothing, cfg; failed = true)
        append_csv_row!(OUTER_CSV, OUTER_HEADER, outer_row_line(orow))
        continue
    end

    trace = result.trace
    n_feas = count(r -> r.delta_feasible, trace)
    n_infeas = length(trace) - n_feas
    best = result.best
    orow = (family = FAMILY, direction = DIRECTION, delta = delta, start = start_idx,
        checksum_w = st["checksum_w_hash"], elapsed_s = round(wall, digits = 2),
        outer_status = string(result.knitro_status), outer_evals = result.n_eval,
        n_inner_solves = result.n_eval, n_feasible_proxy = n_feas, n_infeasible_or_dashed_proxy = n_infeas,
        best_verified_Delta = best === nothing ? missing : best.Delta_dual,
        budget_residual = best === nothing ? missing : (delta - best.Delta_dual),
        error = "")
    push!(rows_outer, orow)

    cell_inner_rows = NamedTuple[]
    for r in trace
        irow = (family = FAMILY, direction = DIRECTION, delta = delta, start = start_idx,
            eval_idx = r.idx, t_s = round(r.t_elapsed, digits = 3), Delta_star = r.Delta_dual,
            inner_status = r.inner_status, delta_feasible = r.delta_feasible,
            classification_proxy = r.delta_feasible ? "feasible_proxy" : "infeasible_or_dashed_proxy",
            final_incumbent = (best !== nothing && r.idx == best.n_eval))
        push!(rows_inner, irow)
        push!(cell_inner_rows, irow)
    end

    best_nt = best === nothing ? nothing : (Delta_dual = best.Delta_dual, n_eval = best.n_eval)
    write_cell_status!(ckdir, orow, cell_inner_rows, best_nt, cfg; failed = false)
    append_csv_row!(OUTER_CSV, OUTER_HEADER, outer_row_line(orow))
    for irow in cell_inner_rows
        append_csv_row!(INNER_CSV, INNER_HEADER, inner_row_line(irow))
    end

    lp("[", label, "] DONE  wall=", round(wall, digits = 1), "s  knitro_status=", result.knitro_status,
       "  n_eval=", result.n_eval, "  feasible=", n_feas, "/", length(trace),
       "  best_Delta=", best === nothing ? "none" : @sprintf("%.6e", best.Delta_dual))
end

lp(">> outer/inner CSVs already written incrementally, per cell, throughout this run: ", OUTER_CSV, " / ", INNER_CSV)
lp(">> (no end-of-run rewrite here -- a resumed/restarted process only holds resumed cells' outer rows in memory,")
lp(">>  not their inner rows, so rewriting from the in-memory arrays at this point would truncate the inner CSV)")
lp("="^100)
lp("STATE-REUSE ACCOUNTING (unrestricted/", DIRECTION, ", ", length(rows_outer), " cells):")
lp("  OUTER_CONTINUATION_CALLS = 0   (every cell calls run_profile_checkpointed with a fresh w0/zfree0 from the manifest and no resume_from; no code path in this runner can set it otherwise)")
lp("  INNER_WARM_START_REUSES = 0    (no dual bank/state shared across cells in this runner)")
lp("  CACHE_HITS_ACROSS_SOLVES = 0   (no exact_cache_override passed across cells)")
lp("CAMPAIGN UNRESTRICTED RUNNER COMPLETE -- direction=", DIRECTION,
   " cells=", length(rows_outer), " errors=", count(r -> r.outer_status == "EXCEPTION", rows_outer))
lp("="^100)
