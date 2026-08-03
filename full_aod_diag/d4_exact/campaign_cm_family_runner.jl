# campaign_cm_family_runner.jl -- five-family independent-multistart shakedown (2026-07-28)
#
# Covers the four CM-code-path families: flexible_cm, common_frechet, cm_meanzc, origin_zc.
# (unrestricted uses campaign_unrestricted_runner.jl -- separate self-contained include list,
# matching smoke_delta1_unrestricted.jl's own convention.)
#
# Usage:
#   OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 julia --project=. -t 10 \
#       campaign_cm_family_runner.jl <family> <direction> <manifest_json> <outroot> [maxtime_real=30.0]
#
#   family:    flexible_cm | common_frechet | cm_meanzc | origin_zc
#   direction: upper | lower
#
#   ZC Hessian backend production integration (2026-08-01): cm_meanzc/origin_zc's validated H_ZZ/
#   H_CZ/H_EZ backends (blas_syrk/draw_chunk_reordered/drawmajor_v2) need real Julia threads (>=~10)
#   for core_hessian_backend=exact_winner_pair_parallel's own parallelism -- at -t 1 that backend
#   cannot parallelize at all, which this session's own measurement harness discovered the hard way
#   (see ZC_MEASUREMENT_METHODOLOGY_CORRECTION_2026-08-01.md). `-t 10` / `OPENBLAS_NUM_THREADS=8` is
#   the validated ten-by-ten deployment target for these two families specifically; flexible_cm/
#   common_frechet/unrestricted were not part of this validation and keep whatever thread count an
#   operator already uses for them.
#
# No continuation: every (delta,start) cell calls the real public checkpointed driver FRESH
# (resume_from=nothing, a brand-new ckpt_dir), so each cell gets its own outer model, incumbent
# ledger, mutable exact-cache, dual bank (off by default), and checkpoint namespace. Only the
# shared IMMUTABLE production context (ctx/pcx, built once per process) and the ex-ante manifest's
# stored start coordinates/duals are reused across cells -- exactly what the task spec allows.
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl", "postmerge_smoke_diagnostics.jl", "cross_hessian_live_stash_2026-07-28.jl"]
    include(joinpath(_D4E, f))
end
include(joinpath(_D4E, "json_lite.jl"))
include(joinpath(_D4E, "campaign_cell_io.jl"))
using Printf, Dates, Statistics

lp(xs...) = (println(xs...); flush(stdout))

const FAMILY        = ARGS[1]
const DIRECTION      = ARGS[2]
const MANIFEST_PATH  = ARGS[3]
const OUTROOT        = ARGS[4]
const MAXTIME        = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 30.0
const DELTAS_OVERRIDE = length(ARGS) >= 6 && !isempty(ARGS[6]) ? parse.(Float64, split(ARGS[6], ",")) : nothing
const STARTS_OVERRIDE = length(ARGS) >= 7 && !isempty(ARGS[7]) ? parse.(Int, split(ARGS[7], ",")) : nothing

FAMILY in ("flexible_cm", "common_frechet", "cm_meanzc", "origin_zc") ||
    error("campaign_cm_family_runner: unknown family $FAMILY")
DIRECTION in ("upper", "lower") || error("campaign_cm_family_runner: direction must be upper|lower, got $DIRECTION")
const FIND_SMALLEST = DIRECTION == "upper"

const DELTAS = DELTAS_OVERRIDE === nothing ? [0.01, 0.1, 0.5, 1.0, 2.0] : DELTAS_OVERRIDE

# ScientificManifest wiring (2026-08-03 hardening): every scientific setting below is now
# REQUIRED at every function this script calls (run_cm_upper/lower_checkpointed,
# run_originzc_upper/lower_checkpointed) -- there is no more "leave it at the function default"
# option anywhere in that call chain. This script's own single source of truth for those values
# is the ScientificManifest TOML file, not locally hardcoded consts (which is what allowed the
# earlier sigma confusion: this script and the functions it called each had their own separate
# idea of what the default should be). Override the manifest path via
# SCIENTIFIC_MANIFEST_TOML=/path/to/other.toml; there is no silent fallback if it's missing.
include(joinpath(_D4E, "..", "..", "scientific_manifest", "ScientificManifest.jl"))
using .ScientificManifestMod
const SCI_MANIFEST_PATH = get(ENV, "SCIENTIFIC_MANIFEST_TOML",
    joinpath(_D4E, "..", "..", "configs", "fullA_production_2026-08-03.toml"))
isfile(SCI_MANIFEST_PATH) ||
    error("campaign_cm_family_runner: ScientificManifest not found at $SCI_MANIFEST_PATH -- " *
          "this script refuses to run without one (set SCIENTIFIC_MANIFEST_TOML to override).")
const SCI = read_manifest_toml(SCI_MANIFEST_PATH)
let problems = validate_manifest(SCI)
    isempty(problems) || error("campaign_cm_family_runner: ScientificManifest at $SCI_MANIFEST_PATH " *
        "failed validation:\n  " * join(problems, "\n  "))
end
lp("ScientificManifest loaded: ", SCI_MANIFEST_PATH, "  sigma=", SCI.sigma, " W=", SCI.W,
   " draw_design=", SCI.draw_design, " draw_seed=", SCI.draw_seed,
   " destination_sample=", SCI.destination_sample, " exclude_diagonal_gravity=", SCI.exclude_diagonal_gravity,
   " gravity_exclude_cells=", SCI.gravity_exclude_cells)

const W = SCI.W
const DRAW_DESIGN = SCI.draw_design
const DRAW_SEED = SCI.draw_seed
const CM_L = SCI.L
const MEANZC_K = SCI.K_mean   # NOTE: the manifest currently has one shared K_mean/K_pair pair
const ORIGINZC_K = SCI.K_pair # for both families -- true today (both =1 in production) but would
                               # need separate per-family fields if that ever changes.

lp("="^100)
lp("CAMPAIGN CM-FAMILY RUNNER -- ", Dates.now(), "  family=", FAMILY, " direction=", DIRECTION,
   " (find_smallest=", FIND_SMALLEST, ")  maxtime_real=", MAXTIME)
lp("continuation_enabled = false")
lp("prior_delta_state_loaded = false")
lp("prior_direction_state_loaded = false")
lp("cross_start_state_loaded = false")
lp("julia threads=", Threads.nthreads(), " BLAS threads=", BLAS.get_num_threads())
# ZC Hessian backend production integration (2026-08-01): a real, measured pitfall this session --
# running cm_meanzc/origin_zc with too few Julia threads starves core_hessian_backend=
# exact_winner_pair_parallel and the H_CZ/H_EZ candidates of the real parallelism they need,
# silently inflating wall time (~2x, no error, no correctness change) rather than failing loudly.
# Warn, don't error -- an operator may have a deliberate reason to run fewer threads.
FAMILY in ("cm_meanzc", "origin_zc") && Threads.nthreads() < 10 &&
    lp("WARNING: family=", FAMILY, " is running with Threads.nthreads()=", Threads.nthreads(),
       " < 10 -- the validated H_ZZ/H_CZ/H_EZ backend speedups assume >=10 Julia threads for ",
       "core_hessian_backend=exact_winner_pair_parallel's own parallelism; expect real but ",
       "smaller wall-clock gains than ZC_HESSIAN_BACKEND_CLOSEOUT_MASTER_2026-08-01.md reports.")
lp("="^100)

manifest = json_load(MANIFEST_PATH)
all_starts = manifest["starts"]
length(all_starts) == 5 || error("manifest at $MANIFEST_PATH does not have exactly 5 starts (has $(length(all_starts)))")
starts = STARTS_OVERRIDE === nothing ? all_starts : filter(s -> Int(s["index"]) in STARTS_OVERRIDE, all_starts)
isempty(starts) && error("STARTS_OVERRIDE=$STARTS_OVERRIDE matched no manifest starts")
nu_meanzc = jf64(manifest["shared_extra_coordinates"]["cm_meanzc_nu"])
nu_originzc = jf64(manifest["shared_extra_coordinates"]["origin_zc_nu"])
lp(">> manifest loaded: ", length(starts), " starts, nu_meanzc=", nu_meanzc, " |nu_originzc|=", length(nu_originzc))

const SNAPS = nested_grid_sequence([10, 20, 50])
const PROBS_L50 = SNAPS[CM_L]

# recompute checksums for a fresh cross-check against the manifest's own stored hashes (defense
# against a stale/corrupted manifest silently feeding wrong coordinates into 25 real KNITRO solves)
for st in starts
    w_a = jf64(st["w_transformed_a"])
    h = string(hash(w_a), base = 16)
    h == st["checksum_w_hash"] || error("start $(st["index"]): recomputed checksum_w_hash mismatch " *
        "($(h) vs manifest $(st["checksum_w_hash"])) -- refusing to run a possibly-corrupted manifest.")
end
lp(">> all 5 start checksums re-verified against manifest.")

print_no_h_bundle_facts(FAMILY)
reset_no_h_counters!()

rows_outer = NamedTuple[]
rows_inner = NamedTuple[]

const OUTER_CSV = joinpath(OUTROOT, "$(FAMILY)_$(DIRECTION)_outer_log.csv")
const INNER_CSV = joinpath(OUTROOT, "$(FAMILY)_$(DIRECTION)_inner_log.csv")
const OUTER_HEADER = "family,direction,delta,start,checksum_w,elapsed_s,outer_status,outer_evals,n_inner_solves," *
                     "n_verified,n_approximate_proxy,n_dashed_or_infeasible_proxy,best_verified_gp,best_verified_Delta," *
                     "budget_residual,error"
const INNER_HEADER = "family,direction,delta,start,eval_idx,t_s,gp,Delta_star,feasible,verified_success,classification_proxy,final_incumbent"
outer_row_line(r) = string(r.family, ",", r.direction, ",", r.delta, ",", r.start, ",", r.checksum_w, ",",
    r.elapsed_s, ",", r.outer_status, ",", r.outer_evals, ",", r.n_inner_solves, ",",
    r.n_verified, ",", r.n_approximate_proxy, ",", r.n_dashed_or_infeasible_proxy, ",",
    r.best_verified_gp, ",", r.best_verified_Delta, ",", r.budget_residual, ",",
    "\"", replace(string(r.error), "\"" => "'"), "\"")
inner_row_line(r) = string(r.family, ",", r.direction, ",", r.delta, ",", r.start, ",", r.eval_idx, ",", r.t_s, ",",
    r.gp, ",", r.Delta_star, ",", r.feasible, ",", r.verified_success, ",", r.classification_proxy, ",", r.final_incumbent)

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
                n_approximate_proxy = get(prior, "n_approximate_proxy", missing),
                n_dashed_or_infeasible_proxy = get(prior, "n_dashed_or_infeasible_proxy", missing),
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
    lp("[", label, "] family=", FAMILY, " direction=", DIRECTION, " delta=", delta, " start=", start_idx,
       " checksum_w=", st["checksum_w_hash"], " ckdir=", ckdir)
    lp("outer_initial_start_id = ", start_idx)
    lp("outer_initial_coordinate_checksum = ", st["checksum_w_hash"])
    lp("outer_initialized_from_prior_solution = false")
    lp("outer_initialized_from_prior_delta_solution = false")
    lp("outer_initialized_from_prior_direction_solution = false")
    lp("outer_initialized_from_other_start_solution = false")
    lp("mutable_workspace_reuse = allowed (shared immutable ctx/pcx only; no cross-cell state observed)")
    lp("inner_dual_warm_start_reuse = allowed_not_used (resume_from=nothing, fresh dual bank per cell)")
    lp("exact_cache_reuse = allowed_when_key_matches_not_used (resume_from=nothing, fresh SafeExactCache per cell)")

    w0 = FAMILY == "cm_meanzc"  ? vcat(w_a, log.(nu_meanzc)) :
         FAMILY == "origin_zc" ? vcat(w_a, log.(nu_originzc)) : copy(w_a)

    t0 = time()
    errored = false; errmsg = ""
    result = nothing
    try
        if FAMILY == "origin_zc"
            fn = FIND_SMALLEST ? run_originzc_upper_checkpointed : run_originzc_lower_checkpointed
            result = fn(w0; W = W, delta = delta, draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED,
                distribution_restriction = :origin_specific_moments_zero_covariance,
                K_mean = ORIGINZC_K, K_pair = ORIGINZC_K,
                exclude_diagonal_gravity = SCI.exclude_diagonal_gravity,
                gravity_exclude_cells = SCI.gravity_exclude_cells, σHat = SCI.sigma,
                destination_sample = SCI.destination_sample,
                ckpt_dir = ckdir, run_id = label, label = label,
                checkpoint_interval_s = 3600.0, maxtime_real = MAXTIME, verbose = true)
        else
            fn = FIND_SMALLEST ? run_cm_upper_checkpointed : run_cm_lower_checkpointed
            extra = FAMILY == "common_frechet" ? (marginal_restriction = :common_frechet, cm_hessian_backend = :structured, cm_gradient_backend = :cplus) :
                    FAMILY == "cm_meanzc"      ? (cm_extension = :cm_plus_moments, meanzc_K_mean = MEANZC_K, meanzc_K_pair = MEANZC_K) :
                    NamedTuple()
            result = fn(w0; W = W, delta = delta, draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED,
                L = CM_L, contrasts = :orthonormal, probs = PROBS_L50,
                exclude_diagonal_gravity = SCI.exclude_diagonal_gravity,
                gravity_exclude_cells = SCI.gravity_exclude_cells, σHat = SCI.sigma,
                destination_sample = SCI.destination_sample,
                ckpt_dir = ckdir, run_id = label, label = label,
                checkpoint_interval_s = 3600.0, maxtime_real = MAXTIME, verbose = true, extra...)
        end
    catch e
        errored = true
        errmsg = sprint(showerror, e)
        lp("[", label, "] *** EXCEPTION *** ", typeof(e), ": ", errmsg[1:min(end, 800)])
    end
    wall = time() - t0
    cfg = Dict{String,Any}("family" => FAMILY, "direction" => DIRECTION, "delta" => delta, "start_id" => start_idx,
        "W" => W, "draw_design" => string(DRAW_DESIGN), "draw_seed" => DRAW_SEED, "cm_L" => CM_L,
        "scientific_manifest_path" => SCI_MANIFEST_PATH, "sigma" => SCI.sigma,
        "destination_sample" => string(SCI.destination_sample),
        "exclude_diagonal_gravity" => SCI.exclude_diagonal_gravity,
        "gravity_exclude_cells" => SCI.gravity_exclude_cells,
        "maxtime_real" => MAXTIME, "checksum_w_hash" => st["checksum_w_hash"], "run_id" => label,
        "attempt" => prior_attempts + 1, "outer_initialized_from_prior_solution" => false,
        "outer_initialized_from_prior_delta_solution" => false, "outer_initialized_from_prior_direction_solution" => false,
        "outer_initialized_from_other_start_solution" => false)

    if errored
        orow = (family = FAMILY, direction = DIRECTION, delta = delta, start = start_idx,
            checksum_w = st["checksum_w_hash"], elapsed_s = round(wall, digits = 2), outer_status = "EXCEPTION",
            outer_evals = missing, n_inner_solves = missing, n_verified = missing, n_approximate_proxy = missing,
            n_dashed_or_infeasible_proxy = missing, best_verified_gp = missing, best_verified_Delta = missing,
            budget_residual = missing, error = errmsg[1:min(end, 300)])
        push!(rows_outer, orow)
        write_cell_status!(ckdir, orow, NamedTuple[], nothing, cfg; failed = true)
        append_csv_row!(OUTER_CSV, OUTER_HEADER, outer_row_line(orow))
        continue
    end

    trace = result.trace
    n_verified = count(r -> r.verified, trace)
    n_approx_proxy = count(r -> !r.verified && isfinite(r.Delta), trace)
    n_dashed_proxy = count(r -> !isfinite(r.Delta), trace)
    best = result.best
    orow = (family = FAMILY, direction = DIRECTION, delta = delta, start = start_idx,
        checksum_w = st["checksum_w_hash"], elapsed_s = round(wall, digits = 2),
        outer_status = string(result.knitro_status), outer_evals = result.n_eval,
        n_inner_solves = result.n_eval, n_verified = n_verified, n_approximate_proxy = n_approx_proxy,
        n_dashed_or_infeasible_proxy = n_dashed_proxy,
        best_verified_gp = best === nothing ? missing : best.gp,
        best_verified_Delta = best === nothing ? missing : best.Delta,
        budget_residual = best === nothing ? missing : (delta - best.Delta),
        error = "")
    push!(rows_outer, orow)

    cell_inner_rows = NamedTuple[]
    for r in trace
        irow = (family = FAMILY, direction = DIRECTION, delta = delta, start = start_idx,
            eval_idx = r.idx, t_s = round(r.t, digits = 3), gp = r.gp, Delta_star = r.Delta,
            feasible = r.feasible, verified_success = r.verified,
            classification_proxy = r.verified ? "verified" : (isfinite(r.Delta) ? "approximate_proxy" : "dashed_or_infeasible_proxy"),
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
       "  best_Delta=", best === nothing ? "none" : @sprintf("%.6e", best.Delta))
end

lp(">> outer/inner CSVs already written incrementally, per cell, throughout this run: ", OUTER_CSV, " / ", INNER_CSV)
lp(">> (no end-of-run rewrite here -- a resumed/restarted process only holds resumed cells' outer rows in memory,")
lp(">>  not their inner rows, so rewriting from the in-memory arrays at this point would truncate the inner CSV)")
print_no_h_counters(FAMILY)
lp("="^100)
lp("STATE-REUSE ACCOUNTING (", FAMILY, "/", DIRECTION, ", ", length(rows_outer), " cells):")
lp("  OUTER_CONTINUATION_CALLS = 0   (every cell called with resume_from=nothing, a literal constant at every call site -- no code path in this runner can set it otherwise)")
lp("  INNER_WARM_START_REUSES = 0    (use_dual_bank left at its default; no dual bank shared across cells)")
lp("  CACHE_HITS_ACROSS_SOLVES = 0   (no exact_cache_override passed; each cell gets a fresh SafeExactCache internally)")
lp("CAMPAIGN CM-FAMILY RUNNER COMPLETE -- family=", FAMILY, " direction=", DIRECTION,
   " cells=", length(rows_outer), " errors=", count(r -> r.outer_status == "EXCEPTION", rows_outer))
lp("="^100)
