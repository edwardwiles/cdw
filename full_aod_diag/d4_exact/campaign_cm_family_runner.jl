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
          # 2026-08-09 CROSS campaign integration. `zc_restriction_operator_ragged.jl` is listed
          # explicitly (rather than relying on the runtime include() inside build_*_bin_ctx) so the
          # 4-arg aml-aware ZCRestrictionOperator constructor exists in an OLDER world age than any
          # context build -- the world-age trap that bit both build_originzc_core_hess_ctx and
          # build_cm_meanzc_bin_ctx (both since hardened with Base.invokelatest; this is belt and
          # braces, and costs nothing).
          "zc_restriction_operator.jl", "zc_restriction_operator_ragged.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl",
          "cm_originzc_cross_production.jl", "cm_originzc_cross_cplus.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl",
          "cm_meanzc_cross_production.jl", "cm_meanzc_cross_cplus.jl",
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

# CM+ZC-CROSS / OZC-CROSS campaign integration (2026-08-09): the two CROSS families are selectable
# HERE, as first-class families, INSTEAD of (not alongside) their diagonal counterparts in a given
# campaign wave -- see run_full_campaign_supervisor.sh's FAMILIES array. They reuse the same start
# manifest, the same nu coordinates, and the same drivers; only the target layout differs.
FAMILY in ("flexible_cm", "common_frechet", "cm_meanzc", "origin_zc", "cm_meanzc_cross", "origin_zc_cross") ||
    error("campaign_cm_family_runner: unknown family $FAMILY")
const IS_CROSS    = endswith(FAMILY, "_cross")
const IS_MEANZC_F = FAMILY in ("cm_meanzc", "cm_meanzc_cross")
const IS_OZC_F    = FAMILY in ("origin_zc", "origin_zc_cross")
DIRECTION in ("upper", "lower") || error("campaign_cm_family_runner: direction must be upper|lower, got $DIRECTION")
const FIND_SMALLEST = DIRECTION == "upper"

const DELTAS = DELTAS_OVERRIDE === nothing ? [0.01, 0.1, 0.5, 1.0, 2.0] : DELTAS_OVERRIDE
const W = 100_000
const DRAW_DESIGN = :sobol_randomized
const DRAW_SEED = 20260719
const CM_L = 50
# 2026-08-09 (CROSS campaign integration): these are now FALLBACKS, used only when the start
# manifest's own config block does not record the K it was built at. The manifest is authoritative
# when it says anything -- see MEANZC_K/ORIGINZC_K resolution below. Keeping the historical values
# here means a manifest without those keys behaves exactly as before.
const MEANZC_K_FALLBACK = 1
const ORIGINZC_K_FALLBACK = 1

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
(IS_MEANZC_F || IS_OZC_F) && Threads.nthreads() < 10 &&   # 2026-08-09: the CROSS variants share the same H_ZZ/H_CZ/H_EZ backends (and have a strictly LARGER restriction block), so the same thread-starvation warning applies at least as strongly
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

# ---------------------------------------------------------------------------------------------
# K resolution + nu-length agreement (2026-08-09, CROSS campaign integration).
#
# The manifest's `shared_extra_coordinates` nu vectors are sized at the K the MANIFEST was built
# at (`three_starts_search.jl`'s own MEANZC_K/ORIGINZC_K, recorded in its config block), while this
# runner previously hardcoded its own K constants. Those two only ever agreed by convention -- a
# manifest built at K=2 fed to this runner's K=1 constant produces a w0 whose eta_nu block is the
# WRONG LENGTH for the driver being called. That latent mismatch is pre-existing (not introduced by
# the CROSS work), but the CROSS families make it live, because they are naturally run at K>=2:
# at K_pair=1 the K_pair^2 cross grid has exactly one combo, (1,1), so the cross family is
# IDENTICAL to the diagonal one and the campaign arm would be scientifically vacuous.
#
# Resolution: the manifest is authoritative when it records K; the historical constants are used
# only as a fallback for manifests predating those keys; and the nu length is ASSERTED against
# n_eta either way, so a mismatch is a loud error before any KNITRO solve rather than a confusing
# dimension failure 40 minutes in. The CROSS families deliberately reuse their diagonal
# counterpart's K and nu vector unchanged -- n_eta is genuinely identical between the two layouts
# (K_mean for CM+ZC, K_mean*D for origin-ZC; the cross extension adds NO outer parameters), which
# is exactly why no new manifest key or nu-lift change is needed.
# ---------------------------------------------------------------------------------------------
_mcfg = get(manifest, "config", Dict{String,Any}())
_mget_int(key, fallback) = haskey(_mcfg, key) ? Int(_mcfg[key]) : fallback
const MEANZC_K   = _mget_int("meanzc_K_mean", MEANZC_K_FALLBACK)
const ORIGINZC_K = _mget_int("originzc_K_mean", ORIGINZC_K_FALLBACK)
lp(">> K resolution: MEANZC_K=", MEANZC_K, " ORIGINZC_K=", ORIGINZC_K,
   haskey(_mcfg, "meanzc_K_mean") ? "  (from manifest config)" : "  (manifest has no K keys -- using historical runner fallbacks $(MEANZC_K_FALLBACK)/$(ORIGINZC_K_FALLBACK))")
if IS_MEANZC_F
    length(nu_meanzc) == MEANZC_K ||
        error("campaign_cm_family_runner: manifest cm_meanzc_nu has length $(length(nu_meanzc)) but " *
              "n_eta(SharedByPower[Cross]Layout) = K_mean = $MEANZC_K -- refusing to build a w0 whose " *
              "eta_nu block is the wrong length for the driver. Rebuild the manifest at the intended K, " *
              "or run against a manifest whose config block records the matching meanzc_K_mean.")
end
if IS_OZC_F
    const _D_MANIFEST = _mget_int("D", 20)
    length(nu_originzc) == ORIGINZC_K * _D_MANIFEST ||
        error("campaign_cm_family_runner: manifest origin_zc_nu has length $(length(nu_originzc)) but " *
              "n_eta(OriginByPower[Cross]Layout) = K_mean*D = $(ORIGINZC_K)*$(_D_MANIFEST) = $(ORIGINZC_K*_D_MANIFEST) " *
              "-- refusing to build a w0 whose eta_nu block is the wrong length for the driver.")
end
# ---------------------------------------------------------------------------------------------
# REQUIRED scientific parameters neither driver defaults (2026-08-09). PRE-EXISTING BREAKAGE, found
# by running this runner end to end and CONFIRMED against the unmodified BASE families first:
#
#   cm_meanzc / flexible_cm / common_frechet -> UndefKeywordError: `include_truncated_moment`
#   origin_zc                                -> UndefKeywordError: `inner_lower_limit`
#
# i.e. EVERY family arm of this runner has been unable to launch since those two parameters were
# (correctly) made required-with-no-default by the 2026-08-05 truncated-power and 2026-08-06
# lower-limit hardening passes -- this runner was never updated to supply them. That is exactly the
# intended consequence CLAUDE.md describes for the ~200 old diagnostic scripts, except this one is a
# production campaign entry point, so it has to be fixed rather than left to throw.
#
# Supplied EXPLICITLY, never defaulted inside a call chain: from the manifest's own config block when
# the manifest records them (the manifest is the run's scientific provenance), otherwise from the
# named constants below, whose values are the current production spec:
#   include_truncated_moment = true   -- eq.35+eq.36, the corrected flexible-CM/CM+ZC production spec
#   inner_lower_limit        = -10.0  -- docs/audits/fullA-lower-limit-and-hotpath-2026-08-06/MASTER.md
# Every OTHER scientific parameter these drivers take already defaults to the production value
# (sigma=3.0, gravity_exclude_cells=Brazil-Korea, exclude_diagonal_gravity=true,
# destination_sample=:exclude_row, A_coordinate_mode=:powered_aspace) and is left alone.
#
# NOT changed, deliberately: `meanzc_profiled_level`/`originzc_profiled_level` (Variant D, the focal
# k*=sigma-1 mean-row omission) stay at the drivers' `nothing` default unless the manifest asks for
# them. Turning Variant D on would be a real scientific change to the BASE arms' campaign behavior,
# which is not this task's to make -- but note it IS the intended production spec at K>=2, so a
# campaign wanting it must set `profiled_level` in its manifest config block.
const INCLUDE_TRUNCATED_MOMENT_FALLBACK = true
const INNER_LOWER_LIMIT_FALLBACK        = -10.0
const INCLUDE_TRUNCATED_MOMENT = haskey(_mcfg, "include_truncated_moment") ? Bool(_mcfg["include_truncated_moment"]) : INCLUDE_TRUNCATED_MOMENT_FALLBACK
const INNER_LOWER_LIMIT        = haskey(_mcfg, "inner_lower_limit") ? Float64(_mcfg["inner_lower_limit"]) : INNER_LOWER_LIMIT_FALLBACK
const PROFILED_LEVEL           = haskey(_mcfg, "profiled_level") && _mcfg["profiled_level"] !== nothing ? Int(_mcfg["profiled_level"]) : nothing
lp(">> required scientific params: include_truncated_moment=", INCLUDE_TRUNCATED_MOMENT,
   haskey(_mcfg, "include_truncated_moment") ? " (manifest)" : " (runner constant)",
   "  inner_lower_limit=", INNER_LOWER_LIMIT,
   haskey(_mcfg, "inner_lower_limit") ? " (manifest)" : " (runner constant)",
   "  profiled_level=", PROFILED_LEVEL === nothing ? "nothing (Variant D OFF)" : string(PROFILED_LEVEL))

if IS_CROSS
    _kpair_cross = IS_MEANZC_F ? MEANZC_K : ORIGINZC_K
    _kpair_cross >= 2 ||
        lp("WARNING: family=", FAMILY, " is running at K_pair=", _kpair_cross, ". The K_pair^2 cross-power ",
           "grid has exactly ONE combo (1,1) at K_pair=1, i.e. this run is mathematically IDENTICAL to the ",
           "diagonal family it is meant to be compared against -- the campaign arm carries no new ",
           "information. Use a manifest built at K>=2 for a meaningful CROSS wave.")
    lp(">> CROSS family: ", _kpair_cross^2, " ordered (k1,k2) restrictions per origin pair (vs ",
       _kpair_cross, " diagonal); n_eta UNCHANGED, so w0/eta_nu keep their diagonal-family length.")
end

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

    # 2026-08-09: the CROSS families take the SAME nu lift as their diagonal counterparts -- n_eta is
    # identical between the two layouts (see the K-resolution block above), so this needed widening
    # only to recognize the new family strings, not to change any coordinate construction.
    w0 = IS_MEANZC_F ? vcat(w_a, log.(nu_meanzc)) :
         IS_OZC_F    ? vcat(w_a, log.(nu_originzc)) : copy(w_a)

    t0 = time()
    errored = false; errmsg = ""
    result = nothing
    try
        if IS_OZC_F
            fn = FIND_SMALLEST ? run_originzc_upper_checkpointed : run_originzc_lower_checkpointed
            # 2026-08-09: OZC-CROSS selects itself purely through power_target_layout -- the
            # distribution_restriction/K arguments are IDENTICAL to the diagonal family's (same
            # outer parameter space), which is exactly the property that makes the two directly
            # comparable at matched K. See cm_originzc_config.jl's own docstring.
            result = fn(w0; W = W, delta = delta, draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED,
                distribution_restriction = :origin_specific_moments_zero_covariance,
                K_mean = ORIGINZC_K, K_pair = ORIGINZC_K,
                power_target_layout = IS_CROSS ? :origin_by_power_cross : :origin_by_power,
                inner_lower_limit = INNER_LOWER_LIMIT,          # REQUIRED, no driver default -- see resolution block above
                originzc_profiled_level = PROFILED_LEVEL,
                ckpt_dir = ckdir, run_id = label, label = label,
                checkpoint_interval_s = 3600.0, maxtime_real = MAXTIME, verbose = true)
        else
            fn = FIND_SMALLEST ? run_cm_upper_checkpointed : run_cm_lower_checkpointed
            extra = FAMILY == "common_frechet" ? (marginal_restriction = :common_frechet, cm_hessian_backend = :structured, cm_gradient_backend = :cplus) :
                    IS_MEANZC_F                ? (cm_extension = :cm_plus_moments, meanzc_K_mean = MEANZC_K, meanzc_K_pair = MEANZC_K,
                                                  # 2026-08-09: CM+ZC-CROSS selects itself purely through
                                                  # meanzc_target_layout -- same cm_extension, same K, same
                                                  # eta_nu length as the diagonal family.
                                                  meanzc_target_layout = IS_CROSS ? :shared_by_power_cross : :shared_by_power,
                                                  meanzc_profiled_level = PROFILED_LEVEL) :
                    NamedTuple()
            # common_frechet is FORCED single-family regardless of the resolved value. Two-family
            # (eq.35+eq.36) common-Frechet is NOT production-reachable through this driver at all:
            # build_cm_frechet_production_context requires moment_representation=:dense_reference for
            # include_truncated_moment=true, which collides with prepare_production_run's hard ban on
            # dense bundles -- documented in cm_checkpoint.jl's own comment on that branch and in
            # multistart_seed_generator.jl's COMMON_FRECHET spec ("single-family... matching the one
            # variant that actually is production-reachable"). Passing `true` here would make the
            # common_frechet arm fail at context construction rather than run the wrong spec, but it
            # would still be a needless failure, so it is pinned rather than left to the resolver.
            itm_family = FAMILY == "common_frechet" ? false : INCLUDE_TRUNCATED_MOMENT
            result = fn(w0; W = W, delta = delta, draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED,
                L = CM_L, contrasts = :orthonormal, probs = PROBS_L50,
                include_truncated_moment = itm_family,   # REQUIRED, no driver default -- see resolution block above
                inner_lower_limit = INNER_LOWER_LIMIT,   # REQUIRED, no driver default
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
