# paper_upper_v1 orchestrator: runs ONE (family, start) chain through its full
# delta = 0.1 -> 0.5 -> 1 -> 2 sequence, one process per chain (matches the protocol's
# concurrency model: 2 starts x 5 families = 10 simultaneous processes per Phase-I wave).
#
# This is the "cell runner" for Phase I discovery. It does NOT decide wave sequencing or CPU
# affinity -- that is scripts/launch_wave.sh's job. This script's only inputs are: which family,
# which start, and where the campaign root/seeds live; it runs to completion (or is killed) and
# writes its own resume-safe, immutable-once-written per-delta results underneath
# <campaign_root>/phase1_discovery/<FAMILY_ID>/<START_ID>/delta_<d>/.
#
# Usage:
#   julia --project=. -t 10 paper_upper_v1_orchestrator/family_start_chain.jl \
#       <protocol_toml> <FAMILY_ID> <START_ID> <campaign_root>
#
# Idempotent: if delta_<d>/resume_bundle.jls already exists AND its recorded protocol_sha/
# manifest fields match the current protocol, that delta is SKIPPED (not re-run) -- matches
# "never rerun a completed stage" (protocol section 21).

using TOML, Serialization, Dates, Printf, SHA

const D4E = joinpath(@__DIR__, "..", "full_aod_diag", "d4_exact")
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl", "lfix_cm_aware.jl",
          "cm_hessian_architectures.jl", "cm_production_bundle.jl", "compressed_factual_buffer_reuse.jl", "cm_screen_bridge.jl", "gradient_workspace.jl",
          "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl",
          "cm_outer_driver.jl", "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl",
          "cm_meanzc_cplus.jl", "incumbent_logic.jl", "cm_checkpoint.jl",
          "cm_originzc_target_layout.jl", "zc_restriction_operator_ragged.jl", "cm_aspace_coordinate.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_config.jl",
          "cm_originzc_checkpoint.jl", "direction_bounds.jl", "cm_frechet_hessian.jl", "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl", "country_resolve.jl",
          "cross_delta_cache.jl", "compressed_moments.jl", "canonical_price_precompute_workspace.jl",
          "hard_score_b_cache.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "lfix_buffer_reuse.jl", "lfix_base_workspace_pooled.jl", "lfix_kbplus_workspace.jl",
          "bandwidth_cache_policy.jl", "fast_range_screen.jl", "dual_bank.jl", "negative_cache.jl",
          "dual_bank_ab_harness.jl", "reusable_context.jl", "organic_failure_capture.jl",
          "knitro_status.jl", "knitro_version_check.jl",
          # Unrestricted-family support: c10_d20_production_driver.jl defines ScreenCounters/
          # screened_eval/run_profile_checkpointed/etc. directly in its OWN body (not via a
          # sub-include), and flexible_theta_aspace_production.jl's own function signatures
          # reference ScreenCounters as a TYPE ANNOTATION (resolved at definition/parse time, not
          # call time) -- it MUST come before flexible_theta.jl/flexible_theta_aspace_production.jl.
          "c10_d20_production_driver.jl",
          "flexible_theta.jl", "flexible_theta_aspace_production.jl", "outer_coordinate_layout.jl",
          "c10_d20_production_driver_unified.jl",
          "multistart_seed_generator.jl",
          # CM + pairwise-quantile family (#7, 2026-08-12). The pairwise-quantile BASE files are
          # dependencies of it -- this family's restriction rows ARE the standalone family's rows at
          # a mu common across origins, so it reuses that family's operator, thread scratch, Hessian
          # tables and economic cross block wholesale (see cm_pairwise_quantile_hessian.jl's header).
          #
          # DELIBERATELY NOT INCLUDED: `pairwise_quantile_checkpoint.jl`. That file defines the
          # STANDALONE family's own campaign driver, which `call_driver` already has an arm for but
          # which is NOT currently reachable from this script (the include was never added; standalone
          # PQ campaign runs go through run_pq_multistart_seed_chain.jl instead). Adding it here as a
          # side effect of wiring family #7 would silently change which driver that arm resolves to,
          # mid-campaign, on a family another session is actively working on. Flagged, not fixed here.
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_production.jl",
          "cm_pairwise_quantile_config.jl", "cm_pairwise_quantile_moments.jl",
          "cm_pairwise_quantile_hessian.jl", "cm_pairwise_quantile_hessian_assembly.jl",
          "cm_pairwise_quantile_lookup_kernels.jl", "cm_pairwise_quantile_production.jl",
          "cm_pairwise_quantile_verification.jl", "cm_pairwise_quantile_outer_production.jl", "cm_pairwise_quantile_cplus.jl",
          "cm_pairwise_quantile_checkpoint.jl"]
    include(joinpath(D4E, f))
end

lp(xs...) = (println(xs...); flush(stdout))

# ============================================================================
# 1. CLI / manifest loading
# ============================================================================

const PROTOCOL_TOML = ARGS[1]
const FAMILY_ID = ARGS[2]
const START_ID = ARGS[3]           # e.g. "S0", "S1", ...
const CAMPAIGN_ROOT = ARGS[4]

const MANIFEST = TOML.parsefile(PROTOCOL_TOML)
const SCI = MANIFEST["scientific"]
const FAM = MANIFEST["families"][FAMILY_ID]
const BUD = MANIFEST["budgets"]
const ALG = MANIFEST["algorithms"]

"TOML kwarg dict -> NamedTuple with BOTH keys and string values symbolized (e.g. outer_direct_hessopt=\"sr1\" -> outer_direct_hessopt=:sr1)."
algo_kwargs_from(d) = NamedTuple(Symbol(k) => (v isa String ? Symbol(v) : v) for (k, v) in d)

lp("="^100)
lp("family_start_chain: protocol=", MANIFEST["protocol"]["name"], " family=", FAMILY_ID, " start=", START_ID)
lp("="^100)

sym(s) = Symbol(s)

# ============================================================================
# 2. ctx (shared across all 4 deltas -- rebuilt per delta via set_context_delta! where available,
#    else fresh d20_real_setup_design per delta since delta only changes ctx.obj.δ/ctx.δ, both
#    cheap fields; correctness matters far more than the small rebuild cost here).
# ============================================================================

function build_ctx_at_delta(delta::Float64)
    grav = default_gravity_exclude_cells_brazil_korea()
    ctx_raw = d20_real_setup_design(W = SCI["W"], δ = delta, find_smallest = SCI["find_smallest"],
        draw_design = sym(SCI["draw_design"]), draw_seed = SCI["draw_seed"],
        destination_sample = sym(SCI["destination_sample"]), exclude_diagonal_gravity = SCI["exclude_diagonal_gravity"],
        gravity_exclude_cells = grav, σHat = SCI["sigma"], inner_lower_limit = SCI["inner_lower_limit"])
    return attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)
end

# ============================================================================
# 3. Family driver dispatch -- ONE call site per family that builds the exact kwargs from the
#    manifest's [families.<ID>.kwargs] block, so the manifest (not this file) is the single
#    source of truth for family-defining kwargs.
# ============================================================================

# `const` is illegal inside a function body, so this lives at module level next to its only user.
const NO_PROBS_DRIVERS = ("run_pairwise_quantile_upper_checkpointed",
                          "run_cm_pairwise_quantile_upper_checkpointed")

function fam_kwargs()
    haskey(FAM, "kwargs") || return NamedTuple()
    kw = NamedTuple(sym(k) => (v isa String ? sym(v) : v) for (k, v) in FAM["kwargs"] if k != "notes")
    # CM-family drivers require `probs` as an explicit, no-default kwarg (exact cutpoints, not
    # re-derived from L alone) -- auto-resolve via the SAME production convention
    # (resolve_cm_probs, multistart_seed_generator.jl) any time the manifest specifies `L` without
    # its own explicit `probs` override.
    #
    # EXCEPT for the two families whose `L` is a number of quantile BINS and has nothing to do with
    # a CM contrast grid. Neither driver has a `probs` kwarg at all, so injecting one here would
    # make every call to them fail with an unsupported-keyword MethodError. The guard is on the
    # DRIVER, not on the presence of `L`, because `L` is exactly what these families share with the
    # CM ones.
    #
    #   run_pairwise_quantile_upper_checkpointed     -- `L` = quantile bins per origin; no CM grid.
    #   run_cm_pairwise_quantile_upper_checkpointed  -- `L` = PQ bins on the SHARED reference
    #       marginal, while the CM grid arrives separately as `cm_grid_size`. This family DOES have
    #       a CM grid, but builds it internally and passes `probs` to CM itself, because it needs
    #       the explicit `k/G` grid rather than CM's default (whose spacing is 0.0195918 at G=50, so
    #       NO L divides it -- memory `cm-default-grid-is-not-k-over-g`). Injecting
    #       `resolve_cm_probs(kw.L)` here would therefore be wrong twice over: wrong grid, derived
    #       from the wrong L.
    #
    # BUCKETS -> LEVELS, the one translation in this file (2026-08-12). The manifest's `L` is the
    # CM grid size in equal-mass BUCKETS -- `L = 50` means 50 buckets of mass exactly 1/50 -- and
    # `resolve_cm_probs` returns that grid's `L-1` interior cutpoints (`p=1` excluded: structurally
    # zero moment column, singular KKT; see `cm_equal_mass_probs`). Every CM driver/builder below
    # this line counts LEVELS in its own `L` kwarg, so `L` is overwritten with `length(probs)`
    # here, once, and the invariant `L == length(probs)` then holds everywhere downstream (it is
    # independently enforced by `precalc_common_marginals_cdf` and by `run_cm_upper_checkpointed`'s
    # own top-of-function check). Consequence, disclosed rather than hidden: a checkpoint written
    # from a manifest `L = 50` records `cm_L = 49`. That is the level count, and it is correct.
    if haskey(kw, :L) && !haskey(kw, :probs) && !(FAM["driver"] in NO_PROBS_DRIVERS)
        probs = resolve_cm_probs(kw.L)
        lp("[fam_kwargs] CM grid: L=", kw.L, " equal-mass buckets (mass ", 1 / kw.L, " each) -> ",
           length(probs), " cutpoints k/", kw.L, " for k=1..", kw.L - 1,
           "; driver kwarg L is the LEVEL count ", length(probs), ", not the bucket count ", kw.L)
        kw = merge(kw, (L = length(probs), probs = probs))
    end
    return kw
end

"Calls the family's real checkpointed driver. `extra` overrides/adds kwargs (delta, ckpt_dir, label, objective_mode, gp_fixed, maxtime_real, resume_from, ...)."
function call_driver(w0::Vector{Float64}; extra::NamedTuple)
    grav = default_gravity_exclude_cells_brazil_korea()
    common = (draw_design = sym(SCI["draw_design"]), draw_seed = SCI["draw_seed"],
        inner_lower_limit = SCI["inner_lower_limit"], destination_sample = sym(SCI["destination_sample"]),
        exclude_diagonal_gravity = SCI["exclude_diagonal_gravity"], gravity_exclude_cells = grav,
        σHat = SCI["sigma"], find_smallest = SCI["find_smallest"], W = SCI["W"], z_halfwidth = SCI["z_halfwidth"])
    kw = merge(common, fam_kwargs(), extra)
    driver = FAM["driver"]
    if driver == "run_cm_upper_checkpointed"
        return run_cm_upper_checkpointed(w0; kw...)
    elseif driver == "run_originzc_upper_checkpointed"
        return run_originzc_upper_checkpointed(w0; kw...)
    elseif driver == "run_pairwise_quantile_upper_checkpointed"
        # Pairwise-quantile-independence family (2026-08-10, version B: fixed cutoffs + free bin
        # masses). Fits the uniform convention with no wrapper: same `common` scientific kwargs,
        # same objective_mode/gp_fixed Stage B mechanism, same NamedTuple return shape
        # (.knitro_status/.n_eval/.n_grad/.best) run_stage reads. Its own three required kwargs --
        # `L` (quantile bins), `cutoff_source` (:frechet_theoretical|:empirical_quantile) and
        # `min_bin_count` (bin non-degeneracy floor) -- arrive generically through fam_kwargs() from
        # the protocol's [families.<ID>.kwargs] sub-table, so no per-family kwarg-NAME logic is
        # needed here. (`cutoff_source` must reach the driver as a Symbol; the protocol reader's own
        # kwarg coercion handles that the same way it does for every other Symbol-valued kwarg.)
        return run_pairwise_quantile_upper_checkpointed(w0; kw...)
    elseif driver == "run_cm_pairwise_quantile_upper_checkpointed"
        # CM + pairwise-quantile family (#7, 2026-08-12). Fits the uniform convention with no
        # wrapper: same `common` scientific kwargs, same objective_mode/gp_fixed Stage B mechanism,
        # same NamedTuple return shape (.knitro_status/.n_eval/.n_grad/.best) run_stage reads.
        #
        # Its own required kwargs arrive generically through fam_kwargs() from the protocol's
        # [families.<ID>.kwargs] sub-table -- `L` (PQ bins on the shared reference marginal),
        # `cm_grid_size` (G, which L must divide), `cm_moment_families` (1 = eq.35, 2 = +eq.36),
        # `contrasts`, `min_bin_count`, `mass_start`, and `inner_opt`.
        #
        # `inner_opt` is REQUIRED and must be the exact-Hessian file ("ek_inner_cmpq.opt"). That is
        # not a tuning preference: measured at production n_x=412, the FG-only path hits -400 after
        # 16,734 evaluations while the exact-Hessian path reaches nStatus=0 in twelve. A protocol
        # arm that omits it fails with UndefKeywordError; one that names a non-exact file fails
        # inside the inner registration. Both are deliberate hard errors, not silent downgrades.
        return run_cm_pairwise_quantile_upper_checkpointed(w0; kw...)
    else
        error("call_driver: unknown driver '$driver' for family $FAMILY_ID -- Unrestricted goes through call_unrestricted_driver, not call_driver")
    end
end

# ---- Unrestricted: distinct call convention (positional label/find_smallest/w_start, its own
# fixed-theta OuterCoordinateLayout, and a SEPARATE unconstrained-objective=Delta driver
# (run_profile_checkpointed) rather than an objective_mode kwarg -- see
# /bbkinghome/edav/repo_scratch/fullA_unrestricted_sigma2_sigma4_delta1_upper_2026-08-08/
# stage_driver.jl for the real, already-validated campaign template this mirrors. ----

const UNRESTRICTED_LAYOUT = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :powered_aspace, gp_coordinate_mode = :raw)

"Returns a NamedTuple shaped like call_driver's own CM-family return (.best/.knitro_status/.n_eval/.n_grad) so run_stage/classify_stage work unchanged for both families."
function call_unrestricted_driver(w0::Vector{Float64}, delta::Float64; label::String, ckpt_dir::String,
                                   maxtime_real::Float64, checkpoint_interval_s::Float64, algo_kwargs::NamedTuple,
                                   gp_fixed::Union{Nothing,Float64} = nothing, verbose::Bool = true)
    grav = default_gravity_exclude_cells_brazil_korea()
    # Stage A/C (gp_fixed===nothing, objective_mode=:min_gp) and Stage B/R0 (gp_fixed set,
    # objective_mode=:min_delta_fixed_gp) now go through the SAME run_polish_checkpointed_unified
    # driver, SAME UNRESTRICTED_LAYOUT, SAME context-build machinery throughout -- 2026-08-09 fix.
    # Previously Stage B routed through a SEPARATE, older function (run_profile_checkpointed) with
    # its own hardcoded :legacy_z coordinate machinery and no `layout` kwarg at all, requiring an
    # error-prone coordinate conversion between the two that -- despite being independently proven
    # mathematically exact against a real production-scale verified-feasible checkpoint -- still
    # did not resolve Stage B's systematic infeasibility (see the paper-upper-v1-unrestricted-
    # stage-b-layout-bug-2026-08-09 memory for that full, ultimately-inconclusive investigation).
    # Using the identical driver/layout for both stages removes the cross-function boundary
    # entirely rather than trying to bridge it correctly -- exactly mirroring how the OTHER four
    # families already do Stage B (run_cm_upper_checkpointed/run_originzc_upper_checkpointed with
    # objective_mode=:min_delta_fixed_gp, added 2026-08-08) via one shared driver, not two.
    r = run_polish_checkpointed_unified(label, SCI["find_smallest"], w0;
        layout = UNRESTRICTED_LAYOUT, theta_lo = NaN, theta_hi = NaN,
        maxtime_real = maxtime_real, W_in = SCI["W"], delta_in = delta,
        draw_seed_in = SCI["draw_seed"], draw_design_in = sym(SCI["draw_design"]),
        ckpt_dir = ckpt_dir, checkpoint_interval_s = checkpoint_interval_s,
        destination_sample = sym(SCI["destination_sample"]), exclude_diagonal_gravity = SCI["exclude_diagonal_gravity"],
        gravity_exclude_cells = grav, σHat = SCI["sigma"], inner_lower_limit = SCI["inner_lower_limit"],
        gp_fixed = gp_fixed, objective_mode = gp_fixed === nothing ? :min_gp : :min_delta_fixed_gp,
        algo_kwargs...)
    best = r.best_feasible === nothing ? nothing : (gp = r.best_feasible.gp, w = r.best_feasible.w,
        Delta = r.best_feasible.Delta, n_eval = r.best_feasible.n_eval, t = r.best_feasible.t_elapsed)
    return (best = best, knitro_status = r.knitro_status, n_eval = r.n_eval, n_grad = r.n_grad_calls)
end

# ============================================================================
# 4. Verified Delta* at a point, family-agnostic, using the SAME evaluators the multistart
#    generator + checkpointed drivers use -- via one real KNITRO solve at maxtime effectively 0
#    is not meaningful (KNITRO always runs >=1 iterate), so this reuses objective_mode=:min_gp
#    with an absurdly small maxtime_real as a "just tell me Delta* at w0, verified" probe when
#    no cheaper direct value-only evaluator is wired for this family in this script. This keeps
#    every Delta* number origin-anchored to a real, verified production solve.
# ============================================================================

function verify_delta_star(w0::Vector{Float64}, delta_for_ctx::Float64; label::String, ckpt_dir::String)
    r = call_driver(copy(w0); extra = (delta = delta_for_ctx, ckpt_dir = ckpt_dir, label = label,
        maxtime_real = 5.0, checkpoint_interval_s = 3600.0, verbose = false, use_dual_bank = false))
    return r.best === nothing ? (Delta = Inf, verified = false) : (Delta = r.best.Delta, verified = true)
end

# ============================================================================
# 5. Stage runner: one call to the family's driver, one wall-clock budget, one status
#    classification. Returns a NamedTuple record matching the protocol's "record at every stage"
#    requirement (a genuinely populated SUBSET of the full field list in the task brief -- fields
#    that are not directly exposed by the current driver return value, e.g. per-callback screen-
#    rejection counters split by kind, are NOT fabricated here; `screen_summary`/`trace` from the
#    driver's own return value are attached as-is for anyone who needs the full detail).
# ============================================================================

function classify_stage(r, elapsed_s::Float64, budget_s::Float64, start_Delta::Union{Nothing,Float64})
    if r isa Exception
        return :SOLVER_FAILURE
    end
    b = r.best
    if b === nothing
        return elapsed_s >= budget_s - 1.0 ? :TIME_LIMIT_STALLED : :NO_FEASIBLE_IMPROVEMENT
    end
    converged = r.knitro_status in (0,)
    converged && return :CONVERGED
    if elapsed_s >= budget_s - 1.0
        improved = start_Delta === nothing || b.Delta < start_Delta - MANIFEST["convergence"]["material_delta_improvement_threshold"]
        return improved ? :TIME_LIMIT_PROGRESSING : :TIME_LIMIT_STALLED
    end
    return :CONVERGED   # terminated early for a reason other than the wall clock (e.g. KNITRO's own convergence tolerance)
end

function run_stage(kind::Symbol, w0::Vector{Float64}, delta::Float64, budget_minutes::Real;
                    ckpt_dir::String, label::String, algo_kwargs::NamedTuple, gp_fixed::Union{Nothing,Float64} = nothing,
                    start_Delta::Union{Nothing,Float64} = nothing)
    mkpath(ckpt_dir)
    budget_s = Float64(budget_minutes) * 60
    t0 = time()
    r = try
        if FAM["driver"] == "run_polish_checkpointed_unified"
            call_unrestricted_driver(copy(w0), delta; label = label, ckpt_dir = ckpt_dir,
                maxtime_real = budget_s, checkpoint_interval_s = min(90.0, budget_s / 2),
                algo_kwargs = algo_kwargs, gp_fixed = gp_fixed)
        else
            extra = merge((delta = delta, ckpt_dir = ckpt_dir, label = label, maxtime_real = budget_s,
                           checkpoint_interval_s = min(90.0, budget_s / 2), verbose = true), algo_kwargs)
            if gp_fixed !== nothing
                extra = merge(extra, (gp_fixed = gp_fixed, objective_mode = :min_delta_fixed_gp))
            end
            call_driver(copy(w0); extra = extra)
        end
    catch e
        lp("  [", label, "] STAGE ", kind, " THREW: ", sprint(showerror, e, catch_backtrace()))
        e
    end
    elapsed = time() - t0
    status = classify_stage(r, elapsed, budget_s, start_Delta)
    best = (!(r isa Exception) && r.best !== nothing) ? r.best : nothing
    lp("  [", label, "] STAGE ", kind, " done: status=", status, " wall=", round(elapsed, digits = 1),
       "s best_Delta=", best === nothing ? "none" : best.Delta)
    return (kind = kind, status = status, wall_s = elapsed, budget_s = budget_s, best = best,
            knitro_status = r isa Exception ? nothing : r.knitro_status,
            n_eval = r isa Exception ? nothing : r.n_eval, n_grad = r isa Exception ? nothing : r.n_grad,
            ckpt_dir = ckpt_dir, label = label, threw = r isa Exception ? sprint(showerror, r) : nothing)
end

# ============================================================================
# 6. One delta cell: R0 (if needed) -> A -> B -> C, per protocol §7-9 / Addendum C-E.
# ============================================================================

function run_delta_cell(w_start::Vector{Float64}, delta::Float64, delta_dir::String;
                         seed_Delta_at_delta::Union{Nothing,Float64} = nothing)
    mkpath(delta_dir)
    bundle_path = joinpath(delta_dir, "resume_bundle.jls")
    if isfile(bundle_path)
        b = deserialize(bundle_path)
        if get(b, :protocol_sha, nothing) == MANIFEST["source"]["protocol_sha"]
            lp("  delta=", delta, " already completed (resume_bundle exists, protocol_sha matches) -- SKIPPING")
            return b
        else
            error("run_delta_cell: existing resume_bundle at $bundle_path has a DIFFERENT protocol_sha " *
                  "than the current manifest -- refusing to silently overwrite (protocol section 21).")
        end
    end

    stages = NamedTuple[]
    gp0 = w_start[1]

    needs_r0 = delta == MANIFEST["scientific"]["deltas"][1] &&
        (seed_Delta_at_delta === nothing || seed_Delta_at_delta > 0.1 + 1e-9)
    w_for_A = w_start
    start_feasible = true

    if needs_r0
        st = run_stage(:R0, w_start, delta, BUD["discovery_infeasible_start"]["stage_R0_initial_restoration_minutes"];
            ckpt_dir = joinpath(delta_dir, "stage_R0"), label = "$(FAMILY_ID)_$(START_ID)_d$(delta)_R0",
            algo_kwargs = algo_kwargs_from(ALG["primary_kwargs"]), gp_fixed = gp0)
        push!(stages, st)
        if st.best !== nothing && st.best.Delta <= 0.1 + 1e-6
            w_for_A = st.best.w
        else
            start_feasible = false
        end
    end

    budgets = needs_r0 ? BUD["discovery_infeasible_start"] : BUD["discovery_feasible_start"]

    if !start_feasible
        result = (family = FAMILY_ID, start = START_ID, delta = delta, stages = stages,
                  final_status = :START_NOT_RESTORED_TO_DELTA, best = stages[end].best,
                  protocol_sha = MANIFEST["source"]["protocol_sha"], written_utc = string(now(UTC)))
        atomic_serialize(bundle_path, result)
        return result
    end

    stA = run_stage(:A, w_for_A, delta, budgets["stage_A_primary_push_minutes"];
        ckpt_dir = joinpath(delta_dir, "stage_A"), label = "$(FAMILY_ID)_$(START_ID)_d$(delta)_A",
        algo_kwargs = algo_kwargs_from(ALG["primary_kwargs"]))
    push!(stages, stA)
    w_after_A = stA.best === nothing ? w_for_A : stA.best.w
    gp_after_A = w_after_A[1]

    stB = run_stage(:B, w_after_A, delta, budgets["stage_B_fixed_gp_restoration_minutes"];
        ckpt_dir = joinpath(delta_dir, "stage_B"), label = "$(FAMILY_ID)_$(START_ID)_d$(delta)_B",
        algo_kwargs = algo_kwargs_from(ALG["primary_kwargs"]), gp_fixed = gp_after_A,
        start_Delta = stA.best === nothing ? nothing : stA.best.Delta)
    push!(stages, stB)
    # Stage B improves Delta only (gp fixed) -- if it found no better verified point, fall back to
    # Stage A's own incumbent (never regress).
    w_after_B = (stB.best !== nothing && (stA.best === nothing || stB.best.Delta <= stA.best.Delta)) ? stB.best.w : w_after_A

    stC = run_stage(:C, w_after_B, delta, budgets["stage_C_alternate_push_minutes"];
        ckpt_dir = joinpath(delta_dir, "stage_C"), label = "$(FAMILY_ID)_$(START_ID)_d$(delta)_C",
        algo_kwargs = algo_kwargs_from(ALG["alternate_kwargs"]),
        start_Delta = stB.best === nothing ? (stA.best === nothing ? nothing : stA.best.Delta) : stB.best.Delta)
    push!(stages, stC)

    # Cell-level incumbent selection (2026-08-09 fix): pick the FEASIBLE stage result with the
    # best gp (per is_better_polish/find_smallest -- the SAME criterion every stage's own cb_F!
    # already uses internally to pick ITS OWN best), not argmin(Delta). Delta is a feasibility
    # BUDGET, not a quality score: Stage A/C each already solve "best gp subject to Delta<=target"
    # (their own .Delta reflects how much budget got used getting there, nothing more), while
    # Stage B/R0 solve "best Delta at a FIXED gp" (a restoration probe -- gp is pinned, so its own
    # Delta can be made arbitrarily small without ever improving gp/kappa at all). argmin(Delta)
    # therefore almost always picked Stage B whenever it succeeded -- discarding any better gp
    # Stage C found, purely because B's Delta undercuts C's by construction, not because B found a
    # better answer. Confirmed live: UNRESTRICTED/S2/delta=0.1 had stA.gp=0.970358, stB.gp=0.970358
    # (pinned, Delta=0.054), stC.gp=0.969905 (strictly better, Delta=0.096, still feasible) -- the
    # old logic reported Stage B's worse-gp point. All per-stage results remain fully preserved in
    # `stages` regardless of which one wins here, so this is non-destructive.
    all_bests = filter(!isnothing, [s.best for s in stages])
    overall_best = nothing
    for cand in all_bests
        if is_better_polish(cand.gp, overall_best === nothing ? nothing : overall_best.gp, SCI["find_smallest"])
            overall_best = cand
        end
    end
    final_status = stC.status

    result = (family = FAMILY_ID, start = START_ID, delta = delta, stages = stages,
              final_status = final_status, best = overall_best,
              protocol_sha = MANIFEST["source"]["protocol_sha"], written_utc = string(now(UTC)))
    atomic_serialize(bundle_path, result)
    return result
end

function atomic_serialize(path::String, obj)
    tmp = path * ".tmp"
    serialize(tmp, obj)
    mv(tmp, path; force = true)
end

# ============================================================================
# 7. Seed loading (Phase-I delta=0.1 entry point)
# ============================================================================

function load_seed_w0()
    seeds_dir = joinpath(CAMPAIGN_ROOT, "seeds", "seeds", START_ID)
    econ = deserialize(joinpath(seeds_dir, "economic_seed.jls"))
    fam_dir = joinpath(seeds_dir, FAMILY_ID)
    if !isdir(fam_dir)
        # UNRESTRICTED / COMMON_MARGINALS / COMMON_FRECHET have no nu block -- pure economic vector.
        return copy(econ.economic_vector), nothing
    end
    fseed = deserialize(joinpath(fam_dir, "full_outer_seed.jls"))
    if isempty(fseed.nu_values)
        return copy(econ.economic_vector), fseed.Delta_star
    end
    # fseed.nu_values is ALREADY the ACTIVE (omission-reduced) nu vector -- exactly what the outer
    # KNITRO search vector's own nu block holds (cb_F!: `νvec_active = exp.(w[D2_econ+1:end])`).
    # The driver reconstructs the DENSE nu internally via scatter_nu_eff + its own freshly-computed
    # originzc_profiled_nu_value/meanzc_profiled_nu_value (a function of the CURRENT point, not a
    # fixed seed-time value) on every evaluation -- it must NEVER be pre-expanded to dense here.
    # Confirmed live 2026-08-08: pre-expanding to dense here silently added the omitted focal
    # coordinate as an EXTRA free dimension, producing DimensionMismatch(380 vs 379) deep inside
    # cm_z_from_a (a genuinely dense/active length confusion, not a production driver bug -- see
    # multistart_seed_generator.jl's own dense_nu_for_solve, which is the ONLY place expansion to
    # dense is supposed to happen, and it happens per-eval inside the driver, not at seed load time).
    return vcat(econ.economic_vector, log.(fseed.nu_values)), fseed.Delta_star
end

# ============================================================================
# 8. Chain driver: delta = 0.1 -> 0.5 -> 1 -> 2, continuation rule = best verified feasible at
#    the previous delta (protocol §7).
# ============================================================================

function main()
    deltas = Float64.(MANIFEST["scientific"]["deltas"])
    w0, seed_Delta = load_seed_w0()
    lp("Loaded seed for ", FAMILY_ID, "/", START_ID, ": len(w0)=", length(w0), " seed_Delta_star=", seed_Delta)

    prev_best_w = w0
    for (i, delta) in enumerate(deltas)
        delta_dir = joinpath(CAMPAIGN_ROOT, "phase1_discovery", FAMILY_ID, START_ID, "delta_$(delta)")
        lp("-"^100); lp(FAMILY_ID, "/", START_ID, " delta=", delta, " starting from w[1]=gp=", prev_best_w[1])
        result = run_delta_cell(prev_best_w, delta, delta_dir; seed_Delta_at_delta = (i == 1 ? seed_Delta : nothing))
        lp(FAMILY_ID, "/", START_ID, " delta=", delta, " FINAL status=", result.final_status,
           " best_Delta=", result.best === nothing ? "none" : result.best.Delta)
        if result.best !== nothing
            prev_best_w = result.best.w
        elseif result.final_status == :START_NOT_RESTORED_TO_DELTA
            # per addendum §D/E: keep the R0 stage's own best restored point (may not satisfy this
            # delta) as the seed for the NEXT delta -- never silently reuse the ORIGINAL seed.
            prev_best_w = result.stages[end].best === nothing ? prev_best_w : result.stages[end].best.w
        end
        # else: no verified point at all this delta -- carry the previous delta's incumbent forward unchanged.
    end
    lp("="^100); lp(FAMILY_ID, "/", START_ID, " CHAIN COMPLETE"); lp("="^100)
end

main()
