# ============================================================================
# Continuation 10, Sections 5+6: THE production D=20 driver.
#
# This is the file the coordinating session should use for Section 10's real
# frontier runs (see docs/fullA_D20_checkpoint_resume_report.md, which states
# this explicitly). Built as a direct extension of
# `c9_phase8_d20_pilot.jl`'s `profile_minimize`/`joint_polish` architecture
# (same production config: compressed moments, h_mode=:cached bandwidth
# policy, multi_method=:top3, SR1 outer Hessian -- unchanged, already gated
# PASS by Continuation 9 Phases 6/7/8). NOT a rewrite from scratch.
#
# Two things are new relative to that pilot driver:
#
#   SECTION 5 -- every evaluation goes through `evaluate_fullA_screened_ranged`
#   (fast_range_screen.jl, integration/fullA-fast-range-screen -- promoted to
#   THE production screening path; the underlying pairwise/witness/zero-winner
#   checks are infeasibility_screen.jl's, unchanged), reusing the ctx-level
#   `pairwise`/`witness` structures `context_real_d20.jl::d20_real_setup`
#   builds ONCE at context-construction time plus a per-ctx
#   `RangedScreenContext` (`rsc`, built once via `build_ranged_screen_context`
#   right after `ctx`) in the order: pairwise cert -> pre-winner envelope
#   cert -> witness -> destination winner-scan (fused zero-winner + winning-
#   range checks, single pass) -> general range-screen safety net (reusing
#   the already-built CompressedFactual) -> CC inner solve. A raw, unscreened
#   `evaluate_fullA_fast` call no longer appears anywhere in this driver's
#   hot path. See docs/fullA_fast_range_screen_production_integration.md for
#   the validation/benchmark backing this promotion (real D=20/W=80,000, 0
#   false positives across the recovered pathology catalogue, no measurable
#   overhead on warm feasible calls).
#
#   SECTION 6 -- checkpoint/resume. Every accepted KNITRO outer iterate (via
#   `KN_set_newpt_callback`), every new best-feasible point, every
#   `checkpoint_interval_s` of wall time, and the end of every profile/polish
#   stage triggers a checkpoint (`D20Checkpoint`, serialized via the stdlib
#   `Serialization` module -- no extra dependency). A checkpoint captures
#   everything askED for that this investigation's KNITRO.jl API surface
#   actually exposes: g, full+reduced log-A, the CC inner dual warm start
#   (`ctx.obj.x`), the current best feasible point, the FD bandwidth cache,
#   enough state to cheaply rebuild the SAME infeasibility-witness (the draw
#   seed -- see the note below on why this had to be introduced), the
#   branch/delta, and the draw seed. It does NOT capture KNITRO's internal
#   quasi-Newton (SR1/BFGS/L-BFGS) Hessian-approximation state -- the
#   KNITRO.jl/C API does not expose extracting or reinjecting that across
#   separate `KN_new()` instances (confirmed by inspecting `names(KNITRO,
#   all=true)` for anything hessian/state/restart-shaped -- nothing found
#   beyond `KN_get_hessian_values`, which reports the CURRENT exact-Hessian
#   evaluation for exact-Hessian modes, not the internal QN approximation
#   for hessopt=SR1/BFGS/L-BFGS this driver actually uses). A resumed run
#   restarts its outer Hessian approximation from KNITRO's own default
#   initialization -- same as any fresh warm-started solve.
#
# A REAL, non-obvious finding this task surfaced: `context_real_d20.jl`'s
# underlying `importData`/`genRands.jl::genExpRands!` draws the W Frechet
# simulation support via the GLOBAL Julia RNG (`rand!(U)`), UNSEEDED for the
# real-data (`fakeData==3`) path (only the synthetic `fakeData in (1,2)` paths
# call `Random.seed!`). This means `ctx.U` -- and therefore every Delta_dual/
# gravity/moment-residual number this whole investigation computes -- is NOT
# reproducible across separate Julia processes unless something seeds the RNG
# first. A checkpoint/resume driver by definition restarts in a NEW process,
# so THIS driver introduces the explicit `Random.seed!(draw_seed)` call
# (immediately before `d20_real_setup`) that the rest of this investigation's
# single-process scripts never needed, and records `draw_seed` in every
# checkpoint so a resume reconstructs the bit-identical `ctx.U` (and hence
# `ctx.pairwise`/`ctx.witness`/every downstream calculation). This is
# validated directly, not assumed -- see the bottom of this file / the
# checkpoint-resume report.
#
# DRAW DESIGN (added on top of the above, additive, opt-in): every ctx build
# in this file now goes through `draw_design.jl::d20_real_setup_design`
# instead of calling `Random.seed!(draw_seed); d20_real_setup(...)` directly.
# For the default `draw_design=:pseudorandom` this is the EXACT same call
# sequence (see draw_design.jl's own header) -- zero behavior change, see
# docs/fullA_D20_draw_design_overhead_report.md for the measured (negligible)
# overhead of the added metadata/checksum logging. `:sobol_randomized` and
# `:halton_scrambled` select the two QMC designs ported from
# `diag/fullA-d20-qmc-delta1` (gravity-fullA-d20-qmc-delta1 worktree, tip
# 5882c16 -- now subsumed by this file). Every checkpoint records
# `draw_design` plus both draw checksums (uniform + transformed); resume
# hard-errors on any mismatch (see `load_checkpoint`/resume blocks below) so a
# resumed run can never silently continue with a different draw realization
# than the one it checkpointed against.
# ============================================================================
include(joinpath(@__DIR__, "draw_design.jl"))   # -> d20_real_setup_design (wraps context_real_d20.jl's d20_real_setup + the QMC designs); includes infeasibility_screen.jl transitively (Section 5)
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "cross_delta_cache.jl"))   # allocation/cache-cleanup task §12: CrossDeltaExactCache -- opt-in via exact_cache_override= (run_profile_checkpointed/run_polish_checkpointed) or cross_delta= (run_staged_delta5_continuation, staged_delta5.jl); default off, unused unless explicitly passed
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))   # Continuation 10 Section 9: structured dense-materialize, used by compressed_live.jl / infeasibility_screen.jl
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "lfix_buffer_reuse.jl"))   # Continuation 11 Section 2: validated bit-identical vs composite_gradient_at_fast (0.0 diff, 5 points incl. trajectory test), ~1.1-1.9x faster; now the default gradient below
include(joinpath(@__DIR__, "gradient_workspace.jl"))   # allocation/cache-cleanup task §7: GradWorkspacePool + composite_gradient_at_fast_pooled -- opt-in via use_pooled_gradient= below, verified bit-identical to composite_gradient_at_fast_buffered (test_gradient_workspace.jl, 9/9) and ~4.9x lower allocation (756.5 MB vs 3690.0 MB at a real D=20/W=80000 point); default OFF, old buffered path unchanged when omitted
include(joinpath(@__DIR__, "lfix_base_workspace_pooled.jl"))   # finalization task Phase 3: Backend A+ (composite_gradient_at_Aplus) -- opt-in via price_cache_backend=:aplus
include(joinpath(@__DIR__, "lfix_factorized_workspace.jl"))   # finalization task Phase 3: Backend C+ (composite_gradient_at_Cplus) -- opt-in via price_cache_backend=:cplus
include(joinpath(@__DIR__, "lfix_kbplus_workspace.jl"))   # finalization task Phase 4: Backend :kbplus (composite_gradient_at_KBplus, ratio-based no-W-scale-exp reconstruction) -- opt-in via price_cache_backend=:kbplus
include(joinpath(@__DIR__, "bandwidth_cache_policy.jl"))
include(joinpath(@__DIR__, "fast_range_screen.jl"))   # pre-winner envelope + fused winning-range + general safety-net screens -- THE production screening path via evaluate_fullA_screened_ranged, wired into screened_eval below (used by every cb_F!/cb_G!/cb_newpt! callback); see docs/fullA_fast_range_screen_production_integration.md
include(joinpath(@__DIR__, "dual_bank.jl"))   # successful-dual/KKT-scored small warm-start bank, ported from diag/fullA-d20-warmstart-replay; opt-in via use_dual_bank= on run_profile_checkpointed/run_polish_checkpointed, wired into screened_eval below
include(joinpath(@__DIR__, "negative_cache.jl"))   # negative-cache audit (integration/fullA-negative-cache-audit): typed ConfirmedNegativeResult + SafeNegativeCache, opt-in via use_neg_cache= below; default OFF, zero behavior change unless explicitly enabled -- see docs/fullA_negative_cache_audit.md
include(joinpath(@__DIR__, "dual_bank_ab_harness.jl"))   # DualBankABStats type (needed by screened_eval's signature below) + the A/B harness itself -- see docs/fullA_driver_delta5_diagnostics_handoff.md §12
include(joinpath(@__DIR__, "incumbent_logic.jl"))   # pure, KNITRO-free incumbent seed/compare helpers -- see docs/fullA_driver_delta5_diagnostics_handoff.md §3
include(joinpath(@__DIR__, "direction_bounds.jl"))   # direction-aware gp box split at the Frechet benchmark -- see docs/fullA_driver_delta5_diagnostics_handoff.md's gamma-bounds addendum section
include(joinpath(@__DIR__, "knitro_status.jl"))   # KNITRO termination-status decoder + native per-solve diagnostics -- see docs/fullA_driver_delta5_diagnostics_handoff.md §9-10
include(joinpath(@__DIR__, "reusable_context.jl"))   # build_fullA_context / set_context_delta! -- see docs/fullA_driver_delta5_diagnostics_handoff.md §5
include(joinpath(@__DIR__, "organic_failure_capture.jl"))   # organic -300 failure archive + replay -- see docs/fullA_driver_delta5_diagnostics_handoff.md §7-8
using KNITRO, Printf, Dates, Random, Statistics, Serialization
using LinearAlgebra: norm, dot

include(joinpath(@__DIR__, "knitro_version_check.jl"))
const LOADED_KNITRO_RELEASE = verify_knitro_version()

const FEASIBLE_CODES = (0, -100, -101, -103)
const SOLVER_STATE_NOTE = "KNITRO's internal quasi-Newton (SR1/BFGS/L-BFGS) Hessian-approximation " *
    "state is NOT exposed by the KNITRO.jl/C API for extraction+reinjection across separate " *
    "KN_new() instances -- checked via names(KNITRO, all=true); nothing hessian/state/restart-shaped " *
    "beyond KN_get_hessian_values (exact-Hessian-mode only, not applicable to this driver's SR1 " *
    "default). NOT checkpointed. A resumed run restarts its outer Hessian approximation from " *
    "KNITRO's own default initialization."

# ============================================================================
# CHECKPOINT STRUCTURE (Section 6)
# ============================================================================
struct D20Checkpoint
    schema::Int
    run_id::String
    label::String
    branch::Symbol            # :upper or :lower
    find_smallest::Bool
    delta::Float64
    W::Int
    draw_seed::Int
    g::Float64                 # gamma'_focal at checkpoint time
    zfree::Vector{Float64}      # reduced A-block free coords (length D^2-1) at checkpoint time
    logA_full::Matrix{Float64}  # D x D, full log(Aod_theta), gravity-feasible by construction
    dual_warm_start::Vector{Float64}    # copy of ctx.obj.x at checkpoint time
    bandwidth_cache::Dict{Int,Float64}
    best_feasible::Any          # NamedTuple or nothing -- current best feasible point tracked so far
    n_eval::Int
    knitro_iter::Int
    wall_elapsed::Float64
    checkpoint_reason::Symbol   # :iteration | :new_best | :wall_interval | :stage_complete
    screen_counts::NamedTuple   # (pairwise=.., witness=.., winner=.., envelope=.., winning_range=.., safety_net=.., passed=..) cumulative at ckpt time -- loosely typed field, old 4-key checkpoints from before the fast-range-screen wiring still deserialize fine (never destructured on resume, report-only)
    # ---- fields ONLY for the resume-reproducibility acceptance test, not needed for optimization ----
    verify_Delta_dual::Float64
    verify_gravity_value::Float64
    verify_max_abs_moment_kkt_resid::Float64
    verify_moment_resid_norm::Float64
    solver_state_note::String
    # ---- schema 2 (draw-design port, additive): which of the three validated draw
    # designs (draw_design.jl) produced ctx.U, plus checksums of both the recovered
    # raw-uniform draws and the transformed Exp(1) draws (draw_design_meta). A resume
    # regenerates ctx via the SAME (draw_design, draw_seed) and hard-errors if either
    # checksum doesn't match -- see the resume blocks in run_profile_checkpointed /
    # run_polish_checkpointed below. ----
    draw_design::Symbol
    draw_checksum_uniform::String
    draw_checksum_transformed::String
    # ---- schema 3 (KNITRO version-check addition, additive): the native KN_get_release() string
    # active when this checkpoint was written, so a resume can be checked against (or at least
    # reported against) the solver version that produced it. ----
    knitro_version::String
end

"Schema 1 checkpoints (pre-draw-design-port) do not have draw_design/checksum fields; schema 2 checkpoints do not have knitro_version. Schema must be 3 to resume through this file's resume validation."
const CHECKPOINT_SCHEMA = 3

# AUD-09 fix: a cold-recomputed check at the checkpoint's own recorded point must be a HARD
# resume gate, not merely a logged discrepancy. Since context/draws are already separately
# hard-gated to match exactly (draw checksums above), a cold re-evaluation at the SAME point
# under the SAME context should reproduce the checkpoint's recorded Delta/gravity/KKT-residual/
# moment-mean to near machine precision -- these tolerances are tighter than
# VerifiedSuccessTolerances (oracle.jl), which governs a NEW point's own internal consistency,
# not a same-point reproducibility check.
Base.@kwdef struct ResumeTolerances
    delta_tol::Float64 = 1e-6
    gravity_tol::Float64 = 1e-6
    kkt_tol::Float64 = 1e-6
    moment_mean_norm_tol::Float64 = 1e-6
end
const DEFAULT_RESUME_TOL = ResumeTolerances()

"AUD-09 fix: hard-reject a resume whose cold-recomputed verification point disagrees with the
checkpoint's own recorded values beyond tolerance, instead of only logging the discrepancy.
`label`/`fn` are for the error message only."
function check_resume_tolerances!(label::AbstractString, fn::AbstractString,
        d_delta::Float64, d_grav::Float64, d_kkt::Float64, d_mr::Float64;
        tol::ResumeTolerances = DEFAULT_RESUME_TOL)
    bad = String[]
    d_delta <= tol.delta_tol || push!(bad, "|ΔDelta_dual|=$(d_delta) > $(tol.delta_tol)")
    d_grav <= tol.gravity_tol || push!(bad, "|Δgravity_value|=$(d_grav) > $(tol.gravity_tol)")
    d_kkt <= tol.kkt_tol || push!(bad, "|Δmax_abs_moment_kkt_resid|=$(d_kkt) > $(tol.kkt_tol)")
    d_mr <= tol.moment_mean_norm_tol || push!(bad, "|Δ||benchmark_unweighted_moment_mean|||=$(d_mr) > $(tol.moment_mean_norm_tol)")
    isempty(bad) || error("$(fn)($(label)): cold-recomputed resume verification exceeds tolerance " *
        "-- refusing to resume (AUD-09; docs/fullA_independent_audit_remediation.md): " * join(bad, "; "))
    return nothing
end

"Atomic-ish checkpoint write: serialize to a .tmp file then mv, so a crash mid-write never leaves a half-written checkpoint that a resume could load."
function save_checkpoint(path::AbstractString, ckpt::D20Checkpoint)
    tmp = path * ".tmp"
    serialize(tmp, ckpt)
    mv(tmp, path; force = true)
    return path
end
function load_checkpoint(path::AbstractString)
    ckpt = deserialize(path)::D20Checkpoint
    ckpt.schema == CHECKPOINT_SCHEMA ||
        error("load_checkpoint($path): schema=$(ckpt.schema), expected $(CHECKPOINT_SCHEMA) -- " *
              "this checkpoint predates either the draw-design port (schema 1, no " *
              "draw_design/checksum fields) or the KNITRO version-check addition (schema 2, no " *
              "knitro_version field). Start a fresh run instead of resuming from an older-schema " *
              "checkpoint.")
    return ckpt
end

"""
    guard_checkpoint_path(path, draw_design, checksum_uniform, checksum_transformed)

Namespacing safety net (§5 of the draw-design port brief): if a checkpoint already
exists at `path` from a DIFFERENT draw_design or draw checksum, refuse to overwrite
it. This is deliberately a file-path-level guard, not a change to any cache-key
struct (oracle.jl's `FullAEvalKey` etc.) -- those are being modified concurrently by
a sibling workstream, and this driver's own inner-loop calls already run with
`cache=nothing, use_cache=false` (see `screened_eval`), so there is no shared-cache
collision risk to fix there. The real collision risk is two different-design runs
pointed at the SAME `ckpt_dir`/`label`, which would otherwise silently clobber each
other's checkpoint file; this catches that at every checkpoint write, not just once
at startup, since `ckpt_dir` is caller-supplied and long-running processes can be
misconfigured mid-flight.
"""
function guard_checkpoint_path(path::AbstractString, draw_design::Symbol, checksum_uniform::AbstractString, checksum_transformed::AbstractString)
    isfile(path) || return nothing
    # Remediation task Part E (finding F12): a pre-schema-3 file at `path` makes
    # `deserialize(path)::D20Checkpoint` throw a raw type/field-mismatch error instead of the
    # informative schema message `load_checkpoint` gives -- wrap with the same guidance.
    local prior
    try
        prior = deserialize(path)::D20Checkpoint
    catch e
        error("guard_checkpoint_path($path): could not deserialize an existing file at this path " *
              "as the current D20Checkpoint schema (schema=$CHECKPOINT_SCHEMA) -- it likely " *
              "predates a schema change (see load_checkpoint's error message for the schema " *
              "history). Underlying error: $(sprint(showerror, e)). Start a fresh run with a " *
              "different ckpt_dir/label instead of resuming from an incompatible checkpoint.")
    end
    if prior.draw_design != draw_design || prior.draw_checksum_uniform != checksum_uniform || prior.draw_checksum_transformed != checksum_transformed
        error("guard_checkpoint_path($path): an existing checkpoint at this path was built under a " *
              "DIFFERENT draw design/checksum (design=:$(prior.draw_design), " *
              "checksum=($(prior.draw_checksum_uniform),$(prior.draw_checksum_transformed))) than the " *
              "current run (design=:$(draw_design), checksum=($(checksum_uniform),$(checksum_transformed))). " *
              "Refusing to overwrite -- use a design-namespaced ckpt_dir, e.g. " *
              "results/<draw_design>/seed_<draw_seed>/... (see docs/fullA_D20_draw_design_*.md).")
    end
    return nothing
end

x_free_from_w(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

# ============================================================================
# Screened evaluation wrapper (Section 5, extended by the fast-range-screen
# integration): EVERY call in this driver goes through this, never raw
# evaluate_fullA_fast. Tracks per-stage rejection counts for the report.
#
# Now routes through evaluate_fullA_screened_ranged (fast_range_screen.jl) --
# the pre-winner envelope + fused winning-range + general safety-net screens
# are THE production path, not an opt-in alternative alongside the old one.
# When `rsc.envelope === nothing` (unsupported ctx -- see fast_range_screen.jl's
# EnvelopeUnsupportedContext), evaluate_fullA_screened_ranged itself falls
# back to the existing zero-winner-only screen_hard_winners automatically, so
# this wrapper needs no separate fallback branch.
# ============================================================================
mutable struct ScreenCounters
    pairwise::Int
    witness::Int
    winner::Int
    envelope::Int
    winning_range::Int
    safety_net::Int
    passed::Int
    rejections::Vector{NamedTuple}   # audit trail: (stage, o, d, n_eval)
end
ScreenCounters() = ScreenCounters(0, 0, 0, 0, 0, 0, 0, NamedTuple[])
as_namedtuple(sc::ScreenCounters) = (pairwise = sc.pairwise, witness = sc.witness, winner = sc.winner,
                                      envelope = sc.envelope, winning_range = sc.winning_range,
                                      safety_net = sc.safety_net, passed = sc.passed)

function screened_eval(xf::AbstractVector{Float64}, ctx, rsc::RangedScreenContext, sc::ScreenCounters,
        n_eval_ref::Ref{Int}; warm::Bool = true, bank::Union{Nothing,DualBank} = nothing,
        zfree::Union{Nothing,AbstractVector{Float64}} = nothing,
        exact_cache::Union{Nothing,SafeExactCache,CrossDeltaExactCache} = nothing,   # BUGFIX
        # (finalization task Phase 2B, found live): this was Union{Nothing,SafeExactCache} only.
        # Julia enforces keyword-argument type annotations at the call site (confirmed: a
        # minimal repro throws TypeError, not silently widening) -- so every cb_F!/cb_G! call in
        # run_profile_checkpointed/run_polish_checkpointed that forwards exact_cache=exact_cache
        # would TypeError the instant exact_cache held a CrossDeltaExactCache, i.e. the instant
        # cross_delta=true was actually used through the real driver. evaluate_fullA_screened_
        # ranged's own `cache` parameter (fast_range_screen.jl) is untyped and always accepted
        # CrossDeltaExactCache fine via the generic _cache_lookup/_cache_store! dispatch -- this
        # wrapper's narrower annotation was the only thing blocking it. This is why no staged
        # cross_delta=true continuation had ever been observed to complete through the production
        # driver: it could not have, it TypeErrors on the first callback.
        neg_cache::Union{Nothing,SafeNegativeCache} = nothing,
        ab_stats::Union{Nothing,DualBankABStats} = nothing)   # task §12: opt-in instrumentation for
        # the successful-dual-bank A/B harness (dual_bank_ab_harness.jl) -- captures the
        # select_warm_start label/candidate-count/scoring-wall-time this function already computes
        # and previously discarded (`_label` below). nothing (default) = zero overhead, unchanged
        # behavior.
    # Negative-cache audit (Policy B, opt-in): a CONFIRMED negative (see negative_cache.jl,
    # confirm_and_maybe_cache_negative!) short-circuits here with ZERO KNITRO call, same as an
    # exact_cache positive hit. Checked first (cheap dict lookup) -- default nothing, so every
    # existing caller that never passes neg_cache= sees IDENTICAL behavior to before this file
    # existed (this whole block is a no-op when neg_cache === nothing).
    if neg_cache !== nothing
        key = FullAEvalKey(collect(xf), ctx.obj.δ, ctx.obj.find_smallest, ctx.obj.inner_loop_opt, :hard, context_fingerprint(ctx))
        hit = negcache_lookup(neg_cache, key)
        if hit !== nothing
            # template: cheapest available same-shape NamedTuple -- a fresh screen-only infeasible_result
            # skeleton at this xf (never calls KNITRO; matches oracle.jl's own field set exactly)
            θ_full_t = CS.reconstruct_full(xf, ctx.m)
            template = infeasible_result(xf, θ_full_t, ctx, :negative_cache_hit, 0, 0, 0, 0.0, "", warm)
            result = negative_result_namedtuple(xf, template, hit)
            return result, (screen_status = :negative_cache_hit, elapsed = 0.0)
        end
    end
    # Successful-dual/KKT-scored bank (dual_bank.jl): only engages on warm calls (a cold call
    # resets obj.x .= NaN itself downstream, making any assignment here moot) and only when the
    # caller supplied both a bank and the reduced (zfree) coordinates it's indexed on. Falls back
    # silently to production's existing single-slot ctx.obj.x behavior (no-op here) on a
    # TiedWinnerError while building the scoring factual -- never lets a diagnostic scoring step
    # abort a real optimization callback.
    if warm && bank !== nothing && zfree !== nothing
        θ_full_score = CS.reconstruct_full(xf, ctx.m)
        cf_score = try
            build_compressed_factual(θ_full_score, ctx; check_ties = true)
        catch e
            e isa TiedWinnerError ? nothing : rethrow()
        end
        if cf_score !== nothing
            t_score0 = ab_stats === nothing ? NaN : time()
            n_cands_before = length(bank.history) + 1   # +1 for the always-present :neutral candidate; :actual/:last_accepted/:nearest are conditional, see dual_bank.jl select_warm_start
            x0, _label = select_warm_start(bank, ctx.obj, cf_score, zfree)
            if ab_stats !== nothing
                record_ab_selection!(ab_stats, _label, n_cands_before, time() - t_score0)
            end
            ctx.obj.x .= x0
        end
    end
    result, screen_meta = evaluate_fullA_screened_ranged(xf, ctx, rsc; moment_representation = :compressed,
        cache = exact_cache, use_cache = exact_cache !== nothing, warm = warm, tag = "",
        pairwise = ctx.pairwise, witness = ctx.witness, use_witness = ctx.witness !== nothing)
    if bank !== nothing && zfree !== nothing && result.inner_status in FEASIBLE_CODES
        record_success!(bank, n_eval_ref[], zfree, vcat(result.zeta, result.lambda))
    end
    st = screen_meta.screen_status
    if st === :pairwise_certified_infeasible
        sc.pairwise += 1
        push!(sc.rejections, (stage = :pairwise, o = screen_meta.worst_o, d = screen_meta.worst_d, n_eval = n_eval_ref[]))
    elseif st === :witness_certified_infeasible
        sc.witness += 1
        push!(sc.rejections, (stage = :witness, o = screen_meta.worst_o, d = screen_meta.worst_d, n_eval = n_eval_ref[]))
    elseif st === :winner_scan_infeasible
        sc.winner += 1
        push!(sc.rejections, (stage = :winner, o = screen_meta.worst_o, d = screen_meta.worst_d, n_eval = n_eval_ref[]))
    elseif st === :EXACT_INFEASIBLE_PREWINNER_ENVELOPE
        sc.envelope += 1
        push!(sc.rejections, (stage = :envelope, o = screen_meta.worst_o, d = screen_meta.worst_d, n_eval = n_eval_ref[]))
    elseif st === :EXACT_INFEASIBLE_WINNING_RANGE
        sc.winning_range += 1
        push!(sc.rejections, (stage = :winning_range, o = screen_meta.worst_o, d = screen_meta.worst_d, n_eval = n_eval_ref[]))
    elseif st === :EXACT_INFEASIBLE_MOMENT_RANGE
        sc.safety_net += 1
        push!(sc.rejections, (stage = :safety_net, o = get(screen_meta, :certificate, nothing) === nothing ? 0 : screen_meta.certificate.origin,
                               d = get(screen_meta, :certificate, nothing) === nothing ? 0 : screen_meta.certificate.destination, n_eval = n_eval_ref[]))
    else
        sc.passed += 1
    end
    return result, screen_meta
end

# ============================================================================
# PROFILE stage: fixed g, minimize Delta_dual over the A-block only.
# Direct extension of c9_phase8_d20_pilot.jl::profile_minimize with
# screening (Section 5) + checkpointing (Section 6) wired in.
# ============================================================================
const VALID_PRICE_CACHE_BACKENDS = (:buffered, :pooled, :aplus, :cplus, :kbplus)

"""
    resolve_price_cache_backend(label, use_pooled_gradient, price_cache_backend) -> Symbol

Finalization task Phase 3: single source of truth for which gradient backend a driver call
uses, reconciling the OLD `use_pooled_gradient::Union{Nothing,Bool}` flag (nothing = caller
didn't specify) with the NEW `price_cache_backend::Union{Nothing,Symbol}` selector (nothing =
caller didn't specify). Never lets one silently override the other:
- neither given -> `:cplus` (finalization task Phase 6 default, changed from `:buffered`
  2026-07-22: C+ is 4.0-4.2x faster / 66.8x less memory than the prior default at real
  D=20/W=80,000 points, correct to ~4.3e-17, and its remaining gates -- independent
  optimized-value directional check, cross_delta+backend integration, checkpoint/resume,
  D=20 short trajectories at δ=1/δ=2 -- all closed. See
  docs/fullA_ALLOCATION_CROSSDELTA_KB_GATE_2026-07-22.md §6 for the full adoption record,
  including why `:kbplus` -- also fully correct -- was NOT chosen (measured ~15-17% slower
  than C+, despite eliminating the W-scale `exp` calls it was built to remove).
- explicit `use_pooled_gradient=false` (OLD API) still means `:buffered`, literally, not the
  new default -- backward compatibility for existing callers of the old boolean flag is
  preserved exactly, never silently reinterpreted.
- only one of the two kwargs given (other than the `use_pooled_gradient=false` case above) ->
  that one wins.
- both given, consistent (`use_pooled_gradient=true` + `price_cache_backend=:pooled`, or
  `use_pooled_gradient=false` + any non-`:pooled` backend) -> the explicit backend.
- both given, contradictory (e.g. `use_pooled_gradient=true` + `price_cache_backend=:cplus`) ->
  hard error, not a silent pick.
"""
function resolve_price_cache_backend(label::AbstractString, use_pooled_gradient::Union{Nothing,Bool},
        price_cache_backend::Union{Nothing,Symbol})
    if price_cache_backend !== nothing
        price_cache_backend in VALID_PRICE_CACHE_BACKENDS ||
            error("$label: price_cache_backend=:$price_cache_backend not recognized -- must be one of $VALID_PRICE_CACHE_BACKENDS")
        if use_pooled_gradient !== nothing
            implied = use_pooled_gradient ? :pooled : :buffered
            (implied == price_cache_backend || (!use_pooled_gradient && price_cache_backend != :pooled)) ||
                error("$label: contradictory gradient-backend selection -- use_pooled_gradient=$use_pooled_gradient " *
                      "(implies :$implied) but price_cache_backend=:$price_cache_backend was also given explicitly. " *
                      "Pass only one of these two kwargs.")
        end
        return price_cache_backend
    elseif use_pooled_gradient !== nothing
        return use_pooled_gradient ? :pooled : :buffered
    else
        return :cplus   # finalization task Phase 6 default (was :buffered) -- see docstring above
    end
end

function run_profile_checkpointed(label::String, g_in::Float64, find_smallest_in::Bool, zfree_start_in::Vector{Float64};
        maxtime_real::Float64 = 900.0, hessopt_tag::String = "sr1",
        W_in::Int = 80000, delta_in::Float64 = 1.0, draw_seed_in::Int = 20260719,
        draw_design_in::Union{Nothing,Symbol} = nothing,   # nothing = "caller did not specify" (distinct from
        # explicitly passing :pseudorandom!) -- see the resume block: this sentinel is required to tell
        # "no opinion, inherit whatever the checkpoint used" apart from "I explicitly want :pseudorandom",
        # which must still hard-error if the checkpoint was actually built under a different design.
        ckpt_dir::AbstractString, checkpoint_interval_s::Float64 = 90.0,
        resume_from::Union{Nothing,AbstractString} = nothing,
        logio::Union{Nothing,IO} = nothing,
        use_dual_bank::Bool = true, dual_bank_size::Int = 8, use_exact_cache::Bool = true,
        exact_cache_override::Union{Nothing,SafeExactCache,CrossDeltaExactCache} = nothing,   # allocation/
        # cache-cleanup task §12: when given, USE THIS cache instead of constructing a fresh
        # SafeExactCache() internally -- lets a caller (e.g. run_staged_delta5_continuation) thread
        # ONE CrossDeltaExactCache across every stage of a staged continuation so an exact hit from
        # a DIFFERENT delta-stage (same x_free/find_smallest/context) is served without a re-solve.
        # Default nothing: zero behavior change (falls back to use_exact_cache ? SafeExactCache() :
        # nothing, exactly as before this kwarg existed) -- CrossDeltaExactCache's own correctness
        # (context-fingerprinted key, AUD-08) is a precondition for safely enabling this; see
        # docs/fullA_independent_audit_remediation.md AUD-08 and docs/fullA_postmerge_allocation_
        # productionization.md.
        organic_failures::Union{Nothing,OrganicFailureCollector} = nothing,   # opt-in archive of the
        # first N genuine (post-screen, real KNITRO) inner failures (task §7); nothing (default) = no capture.
        use_pooled_gradient::Union{Nothing,Bool} = nothing,   # allocation/cache-cleanup task §7: opt-in
        # composite_gradient_at_fast_pooled (GradWorkspacePool) instead of composite_gradient_
        # at_fast_buffered. Verified bit-identical (test_gradient_workspace.jl, 9/9) and ~4.9x
        # lower allocation (756.5 MB vs 3690.0 MB at a real D=20/W=80000 point) on this branch.
        # Default nothing (finalization task Phase 3: was `false`; widened to a nothing-sentinel
        # so resolve_price_cache_backend can tell "not specified" from "explicitly false" and
        # reconcile this OLD flag with the NEW price_cache_backend= selector below without one
        # silently overriding the other). nothing behaves exactly like the old `false` default
        # when price_cache_backend is also omitted.
        price_cache_backend::Union{Nothing,Symbol} = nothing,   # finalization task Phase 3: single
        # selector for which gradient backend to use -- :buffered (default) | :pooled (=
        # composite_gradient_at_fast_pooled, GradWorkspacePool only) | :aplus (persistent two-
        # tensor LFixBaseWorkspace + GradWorkspacePool, bit-identical to :pooled, ~1.1-1.2x
        # faster) | :cplus (factorized O(W*D) LFixFactorizedWorkspace + GradWorkspacePool,
        # ~4x faster / ~67x less allocation than :buffered at D=20/W=80000, correct to ~1e-17;
        # see docs/fullA_factorized_price_production_gate.md). Reconciled with the older
        # use_pooled_gradient kwarg via resolve_price_cache_backend -- see that function's
        # docstring for the exact precedence/contradiction rules. nothing (default): behavior
        # governed entirely by use_pooled_gradient, i.e. zero change for existing callers.
        maxit_override::Union{Nothing,Int} = nothing,   # finalization task Phase 2B: override the
        # hardcoded maxit=1_000_000 outer-iteration cap so two arms of an A/B comparison (e.g.
        # cross_delta on/off) can be capped at an IDENTICAL iteration count, not just an identical
        # wall-clock budget -- otherwise a faster arm could simply get more iterations for free in
        # a wall-clock-limited comparison, confounding "cache helped" with "cache left more time".
        # Default nothing: behavior unchanged (maxit=1_000_000, effectively unbounded, terminated
        # by maxtime_real as before this kwarg existed).
        allow_direction_box_migration::Bool = false)   # addendum: by default, a fixed g (fresh or
        # resumed) on the wrong side of the Frechet benchmark for its own direction is REJECTED with a
        # hard error (this stage fixes g, so there is no zfree-only box to widen -- the check is purely
        # a validity gate on the caller's own g). See direction_bounds.jl.
    lp(xs...) = (println(xs...); logio !== nothing && (println(logio, xs...); flush(logio)); flush(stdout))

    mkpath(ckpt_dir)
    resumed = resume_from === nothing ? nothing : load_checkpoint(resume_from)
    if resumed !== nothing && resumed.checkpoint_reason == :stage_complete_unverified
        println("WARNING: resuming from a checkpoint whose terminal point FAILED verification at write ",
                "time (checkpoint_reason=:stage_complete_unverified, AUD-10) -- best[]/best_feasible[] ",
                "inside this checkpoint is still the correct scientific incumbent; only the raw terminal ",
                "solver-state fields are suspect.")
    end
    g = g_in; find_smallest = find_smallest_in; zfree_start = copy(zfree_start_in)
    W = W_in; delta = delta_in; draw_seed = draw_seed_in
    draw_design = draw_design_in === nothing ? :pseudorandom : draw_design_in
    bandwidth_cache = Dict{Int,Float64}()
    if resumed !== nothing
        g = resumed.g; find_smallest = resumed.find_smallest; zfree_start = copy(resumed.zfree)
        W = resumed.W; delta = resumed.delta; draw_seed = resumed.draw_seed
        # draw_design must match what the caller EXPLICITLY asked for -- if the caller passed nothing
        # (no opinion), silently inherit the checkpoint's own design; if the caller passed a specific
        # design (including :pseudorandom) that conflicts with the checkpoint's recorded design, hard-error.
        # Never silently resume under a DIFFERENT design than the run was checkpointed with. See §5 of the
        # draw-design port brief.
        if draw_design_in !== nothing && draw_design_in != resumed.draw_design
            error("run_profile_checkpointed($label): resume draw_design mismatch -- checkpoint has " *
                  ":$(resumed.draw_design), caller requested :$(draw_design_in). Refusing to resume.")
        end
        draw_design = resumed.draw_design
        bandwidth_cache = copy(resumed.bandwidth_cache)
        lp("[", label, "] RESUMING from ", resume_from, " (reason=", resumed.checkpoint_reason,
           " n_eval=", resumed.n_eval, " knitro_iter=", resumed.knitro_iter, " wall_elapsed=", resumed.wall_elapsed, "s)")
    end

    ctx = d20_real_setup_design(W = W, δ = delta, find_smallest = find_smallest,
                                 draw_design = draw_design, draw_seed = draw_seed)
    pe = build_pivot_elimination(ctx)
    D = ctx.D; D2 = D^2; n = D2 - 1
    rsc = build_ranged_screen_context(ctx)
    resolved_backend = resolve_price_cache_backend(label, use_pooled_gradient, price_cache_backend)
    # one pool/workspace per ctx (built ONCE here), not per gradient call -- see docs/
    # fullA_factorized_price_production_gate.md for A+/C+'s own persistence/aliasing design.
    grad_pool = resolved_backend in (:pooled, :aplus, :cplus, :kbplus) ? build_grad_workspace_pool(W) : nothing
    lfix_ws = resolved_backend == :aplus ? build_lfix_base_workspace(D, W) : nothing
    lfix_c_ws = resolved_backend == :cplus ? build_lfix_factorized_workspace(D, W) : nothing
    lfix_kb_ws = resolved_backend == :kbplus ? build_lfix_kbplus_workspace(D, W) : nothing
    lp("[", label, "] ctx built, D=", D, " W=", W, " draw_seed=", draw_seed, " draw_design=", draw_design,
       " price_cache_backend=", resolved_backend,
       " draw_checksum=(", ctx.draw_meta.checksum_uniform, ",", ctx.draw_meta.checksum_transformed, ")",
       " screen_setup_wall=", ctx.screen_setup_wall,
       " envelope_screen_supported=", rsc.envelope !== nothing,
       rsc.envelope === nothing ? " (reason: $(rsc.unsupported_reason))" : "")

    # Direction-aware validity gate on the fixed g (addendum): reject -- do not clamp -- a
    # g (fresh or resumed) on the wrong side of the Frechet benchmark for its own direction.
    if !allow_direction_box_migration
        validate_gp_in_direction_box(g, ctx, find_smallest; label = label,
            what = resumed !== nothing ? "resumed checkpoint's fixed g" : "supplied fixed g")
    end

    if resumed !== nothing
        # Hard reproducibility gate (§4 of the draw-design port brief): the regenerated draws for THIS
        # process must checksum-match what the checkpoint recorded, or resume is refused outright --
        # this is stronger than the pre-existing "RESUME VALIDATION" numeric-diff print below (which
        # only reports differences, it doesn't gate anything).
        if ctx.draw_meta.checksum_uniform != resumed.draw_checksum_uniform ||
           ctx.draw_meta.checksum_transformed != resumed.draw_checksum_transformed
            error("run_profile_checkpointed($label): regenerated draws do not match checkpoint's recorded " *
                  "checksums -- checkpoint (uniform=$(resumed.draw_checksum_uniform), " *
                  "transformed=$(resumed.draw_checksum_transformed)) vs regenerated " *
                  "(uniform=$(ctx.draw_meta.checksum_uniform), transformed=$(ctx.draw_meta.checksum_transformed)). " *
                  "Refusing to resume with mismatched draws.")
        end
        ctx.obj.x .= resumed.dual_warm_start
        r_verify, _ = evaluate_fullA_screened_ranged(x_free_from_w(vcat(g, zfree_start), pe), ctx, rsc;
            moment_representation = :compressed, cache = nothing, use_cache = false, warm = true,
            pairwise = ctx.pairwise, witness = ctx.witness, use_witness = ctx.witness !== nothing)
        d_delta = abs(r_verify.Delta_dual - resumed.verify_Delta_dual)
        d_grav = abs(r_verify.gravity_value - resumed.verify_gravity_value)
        d_kkt = abs(r_verify.max_abs_moment_kkt_resid - resumed.verify_max_abs_moment_kkt_resid)
        d_mr = abs(norm(r_verify.benchmark_unweighted_moment_mean) - resumed.verify_moment_resid_norm)
        lp("[", label, "] RESUME VALIDATION at checkpoint's own point: ",
           "|ΔDelta_dual|=", d_delta, " |Δgravity_value|=", d_grav,
           " |Δmax_abs_moment_kkt_resid|=", d_kkt, " |Δ||benchmark_unweighted_moment_mean|||=", d_mr)
        lp("[", label, "]   original: Delta_dual=", resumed.verify_Delta_dual, " gravity=", resumed.verify_gravity_value)
        lp("[", label, "]   resumed:  Delta_dual=", r_verify.Delta_dual, " gravity=", r_verify.gravity_value)
        check_resume_tolerances!(label, "run_profile_checkpointed", d_delta, d_grav, d_kkt, d_mr)   # AUD-09
    end

    sc = ScreenCounters()
    n_eval = Ref(resumed !== nothing ? resumed.n_eval : 0)
    bank = use_dual_bank ? DualBank(dual_bank_size) : nothing
    exact_cache = exact_cache_override !== nothing ? exact_cache_override : (use_exact_cache ? SafeExactCache() : nothing)

    # seed the compressed warm-start cache with a cold solve first (same fix c9_phase8_d20_pilot.jl
    # found necessary -- a fresh ctx's very first warm=true call has no prior state to warm-start
    # from and can spuriously report infeasible).
    r_seed, _ = screened_eval(x_free_from_w(vcat(g, zfree_start), pe), ctx, rsc, sc, n_eval; warm = false)
    lp("[", label, "] warm-cache seed (cold): inner_status=", r_seed.inner_status, " Delta=", r_seed.Delta_dual,
       " screen_status=", get(r_seed, :screen_status, :unknown))
    r_seed.inner_status in FEASIBLE_CODES || error("run_profile_checkpointed($label): start point not inner-feasible, cannot proceed")

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt"))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", maxit_override === nothing ? 1_000_000 : maxit_override)
    KNITRO.KN_set_param_by_name(kc, "algorithm", 3)
    xIndices = KNITRO.KN_add_vars(kc, n)
    z_halfwidth = 30.0   # see c9_phase8_d20_pilot.jl's box-bounds root-cause comment
    KNITRO.KN_set_var_lobnds_all(kc, zfree_start .- z_halfwidth)
    KNITRO.KN_set_var_upbnds_all(kc, zfree_start .+ z_halfwidth)
    KNITRO.KN_set_var_primal_init_values_all(kc, zfree_start)

    last_F_state = Ref{Union{Nothing,NamedTuple}}(nothing)
    # See incumbent_logic.jl / docs/fullA_driver_delta5_diagnostics_handoff.md §3: seed the
    # incumbent from the already-cold-verified start point (r_seed) rather than `nothing`.
    seed_cand_feasible = r_seed.inner_status in FEASIBLE_CODES && isfinite(r_seed.Delta_dual)
    seed_cand = (zfree = copy(zfree_start), Delta_dual = r_seed.Delta_dual, gravity_value = r_seed.gravity_value,
                 max_abs_moment_kkt_resid = r_seed.max_abs_moment_kkt_resid, inner_status = r_seed.inner_status,
                 t_elapsed = 0.0, n_eval = n_eval[])
    best = Ref{Union{Nothing,NamedTuple}}(seed_incumbent(resumed !== nothing ? resumed.best_feasible : nothing,
                                                           seed_cand_feasible, seed_cand))
    n_grad_calls = Ref(0)
    policy = BandwidthCachePolicy()
    policy.cache = bandwidth_cache
    trace = NamedTuple[]
    t_start = time()
    last_ckpt_wall = Ref(time())
    knitro_iter = Ref(resumed !== nothing ? resumed.knitro_iter : 0)
    run_id = "c10_prod_$(label)_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"

    function do_checkpoint(reason::Symbol, w_current::Vector{Float64}, r::NamedTuple)
        zfree_now = w_current[2:end]
        logA_full = pivot_expand(zfree_now, pe)
        ckpt = D20Checkpoint(CHECKPOINT_SCHEMA, run_id, label, find_smallest ? :upper : :lower, find_smallest, delta, W, draw_seed,
            w_current[1], copy(zfree_now), logA_full, copy(ctx.obj.x), copy(policy.cache),
            best[], n_eval[], knitro_iter[], time() - t_start, reason, as_namedtuple(sc),
            r.Delta_dual, r.gravity_value, r.max_abs_moment_kkt_resid, norm(r.benchmark_unweighted_moment_mean),
            SOLVER_STATE_NOTE, draw_design, ctx.draw_meta.checksum_uniform, ctx.draw_meta.checksum_transformed,
            LOADED_KNITRO_RELEASE)
        latest_path = joinpath(ckpt_dir, "$(label)_latest.jls")
        guard_checkpoint_path(latest_path, draw_design, ctx.draw_meta.checksum_uniform, ctx.draw_meta.checksum_transformed)
        save_checkpoint(latest_path, ckpt)
        reason in (:new_best, :stage_complete, :stage_complete_unverified) && save_checkpoint(joinpath(ckpt_dir, "$(label)_$(reason)_neval$(n_eval[]).jls"), ckpt)
        return ckpt
    end

    n_cold_retries = Ref(0); n_rejected = Ref(0)
    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        zfree = evalRequest.x
        w = vcat(g, zfree)
        xf = x_free_from_w(w, pe)
        dual_before = copy(ctx.obj.x)   # snapshot for organic-failure capture (task §7), before screened_eval mutates it
        r, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = true, bank = bank, zfree = zfree, exact_cache = exact_cache)
        warm_source = :warm
        if !(r.inner_status in FEASIBLE_CODES)
            n_cold_retries[] += 1
            # BUGFIX (negative-cache audit): pass exact_cache through on the cold retry -- previously
            # omitted, so a cold-retry SUCCESS was never written to the positive cache (see
            # docs/fullA_negative_cache_audit.md). Same fix applied to run_polish_checkpointed's cb_F!/cb_G!.
            r, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = false, exact_cache = exact_cache)
            warm_source = :warm_then_cold
        end
        if organic_failures !== nothing
            maybe_capture_organic_failure!(organic_failures, label, w, xf, r, ctx, pe, sc, n_eval[], knitro_iter[],
                delta, find_smallest, draw_design, draw_seed, warm_source, dual_before, resume_from)
        end
        if !(r.inner_status in FEASIBLE_CODES) || !isfinite(r.Delta_dual)
            n_rejected[] += 1
            reject_point(w[1], "run_profile_checkpointed($label): infeasible/non-finite point (inner_status=$(r.inner_status)), rejecting")
        end
        Δ = r.Delta_dual
        evalResult.obj[1] = Δ
        n_eval[] += 1
        t_el = time() - t_start
        # AUD-03 fix: on a cache hit, screened_eval never re-solved, so ctx.obj.arg1 (m_star) can
        # still hold a DIFFERENT point's state (whatever the last real inner solve left behind).
        # solve_base_state always performs a fresh solve, so it is the only way to guarantee
        # correct m_star here; the non-cache-hit branch keeps the fast hand-built path since
        # ctx.obj.arg1 genuinely is fresh for that case (populated by the primal_weight_recovery
        # call inside screened_eval moments ago). See docs/fullA_independent_audit_remediation.md
        # AUD-03 and its A/B/A regression test.
        base = r.cache_hit ? solve_base_state(xf, ctx) :
            BaseDualState(collect(xf), r.θ_full, r.zeta, r.lambda, copy(ctx.obj.arg1), r.inner_status)
        last_F_state[] = (w = copy(w), base = base)
        # AUD-04 fix: a candidate incumbent must be a scientifically VERIFIED solve (residual/gap
        # tolerances pass, not merely inner_status in FEASIBLE_CODES) before it can replace the
        # best-known point.
        is_new_best = is_verified_success(r) &&
            is_better_profile(Δ, best[] === nothing ? nothing : best[].Delta_dual)
        if is_new_best
            best[] = (zfree = copy(zfree), Delta_dual = Δ, gravity_value = r.gravity_value,
                      max_abs_moment_kkt_resid = r.max_abs_moment_kkt_resid, inner_status = r.inner_status,
                      t_elapsed = t_el, n_eval = n_eval[])
        end
        push!(trace, (idx = n_eval[], t_elapsed = t_el, Delta_dual = Δ, inner_status = r.inner_status,
                       delta_feasible = Δ <= ctx.δ + 1e-6))
        if n_eval[] <= 5 || n_eval[] % 10 == 0
            lp("  [", label, "] eval ", n_eval[], " t=", round(t_el, digits = 1), "s Delta=", Δ, " status=", r.inner_status,
               " screens(pw/wt/wn/env/wr/sn/pass)=", sc.pairwise, "/", sc.witness, "/", sc.winner, "/", sc.envelope, "/", sc.winning_range, "/", sc.safety_net, "/", sc.passed)
        end
        if is_new_best
            do_checkpoint(:new_best, w, r)
        end
        if time() - last_ckpt_wall[] > checkpoint_interval_s
            do_checkpoint(:wall_interval, w, r)
            last_ckpt_wall[] = time()
        end
        return 0
    end
    n_g_recompute = Ref(0)
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        zfree = evalRequest.x
        w = vcat(g, zfree)
        xf = x_free_from_w(w, pe)
        shared = last_F_state[]
        base = shared !== nothing && shared.w == w ? shared.base : nothing
        if base === nothing
            n_g_recompute[] += 1
            r_g, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = true, bank = bank, zfree = zfree, exact_cache = exact_cache)
            if !(r_g.inner_status in FEASIBLE_CODES)
                r_g, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = false, exact_cache = exact_cache)   # BUGFIX, see cb_F! above
            end
            r_g.inner_status in FEASIBLE_CODES || reject_point(w[1], "run_profile_checkpointed($label): cb_G! could not recompute a feasible base state")
            # AUD-03 fix: same reasoning as cb_F! above -- do not trust ctx.obj.arg1 on a cache hit.
            base = r_g.cache_hit ? solve_base_state(xf, ctx) :
                BaseDualState(collect(xf), r_g.θ_full, r_g.zeta, r_g.lambda, copy(ctx.obj.arg1), r_g.inner_status)
        end
        invalidated, reason = maybe_invalidate!(policy, w)
        if resolved_backend == :pooled
            gfull, meta = composite_gradient_at_fast_pooled(xf, ctx, pe, grad_pool; base = base, threaded = true,
                                                       h_mode = :cached, bandwidth_cache = policy.cache)
            record_hits!(policy, meta.cache_hits[2:end])   # pooled/A+/C+ paths have no tie_fallback branch (unlike lfix_buffer_reuse.jl's meta, which always reports tie_fallback=false anyway)
        elseif resolved_backend == :aplus
            gfull, meta = composite_gradient_at_Aplus(xf, ctx, pe, grad_pool, lfix_ws; base = base, threaded = true,
                                                       h_mode = :cached, bandwidth_cache = policy.cache)
            record_hits!(policy, meta.cache_hits[2:end])
        elseif resolved_backend == :cplus
            gfull, meta = composite_gradient_at_Cplus(xf, ctx, pe, grad_pool, lfix_c_ws; base = base, threaded = true,
                                                       h_mode = :cached, bandwidth_cache = policy.cache)
            record_hits!(policy, meta.cache_hits[2:end])
        elseif resolved_backend == :kbplus
            gfull, meta = composite_gradient_at_KBplus(xf, ctx, pe, grad_pool, lfix_kb_ws; base = base, threaded = true,
                                                       h_mode = :cached, bandwidth_cache = policy.cache)
            record_hits!(policy, meta.cache_hits[2:end])
        else
            gfull, meta = composite_gradient_at_fast_buffered(xf, ctx, pe; base = base, threaded = true,
                                                       h_mode = :cached, bandwidth_cache = policy.cache)
            meta.tie_fallback || record_hits!(policy, meta.cache_hits[2:end])
        end
        n_grad_calls[] += 1
        evalResult.objGrad .= gfull[2:end]
        return 0
    end
    # Section 6: checkpoint after every accepted KNITRO outer iterate. Cheap here (a warm
    # re-evaluation via the compressed path, plus a small serialize) relative to the gradient
    # call the SAME iterate already paid for.
    function cb_newpt!(kc2, x, lambda, user_data)
        knitro_iter[] += 1
        w_now = vcat(g, x)
        xf_now = x_free_from_w(w_now, pe)
        r_now, _ = screened_eval(xf_now, ctx, rsc, sc, n_eval; warm = true, bank = bank, zfree = collect(x), exact_cache = exact_cache)
        if r_now.inner_status in FEASIBLE_CODES && isfinite(r_now.Delta_dual)
            do_checkpoint(:iteration, w_now, r_now)
        end
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!)
    KNITRO.KN_set_newpt_callback(kc, cb_newpt!)

    KNITRO.KN_solve(kc)
    wall_ext = time() - t_start
    nStatus_code, _, xsol, _ = KNITRO.KN_get_solution(kc)
    # Native, KNITRO-reported per-solve diagnostics for THIS OUTER kc only, queried before KN_free
    # (task §9/§10) -- distinct from n_eval/n_grad_calls/sc.* below, which are this driver's own
    # hand-rolled counters accumulated across every cb_F!/cb_G! call of this one outer solve
    # (themselves genuinely per-call since sc/n_eval/n_grad_calls are all freshly constructed at
    # the top of this function, never shared across separate run_profile_checkpointed calls).
    # Note this reports the OUTER polish/profile NLP's own iteration count, NOT the INNER CC dual
    # solve's -- each screened_eval call below may trigger its own separate inner KNITRO solve with
    # its own counters, audited separately (see docs/fullA_driver_delta5_diagnostics_handoff.md §9).
    native_outer_diag = full_status_record(nStatus_code, kc)
    KNITRO.KN_free(kc)

    b = best[]
    lp("[", label, "] PROFILE DONE: status=", nStatus_code, " (", native_outer_diag.status_name, "/",
       native_outer_diag.status_category, ") wall_ext=", round(wall_ext, digits = 1),
       "s n_eval=", n_eval[], " n_grad_calls=", n_grad_calls[],
       " native_outer_iters=", native_outer_diag.n_iters, " native_outer_fc_evals=", native_outer_diag.n_fc_evals,
       " native_outer_ga_evals=", native_outer_diag.n_ga_evals,
       " screens(pw/wt/wn/env/wr/sn/pass)=", sc.pairwise, "/", sc.witness, "/", sc.winner, "/", sc.envelope, "/", sc.winning_range, "/", sc.safety_net, "/", sc.passed)
    if b !== nothing
        lp("  best: Delta=", b.Delta_dual, " gravity=", b.gravity_value, " found_at_eval=", b.n_eval)
    end
    # final stage-complete checkpoint at the terminal point
    w_final = vcat(g, collect(xsol))
    r_final, _ = screened_eval(x_free_from_w(w_final, pe), ctx, rsc, sc, n_eval; warm = true, bank = bank, zfree = w_final[2:end], exact_cache = exact_cache)
    # AUD-10 fix: the terminal evaluation is not itself gated by anything upstream (unlike
    # is_new_best, which already requires is_verified_success). Flag -- do not silently
    # checkpoint as an ordinary :stage_complete -- a terminal point that fails verification, so a
    # human/resume caller cannot mistake it for a scientifically usable stopping point. best[] (the
    # separately, correctly gated incumbent) is unaffected either way.
    if !(r_final.inner_status in FEASIBLE_CODES) || !is_verified_success(r_final)
        lp("[", label, "] WARNING: terminal point failed verification (inner_status=", r_final.inner_status,
           ", class=", classify_inner_result(r_final), ") -- checkpointing as :stage_complete_unverified, ",
           "NOT :stage_complete. best[]=", best[] === nothing ? "nothing" : "Delta=$(best[].Delta_dual)",
           " remains the correct resume/incumbent state (AUD-10).")
        final_ckpt = do_checkpoint(:stage_complete_unverified, w_final, r_final)
    else
        final_ckpt = do_checkpoint(:stage_complete, w_final, r_final)
    end

    return (label = label, g = g, find_smallest = find_smallest, ctx = ctx, pe = pe,
            knitro_status = nStatus_code, native_outer_diag = native_outer_diag, wall_ext = wall_ext,
            n_eval = n_eval[], n_grad_calls = n_grad_calls[],
            zfree_terminal = collect(xsol), best = b, trace = trace, screen_counts = as_namedtuple(sc),
            screen_rejections = sc.rejections, final_checkpoint = final_ckpt,
            ckpt_path = joinpath(ckpt_dir, "$(label)_latest.jls"))
end

# ============================================================================
# POLISH stage: joint (gamma', A) constrained solve, warm-started from a
# profile's terminal A-block. Same screening+checkpointing discipline as
# run_profile_checkpointed above; direct extension of
# c9_phase8_d20_pilot.jl::joint_polish.
# ============================================================================
function run_polish_checkpointed(label::String, find_smallest_in::Bool, g_start_in::Float64, zfree_start_in::Vector{Float64};
        maxtime_real::Float64 = 450.0, hessopt_tag::String = "sr1",
        W_in::Int = 80000, delta_in::Float64 = 1.0, draw_seed_in::Int = 20260719,
        draw_design_in::Union{Nothing,Symbol} = nothing,   # nothing = "caller did not specify" -- see
        # run_profile_checkpointed's identical parameter for why this sentinel (vs defaulting to
        # :pseudorandom) is required for the mismatch guard below to work correctly.
        ckpt_dir::AbstractString, checkpoint_interval_s::Float64 = 90.0,
        resume_from::Union{Nothing,AbstractString} = nothing,
        logio::Union{Nothing,IO} = nothing,
        inner_opt_override::Union{Nothing,AbstractString} = nothing,
        skip_cold_retry::Bool = true,   # validated 2026-07-20: cold retry rescued 0/30 warm failures (all genuine
        use_dual_bank::Bool = true, dual_bank_size::Int = 8, use_exact_cache::Bool = true,
        exact_cache_override::Union{Nothing,SafeExactCache,CrossDeltaExactCache} = nothing,   # allocation/
        # cache-cleanup task §12: when given, USE THIS cache instead of constructing a fresh
        # SafeExactCache() internally -- lets a caller (e.g. run_staged_delta5_continuation) thread
        # ONE CrossDeltaExactCache across every stage of a staged continuation so an exact hit from
        # a DIFFERENT delta-stage (same x_free/find_smallest/context) is served without a re-solve.
        # Default nothing: zero behavior change (falls back to use_exact_cache ? SafeExactCache() :
        # nothing, exactly as before this kwarg existed) -- CrossDeltaExactCache's own correctness
        # (context-fingerprinted key, AUD-08) is a precondition for safely enabling this; see
        # docs/fullA_independent_audit_remediation.md AUD-08 and docs/fullA_postmerge_allocation_
        # productionization.md.
        use_neg_cache::Bool = false, neg_cache_code_version::String = "unknown",   # negative-cache audit,
        # Policy B (docs/fullA_negative_cache_audit.md): opt-in, default OFF (byte-identical behavior to
        # before this kwarg existed when omitted). Requires skip_cold_retry=false to ever have a
        # confirmation attempt to promote from -- with skip_cold_retry=true (this function's own
        # default) there is only ever a single (warm) attempt per point, which is NEVER cached here
        # regardless of use_neg_cache (a lone failure is a TransientFailureResult by construction, see
        # negative_cache.jl -- only a CONFIRMED compatible second failure is eligible).
        reuse::Union{Nothing,NamedTuple} = nothing,   # a prior build_fullA_context(...) result (task §5) --
        # when given, SKIPS the ~65-83s d20_real_setup_design/build_pivot_elimination/
        # build_ranged_screen_context rebuild entirely and instead cheaply overrides delta on the
        # existing context (set_context_delta!, reusable_context.jl). Ignored (with a warning) when
        # resuming, since a resumed checkpoint's own W/find_smallest/draw_design/draw_seed take
        # precedence and must still be validated against whatever context is actually used.
        organic_failures::Union{Nothing,OrganicFailureCollector} = nothing,   # opt-in archive of the
        # first N genuine (post-screen, real KNITRO) inner failures (task §7); nothing (default) = no
        # capture, zero overhead beyond one is_organic_failure check per evaluation.
        use_pooled_gradient::Union{Nothing,Bool} = nothing,   # see run_profile_checkpointed's identical kwarg
        price_cache_backend::Union{Nothing,Symbol} = nothing,   # see run_profile_checkpointed's identical kwarg
        maxit_override::Union{Nothing,Int} = nothing,   # see run_profile_checkpointed's identical kwarg
        allow_direction_box_migration::Bool = false,   # addendum: by default, a start point (fresh or
        # resumed) on the wrong side of the Frechet benchmark for its own direction is REJECTED with a
        # hard error, not silently clamped. Set true only for an explicit, deliberate migration of a
        # pre-fix checkpoint/start point -- see direction_bounds.jl.
        full_trace_ref::Union{Nothing,Ref{Vector{NamedTuple}}} = nothing)   # remediation task Part B:
        # instrumentation for the corrected Phase-1 directional diagnostic (c24_phase1_directional_broad.jl):
        # when given, every cb_F! eval additionally pushes (w=copy(w), accepted=is_new_best) into this
        # Ref'd vector so a caller can reconstruct full outer-loop step DIRECTIONS (not just the scalar
        # gp the existing `trace` NamedTuple records) from a real short production trajectory. Purely
        # additive -- nothing is read here unless a caller explicitly passes a Ref; zero behavior/
        # allocation change otherwise.
    # primal infeasibility, confirmed by clean KNITRO -300/unbounded status on both attempts) while costing an
    # extra ~13.5s per rejected point; skipping it is a pure win (identical kappa reached in matched A/B tests,
    # ~1.4x more outer attempts explored per unit time). See docs handoff for the full investigation.
    lp(xs...) = (println(xs...); logio !== nothing && (println(logio, xs...); flush(logio)); flush(stdout))

    mkpath(ckpt_dir)
    resumed = resume_from === nothing ? nothing : load_checkpoint(resume_from)
    if resumed !== nothing && resumed.checkpoint_reason == :stage_complete_unverified
        println("WARNING: resuming from a checkpoint whose terminal point FAILED verification at write ",
                "time (checkpoint_reason=:stage_complete_unverified, AUD-10) -- best[]/best_feasible[] ",
                "inside this checkpoint is still the correct scientific incumbent; only the raw terminal ",
                "solver-state fields are suspect.")
    end
    find_smallest = find_smallest_in; g_start = g_start_in; zfree_start = copy(zfree_start_in)
    W = W_in; delta = delta_in; draw_seed = draw_seed_in
    draw_design = draw_design_in === nothing ? :pseudorandom : draw_design_in
    bandwidth_cache = Dict{Int,Float64}()
    if resumed !== nothing
        find_smallest = resumed.find_smallest; g_start = resumed.g; zfree_start = copy(resumed.zfree)
        W = resumed.W; delta = resumed.delta; draw_seed = resumed.draw_seed
        if draw_design_in !== nothing && draw_design_in != resumed.draw_design
            error("run_polish_checkpointed($label): resume draw_design mismatch -- checkpoint has " *
                  ":$(resumed.draw_design), caller requested :$(draw_design_in). Refusing to resume.")
        end
        draw_design = resumed.draw_design
        bandwidth_cache = copy(resumed.bandwidth_cache)
        lp("[", label, "] RESUMING from ", resume_from, " (reason=", resumed.checkpoint_reason,
           " n_eval=", resumed.n_eval, " knitro_iter=", resumed.knitro_iter, ")")
    end

    ctx_reused = false
    if reuse !== nothing && resumed === nothing
        if reuse_matches(reuse; W = W, find_smallest = find_smallest, draw_design = draw_design, draw_seed = draw_seed)
            ctx = set_context_delta!(reuse.ctx, delta)
            pe = reuse.pe; rsc = reuse.rsc
            ctx_reused = true
            lp("[", label, "] REUSING context (task §5): delta overridden to ", delta,
               " -- skipped the ~65-83s d20_real_setup_design/pe/rsc rebuild")
        else
            lp("[", label, "] WARNING: reuse= context does not match requested W/find_smallest/draw_design/",
               "draw_seed -- falling back to a fresh build (this should not happen if the caller threads a ",
               "single build_fullA_context(...) result through matching stages; see task §5)")
        end
    elseif reuse !== nothing && resumed !== nothing
        lp("[", label, "] NOTE: reuse= given but also resuming from a checkpoint -- resumed W/find_smallest/",
           "draw_design/draw_seed take precedence; reuse= is ignored this call (rebuild from scratch, matching ",
           "the checkpoint's own recorded provenance) rather than risk silently reusing a mismatched context.")
    end
    if !ctx_reused
        ctx = inner_opt_override === nothing ?
            d20_real_setup_design(W = W, δ = delta, find_smallest = find_smallest, draw_design = draw_design, draw_seed = draw_seed) :
            d20_real_setup_design(W = W, δ = delta, find_smallest = find_smallest, draw_design = draw_design, draw_seed = draw_seed, inner_loop_opt = inner_opt_override)
        pe = build_pivot_elimination(ctx)
        rsc = build_ranged_screen_context(ctx)
    end
    D = ctx.D; D2 = D^2
    resolved_backend = resolve_price_cache_backend(label, use_pooled_gradient, price_cache_backend)
    # one pool/workspace per ctx (built ONCE here), not per gradient call -- see docs/
    # fullA_factorized_price_production_gate.md for A+/C+'s own persistence/aliasing design.
    grad_pool = resolved_backend in (:pooled, :aplus, :cplus, :kbplus) ? build_grad_workspace_pool(W) : nothing
    lfix_ws = resolved_backend == :aplus ? build_lfix_base_workspace(D, W) : nothing
    lfix_c_ws = resolved_backend == :cplus ? build_lfix_factorized_workspace(D, W) : nothing
    lfix_kb_ws = resolved_backend == :kbplus ? build_lfix_kbplus_workspace(D, W) : nothing
    lp("[", label, "] ctx built, D=", D, " W=", W, " draw_seed=", draw_seed, " draw_design=", draw_design,
       " price_cache_backend=", resolved_backend,
       " draw_checksum=(", ctx.draw_meta.checksum_uniform, ",", ctx.draw_meta.checksum_transformed, ")",
       " screen_setup_wall=", ctx.screen_setup_wall,
       " inner_opt=", inner_opt_override === nothing ? "default" : inner_opt_override,
       " envelope_screen_supported=", rsc.envelope !== nothing,
       rsc.envelope === nothing ? " (reason: $(rsc.unsupported_reason))" : "")

    w0 = vcat(g_start, zfree_start)
    # Direction-aware gp box (addendum, "correct the outer gamma bounds for upper and
    # lower runs"): reject -- do not clamp -- a start point (fresh or resumed) that lies
    # on the wrong side of the Frechet benchmark for its own declared direction. See
    # direction_bounds.jl for the full derivation/evidence.
    if !allow_direction_box_migration
        validate_gp_in_direction_box(w0[1], ctx, find_smallest; label = label,
            what = resumed !== nothing ? "resumed checkpoint's (g, zfree)" : "supplied start point")
    end
    if resumed !== nothing
        if ctx.draw_meta.checksum_uniform != resumed.draw_checksum_uniform ||
           ctx.draw_meta.checksum_transformed != resumed.draw_checksum_transformed
            error("run_polish_checkpointed($label): regenerated draws do not match checkpoint's recorded " *
                  "checksums -- checkpoint (uniform=$(resumed.draw_checksum_uniform), " *
                  "transformed=$(resumed.draw_checksum_transformed)) vs regenerated " *
                  "(uniform=$(ctx.draw_meta.checksum_uniform), transformed=$(ctx.draw_meta.checksum_transformed)). " *
                  "Refusing to resume with mismatched draws.")
        end
        ctx.obj.x .= resumed.dual_warm_start
        r_verify, _ = evaluate_fullA_screened_ranged(x_free_from_w(w0, pe), ctx, rsc; moment_representation = :compressed,
            cache = nothing, use_cache = false, warm = true, pairwise = ctx.pairwise, witness = ctx.witness,
            use_witness = ctx.witness !== nothing)
        d_delta = abs(r_verify.Delta_dual - resumed.verify_Delta_dual)
        d_grav = abs(r_verify.gravity_value - resumed.verify_gravity_value)
        d_kkt = abs(r_verify.max_abs_moment_kkt_resid - resumed.verify_max_abs_moment_kkt_resid)
        d_mr = abs(norm(r_verify.benchmark_unweighted_moment_mean) - resumed.verify_moment_resid_norm)
        lp("[", label, "] RESUME VALIDATION: |ΔDelta_dual|=", d_delta,
           " |Δgravity_value|=", d_grav, " |Δmax_abs_moment_kkt_resid|=", d_kkt,
           " |Δ||benchmark_unweighted_moment_mean|||=", d_mr)
        check_resume_tolerances!(label, "run_polish_checkpointed", d_delta, d_grav, d_kkt, d_mr)   # AUD-09
    end

    sc = ScreenCounters()
    n_eval = Ref(resumed !== nothing ? resumed.n_eval : 0)
    bank = use_dual_bank ? DualBank(dual_bank_size) : nothing
    exact_cache = exact_cache_override !== nothing ? exact_cache_override : (use_exact_cache ? SafeExactCache() : nothing)
    neg_cache = use_neg_cache ? SafeNegativeCache{FullAEvalKey}() : nothing
    n_neg_confirmed = Ref(0)
    r0, _ = screened_eval(x_free_from_w(w0, pe), ctx, rsc, sc, n_eval; warm = false)
    lp("[", label, "] polish start point: inner_status=", r0.inner_status, " Delta=", r0.Delta_dual)
    r0.inner_status in FEASIBLE_CODES || error("run_polish_checkpointed($label): start point not inner-feasible, cannot proceed")

    z_halfwidth = 30.0
    # Direction-aware gp box, split at the Frechet benchmark (addendum) -- replaces the old
    # full-range [ctx.bounds.γp_lo, ctx.bounds.γp_hi] box, which let the "upper" and "lower"
    # searches cross into each other's territory. See direction_bounds.jl.
    gp_dir_lo, gp_dir_hi = direction_gamma_bounds(ctx, find_smallest)
    w_lo = vcat(gp_dir_lo, zfree_start .- z_halfwidth)
    w_hi = vcat(gp_dir_hi, zfree_start .+ z_halfwidth)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt"))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", maxit_override === nothing ? 1_000_000 : maxit_override)
    xIndices = KNITRO.KN_add_vars(kc, D2)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w0)
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], ctx.δ)

    last_F_state = Ref{Union{Nothing,NamedTuple}}(nothing)
    # See incumbent_logic.jl / docs/fullA_driver_delta5_diagnostics_handoff.md §3: seed the
    # incumbent from the already-cold-verified start point (r0), gated on the SAME delta-budget
    # feasibility test (`feasible = Δ <= ctx.δ + 1e-6`) cb_F! uses below, rather than `nothing`.
    seed_cand_feasible = r0.inner_status in FEASIBLE_CODES && isfinite(r0.Delta_dual) && r0.Delta_dual <= ctx.δ + 1e-6
    seed_cand = (gp = w0[1], w = copy(w0), Delta = r0.Delta_dual, gravity = r0.gravity_value,
                 kkt = r0.max_abs_moment_kkt_resid, inner_status = r0.inner_status, t_elapsed = 0.0,
                 n_eval = n_eval[])
    best_feasible = Ref{Any}(seed_incumbent(resumed !== nothing ? resumed.best_feasible : nothing,
                                             seed_cand_feasible, seed_cand))
    n_grad_calls = Ref(0)
    policy = BandwidthCachePolicy(); policy.cache = bandwidth_cache
    trace = NamedTuple[]
    t_start = time(); last_ckpt_wall = Ref(time())
    knitro_iter = Ref(resumed !== nothing ? resumed.knitro_iter : 0)
    run_id = "c10_prod_$(label)_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"

    function do_checkpoint(reason::Symbol, w_current::Vector{Float64}, r::NamedTuple)
        zfree_now = w_current[2:end]
        logA_full = pivot_expand(zfree_now, pe)
        ckpt = D20Checkpoint(CHECKPOINT_SCHEMA, run_id, label, find_smallest ? :upper : :lower, find_smallest, delta, W, draw_seed,
            w_current[1], copy(zfree_now), logA_full, copy(ctx.obj.x), copy(policy.cache),
            best_feasible[], n_eval[], knitro_iter[], time() - t_start, reason, as_namedtuple(sc),
            r.Delta_dual, r.gravity_value, r.max_abs_moment_kkt_resid, norm(r.benchmark_unweighted_moment_mean), SOLVER_STATE_NOTE,
            draw_design, ctx.draw_meta.checksum_uniform, ctx.draw_meta.checksum_transformed,
            LOADED_KNITRO_RELEASE)
        latest_path = joinpath(ckpt_dir, "$(label)_latest.jls")
        guard_checkpoint_path(latest_path, draw_design, ctx.draw_meta.checksum_uniform, ctx.draw_meta.checksum_transformed)
        save_checkpoint(latest_path, ckpt)
        reason in (:new_best, :stage_complete, :stage_complete_unverified) && save_checkpoint(joinpath(ckpt_dir, "$(label)_$(reason)_neval$(n_eval[]).jls"), ckpt)
        return ckpt
    end

    n_cold_retries = Ref(0); n_rejected = Ref(0)
    warm_cold_trace = NamedTuple[]   # additive diagnostic: filled only on warm-attempt failure (rare), read by caller after KN_solve
    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w, pe)
        dual_before = copy(ctx.obj.x)   # snapshot for organic-failure capture (task §7), before screened_eval mutates it
        t_warm0 = time()
        r_warm_result, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = true, bank = bank, zfree = w[2:end], exact_cache = exact_cache, neg_cache = neg_cache)
        r = r_warm_result
        t_warm = time() - t_warm0
        cold_time = NaN; cold_status = missing
        if !(r.inner_status in FEASIBLE_CODES)
            n_cold_retries[] += 1
            warm_status = r.inner_status
            if !skip_cold_retry
                t_cold0 = time()
                # BUGFIX (negative-cache audit): this retry previously omitted `exact_cache=exact_cache`,
                # meaning a cold-retry SUCCESS (a real, feasible answer) was silently never written to the
                # positive exact-point cache -- every future revisit of the same point re-paid the full
                # cold-solve cost even though the point itself is genuinely feasible and cacheable per
                # oracle.jl's own `is_cacheable_result`. Fixed by passing it through, matching every other
                # screened_eval call site in this file.
                r, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = false, exact_cache = exact_cache)
                cold_time = time() - t_cold0
                cold_status = r.inner_status
                # Negative-cache audit (Policy B, opt-in via use_neg_cache=): a confirmed compatible
                # failure (both warm and cold attempts are the same unbounded-family KNITRO code) is
                # promoted to the negative cache here -- this is the ONLY place in this driver a negative
                # entry is ever written, and only after two materially different starts (warm-started from
                # whatever ctx.obj.x/bank held, vs a genuinely cold zeros start) agree.
                if neg_cache !== nothing && !(cold_status in FEASIBLE_CODES) && compatible_failure(warm_status, cold_status)
                    key = FullAEvalKey(collect(xf), ctx.obj.δ, ctx.obj.find_smallest, ctx.obj.inner_loop_opt, :hard, context_fingerprint(ctx))
                    entry = ConfirmedNegativeResult(warm_status, :warm_production_slot, cold_status, :cold_neutral,
                        (first_Delta = NaN, confirm_Delta = NaN, warm_time = t_warm, cold_time = cold_time),
                        neg_cache_code_version, now())
                    negcache_store!(neg_cache, key, entry)
                    n_neg_confirmed[] += 1
                end
            end
            push!(warm_cold_trace, (n_eval = n_eval[], gp = w[1], warm_time = t_warm, warm_status = warm_status,
                                     cold_time = cold_time, cold_status = cold_status,
                                     rescued = !skip_cold_retry && r.inner_status in FEASIBLE_CODES && isfinite(r.Delta_dual)))
        end
        if organic_failures !== nothing
            maybe_capture_organic_failure!(organic_failures, label, w, xf, r, ctx, pe, sc, n_eval[], knitro_iter[],
                delta, find_smallest, draw_design, draw_seed, skip_cold_retry ? :warm_only : :warm_then_cold,
                dual_before, resume_from)
        end
        if !(r.inner_status in FEASIBLE_CODES) || !isfinite(r.Delta_dual)
            n_rejected[] += 1
            reject_point(w[1], "run_polish_checkpointed($label): infeasible/non-finite point (inner_status=$(r.inner_status)), rejecting")
        end
        Δ = r.Delta_dual
        evalResult.obj[1] = find_smallest ? w[1] : -w[1]
        evalResult.c[1] = Δ
        n_eval[] += 1
        t_el = time() - t_start
        feasible = Δ <= ctx.δ + 1e-6
        # AUD-03 fix: on a cache hit, screened_eval never re-solved, so ctx.obj.arg1 (m_star) can
        # still hold a DIFFERENT point's state (whatever the last real inner solve left behind).
        # solve_base_state always performs a fresh solve, so it is the only way to guarantee
        # correct m_star here; the non-cache-hit branch keeps the fast hand-built path since
        # ctx.obj.arg1 genuinely is fresh for that case (populated by the primal_weight_recovery
        # call inside screened_eval moments ago). See docs/fullA_independent_audit_remediation.md
        # AUD-03 and its A/B/A regression test.
        base = r.cache_hit ? solve_base_state(xf, ctx) :
            BaseDualState(collect(xf), r.θ_full, r.zeta, r.lambda, copy(ctx.obj.arg1), r.inner_status)
        last_F_state[] = (w = copy(w), base = base)
        # AUD-04 fix: same reasoning as cb_F! above -- feasibility (Delta<=delta) alone is not a
        # verified solve.
        is_new_best = feasible && is_verified_success(r) &&
            is_better_polish(w[1], best_feasible[] === nothing ? nothing : best_feasible[].gp, find_smallest)
        if is_new_best
            best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ, gravity = r.gravity_value,
                                kkt = r.max_abs_moment_kkt_resid, inner_status = r.inner_status,
                                t_elapsed = t_el, n_eval = n_eval[])
        end
        push!(trace, (idx = n_eval[], t_elapsed = t_el, gp = w[1], Delta_dual = Δ, inner_status = r.inner_status, feasible = feasible))
        full_trace_ref !== nothing && push!(full_trace_ref[], (idx = n_eval[], w = copy(w), accepted = is_new_best, feasible = feasible))
        if n_eval[] <= 5 || n_eval[] % 10 == 0
            lp("  [", label, "] eval ", n_eval[], " t=", round(t_el, digits = 1), "s gp=", w[1], " Delta=", Δ,
               " screens(pw/wt/wn/env/wr/sn/pass)=", sc.pairwise, "/", sc.witness, "/", sc.winner, "/", sc.envelope, "/", sc.winning_range, "/", sc.safety_net, "/", sc.passed)
        end
        if is_new_best
            do_checkpoint(:new_best, w, r)
        end
        if time() - last_ckpt_wall[] > checkpoint_interval_s
            do_checkpoint(:wall_interval, w, r)
            last_ckpt_wall[] = time()
        end
        return 0
    end
    n_g_recompute = Ref(0)
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w, pe)
        shared = last_F_state[]
        base = shared !== nothing && shared.w == w ? shared.base : nothing
        if base === nothing
            n_g_recompute[] += 1
            r_g, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = true, bank = bank, zfree = w[2:end], exact_cache = exact_cache, neg_cache = neg_cache)
            g_warm_status = r_g.inner_status
            if !(r_g.inner_status in FEASIBLE_CODES)
                # BUGFIX (negative-cache audit, same as cb_F! above): pass exact_cache through so a
                # cold-retry success here is cacheable too, not silently recomputed every time.
                r_g, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = false, exact_cache = exact_cache)
                if neg_cache !== nothing && !(r_g.inner_status in FEASIBLE_CODES) && compatible_failure(g_warm_status, r_g.inner_status)
                    key = FullAEvalKey(collect(xf), ctx.obj.δ, ctx.obj.find_smallest, ctx.obj.inner_loop_opt, :hard, context_fingerprint(ctx))
                    entry = ConfirmedNegativeResult(g_warm_status, :warm_production_slot, r_g.inner_status, :cold_neutral,
                        (first_Delta = NaN, confirm_Delta = NaN, warm_time = NaN, cold_time = NaN),
                        neg_cache_code_version, now())
                    negcache_store!(neg_cache, key, entry)
                    n_neg_confirmed[] += 1
                end
            end
            r_g.inner_status in FEASIBLE_CODES || reject_point(w[1], "run_polish_checkpointed($label): cb_G! could not recompute a feasible base state")
            # AUD-03 fix: same reasoning as cb_F! above -- do not trust ctx.obj.arg1 on a cache hit.
            base = r_g.cache_hit ? solve_base_state(xf, ctx) :
                BaseDualState(collect(xf), r_g.θ_full, r_g.zeta, r_g.lambda, copy(ctx.obj.arg1), r_g.inner_status)
        end
        invalidated, reason = maybe_invalidate!(policy, w)
        if resolved_backend == :pooled
            gfull, meta = composite_gradient_at_fast_pooled(xf, ctx, pe, grad_pool; base = base, threaded = true,
                                                       h_mode = :cached, bandwidth_cache = policy.cache)
            record_hits!(policy, meta.cache_hits[2:end])   # pooled/A+/C+ paths have no tie_fallback branch (unlike lfix_buffer_reuse.jl's meta, which always reports tie_fallback=false anyway)
        elseif resolved_backend == :aplus
            gfull, meta = composite_gradient_at_Aplus(xf, ctx, pe, grad_pool, lfix_ws; base = base, threaded = true,
                                                       h_mode = :cached, bandwidth_cache = policy.cache)
            record_hits!(policy, meta.cache_hits[2:end])
        elseif resolved_backend == :cplus
            gfull, meta = composite_gradient_at_Cplus(xf, ctx, pe, grad_pool, lfix_c_ws; base = base, threaded = true,
                                                       h_mode = :cached, bandwidth_cache = policy.cache)
            record_hits!(policy, meta.cache_hits[2:end])
        elseif resolved_backend == :kbplus
            gfull, meta = composite_gradient_at_KBplus(xf, ctx, pe, grad_pool, lfix_kb_ws; base = base, threaded = true,
                                                       h_mode = :cached, bandwidth_cache = policy.cache)
            record_hits!(policy, meta.cache_hits[2:end])
        else
            gfull, meta = composite_gradient_at_fast_buffered(xf, ctx, pe; base = base, threaded = true,
                                                       h_mode = :cached, bandwidth_cache = policy.cache)
            meta.tie_fallback || record_hits!(policy, meta.cache_hits[2:end])
        end
        n_grad_calls[] += 1
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = find_smallest ? 1.0 : -1.0
        evalResult.jac .= gfull
        return 0
    end
    function cb_newpt!(kc2, x, lambda, user_data)
        knitro_iter[] += 1
        xf_now = x_free_from_w(x, pe)
        r_now, _ = screened_eval(xf_now, ctx, rsc, sc, n_eval; warm = true, bank = bank, zfree = collect(x[2:end]), exact_cache = exact_cache)
        if r_now.inner_status in FEASIBLE_CODES && isfinite(r_now.Delta_dual)
            do_checkpoint(:iteration, collect(x), r_now)
        end
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)
    KNITRO.KN_set_newpt_callback(kc, cb_newpt!)

    KNITRO.KN_solve(kc)
    wall_ext = time() - t_start
    nStatus_code, _, xsol, _ = KNITRO.KN_get_solution(kc)
    # See the matching comment in run_profile_checkpointed above (§9/§10): native per-solve
    # diagnostics for THIS OUTER kc, queried before KN_free.
    native_outer_diag = full_status_record(nStatus_code, kc)
    KNITRO.KN_free(kc)

    b = best_feasible[]
    σ = ctx.σ
    κ = NaN
    if b !== nothing
        κ = 1 - b.gp^(σ / (σ - 1))
    end
    lp("[", label, "] POLISH DONE: status=", nStatus_code, " (", native_outer_diag.status_name, "/",
       native_outer_diag.status_category, ") wall_ext=", round(wall_ext, digits = 1),
       "s n_eval=", n_eval[], " kappa=", κ,
       " native_outer_iters=", native_outer_diag.n_iters, " native_outer_fc_evals=", native_outer_diag.n_fc_evals,
       " native_outer_ga_evals=", native_outer_diag.n_ga_evals,
       " screens(pw/wt/wn/env/wr/sn/pass)=", sc.pairwise, "/", sc.witness, "/", sc.winner, "/", sc.envelope, "/", sc.winning_range, "/", sc.safety_net, "/", sc.passed,
       " n_cold_retries(F)=", n_cold_retries[], " n_rejected(F)=", n_rejected[], " n_g_recompute=", n_g_recompute[],
       neg_cache !== nothing ? " n_neg_confirmed=$(n_neg_confirmed[]) neg_cache_size=$(length(neg_cache))" : "")

    w_final = collect(xsol)
    r_final, _ = screened_eval(x_free_from_w(w_final, pe), ctx, rsc, sc, n_eval; warm = true, bank = bank, zfree = w_final[2:end], exact_cache = exact_cache)
    # AUD-10 fix: see run_profile_checkpointed's identical fix above for the full rationale.
    if !(r_final.inner_status in FEASIBLE_CODES) || !is_verified_success(r_final)
        lp("[", label, "] WARNING: terminal point failed verification (inner_status=", r_final.inner_status,
           ", class=", classify_inner_result(r_final), ") -- checkpointing as :stage_complete_unverified, ",
           "NOT :stage_complete. best_feasible[]=", best_feasible[] === nothing ? "nothing" : "gp=$(best_feasible[].gp) Delta=$(best_feasible[].Delta)",
           " remains the correct resume/incumbent state (AUD-10).")
        final_ckpt = do_checkpoint(:stage_complete_unverified, w_final, r_final)
    else
        final_ckpt = do_checkpoint(:stage_complete, w_final, r_final)
    end

    return (label = label, find_smallest = find_smallest, ctx = ctx, pe = pe,
            knitro_status = nStatus_code, native_outer_diag = native_outer_diag, wall_ext = wall_ext,
            n_eval = n_eval[], n_grad_calls = n_grad_calls[],
            best_feasible = b, kappa = κ, trace = trace, screen_counts = as_namedtuple(sc),
            screen_rejections = sc.rejections, final_checkpoint = final_ckpt,
            n_cold_retries = n_cold_retries[], n_rejected = n_rejected[], n_g_recompute = n_g_recompute[],
            warm_cold_trace = warm_cold_trace,
            n_neg_confirmed = n_neg_confirmed[], neg_cache_size = neg_cache !== nothing ? length(neg_cache) : 0,
            ckpt_path = joinpath(ckpt_dir, "$(label)_latest.jls"))
end
