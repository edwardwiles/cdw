# ============================================================================
# CM production checkpoint schema (allocation/cache cleanup task, §11).
#
# GAP (confirmed by direct code reading): `run_cm_upper` (cm_outer_driver.jl) has NO
# checkpoint/resume support at all -- it is a single-shot KNITRO call with no periodic
# save and no resume path. The one place CM runs ARE checkpointed today
# (`c13_d20_cm_upper_continuation.jl`'s `save_stage`) writes a bare, ad-hoc NamedTuple
# (`L, probs, kappa, knitro_status, wall, n_eval, n_grad, best_w, best_Delta, xsol,
# timestamp`) with NO schema version, NO draw-design/checksum validation, NO KNITRO-version
# field, and no corresponding load/resume logic whatsoever -- it is write-only, used only to
# hand `best_w` forward into the NEXT (coarser->finer) grid stage, never to resume an
# INTERRUPTED run of the SAME stage. This is a materially weaker guarantee than the
# unrestricted path's schema-3 `D20Checkpoint` (c10_d20_production_driver.jl), which
# validates draw checksums, KNITRO version, and refuses to resume an incompatible config.
#
# This file adds `CMCheckpoint` (same rigor as `D20Checkpoint`, plus the CM-specific fields
# the brief lists: grid size, exact cutpoints, contrasts/basis, Hessian backend) and
# `run_cm_upper_checkpointed`, a NEW wrapper (additive -- `run_cm_upper`/`cm_outer_driver.jl`
# are UNCHANGED) that periodically checkpoints the best exact-feasible incumbent (matching
# `run_polish_checkpointed`'s own `:new_best`/`:wall_interval`/`:stage_complete` discipline)
# and can resume from a prior `CMCheckpoint`, rebuilding `ctx`/`pcx` from the checkpoint's own
# recorded (W, δ, draw_design, draw_seed, L, contrasts, probs) and hard-refusing resume if the
# regenerated draw checksums don't match (same guarantee schema-3 gives the unrestricted path).
#
# Verified in test_cm_checkpoint.jl: an interrupted-then-resumed run (fresh process) reproduces
# the same best-feasible incumbent's Delta/kappa to bit-for-bit agreement.
# ============================================================================
using Serialization, Dates

const CM_CHECKPOINT_SCHEMA = 3
# Bumped 1 -> 2 (remediation task Part A, finding F1): schema-1 checkpoints computed cb_F!'s
# reported/constrained Delta as `-base.ζstar`, which silently omits mean(Psi(q*)) and overstates
# the divergence at tail-active points (any draw with recovered weight m* > e). A schema-1
# checkpoint's `best_feasible.Delta` and `feasible` flags are NOT trustworthy -- do not resume
# from one. Use `migrate_cm_checkpoint_v1_candidate` to recover just the incumbent vector as a
# fresh start point, then cold-re-evaluate it with `cm_production_value_verified` before trusting
# any Delta/feasibility for it.
#
# Bumped 2 -> 3 (Part II.4 follow-up, 2026-07-23): adds `cm_gradient_backend` to the persisted
# schema (see `CMCheckpointV3`'s own field comment). IMPORTANT Julia-Serialization gotcha,
# discovered live this session (not assumed): `Serialization` resolves a struct field-for-field by
# the TYPE NAME recorded inside the file, looked up in the CURRENT session -- it is NOT safe to
# just add a field to the EXISTING `CMCheckpoint` struct under the same name, because every old
# file's embedded type reference (`Main.CMCheckpoint`) would then resolve to the NEW, longer
# layout and misread the byte stream (confirmed empirically: this throws `EOFError`, not a clean/
# catchable type error, both for `deserialize(path)::CMCheckpoint` AND for an attempted read into
# a DIFFERENTLY-NAMED struct with the old layout -- the type lookup happens via the name IN THE
# FILE, not the annotation at the call site). The correct fix is to version the TYPE NAME: leave
# `CMCheckpoint` (below) permanently unchanged as the schema-1/2 legacy layout -- every existing
# checkpoint, including the entire completed `production_runs/cm_campaign_2026-07-22`, remains
# loadable through it forever -- and introduce `CMCheckpointV3` as a distinct type for schema>=3.
# `load_cm_checkpoint` tries `CMCheckpointV3` first, falls back to `CMCheckpoint`, and upgrades.
# Verified against a real copy of chain1/delta_0.1/stage_latest.jls, not merely asserted.

"""
    CMCheckpoint

Legacy (schema 1/2) checkpoint layout. Kept PERMANENTLY UNCHANGED for backward-compat reads only
-- every schema-1/2 file ever written (including the entire 2026-07-22 real production campaign)
has this exact type name+layout embedded in its own serialized bytes, and Julia's `Serialization`
looks up structs by that embedded name, not by whatever the current top-of-tree definition is (see
the `CM_CHECKPOINT_SCHEMA` comment above for how this was discovered). `save_cm_checkpoint` never
constructs this type -- new checkpoints are always `CMCheckpointV3`.
"""
struct CMCheckpoint
    schema::Int
    run_id::String
    label::String
    branch::Symbol                 # :cm_upper (only direction wired today; kept for parity/future :cm_lower)
    find_smallest::Bool
    delta::Float64
    W::Int
    draw_seed::Int
    draw_design::Symbol
    draw_checksum_uniform::String
    draw_checksum_transformed::String
    # ---- CM configuration (task §11's explicit field list) ----
    cm_L::Int
    cm_probs::Vector{Float64}       # exact cutpoints -- NOT re-derived from L on resume, taken verbatim
    cm_contrasts::Symbol
    cm_grid_rule::Symbol            # :equal | :nested_family -- metadata only (probs is authoritative)
    cm_basis::Symbol                # :cumulative | :interval
    cm_hessian_backend::Symbol      # :structured | :dense_reference
    # ---- outer-point / incumbent state ----
    g::Float64
    zfree::Vector{Float64}
    logA_full::Matrix{Float64}
    dual_warm_start::Vector{Float64}
    bandwidth_cache::Dict{Int,Float64}
    best_feasible::Any              # NamedTuple or nothing
    n_eval::Int
    n_grad::Int
    wall_elapsed::Float64
    wall_budget_remaining::Float64
    checkpoint_reason::Symbol       # :new_best | :wall_interval | :stage_complete | :stage_complete_unverified
    knitro_version::String
end

"""
    CMCheckpointV3

CM-production checkpoint layout, schema>=3 (Part II.4 follow-up, 2026-07-23). Identical to
`CMCheckpoint` except for one new field, `cm_gradient_backend` (see below) -- given a NEW type
name specifically because Julia's `Serialization` cannot safely add a field to an existing struct
name (see `CM_CHECKPOINT_SCHEMA`'s comment). This is the type `save_cm_checkpoint` always
constructs going forward; `CMCheckpoint` (above) is retained only for reading old files.
"""
struct CMCheckpointV3
    schema::Int
    run_id::String
    label::String
    branch::Symbol
    find_smallest::Bool
    delta::Float64
    W::Int
    draw_seed::Int
    draw_design::Symbol
    draw_checksum_uniform::String
    draw_checksum_transformed::String
    cm_L::Int
    cm_probs::Vector{Float64}
    cm_contrasts::Symbol
    cm_grid_rule::Symbol
    cm_basis::Symbol
    cm_hessian_backend::Symbol
    cm_gradient_backend::Symbol     # :reference | :cplus. Previously recorded ONLY in an
                                     # unversioned sidecar (<label>_gradient_backend.txt), never
                                     # validated on resume -- a checkpoint file alone could not
                                     # reveal which backend produced it. Now part of the schema.
    g::Float64
    zfree::Vector{Float64}
    logA_full::Matrix{Float64}
    dual_warm_start::Vector{Float64}
    bandwidth_cache::Dict{Int,Float64}
    best_feasible::Any
    n_eval::Int
    n_grad::Int
    wall_elapsed::Float64
    wall_budget_remaining::Float64
    checkpoint_reason::Symbol
    knitro_version::String
end

"Atomic-ish checkpoint write, same discipline as `save_checkpoint` (D20Checkpoint): serialize to a .tmp file then mv, so a crash mid-write never leaves a half-written checkpoint."
function save_cm_checkpoint(path::AbstractString, ckpt::CMCheckpointV3)
    tmp = path * ".tmp"
    serialize(tmp, ckpt)
    mv(tmp, path; force = true)
    return path
end

"Upgrades a legacy schema-2 `CMCheckpoint` to `CMCheckpointV3`, filling `cm_gradient_backend = :reference` -- CORRECT (not a guess) for every schema-2 file that exists, since :reference was the kwarg's own default throughout schema-2's entire lifetime and the only value the real 2026-07-22 campaign's stage runner ever passed (confirmed by direct read of cm_production_stage_runner.jl, which never sets cm_gradient_backend)."
function upgrade_schema2(old::CMCheckpoint)
    return CMCheckpointV3(old.schema, old.run_id, old.label, old.branch, old.find_smallest, old.delta,
        old.W, old.draw_seed, old.draw_design, old.draw_checksum_uniform, old.draw_checksum_transformed,
        old.cm_L, old.cm_probs, old.cm_contrasts, old.cm_grid_rule, old.cm_basis, old.cm_hessian_backend,
        :reference,   # cm_gradient_backend -- implicit, correct default for schema 2
        old.g, old.zfree, old.logA_full, old.dual_warm_start, old.bandwidth_cache, old.best_feasible,
        old.n_eval, old.n_grad, old.wall_elapsed, old.wall_budget_remaining, old.checkpoint_reason,
        old.knitro_version)
end

"""
    load_cm_checkpoint(path) -> CMCheckpointV3

Tries the CURRENT (schema>=3, `CMCheckpointV3`) shape first; falls back to the legacy
`CMCheckpoint` shape (schema 1/2, the ONLY prior layout that ever existed) and upgrades it via
`upgrade_schema2`. Schema-1 files are still hard-refused below (semantically untrustworthy Delta,
unrelated to the schema-3 layout change) -- this fallback only concerns the byte LAYOUT, not
schema-1's own known defect. Always returns a `CMCheckpointV3` (uniform shape for every caller
downstream of this function, regardless of which schema the file on disk actually is).
"""
function load_cm_checkpoint(path::AbstractString)
    ckpt = try
        deserialize(path)::CMCheckpointV3
    catch e
        (e isa TypeError || e isa EOFError || e isa MethodError) || rethrow()
        local old
        try
            old = deserialize(path)::CMCheckpoint
        catch
            error("load_cm_checkpoint($path): failed to deserialize under BOTH the current " *
                  "CMCheckpointV3 layout and the legacy CMCheckpoint (schema 1/2) layout -- this " *
                  "file is not a recognized CM checkpoint (corrupt, truncated, or an even " *
                  "older/unrelated format).")
        end
        upgrade_schema2(old)
    end
    if ckpt.schema == 1
        error("load_cm_checkpoint($path): schema=1, expected $(CM_CHECKPOINT_SCHEMA) -- schema-1 " *
              "checkpoints stored Delta as `-zeta_star` (remediation task Part A, finding F1), NOT the " *
              "canonical Delta_dual = -(mean(Psi(q*))+zeta*); their best_feasible.Delta/feasible flags " *
              "are NOT trustworthy and must not be resumed from directly. Call " *
              "`migrate_cm_checkpoint_v1_candidate($path)` to recover the incumbent w-vector as a fresh " *
              "START POINT only, then cold-re-evaluate it with cm_production_value_verified before " *
              "trusting any Delta/feasibility for it.")
    end
    ckpt.schema in (2, CM_CHECKPOINT_SCHEMA) ||
        error("load_cm_checkpoint($path): schema=$(ckpt.schema), expected 2 or $(CM_CHECKPOINT_SCHEMA) -- " *
              "this checkpoint predates the CM checkpoint-schema unification (task §11), e.g. a bare " *
              "ad-hoc NamedTuple from c13_d20_cm_upper_continuation.jl's old save_stage. Start a fresh " *
              "run instead of resuming from an incompatible checkpoint.")
    return ckpt
end

"""
    migrate_cm_checkpoint_v1_candidate(path) -> (w_candidate, provenance)

Recover ONLY the incumbent w-vector (`g`, `zfree`) from a pre-remediation schema-1 `CMCheckpoint`
as a possible fresh start point -- per task Part A, a schema-1 checkpoint's `best_feasible.Delta`
and `feasible` flag were computed via the F1-buggy `-zeta_star` shorthand and must NOT be trusted.
`provenance.stored_Delta_DO_NOT_TRUST` is returned only for audit/comparison logging; callers must
cold-re-evaluate `w_candidate` with `cm_production_value_verified` (or
`run_cm_upper_checkpointed`'s own cb_F!, which now does this correctly) before treating it as
feasible or as an incumbent. The `CMCheckpoint` struct layout is unchanged between schema 1 and 2
(only the semantics of the stored Delta changed), so plain `deserialize` reconstructs it directly.
"""
function migrate_cm_checkpoint_v1_candidate(path::AbstractString)
    raw = deserialize(path)::CMCheckpoint
    raw.schema == 1 || error("migrate_cm_checkpoint_v1_candidate($path): expected schema=1, got $(raw.schema)")
    w_candidate = vcat(raw.g, raw.zfree)
    stored_Delta = raw.best_feasible === nothing ? NaN : raw.best_feasible.Delta
    provenance = (run_id = raw.run_id, label = raw.label, n_eval = raw.n_eval,
                  checkpoint_reason = raw.checkpoint_reason, stored_Delta_DO_NOT_TRUST = stored_Delta)
    return (w_candidate = w_candidate, provenance = provenance)
end

"""
    run_cm_upper_checkpointed(ctx, w0; L, contrasts=:anchored, probs, delta=1.0,
        maxtime_real=180.0, ckpt_dir, run_id, label, checkpoint_interval_s=90.0,
        resume_from=nothing, cm_hessian_backend=:structured, kwargs...) -> NamedTuple

Checkpointed CM outer loop, same objective/constraint/gradient path as `run_cm_upper`
(cm_outer_driver.jl, UNCHANGED, reused not reimplemented) but with `D20Checkpoint`-grade
resume support. `resume_from` (a `CMCheckpoint` path) regenerates `ctx`/`pcx` from the
checkpoint's own recorded provenance and REFUSES to resume (hard error, not a silent
best-effort) if the regenerated draw checksums don't match -- same guarantee schema-3 gives
the unrestricted path.
"""
function run_cm_upper_checkpointed(w0::Union{Nothing,Vector{Float64}} = nothing;
        W::Int = 80000, delta::Float64 = 1.0, draw_design::Symbol = :sobol_randomized, draw_seed::Int = 20260719,
        L::Int = 10, contrasts::Symbol = :anchored, probs::Union{Nothing,AbstractVector{Float64}} = nothing,
        cm_hessian_backend::Symbol = :structured, cm_grid_rule::Symbol = :equal,
        maxtime_real::Float64 = 180.0, opt_file::String = "csw_outer_wallclock_sr1.opt",
        z_halfwidth::Float64 = 30.0,
        ckpt_dir::AbstractString, run_id::String = string(Dates.now()), label::String = "cm_upper",
        checkpoint_interval_s::Float64 = 90.0, resume_from::Union{Nothing,AbstractString} = nothing,
        verbose::Bool = true,
        cm_gradient_backend::Symbol = :reference,   # overnight task 2026-07-22 (docs/CM_GRADIENT_ALGEBRA_TRACE_2026-07-22.md):
        # :reference (production default, UNCHANGED behavior -- cm_production_gradient/
        # composite_gradient_at_fast, byte-identical to every run before this kwarg existed) |
        # :cplus (experimental -- cm_production_gradient_cplus/composite_gradient_at_Cplus_from_cache,
        # lfix_cm_cplus.jl). Part II.4 follow-up (2026-07-23): now VALIDATED against a resumed
        # checkpoint's own persisted `cm_gradient_backend` (schema>=3; schema-2 checkpoints are
        # treated as :reference, see `upgrade_schema2`). A mismatch is a HARD ERROR unless
        # `allow_backend_switch=true` is also passed -- see that kwarg's own doc for the policy
        # rationale (base-state/incumbent correctness is backend-independent, proven in
        # docs/CM_GRADIENT_ALGEBRA_TRACE_2026-07-22.md and empirically confirmed to 1e-13..1e-16
        # across every real D=20 point tested this session, but the persisted `bandwidth_cache`
        # was tuned under the ORIGINAL backend's own selection formula and is cleared on a
        # deliberate switch rather than silently reused across backends).
        allow_backend_switch::Bool = false,   # explicit override required to resume under a
        # DIFFERENT cm_gradient_backend than the checkpoint was written with. Ignored for a fresh
        # (non-resumed) run. Switching is not silently allowed even though it is
        # correctness-preserving (see above) -- the brief's own instruction is to make this an
        # explicit, audited choice, not an invisible default.
        shadow_stats::Union{Nothing,Dict{Symbol,Any}} = nothing,   # Part III.2 follow-up (2026-07-23):
        # opt-in, ADDITIVE-ONLY shadow instrumentation for the complete-state cache hit-opportunity
        # question (docs/CM_CACHE_PROCESS_LIFECYCLE_AUDIT_2026-07-23.md). `nothing` (the default) is
        # exactly zero behavioral/perf cost -- no dict lookups, no extra allocation, byte-identical
        # to every run before this kwarg existed. When a `Dict{Symbol,Any}` is passed (caller-owned,
        # pre-populated with :f_keys=>Dict{UInt64,Int}(), :g_keys=>Dict{UInt64,Int}(),
        # :g_same_as_last_F=>Ref(0), :g_could_have_hit_cache=>Ref(0)), cb_F!/cb_G! record the
        # fingerprinted point key (SAME `hash(round.(x,digits=12))` scheme
        # COMPLETE_STATE_CACHE_DESIGN_2026-07-22.md's own `outer_point_key` uses) on every call --
        # NEVER reads from it to change behavior, purely a counter. Does not itself use any cache;
        # measures what a cache COULD have hit.
        heartbeat_interval_s::Union{Nothing,Float64} = nothing)   # remediation task Part B:
        # opt-in liveness watchdog (nothing = off, zero overhead, the default). When set, a
        # background Timer logs, every heartbeat_interval_s, how long it has been since the last
        # cb_F!/cb_G! callback RETURNED. Purpose: distinguish, with real evidence, "ordinary long
        # callback" (heartbeats show a callback in flight, but n_eval/n_grad keep advancing
        # release-over-release) from "stuck inside a single callback" (one callback's wall time
        # alone exceeds several heartbeat intervals with nothing else changing) from "process
        # killed externally" (log simply stops -- see `signal 15: Terminated`, which is Julia's
        # own SIGTERM handler dumping a backtrace, NOT evidence of an uncaught exception or an
        # internal crash; a genuine uncaught Julia exception self-terminates with an ERROR/
        # nonzero exit code and needs no external signal -- confirmed live in
        # test_threaded_exception_propagation.jl). Does NOT attempt to time individual
        # sub-phases (moment construction / Hessian / BLAS / GC) -- that finer breakdown is Part
        # D's scope (full production timing instrumentation), not duplicated here.
    lp(xs...) = (println(xs...); flush(stdout))
    mkpath(ckpt_dir)

    cm_gradient_backend in (:reference, :cplus) ||
        error("run_cm_upper_checkpointed($label): cm_gradient_backend must be :reference|:cplus, got :$cm_gradient_backend")
    lp("[", label, "] cm_gradient_backend=", cm_gradient_backend,
       cm_gradient_backend == :cplus ? " (EXPERIMENTAL -- overnight task 2026-07-22, opt-in)" : " (production default)")
    write(joinpath(ckpt_dir, "$(label)_gradient_backend.txt"),
          "cm_gradient_backend=$(cm_gradient_backend)\nrun_id=$(run_id)\nrecorded_at=$(Dates.now())\n")

    resumed = resume_from === nothing ? nothing : load_cm_checkpoint(resume_from)
    find_smallest = true   # :cm_upper is the only wired direction today, matches run_cm_upper's own docstring
    backend_switched = false   # Part II.4 follow-up -- set true below only on an explicit, audited cross-backend resume

    if resumed !== nothing
        W = resumed.W; delta = resumed.delta; draw_design = resumed.draw_design; draw_seed = resumed.draw_seed
        L = resumed.cm_L; probs = resumed.cm_probs; contrasts = resumed.cm_contrasts
        cm_hessian_backend = resumed.cm_hessian_backend; cm_grid_rule = resumed.cm_grid_rule
        lp("[", label, "] RESUMING from ", resume_from, " (n_eval=", resumed.n_eval, " n_grad=", resumed.n_grad,
           " wall_elapsed=", round(resumed.wall_elapsed, digits = 1), "s)")
        # Part II.4 follow-up (2026-07-23): backend provenance check. resumed.cm_gradient_backend
        # is always present now (schema-2 files are upgraded to :reference by load_cm_checkpoint).
        backend_switched = resumed.cm_gradient_backend != cm_gradient_backend
        if backend_switched
            if !allow_backend_switch
                error("run_cm_upper_checkpointed($label): checkpoint was written with " *
                      "cm_gradient_backend=:$(resumed.cm_gradient_backend), but this call requests " *
                      ":$(cm_gradient_backend) -- refusing to silently switch backends on resume. " *
                      "Pass allow_backend_switch=true if this is intentional (base-state/incumbent " *
                      "correctness is backend-independent -- see the kwarg's own docstring -- but the " *
                      "switch is audited, not silent, and clears the persisted bandwidth_cache).")
            end
            lp("[", label, "] *** BACKEND SWITCH ON RESUME *** checkpoint backend=:",
               resumed.cm_gradient_backend, " -> requested backend=:", cm_gradient_backend,
               " (allow_backend_switch=true, explicit override) -- clearing persisted bandwidth_cache",
               " (was tuned under the OLD backend's own selection formula, not safe to reuse as-is).")
            write(joinpath(ckpt_dir, "$(label)_backend_switch_audit.txt"),
                  "BACKEND SWITCH ON RESUME\n" *
                  "resumed_from=$(resume_from)\n" *
                  "checkpoint_backend=$(resumed.cm_gradient_backend)\n" *
                  "requested_backend=$(cm_gradient_backend)\n" *
                  "run_id=$(run_id)\nlabel=$(label)\nswitched_at=$(Dates.now())\n" *
                  "bandwidth_cache_cleared=true\n" *
                  "n_eval_at_switch=$(resumed.n_eval)\nn_grad_at_switch=$(resumed.n_grad)\n")
        end
        # AUD-10 fix (matches c10_d20_production_driver.jl's identical resume warning): the
        # checkpoint's own best_feasible[] was already gated by is_verified_success in cb_F!
        # above (or is nothing) -- it remains the correct scientific incumbent regardless of
        # whether the RAW terminal solver state that triggered :stage_complete_unverified was
        # itself verified.
        if resumed.checkpoint_reason == :stage_complete_unverified
            lp("WARNING: resuming from a checkpoint whose terminal point FAILED verification at write ",
               "time (checkpoint_reason=:stage_complete_unverified, AUD-04/AUD-10) -- best_feasible[] ",
               "inside this checkpoint is still the correct scientific incumbent; only the raw terminal ",
               "solver-state fields are suspect.")
        end
    end

    ctx = d20_real_setup_design(W = W, δ = delta, find_smallest = find_smallest, draw_design = draw_design, draw_seed = draw_seed)
    pe = build_pivot_elimination(ctx)

    if resumed !== nothing
        if ctx.draw_meta.checksum_uniform != resumed.draw_checksum_uniform ||
           ctx.draw_meta.checksum_transformed != resumed.draw_checksum_transformed
            error("run_cm_upper_checkpointed($label): draw checksum MISMATCH on resume -- regenerated " *
                  "draws (design=:$(draw_design), seed=$(draw_seed)) do not match the checkpoint's own " *
                  "recorded checksums. Refusing to resume from a different problem instance.")
        end
        w0 = vcat(resumed.g, resumed.zfree)
    elseif w0 === nothing
        error("run_cm_upper_checkpointed($label): w0 required for a fresh (non-resumed) run")
    end

    probs === nothing && error("run_cm_upper_checkpointed($label): probs required (exact cutpoints, not re-derived from L)")
    pcx = build_cm_production_context(ctx, CS; L = L, contrasts = contrasts, probs = probs)

    # overnight task 2026-07-22: pool/workspace for cm_gradient_backend=:cplus only -- zero cost
    # (nothing allocated) when the default :reference backend is in effect.
    cplus_pool = cm_gradient_backend == :cplus ? build_grad_workspace_pool(size(ctx.obj.U, 1)) : nothing
    cplus_ws = cm_gradient_backend == :cplus ? build_lfix_factorized_workspace(ctx.D, size(ctx.obj.U, 1)) : nothing

    D2 = length(w0)
    gp_lo, gp_hi = ctx.bounds.γp_lo, ctx.bounds.γp_hi
    w_lo = vcat(gp_lo, w0[2:end] .- z_halfwidth)
    w_hi = vcat(gp_hi, w0[2:end] .+ z_halfwidth)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, opt_file))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
    xIndices = KNITRO.KN_add_vars(kc, D2)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w0)
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], delta)

    last_F_state = Ref{Any}(nothing)
    best_feasible = Ref{Any}(resumed !== nothing ? resumed.best_feasible : nothing)
    n_eval = Ref(resumed !== nothing ? resumed.n_eval : 0)
    n_grad = Ref(resumed !== nothing ? resumed.n_grad : 0)
    trace = NamedTuple[]
    bandwidth_cache = (resumed !== nothing && !backend_switched) ? copy(resumed.bandwidth_cache) : Dict{Int,Float64}()
    t_start = time()
    prior_wall = resumed !== nothing ? resumed.wall_elapsed : 0.0
    last_ckpt_t = Ref(time())
    last_activity_t = Ref(time())   # updated at the END of every cb_F!/cb_G! call
    last_activity_kind = Ref(:none)
    heartbeat_timer = nothing
    if heartbeat_interval_s !== nothing
        heartbeat_timer = Timer(heartbeat_interval_s; interval = heartbeat_interval_s) do _
            since = time() - last_activity_t[]
            lp("[", label, "] HEARTBEAT t=", round(time() - t_start, digits = 1),
               "s  last_callback=", last_activity_kind[], "  ", round(since, digits = 1),
               "s since last callback RETURNED  n_eval=", n_eval[], " n_grad=", n_grad[],
               since > 4 * heartbeat_interval_s ?
                   "  ** no callback has returned for >4 heartbeat intervals -- either an ordinary" *
                   " long single callback (e.g. a slow/near-infeasible inner solve) or a stall;" *
                   " this heartbeat cannot distinguish those two without per-phase timers (Part D)" : "")
        end
    end
    knitro_version = try
        KNITRO.KN_get_release()
    catch
        "unknown"
    end

    function do_checkpoint(reason::Symbol, w_current::Vector{Float64})
        zfree_now = w_current[2:end]
        logA_full = pivot_expand(zfree_now, pe)
        ckpt = CMCheckpointV3(CM_CHECKPOINT_SCHEMA, run_id, label, :cm_upper, find_smallest, delta, W, draw_seed,
            draw_design, ctx.draw_meta.checksum_uniform, ctx.draw_meta.checksum_transformed,
            L, collect(probs), contrasts, cm_grid_rule, :cumulative, cm_hessian_backend, cm_gradient_backend,
            w_current[1], copy(zfree_now), logA_full, copy(ctx.obj.x), copy(bandwidth_cache),
            best_feasible[], n_eval[], n_grad[], prior_wall + (time() - t_start),
            maxtime_real - (time() - t_start), reason, knitro_version)
        path = joinpath(ckpt_dir, "$(label)_latest.jls")
        save_cm_checkpoint(path, ckpt)
        last_ckpt_t[] = time()
        return ckpt
    end

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w, pe)
        local base, verify
        try
            _, base, verify = cm_production_value_verified(xf, pcx)
        catch e
            # Closure task Phase 3B: narrowed further to the dedicated CMExpectedSolveFailure
            # type (cm_production_bundle.jl) -- see cm_outer_driver.jl's identical fix for the
            # full rationale (a bare ErrorException is also what an ordinary programming bug
            # raises, so `e isa ErrorException` alone could silently swallow a real bug).
            e isa CMExpectedSolveFailure || rethrow()
            reject_point(w[1], "run_cm_upper_checkpointed($label): infeasible/failed inner solve at this point")
        end
        # Remediation fix (task Part A, finding F1): this used to read `Δ = -base.ζstar`, which
        # silently omits mean(Psi(q*)) and overstates the divergence whenever any recovered
        # weight m* exceeds e (the hybrid divergence's quadratic-branch threshold). `verify`
        # (returned by cm_production_value_verified, computed in archC_verified_state) already
        # carries the canonical Delta_dual = -(mean(Psi(q*))+zeta*) == cbuf[1]/1e10 -- use it
        # directly rather than recomputing or re-deriving. Verified live at real D=20/W=80,000/
        # L=50 points (remediation_a1_verify_delta_dual_identity.jl): the identity holds to
        # machine precision, and the two diverge (by a real, if usually small, amount) at
        # tail-active points.
        Δ = verify.Delta_dual
        evalResult.obj[1] = w[1]
        evalResult.c[1] = Δ
        n_eval[] += 1
        last_F_state[] = (w = copy(w), base = base)
        feasible = isfinite(Δ) && Δ <= delta + 1e-6
        # AUD-04 gate (matches c10_d20_production_driver.jl's cb_F! pattern): feasibility
        # (Delta<=delta) alone is not a verified solve -- KNITRO's own statuses 0/-100/-101/-103
        # are tolerance-based stops, not an optimality certificate. Require
        # classify_inner_result(verify) == VerifiedSolved before this point may become the
        # incumbent -- see docs/fullA_independent_audit_remediation.md AUD-04.
        verified = is_verified_success(verify)
        is_new_best = feasible && verified && (best_feasible[] === nothing || w[1] < best_feasible[].gp)
        if is_new_best
            best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ, n_eval = n_eval[], t = prior_wall + (time() - t_start))
            do_checkpoint(:new_best, collect(w))
        elseif time() - last_ckpt_t[] >= checkpoint_interval_s
            do_checkpoint(:wall_interval, collect(w))
        end
        push!(trace, (idx = n_eval[], t = time() - t_start, gp = w[1], Delta = Δ, feasible = feasible, verified = verified))
        if verbose && (n_eval[] <= 3 || n_eval[] % 20 == 0)
            lp("  eval ", n_eval[], " t=", round(time() - t_start, digits = 1), "s gp=", w[1], " Delta=", Δ, " feasible=", feasible, " verified=", verified)
        end
        if shadow_stats !== nothing
            fkey = hash(round.(w, digits = 12))
            fkeys = shadow_stats[:f_keys]::Dict{UInt64,Int}
            fkeys[fkey] = get(fkeys, fkey, 0) + 1
        end
        last_activity_t[] = time(); last_activity_kind[] = :cb_F!
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w, pe)
        shared = last_F_state[]
        base = (shared !== nothing && shared.w == w) ? shared.base : nothing
        if shadow_stats !== nothing
            gkey = hash(round.(w, digits = 12))
            gkeys = shadow_stats[:g_keys]::Dict{UInt64,Int}
            gkeys[gkey] = get(gkeys, gkey, 0) + 1
            same_as_last_F = base !== nothing   # already handled, free -- not a cache opportunity
            if same_as_last_F
                (shadow_stats[:g_same_as_last_F]::Base.RefValue{Int})[] += 1
            else
                # A genuine potential cache hit exists iff THIS exact point was already solved by
                # an EARLIER cb_F! call in THIS SAME PROCESS (fkeys already has a prior entry for
                # this key, counted BEFORE this cb_G! call could itself have contributed one via
                # a later cb_F! -- fkeys is only ever incremented in cb_F!, never here, so this is
                # a clean pre-existing-solve check, not self-referential).
                fkeys = shadow_stats[:f_keys]::Dict{UInt64,Int}
                if get(fkeys, gkey, 0) > 0
                    (shadow_stats[:g_could_have_hit_cache]::Base.RefValue{Int})[] += 1
                end
            end
        end
        gfull, meta = if cm_gradient_backend == :cplus
            cm_production_gradient_cplus(xf, pcx, ctx, pe, cplus_pool, cplus_ws; base = base, threaded = true,
                h_mode = :cached, bandwidth_cache = bandwidth_cache)
        else
            cm_production_gradient(xf, pcx, ctx, pe; base = base, threaded = true,
                h_mode = :cached, bandwidth_cache = bandwidth_cache)
        end
        n_grad[] += 1
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = 1.0
        evalResult.jac .= gfull
        last_activity_t[] = time(); last_activity_kind[] = :cb_G!
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)

    try
        KNITRO.KN_solve(kc)
    finally
        heartbeat_timer !== nothing && close(heartbeat_timer)
    end
    wall_ext = time() - t_start
    nStatus, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    σ = ctx.σ
    b = best_feasible[]
    κ = b === nothing ? NaN : 1 - b.gp^(σ / (σ - 1))

    # AUD-04/AUD-10 fix (matches c10_d20_production_driver.jl's run_profile_checkpointed/
    # run_polish_checkpointed final-checkpoint gate): re-verify the terminal point independently
    # before checkpointing it as a normal :stage_complete -- KNITRO's own reported outer status
    # says nothing about whether the INNER dual solve at that point passed the AUD-04
    # residual/gap checks. best_feasible[] (already gated by is_verified_success in cb_F! above)
    # remains the correct resume/incumbent state regardless of this outcome.
    xf_final = x_free_from_w(collect(xsol), pe)
    local verify_final
    try
        _, _, verify_final = cm_production_value_verified(xf_final, pcx)
    catch e
        # Closure task Phase 3B: narrowed further to the dedicated CMExpectedSolveFailure type
        # (cm_production_bundle.jl) -- see cm_outer_driver.jl's identical fix for the full
        # rationale. A genuine programming bug here must propagate (this is post-solve diagnostic
        # verification, not a KNITRO callback, so there is no DomainError-vs-KN_RC_CALLBACK_ERR
        # contract to preserve by catching broadly).
        e isa CMExpectedSolveFailure || rethrow()
        verify_final = (inner_status = -300,)   # inner solve failed outright at the terminal point -- unverified by construction
    end
    if is_verified_success(verify_final)
        final_ckpt = do_checkpoint(:stage_complete, collect(xsol))
    else
        lp("[", label, "] WARNING: terminal point failed verification (class=", classify_inner_result(verify_final),
           ") -- checkpointing as :stage_complete_unverified, NOT :stage_complete. best_feasible[]=",
           best_feasible[] === nothing ? "nothing" : "gp=$(best_feasible[].gp) Delta=$(best_feasible[].Delta)",
           " remains the correct resume/incumbent state (AUD-10).")
        final_ckpt = do_checkpoint(:stage_complete_unverified, collect(xsol))
    end
    return (knitro_status = nStatus, wall = wall_ext, n_eval = n_eval[], n_grad = n_grad[],
            best = b, kappa = κ, xsol = collect(xsol), trace = trace, final_checkpoint = final_ckpt,
            ckpt_path = joinpath(ckpt_dir, "$(label)_latest.jls"))
end
