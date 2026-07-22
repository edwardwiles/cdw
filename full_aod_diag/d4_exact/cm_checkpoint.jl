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

const CM_CHECKPOINT_SCHEMA = 1

"""
    CMCheckpoint

CM-production analogue of `D20Checkpoint` (c10_d20_production_driver.jl), covering
everything a resume needs to (a) validate it is reconstructing the IDENTICAL problem
instance and (b) restart the outer KNITRO search from the best exact-feasible incumbent
found so far -- plus the CM-specific configuration (`D20Checkpoint` has no notion of a
common-marginals grid at all).
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

"Atomic-ish checkpoint write, same discipline as `save_checkpoint` (D20Checkpoint): serialize to a .tmp file then mv, so a crash mid-write never leaves a half-written checkpoint."
function save_cm_checkpoint(path::AbstractString, ckpt::CMCheckpoint)
    tmp = path * ".tmp"
    serialize(tmp, ckpt)
    mv(tmp, path; force = true)
    return path
end

function load_cm_checkpoint(path::AbstractString)
    ckpt = deserialize(path)::CMCheckpoint
    ckpt.schema == CM_CHECKPOINT_SCHEMA ||
        error("load_cm_checkpoint($path): schema=$(ckpt.schema), expected $(CM_CHECKPOINT_SCHEMA) -- " *
              "this checkpoint predates the CM checkpoint-schema unification (task §11), e.g. a bare " *
              "ad-hoc NamedTuple from c13_d20_cm_upper_continuation.jl's old save_stage. Start a fresh " *
              "run instead of resuming from an incompatible checkpoint.")
    return ckpt
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
        verbose::Bool = true)
    lp(xs...) = (println(xs...); flush(stdout))
    mkpath(ckpt_dir)

    resumed = resume_from === nothing ? nothing : load_cm_checkpoint(resume_from)
    find_smallest = true   # :cm_upper is the only wired direction today, matches run_cm_upper's own docstring

    if resumed !== nothing
        W = resumed.W; delta = resumed.delta; draw_design = resumed.draw_design; draw_seed = resumed.draw_seed
        L = resumed.cm_L; probs = resumed.cm_probs; contrasts = resumed.cm_contrasts
        cm_hessian_backend = resumed.cm_hessian_backend; cm_grid_rule = resumed.cm_grid_rule
        lp("[", label, "] RESUMING from ", resume_from, " (n_eval=", resumed.n_eval, " n_grad=", resumed.n_grad,
           " wall_elapsed=", round(resumed.wall_elapsed, digits = 1), "s)")
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
    bandwidth_cache = resumed !== nothing ? copy(resumed.bandwidth_cache) : Dict{Int,Float64}()
    t_start = time()
    prior_wall = resumed !== nothing ? resumed.wall_elapsed : 0.0
    last_ckpt_t = Ref(time())
    knitro_version = try
        KNITRO.KN_get_release()
    catch
        "unknown"
    end

    function do_checkpoint(reason::Symbol, w_current::Vector{Float64})
        zfree_now = w_current[2:end]
        logA_full = pivot_expand(zfree_now, pe)
        ckpt = CMCheckpoint(CM_CHECKPOINT_SCHEMA, run_id, label, :cm_upper, find_smallest, delta, W, draw_seed,
            draw_design, ctx.draw_meta.checksum_uniform, ctx.draw_meta.checksum_transformed,
            L, collect(probs), contrasts, cm_grid_rule, :cumulative, cm_hessian_backend,
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
            throw(DomainError(w[1], "run_cm_upper_checkpointed($label): infeasible/failed inner solve at this point"))
        end
        Δ = -base.ζstar
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
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w, pe)
        shared = last_F_state[]
        base = (shared !== nothing && shared.w == w) ? shared.base : nothing
        gfull, meta = cm_production_gradient(xf, pcx, ctx, pe; base = base, threaded = true,
            h_mode = :cached, bandwidth_cache = bandwidth_cache)
        n_grad[] += 1
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = 1.0
        evalResult.jac .= gfull
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)

    KNITRO.KN_solve(kc)
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
