# Performance closeout task (2026-08-02), Section 13 follow-up: a genuinely matched FULL-vs-REDUCED
# outer-search comparison requires BOTH arms to solve the SAME optimization problem. Investigation
# found they did not: the existing REDUCED-path scaffold (`_run_profiled_outer_knitro_loop`,
# profiled_production_outer_runner_2026-08-01.jl) MINIMIZES Delta_dual directly with gp held
# within a tiny +/-0.05 box around the start point (its own `gp_bounds_halfwidth` default) -- it
# never enforces Delta<=delta as a hard constraint. The FULL-path production drivers
# (`run_cm_upper`, cm_outer_driver.jl; `run_originzc_upper_checkpointed`,
# cm_originzc_checkpoint.jl) solve the OPPOSITE, genuinely constrained problem: minimize gp subject
# to Delta_dual(w)<=delta. These are not two formulations of the same problem -- comparing their
# raw outputs directly would compare different objectives, not formulation speed.
#
# This file adds ONE minimal top-level KNITRO driver, `run_profiled_upper_constrained`, that gives
# the REDUCED path the SAME constrained problem shape as `run_cm_upper` (mirrored line-for-line:
# same objective/constraint/box/algorithm setup), while reusing the REDUCED path's existing
# evaluator (`evaluate_fn`) and gradient (`shared_family_outer_gradient`) completely unchanged --
# no new economics, no new evaluator, no new gradient, only a different top-level KNITRO wiring so
# both arms solve the literal same problem. This is the smallest change that makes the comparison
# meaningful.
#
# profiled-outer-production-readiness-2026-08-03 (task 2), section 5: checkpoint/resume, ADDED
# below (checkpoint_path/checkpoint_interval_s/resume_from kwargs, all default `nothing`/60.0 so
# every existing caller -- run_outer_originzc_reduced_constrained_2026-08-02.jl,
# run_outer_flexcm_reduced_constrained_2026-08-02.jl -- is byte-identical when it passes neither).
# Before this, `run_profiled_upper_constrained` (the genuine constrained REDUCED runner the task
# names explicitly) had NO checkpoint support at all -- confirmed by reading this file's own
# un-augmented body, 2026-08-03 -- unlike all three real FULL production entry points
# (run_polish_checkpointed_unified/run_cm_upper_checkpointed/run_originzc_upper_checkpointed),
# which already implement resume_from. This also fixes a real gap in the OLDER gp-fixed scaffold's
# own checkpoint writer (`_write_profiled_checkpoint`, profiled_production_outer_runner_2026-08-01.jl):
# that function hardcodes W=0, draw_seed=0, draw_design=:sobol_randomized, draw_checksum_uniform/
# transformed="", delta=1.0, destination_sample=:unrecorded, row_idx=nothing, D_dest=0 -- real
# values that WERE available in its caller's scope but were never threaded through to the
# checkpoint-writing helper. `_write_profiled_constrained_checkpoint` below takes `ctx` and
# `delta` explicitly and records the real values instead.
isdefined(Main, :CMCheckpointV11) ||
    error("profiled_production_outer_constrained_2026-08-02.jl requires cm_checkpoint.jl to be included first (checkpoint/resume, task 2 section 5).")
isdefined(Main, :build_profiled_production_config) ||
    error("profiled_production_outer_constrained_2026-08-02.jl requires profiled_ab_comparability_and_plumbing_2026-08-01.jl to be included first (checkpoint namespace, task 2 section 5).")
using KNITRO

"""
    _write_profiled_constrained_checkpoint(path, label, fctx, ctx, delta, config, w_full, best_feasible,
        n_eval, n_grad, wall_elapsed, wall_budget_remaining) -> Nothing

Task 2 section 5: real, versioned `CMCheckpointV11` writer for `run_profiled_upper_constrained`.
Same field-reuse discipline as the existing `_write_profiled_checkpoint`
(profiled_production_outer_runner_2026-08-01.jl) -- family is stored in the `cm_extension` slot
via `Symbol(family_kind(fctx))` (that struct has no dedicated `family` field; this matches the
existing checkpoint's own convention, not a new overload invented here) -- but reads
W/draw_seed/draw_design/draw_checksum_uniform/draw_checksum_transformed/delta/find_smallest/
destination_sample/row_idx/D_dest from the REAL `ctx`/`delta` this driver actually has in scope,
instead of the zero/placeholder values `_write_profiled_checkpoint` hardcodes (that function's
caller never threaded `ctx`/`delta` through to it -- fixed here by taking both as parameters).
`best_feasible` is stored as-is (the driver's own richer `(gp,w,Delta,n_eval,t_elapsed)` NamedTuple,
not down-cast to the older `(w,Delta)` shape) -- the field is `::Any`, so this is a valid, if
driver-specific, shape a resumer of THIS driver's own checkpoints already knows how to read.

`run_profiled_upper_constrained` is used with BOTH `d20_real_setup_design` contexts (real Monte
Carlo draws: `ctx.W`/`ctx.draw_design`/`ctx.draw_seed`/`ctx.draw_meta`/`ctx.destination_sample`/
`ctx.row_idx`/`ctx.D_dest` all present) AND `d4_exact_setup` contexts (D4 is an analytically
EXACT synthetic economy -- no draws at all, confirmed by reading its own return NamedTuple, which
has none of those seven fields). Every draw/W/destination field below is read via `hasproperty`
with an explicit neutral fallback (`W=0`, `draw_design=:none`, checksums `""`,
`destination_sample=:not_applicable`, `row_idx=nothing`, `D_dest=ctx.D`) rather than assuming
either context shape -- this is what makes `_write_profiled_constrained_checkpoint`/the resume
validation below work at D4 (this file's own fast test) without erroring on a missing field.
"""
function _write_profiled_constrained_checkpoint(path::AbstractString, label::String, fctx, ctx,
        delta::Float64, config, w_full::Vector{Float64}, best_feasible,
        n_eval::Int, n_grad::Int, wall_elapsed::Float64, wall_budget_remaining::Float64)
    fam = family_kind(fctx)
    has_draws = hasproperty(ctx, :draw_meta) && ctx.draw_meta !== nothing
    draw_cksum_u = has_draws ? ctx.draw_meta.checksum_uniform : ""
    draw_cksum_t = has_draws ? ctx.draw_meta.checksum_transformed : ""
    ckpt = CMCheckpointV11(CM_CHECKPOINT_SCHEMA_V11, "profiled_constrained_$(label)", label, :cm_upper,
        ctx.find_smallest, delta,
        hasproperty(ctx, :W) ? ctx.W : 0,
        hasproperty(ctx, :draw_seed) ? ctx.draw_seed : 0,
        hasproperty(ctx, :draw_design) ? ctx.draw_design : :none,
        draw_cksum_u, draw_cksum_t,
        0, Float64[], :anchored, :equal, :orthonormal, :structured, :cplus,
        Symbol(fam), 0, 0, :direct, 1,
        w_full[1], w_full[2:end], Float64[], zeros(0, 0), Float64[], Dict{Int,Float64}(),
        best_feasible, n_eval, n_grad, wall_elapsed, wall_budget_remaining, :wall_interval,
        "unknown",
        hasproperty(ctx, :destination_sample) ? ctx.destination_sample : :not_applicable,
        hasproperty(ctx, :row_idx) ? ctx.row_idx : nothing,
        hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D,
        :common_flexible, :legacy_z,
        config.economic_parameterization, config.stable_layout_digest, string(config.inner_dual_layout_digest),
        config.hessian_backends.H_ZZ, config.hessian_backends.H_CZ, config.hessian_backends.H_EZ,
        config.hessian_backends.source, config.full_a_recovery_convention, config.checkpoint_namespace)
    save_cm_checkpoint(path, ckpt)
    return nothing
end

"""
    run_profiled_upper_constrained(label, w0; fctx, evaluate_fn, ctx, pe, delta=1.0,
        maxtime_real=180.0, hessopt_tag="sr1", z_halfwidth=30.0,
        gp_lo=ctx.bounds.γp_lo, gp_hi=ctx.bounds.γp_hi,
        use_screen=false, cache=nothing, verbose=true,
        checkpoint_path=nothing, checkpoint_interval_s=60.0, resume_from=nothing) -> NamedTuple

Constrained-upper-bound KNITRO outer loop for the REDUCED/profiled path, structurally IDENTICAL to
`run_cm_upper`/`run_originzc_upper_checkpointed`'s own problem (minimize w[1]=gp subject to
Delta_dual(w)<=delta, box bounds on gp and the remaining free coordinates, Direct+SR1 algorithm),
but evaluated via the REDUCED path's own `evaluate_fn`/`shared_family_outer_gradient` -- the same
functions `_run_profiled_outer_knitro_loop` already calls, unchanged. Returns the best verified
FEASIBLE incumbent (smallest gp with Delta<=delta+1e-6), matching `run_cm_upper`'s own
`best_feasible` semantics exactly, so results are directly comparable to the FULL-arm driver's own
return shape.

Checkpoint/resume (task 2 section 5, ADDED): if `checkpoint_path` is given, a `CMCheckpointV11` is
written (via `_write_profiled_constrained_checkpoint`) every `checkpoint_interval_s` seconds AND on
every new verified-feasible incumbent -- same trigger discipline `_run_profiled_outer_knitro_loop`
already uses. If `resume_from` is given, the checkpoint is loaded (`load_cm_checkpoint_v11`) and
validated on FIVE independent axes before anything else runs, matching `run_cm_upper_checkpointed`'s
own hard-refuse-on-mismatch discipline (no silent override, no escape hatch): (1) checkpoint
namespace (`assert_checkpoint_compatible` -- economic_parameterization + family + outer layout
digest all folded in), (2) `find_smallest` (this driver only ever solves the upper/minimize-gp
direction; a checkpoint written under a different direction is refused, not silently reinterpreted),
(3) `W`, (4) `delta`, (5) `draw_design`+`draw_seed`+both draw checksums (a checkpoint whose draws
don't match this call's actual `ctx` is refused -- the checksums are the only genuine proof, not
merely the seed integers matching). On success, `w0` is overridden by the checkpoint's own
`vcat(g, zfree)`, and `n_eval`/`n_grad`/`best_feasible`/prior wall time are seeded from it so the
returned `n_eval`/`n_grad`/`wall` reflect the FULL history across the resume, not just this call.
"""
function run_profiled_upper_constrained(label::String, w0::Vector{Float64}; fctx,
        evaluate_fn::Function, ctx, pe::PivotGravityElimOnRetained,
        delta::Float64 = 1.0, maxtime_real::Float64 = 180.0, hessopt_tag::String = "sr1",
        z_halfwidth::Float64 = 30.0,
        gp_lo::Float64 = ctx.bounds.γp_lo, gp_hi::Float64 = ctx.bounds.γp_hi,
        use_screen::Bool = false, cache = nothing,
        trace_csv::Union{Nothing,AbstractString} = nothing,
        verbose::Bool = true,
        checkpoint_path::Union{Nothing,AbstractString} = nothing,
        checkpoint_interval_s::Float64 = 60.0,
        resume_from::Union{Nothing,AbstractString} = nothing)
    lp(xs...) = (println(xs...); flush(stdout))
    fam = family_kind(fctx)
    config = build_profiled_production_config(fctx; economic_parameterization = :profiled_destination_scales)

    n_eval_seed = 0; n_grad_seed = 0; best_feasible_seed = nothing; prior_wall = 0.0
    if resume_from !== nothing
        resumed = load_cm_checkpoint_v11(resume_from)
        assert_checkpoint_compatible(config, resumed.checkpoint_namespace)
        resumed.find_smallest ||
            error("run_profiled_upper_constrained($label): checkpoint at $resume_from was written " *
                  "with find_smallest=false, but this driver only ever solves the upper/minimize-gp " *
                  "direction -- refusing to resume a different direction under the same driver.")
        W_cur = hasproperty(ctx, :W) ? ctx.W : 0
        resumed.W == W_cur ||
            error("run_profiled_upper_constrained($label): checkpoint W=$(resumed.W) != this call's ctx W=$(W_cur) -- refusing to resume across a different W.")
        isapprox(resumed.delta, delta; atol = 1e-12) ||
            error("run_profiled_upper_constrained($label): checkpoint delta=$(resumed.delta) != this call's delta=$delta -- refusing to resume across a different delta.")
        # d20_real_setup_design contexts carry real Monte Carlo draws (ctx.draw_design/draw_seed/
        # draw_meta); d4_exact_setup contexts (D4, analytically exact) have none of these fields at
        # all -- hasproperty-guarded so this driver's checkpoint/resume works at both scales.
        has_draws_cur = hasproperty(ctx, :draw_meta) && ctx.draw_meta !== nothing
        has_draws_ckpt = !isempty(resumed.draw_checksum_uniform) || !isempty(resumed.draw_checksum_transformed)
        if has_draws_cur != has_draws_ckpt
            error("run_profiled_upper_constrained($label): checkpoint has_draws=$has_draws_ckpt but this call's ctx has_draws=$has_draws_cur -- refusing to resume across a draw-based context (e.g. D20) and a draw-free one (e.g. D4).")
        end
        if has_draws_cur
            draw_design_cur = hasproperty(ctx, :draw_design) ? ctx.draw_design : :none
            draw_seed_cur = hasproperty(ctx, :draw_seed) ? ctx.draw_seed : 0
            (resumed.draw_design == draw_design_cur && resumed.draw_seed == draw_seed_cur) ||
                error("run_profiled_upper_constrained($label): checkpoint draw_design/draw_seed=" *
                      ":$(resumed.draw_design)/$(resumed.draw_seed) != this call's ctx=:$(draw_design_cur)/$(draw_seed_cur) -- refusing to resume across a different draw configuration.")
            (ctx.draw_meta.checksum_uniform == resumed.draw_checksum_uniform &&
             ctx.draw_meta.checksum_transformed == resumed.draw_checksum_transformed) ||
                error("run_profiled_upper_constrained($label): draw checksum MISMATCH on resume -- " *
                      "checkpoint recorded uniform=$(resumed.draw_checksum_uniform) transformed=$(resumed.draw_checksum_transformed), " *
                      "this call's ctx has uniform=$(ctx.draw_meta.checksum_uniform) transformed=$(ctx.draw_meta.checksum_transformed) -- " *
                      "same draw_design/draw_seed produced different draws (non-determinism bug) or the checkpoint predates a draw-generation change; refusing to resume.")
        else
            lp("[$label] resume: no draw metadata on either side (draw-free context, e.g. D4) -- skipping draw-checksum validation")
        end
        length(vcat(resumed.g, resumed.zfree)) == length(w0) ||
            error("run_profiled_upper_constrained($label): checkpoint outer-vector length=$(length(vcat(resumed.g, resumed.zfree))) != this call's w0 length=$(length(w0)) -- refusing to resume across a different layout.")
        w0 = vcat(resumed.g, resumed.zfree)
        n_eval_seed = resumed.n_eval; n_grad_seed = resumed.n_grad
        best_feasible_seed = resumed.best_feasible
        prior_wall = resumed.wall_elapsed
        lp("[$label] RESUMING from $resume_from (n_eval=$(resumed.n_eval) n_grad=$(resumed.n_grad) " *
           "wall_elapsed=$(round(resumed.wall_elapsed,digits=1))s gp=$(w0[1]))")
    end

    D2 = length(w0)
    w_lo = vcat(gp_lo, w0[2:end] .- z_halfwidth)
    w_hi = vcat(gp_hi, w0[2:end] .+ z_halfwidth)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt"))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
    # Same explicit Direct+SR1 pin the unconstrained REDUCED scaffold uses (task's own Section 13
    # ask), via the existing helper -- not re-derived.
    set_outer_algorithm_direct!(kc, KNITRO_HESSOPT_SR1)
    xIndices = KNITRO.KN_add_vars(kc, D2)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w0)
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], delta)

    last_w = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    last_ev = Ref{Any}(nothing)
    best_feasible = Ref{Any}(best_feasible_seed)
    n_eval = Ref(n_eval_seed); n_grad = Ref(n_grad_seed); trace = NamedTuple[]
    t_start = time()
    t_last_ckpt = Ref(t_start)

    # solve_at: copied verbatim from _run_profiled_outer_knitro_loop's own screen/cache dispatch
    # (profiled_production_outer_runner_2026-08-01.jl) -- same screen precheck, same exact-cache
    # keying, same evaluate_fn call. Not re-derived.
    function solve_at(w_full::Vector{Float64})
        if use_screen
            try
                profiled_cm_screen_precheck!(w_full, ctx, pe; use_witness = false)
            catch e
                e isa CMExpectedSolveFailure || rethrow()
                return nothing
            end
        end
        if cache !== nothing
            key = ProfiledCMProductionEvalKey(:profiled_destination_scales, w_full, Float64[],
                ctx.δ, ctx.find_smallest, fam, stable_layout_digest(fctx), "")
            ev, _ = profiled_cm_cache_lookup_or_compute!(cache, key,
                () -> (ev = evaluate_fn(w_full, fctx); (ev, (inner_status = ev.result.inner_status,))))
            return ev
        end
        return evaluate_fn(w_full, fctx)
    end

    ev0 = solve_at(w0)
    ev0 !== nothing && ev0.result.inner_status in (0, -100, -101, -103) ||
        error("run_profiled_upper_constrained($label): start point not inner-feasible/not screen-passing")
    lp("[$label] CONSTRAINED start: gp0=$(w0[1]) Delta0=$(-ev0.result.zeta) status=$(ev0.result.inner_status)")

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        # solve_at (via evaluate_fn) can throw CMExpectedSolveFailure for a genuinely
        # infeasible/unbounded inner solve at a KNITRO trial point (a NORMAL, expected occurrence
        # during line search, not a bug) -- run_cm_upper's own cb_F! (cm_outer_driver.jl) catches
        # exactly this and rejects the point instead of propagating; this driver must do the same,
        # or an uncaught exception here aborts the ENTIRE outer search with a fatal KNITRO callback
        # error (confirmed live: exactly this happened on the first real W=100,000 run, cutting a
        # 600s budget short at 471s after 20 real evals -- not a fluke, a real missing catch).
        t_call0 = time()
        local ev
        try
            ev = solve_at(w)
        catch e
            e isa CMExpectedSolveFailure || rethrow()
            evalResult.obj[1] = w[1]
            evalResult.c[1] = 1e10
            n_eval[] += 1
            t_el = time() - t_start
            verbose && lp("  [$label] eval $(n_eval[]) t=$(round(t_el,digits=1))s gp=$(w[1]) REJECTED (CMExpectedSolveFailure) call_dur=$(round(time()-t_call0,digits=1))s")
            return 0
        end
        if ev === nothing || !(ev.result.inner_status in (0, -100, -101, -103))
            evalResult.obj[1] = w[1]
            evalResult.c[1] = 1e10   # certainly-infeasible sentinel, mirrors run_cm_upper's reject_point convention
            n_eval[] += 1
            t_el = time() - t_start
            verbose && lp("  [$label] eval $(n_eval[]) t=$(round(t_el,digits=1))s gp=$(w[1]) REJECTED (ev=$(ev === nothing ? "screened-out" : "inner_status=$(ev.result.inner_status)")) call_dur=$(round(time()-t_call0,digits=1))s")
            return 0
        end
        Δ = -ev.result.zeta
        evalResult.obj[1] = w[1]
        evalResult.c[1] = Δ
        n_eval[] += 1
        last_w[] = copy(w); last_ev[] = ev
        t_el = time() - t_start
        feasible = isfinite(Δ) && Δ <= delta + 1e-6
        verified = ev.result.inner_status in (0, -100, -101, -103)
        is_new_best = feasible && verified && (best_feasible[] === nothing || w[1] < best_feasible[].gp)
        if is_new_best
            best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ, n_eval = n_eval[], t_elapsed = t_el)
        end
        push!(trace, (idx = n_eval[], t_elapsed = t_el, gp = w[1], Delta = Δ, feasible = feasible, verified = verified))
        if checkpoint_path !== nothing && (time() - t_last_ckpt[] >= checkpoint_interval_s || is_new_best)
            _write_profiled_constrained_checkpoint(checkpoint_path, label, fctx, ctx, delta, config,
                copy(w), best_feasible[], n_eval[], n_grad[], prior_wall + t_el, maxtime_real - t_el)
            t_last_ckpt[] = time()
        end
        # Unconditional per-eval timing (call_dur = wall time of THIS solve_at call alone, not
        # cumulative) -- was previously gated to every-10th eval, which silently hid two evals
        # (5,6) that landed in the reject branches above during a real run and made an ~80s gap
        # look mysterious. Always print now; still cheap (one println per outer eval, not per draw).
        verbose && lp("  [$label] eval $(n_eval[]) t=$(round(t_el,digits=1))s gp=$(w[1]) Delta=$Δ feasible=$feasible verified=$verified call_dur=$(round(time()-t_call0,digits=1))s")
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        t_call0 = time()
        w = evalRequest.x
        local ev
        reused = false
        if last_w[] !== nothing && last_w[] == w
            ev = last_ev[]; reused = true
        else
            # Defensive: KNITRO normally only requests a gradient at a point cb_F! already
            # evaluated successfully, so this branch should rarely execute -- but guard it with
            # the same CMExpectedSolveFailure catch as cb_F! regardless, for robustness.
            try
                ev = solve_at(w)
            catch e
                e isa CMExpectedSolveFailure || rethrow()
                evalResult.objGrad .= 0.0; evalResult.objGrad[1] = 1.0
                evalResult.jac .= 0.0
                verbose && lp("  [$label] grad $(n_grad[]+1) t=$(round(time()-t_start,digits=1))s REJECTED (CMExpectedSolveFailure, re-solve path -- NOT reused from cb_F!) call_dur=$(round(time()-t_call0,digits=1))s")
                return 0
            end
        end
        t_solve_done = time()
        g, meta = shared_family_outer_gradient(collect(Float64, w), ctx, fctx, ev)
        n_grad[] += 1
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = 1.0
        evalResult.jac .= g
        # Unconditional timing, split into (re-solve time if not reused) + (gradient-engine time) --
        # this is the OTHER candidate (besides a slow cb_F! reject) for where wall time could go
        # unaccounted-for between printed cb_F! evals.
        verbose && lp("  [$label] grad $(n_grad[]) t=$(round(time()-t_start,digits=1))s reused_from_cbF=$reused solve_dur=$(round(t_solve_done-t_call0,digits=1))s grad_engine_dur=$(round(time()-t_solve_done,digits=1))s")
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)

    KNITRO.KN_solve(kc)
    wall = prior_wall + (time() - t_start)   # cumulative across resumes, not just this call
    nStatus, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    if checkpoint_path !== nothing
        _write_profiled_constrained_checkpoint(checkpoint_path, label, fctx, ctx, delta, config,
            copy(collect(Float64, xsol)), best_feasible[], n_eval[], n_grad[], wall, max(0.0, maxtime_real - (time() - t_start)))
    end

    if trace_csv !== nothing
        open(trace_csv, "w") do io
            println(io, "idx,t_elapsed,gp,Delta,feasible,verified")
            for r in trace
                println(io, "$(r.idx),$(r.t_elapsed),$(r.gp),$(r.Delta),$(r.feasible),$(r.verified)")
            end
        end
    end

    b = best_feasible[]
    lp("[$label] DONE: status=$nStatus wall=$(round(wall,digits=1))s n_eval=$(n_eval[]) n_grad=$(n_grad[])")
    b !== nothing && lp("  best: gp=$(b.gp) Delta=$(b.Delta) found_at_eval=$(b.n_eval) t=$(b.t_elapsed)s")

    return (label = label, family = fam, knitro_status = nStatus, wall = wall,
        n_eval = n_eval[], n_grad = n_grad[], best = b, xsol = collect(xsol), trace = trace)
end
