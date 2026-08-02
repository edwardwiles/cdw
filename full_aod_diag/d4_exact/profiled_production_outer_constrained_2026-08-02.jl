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
using KNITRO

"""
    run_profiled_upper_constrained(label, w0; fctx, evaluate_fn, ctx, pe, delta=1.0,
        maxtime_real=180.0, hessopt_tag="sr1", z_halfwidth=30.0,
        gp_lo=ctx.bounds.γp_lo, gp_hi=ctx.bounds.γp_hi,
        use_screen=false, cache=nothing, verbose=true) -> NamedTuple

Constrained-upper-bound KNITRO outer loop for the REDUCED/profiled path, structurally IDENTICAL to
`run_cm_upper`/`run_originzc_upper_checkpointed`'s own problem (minimize w[1]=gp subject to
Delta_dual(w)<=delta, box bounds on gp and the remaining free coordinates, Direct+SR1 algorithm),
but evaluated via the REDUCED path's own `evaluate_fn`/`shared_family_outer_gradient` -- the same
functions `_run_profiled_outer_knitro_loop` already calls, unchanged. Returns the best verified
FEASIBLE incumbent (smallest gp with Delta<=delta+1e-6), matching `run_cm_upper`'s own
`best_feasible` semantics exactly, so results are directly comparable to the FULL-arm driver's own
return shape.
"""
function run_profiled_upper_constrained(label::String, w0::Vector{Float64}; fctx,
        evaluate_fn::Function, ctx, pe::PivotGravityElimOnRetained,
        delta::Float64 = 1.0, maxtime_real::Float64 = 180.0, hessopt_tag::String = "sr1",
        z_halfwidth::Float64 = 30.0,
        gp_lo::Float64 = ctx.bounds.γp_lo, gp_hi::Float64 = ctx.bounds.γp_hi,
        use_screen::Bool = false, cache = nothing,
        trace_csv::Union{Nothing,AbstractString} = nothing,
        verbose::Bool = true)
    lp(xs...) = (println(xs...); flush(stdout))
    fam = family_kind(fctx)
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
    best_feasible = Ref{Any}(nothing)
    n_eval = Ref(0); n_grad = Ref(0); trace = NamedTuple[]
    t_start = time()

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
    wall = time() - t_start
    nStatus, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

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
