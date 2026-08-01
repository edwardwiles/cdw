# ============================================================================
# Claude Code task 2026-08-01, §14: matched full-vs-profiled outer A/B
# harness. SURGICAL by design (per live user feedback): the FULL/reference
# arm is literally `run_profile_checkpointed` (production driver,
# c10_d20_production_driver.jl) called unmodified -- zero new code for that
# arm. Only the PROFILED arm needs a new (small) outer loop, since no
# production driver exists for `economic_parameterization=:profiled_destination_scales`
# yet; it reuses the SAME outer-KNITRO configuration (csw_outer_wallclock_sr1.opt,
# algorithm=3, z_halfwidth=30) and the already-built/gated
# evaluate_profiled_point / profiled_composite_gradient_at.
#
# Screens (task §13): the profiled arm has NO screens at all by construction
# (evaluate_profiled_point never calls evaluate_fullA_screened_ranged); the
# full/reference arm is called with `use_general_range_safety_net=false`
# (explicit opt-out) -- `use_witness` is already false by default under
# `destination_sample=:exclude_row`'s production ctx (no witness built), and
# the pre-winner envelope screen is already disabled by default under
# `:exclude_row`. The pairwise certificate and the fused zero-winner/winning-
# range scan (screen_hard_winners_ranged) have NO disable kwarg in production
# (call-graph doc §9) -- this is a documented, unavoidable asymmetry for this
# first A/B (both are cheap CERTIFICATES that only reject genuinely infeasible
# points, not a source of false rejection at a validated calibration-adjacent
# trajectory), not silently ignored.
# ============================================================================

isdefined(Main, :evaluate_profiled_point) || error("profiled_outer_ab_harness_2026-08-01.jl requires profiled_outer_evaluator_2026-08-01.jl to be included first.")
isdefined(Main, :profiled_composite_gradient_at) || error("profiled_outer_ab_harness_2026-08-01.jl requires profiled_outer_gradient_fd_2026-08-01.jl to be included first.")

using KNITRO, Dates, CSV, DataFrames

"""
    run_profiled_outer_search(label, w_start; ctx, spec, pe, maxtime_real=1800.0,
                               hessopt_tag="sr1", maxit_override=nothing, trace_csv=nothing) -> NamedTuple

Minimal outer KNITRO loop for the profiled parameterization, matching
`run_profile_checkpointed`'s outer-solver configuration exactly (same .opt
file, same hardcoded algorithm=3, same z_halfwidth=30 box) but with NO
screens, NO checkpoint/resume schema (out of scope, task §18), and the
gradient computed via `profiled_composite_gradient_at` (fixed-dual FD, §9).
`find_smallest` is read from `ctx.find_smallest` (the ctx passed in already
encodes the search direction, matching how `evaluate_profiled_point` uses it
implicitly through `ctx.obj`).
"""
function run_profiled_outer_search(label::String, w_start::Vector{Float64}; ctx, spec::AnchorSpec,
        pe::PivotGravityElimOnRetained, maxtime_real::Float64 = 1800.0, hessopt_tag::String = "sr1",
        maxit_override::Union{Nothing,Int} = nothing, trace_csv::Union{Nothing,AbstractString} = nothing,
        gradient_fn::Function = profiled_composite_gradient_at)   # swappable: profiled_composite_gradient_at
        # (full-rebuild FD, §9 original) or profiled_composite_gradient_at_incremental
        # (O(1)-per-changed-cell, profiled_lfix_incremental_2026-08-01.jl) -- same call signature
        # (w, ctx, spec, pe, ev) -> (g, meta) for both, so this is a pure drop-in.
    # BUGFIX (live user-prompted verification, 2026-08-01): gp MUST be held fixed for the
    # "profile stage" (matching run_profile_checkpointed's own contract exactly -- there, g_in is a
    # separate scalar argument captured by closure, NEVER added to KNITRO's free-variable vector at
    # all). An earlier version of this function put ALL of w_start (including gp, w[1]) into
    # KN_add_vars with a +-30 box, letting KNITRO silently drift gp back toward its easy,
    # well-fitting calibration value instead of being held at the caller's fixed target -- caught via
    # a live LFD/gravity verification the user asked for after a suspiciously large (509x) A/B gap:
    # the "best" checkpoint's own gp had drifted from 0.9831 (the intended fixed value) back to
    # 0.9930 (~calibration), which is not a legitimate profile-stage comparison at all.
    gp_fixed = w_start[1]
    r_free_start = w_start[2:end]
    n_free = length(r_free_start)
    lp(xs...) = (println(xs...); flush(stdout))

    t_start = time()
    n_eval = Ref(0); n_grad_calls = Ref(0)
    trace = NamedTuple[]
    best = Ref{Union{Nothing,NamedTuple}}(nothing)
    last_w = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    last_ev = Ref{Any}(nothing)

    ev0 = evaluate_profiled_point(w_start, ctx, spec, pe)
    ev0.result.inner_status in (0, -100, -101, -103) || error("run_profiled_outer_search($label): start point not inner-feasible (status=$(ev0.result.inner_status))")
    lp("[$label] seed: gp_fixed=$gp_fixed  Delta_dual=$(ev0.result.Delta_dual)  status=$(ev0.result.inner_status)")

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt"))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", maxit_override === nothing ? 1_000_000 : maxit_override)
    KNITRO.KN_set_param_by_name(kc, "algorithm", 3)
    xIndices = KNITRO.KN_add_vars(kc, n_free)
    z_halfwidth = 30.0
    KNITRO.KN_set_var_lobnds_all(kc, r_free_start .- z_halfwidth)
    KNITRO.KN_set_var_upbnds_all(kc, r_free_start .+ z_halfwidth)
    KNITRO.KN_set_var_primal_init_values_all(kc, r_free_start)

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        r_free = collect(Float64, evalRequest.x)
        w = vcat(gp_fixed, r_free)
        ev = evaluate_profiled_point(w, ctx, spec, pe)
        if !(ev.result.inner_status in (0, -100, -101, -103)) || !isfinite(ev.result.Delta_dual)
            evalResult.obj[1] = 1e10
            n_eval[] += 1
            return 0
        end
        Δ = ev.result.Delta_dual
        evalResult.obj[1] = Δ
        n_eval[] += 1
        last_w[] = w; last_ev[] = ev
        t_el = time() - t_start
        verified = ev.result.primal_dual_gap < 1e-3 && ev.result.mean_m_resid < 1e-6 && ev.result.max_abs_moment_kkt_resid < 1e-3
        is_new_best = verified && (best[] === nothing || (ctx.find_smallest ? Δ < best[].Delta_dual : Δ > best[].Delta_dual))
        if is_new_best
            best[] = (w = copy(w), Delta_dual = Δ, t_elapsed = t_el, n_eval = n_eval[])
        end
        push!(trace, (idx = n_eval[], t_elapsed = t_el, Delta_dual = Δ, inner_status = ev.result.inner_status, verified = verified))
        if n_eval[] <= 5 || n_eval[] % 10 == 0
            lp("  [$label] eval $(n_eval[]) t=$(round(t_el,digits=1))s Delta=$Δ status=$(ev.result.inner_status)")
        end
        return 0
    end

    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        r_free = collect(Float64, evalRequest.x)
        w = vcat(gp_fixed, r_free)
        ev = (last_w[] !== nothing && last_w[] == w) ? last_ev[] : evaluate_profiled_point(w, ctx, spec, pe)
        g, meta = gradient_fn(w, ctx, spec, pe, ev)
        n_grad_calls[] += 1
        evalResult.objGrad .= g[2:end]   # drop gp component -- gp is fixed, not a KNITRO free variable (matches production's own gfull[2:end] convention)
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!)

    KNITRO.KN_solve(kc)
    wall_ext = time() - t_start
    nStatus_code, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    lp("[$label] PROFILED SEARCH DONE: status=$nStatus_code wall=$(round(wall_ext,digits=1))s n_eval=$(n_eval[]) n_grad=$(n_grad_calls[])")
    b = best[]
    b !== nothing && lp("  best: Delta=$(b.Delta_dual) found_at_eval=$(b.n_eval) t=$(b.t_elapsed)s")

    if trace_csv !== nothing
        CSV.write(trace_csv, DataFrame(trace))
    end

    return (label = label, knitro_status = nStatus_code, wall_ext = wall_ext, n_eval = n_eval[],
        n_grad_calls = n_grad_calls[], zfree_terminal = collect(Float64, xsol), best = b, trace = trace)
end
