# ============================================================================
# Claude Code task 2026-08-01 (parallel outer-gradient workstream), §17: the
# matched, all-family outer-search A/B harness. Generalizes the already-gated
# unrestricted-only `run_profiled_outer_search`
# (profiled_outer_ab_harness_2026-08-01.jl) to any family satisfying the
# five-accessor contract, by taking the family's own `evaluate_fn` and
# `family_ctx_builder` as arguments instead of hard-coding
# `evaluate_profiled_point`, and by computing the PROFILED arm's gradient
# through `shared_family_outer_gradient` UNCONDITIONALLY -- every family's
# profiled arm uses the literal same gradient method (task §8/§17: "The
# harness must explicitly hold `gp` fixed"; also matches mission's "one
# shared economic A/gp gradient engine" requirement at the harness level, not
# just at the unit-test level).
#
# The FULL/reference arm stays exactly what task §17 asks for: whatever
# existing full-gamma-normalized production driver a family already uses
# ("same outer solver", "same inner warm-start policy", "same restriction
# backend") -- this file does NOT reimplement it; callers pass their own
# `full_formulation_runner` closure. For the unrestricted family this is
# `run_profile_checkpointed` (task's own §14-era note, reused unmodified,
# same as the original harness).
#
# STATUS 2026-08-01: only the unrestricted family has a real `evaluate_fn`
# available in this worktree (`evaluate_profiled_point`). The four restricted
# families' own evaluators are owned by the separate, still-in-flight inner
# workstream (architecture/profiled-restricted-inner-endtoend-2026-08-01 or
# its descendant) and are NOT wired here -- `restricted_family_evaluator_
# not_ready` below is an explicit, loud placeholder (never a silent
# fallback to the unrestricted evaluator) so a caller cannot accidentally run
# a "restricted family" A/B that is secretly unrestricted. Once the inner
# branch exposes a real evaluator per family, only the four
# `build_<family>_ab_arm` closures at the bottom of this file need a real
# `evaluate_fn`; `run_profiled_family_outer_search` itself needs no change.
# ADDITIVE ONLY.
# ============================================================================

isdefined(Main, :shared_family_outer_gradient) || error("PROFILED_ALL_FAMILY_OUTER_AB_HARNESS_2026-08-01.jl requires profiled_shared_economic_gradient_engine_2026-08-01.jl to be included first.")
isdefined(Main, :validate_family_layout_contract) || error("PROFILED_ALL_FAMILY_OUTER_AB_HARNESS_2026-08-01.jl requires profiled_outer_gradient_layout_contract_2026-08-01.jl to be included first.")

using KNITRO, Dates, CSV, DataFrames

"""
    restricted_family_evaluator_not_ready(family::Symbol) -> Nothing

Loud placeholder for the four restricted families' own `evaluate_fn` (task
§19: "If an inner bug is discovered, provide a minimal reproducer to the
inner workstream rather than fixing it here" -- symmetrically, this
workstream does not silently stand up a fake restricted-family evaluator
either). Calling `run_profiled_family_outer_search` for a restricted family
before the inner branch's live evaluator is wired throws this, by design.
"""
function restricted_family_evaluator_not_ready(family::Symbol)
    error("restricted_family_evaluator_not_ready(:$family): the inner workstream " *
          "(architecture/profiled-restricted-inner-endtoend-2026-08-01 or its descendant) " *
          "has not yet exposed a live evaluate_fn for family :$family. Wire a real evaluator " *
          "(same NamedTuple shape as evaluate_profiled_point's return) and pass it as " *
          "`evaluate_fn` -- do not substitute the unrestricted evaluator.")
end

"""
    run_profiled_family_outer_search(label, w_start; ctx, evaluate_fn, family_ctx_builder,
        maxtime_real=1800.0, hessopt_tag="sr1", maxit_override=nothing, trace_csv=nothing) -> NamedTuple

Family-generic profiled outer KNITRO loop (task §17). `evaluate_fn(w, ctx) ->
ev`-shaped NamedTuple (same fields `evaluate_profiled_point` returns: at
least `result` with `.inner_status`/`.Delta_dual`/`.primal_dual_gap`/
`.mean_m_resid`/`.max_abs_moment_kkt_resid`, plus whatever `family_ctx_builder`
and `shared_family_outer_gradient` need). `family_ctx_builder(ctx, ev) -> fctx`
must return an object satisfying the five-accessor contract (task §7);
`spec`/`pe` are read off it via `profiled_anchor_spec`/
`profiled_outer_coordinate_layout`, never passed separately, so there is no
way for the harness to silently use a stale spec/pe (task §9's own
requirement, `pe.spec === spec`, is enforced by `validate_family_layout_contract`
on every gradient call). The gradient is ALWAYS `shared_family_outer_gradient`
-- no swappable `gradient_fn` argument (unlike the unrestricted-only
predecessor harness) -- so every family's profiled arm provably uses the one
shared A/gp method.
"""
function run_profiled_family_outer_search(label::String, w_start::Vector{Float64}; ctx,
        evaluate_fn::Function, family_ctx_builder::Function,
        maxtime_real::Float64 = 1800.0, hessopt_tag::String = "sr1",
        maxit_override::Union{Nothing,Int} = nothing, trace_csv::Union{Nothing,AbstractString} = nothing)
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
    last_fctx = Ref{Any}(nothing)

    ev0 = evaluate_fn(w_start, ctx)
    ev0.result.inner_status in (0, -100, -101, -103) || error("run_profiled_family_outer_search($label): start point not inner-feasible (status=$(ev0.result.inner_status))")
    fctx0 = family_ctx_builder(ctx, ev0)
    validate_family_layout_contract(fctx0)  # fail loudly before spending any KNITRO wall-clock
    pe = profiled_outer_coordinate_layout(fctx0)
    lp("[$label] family=:$(family_kind(fctx0))  seed: gp_fixed=$gp_fixed  Delta_dual=$(ev0.result.Delta_dual)  status=$(ev0.result.inner_status)")

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
        w = vcat(gp_fixed, r_free)   # gp held fixed -- task §17's own requirement, never a free KNITRO variable
        ev = evaluate_fn(w, ctx)
        if !(ev.result.inner_status in (0, -100, -101, -103)) || !isfinite(ev.result.Delta_dual)
            evalResult.obj[1] = 1e10
            n_eval[] += 1
            return 0
        end
        Δ = ev.result.Delta_dual
        evalResult.obj[1] = Δ
        n_eval[] += 1
        last_w[] = w; last_ev[] = ev; last_fctx[] = family_ctx_builder(ctx, ev)
        t_el = time() - t_start
        verified = ev.result.primal_dual_gap < 1e-3 && ev.result.mean_m_resid < 1e-6 && ev.result.max_abs_moment_kkt_resid < 1e-3
        find_smallest = hasproperty(ctx, :find_smallest) ? ctx.find_smallest : true
        is_new_best = verified && (best[] === nothing || (find_smallest ? Δ < best[].Delta_dual : Δ > best[].Delta_dual))
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
        if last_w[] !== nothing && last_w[] == w
            ev = last_ev[]; fctx = last_fctx[]
        else
            ev = evaluate_fn(w, ctx); fctx = family_ctx_builder(ctx, ev)
        end
        g, meta = shared_family_outer_gradient(w, ctx, fctx, ev)
        n_grad_calls[] += 1
        evalResult.objGrad .= g[2:end]   # drop gp component -- gp is fixed, not a KNITRO free variable
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!)

    KNITRO.KN_solve(kc)
    wall_ext = time() - t_start
    nStatus_code, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    lp("[$label] PROFILED FAMILY SEARCH DONE: status=$nStatus_code wall=$(round(wall_ext,digits=1))s n_eval=$(n_eval[]) n_grad=$(n_grad_calls[])")
    b = best[]
    b !== nothing && lp("  best: Delta=$(b.Delta_dual) found_at_eval=$(b.n_eval) t=$(b.t_elapsed)s")

    if trace_csv !== nothing
        CSV.write(trace_csv, DataFrame(trace))
    end

    return (label = label, family = family_kind(fctx0), knitro_status = nStatus_code, wall_ext = wall_ext,
        n_eval = n_eval[], n_grad_calls = n_grad_calls[], zfree_terminal = collect(Float64, xsol),
        best = b, trace = trace)
end

# ----------------------------------------------------------------------------
# Family arm builders (task §17's "Required families: unrestricted, flexible
# CM, ZC-only, CM+ZC" -- common Frechet "may be added afterward"). Each
# returns (evaluate_fn, family_ctx_builder) ready to pass to
# `run_profiled_family_outer_search`.
# ----------------------------------------------------------------------------

"unrestricted_ab_arm(ctx, spec, pe) -> (evaluate_fn, family_ctx_builder) -- the one REAL, ready-now family."
function unrestricted_ab_arm(ctx, spec::AnchorSpec, pe::PivotGravityElimOnRetained)
    evaluate_fn = (w, c) -> evaluate_profiled_point(w, c, spec, pe)
    family_ctx_builder = (c, ev) -> build_unrestricted_family_ctx(c, spec, pe, ev)
    return evaluate_fn, family_ctx_builder
end

"flexible_cm_ab_arm() -- NOT READY: inner evaluator not yet exposed. See restricted_family_evaluator_not_ready."
flexible_cm_ab_arm() = ((w, c) -> restricted_family_evaluator_not_ready(:flexible_CM),
    (c, ev) -> restricted_family_evaluator_not_ready(:flexible_CM))
"common_frechet_ab_arm() -- NOT READY: inner evaluator not yet exposed."
common_frechet_ab_arm() = ((w, c) -> restricted_family_evaluator_not_ready(:common_Frechet),
    (c, ev) -> restricted_family_evaluator_not_ready(:common_Frechet))
"zc_only_ab_arm() -- NOT READY: inner evaluator not yet exposed."
zc_only_ab_arm() = ((w, c) -> restricted_family_evaluator_not_ready(:ZC_only),
    (c, ev) -> restricted_family_evaluator_not_ready(:ZC_only))
"cm_plus_zc_ab_arm() -- NOT READY: inner evaluator not yet exposed."
cm_plus_zc_ab_arm() = ((w, c) -> restricted_family_evaluator_not_ready(:CM_plus_ZC),
    (c, ev) -> restricted_family_evaluator_not_ready(:CM_plus_ZC))
