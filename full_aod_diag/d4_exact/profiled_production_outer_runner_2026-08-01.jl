# ============================================================================
# Production outer bridge task (2026-08-01), §13: a production outer-runner
# SCAFFOLD supporting the live family combined outer vector, in two EXPLICIT
# modes -- not just the fixed-gp A/B harness (`run_profiled_family_outer_search`,
# `PROFILED_ALL_FAMILY_OUTER_AB_HARNESS_2026-08-01.jl`), which intentionally
# fixes gp and optimizes only the profiled A coordinates (a matched-
# parameterization EXPERIMENT, not the full production outer problem).
#
#   :fixed_gp_parameterization_ab   -- thin pass-through to the EXISTING,
#     already-gated harness, UNCHANGED. gp fixed, never a KNITRO free
#     variable. This mode's manifest free-coordinate list is exactly
#     `combined_outer_coordinate_names(fctx)[2:end]` (drops gp).
#
#   :production_bound_search        -- gp free WHEN the family/context's own
#     production convention treats it as free (task §13's actual production
#     problem). Mechanically: same cb_F!/cb_G! shape as the AB harness, but
#     (a) gp is a KNITRO free variable with its own bounds, (b) `cb_G!` does
#     NOT drop `g[1]` -- the full `shared_family_outer_gradient` output is
#     used, since that function's own docstring already promises
#     `grad[1] = dK*/dgp`. This is a MECHANICAL extension of the exact same
#     gradient call the AB harness already uses -- no new gradient formula.
#
# What this scaffold does NOT do (honestly, not silently): wire the full
# production KNITRO option stack (screens, exact cache, dual-bank policy,
# checkpointing/continuation) -- those are real, substantial production
# subsystems (`cm_dual_bank_production.jl`, `cm_exact_cache_production.jl`,
# `cm_screen_bridge.jl`, `cm_checkpoint.jl`) this task's file-ownership
# boundary does not license rewriting, and no live restricted-family
# evaluator exists yet to exercise them meaningfully through this bridge.
# Each mode's manifest below has an explicit `production_subsystems` field
# recording, per subsystem, whether it is `:inherited_from_ab_harness`,
# `:not_yet_wired`, or a concrete value -- never silently omitted.
# ADDITIVE ONLY.
# ============================================================================

isdefined(Main, :run_profiled_family_outer_search) ||
    error("profiled_production_outer_runner_2026-08-01.jl requires PROFILED_ALL_FAMILY_OUTER_AB_HARNESS_2026-08-01.jl to be included first.")
isdefined(Main, :stable_layout_digest) ||
    error("profiled_production_outer_runner_2026-08-01.jl requires profiled_stable_layout_digest_2026-08-01.jl to be included first.")

using KNITRO, Dates, CSV, DataFrames

const VALID_OUTER_RUNNER_MODES = (:fixed_gp_parameterization_ab, :production_bound_search)

"""
    OuterRunManifest

Immutable record of what a given outer run actually did -- task §13 ("each
mode must record its free-coordinate names in the manifest") + §14/§15's
comparability/plumbing needs. Every field is filled with either a real value
or an explicit placeholder Symbol (`:not_yet_wired`, `:unknown_pending_inner`)
-- never silently absent.
"""
struct OuterRunManifest
    mode::Symbol
    family::Symbol
    label::String
    free_coordinate_names::Vector{String}
    n_free::Int
    gp_free::Bool
    hessopt_tag::String
    knitro_opt_file::String
    stable_layout_digest::String
    economic_parameterization::Symbol   # :profiled_destination_scales (this whole branch) -- see §15
    production_subsystems::NamedTuple
    timestamp::String
end

"""
    default_production_subsystems_manifest() -> NamedTuple

Task §14's required fields, each honestly tagged (see file header). Populate
per-family concrete values as they become determinable; `:not_yet_wired`
must never be silently swapped for a made-up concrete value.
"""
default_production_subsystems_manifest() = (
    dual_bank_policy = :not_yet_wired,
    obj_x_reuse_policy = :inherited_from_ab_harness,   # AB harness's cb_G! reuses last_ev/last_fctx when w unchanged -- see run_profiled_family_outer_search
    exact_cache_policy = :not_yet_wired,
    screen_set = :not_yet_wired,
    restriction_backend = :not_yet_wired,
    solver_options_file = :inherited_from_ab_harness,
)

_combined_names_or_fallback(fctx) =
    isdefined(Main, :combined_outer_coordinate_names) ? combined_outer_coordinate_names(fctx) :
        ["gp"; ["r_free_$k" for k in 1:(outer_dim_profiled(profiled_outer_coordinate_layout(fctx)) - 1)]]

"""
    run_profiled_production_outer(mode, label, w_start; ctx, evaluate_fn, family_ctx_builder,
        maxtime_real=1800.0, hessopt_tag="sr1", maxit_override=nothing, trace_csv=nothing,
        gp_bounds_halfwidth=0.05) -> (result, manifest)

Task §13 entry point. `mode in VALID_OUTER_RUNNER_MODES`, throws on anything
else (no silent default mode).
"""
function run_profiled_production_outer(mode::Symbol, label::String, w_start::Vector{Float64}; ctx,
        evaluate_fn::Function, family_ctx_builder::Function,
        maxtime_real::Float64 = 1800.0, hessopt_tag::String = "sr1",
        maxit_override::Union{Nothing,Int} = nothing, trace_csv::Union{Nothing,AbstractString} = nothing,
        gp_bounds_halfwidth::Float64 = 0.05)
    mode in VALID_OUTER_RUNNER_MODES ||
        error("run_profiled_production_outer: mode=:$mode not in $VALID_OUTER_RUNNER_MODES")

    ev0 = evaluate_fn(w_start, ctx)
    fctx0 = family_ctx_builder(ctx, ev0)
    validate_family_layout_contract(fctx0)
    names = _combined_names_or_fallback(fctx0)
    digest = stable_layout_digest(fctx0)
    opt_file = joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt")

    if mode == :fixed_gp_parameterization_ab
        result = run_profiled_family_outer_search(label, w_start; ctx, evaluate_fn, family_ctx_builder,
            maxtime_real, hessopt_tag, maxit_override, trace_csv)
        manifest = OuterRunManifest(mode, family_kind(fctx0), label, names[2:end], length(names) - 1, false,
            hessopt_tag, opt_file, digest, :profiled_destination_scales,
            default_production_subsystems_manifest(), string(now()))
        return result, manifest
    end

    # mode == :production_bound_search
    result = _run_production_bound_search(label, w_start; ctx, evaluate_fn, family_ctx_builder,
        maxtime_real, hessopt_tag, maxit_override, trace_csv, gp_bounds_halfwidth)
    manifest = OuterRunManifest(mode, family_kind(fctx0), label, names, length(names), true,
        hessopt_tag, opt_file, digest, :profiled_destination_scales,
        default_production_subsystems_manifest(), string(now()))
    return result, manifest
end

"""
    _run_production_bound_search(label, w_start; ...) -> NamedTuple

Task §13's `:production_bound_search` mode: SAME cb_F!/cb_G! shape as
`run_profiled_family_outer_search` (`PROFILED_ALL_FAMILY_OUTER_AB_HARNESS_2026-08-01.jl`,
reproduced here rather than parameterized into that function so the
already-gated AB harness stays byte-for-byte unedited), except gp IS a free
KNITRO variable (own bounds, `gp_bounds_halfwidth` around the start value)
and `cb_G!` passes the FULL `shared_family_outer_gradient` output (`g[1]` =
dK*/dgp included, not dropped).
"""
function _run_production_bound_search(label::String, w_start::Vector{Float64}; ctx,
        evaluate_fn::Function, family_ctx_builder::Function,
        maxtime_real::Float64, hessopt_tag::String,
        maxit_override::Union{Nothing,Int}, trace_csv::Union{Nothing,AbstractString},
        gp_bounds_halfwidth::Float64)
    n_total = length(w_start)
    lp(xs...) = (println(xs...); flush(stdout))

    t_start = time()
    n_eval = Ref(0); n_grad_calls = Ref(0)
    trace = NamedTuple[]
    best = Ref{Union{Nothing,NamedTuple}}(nothing)
    last_w = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    last_ev = Ref{Any}(nothing)
    last_fctx = Ref{Any}(nothing)

    ev0 = evaluate_fn(w_start, ctx)
    ev0.result.inner_status in (0, -100, -101, -103) || error("_run_production_bound_search($label): start point not inner-feasible (status=$(ev0.result.inner_status))")
    fctx0 = family_ctx_builder(ctx, ev0)
    validate_family_layout_contract(fctx0)
    lp("[$label] :production_bound_search family=:$(family_kind(fctx0))  gp_start=$(w_start[1])  Delta_dual=$(ev0.result.Delta_dual)  status=$(ev0.result.inner_status)")

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt"))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", maxit_override === nothing ? 1_000_000 : maxit_override)
    KNITRO.KN_set_param_by_name(kc, "algorithm", 3)
    xIndices = KNITRO.KN_add_vars(kc, n_total)
    z_halfwidth = 30.0
    lo = copy(w_start) .- z_halfwidth; lo[1] = w_start[1] - gp_bounds_halfwidth
    hi = copy(w_start) .+ z_halfwidth; hi[1] = w_start[1] + gp_bounds_halfwidth
    KNITRO.KN_set_var_lobnds_all(kc, lo)
    KNITRO.KN_set_var_upbnds_all(kc, hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w_start)

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = collect(Float64, evalRequest.x)
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
        push!(trace, (idx = n_eval[], t_elapsed = t_el, Delta_dual = Δ, gp = w[1], inner_status = ev.result.inner_status, verified = verified))
        if n_eval[] <= 5 || n_eval[] % 10 == 0
            lp("  [$label] eval $(n_eval[]) t=$(round(t_el,digits=1))s Delta=$Δ gp=$(w[1]) status=$(ev.result.inner_status)")
        end
        return 0
    end

    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = collect(Float64, evalRequest.x)
        if last_w[] !== nothing && last_w[] == w
            ev = last_ev[]; fctx = last_fctx[]
        else
            ev = evaluate_fn(w, ctx); fctx = family_ctx_builder(ctx, ev)
        end
        g, meta = shared_family_outer_gradient(w, ctx, fctx, ev)
        n_grad_calls[] += 1
        evalResult.objGrad .= g   # FULL gradient, gp included -- gp is free here, unlike the AB harness
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!)

    KNITRO.KN_solve(kc)
    wall_ext = time() - t_start
    nStatus_code, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    lp("[$label] PRODUCTION BOUND SEARCH DONE: status=$nStatus_code wall=$(round(wall_ext,digits=1))s n_eval=$(n_eval[]) n_grad=$(n_grad_calls[])")
    b = best[]
    b !== nothing && lp("  best: Delta=$(b.Delta_dual) found_at_eval=$(b.n_eval) t=$(b.t_elapsed)s")

    if trace_csv !== nothing
        CSV.write(trace_csv, DataFrame(trace))
    end

    return (label = label, family = family_kind(fctx0), knitro_status = nStatus_code, wall_ext = wall_ext,
        n_eval = n_eval[], n_grad_calls = n_grad_calls[], w_terminal = collect(Float64, xsol),
        best = b, trace = trace)
end
