# fix/profiled-functional-readiness-closeout-2026-08-03, continuation of task §8.4: wire the
# free-eta_nu evaluators/gradients (profiled_zc_free_eta_2026-08-04.jl) into a genuine production
# KNITRO driver, so origin_ZC/CM_plus_ZC can search nu jointly with (gp; A_free) instead of holding
# it fixed. ADDITIVE ONLY -- `run_profiled_upper_constrained`/`_write_profiled_constrained_checkpoint`
# (profiled_production_outer_constrained_2026-08-02.jl) are NOT modified. That driver now serves 3
# genuinely production_ready REDUCED families (unrestricted/flexible_cm/common_frechet as of this
# session); this file adds a NEW, parallel driver rather than risk regressing it. Same discipline
# this session's own warm-start variants used (e.g. reduced_cm_base_state_warm): copy the tested
# function's structure, change only what genuinely differs, do not touch the original.
#
# What genuinely differs from run_profiled_upper_constrained:
#   1. Outer vector is [gp; A_free; eta_nu] (ZCFreeNuOuterLayout, profiled_zc_free_eta_2026-08-04.jl)
#      instead of [gp; A_free] -- D2 = length(w0_econ) + n_eta.
#   2. evaluate_fn_free_nu/gradient_fn_free_nu take (w_econ, eta_nu, fctx, pes)/(w_econ, eta_nu,
#      ctx, fctx, ev) -- the 4-arg free-eta signatures, not the 2-arg fixed-nu ones. Caller passes
#      evaluate_profiled_originzc_point/reduced_originzc_outer_gradient_with_eta (or the cmzc
#      pair) explicitly, same "no default, caller states which family" discipline the base driver
#      already uses for evaluate_fn.
#   3. eta bounds are REQUIRED (no default) and must come from this session's own traced FULL
#      production convention (originzc_default_nu_bounds/meanzc_default_nu_bounds,
#      cm_originzc_config.jl/cm_meanzc_config.jl) -- a real, already-production-tested "deliberately
#      WIDE, 4x safety margin on the hard finite-support interval" policy, not an invented number.
#      Multiple-dispatches correctly on OriginByPowerLayout vs SharedByPowerLayout via fctx.zc_layout.
#   4. CMCheckpointV11's own eta_nu::Vector{Float64} field (already present in the schema, always
#      written as Float64[] by the base driver) is genuinely POPULATED here -- this is the "eta
#      generation cache/checkpoint wiring" task §8.4 asked for. Resume restores eta_nu as the KNITRO
#      warm-start value for the eta block, exactly as g/zfree already restore gp/A_free.
isdefined(Main, :run_profiled_upper_constrained) ||
    error("profiled_zc_free_nu_production_driver_2026-08-04.jl requires profiled_production_outer_constrained_2026-08-02.jl to be included first.")
isdefined(Main, :ZCFreeNuOuterLayout) ||
    error("profiled_zc_free_nu_production_driver_2026-08-04.jl requires profiled_zc_free_eta_2026-08-04.jl to be included first.")
isdefined(Main, :decode_w_mode_to_native) ||
    error("profiled_zc_free_nu_production_driver_2026-08-04.jl requires profiled_coordinate_mode_dispatch_2026-08-04.jl to be included first (task §6.1 continuation, powered A-coordinate mode for origin_zc/cm_meanzc).")
using KNITRO

"""
    _write_profiled_constrained_checkpoint_free_nu(path, label, fctx, ctx, delta, config, w_econ,
        eta_nu, best_feasible, n_eval, n_grad, wall_elapsed, wall_budget_remaining) -> Nothing

Verbatim copy of `_write_profiled_constrained_checkpoint`'s own body (same field-reuse discipline,
same hasproperty-guarded draw/W/destination reads), with ONE real difference: `eta_nu` is the
actual live free-nu value (in `CMCheckpointV11.eta_nu`), not the base function's hardcoded
`Float64[]`. `best_feasible` here is expected to carry `w_econ`/`eta_nu` fields (this driver's own
richer incumbent shape, analogous to the base driver's `(gp,w,Delta,n_eval,t_elapsed)`), not the
base driver's own shape -- the field is `::Any` on both sides so this is valid.
"""
function _write_profiled_constrained_checkpoint_free_nu(path::AbstractString, label::String, fctx, ctx,
        delta::Float64, config, w_econ::Vector{Float64}, eta_nu::Vector{Float64}, best_feasible,
        n_eval::Int, n_grad::Int, wall_elapsed::Float64, wall_budget_remaining::Float64)
    fam = family_kind(fctx)
    has_draws = hasproperty(ctx, :draw_meta) && ctx.draw_meta !== nothing
    draw_cksum_u = has_draws ? ctx.draw_meta.checksum_uniform : ""
    draw_cksum_t = has_draws ? ctx.draw_meta.checksum_transformed : ""
    ckpt = CMCheckpointV11(CM_CHECKPOINT_SCHEMA_V11, "profiled_constrained_freenu_$(label)", label, :cm_upper,
        ctx.find_smallest, delta,
        hasproperty(ctx, :W) ? ctx.W : 0,
        hasproperty(ctx, :draw_seed) ? ctx.draw_seed : 0,
        hasproperty(ctx, :draw_design) ? ctx.draw_design : :none,
        draw_cksum_u, draw_cksum_t,
        0, Float64[], :anchored, :equal, :orthonormal, :structured, :cplus,
        Symbol(fam), 0, 0, :direct, 1,
        w_econ[1], w_econ[2:end], collect(Float64, eta_nu), zeros(0, 0), Float64[], Dict{Int,Float64}(),
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
    run_profiled_upper_constrained_free_nu(label, w0_econ, eta_nu0; fctx, evaluate_fn_free_nu,
        gradient_fn_free_nu, pes, ctx, pe, eta_bounds, delta=1.0, maxtime_real=180.0,
        hessopt_tag="sr1", z_halfwidth=30.0, gp_lo=ctx.bounds.γp_lo, gp_hi=ctx.bounds.γp_hi,
        use_screen=false, cache=nothing, trace_csv=nothing, verbose=true, checkpoint_path=nothing,
        checkpoint_interval_s=60.0, resume_from=nothing) -> NamedTuple

Constrained-upper-bound KNITRO outer loop for origin_ZC/CM_plus_ZC with eta_nu as a genuine free
outer coordinate, structurally mirroring `run_profiled_upper_constrained` (same objective/
constraint/box/algorithm shape: minimize w[1]=gp subject to Delta_dual(w)<=delta, Direct+SR1) but
over the EXTENDED vector `[gp; A_free; eta_nu]` (`ZCFreeNuOuterLayout`), calling the 4-arg free-eta
evaluator/gradient (`evaluate_fn_free_nu`/`gradient_fn_free_nu`, e.g.
`evaluate_profiled_originzc_point`/`reduced_originzc_outer_gradient_with_eta` or the cmzc pair --
caller states which family explicitly, no default, matching the base driver's own `evaluate_fn`
discipline) instead of the 2-arg fixed-nu ones.

`eta_bounds::Vector{NTuple{2,Float64}}` (length `length(eta_nu0)`) is REQUIRED -- pass
`originzc_default_nu_bounds(ctx, fctx.zc_layout)` (dispatches correctly on OriginByPowerLayout vs
SharedByPowerLayout), the real already-production-tested FULL-side convention, not an invented
number. After solving, `originzc_verify_box_not_binding(eta_star, eta_bounds)` should be checked by
the caller (that function's own docstring: widen and re-solve if any level is at its boundary,
never trust a boundary solution silently) -- this driver does NOT auto-widen/re-solve, it only
returns `eta_bounds` in its result so the caller can run that check.

Checkpoint/resume: `CMCheckpointV11.eta_nu` is genuinely populated (task §8.4) via
`_write_profiled_constrained_checkpoint_free_nu`. On resume, `w0_econ = vcat(resumed.g,
resumed.zfree)` and `eta_nu0 = resumed.eta_nu` are BOTH restored from the checkpoint (the base
driver's own resume only restores the former since it never writes a real eta_nu). The same five
independent mismatch axes the base driver validates (namespace, find_smallest, W, delta, draw
design+seed+checksums) are validated here too, PLUS a sixth: `length(resumed.eta_nu) ==
length(eta_nu0)` (refuses resuming a checkpoint written with a different eta dimension, e.g. a
different D or K_mean).
"""
function run_profiled_upper_constrained_free_nu(label::String, w0_econ::Vector{Float64}, eta_nu0::Vector{Float64};
        fctx, evaluate_fn_free_nu::Function, gradient_fn_free_nu::Function, pes,
        ctx, pe::PivotGravityElimOnRetained, eta_bounds::Vector{NTuple{2,Float64}},
        delta::Float64 = 1.0, maxtime_real::Float64 = 180.0, hessopt_tag::String = "sr1",
        z_halfwidth::Float64 = 30.0,
        gp_lo::Float64 = ctx.bounds.γp_lo, gp_hi::Float64 = ctx.bounds.γp_hi,
        use_screen::Bool = false, cache = nothing,
        trace_csv::Union{Nothing,AbstractString} = nothing,
        verbose::Bool = true,
        checkpoint_path::Union{Nothing,AbstractString} = nothing,
        checkpoint_interval_s::Float64 = 60.0,
        resume_from::Union{Nothing,AbstractString} = nothing,
        a_coordinate_mode::Symbol = :profiled_pivot_anchor_relative)
    lp(xs...) = (println(xs...); flush(stdout))
    fam = family_kind(fctx)
    validate_mode_family_compatibility(a_coordinate_mode, fam)
    theta = a_coordinate_mode == :profiled_powered_relative_A ? cm_fixed_theta(ctx) : NaN
    xy = a_coordinate_mode == :profiled_powered_relative_A ? precompute_cm_aspace_xy(ctx) : nothing
    config = build_profiled_production_config(fctx; economic_parameterization = :profiled_destination_scales,
        a_coordinate_mode = a_coordinate_mode)
    n_eta = length(eta_nu0)
    length(eta_bounds) == n_eta ||
        error("run_profiled_upper_constrained_free_nu($label): length(eta_bounds)=$(length(eta_bounds)) != length(eta_nu0)=$n_eta")

    n_eval_seed = 0; n_grad_seed = 0; best_feasible_seed = nothing; prior_wall = 0.0
    if resume_from !== nothing
        resumed = load_cm_checkpoint_v11(resume_from)
        assert_checkpoint_compatible(config, resumed.checkpoint_namespace)
        resumed.find_smallest ||
            error("run_profiled_upper_constrained_free_nu($label): checkpoint at $resume_from was written " *
                  "with find_smallest=false, but this driver only ever solves the upper/minimize-gp " *
                  "direction -- refusing to resume a different direction under the same driver.")
        W_cur = hasproperty(ctx, :W) ? ctx.W : 0
        resumed.W == W_cur ||
            error("run_profiled_upper_constrained_free_nu($label): checkpoint W=$(resumed.W) != this call's ctx W=$(W_cur) -- refusing to resume across a different W.")
        isapprox(resumed.delta, delta; atol = 1e-12) ||
            error("run_profiled_upper_constrained_free_nu($label): checkpoint delta=$(resumed.delta) != this call's delta=$delta -- refusing to resume across a different delta.")
        has_draws_cur = hasproperty(ctx, :draw_meta) && ctx.draw_meta !== nothing
        has_draws_ckpt = !isempty(resumed.draw_checksum_uniform) || !isempty(resumed.draw_checksum_transformed)
        if has_draws_cur != has_draws_ckpt
            error("run_profiled_upper_constrained_free_nu($label): checkpoint has_draws=$has_draws_ckpt but this call's ctx has_draws=$has_draws_cur -- refusing to resume across a draw-based context and a draw-free one.")
        end
        if has_draws_cur
            draw_design_cur = hasproperty(ctx, :draw_design) ? ctx.draw_design : :none
            draw_seed_cur = hasproperty(ctx, :draw_seed) ? ctx.draw_seed : 0
            (resumed.draw_design == draw_design_cur && resumed.draw_seed == draw_seed_cur) ||
                error("run_profiled_upper_constrained_free_nu($label): checkpoint draw_design/draw_seed=" *
                      ":$(resumed.draw_design)/$(resumed.draw_seed) != this call's ctx=:$(draw_design_cur)/$(draw_seed_cur) -- refusing to resume across a different draw configuration.")
            (ctx.draw_meta.checksum_uniform == resumed.draw_checksum_uniform &&
             ctx.draw_meta.checksum_transformed == resumed.draw_checksum_transformed) ||
                error("run_profiled_upper_constrained_free_nu($label): draw checksum MISMATCH on resume -- refusing to resume.")
        else
            lp("[$label] resume: no draw metadata on either side (draw-free context, e.g. D4) -- skipping draw-checksum validation")
        end
        length(vcat(resumed.g, resumed.zfree)) == length(w0_econ) ||
            error("run_profiled_upper_constrained_free_nu($label): checkpoint economic-vector length=$(length(vcat(resumed.g, resumed.zfree))) != this call's w0_econ length=$(length(w0_econ)) -- refusing to resume across a different layout.")
        length(resumed.eta_nu) == n_eta ||
            error("run_profiled_upper_constrained_free_nu($label): checkpoint eta_nu length=$(length(resumed.eta_nu)) != this call's eta_nu0 length=$n_eta -- refusing to resume across a different eta dimension.")
        w0_econ = vcat(resumed.g, resumed.zfree)
        eta_nu0 = collect(Float64, resumed.eta_nu)
        n_eval_seed = resumed.n_eval; n_grad_seed = resumed.n_grad
        best_feasible_seed = resumed.best_feasible
        prior_wall = resumed.wall_elapsed
        lp("[$label] RESUMING from $resume_from (n_eval=$(resumed.n_eval) n_grad=$(resumed.n_grad) " *
           "wall_elapsed=$(round(resumed.wall_elapsed,digits=1))s gp=$(w0_econ[1]) eta_nu=$eta_nu0)")
    end

    layout = ZCFreeNuOuterLayout(length(w0_econ), n_eta)
    # task §6.1 continuation (2026-08-04): same boundary-only mode-awareness as
    # run_profiled_upper_constrained -- w0_econ here is the function's OWN parameter (always native,
    # unchanged contract) UNLESS it was just overwritten by the resume branch above, in which case
    # it is already in a_coordinate_mode units (checkpoints store KNITRO's raw econ block; the
    # namespace check already confirmed this call's mode matches the checkpoint's).
    w0_econ_mode = resume_from === nothing ? encode_w_native_to_mode(w0_econ, pe, a_coordinate_mode, theta, xy) : w0_econ
    w0_econ_native_for_bounds = decode_w_mode_to_native(w0_econ_mode, pe, a_coordinate_mode, theta, xy)
    r_lo_native = w0_econ_native_for_bounds[2:end] .- z_halfwidth
    r_hi_native = w0_econ_native_for_bounds[2:end] .+ z_halfwidth
    r_lo_mode, r_hi_mode = mode_bounds(r_lo_native, r_hi_native, pe, a_coordinate_mode, theta, xy)
    w0 = pack_free_nu_outer(w0_econ_mode, eta_nu0)
    D2 = length(w0)
    w_lo = vcat(gp_lo, r_lo_mode, [b[1] for b in eta_bounds])
    w_hi = vcat(gp_hi, r_hi_mode, [b[2] for b in eta_bounds])

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt"))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
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

    function solve_at(w_full_mode::Vector{Float64})
        w_econ_mode, eta_nu = split_free_nu_outer(w_full_mode, layout)
        w_econ = decode_w_mode_to_native(collect(Float64, w_econ_mode), pe, a_coordinate_mode, theta, xy)
        w_full = vcat(w_econ, eta_nu)
        if use_screen
            try
                profiled_cm_screen_precheck!(collect(Float64, w_econ), ctx, pe; use_witness = false)
            catch e
                e isa CMExpectedSolveFailure || rethrow()
                return nothing
            end
        end
        if cache !== nothing
            key = ProfiledCMProductionEvalKey(:profiled_destination_scales, w_full, collect(Float64, eta_nu),
                ctx.δ, ctx.find_smallest, fam, stable_layout_digest(fctx), "")
            ev, _ = profiled_cm_cache_lookup_or_compute!(cache, key,
                () -> (ev = evaluate_fn_free_nu(w_econ, eta_nu, fctx, pes); (ev, (inner_status = ev.result.inner_status,))))
            return ev
        end
        return evaluate_fn_free_nu(w_econ, eta_nu, fctx, pes)
    end

    ev0 = solve_at(w0)
    ev0 !== nothing && ev0.result.inner_status in (0, -100, -101, -103) ||
        error("run_profiled_upper_constrained_free_nu($label): start point not inner-feasible/not screen-passing")
    lp("[$label] CONSTRAINED start: gp0=$(w0[1]) eta_nu0=$eta_nu0 Delta0=$(-ev0.result.zeta) status=$(ev0.result.inner_status)")

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
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
            evalResult.c[1] = 1e10
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
        w_econ_i, eta_nu_i = split_free_nu_outer(w, layout)
        is_new_best = feasible && verified && (best_feasible[] === nothing || w[1] < best_feasible[].gp)
        if is_new_best
            best_feasible[] = (gp = w[1], w_econ = collect(w_econ_i), eta_nu = collect(eta_nu_i), Delta = Δ, n_eval = n_eval[], t_elapsed = t_el)
        end
        push!(trace, (idx = n_eval[], t_elapsed = t_el, gp = w[1], Delta = Δ, feasible = feasible, verified = verified))
        if checkpoint_path !== nothing && (time() - t_last_ckpt[] >= checkpoint_interval_s || is_new_best)
            _write_profiled_constrained_checkpoint_free_nu(checkpoint_path, label, fctx, ctx, delta, config,
                collect(Float64, w_econ_i), collect(Float64, eta_nu_i), best_feasible[], n_eval[], n_grad[], prior_wall + t_el, maxtime_real - t_el)
            t_last_ckpt[] = time()
        end
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
            try
                ev = solve_at(w)
            catch e
                e isa CMExpectedSolveFailure || rethrow()
                evalResult.objGrad .= 0.0; evalResult.objGrad[1] = 1.0
                evalResult.jac .= 0.0
                verbose && lp("  [$label] grad $(n_grad[]+1) t=$(round(time()-t_start,digits=1))s REJECTED (CMExpectedSolveFailure, re-solve path) call_dur=$(round(time()-t_call0,digits=1))s")
                return 0
            end
        end
        t_solve_done = time()
        w_econ_i_mode, eta_nu_i = split_free_nu_outer(collect(Float64, w), layout)
        # task §6.1 continuation fix: decode the econ block to native before calling
        # gradient_fn_free_nu (which perturbs r_free directly, same as shared_family_outer_gradient),
        # then rescale only the returned A-block portion (g[2:n_econ]) back to mode units -- gp
        # (g[1]) and the eta_nu block (g[n_econ+1:end]) are coordinate-mode-independent.
        w_econ_i = decode_w_mode_to_native(w_econ_i_mode, pe, a_coordinate_mode, theta, xy)
        g_native, meta = gradient_fn_free_nu(w_econ_i, eta_nu_i, ctx, fctx, ev)
        length(g_native) == D2 ||
            error("run_profiled_upper_constrained_free_nu($label): gradient_fn_free_nu returned length $(length(g_native)) but D2=$D2 -- combined-gradient shape must match [gp;A_free;eta_nu]")
        n_econ = length(w_econ_i_mode)
        g = vcat(g_native[1], rescale_gradient_for_mode(g_native[2:n_econ], a_coordinate_mode, theta), g_native[n_econ+1:end])
        n_grad[] += 1
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = 1.0
        evalResult.jac .= g
        verbose && lp("  [$label] grad $(n_grad[]) t=$(round(time()-t_start,digits=1))s reused_from_cbF=$reused solve_dur=$(round(t_solve_done-t_call0,digits=1))s grad_engine_dur=$(round(time()-t_solve_done,digits=1))s")
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)

    KNITRO.KN_solve(kc)
    wall = prior_wall + (time() - t_start)
    nStatus, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    xsol_econ, xsol_eta = split_free_nu_outer(collect(Float64, xsol), layout)
    if checkpoint_path !== nothing
        _write_profiled_constrained_checkpoint_free_nu(checkpoint_path, label, fctx, ctx, delta, config,
            collect(Float64, xsol_econ), collect(Float64, xsol_eta), best_feasible[], n_eval[], n_grad[], wall, max(0.0, maxtime_real - (time() - t_start)))
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
    b !== nothing && lp("  best: gp=$(b.gp) eta_nu=$(b.eta_nu) Delta=$(b.Delta) found_at_eval=$(b.n_eval) t=$(b.t_elapsed)s")

    return (label = label, family = fam, knitro_status = nStatus, wall = wall,
        n_eval = n_eval[], n_grad = n_grad[], best = b, xsol_econ = collect(xsol_econ),
        xsol_eta = collect(xsol_eta), eta_bounds = eta_bounds, trace = trace)
end
