# ================================================================================================
# Production wiring for OriginZCOperatorState (cm_originzc_lookup_kernels.jl) --
# port/shared-inner-fg-operator-and-verification-2026-07-26, Job 1 §3 (ZC only, `G=[E|Z]`).
# Mirrors cm_lookup_production.jl's structure exactly (KNITRO callback registration, Hessian
# adapter pattern, `inner_loop_internal_*` contract) so it reads as "the same pattern applied to
# origin-ZC", not a new design.
#
# SCOPE: opt-in only (`fg_backend=:operator` on `OriginZCCoreHessCtx`, default stays
# `:dense_reference`) until this branch's own D=4/D=20 correctness + performance gates pass (task
# §5.6 flip rule) -- see docs/ORIGIN_ZC_OPERATOR_FG_PORT_2026-07-26.md for gate results and the
# flip decision.
# ================================================================================================

isdefined(Main, :OriginZCOperatorState) || include(joinpath(@__DIR__, "cm_originzc_lookup_kernels.jl"))

"_adapt_hess_cb_for_originzc_operator(hess_cb): same call-site adapter cm_lookup_production.jl uses -- unwraps `userParams::OriginZCOperatorState` to `st.obj` before forwarding to the UNMODIFIED existing Hessian closure (`archA_partitioned_hess_cb_builder(octx)`, which itself expects `userParams` to be the dense `obj`)."
_adapt_hess_cb_for_originzc_operator(hess_cb) =
    (kc, cb, evalRequest, evalResult, userParams) -> hess_cb(kc, cb, evalRequest, evalResult, userParams.obj)

"""
    inner_loop_KNITRO_originzc_operator(obj, st::OriginZCOperatorState; hess_cb_builder) -> (nStatus, objSol, x, lambda_, n_fg, n_hess)

Mirrors `inner_loop_KNITRO_cmlookup_production` exactly.
"""
function inner_loop_KNITRO_originzc_operator(obj, st::OriginZCOperatorState; hess_cb_builder)
    CS.guard_enter_inner_solve!()
    try
        kc = KNITRO.KN_new()
        KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
        KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
        KNITRO.KN_set_var_primal_init_values_all(kc, CS.inner_loop_initial_values(obj))

        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], _callbackEvalFG_inner_originzc_operator!)
        KNITRO.KN_set_cb_user_params(kc, cb, st)
        KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

        n_hess = Ref(0)
        if KNITRO.KN_get_int_param(kc, "hessopt") == 1
            hess_cb_raw = hess_cb_builder(obj)
            hess_cb_adapted = (kc2, cb2, evalRequest, evalResult, userParams) -> begin
                r = _adapt_hess_cb_for_originzc_operator(hess_cb_raw)(kc2, cb2, evalRequest, evalResult, userParams)
                n_hess[] += 1
                return r
            end
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, hess_cb_adapted)
        end
        if obj.complement_index != [0 0]
            CS.inner_loop_complementarity_constraints(kc, obj)
        end

        KNITRO.KN_solve(kc)
        nStatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
        CS.INNER_ITERS_TOTAL[] += CS._kn_num_iters(kc)
        n_fg = st.n_fg_calls
        KNITRO.KN_free(kc)

        return nStatus, objSol, x, lambda_, n_fg, n_hess[]
    finally
        CS.guard_exit_inner_solve!()
    end
end

"KNITRO FG callback protocol: this codebase's `KN_add_eval_callback(kc, true, ...)` registration calls ONE combined callback for both f and g every invocation -- mirrors `_callbackEvalFG_inner_cmlookup!` exactly (see cm_lookup_production.jl / cm_frechet_lookup_production.jl for the same pattern and the KNITRO-wiring bug this must not repeat)."
function _callbackEvalFG_inner_originzc_operator!(kc, cb, evalRequest, evalResult, userParams)
    st = userParams
    x = evalRequest.x
    f = st(x, evalResult.objGrad)
    evalResult.obj[1] = f <= st.obj.lower_limit ? -KNITRO.KN_INFINITY : f
    return 0
end

"""
    inner_loop_internal_originzc_operator(obj, θ_ext, octx::OriginZCCoreHessCtx) -> (K, x, nStatus, n_fg, n_hess)

Mirrors `inner_loop_internal_cmlookup_production`'s contract exactly (same return shape as
`inner_loop_internal_archgeneric`) so `archOZ_base_state`/`archOZ_verified_state` can dispatch to
either with no other code change. Still calls `obj.moments!` once per solve (builds dense `obj.H`,
which the Hessian callback's H_ER/H_RR cross-terms and the tied-winner dense-fallback path both
still need -- see economic_operator.jl's header for why this is not eliminated, out of scope for
this task) -- what changes is that the FG callback itself never reads `obj.H` in the normal
(non-fallback) case.

`octx.fg_zc_op`/`fg_layout` are built EAGERLY at `build_originzc_core_hess_ctx` construction time
(campaign-lifetime, `fg_backend=:operator` only); `octx.fg_lookup_st` is cached here on first call
and reused across every subsequent inner solve at this `octx` (same pattern as `cctx.cmlookup_st`).
"""
function inner_loop_internal_originzc_operator(obj, θ_ext::AbstractVector, octx::OriginZCCoreHessCtx)
    octx.fg_zc_op === nothing &&
        error("inner_loop_internal_originzc_operator: octx.fg_zc_op is nothing -- octx was not built with fg_backend=:operator")
    # NOTE: `octx.n_eta` (= n_mean+n_pair, the RESTRICTION-COLUMN count) is a DIFFERENT quantity
    # from `n_eta(layout)` (= K_mean*D, the eta/nu PARAMETER count) -- octx.n_eta is correct for
    # the Hessian partition width (`archA_partitioned_hess_cb_builder`'s own `n = NCORE+n_eta_total`
    # column count) but WRONG here; θ_ext's trailing block has length `n_eta(layout)`, matching
    # `wrap_moments_with_originzc`'s own `n_eta_total = n_eta(layout)` local (a same-named but
    # different-valued quantity -- do not conflate the two, this was a real bug caught by this
    # branch's own D=4 correctness gate, nStatus=-400 with a corrupted νfull slice).
    n_eta_params = n_eta(octx.fg_layout)
    νfull = @view θ_ext[end-n_eta_params+1:end]

    obj.moments!(@view(obj.H[:, 1]), CS.select_G_from_H(obj, obj.H), θ_ext, obj.U, obj)
    obj.H[:, 2] .= 1.0
    obj.H_save = obj.H[1, 1] * (-1.0)^obj.find_smallest

    if octx.fg_lookup_st === nothing
        octx.fg_lookup_st = OriginZCOperatorState(obj, octx.NCORE - 1, octx.fg_zc_op::ZCRestrictionOperator, octx.fg_layout, octx.core_cf_ref)
    end
    st = octx.fg_lookup_st::OriginZCOperatorState
    reset_for_solve!(st, collect(νfull))

    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_originzc_operator(obj, st;
        hess_cb_builder = _ -> archA_partitioned_hess_cb_builder(octx))

    CS.INNER_SOLVE_COUNT[] += 1
    if nStatus ∉ [0, -100, -101, -103]
        CS.INNER_INFEAS_COUNT[] += 1
    end
    if nStatus ∈ [0, -100, -101, -103]
        obj.x .= x
        return obj.H_save, x, nStatus, n_fg, n_hess
    else
        obj.x .= NaN
        return -1e10, x, nStatus, n_fg, n_hess
    end
end

"""
    _originzc_fg_dispatch(ctx_cm, obj, θ_ext) -> (K, x, nStatus, n_fg, n_hess)

Single dispatch point `archOZ_base_state`/`archOZ_verified_state` call: `:dense_reference` (default)
goes through the UNCHANGED `inner_loop_internal_archgeneric`; `:operator` goes through the new
shared-economic-operator + ZC-restriction-operator path. `ctx_cm.octx.fg_backend` is the single
source of truth for which.
"""
function _originzc_fg_dispatch(ctx_cm, obj, θ_ext::AbstractVector)
    octx = ctx_cm.octx
    if octx.fg_backend === :operator
        return inner_loop_internal_originzc_operator(obj, θ_ext, octx)
    elseif octx.fg_backend === :dense_reference
        return inner_loop_internal_archgeneric(obj, θ_ext; hess_cb_builder = _ -> _originzc_hess_cb_builder(ctx_cm))
    else
        error("_originzc_fg_dispatch: fg_backend must be :dense_reference or :operator, got :$(octx.fg_backend)")
    end
end
