# ================================================================================================
# Production wiring for CMMeanZCOperatorState (cm_meanzc_lookup_kernels.jl) --
# port/shared-inner-fg-operator-and-verification-2026-07-26, Job 1 §3 (CM+ZC, `G=[E|C|Z]`). Mirrors
# cm_originzc_lookup_production.jl / cm_lookup_production.jl's structure exactly.
#
# SCOPE: opt-in only (`inner_fg_backend=:operator` on `CMBinHessCtx`, default stays
# `:dense_reference`) until this branch's own D=4/D=20 correctness + performance gates pass.
# ================================================================================================

isdefined(Main, :CMMeanZCOperatorState) || include(joinpath(@__DIR__, "cm_meanzc_lookup_kernels.jl"))
isdefined(Main, :NO_DENSE_G_COUNTERS) || include(joinpath(@__DIR__, "no_dense_g_counters.jl"))

"Same call-site Hessian adapter cm_lookup_production.jl/cm_originzc_lookup_production.jl use -- unwraps `userParams::CMMeanZCOperatorState` to `st.obj`."
_adapt_hess_cb_for_meanzc_operator(hess_cb) =
    (kc, cb, evalRequest, evalResult, userParams) -> hess_cb(kc, cb, evalRequest, evalResult, userParams.obj)

"KNITRO FG callback: ONE combined callback for f and g every invocation (this codebase's established `KN_add_eval_callback(kc, true, ...)` protocol -- mirrors every other lookup/operator callback in this codebase)."
function _callbackEvalFG_inner_meanzc_operator!(kc, cb, evalRequest, evalResult, userParams)
    st = userParams
    x = evalRequest.x
    f = st(x, evalResult.objGrad)
    evalResult.obj[1] = f <= st.obj.lower_limit ? -KNITRO.KN_INFINITY : f
    return 0
end

"Mirrors `inner_loop_KNITRO_cmlookup_production`/`inner_loop_KNITRO_originzc_operator` exactly."
function inner_loop_KNITRO_meanzc_operator(obj, st::CMMeanZCOperatorState; hess_cb_builder)
    CS.guard_enter_inner_solve!()
    try
        kc = KNITRO.KN_new()
        KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
        KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
        KNITRO.KN_set_var_primal_init_values_all(kc, CS.inner_loop_initial_values(obj))

        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], _callbackEvalFG_inner_meanzc_operator!)
        KNITRO.KN_set_cb_user_params(kc, cb, st)
        KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

        n_hess = Ref(0)
        if KNITRO.KN_get_int_param(kc, "hessopt") == 1
            hess_cb_raw = hess_cb_builder(obj)
            hess_cb_adapted = (kc2, cb2, evalRequest, evalResult, userParams) -> begin
                r = _adapt_hess_cb_for_meanzc_operator(hess_cb_raw)(kc2, cb2, evalRequest, evalResult, userParams)
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

"""
    inner_loop_internal_meanzc_operator(obj, θ_ext, cctx::CMBinHessCtx) -> (K, x, nStatus, n_fg, n_hess)

Mirrors `inner_loop_internal_cmlookup_production`/`inner_loop_internal_originzc_operator`'s
contract exactly. `θ_ext`'s trailing block has length `cctx.meanzc_zc_layout.K_mean` (=
`n_eta(cctx.meanzc_zc_layout)` for `SharedByPowerLayout` -- K_mean itself, NOT K_mean+K_pair; see
`cm_meanzc_moments.jl`'s own `θ_econ = @view θ_ext[1:end-K_mean]`).
"""
function inner_loop_internal_meanzc_operator(obj, θ_ext::AbstractVector, cctx::CMBinHessCtx)
    cctx.meanzc_zc_op === nothing &&
        error("inner_loop_internal_meanzc_operator: cctx.meanzc_zc_op is nothing -- cctx was not built with inner_fg_backend=:operator")
    layout = cctx.meanzc_zc_layout
    n_eta_params = n_eta(layout)   # = K_mean (SharedByPowerLayout), NOT K_mean+K_pair
    νs = @view θ_ext[end-n_eta_params+1:end]

    obj.moments!(@view(obj.H[:, 1]), CS.select_G_from_H(obj, obj.H), θ_ext, obj.U, obj)
    obj.H[:, 2] .= 1.0
    obj.H_save = obj.H[1, 1] * (-1.0)^obj.find_smallest

    if cctx.cmlookup_st === nothing
        bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
        cctx.cmlookup_st = CMMeanZCOperatorState(obj, cctx.ncore_core - 1, cctx.meanzc_zc_op::ZCRestrictionOperator, layout, cctx.core_cf_ref,
            cctx.ncm, cctx.L, cctx.origins, cctx.refIndex1, bins_u, cctx.R; nthreads_use = Threads.nthreads())
    end
    st = cctx.cmlookup_st::CMMeanZCOperatorState
    reset_for_solve!(st, collect(νs))

    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_meanzc_operator(obj, st;
        hess_cb_builder = _ -> archC_hess_cb_builder(cctx))

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
    _meanzc_fg_dispatch(cctx, obj, θ_ext) -> (K, x, nStatus, n_fg, n_hess)

Single dispatch point `archC_meanzc_base_state`/`archC_meanzc_verified_state` call:
`:dense_reference` (default) -> UNCHANGED `inner_loop_internal_archgeneric`; `:operator` -> the new
shared-economic-operator + CM-lookup + ZC-restriction-operator path.
"""
function _meanzc_fg_dispatch(cctx::CMBinHessCtx, obj, θ_ext::AbstractVector)
    if cctx.inner_fg_backend === :operator
        return inner_loop_internal_meanzc_operator(obj, θ_ext, cctx)
    elseif cctx.inner_fg_backend === :dense_reference
        record_generic_dense_fg!()
        return inner_loop_internal_archgeneric(obj, θ_ext; hess_cb_builder = _obj -> archC_hess_cb_builder(cctx))
    else
        error("_meanzc_fg_dispatch: cctx.inner_fg_backend must be :dense_reference or :operator, got :$(cctx.inner_fg_backend)")
    end
end
