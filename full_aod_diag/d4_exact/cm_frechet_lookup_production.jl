# ================================================================================================
# Production wiring for CMFrechetLookupState (cm_frechet_lookup_kernels.jl) -- Phase 5.2
# remediation (2026-07-26). Mirrors cm_lookup_production.jl's structure exactly (same
# hess-callback-adapter trick, same cctx-cached-state pattern, same skip-cm-fill-ref reuse).
# ================================================================================================

"""
    inner_loop_KNITRO_cmfrechetlookup_production(obj, st::CMFrechetLookupState; hess_cb_builder) -> (nStatus, objSol, x, lambda_, n_fg, n_hess)

Common-Frechet analogue of `inner_loop_KNITRO_cmlookup_production`.
"""
function inner_loop_KNITRO_cmfrechetlookup_production(obj, st::CMFrechetLookupState; hess_cb_builder)
    CS.guard_enter_inner_solve!()
    try
        kc = KNITRO.KN_new()
        KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
        KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
        KNITRO.KN_set_var_primal_init_values_all(kc, CS.inner_loop_initial_values(obj))

        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], _callbackEvalFG_inner_cmfrechetlookup!)
        KNITRO.KN_set_cb_user_params(kc, cb, st)
        KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

        n_hess = Ref(0)
        if KNITRO.KN_get_int_param(kc, "hessopt") == 1
            hess_cb_raw = hess_cb_builder(obj)
            hess_cb_adapted = (kc2, cb2, evalRequest, evalResult, userParams) -> begin
                r = _adapt_hess_cb_for_lookup(hess_cb_raw)(kc2, cb2, evalRequest, evalResult, userParams)
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
KNITRO FG callback wrapper for CMFrechetLookupState -- EXACT mirror of
`_callbackEvalFG_inner_cmlookup!` (cm_lookup_live_knitro.jl), not a re-derivation: this codebase's
`KN_add_eval_callback(kc, true, Int32[], cb)` registration calls ONE combined callback for both
`f` and `g` every time (no separate EVALFC/EVALGA dispatch on `evalRequest.evalRequestCode` --
an earlier draft of this function branched on that and fell through to an error return on every
single call, the root cause of a real live bug this session hit: `nStatus` still reported 0/
"success" but every solved point was the untouched KNITRO initial point, `Delta_dual` exactly
`-0.0` in every case -- caught by a standalone unit-style forward/backward comparison against the
dense reference BEFORE suspecting the kernel math, per this project's own verify-before-blaming-
numerics discipline). Includes the SAME `lower_limit` infeasibility guard the real cm_lookup
callback applies, omitted from the earlier draft.
"""
function _callbackEvalFG_inner_cmfrechetlookup!(kc, cb, evalRequest, evalResult, userParams)
    st = userParams::CMFrechetLookupState
    x = evalRequest.x
    f = st(x, evalResult.objGrad)
    evalResult.obj[1] = f <= st.obj.lower_limit ? -KNITRO.KN_INFINITY : f
    return 0
end

"""
    inner_loop_internal_cmfrechetlookup_production(obj, θ, cctx::CMBinHessCtx; hess_cb_builder, nthreads_use=Threads.nthreads())

Common-Frechet analogue of `inner_loop_internal_cmlookup_production` -- same contract
`(K, x, nStatus, n_fg, n_hess)`, same `cctx.cmlookup_st` cache reuse (typed `Any`, shared with
plain-CM's own `CMLookupState` cache -- a `cctx` is only ever built for ONE family, so no
collision risk), same `cctx.skip_cm_fill_ref` wiring for archC_frechet_base_state to skip the
now-wasted dense CM-column fill exactly as archC_base_state does for plain CM (the level block's
own dense fill, `fill_frechet_level_columns_from_bins!`, is ALSO skippable under the same
condition -- see wrap_moments_with_cm_frechet_archB's own skip_cm_fill_ref-gated call below).
"""
function inner_loop_internal_cmfrechetlookup_production(obj, θ::AbstractVector, cctx::CMBinHessCtx,
        level_targets::Vector{Float64}; hess_cb_builder, nthreads_use::Int = Threads.nthreads())
    obj.moments!(@view(obj.H[:, 1]), CS.select_G_from_H(obj, obj.H), θ, obj.U, obj)
    obj.H[:, 2] .= 1.0
    obj.H_save = obj.H[1, 1] * (-1.0)^obj.find_smallest

    if cctx.cmlookup_st === nothing
        bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
        ncm_cm = cctx.ncm - cctx.L   # cctx.ncm = ncm_cm + ncm_level = (D-1)*L + L for common-Frechet
        cctx.cmlookup_st = CMFrechetLookupState(obj, cctx.NCORE, ncm_cm, cctx.L, cctx.L, cctx.D,
            cctx.origins, cctx.refIndex1, bins_u, cctx.R, level_targets;
            nthreads_use = nthreads_use)
    end
    st = cctx.cmlookup_st::CMFrechetLookupState
    st.n_fg_calls = 0

    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_cmfrechetlookup_production(obj, st; hess_cb_builder = hess_cb_builder)

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
