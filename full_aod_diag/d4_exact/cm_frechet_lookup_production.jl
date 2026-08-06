# ================================================================================================
# Production wiring for CMFrechetLookupState (cm_frechet_lookup_kernels.jl) -- Phase 5.2
# remediation (2026-07-26). Mirrors cm_lookup_production.jl's structure exactly (same
# hess-callback-adapter trick, same cctx-cached-state pattern, same skip-cm-fill-ref reuse).
# ================================================================================================

isdefined(Main, :CallbackHealthRecord) || include(joinpath(@__DIR__, "cm_callback_health.jl"))

"""
    inner_loop_KNITRO_cmfrechetlookup_production(obj, st::CMFrechetLookupState; hess_cb_builder) -> (nStatus, objSol, x, lambda_, n_fg, n_hess)

Common-Frechet analogue of `inner_loop_KNITRO_cmlookup_production`.
"""
function inner_loop_KNITRO_cmfrechetlookup_production(obj, st::CMFrechetLookupState; hess_cb_builder)
    CS.guard_enter_inner_solve!()
    health = CallbackHealthRecord()   # 2026-08-06 fake-success guard, see cm_callback_health.jl
    try
        kc = KNITRO.KN_new()
        KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
        KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
        x_initial = CS.inner_loop_initial_values(obj)
        KNITRO.KN_set_var_primal_init_values_all(kc, x_initial)

        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], callback_health_guard(_callbackEvalFG_inner_cmfrechetlookup!, health))
        KNITRO.KN_set_cb_user_params(kc, cb, st)
        KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

        n_hess = Ref(0)
        if KNITRO.KN_get_int_param(kc, "hessopt") == 1
            hess_cb_raw = hess_cb_builder(obj)
            hess_cb_adapted = callback_health_guard((kc2, cb2, evalRequest, evalResult, userParams) -> begin
                r = _adapt_hess_cb_for_lookup(hess_cb_raw)(kc2, cb2, evalRequest, evalResult, userParams)
                n_hess[] += 1
                return r
            end, health)
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, hess_cb_adapted)
        end
        if obj.complement_index != [0 0]
            CS.inner_loop_complementarity_constraints(kc, obj)
        end

        KNITRO.KN_solve(kc)
        nStatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
        assert_no_fake_success!("inner_loop_KNITRO_cmfrechetlookup_production", health, nStatus, st.n_fg_calls, x_initial, x)
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
collision risk).

History: an earlier draft (Phase 5.2, 2026-07-26) gave this priming call a skip variant, found
unsafe (nStatus=-400, commit 5fd6347) and removed. RE-INVESTIGATED 2026-07-27/28
(docs/GOAL10_SKIP_CM_FILL_REF_REMOVAL_2026-07-27.md) on the hypothesis that the winner-bin
H_E,level Hessian path (`winner_pair_cross_hessian_colsum!`/`_esum!`, cm_frechet_hessian.jl, a git
DESCENDANT of the original bugfix by ~7.5h) might have made the skip safe again. D=4 multi-point
testing supported the hypothesis; a real D=20/W=80,000 re-test then DISPROVED it -- both tested
non-calibration points reproduced the exact nStatus=-400 failure with the skip enabled. The `skip_fill`
parameter below is kept (mirrors the other families' shared call signature, and `cctx.moments_skip!`
still exists as a built-but-unused closure) but common-Fréchet's own two call sites
(`archC_frechet_base_state`/`archC_frechet_verified_state`, cm_frechet_cplus.jl) now always pass
`skip_fill=false` -- see those functions' own HISTORY comments for the full chronology and real
D=20 evidence. Do not re-enable without root-causing the actual dependency first.
"""
function inner_loop_internal_cmfrechetlookup_production(obj, θ::AbstractVector, cctx::CMBinHessCtx,
        level_targets::Vector{Float64}; hess_cb_builder, nthreads_use::Int = Threads.nthreads(),
        skip_fill::Bool = false)
    # True no-H operator bundle (2026-07-28 continuation): same dispatch as flexible-CM's
    # inner_loop_internal_cmlookup_production -- see that function's own comment for the rationale.
    if obj isa OperatorPsiBundle
        prime_operator!(obj, θ, cctx.econ_ctx, cctx.core_cf_ref; restriction_state = cctx)
    else
        moments_fn = (skip_fill && cctx.moments_skip! !== nothing) ? cctx.moments_skip! : obj.moments!
        moments_fn(@view(obj.H[:, 1]), CS.select_G_from_H(obj, obj.H), θ, obj.U, obj)
        obj.H[:, 2] .= 1.0
        obj.H_save = obj.H[1, 1] * (-1.0)^obj.find_smallest
    end

    if cctx.cmlookup_st === nothing
        bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
        # 2026-08-06 (levelpow kernel task): `cctx.n_families==2` (set generically by
        # build_cm_bin_ctx from aug.n_families, cm_hessian_architectures.jl -- the SAME shared
        # builder flexible-CM/CM+ZC already use) means `cctx.ncm` is TWO-FAMILY-widened
        # (2*(D-1)*L + 2*L for common-Fréchet: CM_cdf+CM_pow+level+levelpow), and `level_targets`
        # (the caller's own aug.level_targets) is the already-concatenated [level_cdf(L);
        # level_pow(L)]. `cctx.Pow` is likewise already the real z^(sigma-1) matrix, built
        # generically by build_cm_bin_ctx -- no separate computation needed here.
        fam2 = cctx.n_families == 2
        L = cctx.L
        ncm_level_total = fam2 ? 2L : L
        ncm_cm = cctx.ncm - ncm_level_total
        level_targets_cdf = fam2 ? level_targets[1:L] : level_targets
        levelpow_targets = fam2 ? level_targets[L+1:2L] : nothing
        cctx.cmlookup_st = CMFrechetLookupState(obj, cctx.NCORE, ncm_cm, ncm_level_total, cctx.L, cctx.D,
            cctx.origins, cctx.refIndex1, bins_u, cctx.R, level_targets_cdf;
            nthreads_use = nthreads_use, core_cf_ref = cctx.core_cf_ref,
            Pow = fam2 ? cctx.Pow : nothing, levelpow_targets = levelpow_targets)
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
