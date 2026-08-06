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
isdefined(Main, :CallbackHealthRecord) || include(joinpath(@__DIR__, "cm_callback_health.jl"))

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
    health = CallbackHealthRecord()   # 2026-08-06 fake-success guard (task section 3): reset fresh
        # every inner solve -- see cm_callback_health.jl for why this exists (the missing-Pow=
        # DimensionMismatch that used to report nStatus=0/n_fg_calls=0 as a "successful" solve).
    try
        kc = KNITRO.KN_new()
        KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
        KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
        x_initial = CS.inner_loop_initial_values(obj)
        KNITRO.KN_set_var_primal_init_values_all(kc, x_initial)

        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], callback_health_guard(_callbackEvalFG_inner_meanzc_operator!, health))
        KNITRO.KN_set_cb_user_params(kc, cb, st)
        KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

        n_hess = Ref(0)
        if KNITRO.KN_get_int_param(kc, "hessopt") == 1
            hess_cb_raw = hess_cb_builder(obj)
            hess_cb_adapted = callback_health_guard((kc2, cb2, evalRequest, evalResult, userParams) -> begin
                r = _adapt_hess_cb_for_meanzc_operator(hess_cb_raw)(kc2, cb2, evalRequest, evalResult, userParams)
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
        assert_no_fake_success!("inner_loop_KNITRO_meanzc_operator", health, nStatus, st.n_fg_calls, x_initial, x)
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
function inner_loop_internal_meanzc_operator(obj, θ_ext::AbstractVector, cctx::CMBinHessCtx; skip_fill::Bool = false)
    # Legacy-H cleanup (2026-07-28): mirrors inner_loop_internal_cmlookup_production's identical
    # dispatch (cm_lookup_production.jl) -- skip_fill=true only ever safe when cctx.moments_skip!
    # was actually built, falls back to the always-fill obj.moments! otherwise.
    cctx.meanzc_zc_op === nothing &&
        error("inner_loop_internal_meanzc_operator: cctx.meanzc_zc_op is nothing -- cctx was not built with inner_fg_backend=:operator")
    layout = cctx.meanzc_zc_layout
    n_eta_params = n_eta(layout)   # = K_mean (SharedByPowerLayout), NOT K_mean+K_pair
    νs = @view θ_ext[end-n_eta_params+1:end]

    # True no-H operator bundle (2026-07-28 continuation): same dispatch as flexible-CM's
    # inner_loop_internal_cmlookup_production. θ_econ is θ_ext with the trailing ν-block stripped --
    # cf_build/fill_K_directgp!/compressed_gravity_raw only ever read a PREFIX of their theta
    # argument (bounded by ctx.l_full/ctx.Aod_offset), so passing the full θ_ext would also be
    # numerically safe, but stripping matches wrap_moments_with_cm_meanzc's own established
    # convention (cm_meanzc_moments.jl: `θ_econ = @view θ_ext[1:end-K_mean]`) exactly.
    if obj isa OperatorPsiBundle
        θ_econ = @view θ_ext[1:end-n_eta_params]
        prime_operator!(obj, θ_econ, cctx.econ_ctx, cctx.core_cf_ref; restriction_state = cctx)
    else
        moments_fn = (skip_fill && cctx.moments_skip! !== nothing) ? cctx.moments_skip! : obj.moments!
        moments_fn(@view(obj.H[:, 1]), CS.select_G_from_H(obj, obj.H), θ_ext, obj.U, obj)
        obj.H[:, 2] .= 1.0
        obj.H_save = obj.H[1, 1] * (-1.0)^obj.find_smallest
    end

    if cctx.cmlookup_st === nothing
        bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
        # 2026-08-06 (paired-basis-preconditioning pilot continuation), BUG FIX: `Pow` was never
        # threaded through here at all -- defaulted to `nothing` even for a two-family (`cctx.
        # n_families==2`) context, so `st.Pow` stayed `nothing` while `st.ncm` was still the FULL
        # two-family width (2*nO*L). `dual_index!`'s own `fam2 = st.Pow !== nothing` therefore
        # incorrectly evaluated `false`, causing it to `reshape(λ_cm, nO, L)` on the UNSLICED FULL
        # two-family λ_cm (2*nO*L elements) instead of splitting it into cdf/pow halves first --
        # a real, silent `DimensionMismatch` on the very first live KNITRO callback invocation.
        # KNITRO's own C wrapper catches that exception at the FFI boundary (`_try_catch_handler`,
        # KNITRO.jl) and returns `KN_RC_CALLBACK_ERR` -- printed as a generic "exception in puts
        # callback" warning REGARDLESS of which callback actually threw (a KNITRO.jl package
        # quirk, not evidence it was really the output-text callback) -- KNITRO then reports
        # `nStatus=0` with the SOLVE NEVER HAVING RUN (n_fg_calls=0, x = the untouched all-zero
        # initial point) rather than propagating a Julia error, which is what made this so hard to
        # diagnose: no crash, no exception surfaced to the caller, just a silently-unsolved
        # "successful" result. Confirmed root cause live via a real D20/W=100,000/L=50 diagnostic
        # (st.n_fg_calls==0, lambdastar/zetastar == untouched zeros) plus a single-family control
        # at the identical scale that worked correctly (n_fg_calls=7, genuinely converged
        # nonzero lambdastar) -- isolating the difference to exactly this missing kwarg. This is
        # ALSO why the independent CMMeanZCOperatorState-vs-dense-reference gate
        # (test_cmmeanzc_operator_fg_twofamily_2026-08-06.jl, 28/28 PASS) never caught it -- that
        # test constructs CMMeanZCOperatorState directly and DOES pass `Pow=cctx.Pow` itself,
        # exercising the struct's own two-family logic correctly but never exercising this real
        # production call site's own (buggy) construction.
        cctx.cmlookup_st = CMMeanZCOperatorState(obj, cctx.ncore_core - 1, cctx.meanzc_zc_op::ZCRestrictionOperator, layout, cctx.core_cf_ref,
            cctx.ncm, cctx.L, cctx.origins, cctx.refIndex1, bins_u, cctx.R; nthreads_use = Threads.nthreads(),
            Pow = cctx.n_families == 2 ? cctx.Pow : nothing)
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
function _meanzc_fg_dispatch(cctx::CMBinHessCtx, obj, θ_ext::AbstractVector; skip_fill::Bool = false)
    if cctx.inner_fg_backend === :operator
        return inner_loop_internal_meanzc_operator(obj, θ_ext, cctx; skip_fill = skip_fill)
    elseif cctx.inner_fg_backend === :dense_reference
        record_generic_dense_fg!()
        # 2026-08-05 truncated-power task: unconditional Architecture-C dispatch, same as plain
        # flexible CM's archC_base_state -- `hessian_cm_structured!` self-selects the correct
        # internal path for a two-family, ZC-widened (CM+ZC) cctx here (forces its own dense-H
        # CScum2 H_EC fallback, since the winner-bin path's ZC-widened H_CZ cross was not given its
        # own pow extension in this pass -- see that function's own top-of-body comment). This
        # `:dense_reference` branch always has `obj.H` available, so that fallback is safe here.
        hess_builder = _obj -> archC_hess_cb_builder(cctx)
        return inner_loop_internal_archgeneric(obj, θ_ext; hess_cb_builder = hess_builder)
    else
        error("_meanzc_fg_dispatch: cctx.inner_fg_backend must be :dense_reference or :operator, got :$(cctx.inner_fg_backend)")
    end
end
