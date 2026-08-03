# diag/compressed-hessian-operator-audit-2026-07-25, Phase 6: KNITRO wiring
# for the winner-pair Hessian candidate, mirroring compressed_live.jl's
# _callbackEvalH_inner_compressed!/inner_loop_KNITRO_compressed EXACTLY except
# for which Hessian callback is registered -- everything else (FG callback,
# variable/bound setup, option file, complementarity constraints, guards) is
# byte-identical to the current production compressed path, so any timing
# difference measured against it is attributable to the Hessian step alone.
#
# BENCHMARK/CANDIDATE CODE ONLY -- not wired into any production driver.
include(joinpath(@__DIR__, "winner_pair_hessian.jl"))

"Per-inner-solve mutable bundle for the winner-pair Hessian callback: like CompressedCBState, plus a lazily-built WinnerPairHessCtx cache (theta-fixed, built once per inner solve, exactly analogous to CompressedCBState.dense_materialized)."
mutable struct WinnerPairCBState
    obj::Any
    cf::CompressedFactual
    wctx::Union{Nothing,WinnerPairHessCtx}
end
WinnerPairCBState(obj, cf::CompressedFactual) = WinnerPairCBState(obj, cf, nothing)

function _callbackEvalH_inner_winnerpair!(kc, cb, evalRequest, evalResult, userParams)
    st = userParams
    obj = st.obj
    @prof "inner_dual_hessian_callback_winnerpair" begin
        if st.wctx === nothing
            st.wctx = build_winner_pair_ctx(st.cf)
        end
        winner_pair_hessian!(evalResult.hess, obj, st.wctx)
    end
    _INNER_CALL_COUNTERS[].n_hess_calls += 1
    return 0
end

"Winner-pair-Hessian variant of _callbackEvalFG_inner_compressed! -- IDENTICAL body (reused verbatim, not reimplemented), just typed against WinnerPairCBState's field layout (obj/cf, same names) so it can share userParams with _callbackEvalH_inner_winnerpair!."
function _callbackEvalFG_inner_winnerpair!(kc, cb, evalRequest, evalResult, userParams)
    st = userParams
    obj = st.obj
    x = evalRequest.x
    ζ = x[1]
    λ = @view x[2:end]
    @prof "inner_dual_fg_callback_compressed" begin
        f, g_ζ, g_λ, q, _ = compressed_cc_value_grad(ζ, λ, st.cf; Psi! = obj.Psi!, dPsi! = obj.dPsi!)
        evalResult.obj[1] = f <= obj.lower_limit ? -KNITRO.KN_INFINITY : f
        evalResult.objGrad[1] = g_ζ
        @views evalResult.objGrad[2:end] .= g_λ
        obj.arg0 .= q
    end
    _INNER_CALL_COUNTERS[].n_fg_calls += 1
    return 0
end

"""
    inner_loop_KNITRO_compressed_winnerpair(obj, st::WinnerPairCBState) -> (nStatus, objSol, x, lambda_, n_fg_calls, n_hess_calls)

Byte-identical to `inner_loop_KNITRO_compressed` (compressed_live.jl) except
the Hessian callback registered is `_callbackEvalH_inner_winnerpair!` instead
of `_callbackEvalH_inner_compressed!` -- variable/bound setup, option file,
complementarity constraints, and inner-solve guards are all the SAME calls on
the SAME `obj`, so this isolates the Hessian-backend choice as the only
difference from the trusted production path.
"""
function inner_loop_KNITRO_compressed_winnerpair(obj, st::WinnerPairCBState)
    _INNER_CALL_COUNTERS[] = InnerCallCounters(0, 0)
    CS.guard_enter_inner_solve!()
    try
        kc = KNITRO.KN_new()
        KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
        KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
        KNITRO.KN_set_var_primal_init_values_all(kc, CS.inner_loop_initial_values(obj))

        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], _callbackEvalFG_inner_winnerpair!)
        KNITRO.KN_set_cb_user_params(kc, cb, st)
        KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

        if KNITRO.KN_get_int_param(kc, "hessopt") == 1
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, _callbackEvalH_inner_winnerpair!)
        end
        if obj.complement_index != [0 0]
            CS.inner_loop_complementarity_constraints(kc, obj)
        end

        @prof "inner_knitro_dual_solve_winnerpair" begin
            KNITRO.KN_solve(kc)
        end
        nSTatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
        CS.INNER_ITERS_TOTAL[] += CS._kn_num_iters(kc)
        KNITRO.KN_free(kc)

        return nSTatus, objSol, x, lambda_, _INNER_CALL_COUNTERS[].n_fg_calls, _INNER_CALL_COUNTERS[].n_hess_calls
    finally
        CS.guard_exit_inner_solve!()
    end
end
