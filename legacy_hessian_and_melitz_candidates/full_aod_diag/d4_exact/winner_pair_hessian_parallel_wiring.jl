# diag/compressed-hessian-operator-audit-2026-07-25 continuation: KNITRO
# wiring for the pair-ownership parallel winner-pair Hessian, mirroring
# winner_pair_hessian_wiring.jl (serial candidate)/compressed_live.jl
# (production) EXACTLY except for which Hessian callback is registered.
#
# BENCHMARK/CANDIDATE CODE ONLY -- not wired into any production driver.
include(joinpath(@__DIR__, "winner_pair_hessian_parallel.jl"))

"Per-inner-solve mutable bundle for the parallel winner-pair Hessian callback."
mutable struct WinnerPairParallelCBState
    obj::Any
    cf::CompressedFactual
    workspace::WinnerPairParallelWorkspace
    workers::Int
    storage::Symbol
end

function _callbackEvalH_inner_winnerpair_parallel!(kc, cb, evalRequest, evalResult, userParams)
    st = userParams
    obj = st.obj
    @prof "inner_dual_hessian_callback_winnerpair_parallel" begin
        hessian_core_winner_pair!(evalResult.hess, obj.arg2, obj, st.workspace; workers = st.workers, storage = st.storage)
    end
    _INNER_CALL_COUNTERS[].n_hess_calls += 1
    return 0
end

function _callbackEvalFG_inner_winnerpair_parallel!(kc, cb, evalRequest, evalResult, userParams)
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
    inner_loop_KNITRO_compressed_winnerpair_parallel(obj, st::WinnerPairParallelCBState) -> (nStatus, objSol, x, lambda_, n_fg_calls, n_hess_calls)
"""
function inner_loop_KNITRO_compressed_winnerpair_parallel(obj, st::WinnerPairParallelCBState)
    _INNER_CALL_COUNTERS[] = InnerCallCounters(0, 0)
    CS.guard_enter_inner_solve!()
    try
        kc = KNITRO.KN_new()
        KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
        KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
        KNITRO.KN_set_var_primal_init_values_all(kc, CS.inner_loop_initial_values(obj))

        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], _callbackEvalFG_inner_winnerpair_parallel!)
        KNITRO.KN_set_cb_user_params(kc, cb, st)
        KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

        if KNITRO.KN_get_int_param(kc, "hessopt") == 1
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, _callbackEvalH_inner_winnerpair_parallel!)
        end
        if obj.complement_index != [0 0]
            CS.inner_loop_complementarity_constraints(kc, obj)
        end

        @prof "inner_knitro_dual_solve_winnerpair_parallel" begin
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
