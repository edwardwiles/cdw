# ============================================================================
# Continuation 9, Phase 4: end-to-end wiring of the destination-major _v2
# kernels (compressed_cc_kernels_v2.jl) into an ACTUAL KNITRO inner solve,
# structurally mirroring compressed_live.jl's `_callbackEvalFG_inner_
# compressed!`/`inner_loop_KNITRO_compressed`/`inner_loop_internal_compressed`
# exactly (same variable/bound/init-value setup, same lazy dense-Hessian
# materialization decision from Continuation 8 -- NOT re-litigated here) --
# the ONLY difference is calling `compressed_cc_value_grad_v2` instead of
# `compressed_cc_value_grad` in the FG callback.
#
# WHY THIS EXISTS: the isolated per-call kernel benchmark
# (c9_phase4_kernels_v2_d20_bench.jl) measured compressed_cc_value_grad_v2
# ~1.2-1.3x faster than the original in a single, standalone call. This
# investigation's own "verify before causal claims" discipline (and Phase
# 3C's own finding that an isolated FLOP-count argument did NOT survive
# contact with an actual KNITRO solve) means that per-call win is NOT
# automatically assumed to translate to a real end-to-end inner-solve
# speedup -- it is measured directly here, via an actual multi-iteration
# KNITRO solve (many FG calls in sequence, real KNITRO overhead in between),
# not extrapolated from the isolated number.
#
# ADDITIVE ONLY: new function names throughout (`_v2` suffix on everything
# that touches the FG callback), zero changes to compressed_live.jl/
# oracle_fast.jl. Not wired into evaluate_fullA_fast's `moment_representation`
# dispatch (that would require also handling the Hessian-callback tail's
# lazy materialization + lazy post-processing paths compressed_live.jl
# already owns) -- this file's scope is strictly the FG-callback-driven
# inner dual solve itself, for direct comparison against
# inner_loop_internal_compressed.
# ============================================================================

"Same as compressed_live.jl's `_callbackEvalFG_inner_compressed!`, calling `compressed_cc_value_grad_v2` instead of the original."
function _callbackEvalFG_inner_compressed_v2!(kc, cb, evalRequest, evalResult, userParams)
    st = userParams
    obj = st.obj
    x = evalRequest.x
    ζ = x[1]
    λ = @view x[2:end]
    @prof "inner_dual_fg_callback_compressed_v2" begin
        f, g_ζ, g_λ, q, _ = compressed_cc_value_grad_v2(ζ, λ, st.cf; Psi! = obj.Psi!, dPsi! = obj.dPsi!)
        evalResult.obj[1] = f <= obj.lower_limit ? -KNITRO.KN_INFINITY : f
        evalResult.objGrad[1] = g_ζ
        @views evalResult.objGrad[2:end] .= g_λ
        obj.arg0 .= q
    end
    _INNER_CALL_COUNTERS[].n_fg_calls += 1
    return 0
end

"Same as compressed_live.jl's `_callbackEvalH_inner_compressed!` (lazy dense materialization for the Hessian, UNCHANGED -- this file's scope is the FG callback only, per this file's own header)."
function _callbackEvalH_inner_compressed_v2!(kc, cb, evalRequest, evalResult, userParams)
    st = userParams
    obj = st.obj
    @prof "inner_dual_hessian_callback_compressed_v2" begin
        if !st.dense_materialized
            ncolI = st.cf.oci - 1
            materialize_dense_factual!(@view(obj.H[:, 3:2+ncolI]), st.cf)
            fill_gravity_column!(obj, st.grav_raw)
            st.dense_materialized = true
        end
        CS.hessian!(evalResult.hess, obj)
    end
    _INNER_CALL_COUNTERS[].n_hess_calls += 1
    return 0
end

"Same as compressed_live.jl's `inner_loop_KNITRO_compressed`, registering the `_v2` FG callback."
function inner_loop_KNITRO_compressed_v2(obj, st::CompressedCBState)
    _INNER_CALL_COUNTERS[] = InnerCallCounters(0, 0)
    kc = KNITRO.KN_new()
    KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
    KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
    KNITRO.KN_set_var_primal_init_values_all(kc, CS.inner_loop_initial_values(obj))

    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], _callbackEvalFG_inner_compressed_v2!)
    KNITRO.KN_set_cb_user_params(kc, cb, st)
    KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

    if KNITRO.KN_get_int_param(kc, "hessopt") == 1
        KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, _callbackEvalH_inner_compressed_v2!)
    end
    if obj.complement_index != [0 0]
        CS.inner_loop_complementarity_constraints(kc, obj)
    end

    @prof "inner_knitro_dual_solve_compressed_v2" begin
        KNITRO.KN_solve(kc)
    end
    nSTatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
    CS.INNER_ITERS_TOTAL[] += CS._kn_num_iters(kc)
    KNITRO.KN_free(kc)

    return nSTatus, objSol, x, lambda_, _INNER_CALL_COUNTERS[].n_fg_calls, _INNER_CALL_COUNTERS[].n_hess_calls
end

"Same as compressed_live.jl's `inner_loop_internal_compressed`, dispatching to the `_v2` solver loop."
function inner_loop_internal_compressed_v2(obj, θ_full, ctx)
    # EXPLICIT_REFERENCE / benchmark-only (per this file's own module docstring: "Not wired into
    # evaluate_fullA_fast's moment_representation dispatch") -- not a PRODUCTION_HOT_PATH, but
    # updated to the canonical `build_economic_moment_state!` anyway (2026-07-27 task) for
    # consistency with compressed_live.jl's now-fixed call site; zero behavior change for any
    # caller whose ctx has no cf_workspace attached (unchanged allocating fallback).
    cf = @prof "inner_moment_build_compressed_v2" build_economic_moment_state!(θ_full, ctx; check_ties = true)

    W = size(obj.U, 1)
    SW = ctx.γ.SamplingWeights[1:W]
    obj.H[:, 1] .= θ_full[3 + ctx.D] .* SW
    obj.H[:, 2] .= 1.0
    obj.H_save = obj.H[1, 1] * (-1.0)^obj.find_smallest

    grav_raw = compressed_gravity_raw(θ_full, ctx)

    st = CompressedCBState(obj, cf, grav_raw, false)
    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_compressed_v2(obj, st)

    CS.INNER_SOLVE_COUNT[] += 1
    if nStatus ∉ [0, -100, -101, -103]
        CS.INNER_INFEAS_COUNT[] += 1
    end

    if nStatus ∈ [0, -100, -101, -103]
        obj.x .= x
        return obj.H_save, x, nStatus, n_fg, n_hess, st
    else
        obj.x .= NaN
        return -1e10, x, nStatus, n_fg, n_hess, st
    end
end
