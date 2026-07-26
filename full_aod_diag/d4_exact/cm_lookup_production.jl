# ================================================================================================
# Production wiring for the validated lookup-based FG evaluator (cm_lookup_kernels.jl,
# CMLookupState) -- remediation task Phase B1 (production-audit continuation, 2026-07-26).
#
# BACKGROUND: cm_lookup_kernels.jl's CMLookupState/`_callbackEvalFG_inner_cmlookup!` was
# validated (c12i_validate_lookup_fg.jl) to reproduce the dense `obj(x,g)` callable's (f,g)
# machine-precision-exactly, but was never wired into any production entry point -- every
# restricted family's inner KNITRO dual solve still registers the generic
# `_callbackEvalFG_inner_profiled!` (oracle_fast.jl), which performs a dense BLAS.gemv! against
# the fully materialized CM columns of `obj.H` on every single inner-solve iterate (baseline
# audit finding B1, docs/PRODUCTION_5X7_AUDIT_BASELINE_AND_REMEDIATION_SIZING_2026-07-26.md).
#
# SCOPE (Phase B1, this file): plain FLEXIBLE_CM only. CMLookupState's own layout
# `x = [zeta; lambda_core; lambda_cm]` has no room for common-Frechet's level-anchor block or
# CM+ZC's mean/pair block -- extending it for those two families is genuinely new numerical-
# kernel development, not wiring, and is tracked as a separate follow-on item (see
# docs/RESTRICTED_LOOKUP_FG_PRODUCTION_PORT_2026-07-26.md). This file's production dispatch is
# used ONLY when marginal_restriction=:common_flexible and cm_extension=:cm_only (plain CM, no
# widened economic block) -- callers must not invoke it under common_frechet or meanzc configs.
#
# DESIGN: cm_lookup_kernels.jl's own inner_loop_KNITRO_cmlookup (cm_lookup_live_knitro.jl)
# bundles a DENSE Hessian callback for isolated microbenchmark purposes. Production must instead
# combine the lookup FG callback with the EXISTING, already-optimized Architecture-C Hessian
# callback (archC_hess_cb_builder, cm_hessian_architectures.jl) -- per the task brief, "do not
# modify family Hessian mathematics merely to accommodate the FG callback". Because KNITRO
# attaches exactly one `userParams` per (FG, Hessian) callback pair, and the existing Hessian
# closures expect `userParams` to BE the dense `PsiObjectiveBundleImplicit` (`o = userParams`),
# while the lookup FG callback needs `userParams` to BE a `CMLookupState` (`st`), the two are
# reconciled via a THIN adapter (`_adapt_hess_cb_for_lookup`) that unwraps `st.obj` before
# forwarding to the unmodified existing Hessian closure -- zero changes to
# hessian_cm_structured!/`_v2!`/archC_hess_cb_builder themselves.
# ================================================================================================

"""
    _adapt_hess_cb_for_lookup(hess_cb) -> Function

Wraps an EXISTING KNITRO Hessian callback closure (as returned by `archC_hess_cb_builder(cctx)`,
which expects `userParams` to be the dense `obj`) so it can be registered on the SAME `cb` as a
lookup FG callback (whose own `userParams` is a `CMLookupState`, `st`). `st.obj` is the same dense
`PsiObjectiveBundleImplicit` object the unmodified Hessian closure expects -- this is a pure
call-site adapter, no Hessian math is touched.
"""
_adapt_hess_cb_for_lookup(hess_cb) =
    (kc, cb, evalRequest, evalResult, userParams) -> hess_cb(kc, cb, evalRequest, evalResult, userParams.obj)

"""
    inner_loop_KNITRO_cmlookup_production(obj, st::CMLookupState; hess_cb_builder) -> (nStatus, objSol, x, lambda_, n_fg, n_hess)

Production inner KNITRO dual solve using the lookup FG callback (`_callbackEvalFG_inner_cmlookup!`,
cm_lookup_live_knitro.jl) for the hot per-iterate forward/backward, and the CALLER-SUPPLIED
production Hessian callback builder (e.g. `archC_hess_cb_builder(cctx)`, unmodified) for the
Hessian -- mirrors `inner_loop_KNITRO_archgeneric`'s structure/counters exactly so the two are a
fair, matched A/B.
"""
function inner_loop_KNITRO_cmlookup_production(obj, st::CMLookupState; hess_cb_builder)
    CS.guard_enter_inner_solve!()
    try
        kc = KNITRO.KN_new()
        KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
        KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
        KNITRO.KN_set_var_primal_init_values_all(kc, CS.inner_loop_initial_values(obj))

        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], _callbackEvalFG_inner_cmlookup!)
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
    inner_loop_internal_cmlookup_production(obj, θ, ctx, cctx::CMBinHessCtx; hess_cb_builder, method=:interval, nthreads_use=Threads.nthreads())

Mirrors `inner_loop_internal_archgeneric`'s contract exactly (same return shape
`(K, x, nStatus, n_fg, n_hess)`) so `archC_base_state`/`archC_verified_state` can dispatch to
either with no other code change. Builds the dense `obj.H` once (same `moments!` call the
:dense_reference path uses -- CMLookupState's own core-column BLAS slice and the Hessian
callback both need it).

Phase 5.5 remediation (2026-07-26): the `CMLookupState` itself is now built ONCE per `cctx`
(cached on `cctx.cmlookup_st`, the same "built once per campaign, reused every inner solve"
pattern `cctx`'s own `core_ws`/`tls` fields already use) and REUSED across every subsequent
inner solve at this context, rather than rebuilt fresh here every call -- bin
indices/origins/refIndex1/R are immutable for the life of `cctx` (task §5.5: "do not rebuild
CMLookupState per solve if dimensions/context are fixed"). `st.n_fg_calls` is reset to 0 at the
top of each solve so the returned `n_fg` still means "FG calls THIS inner solve", matching the
pre-caching contract.
"""
function inner_loop_internal_cmlookup_production(obj, θ::AbstractVector, cctx::CMBinHessCtx;
        hess_cb_builder, method::Symbol = :suffix, nthreads_use::Int = Threads.nthreads())
    obj.moments!(@view(obj.H[:, 1]), CS.select_G_from_H(obj, obj.H), θ, obj.U, obj)
    obj.H[:, 2] .= 1.0
    obj.H_save = obj.H[1, 1] * (-1.0)^obj.find_smallest

    # method defaults to :suffix (cumulative-basis lookup), NOT :interval, because
    # build_cm_production_context's own `aug = build_cm_augmented_obj(...)` (cm_production_bundle.jl
    # header: "cumulative basis (build_cm_augmented_obj)") stores CM coefficients in the CUMULATIVE
    # basis -- comparing :interval lookup against a cumulative-basis obj.H is exactly the "category
    # error" c12i_validate_lookup_fg.jl's own comments warn about (comparing interval lookup against
    # the cumulative dense reference or vice versa). Confirmed live: passing method=:interval here
    # (an earlier version of this function) caused every real inner solve to report KNITRO nStatus=
    # -400 (infeasible) at the calibration point -- not a KNITRO/production bug, a basis mismatch in
    # this file's own first draft, caught by test_phaseB1_cmlookup_production_correctness.jl.
    if cctx.cmlookup_st === nothing
        bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
        cctx.cmlookup_st = CMLookupState(obj, cctx.NCORE, cctx.ncm, cctx.L, cctx.origins, cctx.refIndex1, bins_u, cctx.R;
                                          method = method, nthreads_use = nthreads_use)
    end
    st = cctx.cmlookup_st::CMLookupState
    st.method == method || error("inner_loop_internal_cmlookup_production: cached CMLookupState was built with method=$(st.method), called with method=$method -- a live method change on a reused cctx is not supported (rebuild cctx instead)")
    st.n_fg_calls = 0

    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_cmlookup_production(obj, st; hess_cb_builder = hess_cb_builder)

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
