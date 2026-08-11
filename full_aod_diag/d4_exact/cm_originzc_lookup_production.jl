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
isdefined(Main, :NO_DENSE_G_COUNTERS) || include(joinpath(@__DIR__, "no_dense_g_counters.jl"))
isdefined(Main, :CallbackHealthRecord) || include(joinpath(@__DIR__, "cm_callback_health.jl"))

"_adapt_hess_cb_for_originzc_operator(hess_cb): same call-site adapter cm_lookup_production.jl uses -- unwraps `userParams::OriginZCOperatorState` to `st.obj` before forwarding to the UNMODIFIED existing Hessian closure (`archA_partitioned_hess_cb_builder(octx)`, which itself expects `userParams` to be the dense `obj`)."
_adapt_hess_cb_for_originzc_operator(hess_cb) =
    (kc, cb, evalRequest, evalResult, userParams) -> hess_cb(kc, cb, evalRequest, evalResult, userParams.obj)

"""
    inner_loop_KNITRO_originzc_operator(obj, st::OriginZCOperatorState; hess_cb_builder) -> (nStatus, objSol, x, lambda_, n_fg, n_hess)

Mirrors `inner_loop_KNITRO_cmlookup_production` exactly.
"""
function inner_loop_KNITRO_originzc_operator(obj, st::OriginZCOperatorState; hess_cb_builder)
    CS.guard_enter_inner_solve!()
    health = CallbackHealthRecord()   # 2026-08-06 fake-success guard, see cm_callback_health.jl
    try
        kc = KNITRO.KN_new()
        KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
        KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
        x_initial = CS.inner_loop_initial_values(obj)
        KNITRO.KN_set_var_primal_init_values_all(kc, x_initial)

        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], callback_health_guard(_callbackEvalFG_inner_originzc_operator!, health))
        KNITRO.KN_set_cb_user_params(kc, cb, st)
        KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

        n_hess = Ref(0)
        if KNITRO.KN_get_int_param(kc, "hessopt") == 1
            hess_cb_raw = hess_cb_builder(obj)
            hess_cb_adapted = callback_health_guard((kc2, cb2, evalRequest, evalResult, userParams) -> begin
                r = _adapt_hess_cb_for_originzc_operator(hess_cb_raw)(kc2, cb2, evalRequest, evalResult, userParams)
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
        assert_no_fake_success!("inner_loop_KNITRO_originzc_operator", health, nStatus, st.n_fg_calls, x_initial, x)
        CS.INNER_ITERS_TOTAL[] += CS._kn_num_iters(kc)
        n_fg = st.n_fg_calls
        KNITRO.KN_free(kc)

        return nStatus, objSol, x, lambda_, n_fg, n_hess[]
    finally
        CS.guard_exit_inner_solve!()
    end
end

"""
    INNER_SOLVE_TRACE

Opt-in per-inner-solve recorder (default `nothing` = one `=== nothing` check, zero cost) for the
2026-08-11 abort-cost study. Set to a `Vector{Any}`; each inner solve appends
`(status, n_fg, n_hess, wall, f_first, first_below)` where `f_first` is the objective at the FIRST
FG evaluation (i.e. at the warm start KNITRO was handed) and `first_below` is the 1-based index of
the first FG call at or under `obj.lower_limit` (0 = never).

The question it answers: when a solve is doomed -- `f <= lower_limit` already at the initial point --
how much work does KNITRO still do before returning? Aggregate counters cannot answer this because
they mix doomed and healthy solves.
"""
const INNER_SOLVE_TRACE = Ref{Any}(nothing)
const _IST_nfg = Ref{Int}(0)
const _IST_f1 = Ref{Float64}(NaN)
const _IST_first_below = Ref{Int}(0)

"KNITRO FG callback protocol: this codebase's `KN_add_eval_callback(kc, true, ...)` registration calls ONE combined callback for both f and g every invocation -- mirrors `_callbackEvalFG_inner_cmlookup!` exactly (see cm_lookup_production.jl / cm_frechet_lookup_production.jl for the same pattern and the KNITRO-wiring bug this must not repeat)."
function _callbackEvalFG_inner_originzc_operator!(kc, cb, evalRequest, evalResult, userParams)
    # 2026-08-11 profiling: this path had NO timing label at all, so a whole-run profile could only
    # report the FG share as an unmeasured residual (outer wall minus Hessian-callback total). Uses
    # @cmhess_prof, NOT @prof, so it is gated by CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[] (default
    # false = a single Ref check, zero overhead) rather than PROF_ENABLED[] (default true) -- this
    # fires once per inner KNITRO iteration in production and must cost nothing when not profiling.
    @cmhess_prof "originZC_FG_callback" begin
        st = userParams
        x = evalRequest.x
        f = st(x, evalResult.objGrad)
        if INNER_SOLVE_TRACE[] !== nothing
            _IST_nfg[] += 1
            _IST_nfg[] == 1 && (_IST_f1[] = f)
            (_IST_first_below[] == 0 && f <= st.obj.lower_limit) && (_IST_first_below[] = _IST_nfg[])
        end
        evalResult.obj[1] = f <= st.obj.lower_limit ? -KNITRO.KN_INFINITY : f
    end
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
function inner_loop_internal_originzc_operator(obj, θ_ext::AbstractVector, octx::OriginZCCoreHessCtx; skip_fill::Bool = false)
    # Legacy-H cleanup (2026-07-28): mirrors inner_loop_internal_cmlookup_production/
    # inner_loop_internal_meanzc_operator's identical dispatch.
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

    # True no-H operator bundle (2026-07-28 continuation): same dispatch as flexible-CM's
    # inner_loop_internal_cmlookup_production.
    if obj isa OperatorPsiBundle
        θ_econ = @view θ_ext[1:end-n_eta_params]
        prime_operator!(obj, θ_econ, octx.econ_ctx, octx.core_cf_ref; restriction_state = octx)
    else
        moments_fn = (skip_fill && octx.moments_skip! !== nothing) ? octx.moments_skip! : obj.moments!
        moments_fn(@view(obj.H[:, 1]), CS.select_G_from_H(obj, obj.H), θ_ext, obj.U, obj)
        obj.H[:, 2] .= 1.0
        obj.H_save = obj.H[1, 1] * (-1.0)^obj.find_smallest
    end

    if octx.fg_lookup_st === nothing
        octx.fg_lookup_st = OriginZCOperatorState(obj, octx.NCORE - 1, octx.fg_zc_op::ZCRestrictionOperator, octx.fg_layout, octx.core_cf_ref)
    end
    st = octx.fg_lookup_st::OriginZCOperatorState
    reset_for_solve!(st, collect(νfull))

    if INNER_SOLVE_TRACE[] !== nothing
        _IST_nfg[] = 0; _IST_f1[] = NaN; _IST_first_below[] = 0
    end
    _ist_t0 = INNER_SOLVE_TRACE[] === nothing ? 0.0 : time()
    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_originzc_operator(obj, st;
        hess_cb_builder = _ -> archA_partitioned_hess_cb_builder(octx))
    INNER_SOLVE_TRACE[] !== nothing && push!(INNER_SOLVE_TRACE[],
        (status = nStatus, n_fg = n_fg, n_hess = n_hess, wall = time() - _ist_t0,
         f_first = _IST_f1[], first_below = _IST_first_below[]))

    CS.INNER_SOLVE_COUNT[] += 1
    if nStatus ∉ [0, -100, -101, -103]
        CS.INNER_INFEAS_COUNT[] += 1
    end
    if nStatus ∈ [0, -100, -101, -103]
        obj.x .= x
        return obj.H_save, x, nStatus, n_fg, n_hess
    else
        # 2026-08-11: only clear the warm-start slot when INNER_KEEP_LAST_GOOD_X[] is false.
        # See that Ref's docstring -- the historical NaN discards the last SUCCESSFUL dual, forcing
        # every post-failure solve cold, which at a high failure rate is nearly all of them.
        INNER_KEEP_LAST_GOOD_X[] || (obj.x .= NaN)
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
function _originzc_fg_dispatch(ctx_cm, obj, θ_ext::AbstractVector; skip_fill::Bool = false)
    octx = ctx_cm.octx
    if octx.fg_backend === :operator
        return inner_loop_internal_originzc_operator(obj, θ_ext, octx; skip_fill = skip_fill)
    elseif octx.fg_backend === :dense_reference
        record_generic_dense_fg!()
        return inner_loop_internal_archgeneric(obj, θ_ext; hess_cb_builder = _ -> _originzc_hess_cb_builder(ctx_cm))
    else
        error("_originzc_fg_dispatch: fg_backend must be :dense_reference or :operator, got :$(octx.fg_backend)")
    end
end
