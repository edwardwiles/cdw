# ============================================================================
# Integration continuation (2026-08-02): genuine, matrix-free reduced+Z(mean/pair) operator FG for
# ZC-only (origin-ZC) -- same surgical-adaptation discipline as flexible CM/common Fréchet.
#
# Relative to OriginZCOperatorState (cm_originzc_lookup_kernels.jl): ONLY the economic ([E]) piece
# is replaced -- and since `ReducedOriginZCOperatorState` below uses the SAME field names
# (`arg0`/`arg1`/`n_econ`/`ctx`/`layout`/`θ_full`/`core_cf_ref`/`B`/`Tslot`) as `ReducedCMLookupState`
# (profiled_reduced_lookup_kernels_2026-08-02.jl), `economic_forward_into_arg0_reduced!`/
# `economic_transpose_into_g1_and_gE_reduced!` are REUSED VERBATIM, UNCHANGED, with zero
# duplication -- Julia dispatches on the field ACCESS pattern, not the struct name. The Z
# (mean/pair) block reuses `restriction_forward!`/`restriction_transpose!`
# (zc_restriction_operator.jl) COMPLETELY UNCHANGED, exactly as OriginZCOperatorState's own
# docstring already states ("gravity itself is OUTER-only, never part of the inner-dual lambda").
#
# NO GRAVITY: OriginZCOperatorState's own dense FG never had one either (confirmed by direct read:
# `x = [zeta; lambda_E; lambda_mean; lambda_pair]`, no gravity term anywhere), so this reduced
# sibling correctly has none either.
# ============================================================================

isdefined(Main, :economic_forward_into_arg0_reduced!) || error("profiled_reduced_originzc_lookup_kernels_2026-08-02.jl requires profiled_reduced_lookup_kernels_2026-08-02.jl to be included first.")
isdefined(Main, :ZCRestrictionOperator) || error("profiled_reduced_originzc_lookup_kernels_2026-08-02.jl requires zc_restriction_operator.jl to be included first.")

"Persistent per-inner-solve state for ZC-only's reduced+Z(mean/pair) operator FG. Field names deliberately match `ReducedCMLookupState` (`arg0`/`arg1`/`n_econ`/`ctx`/`layout`/`θ_full`/`core_cf_ref`/`B`/`Tslot`) so `economic_forward_into_arg0_reduced!`/`economic_transpose_into_g1_and_gE_reduced!` are reusable verbatim."
mutable struct ReducedOriginZCOperatorState
    obj::OperatorPsiBundle
    ctx::Any
    layout::ProfiledEconomicMomentLayout
    θ_full::Vector{Float64}
    n_econ::Int
    op::ZCRestrictionOperator
    zc_layout::Any
    core_cf_ref::Ref{Any}
    zc_ws::ZCRestrictionWorkspace
    arg0::Vector{Float64}
    arg1::Vector{Float64}
    B::Matrix{Float64}
    Tslot::Vector{Float64}
    n_fg_calls::Int
    hw_cache::HessianWeightCache
end

function ReducedOriginZCOperatorState(obj::OperatorPsiBundle, ctx, layout::ProfiledEconomicMomentLayout, θ_full::Vector{Float64},
        op::ZCRestrictionOperator, zc_layout; core_cf_ref::Ref{Any} = Ref{Any}(nothing))
    n_econ = layout.total_reduced_economic_moments
    W = obj.M
    Ddest = cf_ddest_hint(ctx)
    D = ctx.D
    n_x = 1 + n_econ + n_mean(op) + n_pair(op)
    ReducedOriginZCOperatorState(obj, ctx, layout, θ_full, n_econ, op, zc_layout, core_cf_ref,
        ZCRestrictionWorkspace(op), zeros(W), zeros(W), zeros(D, Ddest), zeros(Ddest), 0,
        HessianWeightCache(n_x))
end

"""
    dual_index!(st::ReducedOriginZCOperatorState, x) -> st.arg0

`x = [zeta; beta_econ(n_econ); lambda_mean(n_mean); lambda_pair(n_pair)]` -- no gravity.
"""
function dual_index!(st::ReducedOriginZCOperatorState, x::AbstractVector{Float64})
    op = st.op
    λ_mean = @view x[2+st.n_econ : 1+st.n_econ+n_mean(op)]
    λ_pair = @view x[2+st.n_econ+n_mean(op) : 1+st.n_econ+n_mean(op)+n_pair(op)]

    economic_forward_into_arg0_reduced!(st, x)
    restriction_forward!(st.arg0, λ_mean, λ_pair, op, st.zc_ws)
    return st.arg0
end

"""
    (st::ReducedOriginZCOperatorState)(x, g=Float64[]) -> f

FG evaluator. `[E]` prefix (forward+backward) reuses flexible CM's own reduced economic functions
unchanged; Z (mean/pair) reuses `restriction_forward!`/`restriction_transpose!` unchanged.
"""
function (st::ReducedOriginZCOperatorState)(x::AbstractVector{Float64}, g::AbstractVector{Float64} = Float64[])
    obj = st.obj
    M = obj.M
    ζ = x[1]
    op = st.op

    cf = st.core_cf_ref[]
    dual_index!(st, x)

    obj.Psi!(st.arg1, st.arg0)
    f = sum(st.arg1) / M + ζ

    if length(g) > 0
        economic_transpose_into_g1_and_gE_reduced!(g, st, cf)
        restriction_transpose!((@view g[2+st.n_econ : 1+st.n_econ+n_mean(op)]),
                                (@view g[2+st.n_econ+n_mean(op) : 1+st.n_econ+n_mean(op)+n_pair(op)]),
                                st.arg1, op, st.zc_ws)
    end

    obj.arg0 .= st.arg0
    _publish_dual_index_cache!(st, x)
    st.n_fg_calls += 1
    return f
end

"""
    build_reduced_originzc_operator_bundle(ctx, θ_full, layout, octx; ref_obj=ctx.obj) -> (obj, st)

ZC-only analogue of `build_reduced_cm_operator_bundle`. `octx` is the ALREADY-BUILT reduced
`OriginZCCoreHessCtx` (`build_originzc_core_hess_ctx(...; profiled_layout=layout, ...)`).
"""
function build_reduced_originzc_operator_bundle(ctx, θ_full::AbstractVector{Float64}, layout::ProfiledEconomicMomentLayout,
        octx::OriginZCCoreHessCtx; ref_obj = ctx.obj)
    op = octx.hzz_zc_op
    op === nothing && error("build_reduced_originzc_operator_bundle: octx.hzz_zc_op is nothing -- octx.n_eta must be > 0.")
    zc_layout = octx.hzz_zc_layout
    n = 1 + layout.total_reduced_economic_moments + n_mean(op) + n_pair(op)   # zeta + econ + mean + pair, no gravity
    obj = OperatorPsiBundle(
        δ = ref_obj.δ, find_smallest = ref_obj.find_smallest, γ = ref_obj.γ, l = ref_obj.l,
        inequality_index = Int[], U = ref_obj.U, outer_constr_index = n,
        inner_loop_opt = ref_obj.inner_loop_opt, Psi! = ref_obj.Psi!, dPsi! = ref_obj.dPsi!, ddPsi! = ref_obj.ddPsi!,
        lower_limit = ref_obj.lower_limit,
    )
    st = ReducedOriginZCOperatorState(obj, ctx, layout, collect(Float64, θ_full), op, zc_layout; core_cf_ref = octx.core_cf_ref)
    return obj, st
end

"""
    inner_loop_KNITRO_reduced_originzc(obj, st, octx) -> (nStatus, objSol, x, lambda_, n_fg, n_hess)

Hessian via `archA_partitioned_hess_cb_builder(octx)` (cm_hessian_architectures.jl) -- ZC-only's
OWN Hessian callback builder (NOT flexible CM's `archC_hess_cb_builder`, which dispatches on
`cctx::CMBinHessCtx`, a different struct than `octx::OriginZCCoreHessCtx`).
"""
function inner_loop_KNITRO_reduced_originzc(obj::OperatorPsiBundle, st::ReducedOriginZCOperatorState, octx::OriginZCCoreHessCtx)
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
            hess_cb_raw = archA_partitioned_hess_cb_builder(octx)
            hess_cb_adapted = (kc2, cb2, evalRequest, evalResult, userParams) -> begin
                r = _adapt_hess_cb_for_lookup(hess_cb_raw)(kc2, cb2, evalRequest, evalResult, userParams)
                n_hess[] += 1
                return r
            end
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, hess_cb_adapted)
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
    reduced_originzc_base_state(x_free0, ctx, layout, octx, νfull) -> NamedTuple

Faithful reduced-operator mirror of `archOZ_base_state`, INCLUDING its own critical
`octx.nu_ref[] = collect(νfull)` publish (found live, 2026-08-02: the Hessian callback's own
shared H_ZZ primitive reads `octx.nu_ref[]` -- a SEPARATE box from `st.zc_ws`, which only the FG
side reads -- and refreshes ITS OWN internal ZC workspace from it; omitting this publish crashes
with `BoundsError: attempt to access 0-element Vector{Float64} at index [1]` deep inside
`refresh_zc_targets!`/`mean_targets`, since that internal workspace was never sized/targeted at
all. Mirrors `archOZ_base_state`'s own identical comment: "must happen BEFORE the inner solve").
"""
function reduced_originzc_base_state(x_free0::AbstractVector, ctx, layout::ProfiledEconomicMomentLayout,
        octx::OriginZCCoreHessCtx, νfull::AbstractVector{Float64})
    θ_full0 = CS.reconstruct_full(x_free0, ctx.m)
    obj, st = build_reduced_originzc_operator_bundle(ctx, θ_full0, layout, octx)
    refresh_zc_targets!(st.zc_ws, st.op, st.zc_layout, νfull)
    octx.nu_ref[] = collect(νfull)
    prime_operator!(obj, θ_full0, ctx, octx.core_cf_ref; restriction_state = octx)
    octx.profiled_theta_ref[] = copy(θ_full0)
    octx.fg_lookup_st = st
    octx.fg_backend = :operator
    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_reduced_originzc(obj, st, octx)
    nStatus ∈ (0, -100, -101, -103) || throw(CMExpectedSolveFailure("reduced_originzc_base_state: inner solve failed, nStatus=$nStatus (x_free0=$x_free0)"))
    ζstar = x[1]; λstar = collect(x[2:end])
    return (obj = obj, st = st, inner_status = nStatus, ζstar = ζstar, λstar = λstar, n_fg = n_fg, n_hess = n_hess)
end
