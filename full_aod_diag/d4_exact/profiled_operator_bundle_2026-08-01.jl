# ============================================================================
# Claude Code task 2026-08-01, §9: a REAL unrestricted `OperatorPsiBundle` for
# `economic_parameterization = :profiled_destination_scales`, going through
# the SAME public CC/KNITRO inner solver the `:full_gamma_normalized_reference`
# path uses (`inner_loop_KNITRO_compressed`, compressed_live.jl) -- this file
# is a faithful sibling of that driver, swapping only the two callbacks and
# the dual-vector dimension, exactly mirroring the master report's own §5a
# "concrete, ready-to-execute path" note. ADDITIVE ONLY -- does not modify
# compressed_live.jl, operator_psi_bundle.jl, or any 2026-07-31 file.
#
# Production default is untouched: this file adds a NEW driver function and a
# NEW bundle-construction helper; nothing here is called unless a caller
# explicitly asks for `:profiled_destination_scales` (task requirement:
# PRODUCTION_DEFAULT_CHANGED = false).
# ============================================================================

isdefined(Main, :ProfiledEconomicMomentLayout) || error("profiled_operator_bundle_2026-08-01.jl requires profiled_economic_moment_layout_2026-08-01.jl to be included first.")
isdefined(Main, :reduced_homogeneous_dual_contraction) || error("profiled_operator_bundle_2026-08-01.jl requires reduced_homogeneous_contraction_2026-08-01.jl to be included first.")
isdefined(Main, :ReducedHomogeneousWinnerPairHessCtx) || error("profiled_operator_bundle_2026-08-01.jl requires reduced_homogeneous_hessian_2026-08-01.jl to be included first.")
isdefined(Main, :OperatorPsiBundle) || error("profiled_operator_bundle_2026-08-01.jl requires operator_psi_bundle.jl to be included first.")

"Per-inner-solve mutable bundle passed as KNITRO's userParams for the profiled/reduced callbacks. Faithful sibling of `CompressedCBState` (compressed_live.jl)."
mutable struct ProfiledCBState
    obj::OperatorPsiBundle
    ctx::Any                                             # the economic context (ctx.bi, ctx.γ.LPrime, etc.) -- NOT the bundle
    cf::CompressedFactual
    layout::ProfiledEconomicMomentLayout
    θ_full::Vector{Float64}
    B::Matrix{Float64}                                  # D x Ddest scratch (transpose contraction)
    Tslot::Vector{Float64}                               # Ddest scratch (transpose contraction)
    wctx::Union{Nothing,ReducedHomogeneousWinnerPairHessCtx}   # built lazily on first Hessian call this inner solve
end

ProfiledCBState(obj::OperatorPsiBundle, ctx, cf::CompressedFactual, layout::ProfiledEconomicMomentLayout, θ_full::Vector{Float64}) =
    ProfiledCBState(obj, ctx, cf, layout, θ_full, zeros(cf.D, cf.D_dest), zeros(cf.D_dest), nothing)

"""
    build_profiled_operator_bundle(ctx, θ_full, spec::AnchorSpec; ref_obj=ctx.obj) -> (obj, st)

Builds a genuine `OperatorPsiBundle` sized for the REDUCED (profiled-
destination-scales) dual dimension `n = 1 + layout.total_reduced_economic_moments`,
reusing `ref_obj`'s shared outer parameters (`δ`, `find_smallest`, `γ`, `l`,
`U`, `inner_loop_opt`, `Psi!`/`dPsi!`/`ddPsi!`) -- these describe the CC
inner-dual objective's OWN shape (the delta-grid/weight structure), which
does not depend on how many economic-moment columns feed it, exactly as
`ref_obj.M`/`U` are shared unchanged between the dense and compressed
unrestricted paths already. Also returns the `ProfiledCBState` needed by
`inner_loop_KNITRO_profiled`.
"""
function build_profiled_operator_bundle(ctx, θ_full::AbstractVector{Float64}, spec::AnchorSpec; ref_obj = ctx.obj)
    cf = build_compressed_factual(collect(θ_full), ctx; check_ties = false)
    has_france = cf.cf_col > 0
    layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
    assert_no_factual_price_index_moment(layout)
    n = 1 + layout.total_reduced_economic_moments

    obj = OperatorPsiBundle(
        δ = ref_obj.δ, find_smallest = ref_obj.find_smallest, γ = ref_obj.γ, l = ref_obj.l,
        inequality_index = Int[], U = ref_obj.U, outer_constr_index = n,
        inner_loop_opt = ref_obj.inner_loop_opt, Psi! = ref_obj.Psi!, dPsi! = ref_obj.dPsi!, ddPsi! = ref_obj.ddPsi!,
    )
    st = ProfiledCBState(obj, ctx, cf, layout, collect(Float64, θ_full))
    return obj, st
end

"""
    _callbackEvalFG_inner_profiled!(kc, cb, evalRequest, evalResult, userParams)

Reduced-dimension sibling of `_callbackEvalFG_inner_compressed!`: objective +
gradient w.r.t. `(zeta, beta_reduced)` from `reduced_homogeneous_dual_contraction`/
`reduced_homogeneous_transpose_contraction!` (task §6/§7 kernels), never
touching a full/anchor-inclusive dual vector at all.
"""
function _callbackEvalFG_inner_profiled!(kc, cb, evalRequest, evalResult, userParams)
    st = userParams
    obj = st.obj
    x = evalRequest.x
    ζ = x[1]
    β = @view x[2:end]
    t = reduced_homogeneous_dual_contraction(β, st.cf, st.ctx, st.θ_full, st.layout)
    W = st.cf.W
    q = -ζ .- t
    Psi_q = similar(q)
    dPsi_q = similar(q)
    obj.Psi!(Psi_q, q)
    obj.dPsi!(dPsi_q, q)
    M = obj.M
    evalResult.obj[1] = sum(Psi_q) / M + ζ
    evalResult.objGrad[1] = 1.0 - sum(dPsi_q) / M
    gβ = @view evalResult.objGrad[2:end]
    reduced_homogeneous_transpose_contraction!(gβ, dPsi_q, st.cf, st.ctx, st.θ_full, st.layout, st.B, st.Tslot)
    gβ .*= -1.0 / M
    obj.arg0 .= q
    st.wctx = nothing   # invalidate any cached Hessian context from a previous KNITRO point
    return 0
end

"""
    _callbackEvalH_inner_profiled!(kc, cb, evalRequest, evalResult, userParams)

Reduced-dimension sibling of `_callbackEvalH_inner_compressed!`'s
`:exact_winner_pair_serial` branch: builds (lazily, once per inner solve) a
`ReducedHomogeneousWinnerPairHessCtx` and calls
`reduced_homogeneous_winner_pair_hessian!` directly on `evalResult.hess` --
KNITRO's own packed dense-row-major upper triangle, no dense round-trip, no
legacy `obj.H` read (there is none: `OperatorPsiBundle` has no such field).
"""
function _callbackEvalH_inner_profiled!(kc, cb, evalRequest, evalResult, userParams)
    st = userParams
    obj = st.obj
    if st.wctx === nothing
        st.wctx = build_reduced_homogeneous_winner_pair_ctx(st.cf, st.ctx, st.θ_full, st.layout)
    end
    reduced_homogeneous_winner_pair_hessian!(evalResult.hess, obj, st.wctx)
    return 0
end

"""
    inner_loop_KNITRO_profiled(obj, st::ProfiledCBState) -> (nStatus, objSol, x, lambda_, n_fg_calls, n_hess_calls)

Faithful reduced-dimension mirror of `inner_loop_KNITRO_compressed`
(compressed_live.jl): identical KNITRO variable/bound/init-value setup and
option file, registering the PROFILED/REDUCED callbacks instead. `n_fg_calls`/
`n_hess_calls` are counted locally (this file deliberately does not touch the
shared `_INNER_CALL_COUNTERS` global the production compressed path uses, to
avoid any cross-talk with a concurrently-running production diagnostic).
"""
function inner_loop_KNITRO_profiled(obj::OperatorPsiBundle, st::ProfiledCBState; maxit_override::Union{Nothing,Int} = nothing)
    n_fg = Ref(0); n_hess = Ref(0)
    CS.guard_enter_inner_solve!()
    try
        kc = KNITRO.KN_new()
        KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
        KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
        KNITRO.KN_set_var_primal_init_values_all(kc, CS.inner_loop_initial_values(obj))

        function fg_counted!(kc_, cb_, evalRequest, evalResult, userParams)
            n_fg[] += 1
            return _callbackEvalFG_inner_profiled!(kc_, cb_, evalRequest, evalResult, userParams)
        end
        function h_counted!(kc_, cb_, evalRequest, evalResult, userParams)
            n_hess[] += 1
            return _callbackEvalH_inner_profiled!(kc_, cb_, evalRequest, evalResult, userParams)
        end

        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], fg_counted!)
        KNITRO.KN_set_cb_user_params(kc, cb, st)
        KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)
        # Diagnostic-only override (task §16: "increasing iteration budget" is not one of the
        # forbidden shortcuts -- looser tolerances/different algorithm/regularization/dropping
        # Hessian terms/clamping are -- and this does not touch the shared production .opt file).
        maxit_override !== nothing && KNITRO.KN_set_int_param_by_name(kc, "maxit", maxit_override)

        if KNITRO.KN_get_int_param(kc, "hessopt") == 1
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, h_counted!)
        end

        KNITRO.KN_solve(kc)
        nStatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
        CS.INNER_ITERS_TOTAL[] += CS._kn_num_iters(kc)
        KNITRO.KN_free(kc)
        return nStatus, objSol, x, lambda_, n_fg[], n_hess[]
    finally
        CS.guard_exit_inner_solve!()
    end
end
