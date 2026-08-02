# ============================================================================
# Integration continuation (2026-08-02): genuine, matrix-free reduced+CM-grid+level operator FG for
# common Fréchet -- same surgical-adaptation discipline as
# profiled_reduced_lookup_kernels_2026-08-02.jl (flexible CM).
#
# Relative to CMFrechetLookupState (cm_frechet_lookup_kernels.jl): ONLY the economic ([E]) piece is
# replaced, reusing `economic_forward_into_arg0_reduced!`/`economic_transpose_into_g1_and_gE_
# reduced!` (profiled_reduced_lookup_kernels_2026-08-02.jl) UNCHANGED -- the SAME two functions
# flexible CM's own reduced state uses, since the reduced economic block does not depend on which
# restriction family sits on top of it. The CM ([C]) and level ([F]) blocks are REUSED COMPLETELY
# UNCHANGED (`cm_forward_contribution!`/`cm_transpose_into_g!` from cm_lookup_kernels.jl,
# `frechet_level_suffix_sums!`/`frechet_level_forward_sum!`/`frechet_level_backward_gradient!` from
# cm_frechet_lookup_kernels.jl) -- exactly as CMFrechetLookupState's own docstring already states
# ("only the trailing [F] level-block extension is Fréchet-specific").
#
# NO GRAVITY (same user correction as flexible CM): CMFrechetLookupState's own dense FG never had a
# gravity term either (confirmed by direct read -- its `dual_index!`/functor only ever touch
# economic+CM+level), so this reduced sibling correctly has none either; x = [zeta; beta_econ
# (n_econ); lambda_cm (ncm_cm); lambda_level (ncm_level)].
# ============================================================================

isdefined(Main, :economic_forward_into_arg0_reduced!) || error("profiled_reduced_frechet_lookup_kernels_2026-08-02.jl requires profiled_reduced_lookup_kernels_2026-08-02.jl to be included first.")
isdefined(Main, :frechet_level_suffix_sums!) || error("profiled_reduced_frechet_lookup_kernels_2026-08-02.jl requires cm_frechet_lookup_kernels.jl to be included first.")

"Persistent per-inner-solve state for common Fréchet's reduced+CM-grid+level operator FG. Mirrors `CMFrechetLookupState` exactly (same CM/level scratch, same `hw_cache` contract) plus the reduced-economic-specific `ctx`/`layout`/`θ_full`/`n_econ`/`B`/`Tslot`."
mutable struct ReducedCMFrechetLookupState
    obj::OperatorPsiBundle
    ctx::Any
    layout::ProfiledEconomicMomentLayout
    θ_full::Vector{Float64}
    n_econ::Int
    ncm_cm::Int
    ncm_level::Int
    L::Int
    D::Int
    nO::Int
    origins::Vector{Int}
    refIndex1::Int
    bins::Matrix{<:Unsigned}
    R::Union{Nothing,Matrix{Float64}}
    nbins::Int
    level_targets::Vector{Float64}
    invsqrtD::Float64
    arg0::Vector{Float64}
    arg1::Vector{Float64}
    cm_contrib::Vector{Float64}
    level_contrib::Vector{Float64}
    λmat_block::Matrix{Float64}
    λmat_ext::Matrix{Float64}
    P_level::Vector{Float64}
    hist_partials::Vector{Matrix{Float64}}
    hist_h::Matrix{Float64}
    Hpre::Matrix{Float64}
    g_block::Matrix{Float64}
    g_stored::Matrix{Float64}
    g_level::Vector{Float64}
    core_cf_ref::Ref{Any}
    B::Matrix{Float64}
    Tslot::Vector{Float64}
    n_fg_calls::Int
    hw_cache::HessianWeightCache
end

# Same forward as `ReducedCMLookupState`'s own (profiled_reduced_lookup_kernels_2026-08-02.jl): the
# shared, family-agnostic outer-gradient engine reads `st.cf` directly; this state carries
# `core_cf_ref::Ref{Any}` (the box `prime_operator!` publishes into), so `st.cf` forwards there.
Base.getproperty(st::ReducedCMFrechetLookupState, s::Symbol) = s === :cf ? getfield(st, :core_cf_ref)[] : getfield(st, s)

function ReducedCMFrechetLookupState(obj::OperatorPsiBundle, ctx, layout::ProfiledEconomicMomentLayout, θ_full::Vector{Float64},
        ncm_cm::Int, ncm_level::Int, L::Int, D::Int, origins::Vector{Int}, refIndex1::Int, bins::Matrix{<:Unsigned}, R,
        level_targets::Vector{Float64}; core_cf_ref::Ref{Any} = Ref{Any}(nothing))
    ncm_level == L || error("ReducedCMFrechetLookupState: ncm_level=$ncm_level must equal L=$L")
    length(level_targets) == L || error("ReducedCMFrechetLookupState: length(level_targets)=$(length(level_targets)) != L=$L")
    n_econ = layout.total_reduced_economic_moments
    nO = length(origins)
    M = obj.M
    nbins = L + 1
    Ddest = cf_ddest_hint(ctx)
    ReducedCMFrechetLookupState(obj, ctx, layout, θ_full, n_econ, ncm_cm, ncm_level, L, D, nO, origins, refIndex1, bins, R,
        nbins, level_targets, 1.0 / sqrt(D),
        zeros(M), zeros(M), zeros(M), zeros(M),
        zeros(nO, L), zeros(nO, L + 1), zeros(L + 1),
        [zeros(D, nbins)], zeros(D, nbins), zeros(D, L), zeros(nO, L), zeros(nO, L), zeros(L),
        core_cf_ref, zeros(D, Ddest), zeros(Ddest), 0,
        HessianWeightCache(1 + n_econ + ncm_cm + ncm_level))
end

"""
    dual_index!(st::ReducedCMFrechetLookupState, x) -> st.arg0

`x = [zeta; beta_econ(n_econ); lambda_cm(ncm_cm); lambda_level(ncm_level)]` -- no gravity. `[E]`
prefix calls the SAME reduced-economic function flexible CM's own reduced state uses; `[C|F]` reuse
`cm_forward_contribution!`/the level forward kernels completely unchanged.
"""
function dual_index!(st::ReducedCMFrechetLookupState, x::AbstractVector{Float64})
    λ_cm = @view x[2+st.n_econ:1+st.n_econ+st.ncm_cm]
    λ_level = @view x[2+st.n_econ+st.ncm_cm:1+st.n_econ+st.ncm_cm+st.ncm_level]

    economic_forward_into_arg0_reduced!(st, x)
    cm_forward_contribution!(st, λ_cm, :suffix)

    frechet_level_suffix_sums!(st.P_level, λ_level)
    frechet_level_forward_sum!(st.level_contrib, st.bins, st.D, st.P_level)
    const_term = 0.0
    @inbounds for l in 1:st.L
        const_term += λ_level[l] * st.level_targets[l]
    end
    @inbounds for s in 1:length(st.arg0)
        st.arg0[s] -= st.invsqrtD * st.level_contrib[s] - const_term
    end
    return st.arg0
end

"""
    (st::ReducedCMFrechetLookupState)(x, g=Float64[]) -> f

FG evaluator. `[E]` prefix (forward+backward) reuses flexible CM's own reduced economic functions
unchanged; `[C|F]` reuse the existing unchanged CM/level kernels.
"""
function (st::ReducedCMFrechetLookupState)(x::AbstractVector{Float64}, g::AbstractVector{Float64} = Float64[])
    obj = st.obj
    M = obj.M
    ζ = x[1]

    cf = st.core_cf_ref[]
    dual_index!(st, x)

    obj.Psi!(st.arg1, st.arg0)
    f = sum(st.arg1) / M + ζ

    if length(g) > 0
        sum_dPsi = economic_transpose_into_g1_and_gE_reduced!(g, st, cf)
        cm_transpose_into_g!(g, st, :suffix, st.D, st.n_econ, st.ncm_cm, M)

        frechet_level_backward_gradient!(st.g_level, st.Hpre, st.D, st.L, M, st.invsqrtD, st.level_targets, sum_dPsi)
        @views g[2+st.n_econ+st.ncm_cm:1+st.n_econ+st.ncm_cm+st.ncm_level] .= st.g_level
    end

    obj.arg0 .= st.arg0
    _publish_dual_index_cache!(st, x)
    st.n_fg_calls += 1
    return f
end

"""
    build_reduced_frechet_operator_bundle(ctx, θ_full, layout, cctx; ref_obj=ctx.obj) -> (obj, st)

Common-Fréchet analogue of `build_reduced_cm_operator_bundle`. `cctx` is the ALREADY-BUILT reduced
`CMBinHessCtx` (`build_cm_bin_ctx(...; profiled_layout=layout, ...)` with a `CMFrechetExtension`
attached via `cctx.frechet_ext_cache`).
"""
function build_reduced_frechet_operator_bundle(ctx, θ_full::AbstractVector{Float64}, layout::ProfiledEconomicMomentLayout,
        cctx::CMBinHessCtx, level_targets::Vector{Float64}; ref_obj = ctx.obj)
    ncm_level = cctx.L
    ncm_cm = cctx.ncm - ncm_level
    n = 1 + layout.total_reduced_economic_moments + cctx.ncm   # zeta + econ + CM + level, no gravity
    obj = OperatorPsiBundle(
        δ = ref_obj.δ, find_smallest = ref_obj.find_smallest, γ = ref_obj.γ, l = ref_obj.l,
        inequality_index = Int[], U = ref_obj.U, outer_constr_index = n,
        inner_loop_opt = ref_obj.inner_loop_opt, Psi! = ref_obj.Psi!, dPsi! = ref_obj.dPsi!, ddPsi! = ref_obj.ddPsi!,
    )
    bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
    st = ReducedCMFrechetLookupState(obj, ctx, layout, collect(Float64, θ_full), ncm_cm, ncm_level, cctx.L, cctx.D,
        cctx.origins, cctx.refIndex1, bins_u, cctx.R, level_targets; core_cf_ref = cctx.core_cf_ref)
    return obj, st
end

"""
    inner_loop_KNITRO_reduced_frechet(obj, st, cctx, level_targets) -> (nStatus, objSol, x, lambda_, n_fg, n_hess)

Same wiring as `inner_loop_KNITRO_reduced_cmlookup` (FG via the reused `_callbackEvalFG_inner_
cmlookup!`), but the Hessian uses `archC_frechet_hess_cb_builder(cctx, level_targets)`
(cm_frechet_hessian.jl) -- NOT flexible CM's own `archC_hess_cb_builder` (an earlier draft of this
file used the wrong one, which never dispatches the level-block Hessian at all since it always
calls `hessian_cm_structured!` with `extension=nothing`) -- adapted via the reused
`_adapt_hess_cb_for_lookup`.
"""
function inner_loop_KNITRO_reduced_frechet(obj::OperatorPsiBundle, st::ReducedCMFrechetLookupState, cctx::CMBinHessCtx,
        level_targets::Vector{Float64})
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
            hess_cb_raw = archC_frechet_hess_cb_builder(cctx, level_targets)
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
    reduced_frechet_base_state(x_free0, ctx, layout, cctx, level_targets) -> NamedTuple

Faithful reduced-operator mirror of `archC_frechet_base_state`.
"""
function reduced_frechet_base_state(x_free0::AbstractVector, ctx, layout::ProfiledEconomicMomentLayout,
        cctx::CMBinHessCtx, level_targets::Vector{Float64})
    θ_full0 = CS.reconstruct_full(x_free0, ctx.m)
    obj, st = build_reduced_frechet_operator_bundle(ctx, θ_full0, layout, cctx, level_targets)
    prime_operator!(obj, θ_full0, ctx, cctx.core_cf_ref; restriction_state = cctx)
    cctx.profiled_theta_ref[] = copy(θ_full0)
    cctx.cmlookup_st = st
    cctx.inner_fg_backend = :cm_lookup
    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_reduced_frechet(obj, st, cctx, level_targets)
    nStatus ∈ (0, -100, -101, -103) || throw(CMExpectedSolveFailure("reduced_frechet_base_state: inner solve failed, nStatus=$nStatus (x_free0=$x_free0)"))
    ζstar = x[1]; λstar = collect(x[2:end])
    return (obj = obj, st = st, inner_status = nStatus, ζstar = ζstar, λstar = λstar, n_fg = n_fg, n_hess = n_hess)
end
