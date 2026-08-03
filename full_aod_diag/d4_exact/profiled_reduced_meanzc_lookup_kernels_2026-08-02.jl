# ============================================================================
# Integration continuation (2026-08-02), ZC-lane follow-up: a genuine, matrix-free (zero dense-G) FG
# evaluator for CM+ZC's ("mean ZC") REDUCED (profiled destination-scales) economic layout, combined
# with its EXISTING mean/pair-ZC restriction block (ZCRestrictionOperator/SharedByPowerLayout) and
# CM-grid restriction block, both REUSED COMPLETELY UNCHANGED.
#
# SAME EXPLICIT DESIGN CONSTRAINT as the other three families' own reduced operator FGs (flexible
# CM/common Frechet/origin-ZC, profiled_reduced_lookup_kernels_2026-08-02.jl /
# profiled_reduced_frechet_lookup_kernels_2026-08-02.jl / profiled_reduced_originzc_lookup_kernels_
# 2026-08-02.jl): this is a SURGICAL adaptation of the existing, already-validated, dense-width
# operator FG (`CMMeanZCOperatorState`, cm_meanzc_lookup_kernels.jl) -- the ONLY change is the
# economic block's own moment definitions. Concretely, relative to `CMMeanZCOperatorState`:
#   - `economic_forward!(st.econ_buf, λ_E, cf, st.econ_ws)` (dense, economic_operator.jl) is replaced
#     by `economic_forward_into_arg0_reduced!` (profiled_reduced_lookup_kernels_2026-08-02.jl, REUSED
#     UNCHANGED -- duck-typed on `st.n_econ`/`st.core_cf_ref`/`st.ctx`/`st.θ_full`/`st.layout`, all
#     present on this new state below with the SAME names for exactly this reason).
#   - `economic_transpose!` is likewise replaced by `economic_transpose_into_g1_and_gE_reduced!`
#     (same file, REUSED UNCHANGED), which additionally needs `st.arg1`/`st.B`/`st.Tslot` -- present
#     below, sized exactly as the other three families' own reduced states already do.
#   - The Z (mean/pair) block -- `restriction_forward!`/`restriction_transpose!`/`refresh_zc_targets!`
#     (zc_restriction_operator.jl) against a `ZCRestrictionOperator`+`SharedByPowerLayout` -- and the
#     CM-grid block (`apply_contrast!`/`suffix_sums!`/`cumulative_forward_contribution!`/
#     `build_weighted_histogram!`/`cumulative_backward_gradient!`, cm_lookup_kernels.jl) are REUSED
#     COMPLETELY UNCHANGED, called EXACTLY as `CMMeanZCOperatorState`'s own functor already calls
#     them (same inline style, not the cm_forward_contribution!/cm_transpose_into_g! wrapper flexible
#     CM's own reduced state uses, since THIS family's existing dense state does not use that wrapper
#     either -- matching the existing code's own convention, not introducing a new one).
#   - NO GRAVITY TERM: `CMMeanZCOperatorState`'s own `ncore1` ALREADY excludes gravity (`ncore1 =
#     pregrav = ncore_econ - 1`); this reduced sibling's `n_econ` is the exact same "no gravity"
#     quantity (`layout.total_reduced_economic_moments`), so no change of convention is needed here
#     at all -- gravity was never in this family's inner dual to begin with.
#   - The Hessian callback (`archC_hess_cb_builder(cctx)` -> `hessian_cm_structured!` ->
#     `_fill_cm_HEE!`'s `ncore_core < NCORE` widened branch, cm_hessian_architectures.jl) is REUSED
#     COMPLETELY UNCHANGED -- confirmed by direct code read (`_fill_cm_HEE!` line ~984: requires
#     `n == cctx.ncore_core == 1 + layout.total_reduced_economic_moments` for the profiled case,
#     exactly matching this file's own `n_econ`) -- this is the SAME `CMBinHessCtx`/builder flexible
#     CM's own reduced state already uses; CM+ZC's widening lives entirely in `cctx` (`ncore_core` vs
#     `NCORE`), not in a family-specific builder function.
#
# `cctx.hzz_zc_op`/`cctx.hzz_zc_layout` (built ALWAYS by `build_cm_meanzc_bin_ctx`, independent of
# `inner_fg_backend`, per that function's own comment) are REUSED DIRECTLY here -- immutable
# structural wrappers around the same `Zraw_all`/`Zpairraw_all` matrices, safe to share. A FRESH
# `ZCRestrictionWorkspace` is built for this state's own mutable per-solve target scratch (never
# touch a live Hessian-side state's own buffers -- same convention origin-ZC's own reduced state and
# family adapter already follow).
# ============================================================================

isdefined(Main, :economic_forward_into_arg0_reduced!) || error("profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl requires profiled_reduced_lookup_kernels_2026-08-02.jl to be included first.")
isdefined(Main, :ZCRestrictionOperator) || error("profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl requires zc_restriction_operator.jl to be included first.")
isdefined(Main, :SharedByPowerLayout) || error("profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl requires cm_originzc_target_layout.jl to be included first.")
isdefined(Main, :CMMeanZCOperatorState) || error("profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl requires cm_meanzc_lookup_kernels.jl to be included first.")

"Persistent per-inner-solve state for CM+ZC's ('mean ZC') reduced economic + mean/pair-ZC + CM-grid operator FG. Fields mirror `CMMeanZCOperatorState` exactly (same Z/CM-grid scratch, same `hw_cache` contract) plus the reduced-economic-specific `ctx`/`layout`/`θ_full`/`n_econ`/`B`/`Tslot` (same names `economic_forward_into_arg0_reduced!`/`economic_transpose_into_g1_and_gE_reduced!` expect, by construction)."
mutable struct ReducedCMMeanZCOperatorState
    obj::OperatorPsiBundle
    ctx::Any
    layout::ProfiledEconomicMomentLayout
    θ_full::Vector{Float64}
    n_econ::Int
    zc_op::ZCRestrictionOperator
    zc_layout::Any                   # SharedByPowerLayout(K_mean, K_pair)
    zc_ws::ZCRestrictionWorkspace
    core_cf_ref::Ref{Any}
    # ---- CM-grid block (mirrors CMMeanZCOperatorState's own fields verbatim) ----
    ncm::Int
    L::Int
    nO::Int
    origins::Vector{Int}
    refIndex1::Int
    bins::Matrix{<:Unsigned}
    R::Union{Nothing,Matrix{Float64}}
    nbins::Int
    λmat_ext::Matrix{Float64}
    cm_contrib::Vector{Float64}
    λmat_block::Matrix{Float64}
    hist_partials::Vector{Matrix{Float64}}
    hist_h::Matrix{Float64}
    Hpre::Matrix{Float64}
    g_block::Matrix{Float64}
    g_stored::Matrix{Float64}
    # ---- shared scratch ----
    arg0::Vector{Float64}
    arg1::Vector{Float64}
    B::Matrix{Float64}       # reduced-economic transpose-contraction scratch (D x Ddest)
    Tslot::Vector{Float64}   # reduced-economic transpose-contraction scratch (Ddest)
    n_fg_calls::Int
    hw_cache::HessianWeightCache
end

# Same forward as the other three families' own reduced states (profiled_reduced_lookup_kernels_
# 2026-08-02.jl / profiled_reduced_frechet_lookup_kernels_2026-08-02.jl / profiled_reduced_originzc_
# lookup_kernels_2026-08-02.jl): the shared, family-agnostic outer-gradient engine reads `st.cf`
# directly; this state carries `core_cf_ref::Ref{Any}` (the box `prime_operator!` publishes into).
Base.getproperty(st::ReducedCMMeanZCOperatorState, s::Symbol) = s === :cf ? getfield(st, :core_cf_ref)[] : getfield(st, s)

function ReducedCMMeanZCOperatorState(obj::OperatorPsiBundle, ctx, layout::ProfiledEconomicMomentLayout, θ_full::Vector{Float64},
        zc_op::ZCRestrictionOperator, zc_layout, ncm::Int, L::Int, origins::Vector{Int}, refIndex1::Int,
        bins::Matrix{<:Unsigned}, R; core_cf_ref::Ref{Any} = Ref{Any}(nothing))
    n_econ = layout.total_reduced_economic_moments
    nO = length(origins)
    M = obj.M
    nbins = L + 1
    D = size(bins, 2)
    Ddest = cf_ddest_hint(ctx)
    n_x = 1 + n_econ + n_mean(zc_op) + n_pair(zc_op) + ncm
    ReducedCMMeanZCOperatorState(obj, ctx, layout, θ_full, n_econ, zc_op, zc_layout, ZCRestrictionWorkspace(zc_op), core_cf_ref,
        ncm, L, nO, origins, refIndex1, bins, R, nbins,
        zeros(nO, L + 1), zeros(M), zeros(nO, L), [zeros(D, nbins)], zeros(D, nbins), zeros(D, L), zeros(nO, L), zeros(nO, L),
        zeros(M), zeros(M), zeros(D, Ddest), zeros(Ddest), 0,
        HessianWeightCache(n_x))
end

"Call ONCE per inner solve: refresh Z-block targets for the current outer point's ν (νvec, length K_mean -- SharedByPowerLayout broadcasts each ν_k to every origin/pair), reset counters."
function reset_for_solve!(st::ReducedCMMeanZCOperatorState, νvec::AbstractVector{Float64})
    refresh_zc_targets!(st.zc_ws, st.zc_op, st.zc_layout, νvec)
    st.n_fg_calls = 0
    return st
end

"""
    dual_index!(st::ReducedCMMeanZCOperatorState, x) -> st.arg0

`x = [zeta; beta_econ(n_econ); lambda_mean(n_mean); lambda_pair(n_pair); lambda_cm(ncm)]` -- no
gravity. Computes `st.arg0` in place: economic via the reduced kernel (sets `st.arg0 = -zeta -
t_econ`), then Z (mean/pair) and CM-grid blocks ACCUMULATE into it, exactly mirroring
`CMMeanZCOperatorState.dual_index!`'s own three-block ordering.
"""
function dual_index!(st::ReducedCMMeanZCOperatorState, x::AbstractVector{Float64})
    op = st.zc_op
    λ_mean = @view x[2+st.n_econ : 1+st.n_econ+n_mean(op)]
    λ_pair = @view x[2+st.n_econ+n_mean(op) : 1+st.n_econ+n_mean(op)+n_pair(op)]
    λ_cm = @view x[2+st.n_econ+n_mean(op)+n_pair(op) : 1+st.n_econ+n_mean(op)+n_pair(op)+st.ncm]

    economic_forward_into_arg0_reduced!(st, x)
    restriction_forward!(st.arg0, λ_mean, λ_pair, op, st.zc_ws)

    λmat_stored = reshape(λ_cm, st.nO, st.L)
    apply_contrast!(st.λmat_block, λmat_stored, st.R)
    suffix_sums!(st.λmat_ext, st.λmat_block)
    cumulative_forward_contribution!(st.cm_contrib, st.bins, st.refIndex1, st.origins, st.λmat_ext)
    st.arg0 .-= st.cm_contrib
    return st.arg0
end

"""
    (st::ReducedCMMeanZCOperatorState)(x, g=Float64[]) -> f

FG evaluator, same signature/semantics as `(st::CMMeanZCOperatorState)(x, g)`.
"""
function (st::ReducedCMMeanZCOperatorState)(x::AbstractVector{Float64}, g::AbstractVector{Float64} = Float64[])
    obj = st.obj
    M = obj.M
    op = st.zc_op
    ζ = x[1]

    cf = st.core_cf_ref[]
    dual_index!(st, x)

    obj.Psi!(st.arg1, st.arg0)
    f = sum(st.arg1) / M + ζ

    if length(g) > 0
        economic_transpose_into_g1_and_gE_reduced!(g, st, cf)
        restriction_transpose!((@view g[2+st.n_econ:1+st.n_econ+n_mean(op)]),
                                (@view g[2+st.n_econ+n_mean(op):1+st.n_econ+n_mean(op)+n_pair(op)]),
                                st.arg1, op, st.zc_ws)

        build_weighted_histogram!(st.hist_h, st.hist_partials, st.bins, st.arg1, size(st.bins, 2), st.nbins)
        cumulative_backward_gradient!(st.g_block, st.Hpre, st.hist_h, st.refIndex1, st.origins, st.L, M)
        apply_contrast!(st.g_stored, st.g_block, st.R)
        cm_off = 1 + st.n_econ + n_mean(op) + n_pair(op)
        @views g[cm_off+1:cm_off+st.ncm] .= vec(st.g_stored)
    end

    obj.arg0 .= st.arg0
    _publish_dual_index_cache!(st, x)
    st.n_fg_calls += 1
    return f
end

"""
    build_reduced_meanzc_operator_bundle(ctx, θ_full, layout, cctx; ref_obj=ctx.obj) -> (obj, st)

Builds the `OperatorPsiBundle`/`ReducedCMMeanZCOperatorState` pair for CM+ZC's reduced+mean/pair-ZC+
CM-grid dual problem. `cctx` is the ALREADY-BUILT reduced `CMBinHessCtx`
(`build_cm_meanzc_bin_ctx(...; profiled_layout=layout, ...)`) -- reuses its `core_cf_ref` (SAME box
the unchanged Hessian callback reads) and its always-built `hzz_zc_op`/`hzz_zc_layout` (immutable
structural wrappers, safe to share; this bundle's own `ZCRestrictionWorkspace` is fresh/independent).
"""
function build_reduced_meanzc_operator_bundle(ctx, θ_full::AbstractVector{Float64}, layout::ProfiledEconomicMomentLayout,
        cctx::CMBinHessCtx; ref_obj = ctx.obj)
    n = 1 + layout.total_reduced_economic_moments + n_mean(cctx.hzz_zc_op) + n_pair(cctx.hzz_zc_op) + cctx.ncm   # zeta+econ+mean+pair+CM-grid, no gravity
    obj = OperatorPsiBundle(
        δ = ref_obj.δ, find_smallest = ref_obj.find_smallest, γ = ref_obj.γ, l = ref_obj.l,
        inequality_index = Int[], U = ref_obj.U, outer_constr_index = n,
        inner_loop_opt = ref_obj.inner_loop_opt, Psi! = ref_obj.Psi!, dPsi! = ref_obj.dPsi!, ddPsi! = ref_obj.ddPsi!,
        lower_limit = ref_obj.lower_limit,
    )
    bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
    st = ReducedCMMeanZCOperatorState(obj, ctx, layout, collect(Float64, θ_full), cctx.hzz_zc_op, cctx.hzz_zc_layout,
        cctx.ncm, cctx.L, cctx.origins, cctx.refIndex1, bins_u, cctx.R; core_cf_ref = cctx.core_cf_ref)
    return obj, st
end

"""
    inner_loop_KNITRO_reduced_meanzc(obj, st, cctx) -> (nStatus, objSol, x, lambda_, n_fg, n_hess)

FG callback: `_callbackEvalFG_inner_cmlookup!` (cm_lookup_live_knitro.jl), REUSED UNCHANGED (already
type-agnostic in `st`, same as the other three families' own reduced drivers). Hessian callback:
`archC_hess_cb_builder(cctx)` (cm_hessian_architectures.jl), REUSED UNCHANGED -- CM+ZC's widening is
entirely internal to `cctx` (`ncore_core < NCORE`), the builder call itself is identical to flexible
CM's own.
"""
function inner_loop_KNITRO_reduced_meanzc(obj::OperatorPsiBundle, st::ReducedCMMeanZCOperatorState, cctx::CMBinHessCtx)
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
            hess_cb_raw = archC_hess_cb_builder(cctx)
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
    reduced_meanzc_base_state(x_free0, νvec, ctx, layout, cctx) -> (obj, st, inner_status, ζstar, λstar, n_fg, n_hess)

Faithful reduced-operator mirror of `archC_meanzc_base_state`. Publishes `cctx`'s shared state
(`profiled_theta_ref`, `cmlookup_st`, `inner_fg_backend`) exactly like `reduced_cm_base_state`/
`reduced_frechet_base_state`/`reduced_originzc_base_state` before calling the inner solve.
"""
function reduced_meanzc_base_state(x_free0::AbstractVector, νvec::AbstractVector{Float64}, ctx,
        layout::ProfiledEconomicMomentLayout, cctx::CMBinHessCtx)
    θ_full0 = CS.reconstruct_full(x_free0, ctx.m)
    obj, st = build_reduced_meanzc_operator_bundle(ctx, θ_full0, layout, cctx)
    prime_operator!(obj, θ_full0, ctx, cctx.core_cf_ref; restriction_state = cctx)
    reset_for_solve!(st, νvec)
    # CM+ZC E/C/Z block-partition + H_CZ/H_ZZ release (2026-07-27): publish the CURRENT νvec into
    # cctx.nu_ref for the Hessian callback's shared H_ZZ/H_CZ primitives to read (mirrors
    # archC_meanzc_base_state's own identical line -- SEPARATE box from this state's own st.zc_ws,
    # same class of gap as origin-ZC's octx.nu_ref[] fix). Must happen BEFORE the inner solve.
    cctx.nu_ref[] = collect(νvec)
    cctx.profiled_theta_ref[] = copy(θ_full0)
    cctx.cmlookup_st = st
    cctx.inner_fg_backend = :cm_lookup
    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_reduced_meanzc(obj, st, cctx)
    nStatus ∈ (0, -100, -101, -103) || throw(CMExpectedSolveFailure("reduced_meanzc_base_state: inner solve failed, nStatus=$nStatus (x_free0=$x_free0)"))
    ζstar = x[1]; λstar = collect(x[2:end])
    return (obj = obj, st = st, inner_status = nStatus, ζstar = ζstar, λstar = λstar, n_fg = n_fg, n_hess = n_hess)
end
