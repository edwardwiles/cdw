# ============================================================================
# Integration continuation (2026-08-02): a genuine, matrix-free (zero dense-G) FG evaluator for
# flexible CM's REDUCED (profiled destination-scales) economic layout, combined with its existing
# CM-grid restriction block.
#
# EXPLICIT DESIGN CONSTRAINT (user instruction, 2026-08-02): do not invent new architecture. This
# file is a SURGICAL adaptation of the existing, already-validated, already-optimized dense-width
# operator FG (CMLookupState, cm_lookup_kernels.jl) -- the only real difference between the reduced
# and non-reduced dual problems is the economic block's own moment definitions (reduced/anchor-
# excluded vs full/dense). Concretely, relative to CMLookupState:
#   - economic_forward_into_arg0! (dense, economic_forward!/economic_operator.jl) is replaced by
#     `economic_forward_into_arg0_reduced!` below, which is a near-verbatim copy calling
#     `reduced_homogeneous_dual_contraction` (reduced_homogeneous_contraction_2026-08-01.jl, already
#     built+gated by the inner-endtoend branch) instead.
#   - economic_transpose_into_g1_and_gE! is likewise replaced by
#     `economic_transpose_into_g1_and_gE_reduced!`, calling `reduced_homogeneous_transpose_
#     contraction!` instead of `economic_transpose!`.
#   - `cm_forward_contribution!`/`cm_transpose_into_g!` (cm_lookup_kernels.jl) are REUSED COMPLETELY
#     UNCHANGED -- the CM-grid restriction block's own math does not depend on the economic layout's
#     width at all, and immediately follows economic with no gap (`x = [zeta; econ(n_econ);
#     CM(ncm)]`), so `cm_transpose_into_g!`'s own `ncore1` argument is passed as `n_econ` directly,
#     unchanged from how any other family's dense case would use it.
#   - NO GRAVITY TERM (user correction, 2026-08-02): gravity is legacy -- not an inner moment at all
#     -- and must not appear in this dual problem. An earlier draft of this file added a separate
#     gravity dual index (mirroring `wrap_moments_with_cm_archB`'s own `fill_gravity_column_into!`
#     column), which is exactly why the real KNITRO solve failed (nStatus=-400): the existing
#     Hessian block-builder (`hessian_cm_structured!`, unchanged, see below) sizes `Hfull` as
#     exactly `(NCORE+ncm)x(NCORE+ncm)` with NO row/column for gravity at all (`HEE` covers only
#     zeta+economic, `CM cols` immediately follow with no gap) -- confirmed by direct read of
#     `cctx.Hfull`'s own field type comment and the `cols=NCORE+(l-1)*nO+1:NCORE+l*nO` block
#     assignment. Removing gravity from this file's own x-layout makes it match `Hfull`'s existing
#     dimension exactly, with no Hessian-side change needed.
#   - The Hessian callback (`archC_hess_cb_builder(cctx)` -> `hessian_cm_structured!` ->
#     `_fill_cm_HEE!`) is REUSED COMPLETELY UNCHANGED -- already confirmed (by direct code read) to
#     dispatch straight to `reduced_homogeneous_winner_pair_hessian!`/the winner-pair cross-Hessian
#     kernels for `profiled_layout !== nothing`, never reading a dense G/H at all.
#
# No dense G/H materialization anywhere in this file -- the ONLY per-draw (O(W)) buffers are
# `st.arg0`/`st.arg1`/`st.cm_contrib`/`obj.payoff`, all already-existing O(W) conventions this
# codebase uses throughout (never O(W*n)).
# ============================================================================

isdefined(Main, :ProfiledEconomicMomentLayout) || error("profiled_reduced_lookup_kernels_2026-08-02.jl requires profiled_economic_moment_layout_2026-08-01.jl to be included first.")
isdefined(Main, :reduced_homogeneous_dual_contraction) || error("profiled_reduced_lookup_kernels_2026-08-02.jl requires reduced_homogeneous_contraction_2026-08-01.jl to be included first.")
isdefined(Main, :OperatorPsiBundle) || error("profiled_reduced_lookup_kernels_2026-08-02.jl requires operator_psi_bundle.jl to be included first.")
isdefined(Main, :cm_forward_contribution!) || error("profiled_reduced_lookup_kernels_2026-08-02.jl requires cm_lookup_kernels.jl to be included first.")
isdefined(Main, :HessianWeightCache) || error("profiled_reduced_lookup_kernels_2026-08-02.jl requires operator_hessian_weights.jl to be included first.")

"""
    economic_forward_into_arg0_reduced!(st, x) -> st.arg0

Reduced-economic-layout sibling of `economic_forward_into_arg0!` (cm_lookup_kernels.jl): computes
`st.arg0 = -zeta - t_econ` in place. `t_econ` comes from `reduced_homogeneous_dual_contraction!` (the
SAME "safe by linearity" kernel `materialize_homogeneous_dense_G_reduced!` uses to build G's own
columns -- already gated), written directly into `st.arg0` itself (no fresh W-length allocation on
this hot path -- every family's inner-FG callback calls this once per KNITRO iteration). No gravity
term (user correction, 2026-08-02: gravity is legacy, not an inner moment, must not appear in this
dual problem at all).
"""
function economic_forward_into_arg0_reduced!(st, x::AbstractVector{Float64})
    ζ = x[1]
    β_econ = @view x[2:1+st.n_econ]

    cf = st.core_cf_ref[]
    cf isa CompressedFactual || error("economic_forward_into_arg0_reduced!: core_cf_ref[] is not a CompressedFactual (got $(typeof(cf))) -- prime_operator! must run before any FG/Hessian callback at this outer point.")
    reduced_homogeneous_dual_contraction!(st.arg0, β_econ, cf, st.ctx, st.θ_full, st.layout)
    st.arg0 .= (-ζ) .- st.arg0
    return cf
end

"""
    economic_transpose_into_g1_and_gE_reduced!(g, st, cf) -> sum_dPsi

Reduced-economic-layout sibling of `economic_transpose_into_g1_and_gE!`: same `g[1]` fill, economic
gradient via `reduced_homogeneous_transpose_contraction!` (in place, into `g[2:1+n_econ]`) instead of
the dense `economic_transpose!`. No gravity term.
"""
function economic_transpose_into_g1_and_gE_reduced!(g::AbstractVector{Float64}, st, cf)
    obj = st.obj
    M = obj.M
    obj.dPsi!(st.arg1, st.arg0)
    sum_dPsi = sum(st.arg1)
    g[1] = 1.0 - sum_dPsi / M
    g_econ = @view g[2:1+st.n_econ]
    reduced_homogeneous_transpose_contraction!(g_econ, st.arg1, cf, st.ctx, st.θ_full, st.layout, st.B, st.Tslot)
    g_econ .*= -(1.0 / M)
    return sum_dPsi
end

"Persistent per-inner-solve state for flexible CM's reduced+CM-grid operator FG. Fields mirror `CMLookupState` exactly (same CM-grid scratch, same `hw_cache` contract) plus the reduced-economic-specific `ctx`/`layout`/`θ_full`/`n_econ`/`B`/`Tslot`."
mutable struct ReducedCMLookupState
    obj::OperatorPsiBundle
    ctx::Any
    layout::ProfiledEconomicMomentLayout
    θ_full::Vector{Float64}
    n_econ::Int
    ncm::Int
    L::Int
    nO::Int
    origins::Vector{Int}
    refIndex1::Int
    bins::Matrix{<:Unsigned}
    R::Union{Nothing,Matrix{Float64}}
    method::Symbol
    nbins::Int
    arg0::Vector{Float64}
    arg1::Vector{Float64}
    cm_contrib::Vector{Float64}
    λmat_ext::Matrix{Float64}
    λmat_block::Matrix{Float64}
    hist_partials::Vector{Matrix{Float64}}
    hist_h::Matrix{Float64}
    Hpre::Matrix{Float64}
    g_block::Matrix{Float64}
    g_stored::Matrix{Float64}
    core_cf_ref::Ref{Any}
    B::Matrix{Float64}
    Tslot::Vector{Float64}
    n_fg_calls::Int
    hw_cache::HessianWeightCache
end

# `shared_family_outer_gradient`/`diag_profiled_full_rebuild_gradient` (the pre-existing,
# family-agnostic outer-gradient engine and its independent reference, both built for the
# unrestricted family's own `ProfiledCBState` which carries `cf::CompressedFactual` directly) read
# `st.cf`. This state instead carries `core_cf_ref::Ref{Any}` -- the SAME box `prime_operator!`
# (shared, unchanged, family-agnostic priming already used by every restricted family) publishes
# into -- so `st.cf` is forwarded to `core_cf_ref[]` here rather than duplicating a `cf` field that
# would need to be kept in sync by hand.
Base.getproperty(st::ReducedCMLookupState, s::Symbol) = s === :cf ? getfield(st, :core_cf_ref)[] : getfield(st, s)

function ReducedCMLookupState(obj::OperatorPsiBundle, ctx, layout::ProfiledEconomicMomentLayout, θ_full::Vector{Float64},
        ncm::Int, L::Int, origins::Vector{Int}, refIndex1::Int, bins::Matrix{<:Unsigned}, R;
        method::Symbol = :suffix, core_cf_ref::Ref{Any} = Ref{Any}(nothing))
    method in (:interval, :suffix) || error("ReducedCMLookupState: method must be :interval or :suffix, got $method")
    n_econ = layout.total_reduced_economic_moments
    nO = length(origins)
    M = obj.M
    nbins = L + 1
    D = size(bins, 2)
    Ddest = cf_ddest_hint(ctx)
    ReducedCMLookupState(obj, ctx, layout, θ_full, n_econ, ncm, L, nO, origins, refIndex1, bins, R, method, nbins,
        zeros(M), zeros(M), zeros(M), zeros(nO, L + 1),
        zeros(nO, L), [zeros(D, nbins)], zeros(D, nbins), zeros(D, L), zeros(nO, L), zeros(nO, L),
        core_cf_ref, zeros(D, Ddest), zeros(Ddest), 0,
        HessianWeightCache(1 + n_econ + ncm))
end

"`ctx.D_dest` if present (rectangular D!=Ddest support), else `ctx.D` -- same fallback convention `compressed_gravity_raw` (compressed_live.jl) already uses."
cf_ddest_hint(ctx) = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D

"""
    dual_index!(st::ReducedCMLookupState, x) -> st.arg0

`x = [zeta; beta_econ(n_econ); lambda_cm(ncm)]` -- no gravity. Computes `st.arg0` in place -- same
"extracted for the Hessian-weight cache to share" discipline as `dual_index!(::CMLookupState, x)`.
"""
function dual_index!(st::ReducedCMLookupState, x::AbstractVector{Float64})
    λ_cm = @view x[2+st.n_econ:1+st.n_econ+st.ncm]
    economic_forward_into_arg0_reduced!(st, x)
    cm_forward_contribution!(st, λ_cm, st.method)
    return st.arg0
end

"""
    (st::ReducedCMLookupState)(x, g=Float64[]) -> f

FG evaluator, same signature/semantics as `(st::CMLookupState)(x, g)`. `x = [zeta; beta_econ;
lambda_cm]` -- no gravity.
"""
function (st::ReducedCMLookupState)(x::AbstractVector{Float64}, g::AbstractVector{Float64} = Float64[])
    obj = st.obj
    M = obj.M
    ζ = x[1]

    cf = st.core_cf_ref[]
    dual_index!(st, x)

    obj.Psi!(st.arg1, st.arg0)
    f = sum(st.arg1) / M + ζ

    if length(g) > 0
        economic_transpose_into_g1_and_gE_reduced!(g, st, cf)
        D = size(st.bins, 2)
        # ncore1 = n_econ (economic width, no gravity gap) -- CM immediately follows economic,
        # exactly matching Hfull's own (NCORE+ncm)x(NCORE+ncm) block structure (HEE=zeta+econ,
        # CM cols immediately after, no gap).
        cm_transpose_into_g!(g, st, st.method, D, st.n_econ, st.ncm, M)
    end

    obj.arg0 .= st.arg0   # keep obj in sync for the Hessian callback, same trick as CMLookupState
    _publish_dual_index_cache!(st, x)
    st.n_fg_calls += 1
    return f
end

"""
    build_reduced_cm_operator_bundle(ctx, θ_full, layout, cctx; ref_obj=ctx.obj, method=:suffix) -> (obj, st)

Builds the `OperatorPsiBundle`/`ReducedCMLookupState` pair for flexible CM's reduced+CM-grid dual
problem. `cctx` is the ALREADY-BUILT reduced `CMBinHessCtx` (`build_cm_bin_ctx(...;
profiled_layout=layout, ...)`) -- this function does not build a new one, it wires this bundle's
`core_cf_ref` to the SAME `cctx.core_cf_ref` box the (unchanged) Hessian callback reads, exactly the
existing `inner_loop_internal_cmlookup_production` convention.
"""
function build_reduced_cm_operator_bundle(ctx, θ_full::AbstractVector{Float64}, layout::ProfiledEconomicMomentLayout,
        cctx::CMBinHessCtx; ref_obj = ctx.obj, method::Symbol = :suffix)
    n = 1 + layout.total_reduced_economic_moments + cctx.ncm   # zeta + econ + CM-grid, no gravity; matches Hfull's own (NCORE+ncm) sizing exactly
    obj = OperatorPsiBundle(
        δ = ref_obj.δ, find_smallest = ref_obj.find_smallest, γ = ref_obj.γ, l = ref_obj.l,
        inequality_index = Int[], U = ref_obj.U, outer_constr_index = n,
        inner_loop_opt = ref_obj.inner_loop_opt, Psi! = ref_obj.Psi!, dPsi! = ref_obj.dPsi!, ddPsi! = ref_obj.ddPsi!,
        lower_limit = ref_obj.lower_limit,
    )
    bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
    st = ReducedCMLookupState(obj, ctx, layout, collect(Float64, θ_full), cctx.ncm, cctx.L, cctx.origins, cctx.refIndex1,
        bins_u, cctx.R; method = method, core_cf_ref = cctx.core_cf_ref)
    return obj, st
end

"""
    inner_loop_KNITRO_reduced_cmlookup(obj, st, cctx) -> (nStatus, objSol, x, lambda_, n_fg, n_hess)

FG callback: `_callbackEvalFG_inner_cmlookup!` (cm_lookup_live_knitro.jl), REUSED UNCHANGED -- it is
already type-agnostic in `st` (only needs `st(x,g)` callable + `st.obj.lower_limit`).
Hessian callback: `archC_hess_cb_builder(cctx)` (cm_hessian_architectures.jl), REUSED UNCHANGED,
adapted via the SAME `_adapt_hess_cb_for_lookup` (cm_lookup_production.jl) the existing dense-width
production `:cm_lookup` path already uses to unwrap `st.obj` before forwarding.
"""
function inner_loop_KNITRO_reduced_cmlookup(obj::OperatorPsiBundle, st::ReducedCMLookupState, cctx::CMBinHessCtx)
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
    reduced_cm_base_state(x_free0, ctx, layout, cctx; method=:suffix) -> (obj, st, nStatus, zetastar, lambdastar, n_fg, n_hess)

Faithful reduced-operator mirror of `archC_base_state`: builds the bundle fresh (cheap; only the
KNITRO variable/bound setup, no per-draw work), primes via the UNCHANGED `prime_operator!`
(economic-only theta slice, gravity, `cf`/`core_cf_ref` publish -- zero dense-G/CM-column fill),
then solves.
"""
function reduced_cm_base_state(x_free0::AbstractVector, ctx, layout::ProfiledEconomicMomentLayout, cctx::CMBinHessCtx;
        method::Symbol = :suffix)
    θ_full0 = CS.reconstruct_full(x_free0, ctx.m)
    obj, st = build_reduced_cm_operator_bundle(ctx, θ_full0, layout, cctx; method = method)
    prime_operator!(obj, θ_full0, ctx, cctx.core_cf_ref; restriction_state = cctx)
    cctx.profiled_theta_ref[] = copy(θ_full0)
    cctx.cmlookup_st = st
    cctx.inner_fg_backend = :cm_lookup
    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_reduced_cmlookup(obj, st, cctx)
    nStatus ∈ (0, -100, -101, -103) || throw(CMExpectedSolveFailure("reduced_cm_base_state: inner solve failed, nStatus=$nStatus (x_free0=$x_free0)"))
    ζstar = x[1]; λstar = collect(x[2:end])
    return (obj = obj, st = st, inner_status = nStatus, ζstar = ζstar, λstar = λstar, n_fg = n_fg, n_hess = n_hess)
end
