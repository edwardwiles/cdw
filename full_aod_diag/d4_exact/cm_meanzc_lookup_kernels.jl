# ================================================================================================
# port/shared-inner-fg-operator-and-verification-2026-07-26, Job 1 §3 (CM+ZC, `G=[E|C|Z]`, code
# column order `[E|Z|C]` -- see cm_meanzc_moments.jl's own header, "Mean/pair columns sit BEFORE
# the CM-grid block"). CM+ZC had NO lookup/compressed FG alternative at all before this branch
# (RESTRICTED_OPERATOR_FG_PRODUCTION_PORT_2026-07-26.md §5: "CM+ZC... its entire inner FG callback
# is one dense BLAS.gemv! against the full widened obj.H").
#
# Composition, reusing THREE already-validated pieces, no new algebra:
#   - E: shared economic_forward!/economic_transpose! (economic_operator.jl), against the SAME
#     core_cf_ref[] CompressedFactual CM+ZC's own moments! closure already publishes for the shared
#     H_EE Hessian backend.
#   - Z: restriction_forward!/restriction_transpose! (zc_restriction_operator.jl) -- IDENTICAL code
#     to origin-ZC's, just constructed with a `SharedByPowerLayout(K_mean,K_pair)` instead of
#     `OriginByPowerLayout` (CM+ZC's own mean_columns_direct!/pair_columns! use a SCALAR ν_k per
#     level -- `mean_targets`/`pair_targets` under SharedByPowerLayout reduce to exactly that
#     scalar broadcast to every origin/pair, confirmed by inspection of `target_index`'s
#     `SharedByPowerLayout` method, not re-derived).
#   - C: the CM-grid bin-lookup kernels (`apply_contrast!`/`suffix_sums!`/`build_weighted_histogram!`/
#     `cumulative_backward_gradient!`, cm_lookup_kernels.jl) -- REUSED VERBATIM, same functions
#     `CMLookupState` already calls, not re-copied.
#
# `x = [ζ; λ_E(pregrav); λ_mean(n_mean); λ_pair(n_pair); λ_cm(ncm)]`, matching
# `wrap_moments_with_cm_meanzc`'s own column layout `[economic|mean|pair|CM-grid|gravity]` exactly
# (gravity, the last column, is outer-only -- same convention as origin-ZC, confirmed by this
# file's own D=4 correctness gate, not asserted from reading alone).
# ================================================================================================

isdefined(Main, :economic_forward!) || include(joinpath(@__DIR__, "economic_operator.jl"))
isdefined(Main, :ZCRestrictionOperator) || include(joinpath(@__DIR__, "zc_restriction_operator.jl"))

"""
    CMMeanZCOperatorState

Per-inner-solve mutable bundle: economic block via `core_cf_ref`, Z block via `ZCRestrictionOperator`
+ `SharedByPowerLayout`, CM-grid block via the SAME bin-lookup scratch pattern `CMLookupState` uses
(method=:suffix, matching `cm_lookup_kernels.jl`'s own production-convention comment).
"""
mutable struct CMMeanZCOperatorState
    obj::Any
    ncore1::Int                      # pregrav = ncore_econ - 1 (economic lambda length)
    zc_op::ZCRestrictionOperator
    zc_layout::Any                   # SharedByPowerLayout(K_mean, K_pair)
    core_cf_ref::Ref{Any}
    econ_ws::Union{Nothing,EconomicFGWorkspace}
    econ_ws_for::Any
    zc_ws::ZCRestrictionWorkspace
    # ---- CM-grid block (mirrors CMLookupState's own fields) ----
    ncm::Int
    L::Int
    nO::Int
    origins::Vector{Int}
    refIndex1::Int
    bins::Matrix{<:Unsigned}
    R::Union{Nothing,Matrix{Float64}}
    nbins::Int
    nthreads_use::Int
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
    econ_buf::Vector{Float64}
    n_fg_calls::Int
    n_dense_econ_fallback::Int
end

function CMMeanZCOperatorState(obj, ncore1::Int, zc_op::ZCRestrictionOperator, zc_layout, core_cf_ref::Ref{Any},
        ncm::Int, L::Int, origins::Vector{Int}, refIndex1::Int, bins::Matrix{<:Unsigned}, R;
        nthreads_use::Int = Threads.nthreads())
    W_ = size(obj.U, 1)
    nO = length(origins)
    nbins = L + 1
    D_bins = size(bins, 2)
    nt = max(1, min(nthreads_use, W_))
    hist_partials = [zeros(D_bins, nbins) for _ in 1:nt]
    CMMeanZCOperatorState(obj, ncore1, zc_op, zc_layout, core_cf_ref, nothing, nothing, ZCRestrictionWorkspace(zc_op),
        ncm, L, nO, origins, refIndex1, bins, R, nbins, nthreads_use,
        zeros(nO, L + 1), zeros(W_), zeros(nO, L), hist_partials, zeros(D_bins, nbins), zeros(D_bins, L),
        zeros(nO, L), zeros(nO, L),
        zeros(W_), zeros(W_), zeros(W_), 0, 0)
end

"Call ONCE per inner solve: refresh Z-block targets for the current outer point's ν, reset counters."
function reset_for_solve!(st::CMMeanZCOperatorState, νs::AbstractVector{Float64})
    refresh_zc_targets!(st.zc_ws, st.zc_op, st.zc_layout, νs)
    st.n_fg_calls = 0
    return st
end

function (st::CMMeanZCOperatorState)(x::AbstractVector{Float64}, g::AbstractVector{Float64} = Float64[])
    obj = st.obj
    M = length(st.arg0)
    ncore1 = st.ncore1
    op = st.zc_op

    ζ = x[1]
    λ_E = @view x[2:1+ncore1]
    λ_mean = @view x[2+ncore1 : 1+ncore1+n_mean(op)]
    λ_pair = @view x[2+ncore1+n_mean(op) : 1+ncore1+n_mean(op)+n_pair(op)]
    λ_cm = @view x[2+ncore1+n_mean(op)+n_pair(op) : 1+ncore1+n_mean(op)+n_pair(op)+st.ncm]

    fill!(st.arg0, -ζ)

    cf = st.core_cf_ref[]
    if cf isa CompressedFactual
        if st.econ_ws === nothing || st.econ_ws_for !== cf
            st.econ_ws = economic_operator_workspace(cf)
            st.econ_ws_for = cf
        end
        economic_forward!(st.econ_buf, λ_E, cf, st.econ_ws)
        st.arg0 .-= st.econ_buf
    else
        st.n_dense_econ_fallback += 1
        xsub_ext = vcat(ζ, collect(λ_E))
        @views BLAS.gemv!('N', -1.0, obj.H[:, 2:2+ncore1], xsub_ext, 0.0, st.arg0)
    end

    restriction_forward!(st.arg0, λ_mean, λ_pair, op, st.zc_ws)

    # CM-grid block (reuses cm_lookup_kernels.jl's own suffix-sum forward, unchanged)
    λmat_stored = reshape(λ_cm, st.nO, st.L)
    apply_contrast!(st.λmat_block, λmat_stored, st.R)
    suffix_sums!(st.λmat_ext, st.λmat_block)
    cumulative_forward_contribution!(st.cm_contrib, st.bins, st.refIndex1, st.origins, st.λmat_ext)
    st.arg0 .-= st.cm_contrib

    obj.Psi!(st.arg1, st.arg0)
    f = sum(st.arg1) / M + ζ

    if length(g) > 0
        obj.dPsi!(st.arg1, st.arg0)
        g[1] = 1.0 - sum(st.arg1) / M
        if cf isa CompressedFactual
            g_E = @view g[2:1+ncore1]
            economic_transpose!(g_E, st.arg1, cf, st.econ_ws)
            g_E .*= -(1.0 / M)
        else
            @views BLAS.gemv!('T', -1.0 / M, obj.H[:, 3:2+ncore1], st.arg1, 0.0, g[2:1+ncore1])
        end
        restriction_transpose!((@view g[2+ncore1:1+ncore1+n_mean(op)]),
                                (@view g[2+ncore1+n_mean(op):1+ncore1+n_mean(op)+n_pair(op)]),
                                st.arg1, op, st.zc_ws)

        build_weighted_histogram!(st.hist_h, st.hist_partials, st.bins, st.arg1, size(st.bins, 2), st.nbins)
        cumulative_backward_gradient!(st.g_block, st.Hpre, st.hist_h, st.refIndex1, st.origins, st.L, M)
        apply_contrast!(st.g_stored, st.g_block, st.R)
        cm_off = 1 + ncore1 + n_mean(op) + n_pair(op)
        @views g[cm_off+1:cm_off+st.ncm] .= vec(st.g_stored)
    end

    obj.arg0 .= st.arg0
    st.n_fg_calls += 1
    return f
end
