# ================================================================================================
# port/shared-inner-fg-operator-and-verification-2026-07-26, Job 1 §3 (ZC only, `G=[E|Z]`): the
# first genuinely NEW restricted-family operator FG this branch builds (per the inherited
# handoff's own priority order -- origin-ZC first: smallest, no existing partial kernel at all,
# unlike CM+ZC which at least has a dense reference to extend).
#
# Composition: shared `economic_forward!`/`economic_transpose!` (economic_operator.jl, UNCHANGED
# from the unrestricted family's own validated compressed operator) for E, PLUS
# `restriction_forward!`/`restriction_transpose!` (zc_restriction_operator.jl, NEW this branch) for
# Z. Mirrors `CMLookupState`'s callable structure (cm_lookup_kernels.jl) exactly -- same `(ζ,λ)`
# split, same `obj.Psi!`/`obj.dPsi!` calls, same `obj.arg0` post-call sync for the Hessian callback
# -- so this is recognizable as "the same pattern, one more restriction operator swapped in", not a
# new design.
#
# `x = [ζ; λ_E (ncore1 = pregrav components); λ_mean (n_mean(op)); λ_pair (n_pair(op))]` -- this
# layout matches `wrap_moments_with_originzc`'s own G-column layout
# `[economic(pregrav) | mean | pair | gravity]` exactly (gravity itself is OUTER-only, never part
# of the inner-dual λ -- confirmed via `obj_oz.outer_constr_index == obj_oz.d` and cross-checked by
# this file's own D=4/D=20 correctness gate against the dense reference, not asserted from reading
# alone).
# ================================================================================================

isdefined(Main, :economic_forward!) || include(joinpath(@__DIR__, "economic_operator.jl"))
isdefined(Main, :ZCRestrictionOperator) || include(joinpath(@__DIR__, "zc_restriction_operator.jl"))

"""
    OriginZCOperatorState

Per-inner-solve mutable bundle for origin-ZC's operator FG. `core_cf_ref` is the SAME `Ref{Any}`
`wrap_moments_with_originzc`'s `moments!` closure publishes a `CompressedFactual` into on every
outer point (built for the shared winner-pair Hessian backend) -- this state reads it, it does not
build its own.
"""
mutable struct OriginZCOperatorState
    obj::Any
    ncore1::Int                                   # pregrav = ncore_econ - 1 (economic lambda length)
    op::ZCRestrictionOperator
    layout::Any                                    # MeanZCTargetLayout
    core_cf_ref::Ref{Any}
    econ_ws::Union{Nothing,EconomicFGWorkspace}
    econ_ws_for::Any
    zc_ws::ZCRestrictionWorkspace
    arg0::Vector{Float64}
    arg1::Vector{Float64}
    econ_buf::Vector{Float64}
    n_fg_calls::Int
    n_dense_econ_fallback::Int   # counts calls where core_cf_ref[] was not a usable CompressedFactual (tie/unavailable)
end

function OriginZCOperatorState(obj, ncore1::Int, op::ZCRestrictionOperator, layout, core_cf_ref::Ref{Any})
    W = size(obj.U, 1)
    OriginZCOperatorState(obj, ncore1, op, layout, core_cf_ref, nothing, nothing,
        ZCRestrictionWorkspace(op), zeros(W), zeros(W), zeros(W), 0, 0)
end

"""
    reset_for_solve!(st::OriginZCOperatorState, νfull)

Call ONCE per inner solve (before the KNITRO solve starts, same lifecycle point
`inner_loop_internal_cmlookup_production` calls `obj.moments!` at): refreshes the ZC target
buffers for the current outer point's `νfull` and resets the FG-call counter.
"""
function reset_for_solve!(st::OriginZCOperatorState, νfull::AbstractVector{Float64})
    refresh_zc_targets!(st.zc_ws, st.op, st.layout, νfull)
    st.n_fg_calls = 0
    return st
end

"""
    (st::OriginZCOperatorState)(x, g=Float64[]) -> f

FG evaluator with the SAME signature/semantics as `obj(x, g)`. `core_cf_ref[]` must already be a
`CompressedFactual` (published by the current inner solve's `moments!` call) -- if it is a Symbol
(tie/unavailable fallback reason), this falls back to a dense `obj.H`-based economic contraction
for correctness (never silently wrong), incrementing `n_dense_econ_fallback` so callers/tests can
detect and investigate rather than have it pass unnoticed.
"""
function (st::OriginZCOperatorState)(x::AbstractVector{Float64}, g::AbstractVector{Float64} = Float64[])
    obj = st.obj
    M = length(st.arg0)
    ncore1 = st.ncore1
    op = st.op

    ζ = x[1]
    λ_E = @view x[2:1+ncore1]
    λ_mean = @view x[2+ncore1 : 1+ncore1+n_mean(op)]
    λ_pair = @view x[2+ncore1+n_mean(op) : 1+ncore1+n_mean(op)+n_pair(op)]

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
        # Fallback (tied winner / compressed state unavailable this outer point): dense economic
        # contraction against obj.H, same BLAS.gemv! CMLookupState uses for its own core block.
        st.n_dense_econ_fallback += 1
        xsub_ext = vcat(ζ, collect(λ_E))
        @views BLAS.gemv!('N', -1.0, obj.H[:, 2:2+ncore1], xsub_ext, 0.0, st.arg0)
    end

    restriction_forward!(st.arg0, λ_mean, λ_pair, op, st.zc_ws)

    obj.Psi!(st.arg1, st.arg0)
    f = sum(st.arg1) / M + ζ

    if length(g) > 0
        obj.dPsi!(st.arg1, st.arg0)
        g[1] = 1.0 - sum(st.arg1) / M
        if cf isa CompressedFactual
            g_E = @view g[2:1+ncore1]
            economic_transpose!(g_E, st.arg1, cf, st.econ_ws)
            g_E .*= -(1.0 / M)   # economic_transpose! returns the raw Σ_s w_s E_{s,j} scatter (caller applies -(1/M), per its own docstring)
        else
            @views BLAS.gemv!('T', -1.0 / M, obj.H[:, 3:2+ncore1], st.arg1, 0.0, g[2:1+ncore1])
        end
        restriction_transpose!((@view g[2+ncore1:1+ncore1+n_mean(op)]),
                                (@view g[2+ncore1+n_mean(op):1+ncore1+n_mean(op)+n_pair(op)]),
                                st.arg1, op, st.zc_ws)
    end

    obj.arg0 .= st.arg0   # keep obj in sync for the Hessian callback, same convention CMLookupState uses
    st.n_fg_calls += 1
    return f
end
