# ================================================================================================
# Common-Frechet matrix-free inner FG operator -- Phase 5.2 remediation (2026-07-26).
#
# Extends the validated flexible-CM lookup approach (cm_lookup_kernels.jl, CMLookupState) with the
# common-level-anchor block that turns flexible CM into fixed Frechet, per
# docs/FRECHET_AS_CM_PLUS_LEVEL_MATHEMATICS_2026-07-25.md and the existing dense-reference
# construction this file's forward/backward math is verified against (cm_frechet_level.jl's
# `fill_frechet_level_columns_from_bins!`, the moments!-time dense fill this kernel replaces).
#
# Inner-solve variable layout: `x = [zeta; lambda_core (ncore-1); lambda_cm ((D-1)*L); lambda_level (L)]`
# -- see wrap_moments_with_cm_frechet_archB's own column-layout docstring (cm_frechet_level.jl),
# `[core | CM | level | gravity]`.
#
# CM block: IDENTICAL math to plain flexible CM (reuses cm_lookup_kernels.jl's free functions
# unchanged: apply_contrast!, suffix_sums!, cumulative_forward_contribution!, build_weighted_
# histogram!, prefix_sums!). Common Frechet's own moment construction (wrap_moments_with_cm_
# frechet_archB) always stores the CM block in the CUMULATIVE basis (same fill_cm_columns_from_
# bins! call flexible CM's own production path uses) -- so, unlike CMLookupState, this file does
# not support an :interval method at all; the CM block always uses the suffix-sum/cumulative
# lookup.
#
# Level block (genuinely new -- no existing production inner-FG kernel to reuse):
#   forward:  sum_l lambda_level[l]*G_level[s,l]
#           = invsqrtD * sum_o P_level[bin(s,o)] - sum_l lambda_level[l]*targets[l]
#     where P_level[k] = sum_{l>=k} lambda_level[l] (suffix sum, same convention as the CM block's
#     P), invsqrtD = 1/sqrt(D), and `sum_o P_level[bin(s,o)]` is computed by the EXISTING
#     `frechet_level_forward_sum!` (cm_frechet_cplus.jl, already used unchanged by the Lfix outer-
#     gradient path at the converged lambda* -- reused here unchanged for the LIVE inner-solve
#     lambda, same underlying identity).
#   backward: g_level[l] = -(invsqrtD/M)*Hpre_total[l] + (targets[l]/M)*sum_dPsi
#     where Hpre_total[l] = sum_{o=1}^D Hpre[o,l], Hpre = the SAME (D,L) prefix-sum-of-weighted-
#     histogram buffer the CM block's own cumulative backward gradient already computes (shared,
#     computed ONCE per callback, not duplicated) via `prefix_sums!(Hpre, h, L)`, and
#     sum_dPsi = sum_s dPsi(arg0)[s] (a free byproduct of the core block's own g[1] computation).
# Full derivation: see docs/RESTRICTED_OPERATOR_FG_PRODUCTION_PORT_2026-07-26.md (this session).
# ================================================================================================

isdefined(Main, :CompressedFactual) || include(joinpath(@__DIR__, "compressed_moments.jl"))
isdefined(Main, :economic_forward!) || include(joinpath(@__DIR__, "economic_operator.jl"))

"""
    frechet_level_suffix_sums!(P, λ_level)

`P[k] = sum_{l=k}^{L} λ_level[l]` for `k=1:L`, `P[L+1]=0` -- the level block's own (un-origin-
indexed) analogue of `suffix_sums!`. `λ_level` is length `L`, `P` is length `L+1`.
"""
function frechet_level_suffix_sums!(P::AbstractVector{Float64}, λ_level::AbstractVector{Float64})
    L = length(λ_level)
    acc = 0.0
    @inbounds for k in L:-1:1
        acc += λ_level[k]
        P[k] = acc
    end
    P[L + 1] = 0.0
    return P
end

"""
    cumulative_backward_gradient_from_prefix!(g, Hpre, refIndex1, origins, L, M)

Identical formula to `cumulative_backward_gradient!` (cm_lookup_kernels.jl), but takes an
ALREADY-COMPUTED `Hpre` (D x L prefix-sum-of-histogram) instead of building its own from `h` --
lets the CM and level backward passes share ONE `prefix_sums!` call per callback instead of two.
"""
function cumulative_backward_gradient_from_prefix!(g::AbstractMatrix{Float64}, Hpre::AbstractMatrix{Float64},
                                                     refIndex1::Int, origins::Vector{Int}, L::Int, M::Int)
    nO = length(origins)
    @inbounds for l in 1:L, oi in 1:nO
        g[oi, l] = -(Hpre[origins[oi], l] - Hpre[refIndex1, l]) / M
    end
    return g
end

"""
    frechet_level_backward_gradient!(g_level, Hpre, D, L, M, invsqrtD, targets, sum_dPsi)

`g_level[l] = -(invsqrtD/M)*sum_{o=1}^D Hpre[o,l] + (targets[l]/M)*sum_dPsi`. O(D*L), reusing the
SAME `Hpre` buffer the CM block's own backward gradient already computed this callback.
"""
function frechet_level_backward_gradient!(g_level::AbstractVector{Float64}, Hpre::AbstractMatrix{Float64},
                                           D::Int, L::Int, M::Int, invsqrtD::Float64,
                                           targets::Vector{Float64}, sum_dPsi::Float64)
    @inbounds for l in 1:L
        acc = 0.0
        for o in 1:D
            acc += Hpre[o, l]
        end
        g_level[l] = -(invsqrtD / M) * acc + (targets[l] / M) * sum_dPsi
    end
    return g_level
end

"""
    CMFrechetLookupState

Per-context mutable bundle, common-Frechet analogue of `CMLookupState`. `obj` is the DENSE
CM+level-augmented `PsiObjectiveBundleImplicit` (needed for its core-column H buffer, `M`,
`Psi!`/`dPsi!`, and to serve the Hessian callback unchanged -- Architecture C's Hessian,
`archC_frechet_hess_cb_builder`, is completely independent of this FG kernel). `R` is the `nO x nO`
orthonormal contrast matrix for the CM block (or `nothing` for `:anchored`) -- the level block is
NEVER rotated by `R` (it is a single un-rotated column per threshold, not part of the CM block's
per-threshold `nO`-dimensional rotation, per the math doc's `[C u]`/`[CR u]` construction).
"""
# port/finish-operator-stack-no-dense-G-and-CM-basis-diagnosis-2026-07-26 Phase A item 4: `obj::O`
# (was `obj::Any`) per docs/COMMON_FRECHET_OPERATOR_FG_FINAL_GATE_2026-07-26.md's own recommended
# follow-on ("(a) restructuring the callable so obj's concrete type is captured once at construction
# via a type parameter... rather than read fresh from an Any field on every property access") --
# the prior session measured a real, reproducible 14,066,064 bytes/callback regression here vs
# CMLookupState's near-identical `obj::Any`-fielded but textually SHORTER callable (2,384 bytes),
# suspected to be a devirtualization failure that scales with function-body size/branch count. `O`
# is inferred automatically from the `obj` argument at construction (Julia's default parametric-
# struct outer constructor) -- no forward-type-reference load-order dependency is introduced (this
# was the ONLY reason the field was `Any` in the first place; a type parameter has the same
# load-order-agnostic property, since `O` is resolved from the CONCRETE runtime object passed in,
# never a textual type name that needs `PsiObjectiveBundleImplicit` predeclared).
mutable struct CMFrechetLookupState{O}
    obj::O
    ncore::Int
    ncm_cm::Int         # (D-1)*L
    ncm_level::Int      # L
    L::Int
    D::Int
    nO::Int
    origins::Vector{Int}
    refIndex1::Int
    bins::Matrix{<:Unsigned}
    R::Union{Nothing,Matrix{Float64}}
    nbins::Int
    nthreads_use::Int
    level_targets::Vector{Float64}
    invsqrtD::Float64
    n_fg_calls::Int
    # persistent scratch (Phase 5.5 allocation-hygiene pattern, applied here from the start rather
    # than fixed as a follow-on -- see cm_lookup_kernels.jl's own history for why)
    arg0::Vector{Float64}
    arg1::Vector{Float64}
    cm_contrib::Vector{Float64}
    level_contrib::Vector{Float64}
    xsub::Vector{Float64}
    λmat_block::Matrix{Float64}    # (nO, L)
    λmat_ext::Matrix{Float64}      # (nO, L+1)
    P_level::Vector{Float64}       # (L+1,)
    hist_partials::Vector{Matrix{Float64}}
    hist_h::Matrix{Float64}        # (D, nbins)
    Hpre::Matrix{Float64}          # (D, L) -- SHARED between CM and level backward
    g_block::Matrix{Float64}       # (nO, L)
    g_stored::Matrix{Float64}      # (nO, L)
    g_level::Vector{Float64}       # (L,)
    # Phase A item 4 (second half): shared economic operator retrofit, IDENTICAL pattern/rationale
    # to CMLookupState's own core_cf_ref/econ_ws/econ_ws_for/econ_buf/n_dense_econ_fallback fields
    # -- see that struct's docstring for the full contract. `core_cf_ref` defaults to
    # `Ref{Any}(nothing)` for the standalone constructor (dense fallback, byte-identical to
    # pre-port); only `cm_frechet_lookup_production.jl`'s production wiring passes the real
    # `cctx.core_cf_ref` (populated by `wrap_moments_with_cm_frechet_archB`, same box the shared
    # winner-pair Hessian already reads).
    core_cf_ref::Ref{Any}
    econ_ws::Any
    econ_ws_for::Any
    econ_buf::Vector{Float64}
    n_dense_econ_fallback::Int
end

function CMFrechetLookupState(obj, ncore::Int, ncm_cm::Int, ncm_level::Int, L::Int, D::Int,
                               origins::Vector{Int}, refIndex1::Int, bins::Matrix{<:Unsigned},
                               R::Union{Nothing,Matrix{Float64}}, level_targets::Vector{Float64};
                               nthreads_use::Int = 1, core_cf_ref::Ref{Any} = Ref{Any}(nothing))
    ncm_level == L || error("CMFrechetLookupState: ncm_level=$ncm_level must equal L=$L")
    length(level_targets) == L || error("CMFrechetLookupState: length(level_targets)=$(length(level_targets)) != L=$L")
    nO = length(origins)
    M = size(obj.U, 1)
    ncore1 = ncore - 1
    W = size(bins, 1)
    nbins = L + 1
    Dcheck = size(bins, 2)
    Dcheck == D || error("CMFrechetLookupState: D=$D != size(bins,2)=$Dcheck")
    nt = max(1, min(nthreads_use, W))
    hist_partials = [zeros(D, nbins) for _ in 1:nt]
    CMFrechetLookupState(obj, ncore, ncm_cm, ncm_level, L, D, nO, origins, refIndex1, bins, R,
        nbins, nthreads_use, level_targets, 1.0 / sqrt(D), 0,
        zeros(M), zeros(M), zeros(M), zeros(M),
        zeros(1 + ncore1), zeros(nO, L), zeros(nO, L + 1), zeros(L + 1),
        hist_partials, zeros(D, nbins), zeros(D, L), zeros(nO, L), zeros(nO, L), zeros(L),
        core_cf_ref, nothing, nothing, zeros(M), 0)
end

"""
    (st::CMFrechetLookupState)(x, g=Float64[]) -> f

FG evaluator, same signature/semantics as `obj(x, g)`. `x = [ζ; λ_core; λ_cm; λ_level]`.
"""
function (st::CMFrechetLookupState)(x::AbstractVector{Float64}, g::AbstractVector{Float64} = Float64[])
    obj = st.obj
    M = size(obj.U, 1)
    ncore1 = st.ncore - 1

    ζ = x[1]
    λ_core = @view x[2:1+ncore1]
    λ_cm = @view x[2+ncore1:1+ncore1+st.ncm_cm]
    λ_level = @view x[2+ncore1+st.ncm_cm:1+ncore1+st.ncm_cm+st.ncm_level]

    # ---- forward: arg0 = -(ζ + G_core*λ_core + G_cm*λ_cm + G_level*λ_level) ----
    # Phase A item 4: shared economic_forward!/economic_transpose! when core_cf_ref[] holds a real
    # CompressedFactual, else the original dense obj.H BLAS.gemv! -- identical contract to
    # CMLookupState's own retrofit (cm_lookup_kernels.jl).
    cf = st.core_cf_ref[]
    if cf isa CompressedFactual
        if st.econ_ws === nothing || st.econ_ws_for !== cf
            st.econ_ws = economic_operator_workspace(cf)
            st.econ_ws_for = cf
        end
        economic_forward!(st.econ_buf, λ_core, cf, st.econ_ws)
        st.arg0 .= (-ζ) .- st.econ_buf
    else
        st.n_dense_econ_fallback += 1
        record_dense_economic_G!()
        st.xsub[1] = ζ
        st.xsub[2:end] .= λ_core
        @views BLAS.gemv!('N', -1.0, obj.H[:, 2:2+ncore1], st.xsub, 0.0, st.arg0)
    end

    λmat_stored = reshape(λ_cm, st.nO, st.L)
    apply_contrast!(st.λmat_block, λmat_stored, st.R)
    suffix_sums!(st.λmat_ext, st.λmat_block)
    cumulative_forward_contribution!(st.cm_contrib, st.bins, st.refIndex1, st.origins, st.λmat_ext)
    st.arg0 .-= st.cm_contrib

    frechet_level_suffix_sums!(st.P_level, λ_level)
    frechet_level_forward_sum!(st.level_contrib, st.bins, st.D, st.P_level)
    const_term = 0.0
    @inbounds for l in 1:st.L
        const_term += λ_level[l] * st.level_targets[l]
    end
    @inbounds for s in 1:M
        st.arg0[s] -= st.invsqrtD * st.level_contrib[s] - const_term
    end

    obj.Psi!(st.arg1, st.arg0)
    f = sum(st.arg1) / M + ζ

    if length(g) > 0
        obj.dPsi!(st.arg1, st.arg0)
        sum_dPsi = sum(st.arg1)
        g[1] = 1.0 - sum_dPsi / M
        if cf isa CompressedFactual
            g_E = @view g[2:1+ncore1]
            economic_transpose!(g_E, st.arg1, cf, st.econ_ws)
            g_E .*= -(1.0 / M)
        else
            @views BLAS.gemv!('T', -1.0 / M, obj.H[:, 3:2+ncore1], st.arg1, 0.0, g[2:1+ncore1])
        end

        build_weighted_histogram!(st.hist_h, st.hist_partials, st.bins, st.arg1, st.D, st.nbins)
        prefix_sums!(st.Hpre, st.hist_h, st.L)   # SHARED: CM and level backward both read this

        cumulative_backward_gradient_from_prefix!(st.g_block, st.Hpre, st.refIndex1, st.origins, st.L, M)
        apply_contrast!(st.g_stored, st.g_block, st.R)
        @views g[2+ncore1:1+ncore1+st.ncm_cm] .= vec(st.g_stored)

        frechet_level_backward_gradient!(st.g_level, st.Hpre, st.D, st.L, M, st.invsqrtD, st.level_targets, sum_dPsi)
        @views g[2+ncore1+st.ncm_cm:1+ncore1+st.ncm_cm+st.ncm_level] .= st.g_level
    end

    obj.arg0 .= st.arg0
    st.n_fg_calls += 1
    return f
end
