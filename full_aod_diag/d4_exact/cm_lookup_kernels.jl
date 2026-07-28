# ================================================================================================
# Part B: lookup-based evaluation of the common-marginals contribution to the CC inner dual
# objective/gradient (w.r.t. (ζ,λ) -- the KNITRO inner-solve variables), using the bin indices
# from common_marginals_interval.jl. NEW file; does not modify cc_algo/PsiObjectiveBundle.jl or
# any dense reference file.
#
# BACKGROUND (see cc_algo/PsiObjectiveBundle.jl's PsiObjectiveBundleImplicit callable, the
# production code this replicates the math of): for this ctx (outer_constr_index == d, CM block
# spliced in as INNER moments before the sole outer-only gravity column), the KNITRO inner-solve
# variable vector is `x = [ζ; λ_core (ncore-1 entries); λ_cm (ncm entries)]`, and per FG callback
# call (many times per inner solve):
#   arg0 = -(ζ .+ G_core*λ_core .+ G_cm*λ_cm)              # length-W vector
#   arg1 = Psi!/dPsi!(arg0)
#   f    = sum(arg1)/M + ζ
#   g[1] = 1 - sum(arg1)/M
#   g[core part] = -mean(arg1 .* G_core, dims=1)           # BLAS gemv, O(W*(ncore-1))
#   g[cm part]   = -mean(arg1 .* G_cm,   dims=1)           # BLAS gemv in the dense baseline,
#                                                           # O(W*ncm) = O(W*(D-1)*L)
# `G_cm*λ_cm` (forward) and `mean(arg1.*G_cm)` (backward) are the two operations this file
# replaces with O(W*(D-1)) bin-lookup kernels (independent of L), using the identities proved in
# common_marginals_interval.jl's module docstring:
#   forward (interval basis):  sum_j λ_cm[j]*G_cm[s,j] = sum_oi (λblock[b_{s,o(oi)},oi] - λblock[b_{s,1},oi])
#   backward (interval basis): g_cm[k,oi] = -(1/M)*(h[o(oi),k] - h[refIndex1,k]),  h = weighted histogram of arg1 by bin
# and the cumulative-basis suffix-sum ANALOGS (task Part B.3, diagnostic/equivalence-check only,
# not a competing production candidate):
#   forward (cumulative basis): sum_l nu[l,oi]*1{U_o<=z_l} = P_oi(b_o),  P_oi(k) = sum_{l>=k} nu[l,oi]
#   backward (cumulative basis): g[l,oi] = -(1/M)*(Hpre[o(oi),l] - Hpre[refIndex1,l]),
#                                 Hpre = PREFIX (not suffix) cumulative sum of the histogram h
# (`R` handling: for :orthonormal contrasts, the stored λ/gradient are related to the raw
# "block"-space λ/gradient by `stored = block * R`, `block = stored * R` (R symmetric, `R=R'`,
# see common_marginals_moments.jl::orthonormal_contrast_matrix) -- both forward and backward
# lookups therefore pre/post-multiply the (L x nO) coefficient matrix by R once per FG call, an
# O(L*nO^2) operation, cheap relative to L*W or W.)
# ================================================================================================

using Base.Threads: nthreads as _nthreads, @threads
using LinearAlgebra: mul!

# port/finish-operator-stack-no-dense-G-and-CM-basis-diagnosis-2026-07-26 Phase A item 3: guarantee
# `CompressedFactual`/`economic_forward!`/`economic_transpose!`/`record_dense_economic_G!` are
# defined before CMLookupState's callable (below) can ever reference them at runtime -- this file
# is included directly (without cm_hessian_architectures.jl/cm_production_bundle.jl first) by 50+
# ad hoc scripts across the repo, so these guards must live HERE, not be assumed from caller order.
# Same `isdefined(Main, :CompressedFactual) || include(...)` idiom cm_hessian_architectures.jl
# already uses successfully across those same call sites.
isdefined(Main, :CompressedFactual) || include(joinpath(@__DIR__, "compressed_moments.jl"))
isdefined(Main, :economic_forward!) || include(joinpath(@__DIR__, "economic_operator.jl"))
isdefined(Main, :HessianWeightCache) || include(joinpath(@__DIR__, "operator_hessian_weights.jl"))

# ---- weighted histogram (Part B.2), threaded over draw chunks, thread-local buffers, no atomics ----

"""
    build_weighted_histogram(bins, weights, D, nbins; nthreads_use=1) -> Matrix{Float64}(D, nbins)

`h[o,k] = sum_{s : bins[s,o]==k} weights[s]`, for every origin `o=1:D` (including the reference
origin) and bin `k=1:nbins` (`nbins = L+1`). Threaded over draw-index CHUNKS (contiguous blocks of
rows, one per thread), each thread accumulating into its OWN preallocated `(D, nbins)` buffer
(zero risk of races, no atomics), then reduced by summing the per-thread buffers in a FIXED
(thread-index) order -- deterministic across repeated runs at the same `nthreads_use` (not
bit-identical to a serial run, since chunked partial sums associate additions differently; that is
expected of any parallel reduction and is noted in the report, not hidden).
"""
function build_weighted_histogram(bins::AbstractMatrix{<:Unsigned}, weights::AbstractVector{Float64},
                                   D::Int, nbins::Int; nthreads_use::Int = 1)
    W = length(weights)
    nt = max(1, min(nthreads_use, W))
    partials = [zeros(Float64, D, nbins) for _ in 1:nt]
    chunk = cld(W, nt)
    @threads for t in 1:nt
        lo = (t - 1) * chunk + 1
        hi = min(t * chunk, W)
        buf = partials[t]
        @inbounds for s in lo:hi
            for o in 1:D
                k = Int(bins[s, o])
                buf[o, k] += weights[s]
            end
        end
    end
    h = zeros(Float64, D, nbins)
    @inbounds for t in 1:nt
        h .+= partials[t]
    end
    return h
end

# ---- interval-basis lookup kernels ----
#
# ORIENTATION NOTE: the stored parameter vector's threshold-major layout is
# `col(k,oi) = (k-1)*nO + oi` (oi/origin varies FASTEST within a threshold/bin block). Julia
# reshapes a length-(nO*L) vector into an `(nO, L)` matrix in COLUMN-MAJOR order with linear
# index `(k-1)*nO+oi` for element `[oi,k]` -- i.e. `reshape(v, nO, L)` reproduces this layout
# EXACTLY (row=origin, column=bin/threshold). All lookup matrices below therefore use `(nO, ·)`
# orientation (origin-major-row), NOT `(·, nO)` -- an earlier draft of this file used `(L, nO)`
# and required a reshape that silently transposed origin<->bin; caught by a direct dense-vs-lookup
# forward/backward numerical comparison before this was wired into anything, fixed here.

"""
    apply_contrast(M, R)

`R*M` if `R !== nothing` (R the `nO x nO` symmetric orthonormal-contrast matrix,
`orthonormal_contrast_matrix`; `R' == R` so this same LEFT-multiply is the correct linear map in
BOTH directions -- forward (stored coefficients -> raw "block" coefficients) and backward (raw
block-space gradient -> stored-space gradient); derivation in the module docstring above), else
`M` unchanged (`:anchored`, `R === nothing`). `M` is `(nO, ·)`-oriented (origin = row).
"""
apply_contrast(M::AbstractMatrix{Float64}, R::Union{Nothing,AbstractMatrix{Float64}}) = R === nothing ? M : R * M

"""
    interval_forward_contribution!(out, bins, refIndex1, origins, λmat_ext)

`out[s] = sum_oi (λmat_ext[oi, bins[s,origins[oi]]] - λmat_ext[oi, bins[s,refIndex1]])`,
`λmat_ext` is `nO x (L+1)` with column `L+1` == 0 (the dropped bin). O(W*nO) -- no loop over L.
`out` is OVERWRITTEN (not accumulated).
"""
function interval_forward_contribution!(out::AbstractVector{Float64}, bins::AbstractMatrix{<:Unsigned},
                                         refIndex1::Int, origins::Vector{Int}, λmat_ext::AbstractMatrix{Float64})
    W = length(out); nO = length(origins)
    @inbounds for s in 1:W
        acc = 0.0
        bref = Int(bins[s, refIndex1])
        for oi in 1:nO
            bo = Int(bins[s, origins[oi]])
            acc += λmat_ext[oi, bo] - λmat_ext[oi, bref]
        end
        out[s] = acc
    end
    return out
end

"""
    interval_backward_gradient(h, refIndex1, origins, L, M) -> Matrix{Float64}(nO, L)

`g_block[oi,k] = -(h[origins[oi],k] - h[refIndex1,k]) / M`, `h` the `(D, L+1)` weighted histogram
from `build_weighted_histogram`. O(L*nO) given `h`.
"""
function interval_backward_gradient(h::AbstractMatrix{Float64}, refIndex1::Int, origins::Vector{Int}, L::Int, M::Int)
    nO = length(origins)
    g = Matrix{Float64}(undef, nO, L)
    @inbounds for k in 1:L, oi in 1:nO
        g[oi, k] = -(h[origins[oi], k] - h[refIndex1, k]) / M
    end
    return g
end

# ---- cumulative-basis suffix-sum kernels (Part B.3, DIAGNOSTIC / equivalence cross-check only) ----

"suffix_sums(nu) -> P (nO x (L+1)): P[:,k] = sum_{l=k}^{L} nu[:,l], P[:,L+1] = 0. `nu` is `nO x L`."
function suffix_sums(nu::AbstractMatrix{Float64})
    nO, L = size(nu)
    P = zeros(Float64, nO, L + 1)
    @inbounds for oi in 1:nO
        acc = 0.0
        for k in L:-1:1
            acc += nu[oi, k]
            P[oi, k] = acc
        end
        # P[oi, L+1] stays 0.0
    end
    return P
end

"cumulative_forward_contribution! -- same lookup pattern as interval_forward_contribution!, but with the SUFFIX-SUM-extended matrix P (P[:,L+1]=0 plays the same role as λmat_ext's dropped-bin column)."
cumulative_forward_contribution!(out, bins, refIndex1, origins, P) =
    interval_forward_contribution!(out, bins, refIndex1, origins, P)   # identical lookup shape; P already encodes the suffix sum

"prefix_sums(h, L) -> Hpre (D x L): Hpre[o,l] = sum_{k=1}^{l} h[o,k] (l=1:L; bin L+1 excluded, it's the dropped/uninformative one for the cumulative basis too since CDF thresholds only go up to z_L)."
function prefix_sums(h::AbstractMatrix{Float64}, L::Int)
    D = size(h, 1)
    Hpre = zeros(Float64, D, L)
    @inbounds for o in 1:D
        acc = 0.0
        for l in 1:L
            acc += h[o, l]
            Hpre[o, l] = acc
        end
    end
    return Hpre
end

"cumulative_backward_gradient(h, refIndex1, origins, L, M) -> Matrix{Float64}(nO, L): g[oi,l] = -(Hpre[o(oi),l]-Hpre[ref,l])/M, Hpre = PREFIX sum of h (not suffix -- see module docstring)."
function cumulative_backward_gradient(h::AbstractMatrix{Float64}, refIndex1::Int, origins::Vector{Int}, L::Int, M::Int)
    Hpre = prefix_sums(h, L)
    nO = length(origins)
    g = Matrix{Float64}(undef, nO, L)
    @inbounds for l in 1:L, oi in 1:nO
        g[oi, l] = -(Hpre[origins[oi], l] - Hpre[refIndex1, l]) / M
    end
    return g
end

# ================================================================================================
# Phase 5.5 remediation (2026-07-26): IN-PLACE (`!`-suffixed) analogues of the six allocating
# helpers above, writing into caller-supplied persistent buffers. The allocating originals above
# are UNCHANGED and kept -- they remain the public API `cm_meanzc_production.jl`/
# `cm_frechet_cplus.jl`/`lfix_cm_aware.jl`/`c12i_benchmark_lookup.jl` call directly (once per outer
# point, not hot). These `!` variants exist ONLY to remove the per-KNITRO-FG-callback allocations
# `CMLookupState`'s own callable below used to incur every one of the (many, per inner solve)
# forward/backward evaluations -- task §5.5's "no per-callback vectors, matrices, closures" ask.
# ================================================================================================

"In-place `apply_contrast`: `out .= R*M` (R!==nothing) or `out .= M` (R===nothing, :anchored). `out`/`M` both `(nO, ·)`."
function apply_contrast!(out::AbstractMatrix{Float64}, M::AbstractMatrix{Float64}, R::Union{Nothing,AbstractMatrix{Float64}})
    if R === nothing
        out .= M
    else
        mul!(out, R, M)
    end
    return out
end

"In-place `suffix_sums`: writes into `P` (nO x (L+1)), `P[:,L+1]` left/forced to 0.0. `nu` is `nO x L`."
function suffix_sums!(P::AbstractMatrix{Float64}, nu::AbstractMatrix{Float64})
    nO, L = size(nu)
    @inbounds for oi in 1:nO
        acc = 0.0
        for k in L:-1:1
            acc += nu[oi, k]
            P[oi, k] = acc
        end
        P[oi, L + 1] = 0.0
    end
    return P
end

"""
    build_weighted_histogram!(h, partials, bins, weights, D, nbins) -> h

In-place analogue of `build_weighted_histogram`: `partials` is a `Vector` of `nt` preallocated
`(D, nbins)` thread-local buffers (`nt = length(partials)`, fixed at `CMLookupState` construction
time -- sized from the live worker policy, see `cm_lookup_production.jl`), `h` the `(D, nbins)`
output buffer. Same draw-chunk/no-atomics/deterministic-fixed-order-reduction design as the
allocating original -- see that function's docstring for the full rationale.
"""
function build_weighted_histogram!(h::Matrix{Float64}, partials::Vector{Matrix{Float64}},
                                    bins::AbstractMatrix{<:Unsigned}, weights::AbstractVector{Float64},
                                    D::Int, nbins::Int)
    W = length(weights)
    nt = length(partials)
    chunk = cld(W, nt)
    @threads for t in 1:nt
        lo = (t - 1) * chunk + 1
        hi = min(t * chunk, W)
        buf = partials[t]
        fill!(buf, 0.0)
        @inbounds for s in lo:hi
            for o in 1:D
                k = Int(bins[s, o])
                buf[o, k] += weights[s]
            end
        end
    end
    fill!(h, 0.0)
    @inbounds for t in 1:nt
        h .+= partials[t]
    end
    return h
end

"In-place `prefix_sums`: writes into `Hpre` (D x L). `h` is `(D, nbins)` with `nbins >= L+1`."
function prefix_sums!(Hpre::AbstractMatrix{Float64}, h::AbstractMatrix{Float64}, L::Int)
    D = size(h, 1)
    @inbounds for o in 1:D
        acc = 0.0
        for l in 1:L
            acc += h[o, l]
            Hpre[o, l] = acc
        end
    end
    return Hpre
end

"In-place `interval_backward_gradient`: writes into `g` (nO x L)."
function interval_backward_gradient!(g::AbstractMatrix{Float64}, h::AbstractMatrix{Float64},
                                      refIndex1::Int, origins::Vector{Int}, L::Int, M::Int)
    nO = length(origins)
    @inbounds for k in 1:L, oi in 1:nO
        g[oi, k] = -(h[origins[oi], k] - h[refIndex1, k]) / M
    end
    return g
end

"In-place `cumulative_backward_gradient`: writes into `g` (nO x L), using preallocated `Hpre` (D x L) scratch."
function cumulative_backward_gradient!(g::AbstractMatrix{Float64}, Hpre::AbstractMatrix{Float64}, h::AbstractMatrix{Float64},
                                        refIndex1::Int, origins::Vector{Int}, L::Int, M::Int)
    prefix_sums!(Hpre, h, L)
    nO = length(origins)
    @inbounds for l in 1:L, oi in 1:nO
        g[oi, l] = -(Hpre[origins[oi], l] - Hpre[refIndex1, l]) / M
    end
    return g
end

# ================================================================================================
# Unified FG evaluator state + callable: replicates PsiObjectiveBundleImplicit's (ζ,λ)-gradient
# branch EXACTLY (core columns via the SAME BLAS calls the production callable uses, on the SAME
# dense obj.H buffer; CM columns via the lookup kernels above), for a fixed θ (one inner solve).
# ================================================================================================

"""
    CMLookupState

Per-inner-solve mutable bundle. `obj` is the DENSE CM-augmented `PsiObjectiveBundleImplicit`
(needed for its core-column H buffer, `M`, `Psi!`/`dPsi!`, and to serve the Hessian callback
unchanged); `bins` from `common_marginals_interval.jl`; `method` in `(:interval, :suffix)`.
`R` is the `nO x nO` orthonormal contrast matrix (or `nothing` for `:anchored`).
"""
mutable struct CMLookupState
    obj::Any
    ncore::Int          # obj0.d BEFORE augmentation (core inner cols are 1:ncore-1, CM at ncore:ncore+ncm-1)
    ncm::Int
    L::Int
    nO::Int
    origins::Vector{Int}
    refIndex1::Int
    bins::Matrix{<:Unsigned}
    R::Union{Nothing,Matrix{Float64}}
    method::Symbol
    nbins::Int
    nthreads_use::Int
    arg0::Vector{Float64}
    arg1::Vector{Float64}
    cm_contrib::Vector{Float64}
    λmat_ext::Matrix{Float64}   # (nO, L+1) working buffer for forward pass -- see orientation note above
    n_fg_calls::Int
    # Phase 5.5 remediation (2026-07-26): persistent scratch eliminating every per-FG-callback
    # allocation the original implementation incurred (`vcat`, `apply_contrast`/`suffix_sums`'s own
    # fresh matrix, `build_weighted_histogram`'s `partials`+`h`, `interval_backward_gradient`'s
    # fresh (nO,L) matrix) -- all now write into these buffers instead, allocated ONCE here.
    xsub::Vector{Float64}          # length ncore-1: [ζ; λ_core], replaces per-call `vcat`
    λmat_block::Matrix{Float64}    # (nO, L): R-congruence-applied stored coefficients
    hist_partials::Vector{Matrix{Float64}}   # nt x (D, nbins) thread-local histogram buffers
    hist_h::Matrix{Float64}        # (D, nbins) reduced weighted histogram
    Hpre::Matrix{Float64}          # (D, L) prefix-sum scratch (:suffix/cumulative method only)
    g_block::Matrix{Float64}       # (nO, L) raw block-space backward gradient
    g_stored::Matrix{Float64}      # (nO, L) R-congruence-applied (stored-space) backward gradient
    # port/finish-operator-stack-no-dense-G-and-CM-basis-diagnosis-2026-07-26, Phase A item 3:
    # retrofit the core-column block to the SHARED economic_forward!/economic_transpose!
    # (economic_operator.jl) instead of the dense `obj.H` BLAS.gemv! above, mirroring
    # OriginZCOperatorState's/CMMeanZCOperatorState's identical `core_cf_ref`-driven pattern
    # exactly. `core_cf_ref` defaults to `Ref{Any}(nothing)` (never a `CompressedFactual`) for
    # EVERY pre-existing call site of this constructor (50+ ad hoc scripts across the repo,
    # `c12i_*`/`c13_*`/`c14_*`/etc.) so they take the dense fallback branch unconditionally --
    # byte-identical to pre-port behavior. Only `cm_lookup_production.jl`'s production wiring
    # passes the real `cctx.core_cf_ref`. `econ_ws`/`econ_ws_for` are typed `Any` (not
    # `Union{Nothing,EconomicFGWorkspace}`) purely to avoid a forward type reference --
    # `economic_operator.jl` is not necessarily included yet at every one of this struct's many
    # call sites, same idiom `cctx.cmlookup_st::Any` already uses one file over for the same reason.
    core_cf_ref::Ref{Any}
    econ_ws::Any
    econ_ws_for::Any
    econ_buf::Vector{Float64}
    n_dense_econ_fallback::Int
    # No-moments/no-composite-G task (2026-07-28): same-point cache for the Hessian-weight prep
    # (operator_hessian_weights.jl) -- see that file's own docstring for the full contract.
    hw_cache::HessianWeightCache
end

function CMLookupState(obj, ncore::Int, ncm::Int, L::Int, origins::Vector{Int}, refIndex1::Int,
                        bins::Matrix{<:Unsigned}, R; method::Symbol = :interval, nthreads_use::Int = 1,
                        core_cf_ref::Ref{Any} = Ref{Any}(nothing))
    method in (:interval, :suffix) || error("CMLookupState: method must be :interval or :suffix, got $method")
    nO = length(origins)
    M = size(obj.U, 1)
    ncore1 = ncore - 1
    W = size(bins, 1)
    nbins = L + 1
    D = size(bins, 2)
    nt = max(1, min(nthreads_use, W))
    hist_partials = [zeros(D, nbins) for _ in 1:nt]
    CMLookupState(obj, ncore, ncm, L, nO, origins, refIndex1, bins, R, method, nbins, nthreads_use,
                  zeros(M), zeros(M), zeros(M), zeros(nO, L + 1), 0,
                  zeros(1 + ncore1), zeros(nO, L), hist_partials, zeros(D, nbins), zeros(D, L),
                  zeros(nO, L), zeros(nO, L),
                  core_cf_ref, nothing, nothing, zeros(M), 0,
                  HessianWeightCache(1 + ncore1 + ncm))
end

"""
    dual_index!(st::CMLookupState, x) -> st.arg0

Computes `st.arg0 = r = -ζ·1 - E·λ_core - cm_contribution` in place -- extracted VERBATIM (no
mathematics changed) from this state's own FG functor, so the FG callback and the Hessian-weight
prep (`operator_hessian_weights.jl::operator_prep_for_hessian!`) call the exact same code path.
"""
function dual_index!(st::CMLookupState, x::AbstractVector{Float64})
    obj = st.obj
    ncore1 = st.ncore - 1

    ζ = x[1]
    λ_core = @view x[2:1+ncore1]
    λ_cm = @view x[2+ncore1:1+ncore1+st.ncm]

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
    if st.method == :interval
        st.λmat_ext[:, 1:st.L] .= st.λmat_block
        st.λmat_ext[:, st.L+1] .= 0.0
        interval_forward_contribution!(st.cm_contrib, st.bins, st.refIndex1, st.origins, st.λmat_ext)
    else # :suffix (cumulative basis diagnostic)
        suffix_sums!(st.λmat_ext, st.λmat_block)
        cumulative_forward_contribution!(st.cm_contrib, st.bins, st.refIndex1, st.origins, st.λmat_ext)
    end
    st.arg0 .-= st.cm_contrib
    return st.arg0
end

"""
    (st::CMLookupState)(x, g=Float64[]) -> f

FG evaluator with the SAME signature/semantics as `obj(x, g)` (no `θ`/`constr` support -- not
needed for the hot KNITRO inner-iteration loop, see module docstring). `x = [ζ; λ_core; λ_cm]`.
"""
function (st::CMLookupState)(x::AbstractVector{Float64}, g::AbstractVector{Float64} = Float64[])
    obj = st.obj
    M = size(obj.U, 1)
    ncore1 = st.ncore - 1   # number of core (non-CM, non-gravity) inner columns

    ζ = x[1]

    # No-moments/no-composite-G task (2026-07-28): forward computation extracted into the shared
    # `dual_index!(st, x)` (this file, above) -- now called from BOTH this FG functor and the
    # Hessian-weight prep (operator_hessian_weights.jl::operator_prep_for_hessian!), so the two are
    # provably running the identical code path rather than two independently-maintained copies.
    # `cf` is re-read here (not returned by `dual_index!`) only to select the economic-transpose
    # branch below -- identical value `dual_index!` itself just used internally.
    cf = st.core_cf_ref[]
    dual_index!(st, x)

    obj.Psi!(st.arg1, st.arg0)
    f = sum(st.arg1) / M + ζ

    if length(g) > 0
        obj.dPsi!(st.arg1, st.arg0)
        g[1] = 1.0 - sum(st.arg1) / M
        if cf isa CompressedFactual
            g_E = @view g[2:1+ncore1]
            economic_transpose!(g_E, st.arg1, cf, st.econ_ws)
            g_E .*= -(1.0 / M)   # economic_transpose! returns the raw scatter, caller applies -(1/M) per its own docstring
        else
            @views BLAS.gemv!('T', -1.0 / M, obj.H[:, 3:2+ncore1], st.arg1, 0.0, g[2:1+ncore1])
        end

        build_weighted_histogram!(st.hist_h, st.hist_partials, st.bins, st.arg1, size(st.bins, 2), st.nbins)
        if st.method == :interval
            interval_backward_gradient!(st.g_block, st.hist_h, st.refIndex1, st.origins, st.L, M)
        else
            cumulative_backward_gradient!(st.g_block, st.Hpre, st.hist_h, st.refIndex1, st.origins, st.L, M)
        end
        apply_contrast!(st.g_stored, st.g_block, st.R)
        @views g[2+ncore1:1+ncore1+st.ncm] .= vec(st.g_stored)
    end

    obj.arg0 .= st.arg0   # keep obj in sync for a subsequent dense Hessian callback, same trick as compressed_live.jl
    _publish_dual_index_cache!(st, x)   # let a same-point Hessian call reuse this r instead of recomputing
    st.n_fg_calls += 1
    return f
end
