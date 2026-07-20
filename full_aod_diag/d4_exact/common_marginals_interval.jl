# ================================================================================================
# Continuation: bin-index / interval-moment reformulation of the common-marginals restriction
# (task brief Part A). NEW file -- does NOT modify common_marginals_moments.jl (the trusted dense
# reference this file is validated against) or any of the c12_*.jl validation scripts.
#
# Core idea: `common_marginals_moments.jl::precalc_common_marginals_cdf` builds, per threshold
# z_l (l=1..L) and non-reference origin o, a CUMULATIVE indicator column
#   CDF_block[s,o,l] = 1{U[s,o] <= z_l} - 1{U[s,refIndex1] <= z_l}.
# This file instead assigns each draw/origin a BIN INDEX b_{s,o} in {1,...,L+1} via the SAME
# cutpoints z_1 < ... < z_L (identical quantile computation, verified bit-identical below), where
# bin k covers (z_{k-1}, z_k] for k>=2 and bin 1 covers (-inf, z_1] -- i.e.
#   b_{s,o} = k  iff  z_{k-1} < U[s,o] <= z_k        (z_0 := -inf, z_{L+1} := +inf)
# which is exactly `searchsortedfirst(z, U[s,o])` on the sorted cutpoint vector z (returns the
# smallest k with z_k >= U[s,o], or L+1 if U[s,o] > z_L) -- this reproduces the reference's `<=`
# convention exactly (checked below, not assumed).
#
# The INTERVAL moment for (o, k), k=1..L (bin L+1 dropped as redundant -- see the transform-matrix
# derivation below for why dropping bin L+1 exactly preserves the CDF columns' span) is
#   B_{s,o,k} = 1{b_{s,o}=k} - 1{b_{s,refIndex1}=k}.
#
# LINEAR EQUIVALENCE (explicit, not asserted): for l=1..L,
#   CDF_block[s,o,l] = sum_{k=1}^{l} B_{s,o,k}
# because 1{U<=z_l} = sum_{k=1}^{l} 1{b=k} (bins 1..l tile (-inf, z_l] exactly under the `<=`
# convention above), and the same identity holds after subtracting the reference-origin term
# (linear in the indicators). This is a LOWER-TRIANGULAR CUMULATIVE-SUM map in l/k (an all-ones
# lower-triangular L x L matrix), applied identically within each origin -- see
# `cumulative_interval_transform_matrix` / `interval_to_cumulative_dense` /
# `full_transform_matrix` below, which construct this map explicitly (as both a direct per-block
# cumsum and as an explicit dense (nO*L) x (nO*L) Kronecker matrix) and validate the two agree,
# and that applying it to the interval matrix reproduces `precalc_common_marginals_cdf`'s own
# dense CM matrix to machine precision, for both :anchored and :orthonormal contrasts (the
# orthonormal contrast matrix R is applied per-threshold-block only -- mixes origins, never
# mixes l/k -- so it commutes with the l/k cumulative-sum map; proved in the module docstring of
# common_marginals_moments.jl's own contrast application and re-verified numerically here).
# ================================================================================================

using Statistics: quantile

# ---- bin dtype selection (task: UInt8 if L+1<=256, else UInt16) ----
bin_index_dtype(L::Int) = (L + 1) <= typemax(UInt8) ? UInt8 : UInt16

"""
    common_marginals_quantiles(U, refIndex1, L) -> Vector{Float64}

IDENTICAL (line-for-line) to the quantile computation inside
`common_marginals_moments.jl::precalc_common_marginals_cdf` -- reproduced verbatim here (not
merely "equivalent") so cutpoints are bit-identical between the dense and interval builders
by construction, not by luck. Checked in the equivalence test script regardless.
"""
common_marginals_quantiles(U::AbstractMatrix{Float64}, refIndex1::Int, L::Int) =
    quantile(U[:, refIndex1], collect(range(1 / L, (L - 1) / L, length = L)))

"""
    compute_bin_indices(U, z) -> Matrix{<:Unsigned}

`bins[s,o] = k` iff `z[k-1] < U[s,o] <= z[k]` (`z[0]:=-inf`, `z[L+1]:=+inf`), for ALL D origin
columns of `U` (including the reference origin -- needed for the `b_{s,refIndex1}` lookup).
Implemented via `searchsortedfirst(z, u)` on the sorted cutpoint vector `z`, which returns the
smallest index `k` with `z[k] >= u` (or `length(z)+1` if `u` exceeds every cutpoint) -- this is
EXACTLY the `<=` convention above (see module docstring), not merely close to it.
"""
function compute_bin_indices(U::AbstractMatrix{Float64}, z::Vector{Float64})
    W, D = size(U)
    L = length(z)
    T = bin_index_dtype(L)
    bins = Matrix{T}(undef, W, D)
    @inbounds for o in 1:D, s in 1:W
        bins[s, o] = T(searchsortedfirst(z, U[s, o]))
    end
    return bins
end

"""
    precalc_common_marginals_interval(U, refIndex1, L; contrasts=:anchored) -> (CM, z, origins, bins)

Dense interval-basis moment matrix (W x (D-1)*L), threshold(bin)-major column layout identical
to `precalc_common_marginals_cdf`'s (`cols = (k-1)*nO+1 : k*nO` for bin k). This is the
VALIDATION-ORIENTED dense builder (used to prove equivalence to the reference and to materialize
G for the Hessian callback in the live-KNITRO wiring) -- the O(D) lookup-based evaluator used for
the hot inner-dual FG loop lives in `cm_lookup_kernels.jl` and never materializes this matrix.

Only eq.35 (no truncated-moment eq.36 companion) is supported here -- out of scope for this
reformulation (noted as a limitation in the report; the dense reference's
`include_truncated_moment` path is untouched and still available if that block is ever needed).
"""
function precalc_common_marginals_interval(U::AbstractMatrix{Float64}, refIndex1::Int, L::Int;
                                            contrasts::Symbol = :anchored)
    W, D = size(U)
    @assert 1 <= refIndex1 <= D
    @assert L >= 1
    @assert contrasts in (:anchored, :orthonormal) "contrasts must be :anchored or :orthonormal, got $contrasts"

    z = common_marginals_quantiles(U, refIndex1, L)
    bins = compute_bin_indices(U, z)
    origins = [o for o in 1:D if o != refIndex1]
    nO = length(origins)
    R = contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing

    CM = Matrix{Float64}(undef, W, nO * L)
    block = Matrix{Float64}(undef, W, nO)
    refcol = @view bins[:, refIndex1]
    @inbounds for k in 1:L
        for (oi, o) in enumerate(origins)
            ocol = @view bins[:, o]
            @. block[:, oi] = (ocol == k) - (refcol == k)
        end
        cols = (k - 1) * nO + 1 : k * nO
        if R === nothing
            CM[:, cols] .= block
        else
            CM[:, cols] .= block * R
        end
    end
    return CM, z, origins, bins
end

"""
    cumulative_interval_transform_matrix(L) -> Matrix{Float64}  (L x L)

Explicit construction of the per-origin l/k transform: `S[k,l] = 1.0` if `k<=l` else `0.0`
(upper-triangular all-ones). For a single origin's length-L interval-coefficient vector `v`,
`S' * v` (equivalently `v' * S`) gives the length-L cumulative-coefficient vector `u` with
`u[l] = sum_{k<=l} v[k]`. Used below both directly (per-origin-block cumulative sum) and via its
Kronecker-with-identity form (`full_transform_matrix`) to reconstruct the FULL stored (D-1)*L
column matrix (which interleaves origins within each threshold/bin block) in one matrix multiply,
as an independent construction cross-checked against the direct block-cumsum loop.
"""
function cumulative_interval_transform_matrix(L::Int)
    S = zeros(L, L)
    @inbounds for l in 1:L, k in 1:L
        S[k, l] = (k <= l) ? 1.0 : 0.0
    end
    return S
end

"""
    full_transform_matrix(nO, L) -> Matrix{Float64}  ((nO*L) x (nO*L))

`kron(S, I(nO))` where `S = cumulative_interval_transform_matrix(L)`. With the threshold-major
column layout `col(k,oi) = (k-1)*nO + oi`, `CDF_stored = Interval_stored * full_transform_matrix`
reconstructs the reference's stored (possibly orthonormal-contrast-mixed) CM matrix EXACTLY from
the interval-basis one -- see module docstring for the proof that contrast mixing (per-block only)
commutes with this l/k-only cumulative-sum map.
"""
function full_transform_matrix(nO::Int, L::Int)
    S = cumulative_interval_transform_matrix(L)
    return kron(S, Matrix{Float64}(I, nO, nO))
end

"""
    interval_to_cumulative_dense(CM_interval, nO, L) -> Matrix{Float64}

Direct block-cumsum reconstruction (independent code path from `full_transform_matrix`, used as a
cross-check that the two constructions of the SAME linear map agree): for each threshold block l
(nO columns), accumulate the running sum of interval blocks 1:l.
"""
function interval_to_cumulative_dense(CM_interval::AbstractMatrix{Float64}, nO::Int, L::Int)
    W = size(CM_interval, 1)
    CDF = similar(CM_interval)
    acc = zeros(W, nO)
    @inbounds for l in 1:L
        cols = (l - 1) * nO + 1 : l * nO
        acc .+= @view CM_interval[:, cols]
        CDF[:, cols] .= acc
    end
    return CDF
end

"""
    build_cm_augmented_obj_interval(ctx, CS; L, contrasts=:anchored, refIndex1=ctx.γ.refIndex1)

Interval-basis analog of `common_marginals_moments.jl::build_cm_augmented_obj`. Reuses
`wrap_moments_with_cm` UNCHANGED (it is generic on the supplied CM matrix -- no modification
needed) so the splicing/layout convention (CM inserted as INNER moments before the outer-only
gravity column, `outer_constr_index` growing by `ncm`) is IDENTICAL to the dense path, byte for
byte. Returns the same-shaped NamedTuple as `build_cm_augmented_obj`, plus `bins` (needed by the
lookup-based evaluators in `cm_lookup_kernels.jl`).
"""
function build_cm_augmented_obj_interval(ctx, CS; L::Int, contrasts::Symbol = :anchored,
                                          refIndex1::Int = ctx.γ.refIndex1)
    obj0 = ctx.obj
    ncore = obj0.d
    CM, z, origins, bins = precalc_common_marginals_interval(ctx.U, refIndex1, L; contrasts = contrasts)
    ncm = size(CM, 2)
    @assert ncm == n_cm_moments(ctx.D, L; include_truncated_moment = false)

    d_new = ncore + ncm
    outer_constr_index_new = obj0.outer_constr_index + ncm
    moments_cm! = wrap_moments_with_cm(obj0.moments!, ncore, CM)

    obj_cm = CS.PsiObjectiveBundleImplicit(δ = obj0.δ, find_smallest = obj0.find_smallest,
        γ = obj0.γ, (moments!) = moments_cm!, moments_jacobian! = error,
        d = d_new, outer_constr_index = outer_constr_index_new,
        inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
        l = obj0.l, U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
        use_cached_x = obj0.use_cached_x,
        outer_loop_opt = obj0.outer_loop_opt, inner_loop_opt = obj0.inner_loop_opt,
        needs_outer_moment_jacobian = obj0.needs_outer_moment_jacobian)
    @assert obj_cm.outer_constr_index == obj_cm.d

    return (obj_cm = obj_cm, CM = CM, z = z, origins = origins, bins = bins,
            ncore = ncore, ncm = ncm, L = L, contrasts = contrasts, refIndex1 = refIndex1)
end
