# ================================================================================================
# Continuation 12b: interval (non-cumulative) common-marginals restriction, D=4 full-A_od.
#
# NEW file (deliberately not added to common_marginals_moments.jl -- a sibling agent on branch
# diag/fullA-d4-exact-cm-interval-hessian is independently building bin-index machinery for a
# different purpose in that file's territory; this file duplicates the minimal amount needed
# here rather than depending on unseen work).
#
# Reuses (does not duplicate) `orthonormal_contrast_matrix` and `wrap_moments_with_cm` from
# common_marginals_moments.jl, which must be included first (see conditioning-battery scripts'
# include chain). Everything else here is new.
# ================================================================================================
using Statistics: quantile, std

"""
    precalc_common_marginals_interval(U, refIndex1, L; standardize=false, contrasts=:anchored)

Interval-indicator variant of `precalc_common_marginals_cdf`. Uses the SAME L quantile
thresholds z_1 < ... < z_L of U[:,refIndex1] (probabilities range(1/L,(L-1)/L,length=L), byte
for byte the same construction `precalc_common_marginals_cdf` uses), but each threshold l's
restriction is the interval-membership contrast

    1{z_{l-1} < U_o <= z_l} - 1{z_{l-1} < U_1 <= z_l},   z_0 := -Inf

instead of the cumulative-CDF contrast `1{U_o<=z_l} - 1{U_1<=z_l}`. This drops the implicit top
bin `(z_L, +Inf)`, keeping the restriction count at `(D-1)*L`, matching
`n_cm_moments(D,L)` exactly (same column count as the cumulative version, so `wrap_moments_with_cm`
and `cm_block_to_anchored_residuals` -- both generic on column layout, not on how CM was built --
apply unchanged).

`standardize=true` divides each of the `(D-1)*L` raw (pre-contrast-rotation) interval-indicator
columns by its own empirical std across the W baseline draws, BEFORE any `:orthonormal` rotation
is applied (so standardization always happens in the natural per-origin basis, then rotation -- if
requested -- mixes the already-standardized columns; this matches "each column divided by its own
empirical std under the baseline draws, before insertion into G" per the task brief, read as
"before insertion into the (possibly-rotated) G columns actually shipped").

Returns `(CM, z, origins)`, same shape/contract as `precalc_common_marginals_cdf`.
"""
function precalc_common_marginals_interval(U::AbstractMatrix{Float64}, refIndex1::Int, L::Int;
                                            standardize::Bool = false,
                                            contrasts::Symbol = :anchored)
    W, D = size(U)
    @assert 1 <= refIndex1 <= D
    @assert L >= 1
    @assert contrasts in (:anchored, :orthonormal) "contrasts must be :anchored or :orthonormal, got $contrasts"
    z = quantile(U[:, refIndex1], collect(range(1 / L, (L - 1) / L, length = L)))
    origins = [o for o in 1:D if o != refIndex1]
    nO = length(origins)
    R = contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing

    IND_ref = Matrix{Float64}(undef, W, L)
    @inbounds for l in 1:L
        lo = l == 1 ? -Inf : z[l-1]
        hi = z[l]
        @. IND_ref[:, l] = (U[:, refIndex1] > lo) & (U[:, refIndex1] <= hi)
    end

    CM = Matrix{Float64}(undef, W, nO * L)
    block = Matrix{Float64}(undef, W, nO)
    @inbounds for l in 1:L
        lo = l == 1 ? -Inf : z[l-1]
        hi = z[l]
        for (oi, o) in enumerate(origins)
            @. block[:, oi] = ((U[:, o] > lo) & (U[:, o] <= hi)) - IND_ref[:, l]
        end
        cols = (l - 1) * nO + 1 : l * nO
        CM[:, cols] .= block
    end

    if standardize
        @inbounds for j in 1:size(CM, 2)
            s = std(@view CM[:, j])
            if s > 0
                @views CM[:, j] ./= s
            end
        end
    end

    if R !== nothing
        @inbounds for l in 1:L
            cols = (l - 1) * nO + 1 : l * nO
            @views CM[:, cols] .= CM[:, cols] * R
        end
    end

    return CM, z, origins
end

"""
    build_cm_augmented_obj_from_CM(ctx, CS, CM) -> obj_cm

Generic version of `build_cm_augmented_obj` (common_marginals_moments.jl) that takes an
already-built CM matrix directly instead of calling `precalc_common_marginals_cdf` itself --
lets this file (and any other non-cumulative moment construction) reuse the exact same
`wrap_moments_with_cm` splicing logic and layout convention (CM inserted as INNER moments
immediately before the sole outer gravity column) without re-deriving it.
"""
function build_cm_augmented_obj_from_CM(ctx, CS, CM::Matrix{Float64})
    obj0 = ctx.obj
    ncore = obj0.d
    ncm = size(CM, 2)
    d_new = ncore + ncm
    outer_constr_index_new = obj0.outer_constr_index + ncm
    moments_cm! = wrap_moments_with_cm(obj0.moments!, ncore, CM)
    obj_cm = CS.PsiObjectiveBundleImplicit(δ = obj0.δ, find_smallest = obj0.find_smallest,
        γ = obj0.γ, (moments!) = moments_cm!, moments_jacobian! = error,
        d = d_new, outer_constr_index = outer_constr_index_new,
        inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
        l = obj0.l, U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
        use_cached_x = obj0.use_cached_x,
        threshold_state = obj0.threshold_state,   # 2026-07-24 release fix: was defaulting to Inf (disabled) on every rebuild
        outer_loop_opt = obj0.outer_loop_opt, inner_loop_opt = obj0.inner_loop_opt,
        needs_outer_moment_jacobian = obj0.needs_outer_moment_jacobian)
    @assert obj_cm.outer_constr_index == obj_cm.d
    return obj_cm
end
