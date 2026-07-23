# ============================================================================
# Fixed-Fréchet-marginals moment construction (2026-07-23), Architectures A
# and B. Purely additive: does not modify common_marginals_moments.jl or
# cm_hessian_architectures.jl. See docs/FIXED_FRECHET_MARGINALS_MATH_NOTE_2026-07-23.md
# for the derivation this file implements.
#
# Column layout (task brief §7, deterministic, explicit):
#   columns 1 : (D-1)*L        -- EXISTING contrast block, threshold-major,
#                                  IDENTICAL construction to flexible CM's own
#                                  (precalc_common_marginals_cdf), except the
#                                  thresholds are now the ANALYTIC F* quantiles
#                                  (targets.thresholds) instead of empirical
#                                  quantile(U[:,ref],probs).
#   columns (D-1)*L+1 : D*L     -- NEW common block, one column per grid point
#                                  l, g_common,l(s) = 1{U[s,ref]<=u_l*} - t_l*.
# ncm_frechet = D*L = ncm_flexible + L (one extra column per active grid point).
# ============================================================================

"""
    precalc_frechet_reference_cdf(U, refIndex1, targets::FrechetReferenceTargets; contrasts=:anchored) -> (CM, z, origins)

Architecture-A (dense reference) construction. `z = targets.thresholds`
(analytic, NOT `quantile(U[:,refIndex1],...)`) -- the only substantive
difference from `precalc_common_marginals_cdf`'s own contrast-block loop,
which is otherwise reproduced verbatim (not re-derived) so the two are
guaranteed bit-comparable at equal thresholds (task brief §10.1's
transformed-vs-naive gate). The trailing `L` columns are new.
"""
function precalc_frechet_reference_cdf(U::AbstractMatrix{Float64}, refIndex1::Int,
                                        targets::FrechetReferenceTargets; contrasts::Symbol = :anchored)
    W, D = size(U)
    L = length(targets.probs)
    @assert 1 <= refIndex1 <= D
    @assert contrasts in (:anchored, :orthonormal) "contrasts must be :anchored or :orthonormal, got $contrasts"
    z = targets.thresholds
    p = targets.targets

    origins = [o for o in 1:D if o != refIndex1]
    nO = length(origins)
    R = contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing

    CDF_ref = Matrix{Float64}(undef, W, L)
    @inbounds for l in 1:L
        @. CDF_ref[:, l] = U[:, refIndex1] <= z[l]
    end

    ncols = nO * L + L
    CM = Matrix{Float64}(undef, W, ncols)
    block = Matrix{Float64}(undef, W, nO)
    @inbounds for l in 1:L
        for (oi, o) in enumerate(origins)
            @. block[:, oi] = (U[:, o] <= z[l]) - CDF_ref[:, l]
        end
        cols = (l - 1) * nO + 1 : l * nO
        if R === nothing
            CM[:, cols] .= block
        else
            CM[:, cols] .= block * R
        end
    end
    common_offset = nO * L
    @inbounds for l in 1:L
        @. CM[:, common_offset + l] = CDF_ref[:, l] - p[l]
    end
    return CM, z, origins
end

n_cm_frechet_moments(D::Int, L::Int) = D * L

"""
    build_cm_frechet_augmented_obj(ctx, CS, targets::FrechetReferenceTargets; contrasts=:anchored,
                                    refIndex1=ctx.γ.refIndex1) -> (obj_cm=..., ...)

Architecture-A analog of `build_cm_augmented_obj`, dispatching to the FIXED-
Fréchet restriction. Reuses `wrap_moments_with_cm` UNCHANGED (it is generic
in the CM block's column count -- verified by direct inspection, it never
reads `size(CM,2)` against any external expectation besides what it derives
from `CM` itself), so this file only supplies the new `CM` matrix.
"""
function build_cm_frechet_augmented_obj(ctx, CS, targets::FrechetReferenceTargets; contrasts::Symbol = :anchored,
                                         refIndex1::Int = ctx.γ.refIndex1)
    obj0 = ctx.obj
    ncore = obj0.d
    CM, z, origins = precalc_frechet_reference_cdf(ctx.U, refIndex1, targets; contrasts = contrasts)
    ncm = size(CM, 2)
    @assert ncm == n_cm_frechet_moments(ctx.D, length(targets.probs))

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

    return (obj_cm = obj_cm, CM = CM, z = z, origins = origins, ncore = ncore, ncm = ncm,
            L = length(targets.probs), contrasts = contrasts, refIndex1 = refIndex1, targets = targets)
end

# ============================================================================
# Architecture B: fast per-call construction from bin indices (production
# path). Mirrors wrap_moments_with_cm_archB / build_cm_augmented_obj_archB
# (cm_hessian_architectures.jl) exactly, reusing `fill_cm_columns_from_bins!`
# UNCHANGED for the existing (D-1)*L contrast block and adding a small new
# fill kernel for the trailing L common columns.
# ============================================================================

"Fill the trailing `L` common columns of `Gdest` (a W x L view) directly from bin indices: `Gdest[s,l] = 1{Bidx[s,refIndex1]<=l} - p[l]`."
function fill_cm_frechet_common_columns_from_bins!(Gdest::AbstractMatrix{Float64}, Bidx::AbstractMatrix{Int},
                                                     refIndex1::Int, L::Int, p::Vector{Float64})
    W = size(Gdest, 1)
    @assert size(Gdest, 2) == L
    @inbounds for l in 1:L
        pl = p[l]
        for s in 1:W
            Gdest[s, l] = Float64(Bidx[s, refIndex1] <= l) - pl
        end
    end
    return nothing
end

"""
    wrap_moments_with_cm_frechet_archB(core_moments!, ncore_full, Bidx, origins, refIndex1, L, R, targets_vec; chunk_size)

Architecture B analog for fixed Fréchet: calls `fill_cm_columns_from_bins!`
(UNCHANGED) for the leading `(D-1)*L` contrast columns, then
`fill_cm_frechet_common_columns_from_bins!` for the trailing `L`.
"""
function wrap_moments_with_cm_frechet_archB(core_moments!::Function, ncore_full::Int,
                                             Bidx::Matrix{Int}, origins::Vector{Int}, refIndex1::Int, L::Int,
                                             R::Union{Nothing,Matrix{Float64}}, targets_vec::Vector{Float64};
                                             chunk_size::Int = 2000)
    pregrav = ncore_full - 1
    nO = length(origins)
    Gtmp_cache = Ref{Matrix{Float64}}(Matrix{Float64}(undef, 0, 0))
    return function (K, G, θ, U, obj)
        n = size(U, 1)
        if size(Gtmp_cache[], 1) != n
            Gtmp_cache[] = Matrix{Float64}(undef, n, ncore_full)
        end
        Gtmp = Gtmp_cache[]
        core_moments!(K, Gtmp, θ, U, obj)
        @views G[:, 1:pregrav] .= Gtmp[:, 1:pregrav]
        @views G[:, end] .= Gtmp[:, end]
        contrast_cols = pregrav + 1 : pregrav + L * nO
        common_cols = pregrav + L * nO + 1 : pregrav + L * nO + L
        fill_cm_columns_from_bins!(@view(G[:, contrast_cols]), Bidx, origins, refIndex1, L, R; chunk_size = chunk_size)
        fill_cm_frechet_common_columns_from_bins!(@view(G[:, common_cols]), Bidx, refIndex1, L, targets_vec)
        return nothing
    end
end

"""
    build_cm_frechet_augmented_obj_archB(ctx, CS, targets::FrechetReferenceTargets; contrasts=:anchored,
                                          refIndex1=ctx.γ.refIndex1, chunk_size=2000) -> (obj_cm=..., ...)

Architecture-B analog of `build_cm_augmented_obj_archB`. `Bidx` is built
against the ANALYTIC thresholds `targets.thresholds` (via the existing,
generic `compute_bin_indices`, unchanged -- it takes any `z` vector).
"""
function build_cm_frechet_augmented_obj_archB(ctx, CS, targets::FrechetReferenceTargets; contrasts::Symbol = :anchored,
                                               refIndex1::Int = ctx.γ.refIndex1, chunk_size::Int = 2000)
    obj0 = ctx.obj
    ncore = obj0.d
    L = length(targets.probs)
    z = targets.thresholds
    origins = [o for o in 1:ctx.D if o != refIndex1]
    ncm = n_cm_frechet_moments(ctx.D, L)
    R = contrasts == :orthonormal ? orthonormal_contrast_matrix(ctx.D) : nothing
    Bidx = compute_bin_indices(ctx.U, z)

    d_new = ncore + ncm
    outer_constr_index_new = obj0.outer_constr_index + ncm
    moments_cm! = wrap_moments_with_cm_frechet_archB(obj0.moments!, ncore, Bidx, origins, refIndex1, L, R,
                                                       targets.targets; chunk_size = chunk_size)

    obj_cm = CS.PsiObjectiveBundleImplicit(δ = obj0.δ, find_smallest = obj0.find_smallest,
        γ = obj0.γ, (moments!) = moments_cm!, moments_jacobian! = error,
        d = d_new, outer_constr_index = outer_constr_index_new,
        inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
        l = obj0.l, U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
        use_cached_x = obj0.use_cached_x,
        outer_loop_opt = obj0.outer_loop_opt, inner_loop_opt = obj0.inner_loop_opt,
        needs_outer_moment_jacobian = obj0.needs_outer_moment_jacobian)
    @assert obj_cm.outer_constr_index == obj_cm.d

    return (obj_cm = obj_cm, z = z, origins = origins, ncore = ncore, ncm = ncm, L = L,
            contrasts = contrasts, refIndex1 = refIndex1, Bidx = Bidx, targets = targets)
end
