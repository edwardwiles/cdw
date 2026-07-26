# ================================================================================================
# Fixed Fréchet as flexible CM plus a common-level anchor -- Part II (moment-construction layer).
#
# See docs/FRECHET_AS_CM_PLUS_LEVEL_MATHEMATICS_2026-07-25.md for the full derivation. Summary:
# flexible CM already imposes C'f_l(omega)=0 for each threshold l (C = the anchored/orthonormal
# contrast in common_marginals_moments.jl, columns spanning the mean-zero subspace 1^perp).
# u = ones(D)/sqrt(D) is EXACTLY orthogonal to C's column space in both contrast modes (proved in
# the math doc), so appending ONE extra column per threshold,
#   level_l(omega) = u'f_l(omega) - (u'1)*F*(H_l) = (1/sqrt(D))*sum_o 1{z_o(omega)<H_l} - sqrt(D)*p_l
# (F*(H_l) = p_l exactly, since H_l IS the reference origin's own p_l-quantile -- see math doc §2
# and the prior Fréchet port's own FrechetReferenceTargets convention, "targets: t_l* = p_l"), turns
# flexible CM (D-1)*L restrictions into fixed-Fréchet's DL restrictions, with EXACT equivalence to
# direct country-by-country fixed Fréchet (math doc §3). This is a pure ADDITIVE extension: it does
# not modify common_marginals_moments.jl, common_marginals_interval.jl, or cm_hessian_architectures.jl
# -- it reuses `wrap_moments_with_cm` (generic on the supplied moment matrix, common_marginals_moments.jl)
# and `fill_cm_columns_from_bins!` (cm_hessian_architectures.jl) UNCHANGED, adding only the level
# block's own dense/bin-lookup construction, mirroring their exact patterns.
#
# Feature set is CDF-only (task's explicit scope: "Do not enable the unrequested :cdf_power feature
# set") -- no dependency on theta_star/sigma/scale, only on the SAME probs/z grid CM already builds.
# ================================================================================================

n_frechet_level_moments(L::Int) = L

"""
    frechet_level_probs(L; probs=nothing) -> Vector{Float64}

The SAME probability grid `precalc_common_marginals_cdf`/`common_marginals_quantiles` use by
default (`range(1/L,(L-1)/L,length=L)`), or an explicit caller-supplied grid -- byte-for-byte
identical construction to CM's own, so thresholds and level targets are guaranteed consistent with
whatever grid CM is actually using for this run (task requirement: reuse the exact CM thresholds).
"""
function frechet_level_probs(L::Int; probs::Union{Nothing,AbstractVector{Float64}} = nothing)
    if probs === nothing
        return collect(range(1 / L, (L - 1) / L, length = L))
    end
    @assert length(probs) == L "frechet_level_probs: length(probs)=$(length(probs)) != L=$L"
    return collect(probs)
end

"""
    frechet_level_targets(D, L; probs=nothing) -> Vector{Float64}

`target_l = sqrt(D)*p_l = (u'1)*F*(H_l)` with `u=ones(D)/sqrt(D)`, `F*(H_l)=p_l` (math doc §2: H_l
is BY CONSTRUCTION the reference origin's own p_l-quantile, so the common Fréchet CDF's target
value there is exactly p_l -- the same convention the prior direct-country Fréchet port used,
`FrechetReferenceTargets.targets = p_l`, confirmed in `frechet_reference_targets.jl`). Pure
function of `(D,L,probs)` -- theta-independent, computed once per campaign, like the CM block
itself.
"""
function frechet_level_targets(D::Int, L::Int; probs::Union{Nothing,AbstractVector{Float64}} = nothing)
    p = frechet_level_probs(L; probs = probs)
    return sqrt(D) .* p
end

"""
    precalc_frechet_level_dense(U, z, D, targets) -> Matrix{Float64}  (W x L)

DENSE REFERENCE builder (validation-oriented, mirrors `precalc_common_marginals_cdf`'s own dense
construction pattern -- NOT the hot per-call path, see `fill_frechet_level_columns_from_bins!` for
that). `level[:,l] = (1/sqrt(D))*sum_{o=1}^D 1{U[s,o]<=z[l]} - targets[l]`, using ALL D origin
columns (symmetric, no reference-origin differencing -- this is the one structural difference from
the CM block, which uses only the `D-1` non-reference origins).
"""
function precalc_frechet_level_dense(U::AbstractMatrix{Float64}, z::Vector{Float64}, D::Int,
                                      targets::Vector{Float64})
    W = size(U, 1)
    L = length(z)
    @assert length(targets) == L
    invsqrtD = 1.0 / sqrt(D)
    level = Matrix{Float64}(undef, W, L)
    @inbounds for l in 1:L
        zl = z[l]
        for s in 1:W
            acc = 0.0
            for o in 1:D
                acc += U[s, o] <= zl ? 1.0 : 0.0
            end
            level[s, l] = invsqrtD * acc - targets[l]
        end
    end
    return level
end

"""
    fill_frechet_level_columns_from_bins!(Gdest, Bidx, D, L, targets; chunk_size=2000)

Architecture-B analogue of `fill_cm_columns_from_bins!` (cm_hessian_architectures.jl) for the level
block: `Gdest[s,l] = (1/sqrt(D))*sum_{o=1}^D (Bidx[s,o]<=l) - targets[l]`. O(W*D*L), same asymptotic
shape as `fill_cm_columns_from_bins!`'s O(W*(D-1)*L) -- the level block costs marginally MORE per
threshold (D origins summed vs D-1 differenced) but is the same order, matching the task's
"extra cost attributable to the L level columns, not a different architecture" requirement. `Bidx`
is the SAME `W x D` bin-index matrix CM's own `fill_cm_columns_from_bins!` uses (built once by
`compute_bin_indices`, theta-independent) -- no separate bin computation.
"""
function fill_frechet_level_columns_from_bins!(Gdest::AbstractMatrix{Float64}, Bidx::AbstractMatrix{Int},
                                                D::Int, L::Int, targets::Vector{Float64}; chunk_size::Int = 2000)
    W = size(Gdest, 1)
    @assert size(Gdest, 2) == L
    @assert length(targets) == L
    invsqrtD = 1.0 / sqrt(D)
    cs = min(chunk_size, W)
    start = 1
    @inbounds while start <= W
        stop = min(start + cs - 1, W)
        for l in 1:L
            tl = targets[l]
            for s in start:stop
                acc = 0.0
                for o in 1:D
                    acc += Bidx[s, o] <= l ? 1.0 : 0.0
                end
                Gdest[s, l] = invsqrtD * acc - tl
            end
        end
        start = stop + 1
    end
    return nothing
end

"""
    build_cm_frechet_level_augmented_obj(ctx, CS; L, contrasts=:anchored, probs=nothing,
                                          refIndex1=ctx.γ.refIndex1) -> NamedTuple

DENSE reference construction (Architecture A -- validation-oriented, mirrors
`build_cm_augmented_obj`'s own shape exactly, not the hot production path). Builds the CM block
EXACTLY as `precalc_common_marginals_cdf` does (unchanged, reused), builds the level block via
`precalc_frechet_level_dense`, concatenates `[CM level]` horizontally into ONE `(W x (ncm+L))`
matrix, and hands it to `wrap_moments_with_cm` UNCHANGED (that function is generic on the supplied
moment matrix -- no modification needed, confirming task §5's "reuse the exact CM forward and
transpose operators"). Column layout: CM block first (`(D-1)*L` columns, threshold-major, IDENTICAL
to flexible CM's own layout), level block last (`L` columns, one per threshold) -- see
`docs/COMMON_FRECHET_CM_DRIVER_PORT_2026-07-25.md` for the full machine-readable layout table
(Part II §6 of the task).

Returns the same-shaped NamedTuple as `build_cm_augmented_obj`, plus `level_targets`, `probs`, and
`ncm_cm`/`ncm_level` (the two block sizes) so callers can locate either block within the combined
`ncm = ncm_cm + ncm_level` columns.
"""
function build_cm_frechet_level_augmented_obj(ctx, CS; L::Int, contrasts::Symbol = :anchored,
                                               refIndex1::Int = ctx.γ.refIndex1,
                                               probs::Union{Nothing,AbstractVector{Float64}} = nothing)
    obj0 = ctx.obj
    ncore = obj0.d
    CM, z, origins = precalc_common_marginals_cdf(ctx.U, refIndex1, L; contrasts = contrasts, probs = probs)
    ncm_cm = size(CM, 2)
    @assert ncm_cm == n_cm_moments(ctx.D, L)

    level_probs = frechet_level_probs(L; probs = probs)
    level_targets = frechet_level_targets(ctx.D, L; probs = level_probs)
    LEVEL = precalc_frechet_level_dense(ctx.U, z, ctx.D, level_targets)
    ncm_level = size(LEVEL, 2)
    @assert ncm_level == L

    CMF = hcat(CM, LEVEL)
    ncm = ncm_cm + ncm_level
    @assert ncm == ctx.D * L   # task §3: exactly DL restrictions total

    d_new = ncore + ncm
    outer_constr_index_new = obj0.outer_constr_index + ncm
    moments_cmf! = wrap_moments_with_cm(obj0.moments!, ncore, CMF)

    obj_cmf = CS.PsiObjectiveBundleImplicit(δ = obj0.δ, find_smallest = obj0.find_smallest,
        γ = obj0.γ, (moments!) = moments_cmf!, moments_jacobian! = error,
        d = d_new, outer_constr_index = outer_constr_index_new,
        inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
        l = obj0.l, U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
        use_cached_x = obj0.use_cached_x,
        threshold_state = obj0.threshold_state,
        outer_loop_opt = obj0.outer_loop_opt, inner_loop_opt = obj0.inner_loop_opt,
        needs_outer_moment_jacobian = obj0.needs_outer_moment_jacobian)
    @assert obj_cmf.outer_constr_index == obj_cmf.d

    return (obj_cm = obj_cmf, CM = CMF, z = z, origins = origins, ncore = ncore, ncm = ncm,
            ncm_cm = ncm_cm, ncm_level = ncm_level, L = L, contrasts = contrasts,
            include_truncated_moment = false, refIndex1 = refIndex1,
            level_targets = level_targets, level_probs = level_probs,
            marginal_restriction = :common_frechet)
end
