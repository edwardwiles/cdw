# Pure-CDF common-marginals restriction (CDW eq. 35), built on top of the reduced focal-only CC
# moments (focal_moments.jl). For each non-reference origin o and each of L quantile thresholds
# z_l (evenly-spaced-probability empirical quantiles of the REFERENCE origin's raw baseline draw
# U[:,refIndex1]), imposes
#     E_F[1{U_o <= z_l}] = E_F[1{U_refIndex1 <= z_l}]      for all o != refIndex1, l = 1..L
# as (D-1)*L extra inner-loop moments -- CDW's eq. 35 written in terms of the raw Exp(1) draws U
# rather than the transformed productivity z = U^{-mu}. Since z is a strictly DECREASING function
# of U applied identically across origins (same mu for everyone), matching the CDF of U at
# evenly-spaced probability levels is equivalent (up to relabeling which quantile index is which)
# to matching the CDF of z at the L_CM quantiles of Frechet(1,thetaHat) -- see CDW footnote 24 and
# the legacy implementation this mirrors, prepare_cc/precalcCDFs.jl.
#
# These moments are CONSTANT in theta -- they depend only on the fixed baseline draws U, never on
# the outer parameters (mu, sigma, gamma'_focal, A[.,focal]) -- so they are computed ONCE, before
# the outer loop, and simply appended as extra columns of G. No outer-loop parameters are added,
# and no special autodiff handling is needed: ForwardDiff sees these columns as literal constants
# (their theta-partials are exactly zero) automatically, since the appended values never reference
# theta anywhere in EK_moments_focal_cm! below.
#
# Deviation from the legacy implementation: the reference origin's own comparison against itself
# (o == refIndex1) is DROPPED here (legacy code kept it, trivially all-zero). A constant-zero
# moment column leaves its dual/Lagrange-multiplier direction completely unconstrained in the
# inner KNITRO problem, which is bad for conditioning -- there is no reason to pay for it.
#
# eq. 36 (the companion truncated-(1-sigma)-moment condition, `include_truncated_moment=true`
# below) is ALSO implemented -- CDW impose it alongside eq. 35 in their main results to protect
# the upper-tail/price-index approximation:
#     E_F[z_o(ω)^(1-σ) 1{z_o(ω) < z_l}] = E_F[z_ref(ω)^(1-σ) 1{z_ref(ω) < z_l}]
#
# EXPONENT NOTE: this is written on the raw draws as U^(μHat*(1-σHat)), NOT as a literal
# z^(1-σ) with z=U^(-μHat). Reason: this codebase's "(1-σ)th moment of price" objects are already
# established elsewhere (focal_moments.jl's own price-index/trade-share moments use exactly
# `U[ω,o]^(μ*(1-σ))`, matching price_od^(1-σ) = (w_o·τ_od)^(1-σ)·z_o^(σ-1) under price=w·τ/z,
# z=U^(-μ)) -- i.e. what the MODEL's price index actually needs is z^(σ-1), which written on U is
# U^(μ(1-σ)), and price^(1-σ) is an INCREASING function of productivity z for σ>1, so it correctly
# emphasizes the upper productivity tail, matching CDW's stated motivation ("mitigate risks of
# inadequate upper-tail approximation"). This is also exactly the exponent the legacy
# implementation uses (prepare_cc/precalcCDFs.jl's `Truncated_moment_11`, computed on Ubar with
# exponent `μHat*(1-σHat)`) -- matched here rather than re-derived from the paper's own (possibly
# differently-gauged) z notation, since the legacy code is the validated reference for what was
# actually run. μHat, σHat are the FIXED baseline-calibrated values (not the outer loop's
# currently-searched μ,σ) -- same footnote-24 "fixed at F*" convention as the z_l thresholds
# themselves.

using Statistics: quantile
using LinearAlgebra: I

# ================================================================================================
# Orthonormal country contrasts (alternative to the country-1-anchored differences above).
# ================================================================================================
#
# At each threshold, the anchored block is h_old = B*f(Z), f(Z) the D-vector of country-level
# feature values (the CDF indicator for eq.35, the truncated-(1-sigma) value for eq.36), and
#     B = [-1_n | I_n]     (n = D-1 rows, one per non-reference origin; column 1 = reference).
# All n moments share country 1, inducing correlation across the country dimension: BB' = I_n +
# 1_n*1_n'. Since this holds regardless of F, the SAME reparameterization diagonalizes it for any
# F, so it can be applied once, deterministically, to the fixed feature-difference construction --
# no new outer parameters, no F-dependence, exactly like the anchored version.
#
# R = (BB')^{-1/2} has the closed form (eigendecomposition of aI+b*11' with a=b=1, n=D-1):
#     R = I_n + (1/sqrt(D) - 1) * (1_n*1_n')/(D-1)
# (the 1_n-direction eigenvalue of BB' is 1+n=D, giving eigenvalue 1/sqrt(D) of R there; every
# direction orthogonal to 1_n is unchanged, eigenvalue 1). R is symmetric by construction.
#
# h_new = R*h_old is an EXACT, invertible (det(R)>0) linear reparameterization of the SAME (D-1)
# equality restrictions at that threshold -- E_F[h_old]=0 <=> E_F[h_new]=0 -- applied identically,
# independently, at every threshold and for both feature families (eq.35, eq.36), never mixing
# information across thresholds. Applied at the DRAW level (before any expectation is taken): for
# each threshold's W x (D-1) block M (rows=draws), replace M by M*R' (R symmetric, so M*R'=M*R);
# linearity of expectation over draws then gives E_F[new row] = R*E_F[old row] as required.

"""
    orthonormal_contrast_matrix(D)

R = (BB')^{-1/2} in closed form for the (D-1)x(D-1) anchored-contrast Gram matrix BB'=I+11'. See
the file-level comment above for the derivation. Depends only on D; callers should compute once
and reuse across all thresholds/feature families for a given D.
"""
function orthonormal_contrast_matrix(D::Int)
    n = D - 1
    n >= 1 || throw(ArgumentError("D must be >= 2"))
    return Matrix{Float64}(I, n, n) .+ ((1 / sqrt(D) - 1) / (D - 1)) .* ones(n, n)
end

"Number of extra common-marginals moments for D countries and L quantiles (reference origin
excluded); doubles if the eq.-36 truncated-moment companion is also included. Unaffected by the
choice of contrasts (:anchored vs :orthonormal) -- R is invertible, so the moment COUNT is
identical, only the (fixed, invertible) linear combination changes."
n_cm_moments(D::Int, L::Int; include_truncated_moment::Bool = false) =
    include_truncated_moment ? 2 * (D - 1) * L : (D - 1) * L

"""
    precalc_common_marginals_cdf(U, refIndex1, L; include_truncated_moment=false, μHat=nothing, σHat=nothing, contrasts=:anchored)

Precompute the common-marginals moment matrix, the L quantile thresholds z_l, and the ordered
list of non-reference origins. `U` is the raw W x D baseline Exp(1) draw matrix (== Ubar whenever
thetaConstant=0, i.e. always in this codebase's live sequential-gravity configuration -- see
prepare_cc/createUDerivatives!.jl).

`contrasts`:
  - `:anchored` (default): CDW eq. 35/36 exactly as written, each non-reference origin `o`
    compared directly against the fixed reference `refIndex1`.
  - `:orthonormal`: an EXACT, invertible linear reparameterization of the same restrictions (see
    `orthonormal_contrast_matrix`) -- same economic content, same restriction COUNT, intended to
    remove the country-dimension correlation the anchored form induces (every anchored moment
    shares the reference country) and thereby improve inner-solve conditioning. Target remains
    zero either way (R*0=0).

Column layout: THRESHOLD-MAJOR. The eq. 35 block occupies columns `1:(D-1)*L`, with threshold
`l`'s `(D-1)` columns at `(l-1)*(D-1)+1 : l*(D-1)` (column order within a threshold follows the
returned `origins` list, i.e. this is the RAW/anchored order regardless of `contrasts` -- under
`:orthonormal` these columns hold the ROTATED contrasts, not per-origin residuals; use
`anchored_cm_residuals` for interpretable per-origin diagnostics regardless of which mode
generated the solve). If `include_truncated_moment=true` (requires `μHat`, `σHat`), the eq. 36
companion block follows at offset `(D-1)*L`, same threshold-major layout. Total columns:
`n_cm_moments(D, L; include_truncated_moment)`.
"""
function precalc_common_marginals_cdf(U::AbstractMatrix{Float64}, refIndex1::Int, L::Int;
                                       include_truncated_moment::Bool = false,
                                       μHat::Union{Nothing,Real} = nothing,
                                       σHat::Union{Nothing,Real} = nothing,
                                       contrasts::Symbol = :anchored)
    W, D = size(U)
    @assert 1 <= refIndex1 <= D
    @assert L >= 1
    @assert contrasts in (:anchored, :orthonormal) "contrasts must be :anchored or :orthonormal, got $contrasts"
    include_truncated_moment && @assert(μHat !== nothing && σHat !== nothing,
        "include_truncated_moment=true requires μHat and σHat (the fixed baseline-calibrated values)")
    z = quantile(U[:, refIndex1], collect(range(1 / L, (L - 1) / L, length = L)))
    origins = [o for o in 1:D if o != refIndex1]
    nO = length(origins)
    R = contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing
    CDF_ref = Matrix{Float64}(undef, W, L)
    @inbounds for l in 1:L
        @. CDF_ref[:, l] = U[:, refIndex1] <= z[l]
    end

    ncols = include_truncated_moment ? 2 * nO * L : nO * L
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
            CM[:, cols] .= block * R   # R symmetric (R'=R): draw-level h_new = R*h_old, rows of `block` are h_old'
        end
    end
    if include_truncated_moment
        pw = μHat * (1 - σHat)
        TM_ref = Matrix{Float64}(undef, W, L)
        @inbounds for l in 1:L
            @. TM_ref[:, l] = U[:, refIndex1]^pw * CDF_ref[:, l]
        end
        tm_offset = nO * L
        @inbounds for l in 1:L
            for (oi, o) in enumerate(origins)
                @. block[:, oi] = U[:, o]^pw * (U[:, o] <= z[l]) - TM_ref[:, l]
            end
            cols = tm_offset + (l - 1) * nO + 1 : tm_offset + l * nO
            if R === nothing
                CM[:, cols] .= block
            else
                CM[:, cols] .= block * R
            end
        end
    end
    return CM, z, origins
end

"""
    orthonormal_contrast_matrix_inverse(D)

R^{-1}, closed form (same aI+b*11' eigenstructure as `orthonormal_contrast_matrix`; R's
1_n-eigenvalue is 1/sqrt(D), everywhere else 1, so R^{-1} has 1_n-eigenvalue sqrt(D), everywhere
else 1 -- NOT equal to R itself, since R is symmetric but NOT an involution).
"""
function orthonormal_contrast_matrix_inverse(D::Int)
    n = D - 1
    n >= 1 || throw(ArgumentError("D must be >= 2"))
    return Matrix{Float64}(I, n, n) .+ ((sqrt(D) - 1) / (D - 1)) .* ones(n, n)
end

"""
    cm_block_to_anchored_residuals(meanvec, D, L, nO; include_truncated_moment=false, contrasts=:anchored)

Given the (e.g. LFD-reweighted) mean of a CM block (a length-`ncols` vector, threshold-major
layout as built by `precalc_common_marginals_cdf`), return the SAME quantity re-expressed in the
original anchored (country-vs-refIndex1) coordinates as an `nO x L` matrix (eq. 35) or a tuple of
two such matrices `(eq35, eq36)` if `include_truncated_moment=true` -- i.e. undoes the
R-rotation when `contrasts==:orthonormal` (h_old = R^{-1}*h_new), a no-op when `:anchored`. Lets
diagnostics report "does country o match the reference at threshold l" the same way regardless of
which coordinate system the solver actually used.
"""
function cm_block_to_anchored_residuals(meanvec::AbstractVector, D::Int, L::Int, nO::Int;
                                         include_truncated_moment::Bool = false,
                                         contrasts::Symbol = :anchored)
    Rinv = contrasts == :orthonormal ? orthonormal_contrast_matrix_inverse(D) : nothing
    function block_to_matrix(v::AbstractVector)
        M = Matrix{Float64}(undef, nO, L)
        for l in 1:L
            seg = @view v[(l-1)*nO+1:l*nO]
            M[:, l] .= Rinv === nothing ? seg : Rinv * seg
        end
        return M
    end
    eq35 = block_to_matrix(@view meanvec[1:nO*L])
    include_truncated_moment || return eq35
    eq36 = block_to_matrix(@view meanvec[nO*L+1:2*nO*L])
    return eq35, eq36
end

"""
    EK_moments_focal_cm!(K, G, θ, U, obj)

Wraps `EK_moments_focal!` (columns 1:D+1 = D focal trade shares + 1 focal counterfactual) and
appends `obj.γ.CM_Moments` (columns D+2:end) -- the precomputed, theta-independent CDF
common-marginals block. `size(G,2)` must equal `D + 1 + size(obj.γ.CM_Moments, 2)`.
"""
function EK_moments_focal_cm!(K, G, θ, U, obj)
    D = size(obj.γ.τ, 1)
    EK_moments_focal!(K, @view(G[:, 1:D+1]), θ, U, obj)
    # CM_Moments is precomputed for the FULL W-row baseline draw matrix, but this function is
    # also called by the CC framework's internal derivative machinery with a SHORTER row-prefix
    # of U (e.g. calculate_grad_k_autodiff! uses only the first 2 rows, since K is draw-invariant;
    # calculate_jac_θ_autodiff! uses the first N=Jac_W rows). Always slice CM_Moments down to
    # exactly the number of rows actually being evaluated (a prefix, since every caller slices U
    # as U[1:n,:], never a re-permutation) -- mirrors the legacy convention in moments!.jl's
    # sameMarginalsMoment block (`CDF_Moments[1:W, :]` with W = the LOCAL row count).
    n = size(U, 1)
    @views G[:, D+2:end] .= obj.γ.CM_Moments[1:n, :]
    return nothing
end

"""
    EK_moments_focal_norm_directgp_cm!(K, G, θ, U, obj)

Same as `EK_moments_focal_cm!` but wrapping the production γ_focal≡1-normalized, direct-γ'
objective variant (`EK_moments_focal_norm_directgp!`, focal_moments_directgp.jl) instead of the
plain `EK_moments_focal!`. Column layout identical: 1:D+1 focal moments, D+2:end CM block.
"""
function EK_moments_focal_norm_directgp_cm!(K, G, θ, U, obj)
    D = size(obj.γ.τ, 1)
    EK_moments_focal_norm_directgp!(K, @view(G[:, 1:D+1]), θ, U, obj)
    n = size(U, 1)   # see EK_moments_focal_cm!'s comment: always slice to the actual row count
    @views G[:, D+2:end] .= obj.γ.CM_Moments[1:n, :]
    return nothing
end

"""
    append_cm_moments!(G, offset, CM_Moments)

Copy the precomputed, theta-independent CM block into `G[:, offset+1:offset+size(CM_Moments,2)]`,
slicing `CM_Moments` down to `size(G,1)` rows first. General-purpose helper for drivers (e.g. the
production sequential-gravity driver) that build up G's columns incrementally across several
moment blocks (focal shares, gravity linearization, CM) rather than through a single fixed
wrapper -- see EK_moments_focal_cm!'s docstring for why the row-slicing is necessary.
"""
function append_cm_moments!(G::AbstractMatrix, offset::Int, CM_Moments::AbstractMatrix)
    n = size(G, 1)
    ncm = size(CM_Moments, 2)
    @views G[:, offset+1:offset+ncm] .= CM_Moments[1:n, :]
    return nothing
end
