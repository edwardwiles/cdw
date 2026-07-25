# ============================================================================
# Fixed-Fréchet Q0/Q1/Q2 quantile-axis bases + equation (38) truncated-power
# feature family (task brief Parts III/IV, 2026-07-24 reconciliation task).
#
# Column layout (mirrors cm_frechet_moments.jl exactly, extended with a
# second feature-family block when active):
#   CDF   block (eq.37, always active): 1 : (D-1)*L contrast + (D-1)*L+1:D*L common-pin  = D*L cols
#   POWER block (eq.38, new/optional):   next D*L cols, same contrast+common-pin structure
#   ncm = D*L                (feature_set=:cdf_only)
#   ncm = 2*D*L              (feature_set=:cdf_power)  -- matches draft's 2*D*H exactly at H=L
#
# Q0 (cumulative): existing cm_frechet_moments.jl construction (CDF), extended here with
#                   the analogous POWER block. Reused, not reimplemented.
# Q1 (interval):    NEW. Direct bin-index construction (mirrors common_marginals_interval.jl's
#                   pattern), independently validated against Q0 via the exact linear transform
#                   T = blockdiag(kron(S,I_nO), S) (S = cumulative_interval_transform_matrix(L)),
#                   applied separately to the CDF block and (if active) the POWER block.
# Q2 (whitened):    NEW. A further per-family, per-quantile-block linear map on top of Q1:
#                   W_fam = (chol/eig of the L x L Gram of Q1's own common-pin columns)^{-1/2},
#                   applied as kron(W_fam, I_{nO+1}) (mixes only the l-axis, never origins --
#                   same commuting structure as the existing orthonormal contrast R, which mixes
#                   only origins and never l -- so R, S, and W_fam all commute with each other by
#                   construction, and can be composed in any order).
#
# Every transform here is an EXACT nonsingular linear map on the SAME finite set of moment
# restrictions (task brief Part III.5's requirement) -- verified, not assumed, in
# test_frechet_bases_d4_gates.jl.
# ============================================================================

using LinearAlgebra: I, Symmetric, eigen, cholesky, Diagonal

# ---------------------------------------------------------------------------
# Transform matrices
# ---------------------------------------------------------------------------

"""
    frechet_block_transform_matrix(nO, L) -> Matrix{Float64}  ((nO*L+L) x (nO*L+L))

Interval -> cumulative transform for ONE feature-family block of the fixed-Fréchet layout
((D-1)*L contrast columns, threshold-major, THEN L common/reference-pin columns). Block-diagonal:
the contrast sub-block transforms via `kron(S,I_nO)` (identical to `full_transform_matrix` in
common_marginals_interval.jl -- reused directly), the common sub-block (a single "origin", the
fixed analytic pin) transforms via `S` alone. The two sub-blocks never mix (contrasts and the
common pin are algebraically independent column groups in both bases), so this is exactly
block-diagonal, not merely block-triangular.
"""
function frechet_block_transform_matrix(nO::Int, L::Int)
    Tcontrast = full_transform_matrix(nO, L)      # (nO*L) x (nO*L), from common_marginals_interval.jl
    Scommon = cumulative_interval_transform_matrix(L)  # L x L
    n = nO * L + L
    T = zeros(n, n)
    T[1:nO*L, 1:nO*L] .= Tcontrast
    T[nO*L+1:end, nO*L+1:end] .= Scommon
    return T
end

"""
    frechet_full_transform_matrix(nO, L; feature_set=:cdf_only) -> Matrix{Float64}

Full-layout transform, block-diagonal across feature families when `feature_set=:cdf_power`
(the CDF block and the POWER block are algebraically independent restrictions -- eq.37 and eq.38
never share a column -- so their interval<->cumulative maps never mix either).
"""
function frechet_full_transform_matrix(nO::Int, L::Int; feature_set::Symbol = :cdf_only)
    Tblock = frechet_block_transform_matrix(nO, L)
    feature_set === :cdf_only && return Tblock
    @assert feature_set === :cdf_power "feature_set must be :cdf_only or :cdf_power"
    n = size(Tblock, 1)
    T = zeros(2n, 2n)
    T[1:n, 1:n] .= Tblock
    T[n+1:end, n+1:end] .= Tblock
    return T
end

# ---------------------------------------------------------------------------
# Q1 (interval basis), CDF family (eq. 37)
# ---------------------------------------------------------------------------

"""
    precalc_frechet_reference_interval(U, refIndex1, targets; contrasts=:anchored) -> (CM, z, origins, bins)

Q1 analogue of `precalc_frechet_reference_cdf` (cm_frechet_moments.jl). Bin index
`b_{s,o} = k` iff `z_{k-1} < U[s,o] <= z_k` (`z` = `targets.thresholds`, the FIXED analytic
Fréchet(1,θ*) quantiles -- not empirical, matching the CDF-block convention already used for Q0).
Contrast columns: `B_{s,o,k} = 1{b_{s,o}=k} - 1{b_{s,ref}=k}` (identical in form to
`precalc_common_marginals_interval`, just against fixed thresholds). Common/reference-pin columns:
`B_{s,common,k} = 1{b_{s,ref}=k} - Δp_k`, `Δp_k = p_k - p_{k-1}` (`p_0:=0`) -- the bin-probability
mass under F*, i.e. the exact interval-basis analogue of the Q0 common block's `- p_l` pin.
"""
function precalc_frechet_reference_interval(U::AbstractMatrix{Float64}, refIndex1::Int,
                                             targets::FrechetReferenceTargets; contrasts::Symbol = :anchored)
    W, D = size(U)
    L = length(targets.probs)
    z = targets.thresholds
    p = targets.targets
    dp = Vector{Float64}(undef, L)
    dp[1] = p[1]
    @inbounds for l in 2:L
        dp[l] = p[l] - p[l-1]
    end

    bins = compute_bin_indices(U, z)   # common_marginals_interval.jl, generic on any `z`
    origins = [o for o in 1:D if o != refIndex1]
    nO = length(origins)
    R = contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing

    ncols = nO * L + L
    CM = Matrix{Float64}(undef, W, ncols)
    block = Matrix{Float64}(undef, W, nO)
    refcol = @view bins[:, refIndex1]
    @inbounds for k in 1:L
        for (oi, o) in enumerate(origins)
            ocol = @view bins[:, o]
            @. block[:, oi] = (ocol == k) - (refcol == k)
        end
        cols = (k - 1) * nO + 1 : k * nO
        CM[:, cols] .= (R === nothing ? block : block * R)
    end
    common_offset = nO * L
    @inbounds for k in 1:L
        dpk = dp[k]
        @. CM[:, common_offset + k] = (refcol == k) - dpk
    end
    return CM, z, origins, bins
end

# ---------------------------------------------------------------------------
# Q0 (cumulative), POWER family (eq. 38) -- new, mirrors precalc_frechet_reference_cdf
# ---------------------------------------------------------------------------

"""
    precalc_frechet_reference_power_cdf(U, refIndex1, targets; contrasts=:anchored) -> (CM, z, origins)

Q0 analogue of `precalc_frechet_reference_cdf` for the eq.38 truncated-power feature:
`E_F[z_o^{1-σ} 1{z_o<H_h}] = E_{F*}[z^{1-σ} 1{z<H_h}]`. Contrast columns (empirical, mirrors
`common_marginals_moments.jl`'s `include_truncated_moment` block, `pw=(1-σ)/θ*`):
`g_{o,l}(s) = U[s,o]^pw*1{U[s,o]<=z_l} - U[s,ref]^pw*1{U[s,ref]<=z_l}`. Common/reference-pin
columns (NEW relative to eq.36 -- pinned to the FIXED analytic target, `targets.power_targets`,
not to the empirical reference moment): `g_{common,l}(s) = U[s,ref]^pw*1{U[s,ref]<=z_l} - t^power_l`.
"""
function precalc_frechet_reference_power_cdf(U::AbstractMatrix{Float64}, refIndex1::Int,
                                              targets::FrechetReferenceTargets; contrasts::Symbol = :anchored)
    W, D = size(U)
    L = length(targets.probs)
    z = targets.thresholds
    tpow = targets.power_targets
    pw = (1.0 - targets.sigma) / targets.theta_star

    origins = [o for o in 1:D if o != refIndex1]
    nO = length(origins)
    R = contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing

    TM_ref = Matrix{Float64}(undef, W, L)
    @inbounds for l in 1:L
        @. TM_ref[:, l] = U[:, refIndex1]^pw * (U[:, refIndex1] <= z[l])
    end

    ncols = nO * L + L
    CM = Matrix{Float64}(undef, W, ncols)
    block = Matrix{Float64}(undef, W, nO)
    @inbounds for l in 1:L
        for (oi, o) in enumerate(origins)
            @. block[:, oi] = U[:, o]^pw * (U[:, o] <= z[l]) - TM_ref[:, l]
        end
        cols = (l - 1) * nO + 1 : l * nO
        CM[:, cols] .= (R === nothing ? block : block * R)
    end
    common_offset = nO * L
    @inbounds for l in 1:L
        @. CM[:, common_offset + l] = TM_ref[:, l] - tpow[l]
    end
    return CM, z, origins
end

n_cm_frechet_power_moments(D::Int, L::Int) = D * L

# ---------------------------------------------------------------------------
# Q1 (interval), POWER family
# ---------------------------------------------------------------------------

"""
    precalc_frechet_reference_power_interval(U, refIndex1, targets; contrasts=:anchored) -> (CM, z, origins, bins)

Interval-basis analogue of `precalc_frechet_reference_power_cdf`: within-bin power contributions
(task brief Part III.5's "difference adjacent cumulative truncated moments to obtain within-bin
power contributions"), built DIRECTLY from bin membership (not by explicit differencing of the
Q0 columns -- an independent code path, cross-checked against the Q0 block via
`frechet_block_transform_matrix` in the gate script, exactly as Q1's CDF block already is).
Contrast: `g_{o,k}(s) = U[s,o]^pw*1{b_{s,o}=k} - U[s,ref]^pw*1{b_{s,ref}=k}`. Common/reference-pin:
`g_{common,k}(s) = U[s,ref]^pw*1{b_{s,ref}=k} - Δt^power_k`, `Δt^power_k = t^power_k - t^power_{k-1}`
(`t^power_0 := 0`) -- the exact bin-mass analytic target under F*.
"""
function precalc_frechet_reference_power_interval(U::AbstractMatrix{Float64}, refIndex1::Int,
                                                   targets::FrechetReferenceTargets; contrasts::Symbol = :anchored)
    W, D = size(U)
    L = length(targets.probs)
    z = targets.thresholds
    tpow = targets.power_targets
    pw = (1.0 - targets.sigma) / targets.theta_star
    dtpow = Vector{Float64}(undef, L)
    dtpow[1] = tpow[1]
    @inbounds for l in 2:L
        dtpow[l] = tpow[l] - tpow[l-1]
    end

    bins = compute_bin_indices(U, z)
    origins = [o for o in 1:D if o != refIndex1]
    nO = length(origins)
    R = contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing

    Upw_ref = @. U[:, refIndex1]^pw
    refbin = @view bins[:, refIndex1]

    ncols = nO * L + L
    CM = Matrix{Float64}(undef, W, ncols)
    block = Matrix{Float64}(undef, W, nO)
    @inbounds for k in 1:L
        for (oi, o) in enumerate(origins)
            Upw_o = @. U[:, o]^pw
            obin = @view bins[:, o]
            @. block[:, oi] = Upw_o * (obin == k) - Upw_ref * (refbin == k)
        end
        cols = (k - 1) * nO + 1 : k * nO
        CM[:, cols] .= (R === nothing ? block : block * R)
    end
    common_offset = nO * L
    @inbounds for k in 1:L
        dtk = dtpow[k]
        @. CM[:, common_offset + k] = Upw_ref * (refbin == k) - dtk
    end
    return CM, z, origins, bins
end

# ---------------------------------------------------------------------------
# Unified builder: dispatches (basis, feature_set) onto the above, returns a
# generic (Architecture-A-ready) augmented objective -- NOT the fast structured
# path (see cm_frechet_bases_structured.jl for that).
# ---------------------------------------------------------------------------

n_cm_frechet_moments_full(D::Int, L::Int; feature_set::Symbol = :cdf_only) =
    feature_set === :cdf_only ? D * L : 2 * D * L

"""
    build_cm_frechet_augmented_obj_basis(ctx, CS, targets; basis=:cumulative, feature_set=:cdf_only,
                                          contrasts=:anchored, refIndex1=ctx.γ.refIndex1) -> NamedTuple

Generic (dense-Hessian-compatible) augmented objective for any (basis, feature_set) combination.
`basis in (:cumulative, :interval)`, `feature_set in (:cdf_only, :cdf_power)`. Always returns a
valid `CM` matrix usable with `wrap_moments_with_cm` + `archA_hess_cb_builder` (Architecture A's
dense Hessian construction is generic on any linear-in-weights moment matrix -- it has no
cumulative-specific assumption, unlike Architecture C's bin-table construction).
"""
function build_cm_frechet_augmented_obj_basis(ctx, CS, targets::FrechetReferenceTargets;
                                               basis::Symbol = :cumulative, feature_set::Symbol = :cdf_only,
                                               contrasts::Symbol = :anchored,
                                               refIndex1::Int = ctx.γ.refIndex1)
    @assert basis in (:cumulative, :interval) "basis must be :cumulative or :interval"
    @assert feature_set in (:cdf_only, :cdf_power) "feature_set must be :cdf_only or :cdf_power"
    obj0 = ctx.obj
    ncore = obj0.d

    CM_cdf, z, origins = basis === :cumulative ?
        precalc_frechet_reference_cdf(ctx.U, refIndex1, targets; contrasts = contrasts) :
        precalc_frechet_reference_interval(ctx.U, refIndex1, targets; contrasts = contrasts)[1:3]

    if feature_set === :cdf_only
        CM = CM_cdf
    else
        CM_pow, _, _ = basis === :cumulative ?
            precalc_frechet_reference_power_cdf(ctx.U, refIndex1, targets; contrasts = contrasts) :
            precalc_frechet_reference_power_interval(ctx.U, refIndex1, targets; contrasts = contrasts)[1:3]
        CM = hcat(CM_cdf, CM_pow)
    end
    ncm = size(CM, 2)
    @assert ncm == n_cm_frechet_moments_full(ctx.D, length(targets.probs); feature_set = feature_set)

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
            L = length(targets.probs), basis = basis, feature_set = feature_set,
            contrasts = contrasts, refIndex1 = refIndex1, targets = targets)
end

# ---------------------------------------------------------------------------
# Q2: analytically whitened basis
# ---------------------------------------------------------------------------

"""
    frechet_whiten_matrix_from_common_block(CM_interval_common::AbstractMatrix{Float64}; ridge=1e-10) -> Matrix{Float64}

Given the W x L common/reference-pin block of a Q1 (interval) CM matrix for ONE feature family,
build the L x L whitening matrix `Wmat = V * diag(1/sqrt(max(eig,ridge*max_eig))) * V'` from the
eigendecomposition of `Gram/W` (the empirical -- exactly the population, since `ctx.U` draws ARE
F* by construction -- covariance of the benchmark block). `ridge` floors tiny eigenvalues
(FP-roundoff only, per task brief Part III.5 -- "deterministic regularization only for numerical
roundoff, not to change the feasible set"); with L equally-spaced disjoint-bin indicators and no
p=0/1 endpoint (already excluded upstream), the true population Gram is generically full-rank, so
`ridge` is not expected to bind in practice -- checked explicitly in the gate script.
"""
function frechet_whiten_matrix_from_common_block(CM_common::AbstractMatrix{Float64}; ridge::Float64 = 1e-10)
    W = size(CM_common, 1)
    Gram = Symmetric(CM_common' * CM_common ./ W)
    ev = eigen(Gram)
    maxeig = maximum(ev.values)
    floor_ = ridge * maxeig
    n_floored = count(<(floor_), ev.values)
    invsqrt = 1.0 ./ sqrt.(max.(ev.values, floor_))
    Wmat = ev.vectors * Diagonal(invsqrt) * ev.vectors'
    return Symmetric(Wmat), n_floored
end

"""
    frechet_whiten_block_transform(Wmat, nO) -> Matrix{Float64}  ((nO*L+L) x (nO*L+L))

Assembles ONE feature family's Q2 transform with the SAME block-diagonal structure as
`frechet_block_transform_matrix` (contrast sub-block `kron(Wmat,I_nO)`, common sub-block `Wmat`
alone) -- the whitening matrix `Wmat` (`L x L`) plays exactly the role `S` (the cumulative<->
interval map) plays there: it mixes only the quantile/bin axis `l`, never origins, so it must
respect the same contrast/common column partition, not a combined `kron(Wmat, I_{nO+1})` (which
would incorrectly assume a per-l-contiguous [contrast;common] column layout that the actual CM
matrix -- contrast block for ALL l first, then common block -- does not have).
"""
function frechet_whiten_block_transform(Wmat::AbstractMatrix{Float64}, nO::Int)
    L = size(Wmat, 1)
    Tcontrast = kron(Wmat, Matrix{Float64}(I, nO, nO))
    n = nO * L + L
    T = zeros(n, n)
    T[1:nO*L, 1:nO*L] .= Tcontrast
    T[nO*L+1:end, nO*L+1:end] .= Wmat
    return T
end

"""
    frechet_full_whitening_transform(ctx, targets; feature_set=:cdf_only, contrasts=:anchored) -> (T2, meta)

Builds the FULL Q2 whitening transform on top of the Q1 (interval) basis, per feature family
(block-diagonal across CDF/POWER when both active -- eq.37 and eq.38 never share a column, same
reasoning as `frechet_full_transform_matrix`). Mixes only the l/k axis within a family (never
origins, never across families) -- commutes with the origin contrast `R` and the Q0<->Q1 map `T`
by construction (same argument as `full_transform_matrix`'s docstring).
"""
function frechet_full_whitening_transform(ctx, targets::FrechetReferenceTargets;
                                           feature_set::Symbol = :cdf_only, contrasts::Symbol = :anchored,
                                           refIndex1::Int = ctx.γ.refIndex1, ridge::Float64 = 1e-10)
    D = ctx.D; L = length(targets.probs)
    origins = [o for o in 1:D if o != refIndex1]
    nO = length(origins)

    CM_cdf_i, _, _, _ = precalc_frechet_reference_interval(ctx.U, refIndex1, targets; contrasts = contrasts)
    common_cdf = @view CM_cdf_i[:, nO*L+1:end]
    Wmat_cdf, nfloor_cdf = frechet_whiten_matrix_from_common_block(common_cdf; ridge = ridge)
    Tblk_cdf = frechet_whiten_block_transform(Wmat_cdf, nO)

    feature_set === :cdf_only && return Tblk_cdf, (nfloor_cdf = nfloor_cdf,)

    CM_pow_i, _, _, _ = precalc_frechet_reference_power_interval(ctx.U, refIndex1, targets; contrasts = contrasts)
    common_pow = @view CM_pow_i[:, nO*L+1:end]
    Wmat_pow, nfloor_pow = frechet_whiten_matrix_from_common_block(common_pow; ridge = ridge)
    Tblk_pow = frechet_whiten_block_transform(Wmat_pow, nO)

    n = size(Tblk_cdf, 1)
    T2 = zeros(2n, 2n)
    T2[1:n, 1:n] .= Tblk_cdf
    T2[n+1:end, n+1:end] .= Tblk_pow
    return T2, (nfloor_cdf = nfloor_cdf, nfloor_pow = nfloor_pow)
end
