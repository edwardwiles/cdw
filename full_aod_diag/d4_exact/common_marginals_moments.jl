# ================================================================================================
# Pure-CDF common-marginals restriction (CDW eq. 35), full-A_od variant.
#
# Core math (precalc_common_marginals_cdf, orthonormal contrast machinery,
# n_cm_moments, cm_block_to_anchored_residuals) is ported VERBATIM from the already-validated
# sequential-method implementation, sequential_gravity/common_marginals_moments.jl, branch
# fix/cm-fixed-dual-gradient (commit 8531a89), same git-common-dir as this worktree. That file's
# math is generic on the raw W x D baseline draw matrix U and a reference-origin index -- it has
# no sequential-method-specific dependency -- so it is reused unchanged here rather than
# re-derived. See that file's docstrings for the full CDW eq.35/36 derivation and the
# orthonormal-contrast derivation; not repeated here.
#
# What's NEW in this file (full-A_od glue, not in the sequential source):
#   - `wrap_moments_with_cm`: builds a `moments!`-compatible closure that calls an existing
#     core `moments!` function (e.g. `EK_moments_gammanorm_directgp!`) into the first `ncore`
#     columns of G, then splices the precomputed CM block into the remaining columns. The CM
#     block is THETA-INDEPENDENT (built once from the fixed baseline draws U), so this wrapper
#     adds zero new outer parameters and ForwardDiff sees the appended columns as literal
#     constants automatically -- same reasoning as the sequential file's own
#     `EK_moments_focal_cm!`.
#   - `build_cm_augmented_obj`: constructs a NEW `PsiObjectiveBundleImplicit` (via `CS`) that
#     wraps an existing full-A context's `obj`, WITHOUT mutating the original. `d` and
#     `outer_constr_index` both grow by `ncm` (this codebase's convention, confirmed at D=4:
#     `outer_constr_index == d` for the base economy, i.e. every moment column is an outer
#     constraint moment; the CM block is added under the same convention).
# ================================================================================================

using Statistics: quantile
using LinearAlgebra: I

# ---- ported verbatim from sequential_gravity/common_marginals_moments.jl (fix/cm-fixed-dual-gradient@8531a89) ----

function orthonormal_contrast_matrix(D::Int)
    n = D - 1
    n >= 1 || throw(ArgumentError("D must be >= 2"))
    return Matrix{Float64}(I, n, n) .+ ((1 / sqrt(D) - 1) / (D - 1)) .* ones(n, n)
end

n_cm_moments(D::Int, L::Int; include_truncated_moment::Bool) =
    include_truncated_moment ? 2 * (D - 1) * L : (D - 1) * L

"""
    precalc_common_marginals_cdf(U, refIndex1, L; include_truncated_moment, σHat=nothing, contrasts=:anchored)

Precompute the (W x ncols) common-marginals moment matrix, the L quantile thresholds z_l
(evenly-spaced-probability empirical quantiles of `U[:,refIndex1]`), and the ordered list of
non-reference origins. Column layout: threshold-major, eq.35 (CDF) block first (columns
`1:(D-1)*L`), eq.36 (truncated `(1-σ)`-power) companion (if requested) at offset `(D-1)*L`. See
file header.

`include_truncated_moment` is a REQUIRED kwarg (no default) -- per this repo's standing rule
(never default a scientific parameter that changes which economic restriction is imposed), every
caller must say explicitly whether it wants eq.35 alone or eq.35+eq.36. Production flexible-CM
callers (`build_cm_augmented_obj`, `build_cm_meanzc_augmented_obj`) always pass `true` --
`common_marginals = false`/no-CM production is unaffected, and `cm_frechet_level.jl`'s deliberate
single-family carve-out (fixed Fréchet as CM+level anchor, a structurally different restriction)
continues to pass `false` explicitly.

`σHat` (the trade elasticity σ, required whenever `include_truncated_moment=true`, unused/`nothing`
otherwise) is the ONLY parameter the eq.36 exponent depends on: for non-reference origin `o` and
quantile cutoff `z_l`, eq.36's raw feature is `z_o(ω)^(1-σ) * 1{z_o(ω)<z_l}` (same `z_o(ω)==U[:,o]`
draw and same cutoffs `z_l` eq.35 already uses -- NOT some other "underlying exponential draw",
see docs/fullA_common_marginals_handoff.md section 2's own eq.35 statement, which is on this exact
`U` array). NOTE: an earlier, never-wired-to-production version of this function used
`pw = μHat*(1-σHat)` (an extra `μHat` factor) for this exponent -- that formula does not match
CDW eq.36 and has been removed; `μHat` is no longer a parameter of this function.
"""
function precalc_common_marginals_cdf(U::AbstractMatrix{Float64}, refIndex1::Int, L::Int;
                                       include_truncated_moment::Bool,
                                       σHat::Union{Nothing,Real} = nothing,
                                       contrasts::Symbol = :anchored,
                                       probs::Union{Nothing,AbstractVector{Float64}} = nothing)
    W, D = size(U)
    @assert 1 <= refIndex1 <= D
    @assert L >= 1
    @assert contrasts in (:anchored, :orthonormal) "contrasts must be :anchored or :orthonormal, got $contrasts"
    include_truncated_moment && @assert(σHat !== nothing,
        "include_truncated_moment=true requires σHat (the fixed baseline-calibrated trade elasticity)")
    # Continuation 13, Section 6: `probs=` lets a caller supply an EXPLICIT probability grid (e.g.
    # nested_quantile_grids.jl's genuinely-nested Q_10/Q_20/Q_50) in place of the default grid --
    # the default path (`probs === nothing`) is byte-for-byte unchanged. Remediation task Part E
    # (finding F13): the default is NOT literally `k/L for k=1:L` -- it is `L` points evenly
    # spaced over [1/L, (L-1)/L] (`range(1/L, (L-1)/L, length=L)`), which at L=50 runs
    # 0.02..0.98 with spacing ~=0.0196 (not exactly 1/L), deliberately excluding p=0 and p=1.
    # This is correct/intended behavior (a CDF contrast at p=1 would be degenerate); only the
    # comment previously mislabeled it "the evenly-spaced k/L grid."
    if probs === nothing
        z = quantile(U[:, refIndex1], collect(range(1 / L, (L - 1) / L, length = L)))
    else
        @assert length(probs) == L "precalc_common_marginals_cdf: length(probs)=$(length(probs)) != L=$L"
        z = quantile(U[:, refIndex1], probs)
    end
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
            CM[:, cols] .= block * R
        end
    end
    if include_truncated_moment
        # CDW eq.36: E_F[ z_o'(ω)^(1-σ) · 1{z_o'(ω)<z_l} ], anchored to the reference origin the
        # SAME way eq.35 is (subtract the reference origin's own power-weighted indicator, using
        # the reference's OWN power weight -- NOT the non-reference origin's -- since the power
        # weight is itself origin-and-draw-specific, unlike eq.35's weight-1 indicator).
        pw = 1 - σHat
        Pow = Matrix{Float64}(undef, W, D)   # z_x(ω)^(1-σ) for every origin x (incl. reference), precomputed once
        @inbounds for x in 1:D
            @. Pow[:, x] = U[:, x]^pw
        end
        TM_ref = Matrix{Float64}(undef, W, L)
        @inbounds for l in 1:L
            @. TM_ref[:, l] = Pow[:, refIndex1] * CDF_ref[:, l]
        end
        tm_offset = nO * L
        @inbounds for l in 1:L
            for (oi, o) in enumerate(origins)
                @. block[:, oi] = Pow[:, o] * (U[:, o] <= z[l]) - TM_ref[:, l]
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

function orthonormal_contrast_matrix_inverse(D::Int)
    n = D - 1
    n >= 1 || throw(ArgumentError("D must be >= 2"))
    return Matrix{Float64}(I, n, n) .+ ((sqrt(D) - 1) / (D - 1)) .* ones(n, n)
end

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

# ---- NEW: full-A_od glue ----

"""
    wrap_moments_with_cm(core_moments!, ncore_full, CM) -> Function

Returns a `moments!`-signature closure `(K, G, θ, U, obj) -> nothing`. CRITICAL layout
constraint this respects: in this codebase's `PsiObjectiveBundleImplicit` convention, the moment
columns from `outer_constr_index` to `d` are treated as OUTER-only equality constraints
(evaluated at θ directly, never inner-CC-reweighted -- see `cc_algo/PsiObjectiveBundle.jl`'s
callable, `H[:, 2+outer_constr_index:2+d]`), and this suffix is currently exactly ONE column:
the gravity/orthogonality moment, which `moments/newGravityMoment!.jl` unconditionally writes to
`G[:, end]` of whatever view it is given. The common-marginals restriction, by contrast, is an
INNER moment (a restriction on the least-favorable reweighted F, imposed via its own dual
multiplier, per the task brief) -- so it must be inserted BEFORE the gravity column, not after
it, and `outer_constr_index` must shift by `ncm` so gravity remains the sole outer-only suffix.

Implementation: calls `core_moments!` into a same-eltype temporary buffer `G_tmp` sized
`(size(U,1), ncore_full)` (`ncore_full = obj0.d` before augmentation, i.e. INCLUDING the gravity
column at its original last position), then splices columns 1:(ncore_full-1) (pre-gravity core)
into `G`'s corresponding prefix, `CM` into the next `ncm` columns, and `G_tmp`'s last (gravity)
column into `G`'s new last column. `G_tmp` is freshly allocated per call (not yet a cached
buffer -- fine for correctness/validation; revisit only if profiling shows this call is hot,
per the standing "reduce moment-construction cost" vs "reduce inner-dual eval cost" distinction
in the task brief).
"""
function wrap_moments_with_cm(core_moments!::Function, ncore_full::Int, CM::Matrix{Float64})
    pregrav = ncore_full - 1
    return function (K, G, θ, U, obj)
        n = size(U, 1)
        G_tmp = similar(G, n, ncore_full)
        core_moments!(K, G_tmp, θ, U, obj)
        @views G[:, 1:pregrav] .= G_tmp[:, 1:pregrav]
        @views G[:, ncore_full:ncore_full+size(CM,2)-1] .= CM[1:n, :]
        @views G[:, end] .= G_tmp[:, end]
        return nothing
    end
end

"""
    build_cm_augmented_obj(ctx, CS; L, include_truncated_moment, contrasts=:anchored)

Given a `d4_exact_setup`/`d20_real_setup`-style context `ctx` (must have fields `obj`, `U`, `γ`,
`D`, and -- whenever `include_truncated_moment=true` -- `σ`), builds a NEW
`PsiObjectiveBundleImplicit` with `(D-1)*L` (`include_truncated_moment=false`, eq.35 only) or
`2*(D-1)*L` (`include_truncated_moment=true`, eq.35+eq.36 -- the production flexible-CM spec as of
2026-08-05) extra common-marginals moment columns inserted as INNER (dual-reweighted) moments
BEFORE the existing gravity/orthogonality column (see `wrap_moments_with_cm`'s docstring for why
this ordering matters), leaving `ctx.obj` untouched. `CS` is the `CounterfactualSensitivity`
module (passed explicitly to avoid a hard dependency on how the caller's namespace names it).
refIndex1 is read from `ctx.γ.refIndex1` (this codebase's existing CDF-reference-origin
convention, already used elsewhere) unless overridden. `outer_constr_index` grows by `ncm`
along with `d`, keeping gravity as the sole outer-only suffix column at the new last position.

`include_truncated_moment` is REQUIRED (no default -- see `precalc_common_marginals_cdf`'s own
docstring for the rationale): every caller must say explicitly whether it wants one or two
feature families. Production flexible-CM callers pass `true`; `cm_frechet_level.jl`'s deliberate
single-family carve-out passes `false`.

Returns `(obj_cm, CM, z, origins, ncore, ncm, L, contrasts, include_truncated_moment, refIndex1,
n_families, ncm_cdf, ncm_pow)` -- the last three are new dimension metadata (2026-08-05
truncated-power task): `n_families` is `1` or `2`, `ncm_cdf = (D-1)*L` (the eq.35 sub-block width,
ALWAYS `(D-1)*L` regardless of `n_families`), `ncm_pow` is `(D-1)*L` when `n_families==2` else `0`
(`ncm_cdf + ncm_pow == ncm` always). Every downstream consumer that needs to slice the CM block by
family should use these fields rather than re-deriving `(D-1)*L`/`2*(D-1)*L` locally.
"""
function build_cm_augmented_obj(ctx, CS; L::Int, include_truncated_moment::Bool,
                                 contrasts::Symbol = :anchored,
                                 refIndex1::Int = ctx.γ.refIndex1,
                                 probs::Union{Nothing,AbstractVector{Float64}} = nothing)
    obj0 = ctx.obj
    ncore = obj0.d
    σHat = include_truncated_moment ? ctx.σ : nothing
    CM, z, origins = precalc_common_marginals_cdf(ctx.U, refIndex1, L;
        include_truncated_moment = include_truncated_moment, σHat = σHat,
        contrasts = contrasts, probs = probs)
    ncm = size(CM, 2)
    @assert ncm == n_cm_moments(ctx.D, L; include_truncated_moment = include_truncated_moment)
    nO = length(origins)
    ncm_cdf = nO * L
    ncm_pow = include_truncated_moment ? nO * L : 0
    n_families = include_truncated_moment ? 2 : 1
    @assert ncm_cdf + ncm_pow == ncm

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

    return (obj_cm = obj_cm, CM = CM, z = z, origins = origins, ncore = ncore, ncm = ncm,
            L = L, contrasts = contrasts, include_truncated_moment = include_truncated_moment,
            refIndex1 = refIndex1, n_families = n_families, ncm_cdf = ncm_cdf, ncm_pow = ncm_pow)
end
