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
using SpecialFunctions: gamma   # eq36_theoretical_truncated_moment's upper incomplete gamma (2-arg gamma(s,a))

# 2026-08-05 truncated-power task (corrected): the eq.36 truncated-power feature needs the SAME
# Fréchet-productivity-draw transform (`z_o(ω) = U_o(ω)^{-μ}`, `frechet_power_feature`) the
# ZC/meanZC restrictions use -- reused via this self-include guard (this codebase's own established
# idiom, e.g. cm_hessian_architectures.jl's many `isdefined(Main, :X) || include(...)` guards) so
# every caller of THIS file gets it regardless of whether it separately includes
# cm_meanzc_moments.jl. Do not reintroduce a second `U.^(-μk)` formula here -- call
# `frechet_power_feature` (widened to `k::Real` specifically for this file's non-integer
# `k=1-σ` use, same formula, same behavior for ZC's own integer-`k` callers).
isdefined(Main, :frechet_power_feature) || include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))

# ---- ported verbatim from sequential_gravity/common_marginals_moments.jl (fix/cm-fixed-dual-gradient@8531a89) ----

function orthonormal_contrast_matrix(D::Int)
    n = D - 1
    n >= 1 || throw(ArgumentError("D must be >= 2"))
    return Matrix{Float64}(I, n, n) .+ ((1 / sqrt(D) - 1) / (D - 1)) .* ones(n, n)
end

n_cm_moments(D::Int, L::Int; include_truncated_moment::Bool) =
    include_truncated_moment ? 2 * (D - 1) * L : (D - 1) * L

"""
    theoretical_u_threshold(p::Real) -> Float64

2026-08-05 (user-directed): the CLOSED-FORM (not Monte-Carlo/order-statistic-estimated) `U`-space
threshold to use for a common-marginals cutoff targeting Fréchet quantile PROBABILITY LEVEL `p`,
i.e. the value `u_thresh` such that using it in the EXISTING `1{U_x <= u_thresh}` code structure
(unchanged everywhere else -- forward/transpose kernels, bin-index/Hessian architecture, etc. all
still just consume `z` as an opaque ascending cutoff array and compare it against `U` exactly as
before) is mathematically equivalent to testing the LITERAL eq.35/36 statement
`1{z_x(ω) < z_p}` on the true Fréchet productivity draw `z_x(ω) = U_x(ω)^{-μ}` at target
probability `p`.

Derivation (`T=1`, the confirmed convention -- see `frechet_productivity_from_exponential`,
`cm_meanzc_moments.jl`; `θ = 1/μ` is the Fréchet shape):
  - Closed-form Fréchet quantile at level `p`: solving `P(z<z_p)=p` for `z=U^{-μ}`, `U~Exp(1)`
    gives `z_p = (-log(p))^{-μ}` (`= (T/(-ln p))^{1/θ}` with `T=1`).
  - `1{z_x(ω) < z_p} = 1{U_x(ω) > z_p^{-1/μ}} = 1{U_x(ω) > -log(p)}` (the map `z=U^{-μ}` is
    strictly DECREASING, so the inequality direction flips under it).
  - `1{U_x > c} - 1{U_ref > c} = -(1{U_x <= c} - 1{U_ref <= c})` for any threshold `c` -- an
    overall sign flip that does NOT change the restriction's feasible set (`E_F[X]=0 ⟺
    E_F[-X]=0`), so the EXISTING `1{U_x<=c}-1{U_ref<=c}` code structure, with `c=-log(p)`, imposes
    a restriction with the IDENTICAL feasible set as the literal `1{z_x<z_p}-1{z_ref<z_p}`
    statement -- no inequality-direction rewrite needed anywhere downstream.
  - Grid-symmetry check: this codebase's own equal-probability grid
    (`range(1/L,(L-1)/L,length=L)`) is symmetric under `p -> 1-p` (i.e. `{1-probs[l]} ==
    {probs[L+1-l]}` as a SET), so evaluating `theoretical_u_threshold` AT the grid's own `probs`
    values (rather than at `1 .- probs`) tests the exact same SET of L Fréchet-quantile levels,
    just re-indexed -- and, conveniently, `-log.(1 .- probs)` is monotonically INCREASING in
    `probs` (required: the bin-index architecture needs `z` sorted ascending), which is exactly
    `theoretical_u_threshold.(probs)` reduces to below.
  - `theoretical_u_threshold(p) := -log(1-p)` (substituting the grid-symmetry relabeling
    `p -> 1-p` into `c=-log(p)` above) -- which is ALSO, not coincidentally, exactly the ordinary
    closed-form Exp(1) quantile of `U` at probability `p` (`P(U<=u_p)=p ⟺ u_p=-log(1-p)`): since
    `U` is drawn EXACTLY Exp(1) by construction (not estimated), using its own known closed form
    instead of an empirical/order-statistic estimate is precisely "switch from empirical to
    theoretical," consistent with (and no more than) what the literal eq.35/36 Fréchet-quantile
    statement already requires once the grid-symmetry relabeling above is accounted for.

Empirically validated (not merely derived): `test_cm_truncated_power_2026-08-05.jl`'s theoretical-
cutoff gates check that a large-`W` Monte Carlo average of the resulting features converges to
`eq36_theoretical_truncated_moment` (below) as `W→∞`, which would fail under a sign/direction
error in this derivation.
"""
theoretical_u_threshold(p::Real) = -log(1 - p)

"""
    eq36_theoretical_truncated_moment(z_ℓ::Real, k::Real, μHat::Real) -> Float64

2026-08-05 (user-directed): closed-form population value of `E[z^k · 1{z<z_ℓ}]` for
`z=U^{-μ}`, `U~Exp(1)` (`T=1`), via the upper incomplete gamma function
`Γ(s,a) = ∫_a^∞ t^(s-1)e^(-t) dt`:

    E[z^k · 1{z<z_ℓ}] = Γ(1 - μk, z_ℓ^(-1/μ))

Derivation: `E[z^k·1{z<z_ℓ}] = E[U^{-μk}·1{U>z_ℓ^{-1/μ}}] = ∫_{z_ℓ^{-1/μ}}^∞ u^{-μk}e^{-u}du`,
which is exactly `Γ(1-μk, a)` with `a=z_ℓ^{-1/μ}` and `s-1=-μk`. As `z_ℓ→∞`, `a→0` and
`Γ(s,0)=Γ(s)` (for `s=1-μk>0`), recovering this codebase's own existing UNTRUNCATED
`ν_k=Γ(1-μk)` convention (`cm_meanzc_moments.jl` -- confirmed, not assumed, by direct comparison
in `test_cm_truncated_power_2026-08-05.jl`).

**Verification/reference use ONLY** -- this closed form is never used inside the actual inner
KNITRO dual solve (the restriction imposed there is always the empirical/reweighted-measure
computation over the realized `W` draws, per the whole robust-optimization architecture -- there
is no population-level substitute for optimizing over how those draws get reweighted). Use this
only to check that a Monte-Carlo average of the corresponding raw feature converges to it as
`W→∞`, or as a sanity target at a fixed `(z_ℓ,k,μ)`.
"""
function eq36_theoretical_truncated_moment(z_ℓ::Real, k::Real, μHat::Real)
    a = z_ℓ^(-1 / μHat)
    return gamma(1 - μHat * k, a)
end

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

`σHat` and `μHat`, both required whenever `include_truncated_moment=true` (unused/`nothing`
otherwise), together determine the eq.36 exponent. eq.35 (the CDF family, above) is a pure
rank/indicator statistic, and is therefore invariant to whatever strictly-monotonic representation
`U` happens to be in -- comparing `U[:,o] <= z_l` against a cutoff `z_l` ALSO derived as a quantile
of `U[:,refIndex1]` never mixes representations, so it validly tests "common marginals across
origins" regardless of whether `U` is the raw exponential draw or some monotonic transform of it.
eq.36 is a POWER-WEIGHTED moment and has NO such invariance -- it must be built from the actual
Fréchet productivity draw `z_o(ω) = U_o(ω)^{-μ}` (`μ = 1/θ`, `ctx.μHat`; confirmed via
`fix/zc-frechet-draw-moments-2026-08-05`/`frechet_power_feature`, cm_meanzc_moments.jl, that
`ctx.U` is genuinely the raw, untransformed `Exp(1)` draw in production), NOT from `U` directly.

**Exponent, corrected 2026-08-05 (user-directed, second correction, same day as the U-vs-z fix
below): the paper's eq.36 moment is `z^(σ-1)`, NOT `z^(1-σ)`** -- an earlier pass through this same
task used `1-σ` (matching a first, imprecise reading of the mission brief) and only caught the
sign after the U-vs-z bug below was already fixed and gates were passing; the user corrected it
directly mid-session. `z_o(ω)^(σ-1) = U_o(ω)^{-μ(σ-1)}`, computed via
`frechet_power_feature(U, σHat-1, μHat)` (the SAME canonical helper the ZC/meanZC restrictions
use, widened to a real-valued exponent for this non-integer `k=σ-1` use -- not a second,
independently-derived `U^power` formula). This is a standard CES-aggregator-shaped exponent in
this literature (prices scale as `1/z`, CES weights as `p^(1-σ)~z^(σ-1)`), consistent with the
user's direct correction. `-μ(σ-1)` is small and NEGATIVE at realistic `(μHat,σHat)` (real D20
`μHat≈0.13-0.20`, `σHat≈2.5-3`, giving a `U`-space exponent of roughly `-0.2` to `-0.4`) -- still
comfortably `>-1`, so `E[U^k]` for `U~Exp(1)` remains finite (`Γ(1+k)`, finite whenever `k>-1`); no
new divergence risk from this sign flip.

CONFIRMED BUG, FOUND AND FIXED WITHIN THIS SAME TASK (2026-08-05), independent of the exponent-sign
correction above: an earlier version of this function used `Pow = U.^(1-σHat)` directly (treating
the raw exponential draw as if it were already `z_o(ω)`, the same simplification that is harmless
for eq.35 but not for a power moment). At real D=20 data (σHat=2.5), this diverges: `U~Exp(1)` has
density bounded away from 0 at `U=0`, and exponent `1-σHat=-1.5<=-1` is non-integrable there, so
`E[U^(1-σ)]` is genuinely infinite in population and a SINGLE near-zero draw dominates the entire
sample mean (observed directly: `min(U[:,3])=1.3e-5` produced `Pow[:,3]≈2.1e7` at W=5,000, versus a
well-behaved raw CDF moment of the same order as the other origins) -- this, not any solver or
quantile-grid issue, is what made the two-family restriction spuriously infeasible (KNITRO
nStatus=-300) at real D20 scale while D4's synthetic (non-Exp(1)) draws never exposed it. This bug
was about using the WRONG VARIABLE (`U` instead of `z=U^{-μ}`); it is orthogonal to, and was fixed
before, the exponent-SIGN correction above. See
`docs/audits/cm-add-truncated-power-moments-2026-08-05/MASTER.md` for the full incident writeup.
"""
function precalc_common_marginals_cdf(U::AbstractMatrix{Float64}, refIndex1::Int, L::Int;
                                       include_truncated_moment::Bool,
                                       σHat::Union{Nothing,Real} = nothing,
                                       μHat::Union{Nothing,Real} = nothing,
                                       contrasts::Symbol = :anchored,
                                       probs::Union{Nothing,AbstractVector{Float64}} = nothing)
    W, D = size(U)
    @assert 1 <= refIndex1 <= D
    @assert L >= 1
    @assert contrasts in (:anchored, :orthonormal) "contrasts must be :anchored or :orthonormal, got $contrasts"
    include_truncated_moment && @assert(σHat !== nothing,
        "include_truncated_moment=true requires σHat (the fixed baseline-calibrated trade elasticity)")
    include_truncated_moment && @assert(μHat !== nothing,
        "include_truncated_moment=true requires μHat (the fixed baseline-calibrated Frechet shape exponent, μ=1/θ) " *
        "-- eq.36 is built from the Frechet productivity draw z=U^(-μ), not the raw exponential draw U directly")
    # Continuation 13, Section 6: `probs=` lets a caller supply an EXPLICIT probability grid (e.g.
    # nested_quantile_grids.jl's genuinely-nested Q_10/Q_20/Q_50) in place of the default grid --
    # the default path (`probs === nothing`) is byte-for-byte unchanged. Remediation task Part E
    # (finding F13): the default is NOT literally `k/L for k=1:L` -- it is `L` points evenly
    # spaced over [1/L, (L-1)/L] (`range(1/L, (L-1)/L, length=L)`), which at L=50 runs
    # 0.02..0.98 with spacing ~=0.0196 (not exactly 1/L), deliberately excluding p=0 and p=1.
    # This is correct/intended behavior (a CDF contrast at p=1 would be degenerate); only the
    # comment previously mislabeled it "the evenly-spaced k/L grid."
    # 2026-08-05 (user-directed, beyond the original task brief): cutoffs are now the THEORETICAL
    # closed-form Fréchet quantile, not the empirical (order-statistic) sample quantile of the
    # realized draws -- see `theoretical_u_threshold`'s own docstring for the full derivation
    # (why this is exactly `-log(1-p)`, i.e. the closed-form Exp(1) quantile of `U` itself, even
    # though the target quantile level is stated in terms of the FRÉCHET z=U^(-μ)). Applies to
    # BOTH families (eq.35 and eq.36 must keep sharing the same cutoffs, per the task brief).
    # DELIBERATE CONSEQUENCE (disclosed, not a bug): this changes eq.35's own numeric cutoff
    # values relative to old production (which used the empirical `quantile(U[:,refIndex1],...)`)
    # -- see MASTER.md for the full writeup of why this is intentional.
    probs_used = probs === nothing ? collect(range(1 / L, (L - 1) / L, length = L)) : probs
    @assert probs === nothing || length(probs) == L "precalc_common_marginals_cdf: length(probs)=$(length(probs)) != L=$L"
    z = theoretical_u_threshold.(probs_used)
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
        # CDW eq.36: E_F[ z_o'(ω)^(σ-1) · 1{z_o'(ω)<z_l} ], anchored to the reference origin the
        # SAME way eq.35 is (subtract the reference origin's own power-weighted indicator, using
        # the reference's OWN power weight -- NOT the non-reference origin's -- since the power
        # weight is itself origin-and-draw-specific, unlike eq.35's weight-1 indicator).
        # z_x(ω)^(σ-1) = U_x(ω)^(-μ(σ-1)) for every origin x (incl. reference), precomputed once,
        # via the SAME frechet_power_feature the ZC/meanZC restrictions use (cm_meanzc_moments.jl)
        # -- NOT a plain U.^(σHat-1)/U.^(1-σHat) (both would treat the raw exponential draw as if
        # it were already the Fréchet productivity level -- see this function's own docstring
        # "CONFIRMED BUG" note for the full derivation and the divergence it caused at real scale;
        # the exponent SIGN itself -- σ-1, not 1-σ -- was a separate, later user correction, see
        # the docstring's own "corrected 2026-08-05" note).
        #
        # SECOND BUG, FOUND AND FIXED 2026-08-05 (Architecture C follow-up, user-caught): the
        # indicator direction. `theoretical_u_threshold`'s own derivation establishes
        # `1{z_x(ω)<z_ℓ} = 1{U_x(ω) > c}` (c=z[l], the map z=U^{-μ} is DECREASING) -- but this
        # function reused the SAME `U .<= z[l]` convention eq.35 uses. For eq.35 (a plain,
        # constant-weight-1 indicator) that substitution is harmless: `1{U>c}-1{U_ref>c} =
        # -(1{U<=c}-1{U_ref<=c})`, an overall sign flip, and `E_F[X]=0 ⟺ E_F[-X]=0`. For eq.36 the
        # indicator is weighted by the ORIGIN-SPECIFIC `Pow` factor, and that symmetry breaks:
        # `Pow_o*1{U_o<=c}-Pow_ref*1{U_ref<=c} = (Pow_o-Pow_ref) - (Pow_o*1{U_o>c}-Pow_ref*1{U_ref>c})`
        # -- an ADDITIVE BIAS term `(Pow_o-Pow_ref)` that is exactly zero only at the untouched base
        # (uniform-weight) measure (both origins draw from the identical canonical Fréchet
        # distribution there) but is generically NONZERO under the reweighted measure π the inner
        # KNITRO dual solve actually searches over -- confirmed numerically: under a base measure
        # `code≈-literal` (bias≈0 as expected), but under an arbitrary reweighting,
        # `code+literal` matched the predicted bias `E_π[Pow_o]-E_π[Pow_ref]` to machine precision.
        # This, not any real economic infeasibility, is almost certainly what caused the spurious
        # nStatus=-300/-400 results the Architecture C real-data gates hit. Fixed: use `U .> z[l]`
        # for this family's own indicator (both the non-reference and the reference term).
        Pow = frechet_power_feature(U, σHat - 1, Float64(μHat))
        CDF_ref_pow = Matrix{Float64}(undef, W, L)   # 1{U[:,ref] > z[l]} -- REFLECTED vs CDF_ref
        @inbounds for l in 1:L
            @. CDF_ref_pow[:, l] = U[:, refIndex1] > z[l]
        end
        TM_ref = Matrix{Float64}(undef, W, L)
        @inbounds for l in 1:L
            @. TM_ref[:, l] = Pow[:, refIndex1] * CDF_ref_pow[:, l]
        end
        tm_offset = nO * L
        @inbounds for l in 1:L
            for (oi, o) in enumerate(origins)
                @. block[:, oi] = Pow[:, o] * (U[:, o] > z[l]) - TM_ref[:, l]
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
    μHat = include_truncated_moment ? ctx.μHat : nothing
    CM, z, origins = precalc_common_marginals_cdf(ctx.U, refIndex1, L;
        include_truncated_moment = include_truncated_moment, σHat = σHat, μHat = μHat,
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
