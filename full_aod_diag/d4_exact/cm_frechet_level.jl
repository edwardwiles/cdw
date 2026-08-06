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
    frechet_levelpow_target(z_l::Real, k::Real, μHat::Real) -> Float64

2026-08-05 (user-directed): the population target for ONE thresh_old of the new POW-weighted
common-level restriction, `E_{F*}[ z(ω)^k · 1{z(ω)<Z_l} ]`, where `Z_l` is the SAME Fréchet-space
threshold `theoretical_u_threshold`/`z[l]` already implicitly tests for the (plain, CDF-only) level
restriction and for CM's own eq.36 block (`1{U>c} ⟺ 1{z<Z_l}` with `Z_l=c^{-μ}`, `c=z[l]`).

**Deliberately does NOT call `eq36_theoretical_truncated_moment` with a computed `Z_l` -- takes
`z_l` (the U-space cutoff, i.e. this file's/precalc_common_marginals_cdf's own `z[l]`) DIRECTLY.**
Derivation: `eq36_theoretical_truncated_moment(z_ℓ,k,μ) = Γ(1-μk, z_ℓ^(-1/μ))`. Substituting
`z_ℓ=Z_l=z[l]^{-μ}` gives `a = (z[l]^{-μ})^{-1/μ} = z[l]^{(-μ)(-1/μ)} = z[l]` -- the exponents
cancel EXACTLY, so `E_{F*}[z^k·1{z<Z_l}] = Γ(1-μk, z[l])` with `z[l]` used verbatim, no
power/inverse-power round trip (avoids a numerically noisier `z[l]^(-μ)` then `(...)^(-1/μ)` pair
of operations for the identical mathematical result). Verified two independent ways before use in
any restriction/Hessian code: (1) this exact identity by hand, (2) direct Monte Carlo at real
(μHat,σHat) values converging to this closed form as W grows (see this task's own diagnostic
script, not committed -- matches the SAME "verify via MC before trusting" discipline
`eq36_theoretical_truncated_moment`'s own D4 gate already applied to CM's eq.36 target).

Unlike CM's own eq.36 restriction (a reference-origin DIFFERENCE, always zero-target by
construction), the level restriction is an all-origin SUM with a genuinely nonzero absolute
target -- this function supplies exactly that absolute target, mirroring how
`frechet_level_targets` already supplies the plain-CDF level restriction's own nonzero target
`p_l` (a special case of this same construction at `k=0`: `Γ(1,z_l)=e^{-z_l}`... note this is NOT
literally `p_l` at k=0, since the plain level restriction's own target derivation takes a
different, sign-flip-invariant shortcut valid only for k=0 -- see this file's own header comment
and `theoretical_u_threshold`'s docstring for why k=0's target is legitimately just `p_l`, not
`Γ(1,z_l)`; the two are NOT interchangeable formulas, they happen to answer the same k=0 question
via different valid routes).

**Verification/reference use only for the CDF (k=0) case; for k=σ-1 THIS closed form IS the actual
restriction target** (unlike `eq36_theoretical_truncated_moment`, which per its own docstring is
never used as a live restriction target for CM's own zero-mean eq.36 block) -- the level
restriction has always used an absolute theoretical target (`p_l`), and `levelpow` needs its own
absolute target the same way, computed once, theta-independent, exactly like `frechet_level_targets`
itself.
"""
function frechet_levelpow_target(z_l::Real, k::Real, μHat::Real)
    return gamma(1 - μHat * k, z_l)
end

"""
    frechet_levelpow_targets(D, L, z::Vector{Float64}, σHat, μHat) -> Vector{Float64}

`target_l = sqrt(D) * frechet_levelpow_target(z[l], σHat-1, μHat)` -- the POW-level restriction's
own analog of `frechet_level_targets`, using the SAME `(u'1)=sqrt(D)` scaling and the SAME `z`
cutoff vector `precalc_common_marginals_cdf`/`build_cm_frechet_level_augmented_obj` already build
(byte-identical cutoffs to the plain level block and to CM's own eq.36 block -- required for the
combined restriction to test the coherent SAME set of Fréchet-quantile events across every family).
"""
function frechet_levelpow_targets(D::Int, L::Int, z::Vector{Float64}, σHat::Real, μHat::Real)
    @assert length(z) == L "frechet_levelpow_targets: length(z)=$(length(z)) != L=$L"
    k = σHat - 1
    return sqrt(D) .* [frechet_levelpow_target(z[l], k, μHat) for l in 1:L]
end
# CMFrechetExtension is defined in cm_frechet_hessian.jl (not here) -- every current include-list
# ordering (D=4/D=20 equivalence gates, smoke scripts) loads cm_frechet_hessian.jl either before
# or independently of this file, but the reverse is not true for the two equivalence-gate scripts,
# which load cm_frechet_hessian.jl BEFORE cm_frechet_level.jl.

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
    precalc_frechet_levelpow_dense(U, z, D, Pow, targets) -> Matrix{Float64}  (W x L)

DENSE REFERENCE builder for the POW-weighted common-level restriction (`levelpow`), the
`k=σ-1`-weighted analog of `precalc_frechet_level_dense`. `level_pow[:,l] =
(1/sqrt(D))*sum_{o=1}^D z_o(ω)^(σ-1)·1{z_o(ω)<Z_l} - targets[l]`.

**Indicator direction is `U[s,o] > z[l]` (strict `>`), NOT `<=` like the plain level block.** The
plain level restriction gets away with `<=` only because a weight-1 indicator's restriction is
invariant to an overall sign flip (`E_F[X]=0 ⟺ E_F[-X]=0`, see `theoretical_u_threshold`'s own
docstring) -- that invariance does NOT hold once the indicator carries an origin-and-draw-specific
`Pow` weight (exactly the bug CM's own eq.36 restriction hit once already, section 17 of
docs/... -- fixed there via `U>c`, not `U<=c`; same fix required here for the identical reason).
`Pow` is the SAME `W x D` `z^(σ-1)=U^{-μ(σ-1)}` matrix CM's own eq.36 block already computes
(`frechet_power_feature(U, σHat-1, μHat)`) -- passed in, not recomputed.
"""
function precalc_frechet_levelpow_dense(U::AbstractMatrix{Float64}, z::Vector{Float64}, D::Int,
                                         Pow::AbstractMatrix{Float64}, targets::Vector{Float64})
    W = size(U, 1)
    L = length(z)
    @assert length(targets) == L
    @assert size(Pow) == (W, D) "precalc_frechet_levelpow_dense: size(Pow)=$(size(Pow)) != (W,D)=($W,$D)"
    invsqrtD = 1.0 / sqrt(D)
    level = Matrix{Float64}(undef, W, L)
    @inbounds for l in 1:L
        zl = z[l]
        for s in 1:W
            acc = 0.0
            for o in 1:D
                acc += U[s, o] > zl ? Pow[s, o] : 0.0
            end
            level[s, l] = invsqrtD * acc - targets[l]
        end
    end
    return level
end

"""
    fill_frechet_levelpow_columns_from_bins!(Gdest, Bidx, D, L, Pow, targets; chunk_size=2000)

Architecture-B analogue of `fill_frechet_level_columns_from_bins!` for the POW-weighted level
block. `Gdest[s,l] = (1/sqrt(D))*sum_{o=1}^D Pow[s,o]*(Bidx[s,o]>l) - targets[l]` -- indicator
`Bidx[s,o]>l` (reflected, NOT `<=l`), matching `precalc_frechet_levelpow_dense`'s own U-space `>`
convention (`Bidx[s,o]` is the bin index of `U[s,o]` against the ascending cutoffs `z`, so
`Bidx[s,o]<=l ⟺ U[s,o]<=z[l]`, hence `Bidx[s,o]>l ⟺ U[s,o]>z[l]`, the SAME reflection CM's own
`fill_cm_columns_from_bins!`-adjacent eq.36 bin logic already uses for this same reason).
"""
function fill_frechet_levelpow_columns_from_bins!(Gdest::AbstractMatrix{Float64}, Bidx::AbstractMatrix{Int},
                                                   D::Int, L::Int, Pow::AbstractMatrix{Float64},
                                                   targets::Vector{Float64}; chunk_size::Int = 2000)
    W = size(Gdest, 1)
    @assert size(Gdest, 2) == L
    @assert length(targets) == L
    @assert size(Pow) == (W, D) "fill_frechet_levelpow_columns_from_bins!: size(Pow)=$(size(Pow)) != (W,D)=($W,$D)"
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
                    acc += Bidx[s, o] > l ? Pow[s, o] : 0.0
                end
                Gdest[s, l] = invsqrtD * acc - tl
            end
        end
        start = stop + 1
    end
    return nothing
end

"""
    build_cm_frechet_level_augmented_obj(ctx, CS; L, include_truncated_moment, contrasts=:anchored,
                                          probs=nothing, refIndex1=ctx.γ.refIndex1) -> NamedTuple

DENSE reference construction (Architecture A -- validation-oriented, mirrors
`build_cm_augmented_obj`'s own shape exactly, not the hot production path). Builds the CM block
EXACTLY as `precalc_common_marginals_cdf` does (unchanged, reused), builds the level block via
`precalc_frechet_level_dense`, concatenates `[CM level]` horizontally into ONE `(W x (ncm+L))`
matrix, and hands it to `wrap_moments_with_cm` UNCHANGED (that function is generic on the supplied
moment matrix -- no modification needed, confirming task §5's "reuse the exact CM forward and
transpose operators"). Column layout: CM block first (`(D-1)*L` or `2*(D-1)*L` columns,
threshold-major, IDENTICAL to flexible CM's own layout), level block last (`L` or `2*L` columns) --
see `docs/COMMON_FRECHET_CM_DRIVER_PORT_2026-07-25.md` for the full machine-readable layout table
(Part II §6 of the task).

2026-08-05 (paired-basis-preconditioning pilot, common-Fréchet extension): `include_truncated_moment`
is now a REQUIRED kwarg (no default, per repo rule -- was previously hardcoded `false` here, a
deliberate single-family carve-out CLAUDE.md/CM_CURRENT_SINGLE_BLOCK_SOURCE_MAP.md documented as
"audit only, not rewired"). `true` imposes BOTH families on the CM sub-block (eq.35+eq.36, reusing
`precalc_common_marginals_cdf`'s own already-fixed two-family construction unchanged) AND adds a
NEW `levelpow` block (`precalc_frechet_levelpow_dense`, the POW-weighted analog of the existing
plain `level` block) -- `E_{F*}[z(ω)^{σ-1}·1{z(ω)<Z_l}]`, closed-form target via
`frechet_levelpow_targets`. `false` reproduces the pre-existing CDF-only, level-only behavior
byte-for-byte. Column layout when `true`: `[CM_cdf ((D-1)L) | CM_pow ((D-1)L) | level (L) |
levelpow (L)]` -- CM's own two sub-families stay adjacent (matching flexible CM's own layout
exactly, so the shared `fill_cm_HCC!`/`archC_hess_cb_builder` machinery needs no change to find
them), levelpow placed immediately after level (mirroring how CM_pow follows CM_cdf).

Returns the same-shaped NamedTuple as `build_cm_augmented_obj`, plus `level_targets`, `probs`, and
`ncm_cm`/`ncm_level` (the two block sizes, EACH already including both families when
`include_truncated_moment=true` -- i.e. `ncm_cm=2*(D-1)*L`, `ncm_level=2*L` -- callers that need the
within-family split use the new `ncm_cdf`/`ncm_pow` (CM side) and `ncm_level_cdf`/`ncm_level_pow`
(level side) fields instead) so callers can locate any block within the combined
`ncm = ncm_cm + ncm_level` columns.
"""
function build_cm_frechet_level_augmented_obj(ctx, CS; L::Int, include_truncated_moment::Bool,
                                               contrasts::Symbol = :anchored,
                                               refIndex1::Int = ctx.γ.refIndex1,
                                               probs::Union{Nothing,AbstractVector{Float64}} = nothing)
    obj0 = ctx.obj
    ncore = obj0.d
    σHat = include_truncated_moment ? ctx.σ : nothing
    μHat = include_truncated_moment ? ctx.μHat : nothing
    CM, z, origins = precalc_common_marginals_cdf(ctx.U, refIndex1, L; include_truncated_moment = include_truncated_moment,
                                                   σHat = σHat, μHat = μHat, contrasts = contrasts, probs = probs)
    ncm_cm = size(CM, 2)
    @assert ncm_cm == n_cm_moments(ctx.D, L; include_truncated_moment = include_truncated_moment)
    ncm_cdf = length(origins) * L
    ncm_pow = include_truncated_moment ? ncm_cdf : 0

    level_probs = frechet_level_probs(L; probs = probs)
    level_targets = frechet_level_targets(ctx.D, L; probs = level_probs)
    LEVEL_cdf = precalc_frechet_level_dense(ctx.U, z, ctx.D, level_targets)
    ncm_level_cdf = size(LEVEL_cdf, 2)
    @assert ncm_level_cdf == L

    if include_truncated_moment
        Pow = frechet_power_feature(ctx.U, σHat - 1, Float64(μHat))
        levelpow_targets = frechet_levelpow_targets(ctx.D, L, z, σHat, μHat)
        LEVEL_pow = precalc_frechet_levelpow_dense(ctx.U, z, ctx.D, Pow, levelpow_targets)
        ncm_level_pow = size(LEVEL_pow, 2)
        @assert ncm_level_pow == L
        LEVEL = hcat(LEVEL_cdf, LEVEL_pow)
        level_targets_full = vcat(level_targets, levelpow_targets)
    else
        levelpow_targets = Float64[]
        LEVEL = LEVEL_cdf
        ncm_level_pow = 0
        level_targets_full = level_targets
    end
    ncm_level = size(LEVEL, 2)
    @assert ncm_level == ncm_level_cdf + ncm_level_pow

    CMF = hcat(CM, LEVEL)
    ncm = ncm_cm + ncm_level
    @assert ncm == (include_truncated_moment ? 2 : 1) * ctx.D * L   # task §3 (single-family): D*L restrictions total

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
            ncm_cm = ncm_cm, ncm_level = ncm_level, ncm_cdf = ncm_cdf, ncm_pow = ncm_pow,
            ncm_level_cdf = ncm_level_cdf, ncm_level_pow = ncm_level_pow,
            L = L, contrasts = contrasts,
            include_truncated_moment = include_truncated_moment, refIndex1 = refIndex1,
            level_targets = level_targets_full, level_targets_cdf = level_targets,
            level_targets_pow = levelpow_targets, level_probs = level_probs,
            n_families = include_truncated_moment ? 2 : 1,
            marginal_restriction = :common_frechet)
end

# ================================================================================================
# Architecture-B production path (real production driver's own moment-construction speed, no
# persistent dense W x ncm matrix beyond the throwaway one used for z/origins/target bookkeeping).
# Mirrors wrap_moments_with_cm_archB / build_cm_production_context (cm_production_bundle.jl)
# EXACTLY, adding only the level-block fill call. Hessian side: for now (Part II wiring) this only
# supports `cm_hessian_backend=:dense_reference` (Architecture A -- generic, differentiates the
# augmented obj directly via `_callbackEvalH_inner_profiled!`, unchanged, no level-block-specific
# code needed there). `:structured` (winner-pair-backed Architecture C) requires extending
# CMBinHessCtx for the level block -- Part III, not yet done; guarded against below rather than
# silently producing a wrong Hessian.
# ================================================================================================

isdefined(Main, :NO_DENSE_G_COUNTERS) || include(joinpath(@__DIR__, "no_dense_g_counters.jl"))   # default-flips task (2026-07-27), Task C: record_dense_frechet_g! used below
isdefined(Main, :OperatorPsiBundle) || include(joinpath(@__DIR__, "operator_psi_bundle.jl"))   # true no-H operator bundle (2026-07-28 continuation): OperatorPsiBundle/prime_operator!, load-bearing for build_cm_frechet_production_context below

"""
    wrap_moments_with_cm_frechet_archB(core_moments!, ncore_full, Bidx, origins, refIndex1, L, R, D,
                                        level_targets, ctx; chunk_size, use_compressed_core, core_cf_ref)

Architecture-B analogue of `wrap_moments_with_cm_archB` (cm_hessian_architectures.jl) for the
CM-plus-level restriction: identical core-column construction (same compressed-factual / dense
fallback discipline, same `core_cf_ref` publishing for a future Part III Hessian callback), CM
columns via the UNCHANGED `fill_cm_columns_from_bins!`, plus the level columns via
`fill_frechet_level_columns_from_bins!` appended immediately after. Column layout:
`[core (pregrav) | CM ((D-1)*L) | level (L) | gravity]` -- level block placed after CM, before the
sole trailing gravity column, matching `wrap_moments_with_cm`'s own splicing convention.
"""

function wrap_moments_with_cm_frechet_archB(core_moments!::Function, ncore_full::Int,
                                             Bidx::Matrix{Int}, origins::Vector{Int}, refIndex1::Int, L::Int,
                                             R::Union{Nothing,Matrix{Float64}}, D::Int, level_targets::Vector{Float64},
                                             ctx; chunk_size::Int = 2000, use_compressed_core::Bool = true,
                                             core_cf_ref::Ref{Any} = Ref{Any}(nothing),
                                             skip_fill::Bool = false)
                                             # RE-INTRODUCED then kept UNUSED in production 2026-07-27/28
                                             # (docs/GOAL10_SKIP_CM_FILL_REF_REMOVAL_2026-07-27.md): a
                                             # skip variant of this kwarg existed originally (Phase 5.2,
                                             # 2026-07-26), was found unsafe and removed (commit
                                             # 5fd6347, 2026-07-27 10:52 -- nStatus=-400, see
                                             # docs/COMMON_FRECHET_FG_D20_FINAL_GATE_2026-07-27.md).
                                             # Re-tested on the hypothesis that the winner-bin H_E,level
                                             # Hessian path (winner_pair_cross_hessian_colsum!/_esum!,
                                             # commits e3bce93/d458702, a git DESCENDANT of the bugfix by
                                             # ~7.5h) made the skip safe again -- D=4 multi-point testing
                                             # supported this, but a real D=20/W=80,000 re-test then
                                             # DISPROVED it: both tested non-calibration points reproduced
                                             # the exact nStatus=-400 failure with the skip enabled. The
                                             # `skip_fill` PARAMETER stays (shared call signature with the
                                             # other CM-family wrappers, and `cctx.moments_skip!` still
                                             # exists as a built-but-unused closure for any future
                                             # re-investigation), but common-Fréchet's own two call sites
                                             # (archC_frechet_base_state/archC_frechet_verified_state,
                                             # cm_frechet_cplus.jl) always pass `skip_fill=false` now --
                                             # see those functions' own HISTORY comments for the real
                                             # D=20 evidence. Do not re-enable without root-causing the
                                             # actual dependency first.
    pregrav = ncore_full - 1
    nO = length(origins)
    Gtmp_cache = Ref{Matrix{Float64}}(Matrix{Float64}(undef, 0, 0))
    cs = min(chunk_size, size(Bidx, 1))
    prod_scratch = R === nothing ? nothing : Matrix{Float64}(undef, cs, nO)
    return function (K, G, θ, U, obj)
        n = size(U, 1)
        if size(Gtmp_cache[], 1) != n
            Gtmp_cache[] = Matrix{Float64}(undef, n, ncore_full)
        end
        Gtmp = Gtmp_cache[]
        if use_compressed_core
            # No-moments/no-composite-G task (2026-07-28): check_ties=false -- see the identical
            # change/rationale in cm_hessian_architectures.jl::wrap_moments_with_cm_archB.
            cf = cf_build(θ, ctx; check_ties = false)   # Phase E remediation (2026-07-26): reuses ctx.cf_workspace when attached
            materialize_dense_factual_structured!(@view(Gtmp[:, 1:pregrav]), cf)
            grav_raw = compressed_gravity_raw(θ, ctx)
            fill_gravity_column_into!(@view(Gtmp[:, ncore_full]), grav_raw, ctx, ncore_full)
            fill_K_directgp!(K, θ, ctx)
            core_cf_ref[] = cf
            # No-moments/no-composite-G task (2026-07-28): a live attempt to also gate this dense
            # economic fill behind `!skip_fill` was REVERTED after an unexplained regression was
            # caught in the flexible-CM analogue (test_shared_core_hessian_d4_gates.jl, H_EE
            # mismatch) -- see cm_hessian_architectures.jl::wrap_moments_with_cm_archB's identical
            # note. Not attempted further here since common-Fréchet's own `skip_fill_safe` isn't
            # re-enabled in production anyway (separate, deliberately out-of-scope item).
        else
            core_moments!(K, Gtmp, θ, U, obj)
            core_cf_ref[] = :compressed_state_unavailable
        end
        @views G[:, 1:pregrav] .= Gtmp[:, 1:pregrav]
        @views G[:, end] .= Gtmp[:, end]
        if !skip_fill
            cm_cols = pregrav + 1 : pregrav + L * nO
            fill_cm_columns_from_bins!(@view(G[:, cm_cols]), Bidx, origins, refIndex1, L, R;
                                        chunk_size = chunk_size, prod_scratch = prod_scratch)
            level_cols = pregrav + L * nO + 1 : pregrav + L * nO + L
            fill_frechet_level_columns_from_bins!(@view(G[:, level_cols]), Bidx, D, L, level_targets;
                                                   chunk_size = chunk_size)
            record_dense_frechet_g!()   # fires exactly when the dense CM/level-column fill actually
            # executes -- 0 for the `skip_fill=true` closure variant (installed as cctx.moments_skip!,
            # used only when archC_frechet_base_state's/archC_frechet_verified_state's own skip_fill_safe
            # is true), nonzero for the `skip_fill=false` variant (every other case).
        end
        return nothing
    end
end

"""
    build_cm_frechet_production_context(ctx, CS; L, contrasts=:anchored, probs=nothing,
                                         use_compressed_core=true, cm_hessian_backend=:dense_reference)
        -> (ctx_cm=..., aug=..., hess_cb_builder=...)

Production entry point for the `:common_frechet` marginal restriction mode -- Architecture-B
moment construction (real production speed), reusing `build_cm_frechet_level_augmented_obj`'s dense
`z`/`origins`/`level_targets` bookkeeping (computed once, theta-independent) but building the actual
per-call `G` via bin-lookup, exactly as `build_cm_production_context` does for plain flexible CM.

`cm_hessian_backend`: `:dense_reference` (Architecture A, generic dense differentiation of the
augmented obj -- `archA_hess_cb_builder`, unchanged) or `:structured` (Architecture C, winner-pair
`H_EE` + the level-block extension in `cm_frechet_hessian.jl` -- Part III). Both reuse the UNCHANGED
`build_cm_bin_ctx` (cm_hessian_architectures.jl) for `:structured`, since that function is already
generic on `aug.ncm` (it sizes `Hfull`/scratch from `aug.ncm` alone, with no assumption about what
the extra columns beyond the core mean) -- only the Hessian-FILL step needed a level-aware version.
"""
function build_cm_frechet_production_context(ctx, CS; L::Int, include_truncated_moment::Bool,
                                              contrasts::Symbol = :anchored,
                                              probs::Union{Nothing,AbstractVector{Float64}} = nothing,
                                              use_compressed_core::Bool = true,
                                              cm_hessian_backend::Symbol = :dense_reference,
                                              # No-moments/no-composite-G task (2026-07-28): was
                                              # unconditionally hardcoded `false` below with no
                                              # override -- meaning every real production run took
                                              # the SERIAL `hessian_cm_frechet_structured!` branch
                                              # despite `archC_frechet_hess_cb_builder`'s own comments
                                              # (and `build_cm_bin_ctx`'s own default of `true`)
                                              # describing the threaded branch as "the production
                                              # default". The threaded variant
                                              # (`hessian_cm_frechet_structured_v2!`) is independently
                                              # validated against the serial one at real D=20/W=80,000
                                              # (`test_cm_frechet_threaded_hessian_gates.jl`, PASS per
                                              # docs/COMMON_FRECHET_WINNER_AWARE_HER_RELEASE_2026-07-27.md)
                                              # -- default flipped to `true` here to match flexible-CM's
                                              # own default and actually deliver the documented speedup.
                                              threaded_bins::Bool = true,
                                              inner_fg_backend::Symbol = CM_FRECHET_INNER_FG_BACKEND_DEFAULT[],   # Phase
                                              # 5.2 remediation (2026-07-26): :dense_reference (default
                                              # until gated) | :cm_frechet_lookup (cm_frechet_lookup_
                                              # kernels.jl/cm_frechet_lookup_production.jl). Only
                                              # reachable when cm_hessian_backend=:structured (needs a
                                              # real cctx -- see the check below).
                                              cm_cross_hessian_backend::Symbol = CM_FRECHET_CROSS_HESSIAN_BACKEND_DEFAULT[],   # winner-aware
                                              # H_ER phase (2026-07-27), Section 3: pass-through to build_cm_bin_ctx --
                                              # :dense_reference (default, unchanged until this family's own gates
                                              # pass) | :winner_bin (winner_pair_cross_hessian.jl). Deliberately reads
                                              # a SEPARATE Ref from flexible-CM's own CM_CROSS_HESSIAN_BACKEND_DEFAULT
                                              # -- see that Ref's own docstring (core_exact_hessian.jl).
                                              moment_representation::Symbol = :operator)   # true no-H operator
                                              # bundle -- flipped to match flexible-CM's own default (2026-07-29,
                                              # docs/FRECHET_OPERATOR_DEFAULT_INVESTIGATION_2026-07-29.md). The prior
                                              # :dense_reference default here cited TWO real, reproduced nStatus=-400
                                              # failures (archC_frechet_base_state's HISTORY comment) as grounds for
                                              # caution -- but that history is about `skip_cm_fill_ref`, a DIFFERENT
                                              # mechanism (leaving a still-DENSE bundle's CM/level columns stale while
                                              # claiming nothing reads them; that skip remains permanently reverted,
                                              # unrelated to this flag). `:operator` (OperatorPsiBundle) has no dense
                                              # H field at all -- any real remaining dependency would hard-crash, not
                                              # silently corrupt. Investigated 2026-07-29: real D=20/W=80,000, both
                                              # contrasts, at the EXACT non-calibration outer points that reproduced
                                              # the original failure (near_delta1_perturbed, hard_point_x1.01) --
                                              # BOTH the full inner solve (archC_frechet_base_state) and the full
                                              # outer gradient (cm_frechet_production_gradient_cplus) agree with
                                              # :dense_reference to 0.000e+00 in all 12 tested cells. See that doc for
                                              # the full investigation; :dense_reference remains available as an
                                              # explicit opt-in reference/debug mode.
    cm_hessian_backend in (:dense_reference, :structured) ||
        error("build_cm_frechet_production_context: cm_hessian_backend must be :dense_reference or :structured, got $cm_hessian_backend")
    inner_fg_backend in (:dense_reference, :cm_frechet_lookup) ||
        error("build_cm_frechet_production_context: inner_fg_backend must be :dense_reference or :cm_frechet_lookup, got $inner_fg_backend")
    inner_fg_backend == :cm_frechet_lookup && cm_hessian_backend != :structured &&
        error("build_cm_frechet_production_context: inner_fg_backend=:cm_frechet_lookup requires cm_hessian_backend=:structured (needs a real CMBinHessCtx)")
    moment_representation in (:operator, :dense_reference) ||
        error("build_cm_frechet_production_context: moment_representation must be :operator or :dense_reference, got :$moment_representation")
    moment_representation === :operator && inner_fg_backend !== :cm_frechet_lookup &&
        error("build_cm_frechet_production_context: moment_representation=:operator requires inner_fg_backend=:cm_frechet_lookup")
    # 2026-08-05 (paired-basis-preconditioning pilot, common-Fréchet two-family extension):
    # `wrap_moments_with_cm_frechet_archB` (Architecture B) is still single-family-only (mirrors
    # flexible CM's own `build_cm_production_context` guard: "use_archB_moments=true is
    # incompatible with include_truncated_moment=true") -- `include_truncated_moment=true` still
    # requires `use_compressed_core=false` when `moment_representation=:dense_reference` for this
    # reason.
    #
    # 2026-08-06 (levelpow kernel task): `CMFrechetLookupState`/`:operator` IS now extended for the
    # two-family CM/levelpow blocks (cm_frechet_lookup_kernels.jl's own `Pow`/`levelpow_targets`
    # fields, `_verify_inner_solution_operator_cm_core`'s own `fam2_level` branch,
    # operator_verification.jl) -- `:operator` is a real option here now, not a hard refusal.
    # `cctx.n_families`/`cctx.Pow` are already built generically by `build_cm_bin_ctx`
    # (cm_hessian_architectures.jl) from `aug.n_families`, exactly the same shared path
    # flexible-CM/CM+ZC already use -- no separate Pow computation needed for this family.
    if include_truncated_moment
        moment_representation in (:dense_reference, :operator) ||
            error("build_cm_frechet_production_context: include_truncated_moment=true requires " *
                  "moment_representation=:dense_reference or :operator, got :$moment_representation")
        moment_representation === :dense_reference && use_compressed_core &&
            error("build_cm_frechet_production_context: include_truncated_moment=true with " *
                  "moment_representation=:dense_reference requires use_compressed_core=false " *
                  "(wrap_moments_with_cm_frechet_archB's Architecture-B fill is a pure single-family " *
                  "reconstruction, same reason flexible CM's own build_cm_production_context refuses " *
                  "use_archB_moments=true here) -- the dense aug.obj_cm splice is used directly " *
                  "instead, no Architecture-B fast path.")
    end
    isdefined(Main, :record_cm_feature_context_build!) && record_cm_feature_context_build!()   # Phase 3 (2026-07-26): CM feature immutability counters

    aug = build_cm_frechet_level_augmented_obj(ctx, CS; L = L, include_truncated_moment = include_truncated_moment,
                                                contrasts = contrasts, probs = probs)
    D = ctx.D
    refIndex1 = aug.refIndex1
    R = contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing
    Bidx = Int.(compute_bin_indices(ctx.U, aug.z))

    core_cf_ref = Ref{Any}(nothing)
    obj0 = aug.obj_cm
    moments_archB_skip! = nothing
    if include_truncated_moment && moment_representation === :dense_reference
        # Two-family, dense: obj0 (aug.obj_cm, from build_cm_frechet_level_augmented_obj) is
        # ALREADY the correct dense two-family+levelpow splice (wrap_moments_with_cm, generic) --
        # use it directly, no Architecture-B wrapping.
        obj_cm = obj0
    elseif moment_representation === :dense_reference   # single-family only reaches here (see condition above)
        # Re-tested and RE-ENABLED 2026-07-27 (see wrap_moments_with_cm_frechet_archB's own header
        # comment for the full chronology/rationale): build BOTH the always-fill closure
        # (`moments_archB!`, installed as `obj_cm.moments!`, used by every non-skip path AND by
        # archC_frechet_verified_state's own inner solve) and the skip variant (`moments_archB_skip!`,
        # installed on `cctx.moments_skip!`, used ONLY when archC_frechet_base_state/
        # archC_frechet_verified_state explicitly thread `skip_fill=true` through under production
        # defaults) -- mirrors build_cm_production_context's identical dual-closure pattern exactly.
        moments_archB! = wrap_moments_with_cm_frechet_archB(ctx.obj.moments!, aug.ncore, Bidx, aug.origins,
            refIndex1, L, R, D, aug.level_targets, ctx; use_compressed_core = use_compressed_core, core_cf_ref = core_cf_ref,
            skip_fill = false)
        moments_archB_skip! = wrap_moments_with_cm_frechet_archB(ctx.obj.moments!, aug.ncore, Bidx, aug.origins,
            refIndex1, L, R, D, aug.level_targets, ctx; use_compressed_core = use_compressed_core, core_cf_ref = core_cf_ref,
            skip_fill = true)

        obj_cm = CS.PsiObjectiveBundleImplicit(δ = obj0.δ, find_smallest = obj0.find_smallest,
            γ = obj0.γ, (moments!) = moments_archB!, moments_jacobian! = error,
            d = obj0.d, outer_constr_index = obj0.outer_constr_index,
            inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
            l = obj0.l, U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
            use_cached_x = obj0.use_cached_x, threshold_state = obj0.threshold_state,
            outer_loop_opt = obj0.outer_loop_opt, inner_loop_opt = obj0.inner_loop_opt,
            needs_outer_moment_jacobian = obj0.needs_outer_moment_jacobian)
    else   # :operator -- true no-H production bundle, task Part A, explicit opt-in for this family
        obj_cm = OperatorPsiBundle(δ = obj0.δ, find_smallest = obj0.find_smallest,
            γ = obj0.γ, l = obj0.l, outer_constr_index = obj0.outer_constr_index,
            inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
            U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
            use_cached_x = obj0.use_cached_x, threshold_state = obj0.threshold_state,
            inner_loop_opt = obj0.inner_loop_opt)
    end

    ctx_cm = merge(ctx, (obj = obj_cm,))
    aug = merge(aug, (core_cf_ref = core_cf_ref, moments_skip! = moments_archB_skip!, Bidx = Bidx))
    bins = cm_bin_indices_for(ctx, aug)   # top-level field, matches build_cm_production_context's own pcx shape

    cctx = nothing
    if cm_hessian_backend === :structured
        # 2026-08-05: `_fill_frechet_level_blocks!` (cm_frechet_hessian.jl) two-family extension
        # (H_CM(pow),level / H_CM(cdf,pow),levelpow / H_level(pow),levelpow) is now implemented --
        # see that function's own docstring/derivation. PENDING: independent D4 ForwardDiff/dense-
        # reference verification gate (test_frechet_hessian_structured_vs_dense_d4_twofamily_2026-08-05.jl)
        # before this is trusted for real production use -- do not remove this comment or treat
        # include_truncated_moment=true + :structured as production-ready until that gate is green.
        # inner_fg_backend now genuinely selects the FG callback (Phase 5.2, 2026-07-26) -- previously
        # PINNED to :dense_reference unconditionally here because archC_frechet_base_state/
        # archC_frechet_verified_state (cm_frechet_cplus.jl) never read cctx.inner_fg_backend at all;
        # those two functions now dispatch on it, mirroring plain-CM's archC_base_state/
        # archC_verified_state exactly.
        cctx = build_cm_bin_ctx(ctx, aug; threaded_bins = threaded_bins, inner_fg_backend = inner_fg_backend,
            cm_cross_hessian_backend = cm_cross_hessian_backend)
        # harmonization task (2026-07-28): archC_frechet_hess_cb_builder's public signature is
        # unchanged (still bare level_targets) -- it now internally resolves and caches a
        # persistent CMFrechetExtension on cctx.frechet_ext_cache (_resolve_frechet_ext!,
        # cm_frechet_hessian.jl), so this call site needs no change beyond this comment.
        hess_cb_builder = _obj -> archC_frechet_hess_cb_builder(cctx, aug.level_targets)
        aug = merge(aug, (cctx = cctx,))
    else
        hess_cb_builder = archA_hess_cb_builder
    end
    return (ctx_cm = ctx_cm, aug = aug, bins = bins, cctx = cctx, hess_cb_builder = hess_cb_builder)
end

"""
    print_frechet_startup_manifest(cfg, D::Int, L::Int)

Task Part II §6's required startup manifest, machine-readable and printed once per driver launch
when `cfg.marginal_restriction === :common_frechet`. `frechet_level_count = 1` is PER THRESHOLD (one
level column added per threshold, vs `cm_contrast_count = D-1` CM columns per threshold) -- the
two multiply out to `total_marginal_moments = D*L` exactly (task §3's DL-restrictions claim).
No-op (returns without printing) when `cfg.marginal_restriction !== :common_frechet`, so callers can
call this unconditionally at startup without an external guard. `cfg` is untyped (not `::CMConfig`)
so this file has no include-order dependency on `cm_config.jl` (which itself depends on this file's
`build_cm_frechet_production_context`) -- the two files close a small mutual-reference cycle that
resolves fine in Julia as long as neither uses the other's type at PARSE time, only at call time.
"""
function print_frechet_startup_manifest(cfg, D::Int, L::Int)
    cfg.marginal_restriction === :common_frechet || return nothing
    println("marginal_restriction = common_frechet")
    println("frechet_feature_set = cdf_only")
    println("frechet_basis = cm_contrasts_plus_common_level")
    println("frechet_grid_size = $L")
    println("cm_contrast_count = $(D - 1)")
    println("frechet_level_count = 1")
    println("total_marginal_moments = $(D * L)")
    println("cm_contrasts = $(cfg.contrasts)")
    println("core_hessian_backend = $(cfg.cm_hessian_backend === :dense_reference ? :dense_reference : :exact_winner_pair_parallel)")
    return nothing
end
