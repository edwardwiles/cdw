# ============================================================================
# CM+mean(+ZC)-aware Lfix/composite-gradient outer path, exact analog of
# lfix_cm_aware.jl for the mean/pair moment block. PURELY ADDITIVE:
# lfix_incremental.jl / composite_gradient.jl / lfix_cm_aware.jl are used
# completely UNCHANGED.
#
# Why this works by the same argument lfix_cm_aware.jl documents: the
# (g,zfree) outer-gradient cache (`build_lfix_base_cache`) only ever indexes
# `base.λstar[1:D^2]` and the counterfactual tail check `oci-1 >= D^2+1`
# (always true once ANY extra inner moment block is appended) -- so it
# silently and correctly ignores the mean/pair λ tail exactly as it already
# ignores the CM tail. `meanzc_fixed_contribution` computes the ignored term
# and folds it back into q0 via `with_q0` (lfix_cm_aware.jl), stacked on TOP
# of the (already CM-aware) q0 that `build_lfix_base_cache_cm` produces --
# both corrections are simple subtractions from the same q0 vector, so they
# compose additively regardless of order.
#
# Unlike CM's finite-grid block, the mean/pair columns are RAW dense columns
# (not bin-indexed step functions), so their fixed contribution is a single
# BLAS matrix-vector product -- no suffix-sum lookup trick needed.
# ============================================================================

"""
    meanzc_fixed_contribution(base::BaseDualState, aug) -> Vector{Float64}

`out[s] = λ_mean*'·(Z_s - ν) + λ_pair*'·(Zpair_s - ν²)` for every draw `s`
(the mean-block term only, if `aug.n_pair == 0`). O(W*(D+n_pair)), one or two
BLAS `gemv!`-equivalent calls (`Matrix * Vector`) plus a `Λ_sum` scalar
correction. `aug` must be a `build_cm_meanzc_augmented_obj` result and `base`
must have been solved against `aug.obj_cm` (same precondition
`cm_fixed_contribution` states for its own `aug`/`base` pair) -- checked by
the same length assert pattern.
"""
function meanzc_fixed_contribution(base::BaseDualState, aug)
    ncore_econ = aug.ncore_econ; n_mean = aug.n_mean; n_pair = aug.n_pair
    @assert length(base.λstar) >= ncore_econ - 1 + n_mean + n_pair "base.λstar too short for aug's (ncore_econ,n_mean,n_pair) -- was base solved against aug.obj_cm?"
    ν = aug.nu_ref[]
    λ_mean = @view base.λstar[ncore_econ:ncore_econ+n_mean-1]
    # λ_mean'·(Z_s - ν) = (Zraw*λ_mean)[s] - ν*sum(λ_mean)
    out = aug.Zraw * λ_mean
    out .-= ν * sum(λ_mean)
    if n_pair > 0
        λ_pair = @view base.λstar[ncore_econ+n_mean:ncore_econ+n_mean+n_pair-1]
        out .+= aug.Zpairraw * λ_pair
        out .-= ν^2 * sum(λ_pair)
    end
    return out
end

"""
    build_lfix_base_cache_cm_meanzc(x_free0, ctx_cm, base, ctx, aug, bins; validate_dense=false) -> LFixBaseCache

CM+mean(+ZC)-aware analog of `build_lfix_base_cache_cm`. `aug` here must be a
`build_cm_meanzc_augmented_obj` result (carries `ncore_econ`, `Zraw`,
`Zpairraw`, `nu_ref`, `n_mean`, `n_pair` -- fields `build_cm_augmented_obj`'s
plain CM `aug` does NOT have). The CM-grid contribution is NOT re-added here
-- because `wrap_moments_with_cm_meanzc` builds its OWN CM-grid block
directly (not by composing `wrap_moments_with_cm`), the CM columns and the
mean/pair columns are siblings under the SAME `aug`, so a single combined
correction is applied here using `aug.ncore` (the pre-mean/pair/CM economic
count, same as `aug.ncore_econ`) for the CM slice bounds and the mean/pair
slice bounds documented in `meanzc_fixed_contribution` -- both computed from
the ONE `aug` object.
"""
function build_lfix_base_cache_cm_meanzc(x_free0::AbstractVector, ctx_cm, base::BaseDualState,
                                          ctx, aug, bins::AbstractMatrix{<:Unsigned}; validate_dense::Bool = false)
    cache0 = build_lfix_base_cache(x_free0, ctx_cm, base; validate_dense = validate_dense)
    cm_contrib0 = cm_fixed_contribution_meanzc_layout(base, ctx, aug, bins)
    meanzc_contrib0 = meanzc_fixed_contribution(base, aug)
    return with_q0(cache0, cache0.q0 .- cm_contrib0 .- meanzc_contrib0)
end

"""
    cm_fixed_contribution_meanzc_layout(base, ctx, aug, bins) -> Vector{Float64}

Identical math to `cm_fixed_contribution` (lfix_cm_aware.jl), but re-slices
`base.λstar` at the CM-grid block's actual position under the
`[economic | mean | pair | CM-grid | gravity]` layout
(`build_cm_meanzc_augmented_obj`/`wrap_moments_with_cm_meanzc`) instead of
`cm_fixed_contribution`'s own `[economic | CM-grid | gravity]` slice bounds --
the two column layouts differ (CM-grid starts at `ncore_econ` in the plain CM
`aug`, but at `ncore_econ + n_mean + n_pair` here), so the ORIGINAL
`cm_fixed_contribution` must not be called directly against a
`build_cm_meanzc_augmented_obj` result.
"""
function cm_fixed_contribution_meanzc_layout(base::BaseDualState, ctx, aug, bins::AbstractMatrix{<:Unsigned})
    ncore_econ = aug.ncore_econ; n_mean = aug.n_mean; n_pair = aug.n_pair; ncm = aug.ncm; L = aug.L
    nO = length(aug.origins)
    cm_start = ncore_econ + n_mean + n_pair
    @assert length(base.λstar) >= cm_start - 1 + ncm "base.λstar too short for aug's (ncore_econ,n_mean,n_pair,ncm) -- was base solved against aug.obj_cm?"
    λ_cm = base.λstar[cm_start:cm_start-1+ncm]
    λmat_stored = reshape(λ_cm, nO, L)
    R = aug.contrasts == :orthonormal ? orthonormal_contrast_matrix(ctx.D) : nothing
    λmat_block = apply_contrast(λmat_stored, R)
    P = suffix_sums(λmat_block)
    out = Vector{Float64}(undef, size(bins, 1))
    cumulative_forward_contribution!(out, bins, aug.refIndex1, aug.origins, P)
    return out
end

"""
    composite_gradient_at_fast_cm_meanzc(x_free0, ctx_cm, pe, ctx, aug, bins; base=nothing, cache=nothing, kwargs...)

CM+mean(+ZC)-aware entry point for the (g, A_od) outer gradient -- thin
wrapper around the UNCHANGED `composite_gradient_at_fast`, exact structural
analog of `composite_gradient_at_fast_cm` (lfix_cm_aware.jl). ν is held fixed
at `aug.nu_ref[]`'s current value throughout (Section 5's "the existing
A-coordinate CM gradient must continue to hold the new mean/pair columns
fixed" requirement) -- this function never mutates `nu_ref`, only reads it
once via `build_lfix_base_cache_cm_meanzc`/`meanzc_fixed_contribution`. The
separate `∂Delta_dual/∂η_ν` component (`d_delta_dual_d_eta_nu`,
mean_zero_cov_moments.jl) is NOT computed here -- it is appended by the outer
driver's `cb_G!` as the vector's last entry, since it needs no per-coordinate
work at all (constant across every (g,A_od) probe at a fixed base solve).
"""
function composite_gradient_at_fast_cm_meanzc(x_free0::AbstractVector, ctx_cm, pe, ctx, aug, bins::AbstractMatrix{<:Unsigned};
        base::Union{Nothing,BaseDualState} = nothing, cache = nothing,
        validate_dense::Bool = false, kwargs...)
    base = base === nothing ? solve_base_state(x_free0, ctx_cm) : base
    if cache === nothing
        cache = build_lfix_base_cache_cm_meanzc(x_free0, ctx_cm, base, ctx, aug, bins; validate_dense = validate_dense)
    end
    return composite_gradient_at_fast(x_free0, ctx_cm, pe; base = base, cache = cache, kwargs...)
end
