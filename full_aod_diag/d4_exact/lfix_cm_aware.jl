# ============================================================================
# Continuation 13, Section 4: CM-aware Lfix/composite-gradient outer path.
#
# The common-marginals moment block C_s (precalc_common_marginals_cdf's CM
# matrix / the interval basis's analog) is a function of the FIXED U draws
# and the FIXED quantile cutpoints ONLY -- confirmed directly from
# wrap_moments_with_cm (common_marginals_moments.jl): the CM block is spliced
# into G verbatim from a precomputed `CM` matrix, never touched by
# `core_moments!(K, G_tmp, θ, U, obj)`'s θ-dependent computation. Consequently
# the augmented base-point dual scalar
#     q_s(θ) = -ζ* - λ_G*'G_s(θ) - λ_C*'C_s
# has a CONSTANT `λ_C*'C_s` term across every outer coordinate probe at a
# fixed base dual solve (θ0, ζ*, λ*) -- it needs to be computed exactly ONCE,
# using the SAME O(D)-per-draw cumulative-suffix-sum lookup identity already
# validated in cm_lookup_kernels.jl's `CMLookupState` (`:suffix` method) for
# the live KNITRO inner-solve FG callback, never the dense W x ncm matrix.
#
# This file is PURELY ADDITIVE: lfix_incremental.jl (LFixBaseCache /
# build_lfix_base_cache / lfix_incremental_at / dest_contrib_*) and
# composite_gradient.jl (gamma_component_analytic) are used completely
# UNCHANGED -- the only new mechanism is folding the CM contribution into
# `LFixBaseCache.q0` once, before any coordinate is probed. Every existing
# non-CM call site is unaffected.
#
# Expected include order (mirrors c14_combined_bundle_validate.jl /
# cm_hessian_architectures.jl's own convention):
#   context.jl, winners.jl, oracle.jl, common_marginals_moments.jl,
#   common_marginals_interval.jl, instrumentation.jl, oracle_fast.jl,
#   three_way_derivatives.jl, lfix_incremental.jl, composite_gradient.jl,
#   composite_gradient_fast.jl, cm_lookup_kernels.jl, THEN this file.
# ============================================================================

"""
    with_q0(cache::LFixBaseCache, q0_new::Vector{Float64}) -> LFixBaseCache

Returns a new `LFixBaseCache` identical to `cache` in every field except
`q0`, which is replaced by `q0_new`. Field-generic (built from
`fieldnames(LFixBaseCache)`, not a hardcoded positional list), so this stays
correct if `LFixBaseCache`'s field list is ever extended. This is the ONLY
mechanism this file uses to make the cache CM-aware: every downstream reader
of a `LFixBaseCache` (`lfix_incremental_at`, `dest_contrib_*`,
`gamma_component_analytic`, `select_bandwidth`, ...) only ever reads `q0` as
"the current base-point dual scalar," never re-derives it from `contrib0`/
`cf_contrib0` alone -- so swapping in a CM-augmented `q0` here is sufficient
and correct without touching any of that code.
"""
function with_q0(cache::LFixBaseCache, q0_new::Vector{Float64})
    vals = Any[f === :q0 ? q0_new : getfield(cache, f) for f in fieldnames(LFixBaseCache)]
    return LFixBaseCache(vals...)
end

"""
    cm_bin_indices_for(ctx, aug) -> Matrix{<:Unsigned}

Bin indices for every draw/origin w.r.t. `aug`'s own cumulative-CDF
thresholds (`aug.z`, from `build_cm_augmented_obj`'s
`precalc_common_marginals_cdf` call) -- reuses
`common_marginals_interval.jl::compute_bin_indices` (bit-identical
`searchsortedfirst` logic to `cm_hessian_architectures.jl`'s own
`compute_bin_indices`, but returns the `Unsigned`-typed matrix
`interval_forward_contribution!`/`cumulative_forward_contribution!` require)
rather than re-deriving bin construction a third time. `aug.z` is
CUMULATIVE-basis thresholds regardless of `aug.contrasts` (`:anchored` or
`:orthonormal` only change how the L threshold-columns are linearly combined
within the CM block, never the thresholds themselves).
"""
cm_bin_indices_for(ctx, aug) = compute_bin_indices(ctx.U, aug.z)

"""
    cm_fixed_value_contribution(λ_cm, nO, L, refIndex1, origins, bins, R) -> Vector{Float64}

Shared (C)-block fixed-λ VALUE contribution: `out[s] = λ_cm' C_s`, via the same
`apply_contrast`/`suffix_sums`/`cumulative_forward_contribution!` chain `CMLookupState`'s `:suffix`
method (cm_lookup_kernels.jl) uses for the live inner-solve FG callback, here evaluated ONCE at a
fixed `λ_cm` rather than per-callback. Harmonization (2026-07-29): both `cm_fixed_contribution`
(below, flexible CM) and `frechet_cm_level_fixed_contribution`
(cm_frechet_cplus.jl, common Fréchet's CM sub-block) call this exact function -- previously the
Fréchet side inlined a verbatim copy of this computation because `cm_fixed_contribution` hardcoded
`aug.ncm` as its own tail-slicing bound (wrong for Fréchet, whose `aug.ncm = ncm_cm+ncm_level`).
Taking the already-sliced `λ_cm` (rather than `aug`) as an argument removes that obstacle: each
caller slices its own tail according to its own layout, then shares this one value-contribution
kernel.
"""
function cm_fixed_value_contribution(λ_cm::AbstractVector{Float64}, nO::Int, L::Int, refIndex1::Int,
                                      origins::Vector{Int}, bins::AbstractMatrix{<:Unsigned},
                                      R::Union{Nothing,AbstractMatrix{Float64}})
    λmat_stored = reshape(λ_cm, nO, L)
    λmat_block = apply_contrast(λmat_stored, R)
    P = suffix_sums(λmat_block)
    out = Vector{Float64}(undef, size(bins, 1))
    cumulative_forward_contribution!(out, bins, refIndex1, origins, P)
    return out
end

"""
    cm_fixed_value_contribution_two_family(λ_cm, aug, bins, ctx) -> Vector{Float64}

2026-08-05 truncated-power task: family-count-driven wrapper around `cm_fixed_value_contribution`.
`λ_cm` is the FULL CM-block dual slice (width `aug.ncm`, both families if `aug.n_families==2`).
The eq.35 (CDF) sub-block (the first `aug.ncm_cdf` entries, ALWAYS present) goes through the
EXACT SAME suffix-sum lookup call as before -- byte-identical code path, so the CDF contribution
is bit-for-bit unchanged whether or not a second family exists. When `aug.n_families==2`, the
eq.36 (truncated-power) sub-block's contribution `λ_pow' * C_pow_s` is added via a direct BLAS
matvec against `aug.CM`'s already-precomputed power sub-block -- a literal, unoptimized evaluation
of the same dot-product definition the suffix-sum trick accelerates for the CDF block (no new
formula; the power block's own weighted-indicator structure does not collapse to the same O(W*nO)
suffix-sum identity without a materially new derivation, see
CM_CURRENT_SINGLE_BLOCK_SOURCE_MAP.md section 2 -- and at O(W*(D-1)*L), done ONCE per outer point
(not per Newton iteration, not per-outer-coordinate-probe), this is cheap: no per-outer
rematerialization, `aug.CM` was already built once at context-construction time).
"""
function cm_fixed_value_contribution_two_family(λ_cm::AbstractVector{Float64}, aug, bins::AbstractMatrix{<:Unsigned}, ctx)
    nO = length(aug.origins); L = aug.L
    ncm_cdf = aug.ncm_cdf
    length(λ_cm) == aug.ncm || error("cm_fixed_value_contribution_two_family: length(λ_cm)=$(length(λ_cm)) != aug.ncm=$(aug.ncm)")
    R = aug.contrasts == :orthonormal ? orthonormal_contrast_matrix(ctx.D) : nothing
    λ_cdf = @view λ_cm[1:ncm_cdf]
    out = cm_fixed_value_contribution(λ_cdf, nO, L, aug.refIndex1, aug.origins, bins, R)
    nf = hasproperty(aug, :n_families) ? aug.n_families : 1
    if nf == 2
        ncm_pow = aug.ncm_pow
        λ_pow = @view λ_cm[ncm_cdf+1:ncm_cdf+ncm_pow]
        CM_pow = @view aug.CM[:, ncm_cdf+1:ncm_cdf+ncm_pow]
        out .+= CM_pow * λ_pow
    end
    return out
end

"""
    cm_fixed_contribution(base::BaseDualState, aug, bins) -> Vector{Float64}

`out[s] = λ_C*' C_s` for every draw `s`, O(W*(D-1)) total (no loop over L),
via the cumulative-suffix-sum lookup identity (`cm_lookup_kernels.jl`,
`CMLookupState`'s `:suffix` branch, extracted here as a one-shot call since
the outer gradient never needs the (ζ,λ)-space GRADIENT of this term, only
its fixed VALUE at the base point). `base.λstar`'s CM-block sub-vector is
sliced at `aug.ncore:aug.ncore-1+aug.ncm` -- the same layout
`build_cm_augmented_obj`'s `wrap_moments_with_cm` establishes (core columns
`1:ncore-1`, CM columns `ncore:ncore-1+ncm`, gravity last) and
`CMLookupState`'s own `(st::CMLookupState)(x,g)` callable indexes
identically (`λ_cm = x[2+ncore1:1+ncore1+ncm]`, `ncore1=ncore-1`).
"""
function cm_fixed_contribution(base::BaseDualState, ctx, aug, bins::AbstractMatrix{<:Unsigned})
    ncore = aug.ncore; ncm = aug.ncm
    @assert length(base.λstar) >= ncore - 1 + ncm "base.λstar too short for aug's (ncore,ncm) -- was base solved against aug.obj_cm?"
    λ_cm = base.λstar[ncore:ncore-1+ncm]
    return cm_fixed_value_contribution_two_family(λ_cm, aug, bins, ctx)
end

"""
    build_lfix_base_cache_cm(x_free0, ctx_cm, base, ctx, aug, bins; validate_dense=false) -> LFixBaseCache

The CM-aware analog of `build_lfix_base_cache`. `ctx_cm` must have
`ctx_cm.obj === aug.obj_cm` (the CM-augmented objective the base dual solve
`base` was actually computed against) -- `build_lfix_base_cache` itself is
called COMPLETELY UNCHANGED against `ctx_cm`; because its economic-block
construction (`CONST_d`, `contrib0`, `cf_contrib0`) only ever indexes
`base.λstar[1:D^2]` and the counterfactual-column tail check
`oci-1>=D^2+1` (always true once CM columns are appended, since
`ncore>D^2+1` by construction), it silently and correctly IGNORES the CM
tail of `λstar` -- producing a `q0` that omits `λ_C*'C_s`, exactly the piece
this function adds back in via `cm_fixed_contribution` (computed ONCE, not
per-probe) before returning.

`ctx`/`aug`/`bins` are the plain (non-CM-swapped) context, the
`build_cm_augmented_obj` result, and `cm_bin_indices_for(ctx, aug)` output
respectively -- kept separate from `ctx_cm` because `aug.origins`/`aug.z`
are computed off the plain `ctx.U`, not `ctx_cm` (which only differs from
`ctx` in its `.obj` field).
"""
function build_lfix_base_cache_cm(x_free0::AbstractVector, ctx_cm, base::BaseDualState,
                                   ctx, aug, bins::AbstractMatrix{<:Unsigned}; validate_dense::Bool = false)
    cache0 = build_lfix_base_cache(x_free0, ctx_cm, base; validate_dense = validate_dense)
    cm_contrib0 = cm_fixed_contribution(base, ctx, aug, bins)
    return with_q0(cache0, cache0.q0 .- cm_contrib0)
end

"""
    composite_gradient_at_fast_cm(x_free0, ctx_cm, pe, ctx, aug, bins; base=nothing, cache=nothing, kwargs...)

CM-aware entry point, thin wrapper around the UNCHANGED
`composite_gradient_at_fast`: builds (or reuses, if `cache` is passed) a
CM-augmented `LFixBaseCache` via `build_lfix_base_cache_cm`, then delegates
ALL per-coordinate work (bandwidth selection, incremental FD, threading,
gamma component) to `composite_gradient_at_fast`'s own `cache=` kwarg (see
that function's docstring for why passing a pre-built cache is safe and
already an established pattern there, matching `base=`). Because the CM
contribution is folded into `q0` once and `lfix_incremental_at`/
`dest_contrib_*` never touch `q0`'s CM component (they only ADD/SUBTRACT
deltas for the AFFECTED destinations' ECONOMIC contribution), every
per-coordinate probe automatically reuses the frozen CM term at zero extra
cost -- `needs_outer_moment_jacobian` stays false, no dense `(D-1)*L` matrix
or `jac_h` tensor is ever materialized here.
"""
function composite_gradient_at_fast_cm(x_free0::AbstractVector, ctx_cm, pe, ctx, aug, bins::AbstractMatrix{<:Unsigned};
        base::Union{Nothing,BaseDualState} = nothing, cache::Union{Nothing,LFixBaseCache} = nothing,
        validate_dense::Bool = false, kwargs...)
    base = base === nothing ? solve_base_state(x_free0, ctx_cm) : base
    if cache === nothing
        cache = build_lfix_base_cache_cm(x_free0, ctx_cm, base, ctx, aug, bins; validate_dense = validate_dense)
    end
    return composite_gradient_at_fast(x_free0, ctx_cm, pe; base = base, cache = cache, kwargs...)
end
