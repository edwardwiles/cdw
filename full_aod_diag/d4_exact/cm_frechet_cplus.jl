# ================================================================================================
# Fixed Fréchet as flexible CM plus a common-level anchor -- Part IV (outer gradient / C+ path).
#
# See docs/COMMON_FRECHET_GRADIENT_INTERPRETATION_2026-07-25.md. Summary: both the CM block AND
# the level block are THETA-INDEPENDENT (built once from the fixed baseline draws U, dM/dA=0 at
# fixed theta) -- exactly the property `lfix_cm_aware.jl`/`lfix_cm_cplus.jl` already exploit for
# CM's own restriction: `cm_fixed_contribution` computes lambda_C*'C_s ONCE (not re-evaluated per
# outer-coordinate probe) and folds it into the cached base dual scalar q0 via `with_q0`/`with_q0_C`
# BEFORE the coordinate loop runs -- the coordinate loop itself (`composite_gradient_at_Cplus_from_cache`)
# needs ZERO changes, since it never touches the CM/level tail of lambda* at all once q0 already
# reflects it.
#
# `cm_fixed_contribution` (lfix_cm_aware.jl:83) hardcodes `aug.ncm` as the FULL CM-only tail
# length and reshapes it `(nO, L)` -- for a :common_frechet aug, `aug.ncm = D*L != nO*L`, so it
# cannot be called unmodified (would throw a clear DimensionMismatch, not silently misinterpret
# data). This file adds the level-aware sibling, reusing `apply_contrast`/`suffix_sums`/
# `cumulative_forward_contribution!` (cm_lookup_kernels.jl) UNCHANGED for the CM part, adding one
# new small forward-contribution helper for the level part's different (all-D-origin SUM, not
# reference-differenced) structure -- and, because the level feature carries a NONZERO target
# (same root cause as Part III's Hessian bug), an extra constant term
# `sum(lambda_level .* level_targets)` that has no CM analog (CM's own target is always zero).
# ================================================================================================

"""
    frechet_level_forward_sum!(out, bins, D, λmat_ext)

`out[s] = sum_{o=1}^D λmat_ext[bins[s,o]]`, `λmat_ext` a length-`(L+1)` suffix-sum-extended vector
(column `L+1` == 0, the dropped/beyond-last-threshold bin -- same convention as
`interval_forward_contribution!`). Unlike that function, this one SUMS over all `D` origins rather
than differencing against a reference -- the level feature's own structure (task math doc §2:
`u=ones(D)/sqrt(D)`, symmetric across all origins). O(W*D), no loop over `L`.
"""
function frechet_level_forward_sum!(out::AbstractVector{Float64}, bins::AbstractMatrix{<:Unsigned}, D::Int,
                                     λmat_ext::AbstractVector{Float64})
    W = length(out)
    @inbounds for s in 1:W
        acc = 0.0
        for o in 1:D
            acc += λmat_ext[Int(bins[s, o])]
        end
        out[s] = acc
    end
    return out
end

"""
    frechet_cm_level_fixed_contribution(base::BaseDualState, ctx, aug, bins) -> Vector{Float64}

Level-aware analog of `lfix_cm_aware.jl::cm_fixed_contribution`: computes the FULL fixed
(theta-independent) contribution `lambda_tail*'M_s` per draw `s`, where the tail
`lambda_tail = base.λstar[aug.ncore : aug.ncore-1+aug.ncm]` (length `aug.ncm = D*L`) splits into
`lambda_cm` (first `ncm_cm=(D-1)*L`) and `lambda_level` (last `ncm_level=L`). The CM part is computed
by the IDENTICAL logic `cm_fixed_contribution` uses (same `apply_contrast`/`suffix_sums`/
`cumulative_forward_contribution!` calls -- inlined here, not calling that function directly,
because it hardcodes `aug.ncm` as its own tail length); the level part is new (see
`frechet_level_forward_sum!` above), including the constant target-correction term
`sum(lambda_level .* aug.level_targets)` -- structurally the gradient-side analog of Part III's
Hessian target-correction terms (an additive constant in the level feature contributes an
additive-constant term to `lambda_level'*level_s`, unlike CM's own always-zero-target columns).
"""
function frechet_cm_level_fixed_contribution(base::BaseDualState, ctx, aug, bins::AbstractMatrix{<:Unsigned})
    ncore = aug.ncore; ncm_cm = aug.ncm_cm; ncm_level = aug.ncm_level; L = aug.L
    nO = length(aug.origins)
    D = ctx.D
    @assert length(base.λstar) >= ncore - 1 + ncm_cm + ncm_level "base.λstar too short for aug's (ncore,ncm_cm,ncm_level) -- was base solved against aug.obj_cm?"

    λ_cm = base.λstar[ncore : ncore - 1 + ncm_cm]
    λmat_stored = reshape(λ_cm, nO, L)
    R = aug.contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing
    λmat_block = apply_contrast(λmat_stored, R)
    P_cm = suffix_sums(λmat_block)
    cm_out = Vector{Float64}(undef, size(bins, 1))
    cumulative_forward_contribution!(cm_out, bins, aug.refIndex1, aug.origins, P_cm)

    λ_level = base.λstar[ncore + ncm_cm : ncore - 1 + ncm_cm + ncm_level]
    P_level_mat = suffix_sums(reshape(λ_level, 1, L))   # 1 x (L+1), column L+1 == 0
    invsqrtD = 1.0 / sqrt(D)
    level_out = Vector{Float64}(undef, size(bins, 1))
    frechet_level_forward_sum!(level_out, bins, D, vec(P_level_mat))
    level_out .*= invsqrtD
    level_out .-= sum(λ_level .* aug.level_targets)   # constant target-correction term (see docstring)

    return cm_out .+ level_out
end

"""
    build_lfix_base_cache_cm_frechet_C!(ws, x_free0, ctx_cm, base, ctx, aug, bins; validate_dense=false) -> LFixBaseCacheC

`:common_frechet` analog of `lfix_cm_cplus.jl::build_lfix_base_cache_cm_C!`. Identical structure --
calls `build_lfix_base_cache_C!` UNCHANGED (same reasoning: it only ever indexes
`base.λstar[1:D^2]`, silently and correctly ignoring the whole CM+level tail), then folds in
`frechet_cm_level_fixed_contribution` instead of plain `cm_fixed_contribution`.
"""
function build_lfix_base_cache_cm_frechet_C!(ws::LFixFactorizedWorkspace, x_free0::AbstractVector, ctx_cm,
                                              base::BaseDualState, ctx, aug, bins::AbstractMatrix{<:Unsigned};
                                              validate_dense::Bool = false)
    cache0 = build_lfix_base_cache_C!(ws, x_free0, ctx_cm, base; validate_dense = validate_dense)
    contrib0 = frechet_cm_level_fixed_contribution(base, ctx, aug, bins)
    return with_q0_C(cache0, cache0.q0 .- contrib0)
end

"""
    archC_frechet_base_state(x_free0, ctx_cm, cctx, level_targets) -> BaseDualState

`:common_frechet` analog of `cm_production_bundle.jl::archC_base_state`, using
`archC_frechet_hess_cb_builder(cctx, level_targets)` (Part III) instead of `archC_hess_cb_builder(cctx)`.
"""
function archC_frechet_base_state(x_free0::AbstractVector, ctx_cm, cctx::CMBinHessCtx, level_targets::Vector{Float64})
    obj = ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    K, x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_full0;
        hess_cb_builder = _obj -> archC_frechet_hess_cb_builder(cctx, level_targets))
    nStatus in (0, -100, -101, -103) || throw(CMExpectedSolveFailure("archC_frechet_base_state: inner solve failed, nStatus=$nStatus (x_free0=$x_free0)"))
    ζstar = x[1]; λstar = collect(x[2:end])
    return BaseDualState(collect(x_free0), θ_full0, ζstar, λstar, copy(obj.arg1), nStatus)
end

"""
    composite_gradient_at_Cplus_frechet(x_free0, ctx_cm, pe, ctx, aug, bins, pool, ws, cctx; base=nothing, cache=nothing, kwargs...) -> (g, meta)

`:common_frechet` analog of `lfix_cm_cplus.jl::composite_gradient_at_Cplus_cm`. `base` defaults to
`archC_frechet_base_state` (requires `aug.level_targets`); everything downstream
(`composite_gradient_at_Cplus_from_cache`, the actual coordinate loop) is called UNCHANGED -- the
only new work needed for the outer-gradient path is computing the correct fixed q0 contribution
(`build_lfix_base_cache_cm_frechet_C!`), exactly per this file's header rationale.
"""
function composite_gradient_at_Cplus_frechet(x_free0::AbstractVector, ctx_cm, pe, ctx, aug, bins::AbstractMatrix{<:Unsigned},
        pool::GradWorkspacePool, ws::LFixFactorizedWorkspace, cctx;
        base::Union{Nothing,BaseDualState} = nothing, cache::Union{Nothing,LFixBaseCacheC} = nothing,
        validate_dense::Bool = false, kwargs...)
    base = base === nothing ? archC_frechet_base_state(x_free0, ctx_cm, cctx, aug.level_targets) : base
    if cache === nothing
        cache = build_lfix_base_cache_cm_frechet_C!(ws, x_free0, ctx_cm, base, ctx, aug, bins; validate_dense = validate_dense)
    end
    return composite_gradient_at_Cplus_from_cache(x_free0, ctx_cm, pe, pool, cache; base = base, kwargs...)
end

# ================================================================================================
# Reference (non-factorized) envelope backend -- the "reference envelope gradient" task §10/§11 asks
# the C+ path above to be validated against. Structural twin of `lfix_cm_aware.jl`'s
# `build_lfix_base_cache_cm`/`composite_gradient_at_fast_cm`, using `frechet_cm_level_fixed_contribution`
# in place of `cm_fixed_contribution`. `solve_base_state` (three_way_derivatives.jl) and
# `build_lfix_base_cache`/`composite_gradient_at_fast` are all ALREADY GENERIC on `ctx.obj` (no
# restriction-family-specific code in any of them) -- called completely unchanged, exactly the same
# "reuse, don't duplicate" pattern as the C+ side above.
# ================================================================================================

"""
    build_lfix_base_cache_cm_frechet(x_free0, ctx_cm, base, ctx, aug, bins; validate_dense=false) -> LFixBaseCache

`:common_frechet` analog of `lfix_cm_aware.jl::build_lfix_base_cache_cm`, for the Reference
(non-factorized) backend. `ctx_cm.obj` here is expected to be the DENSE Architecture-A Fréchet obj
(`build_cm_frechet_level_augmented_obj`'s `obj_cm`), matching how `build_lfix_base_cache_cm` itself
is normally paired with the dense reference obj rather than an Architecture-B/C one.
"""
function build_lfix_base_cache_cm_frechet(x_free0::AbstractVector, ctx_cm, base::BaseDualState,
                                           ctx, aug, bins::AbstractMatrix{<:Unsigned}; validate_dense::Bool = false)
    cache0 = build_lfix_base_cache(x_free0, ctx_cm, base; validate_dense = validate_dense)
    contrib0 = frechet_cm_level_fixed_contribution(base, ctx, aug, bins)
    return with_q0(cache0, cache0.q0 .- contrib0)
end

"""
    composite_gradient_at_fast_frechet(x_free0, ctx_cm, pe, ctx, aug, bins; base=nothing, cache=nothing, kwargs...) -> (g, meta)

`:common_frechet` analog of `lfix_cm_aware.jl::composite_gradient_at_fast_cm`. `base` defaults to
`solve_base_state(x_free0, ctx_cm)` (the plain dense inner solve, unchanged, generic on `ctx_cm.obj`).
"""
function composite_gradient_at_fast_frechet(x_free0::AbstractVector, ctx_cm, pe, ctx, aug, bins::AbstractMatrix{<:Unsigned};
        base::Union{Nothing,BaseDualState} = nothing, cache::Union{Nothing,LFixBaseCache} = nothing,
        validate_dense::Bool = false, kwargs...)
    base = base === nothing ? solve_base_state(x_free0, ctx_cm) : base
    if cache === nothing
        cache = build_lfix_base_cache_cm_frechet(x_free0, ctx_cm, base, ctx, aug, bins; validate_dense = validate_dense)
    end
    return composite_gradient_at_fast(x_free0, ctx_cm, pe; base = base, cache = cache, kwargs...)
end
