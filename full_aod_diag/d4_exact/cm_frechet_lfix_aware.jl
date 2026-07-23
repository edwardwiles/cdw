# ============================================================================
# Fixed-Fréchet-marginals CM-aware outer-gradient path (2026-07-23), both
# backends. Mirrors lfix_cm_aware.jl (Reference) and lfix_cm_cplus.jl (C+)
# EXACTLY: the CM-augmented base-point dual scalar
#     q_s(theta) = -zeta* - lambda_G*'G_s(theta) - lambda_C*'C_s
# still has a CONSTANT `lambda_C*'C_s` term across every outer coordinate
# probe (same proof, unchanged -- the fixed-Fréchet CM block is still
# theta-independent, built once from U and the analytic targets). The ONLY
# new piece is `cm_frechet_fixed_contribution`, which folds in the extra
# trailing L "common" columns' contribution alongside the existing (D-1)*L
# contrast columns' contribution (the latter via the EXACT SAME
# `cumulative_forward_contribution!`/`suffix_sums` kernels flexible CM uses,
# unchanged). Purely additive: does not modify lfix_cm_aware.jl or
# lfix_cm_cplus.jl.
# ============================================================================

"""
    cm_frechet_fixed_contribution(base, ctx, aug_frechet, bins) -> Vector{Float64}

`out[s] = lambda_C*' C_s` for the FULL `D*L`-wide fixed-Fréchet CM block
(existing `(D-1)*L` contrast columns + new trailing `L` common columns).
`base.λstar`'s CM-block sub-vector is sliced at
`aug_frechet.ncore : aug_frechet.ncore-1+aug_frechet.ncm`, split into the
leading `nO*L` contrast entries and trailing `L` common entries -- the SAME
layout `build_cm_frechet_augmented_obj(_archB)` establishes (§7 of the math
note).

Contrast part: identical computation to `cm_fixed_contribution`
(lfix_cm_aware.jl), reused (not re-derived) via the same
`cumulative_forward_contribution!`/`suffix_sums` kernels.

Common part: `sum_l λ_common[l] * (1{bin(U[s,ref])<=l} - p[l])
            = P_common[bin(U[s,ref])] - β`, `P_common = suffix_sums` of the
length-`L` `λ_common` vector (viewed as a `1 x L` matrix, i.e. the `nO=1`
degenerate case of the same suffix-sum identity), `β = sum(λ_common .* p)`
the (per-draw-constant) target contribution.
"""
function cm_frechet_fixed_contribution(base::BaseDualState, ctx, aug_frechet, bins::AbstractMatrix{<:Unsigned})
    ncore = aug_frechet.ncore; ncm = aug_frechet.ncm; L = aug_frechet.L
    nO = length(aug_frechet.origins)
    @assert ncm == nO * L + L "cm_frechet_fixed_contribution: ncm=$ncm != nO*L+L=$(nO*L+L)"
    @assert length(base.λstar) >= ncore - 1 + ncm "base.λstar too short for aug_frechet's (ncore,ncm) -- was base solved against aug_frechet.obj_cm?"
    λ_cm_full = base.λstar[ncore:ncore-1+ncm]
    λ_contrast = λ_cm_full[1:nO*L]
    λ_common = λ_cm_full[nO*L+1:nO*L+L]

    R = aug_frechet.contrasts == :orthonormal ? orthonormal_contrast_matrix(ctx.D) : nothing
    λmat_stored = reshape(λ_contrast, nO, L)
    λmat_block = apply_contrast(λmat_stored, R)
    P = suffix_sums(λmat_block)
    out = Vector{Float64}(undef, size(bins, 1))
    cumulative_forward_contribution!(out, bins, aug_frechet.refIndex1, aug_frechet.origins, P)

    p = aug_frechet.targets.targets
    Pc = suffix_sums(reshape(λ_common, 1, L))
    β = sum(λ_common .* p)
    refIndex1 = aug_frechet.refIndex1
    @inbounds for s in 1:length(out)
        out[s] += Pc[1, Int(bins[s, refIndex1])] - β
    end
    return out
end

# ---- Reference backend ----

"CM-aware analog of `build_lfix_base_cache_cm` for fixed Fréchet marginals."
function build_lfix_base_cache_cm_frechet(x_free0::AbstractVector, ctx_cm, base::BaseDualState,
                                           ctx, aug_frechet, bins::AbstractMatrix{<:Unsigned}; validate_dense::Bool = false)
    cache0 = build_lfix_base_cache(x_free0, ctx_cm, base; validate_dense = validate_dense)
    cm_contrib0 = cm_frechet_fixed_contribution(base, ctx, aug_frechet, bins)
    return with_q0(cache0, cache0.q0 .- cm_contrib0)
end

"CM-aware entry point (Reference backend), fixed Fréchet marginals -- structural twin of `composite_gradient_at_fast_cm`."
function composite_gradient_at_fast_cm_frechet(x_free0::AbstractVector, ctx_cm, pe, ctx, aug_frechet, bins::AbstractMatrix{<:Unsigned};
        base::Union{Nothing,BaseDualState} = nothing, cache::Union{Nothing,LFixBaseCache} = nothing,
        validate_dense::Bool = false, kwargs...)
    base = base === nothing ? solve_base_state(x_free0, ctx_cm) : base
    if cache === nothing
        cache = build_lfix_base_cache_cm_frechet(x_free0, ctx_cm, base, ctx, aug_frechet, bins; validate_dense = validate_dense)
    end
    return composite_gradient_at_fast(x_free0, ctx_cm, pe; base = base, cache = cache, kwargs...)
end

# ---- C+ backend ----

"CM-aware analog of `build_lfix_base_cache_cm_C!` for fixed Fréchet marginals."
function build_lfix_base_cache_cm_frechet_C!(ws::LFixFactorizedWorkspace, x_free0::AbstractVector, ctx_cm,
                                              base::BaseDualState, ctx, aug_frechet, bins::AbstractMatrix{<:Unsigned};
                                              validate_dense::Bool = false)
    cache0 = build_lfix_base_cache_C!(ws, x_free0, ctx_cm, base; validate_dense = validate_dense)
    cm_contrib0 = cm_frechet_fixed_contribution(base, ctx, aug_frechet, bins)
    return with_q0_C(cache0, cache0.q0 .- cm_contrib0)
end

"CM-aware entry point (C+ backend), fixed Fréchet marginals -- structural twin of `composite_gradient_at_Cplus_cm`."
function composite_gradient_at_Cplus_cm_frechet(x_free0::AbstractVector, ctx_cm, pe, ctx, aug_frechet, bins::AbstractMatrix{<:Unsigned},
        pool::GradWorkspacePool, ws::LFixFactorizedWorkspace, fctx;
        base::Union{Nothing,BaseDualState} = nothing, cache::Union{Nothing,LFixBaseCacheC} = nothing,
        validate_dense::Bool = false, kwargs...)
    base = base === nothing ? archC_frechet_base_state(x_free0, ctx_cm, fctx) : base
    if cache === nothing
        cache = build_lfix_base_cache_cm_frechet_C!(ws, x_free0, ctx_cm, base, ctx, aug_frechet, bins; validate_dense = validate_dense)
    end
    return composite_gradient_at_Cplus_from_cache(x_free0, ctx_cm, pe, pool, cache; base = base, kwargs...)
end
