# Fills the disclosed gap flagged in cm_frechet_lfix_aware.jl's own docstring: "a fast-lookup
# (non-dense-matvec) contribution kernel for the :cdf_only archB moment path" was never built --
# cm_frechet_production_gradient hard-errors for frechet_feature_set=:cdf_only ("only :cdf_power is
# wired in this port-prep pass"). Since the addendum (2026-07-24) makes :cdf_only the production
# target, the outer-gradient path needs this to run the outer shakedown (task brief §12 requires
# "at least one outer gradient").
#
# Same lookup-kernel discipline as cm_lookup_kernels.jl (suffix-sum forward pass against bin
# indices, O(W*nO) not O(W*ncm) and no dense W x ncm matrix), applied to the CDF-only fixed-Fréchet
# moment layout established in cm_frechet_hessian.jl (contrast block: nO*L columns, column
# (l-1)*nO+oi; common/reference-pin block: L columns, pinned to targets.targets[l] = p[l]).
# Validated against the EXISTING dense reference (`frechet_fixed_contribution` +
# `build_cm_frechet_augmented_obj_basis(...; feature_set=:cdf_only)`, both already used/gated
# elsewhere in this branch) at D=4 in test_frechet_cdf_only_gradient_d4_gates.jl.

"""
    frechet_fixed_contribution_archB(base::BaseDualState, aug, fctx) -> Vector{Float64}

`out[s] = λ_FF*' h^FF_s` for every draw `s`, CDF-only fixed-Fréchet block, computed via bin-index
lookups against `fctx.cctx.Bidx` (already built/cached, same table `build_bin_tables!` consumes) --
O(W*nO), no dense `W x ncm` matrix (`aug.CM` does not exist for the archB/:cdf_only moment path).
`aug` is a `build_cm_frechet_augmented_obj_archB` result; `fctx` a `build_cm_frechet_bin_ctx` result
built against the SAME `aug`. `base.λstar`'s Fréchet-block sub-vector convention (slice
`ncore:ncore-1+ncm`, contrast-then-common column order) matches `frechet_fixed_contribution`
exactly -- same base, same layout, only the evaluation mechanism differs.
"""
function frechet_fixed_contribution_archB(base::BaseDualState, aug, fctx)
    ncore = aug.ncore; ncm = aug.ncm; L = aug.L; nO = length(aug.origins)
    @assert ncm == nO * L + L "layout assumption violated: ncm=$ncm, nO*L+L=$(nO*L+L)"
    @assert length(base.λstar) >= ncore - 1 + ncm "base.λstar too short for aug's (ncore,ncm) -- was base solved against aug.obj_cm?"
    λ_ff = base.λstar[ncore:ncore-1+ncm]
    λ_contrast_stored = reshape(@view(λ_ff[1:nO*L]), nO, L)
    λ_common = @view λ_ff[nO*L+1:nO*L+L]

    cctx = fctx.cctx
    R = cctx.R
    λ_contrast_block = R === nothing ? Matrix(λ_contrast_stored) : R * λ_contrast_stored

    # suffix sums: Qc[oi,k] = sum_{l=k}^{L} lambda_contrast_block[oi,l], Qc[oi,L+1] = 0
    Qc = zeros(nO, L + 1)
    @inbounds for oi in 1:nO
        acc = 0.0
        for l in L:-1:1
            acc += λ_contrast_block[oi, l]
            Qc[oi, l] = acc
        end
    end
    # Qcommon[k] = sum_{l=k}^{L} lambda_common[l], Qcommon[L+1] = 0
    Qcommon = zeros(L + 1)
    acc = 0.0
    @inbounds for l in L:-1:1
        acc += λ_common[l]
        Qcommon[l] = acc
    end
    p = fctx.p
    const_term = 0.0
    @inbounds for l in 1:L
        const_term += λ_common[l] * p[l]
    end

    Bidx = cctx.Bidx   # (W, D) Int, bin index in 1:L+1 per draw/origin, PREFIX convention (1{U<=z_l} <=> Bidx<=l)
    refIndex1 = cctx.refIndex1; origins = cctx.origins
    W = size(Bidx, 1)
    out = Vector{Float64}(undef, W)
    @inbounds for s in 1:W
        bref = Bidx[s, refIndex1]
        acc_s = Qcommon[bref] - const_term
        for (oi, o) in enumerate(origins)
            bo = Bidx[s, o]
            acc_s += Qc[oi, bo] - Qc[oi, bref]
        end
        out[s] = acc_s
    end
    return out
end

"""
    build_lfix_base_cache_cm_frechet_archB(x_free0, ctx_cm, base, aug, fctx; validate_dense=false)

CDF-only analogue of `build_lfix_base_cache_cm_frechet`, using the lookup-based
`frechet_fixed_contribution_archB` instead of a dense `aug.CM` matvec.
"""
function build_lfix_base_cache_cm_frechet_archB(x_free0::AbstractVector, ctx_cm, base::BaseDualState, aug, fctx;
                                                 validate_dense::Bool = false)
    cache0 = build_lfix_base_cache(x_free0, ctx_cm, base; validate_dense = validate_dense)
    ff_contrib0 = frechet_fixed_contribution_archB(base, aug, fctx)
    return with_q0(cache0, cache0.q0 .- ff_contrib0)
end

"""
    cm_frechet_production_gradient_cdf_only(x_free0, fpcx, pe; base=nothing, kwargs...) -> (g, meta)

CDF-only (`frechet_feature_set=:cdf_only`) analogue of `cm_frechet_production_gradient`. Does NOT
modify that function (which remains :cdf_power-only, unchanged, still used by the default
run_frechet_upper.jl driver) -- additive-only, this experimental branch's own CDF-only outer driver
calls this instead. Uses `cm_frechet_base_state`/`cm_frechet_verified_state`'s existing generic
dispatch to solve/reuse, then the lookup-based Lfix cache above.
"""
function cm_frechet_production_gradient_cdf_only(x_free0::AbstractVector, fpcx, pe;
        base::Union{Nothing,BaseDualState} = nothing, kwargs...)
    fpcx.mode === :frechet_reference || error("cm_frechet_production_gradient_cdf_only: fpcx.mode=$(fpcx.mode), expected :frechet_reference")
    fpcx.cfg.frechet_feature_set === :cdf_only || error("cm_frechet_production_gradient_cdf_only: only :cdf_only is supported here (got :$(fpcx.cfg.frechet_feature_set))")
    base = base === nothing ? cm_frechet_base_state(x_free0, fpcx) : base
    cache = build_lfix_base_cache_cm_frechet_archB(x_free0, fpcx.ctx_cm, base, fpcx.aug, fpcx.fctx)
    return composite_gradient_at_fast(x_free0, fpcx.ctx_cm, pe; base = base, cache = cache, kwargs...)
end
