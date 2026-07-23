# ============================================================================
# CM+mean(+ZC) production bundle: ties the mean_zero_cov_moments.jl /
# cm_meanzc_lfix_aware.jl pieces into one inner-solve + gradient path, exact
# structural analog of cm_production_bundle.jl.
#
# Column-layout proof (see math note Section 4 and this repo's own
# hessian_cm_structured! docstring): the augmented H layout is
#     H[:,1]=K, H[:,2]=const-1, H[:,3:...] = [economic | mean | pair | CM-grid | gravity(last)]
# `hessian_cm_structured!`'s `E = H[:,2:1+NCORE]` therefore equals
# `[const | economic]` when `NCORE=ncore_econ` (the plain-CM case) and equals
# `[const | economic | mean | pair]` when `NCORE=ncore_econ+n_mean+n_pair` (this
# file) -- the CM-grid block then starts immediately after at H column
# `2+NCORE`, matching `wrap_moments_with_cm_meanzc`'s column layout exactly.
# So `hessian_cm_structured!`/`build_bin_tables!`/`prefix_sum_tables!`
# (cm_hessian_architectures.jl) are reused COMPLETELY UNCHANGED -- only the
# `NCORE` field of the `CMBinHessCtx` passed in differs. Likewise
# `archC_base_state`/`archC_verified_state`/`cm_production_value`/
# `cm_production_value_verified` (cm_production_bundle.jl) are generic over
# `(ctx_cm, cctx)` and are reused UNCHANGED -- only `cm_production_gradient`
# needs a mean/ZC-aware replacement (it hardcodes `build_lfix_base_cache_cm`).
# ============================================================================

"""
    build_cm_meanzc_bin_ctx(ctx, aug) -> CMBinHessCtx

Widened-`NCORE` analog of `build_cm_bin_ctx` (cm_hessian_architectures.jl):
identical `Bidx`/`z`/`L`/`origins`/`refIndex1`/`contrasts` (built off the SAME
CM grid `aug` carries), but `NCORE = aug.ncore_econ + aug.n_mean + aug.n_pair`
instead of `aug.ncore` -- folding the mean/pair columns into the dense/BLAS
"economic" block (see file header). `aug` must be a
`build_cm_meanzc_augmented_obj` result.
"""
function build_cm_meanzc_bin_ctx(ctx, aug)
    L = aug.L; D = ctx.D; origins = aug.origins; nO = length(origins)
    refIndex1 = aug.refIndex1; z = aug.z
    NCORE = aug.ncore_econ + aug.n_mean + aug.n_pair
    ncm = aug.ncm
    R = aug.contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing
    Bidx = compute_bin_indices(ctx.U, z)
    W = size(ctx.U, 1)
    L1 = L + 1
    return CMBinHessCtx(L, D, nO, origins, refIndex1, z, Bidx, NCORE, ncm, aug.contrasts, R,
        zeros(D, D, L1, L1), zeros(D, NCORE, L1), zeros(D, D, L, L), zeros(D, NCORE, L),
        Matrix{Float64}(undef, W, NCORE), Matrix{Float64}(undef, NCORE + ncm, NCORE + ncm))
end

"""
    build_cm_meanzc_production_context(ctx, CS; L, cm_extension, meanzc_basis=:direct, contrasts=:anchored, probs=nothing, nu_ref=Ref(1.0))
        -> (ctx_cm, aug, bins, cctx)

CM+mean(+ZC) analog of `build_cm_production_context`. `bins` (for
`cm_fixed_contribution_meanzc_layout`) uses the SAME `aug.z`/`ctx.U` the CM
block itself uses, via `cm_bin_indices_for` (lfix_cm_aware.jl, unchanged --
generic over any `aug` carrying `.z`).
"""
function build_cm_meanzc_production_context(ctx, CS; L::Int, cm_extension::Symbol,
        meanzc_basis::Symbol = :direct, contrasts::Symbol = :anchored,
        probs::Union{Nothing,AbstractVector{Float64}} = nothing,
        refIndex1::Int = ctx.γ.refIndex1, nu_ref::Ref{Float64} = Ref(1.0))
    aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, cm_extension = cm_extension,
        contrasts = contrasts, meanzc_basis = meanzc_basis, probs = probs,
        refIndex1 = refIndex1, nu_ref = nu_ref)
    ctx_cm = merge(ctx, (obj = aug.obj_cm,))
    bins = cm_bin_indices_for(ctx, aug)
    cctx = build_cm_meanzc_bin_ctx(ctx, aug)
    return (ctx_cm = ctx_cm, aug = aug, bins = bins, cctx = cctx)
end

"""
    cm_meanzc_production_gradient(x_free0, pcx, ctx, pe; base=nothing, kwargs...) -> (g, meta)

CM+mean(+ZC)-aware analog of `cm_production_gradient` (cm_production_bundle.jl):
same Architecture-C inner solve (`archC_base_state`, reused unchanged), but
`composite_gradient_at_fast_cm_meanzc` (cm_meanzc_lfix_aware.jl) instead of
`composite_gradient_at_fast_cm` for the (g, A_od) gradient block. Does NOT
compute the `∂Delta_dual/∂η_ν` component -- callers needing the full extended
outer gradient must append `d_delta_dual_d_eta_nu(base.λstar, pcx.aug, ν;
mean_m=...)` themselves (needs `verify.m_mean`, which this cheap
gradient-only path does not compute; see `cm_meanzc_production_value_verified`).
"""
function cm_meanzc_production_gradient(x_free0::AbstractVector, pcx, ctx, pe;
        base::Union{Nothing,BaseDualState} = nothing, kwargs...)
    base = base === nothing ? archC_base_state(x_free0, pcx.ctx_cm, pcx.cctx) : base
    cache = build_lfix_base_cache_cm_meanzc(x_free0, pcx.ctx_cm, base, ctx, pcx.aug, pcx.bins)
    return composite_gradient_at_fast_cm_meanzc(x_free0, pcx.ctx_cm, pe, ctx, pcx.aug, pcx.bins; base = base, cache = cache, kwargs...)
end

"""
    cm_meanzc_production_value(x_free0, pcx) -> (K, base)
    cm_meanzc_production_value_verified(x_free0, pcx) -> (K, base, verify)

Reuse `cm_production_value`/`cm_production_value_verified` (cm_production_bundle.jl)
UNCHANGED -- both are generic over `pcx.ctx_cm`/`pcx.cctx`, never reference the
mean/pair column layout. Defined here only as clearly-named aliases so a
caller does not need to remember that reuse is safe.
"""
cm_meanzc_production_value(x_free0::AbstractVector, pcx) = cm_production_value(x_free0, pcx)
cm_meanzc_production_value_verified(x_free0::AbstractVector, pcx) = cm_production_value_verified(x_free0, pcx)

"""
    cm_meanzc_delta_dnu(x_free0, pcx; base=nothing, verify=nothing) -> Float64

Convenience one-call analytic `∂Delta_dual/∂ν` at `x_free0` (Section 6 of the
math note): solves (or reuses) the base state + verify tuple, then calls
`d_delta_dual_d_nu`. `∂Delta_dual/∂η_ν = ν · ∂Delta_dual/∂ν` is the caller's
own `pcx.aug.nu_ref[] * ` multiplication (kept explicit at call sites per the
math note's "do not trust this sign mechanically" -- every η_ν-chain-rule
application should be visibly a `ν * (...)` term, not hidden inside a helper).
"""
function cm_meanzc_delta_dnu(x_free0::AbstractVector, pcx; base = nothing, verify = nothing)
    if base === nothing || verify === nothing
        _, base, verify = cm_meanzc_production_value_verified(x_free0, pcx)
    end
    return d_delta_dual_d_nu(base.λstar, pcx.aug; mean_m = verify.m_mean)
end
