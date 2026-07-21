# ============================================================================
# Continuation 13, Sections 3A + 5: production combined bundle.
#
# Ties together, additively, every already-validated Continuation 12/13 piece
# into ONE inner-solve + gradient path suitable for an outer KNITRO loop:
#   - cumulative basis (build_cm_augmented_obj), nested grids (probs=)
#   - Architecture B moment construction (cached G_tmp, no persistent dense
#     CM copy per call -- cm_hessian_architectures.jl, already built)
#   - Architecture C structured Hessian for the INNER dual solve
#     (cm_hessian_architectures.jl, already built) -- this is what Section
#     9's "wire Architecture C into an actual KNITRO OUTER loop" needed:
#     every inner solve triggered during outer optimization (not just a
#     one-off script) now goes through Architecture C.
#   - the CM-aware Lfix outer gradient (lfix_cm_aware.jl)
# No existing function is modified; this file only composes.
# ============================================================================

"""
    build_cm_production_context(ctx, CS; L, contrasts=:anchored, probs=nothing, use_archB_moments=true)
        -> (ctx_cm, aug, bins, cctx)

`ctx_cm.obj` uses Architecture B's moment construction when
`use_archB_moments=true` (default -- avoids `wrap_moments_with_cm`'s
fresh-`similar` `G_tmp` per call and never stores/copies from a persistent
dense `W x ncm` CM matrix beyond the one built once here for the dense
REFERENCE `aug.CM`/verification; per-call construction goes straight from
bin indices via `fill_cm_columns_from_bins!`). `cctx` is the Architecture-C
bin/scratch context (`build_cm_bin_ctx`), reused for every subsequent inner
solve at this context (bin indices are fixed once the draws U are fixed --
independent of theta).
"""
function build_cm_production_context(ctx, CS; L::Int, contrasts::Symbol = :anchored,
                                      probs::Union{Nothing,AbstractVector{Float64}} = nothing,
                                      use_archB_moments::Bool = true)
    aug = build_cm_augmented_obj(ctx, CS; L = L, contrasts = contrasts, probs = probs)
    obj_cm = aug.obj_cm
    if use_archB_moments
        # NOTE: common_marginals_interval.jl and cm_hessian_architectures.jl both define
        # `compute_bin_indices(U,z)` with overlapping-but-distinct signatures (z::Vector{Float64}
        # vs z::AbstractVector{Float64}) -- Julia's most-specific-method dispatch silently prefers
        # the FORMER (Unsigned-typed, for interval_forward_contribution!) regardless of include
        # order, which is NOT what fill_cm_columns_from_bins! below expects (Matrix{Int}). Force
        # the Int-typed variant explicitly rather than depend on ambient method resolution.
        Bidx = Int.(compute_bin_indices(ctx.U, aug.z))
        R = contrasts == :orthonormal ? orthonormal_contrast_matrix(ctx.D) : nothing
        moments_archB! = wrap_moments_with_cm_archB(ctx.obj.moments!, aug.ncore, Bidx, aug.origins, aug.refIndex1, aug.L, R)
        obj_cm = CS.PsiObjectiveBundleImplicit(δ = obj_cm.δ, find_smallest = obj_cm.find_smallest,
            γ = obj_cm.γ, (moments!) = moments_archB!, moments_jacobian! = error,
            d = obj_cm.d, outer_constr_index = obj_cm.outer_constr_index,
            inequality_index = obj_cm.inequality_index, complement_index = obj_cm.complement_index,
            l = obj_cm.l, U = obj_cm.U, N = obj_cm.N, lower_limit = obj_cm.lower_limit,
            use_cached_x = obj_cm.use_cached_x,
            outer_loop_opt = obj_cm.outer_loop_opt, inner_loop_opt = obj_cm.inner_loop_opt,
            needs_outer_moment_jacobian = obj_cm.needs_outer_moment_jacobian)
    end
    ctx_cm = merge(ctx, (obj = obj_cm,))
    bins = cm_bin_indices_for(ctx, aug)
    cctx = build_cm_bin_ctx(ctx, aug)
    return (ctx_cm = ctx_cm, aug = aug, bins = bins, cctx = cctx)
end

"""
    archC_base_state(x_free0, ctx_cm, cctx) -> BaseDualState

Architecture-C-accelerated drop-in replacement for `solve_base_state` (which
always uses the dense Architecture-A Hessian callback). Uses the SAME shared
FG callback as every other architecture (`inner_loop_internal_archgeneric`
only swaps the HESSIAN callback) -- so `obj.arg1` is populated identically to
how `solve_base_state`'s own call sites already read it
(`copy(ctx.obj.arg1)`, see e.g. `c10_d20_production_driver.jl`), reused here
rather than re-derived: no extra recompute needed, `obj.arg1` is trustworthy
immediately after `KN_solve` converges (KNITRO's own last FG call is always
made AT the reported solution).
"""
function archC_base_state(x_free0::AbstractVector, ctx_cm, cctx::CMBinHessCtx)
    obj = ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    K, x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_full0;
        hess_cb_builder = _obj -> archC_hess_cb_builder(cctx))
    nStatus in (0, -100, -101, -103) || error("archC_base_state: inner solve failed, nStatus=$nStatus (x_free0=$x_free0)")
    ζstar = x[1]; λstar = collect(x[2:end])
    return BaseDualState(collect(x_free0), θ_full0, ζstar, λstar, copy(obj.arg1), nStatus)
end

"""
    cm_production_gradient(x_free0, pcx, ctx, pe; kwargs...) -> (g, meta)

`pcx = build_cm_production_context(...)`'s return value. One-call entry
point: Architecture-C inner solve (`archC_base_state`) + the CM-aware Lfix
gradient (`composite_gradient_at_fast_cm`), fully wired for a KNITRO OUTER
callback's `cb_G!`.
"""
function cm_production_gradient(x_free0::AbstractVector, pcx, ctx, pe;
        base::Union{Nothing,BaseDualState} = nothing, kwargs...)
    base = base === nothing ? archC_base_state(x_free0, pcx.ctx_cm, pcx.cctx) : base
    cache = build_lfix_base_cache_cm(x_free0, pcx.ctx_cm, base, ctx, pcx.aug, pcx.bins)
    return composite_gradient_at_fast(x_free0, pcx.ctx_cm, pe; base = base, cache = cache, kwargs...)
end

"""
    cm_production_value(x_free0, pcx) -> (K, base)

Architecture-C inner solve only (objective value, no gradient) -- for a
KNITRO outer `cb_F!` or a plain feasibility/kappa check.
"""
function cm_production_value(x_free0::AbstractVector, pcx)
    base = archC_base_state(x_free0, pcx.ctx_cm, pcx.cctx)
    K = pcx.ctx_cm.obj.H_save
    return K, base
end
