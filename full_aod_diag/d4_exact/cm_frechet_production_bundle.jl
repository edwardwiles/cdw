# ============================================================================
# Fixed-Fréchet-marginals production context dispatcher + base-state /
# gradient entry points (2026-07-23). Structural twin of
# cm_production_bundle.jl / cm_config.jl's `build_cm_production_context_v2`,
# dispatching on `CMFrechetConfig` instead of `CMConfig`. Purely additive.
# ============================================================================

"Architecture-C base-state solve for fixed Fréchet marginals -- structural twin of `archC_base_state`."
function archC_frechet_base_state(x_free0::AbstractVector, ctx_cm, fctx)
    obj = ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    K, x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_full0;
        hess_cb_builder = _obj -> archC_frechet_hess_cb_builder(fctx))
    nStatus in (0, -100, -101, -103) || throw(CMExpectedSolveFailure("archC_frechet_base_state: inner solve failed, nStatus=$nStatus (x_free0=$x_free0)"))
    ζstar = x[1]; λstar = collect(x[2:end])
    return BaseDualState(collect(x_free0), θ_full0, ζstar, λstar, copy(obj.arg1), nStatus)
end

"""
    archC_frechet_verified_state(x_free0, ctx_cm, fctx) -> (base, verify)

Fixed-Fréchet analog of `archC_verified_state` -- same independent-recompute
diagnostics (`Delta_dual`, `Delta_primal`, `primal_dual_gap`,
`weight_norm_resid`, `max_abs_moment_kkt_resid`, ...), reusing
`primal_divergence`/`kkt_residual_blas` unchanged. Used by the cold verifier
(task brief §9).
"""
function archC_frechet_verified_state(x_free0::AbstractVector, ctx_cm, fctx)
    obj = ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    K, inner_x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_full0;
        hess_cb_builder = _obj -> archC_frechet_hess_cb_builder(fctx))
    nStatus in (0, -100, -101, -103) || throw(CMExpectedSolveFailure("archC_frechet_verified_state: inner solve failed, nStatus=$nStatus (x_free0=$x_free0)"))

    ζstar = inner_x[1]; λstar = collect(inner_x[2:end])
    W = size(obj.U, 1)
    G = CS.select_G_from_H(obj, obj.H)

    ncon = obj.d - obj.outer_constr_index + 2
    cbuf = zeros(ncon)
    obj(inner_x, constr = @view(cbuf[1:ncon]))
    Delta_dual = cbuf[1] / 1e10
    m_weights = copy(obj.arg1)
    p_weights = m_weights ./ sum(m_weights)
    Delta_primal = primal_divergence(m_weights)

    mean_m_resid = abs(sum(m_weights) / W - 1.0)
    nkkt = min(length(λstar), size(G, 2))
    max_abs_moment_kkt_resid = kkt_residual_blas(G, m_weights, nkkt, W)

    base = BaseDualState(collect(x_free0), θ_full0, ζstar, λstar, m_weights, nStatus)
    verify = (inner_status = nStatus, Delta_dual = Delta_dual, Delta_primal = Delta_primal,
              primal_dual_gap = abs(Delta_dual - Delta_primal),
              weight_norm_resid = abs(sum(p_weights) - 1.0),
              mean_m_resid = mean_m_resid, max_abs_moment_kkt_resid = max_abs_moment_kkt_resid,
              m_mean = sum(m_weights) / W, m_min = minimum(m_weights), m_max = maximum(m_weights))
    return base, verify
end

"""
    build_cm_frechet_production_context(ctx, CS, cfg::CMFrechetConfig; L=cfg.cm.cm_grid_size) -> NamedTuple

`marginal_mode=:common_flexible` delegates to the EXISTING
`build_cm_production_context_v2` unchanged (byte-identical results,
`targets=nothing`). `marginal_mode=:frechet_reference` builds the fixed-
Fréchet augmented objective under `cfg.cm.cm_hessian_backend`
(`:structured` -- Architecture B moments + Architecture C Hessian,
production default; `:dense_reference` -- Architecture A, the D=4
correctness baseline).
"""
function build_cm_frechet_production_context(ctx, CS, cfg::CMFrechetConfig; L::Int = cfg.cm.cm_grid_size)
    _cm_frechet_validate(cfg)
    if cfg.marginal_mode === :common_flexible
        pcx = build_cm_production_context_v2(ctx, CS, cfg.cm; L = L)
        return (ctx_cm = pcx.ctx_cm, aug = pcx.aug, hess_cb_builder = pcx.hess_cb_builder,
                cfg = cfg, L = L, targets = nothing, mode = :common_flexible, bins = nothing, fctx = nothing)
    end

    targets = build_frechet_reference_targets(ctx, cfg; L = L)
    if cfg.cm.cm_hessian_backend === :structured
        aug = build_cm_frechet_augmented_obj_archB(ctx, CS, targets; contrasts = cfg.cm.contrasts)
        fctx = build_cm_frechet_bin_ctx(ctx, aug)
        hess_cb_builder = _obj -> archC_frechet_hess_cb_builder(fctx)
    else # :dense_reference
        aug = build_cm_frechet_augmented_obj(ctx, CS, targets; contrasts = cfg.cm.contrasts)
        fctx = nothing
        hess_cb_builder = archA_hess_cb_builder
    end
    ctx_cm = merge(ctx, (obj = aug.obj_cm,))
    bins = cm_bin_indices_for(ctx, aug)
    return (ctx_cm = ctx_cm, aug = aug, hess_cb_builder = hess_cb_builder, cfg = cfg, L = L,
            targets = targets, mode = :frechet_reference, bins = bins, fctx = fctx)
end

"""
    cm_frechet_base_state(x_free0, fpcx) -> BaseDualState

Dispatches to Architecture C (`archC_frechet_base_state`, `fpcx.fctx !==
nothing`) or the dense Architecture A path (`solve_base_state`,
`cm_base_state_v2`'s own generic-hess-cb pattern) depending on
`fpcx.cfg.cm.cm_hessian_backend`.
"""
function cm_frechet_base_state(x_free0::AbstractVector, fpcx)
    if fpcx.fctx !== nothing
        return archC_frechet_base_state(x_free0, fpcx.ctx_cm, fpcx.fctx)
    end
    obj = fpcx.ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free0, fpcx.ctx_cm.m)
    K, x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_full0; hess_cb_builder = fpcx.hess_cb_builder)
    nStatus in (0, -100, -101, -103) || error("cm_frechet_base_state: inner solve failed, nStatus=$nStatus (x_free0=$x_free0)")
    ζstar = x[1]; λstar = collect(x[2:end])
    return BaseDualState(collect(x_free0), θ_full0, ζstar, λstar, copy(obj.arg1), nStatus)
end

"""
    cm_frechet_production_gradient(x_free0, fpcx, ctx, pe; base=nothing, kwargs...) -> (g, meta)

Reference-backend gradient entry point, fixed Fréchet marginals. Requires
`fpcx.mode === :frechet_reference` (for `:common_flexible`, call the
existing `cm_production_gradient` against `fpcx.ctx_cm`/`fpcx.aug` instead).
"""
function cm_frechet_production_gradient(x_free0::AbstractVector, fpcx, ctx, pe; base::Union{Nothing,BaseDualState} = nothing, kwargs...)
    fpcx.mode === :frechet_reference || error("cm_frechet_production_gradient: fpcx.mode=$(fpcx.mode), expected :frechet_reference")
    base = base === nothing ? cm_frechet_base_state(x_free0, fpcx) : base
    cache = build_lfix_base_cache_cm_frechet(x_free0, fpcx.ctx_cm, base, ctx, fpcx.aug, fpcx.bins)
    return composite_gradient_at_fast(x_free0, fpcx.ctx_cm, pe; base = base, cache = cache, kwargs...)
end

"""
    cm_frechet_production_gradient_cplus(x_free0, fpcx, ctx, pe, pool, ws; base=nothing, kwargs...) -> (g, meta)

C+-backend (production default) gradient entry point, fixed Fréchet
marginals. Requires `fpcx.fctx !== nothing` (Architecture C / `:structured`
Hessian backend).
"""
function cm_frechet_production_gradient_cplus(x_free0::AbstractVector, fpcx, ctx, pe, pool::GradWorkspacePool,
        ws::LFixFactorizedWorkspace; base::Union{Nothing,BaseDualState} = nothing, kwargs...)
    fpcx.mode === :frechet_reference || error("cm_frechet_production_gradient_cplus: fpcx.mode=$(fpcx.mode), expected :frechet_reference")
    fpcx.fctx !== nothing || error("cm_frechet_production_gradient_cplus: requires cm_hessian_backend=:structured (Architecture C)")
    base = base === nothing ? archC_frechet_base_state(x_free0, fpcx.ctx_cm, fpcx.fctx) : base
    cache = build_lfix_base_cache_cm_frechet_C!(ws, x_free0, fpcx.ctx_cm, base, ctx, fpcx.aug, fpcx.bins)
    return composite_gradient_at_Cplus_from_cache(x_free0, fpcx.ctx_cm, pe, pool, cache; base = base, kwargs...)
end
