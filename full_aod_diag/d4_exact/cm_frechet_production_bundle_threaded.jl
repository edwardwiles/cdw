# Threaded/syrk CDF-only base-state / verified-state entry points. ADDITIVE only -- does not
# modify cm_frechet_production_bundle.jl's existing archC_frechet_base_state/
# archC_frechet_verified_state (the CDF+POWER default driver, run_frechet_upper.jl, and every
# existing test/gate keep calling those, byte-identical). Structural twins, tail logic copied
# verbatim from cm_frechet_production_bundle.jl (matching that file's own established pattern of
# duplicating the tail rather than parameterizing across a shared closure -- see
# cm_hessian_threaded.jl's header comment for the same discipline applied to the flexible-CM case).
# Task: FIXED_FRECHET_INNER_SOLVER production-feasibility 2026-07-24 (CDF-only addendum).
#
# Requires cm_frechet_hessian_threaded.jl (archC_frechet_hess_cb_builder_v2) and
# cm_frechet_production_bundle.jl (_frechet_require_feasible, frechet_solve_outcome) included first.

"Threaded/syrk analogue of `archC_frechet_base_state`, CDF-only fixed Fréchet. `tls` from `build_thread_local_scratch(fctx.cctx)`."
function archC_frechet_base_state_threaded(x_free0::AbstractVector, ctx_cm, fctx, tls::ThreadLocalBinScratch)
    obj = ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    K, x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_full0;
        hess_cb_builder = _obj -> archC_frechet_hess_cb_builder_v2(fctx; threaded_bins = true, tls = tls, use_syrk = true))
    _frechet_require_feasible(nStatus, "archC_frechet_base_state_threaded", "(x_free0=$x_free0)")
    ζstar = x[1]; λstar = collect(x[2:end])
    return BaseDualState(collect(x_free0), θ_full0, ζstar, λstar, copy(obj.arg1), nStatus)
end

"Threaded/syrk analogue of `archC_frechet_verified_state`, CDF-only fixed Fréchet."
function archC_frechet_verified_state_threaded(x_free0::AbstractVector, ctx_cm, fctx, tls::ThreadLocalBinScratch)
    obj = ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    K, inner_x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_full0;
        hess_cb_builder = _obj -> archC_frechet_hess_cb_builder_v2(fctx; threaded_bins = true, tls = tls, use_syrk = true))
    _frechet_require_feasible(nStatus, "archC_frechet_verified_state_threaded", "(x_free0=$x_free0)")

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
    verify = (inner_status = nStatus, outcome = frechet_solve_outcome(nStatus),
              Delta_dual = Delta_dual, Delta_primal = Delta_primal,
              primal_dual_gap = abs(Delta_dual - Delta_primal),
              weight_norm_resid = abs(sum(p_weights) - 1.0),
              mean_m_resid = mean_m_resid, max_abs_moment_kkt_resid = max_abs_moment_kkt_resid,
              m_mean = sum(m_weights) / W, m_min = minimum(m_weights), m_max = maximum(m_weights))
    return base, verify
end

"""
    cm_frechet_verified_state_threaded(x_free0, fpcx, tls) -> (base, verify)

CDF-only-only dispatcher (errors if `fpcx.cfg.frechet_feature_set !== :cdf_only`) -- this
threaded path is not built for `:common_flexible` or `:cdf_power` in this task.
"""
function cm_frechet_verified_state_threaded(x_free0::AbstractVector, fpcx, tls::ThreadLocalBinScratch)
    fpcx.cfg.frechet_feature_set === :cdf_only ||
        error("cm_frechet_verified_state_threaded: only :cdf_only is supported (got $(fpcx.cfg.frechet_feature_set))")
    return archC_frechet_verified_state_threaded(x_free0, fpcx.ctx_cm, fpcx.fctx, tls)
end

function cm_frechet_base_state_threaded(x_free0::AbstractVector, fpcx, tls::ThreadLocalBinScratch)
    fpcx.cfg.frechet_feature_set === :cdf_only ||
        error("cm_frechet_base_state_threaded: only :cdf_only is supported (got $(fpcx.cfg.frechet_feature_set))")
    return archC_frechet_base_state_threaded(x_free0, fpcx.ctx_cm, fpcx.fctx, tls)
end
