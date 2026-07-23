# ============================================================================
# Origin-specific-ZC (no CM) production bundle: inner solve, Hessian, gradient.
# Companion to cm_originzc_moments.jl. Reuses, UNCHANGED:
#   - Architecture A's generic dense-BLAS Hessian (`archA_hess_cb_builder` =
#     `_callbackEvalH_inner_profiled!`, cm_hessian_architectures.jl) -- this
#     arm has NO CM-grid block, so there is no bin/suffix-sum trick to build;
#     Architecture A is NCORE-only and CM-agnostic by construction (confirmed:
#     it is exactly what CMConfig's `cm_hessian_backend=:dense_reference`
#     already resolves to, and what the plain unrestricted D=20 production
#     driver's own inner solve already uses at n~=402 -- widening it to
#     n~=600-800 for K<=2 mean/pair columns is the SAME architecture at a
#     modestly larger width, not new machinery).
#   - inner_loop_internal_archgeneric / inner_loop_KNITRO_archgeneric
#     (cm_hessian_architectures.jl) -- called with theta_ext =
#     vcat(theta_econ, nu_1,...,nu_{n_eta(layout)}).
#   - composite_gradient_at_fast / composite_gradient_at_Cplus_from_cache --
#     the (g,A_od) outer gradient is computed EXACTLY as in the CM-only path,
#     at every nu_{o,k} held fixed, folded into q0 once per outer evaluation
#     via originzc_fixed_contribution (cm_originzc_moments.jl).
# CMExpectedSolveFailure (cm_production_bundle.jl) is reused for the same
# expected-failure signal, not redefined.
# ============================================================================

using LinearAlgebra: BLAS, dot, norm

"""
    build_originzc_production_context(ctx, CS, layout) -> (ctx_cm, aug)

Analog of `build_cm_meanzc_production_context`, minus the CM-specific
`cctx`/`bins` (there is no CM block for this arm, hence nothing to
precompute for it). `ctx_cm.obj` is `aug.obj_cm`.
"""
function build_originzc_production_context(ctx, CS, layout::MeanZCTargetLayout)
    aug = build_originzc_augmented_obj(ctx, CS, layout)
    ctx_cm = merge(ctx, (obj = aug.obj_cm,))
    return (ctx_cm = ctx_cm, aug = aug)
end

"""
    archOZ_base_state(x_free0, νfull, ctx_cm) -> BaseDualState

Architecture-A inner dual solve at outer point `(x_free0, νfull)`, `νfull`
already exponentiated, length `n_eta(layout)`. Mirrors `archC_meanzc_base_state`
exactly except for the Hessian callback (`archA_hess_cb_builder`, generic --
no `cctx` argument needed).
"""
function archOZ_base_state(x_free0::AbstractVector, νfull::AbstractVector{Float64}, ctx_cm)
    obj = ctx_cm.obj
    θ_econ0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    θ_ext0 = vcat(θ_econ0, νfull)
    K, x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_ext0; hess_cb_builder = archA_hess_cb_builder)
    nStatus in (0, -100, -101, -103) || throw(CMExpectedSolveFailure("archOZ_base_state: inner solve failed, nStatus=$nStatus (x_free0=$x_free0, ν=$νfull)"))
    ζstar = x[1]; λstar = collect(x[2:end])
    return BaseDualState(collect(x_free0), θ_econ0, ζstar, λstar, copy(obj.arg1), nStatus)
end

"""
    archOZ_verified_state(x_free0, νfull, ctx_cm) -> (base::BaseDualState, verify::NamedTuple)

AUD-04-style verified analog of `archOZ_base_state`, mirroring
`archC_meanzc_verified_state`.
"""
function archOZ_verified_state(x_free0::AbstractVector, νfull::AbstractVector{Float64}, ctx_cm)
    obj = ctx_cm.obj
    θ_econ0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    θ_ext0 = vcat(θ_econ0, νfull)
    K, inner_x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_ext0; hess_cb_builder = archA_hess_cb_builder)
    nStatus in (0, -100, -101, -103) || throw(CMExpectedSolveFailure("archOZ_verified_state: inner solve failed, nStatus=$nStatus (x_free0=$x_free0, ν=$νfull)"))

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

    base = BaseDualState(collect(x_free0), θ_econ0, ζstar, λstar, m_weights, nStatus)
    verify = (inner_status = nStatus, Delta_dual = Delta_dual, Delta_primal = Delta_primal,
              primal_dual_gap = abs(Delta_dual - Delta_primal),
              weight_norm_resid = abs(sum(p_weights) - 1.0),
              mean_m_resid = mean_m_resid, max_abs_moment_kkt_resid = max_abs_moment_kkt_resid,
              m_mean = sum(m_weights) / W, m_min = minimum(m_weights), m_max = maximum(m_weights))
    return base, verify
end

"cm_originzc_production_value(x_free0, νfull, pcx) -> (K, base). Inner solve only, no gradient."
function cm_originzc_production_value(x_free0::AbstractVector, νfull::AbstractVector{Float64}, pcx)
    base = archOZ_base_state(x_free0, νfull, pcx.ctx_cm)
    K = pcx.ctx_cm.obj.H_save
    return K, base
end

"cm_originzc_production_value_verified(x_free0, νfull, pcx) -> (K, base, verify). AUD-04-gated analog."
function cm_originzc_production_value_verified(x_free0::AbstractVector, νfull::AbstractVector{Float64}, pcx)
    base, verify = archOZ_verified_state(x_free0, νfull, pcx.ctx_cm)
    K = pcx.ctx_cm.obj.H_save
    return K, base, verify
end

"""
    build_lfix_base_cache_originzc(x_free0, ctx_cm, base, aug, νfull; validate_dense=false) -> LFixBaseCache

No-CM analog of `build_lfix_base_cache_cm_meanzc`: folds ONLY
`originzc_fixed_contribution` into `q0` (no CM-grid block exists for this
arm).
"""
function build_lfix_base_cache_originzc(x_free0::AbstractVector, ctx_cm, base::BaseDualState,
                                         aug, νfull::AbstractVector{Float64}; validate_dense::Bool = false)
    cache0 = build_lfix_base_cache(x_free0, ctx_cm, base; validate_dense = validate_dense)
    contrib0 = originzc_fixed_contribution(base, aug, νfull)
    return with_q0(cache0, cache0.q0 .- contrib0)
end

"""
    cm_originzc_production_gradient(x_free0, νfull, pcx, ctx, pe; base=nothing, verify=nothing, kwargs...) -> (g_ext, meta)

`:reference`-backend entry point for this arm's full outer gradient: the
(g,A_od) block via UNCHANGED `composite_gradient_at_fast` (every nu_{o,k}
held fixed, folded into q0 once), PLUS the analytic
`d(Delta_dual)/d(eta_{o,k})` vector appended as the last `n_eta(layout)`
components. `g_ext` has length `D^2 + n_eta(layout)`.
"""
function cm_originzc_production_gradient(x_free0::AbstractVector, νfull::AbstractVector{Float64}, pcx, ctx, pe;
        base::Union{Nothing,BaseDualState} = nothing, verify = nothing, kwargs...)
    if base === nothing || verify === nothing
        base, verify = archOZ_verified_state(x_free0, νfull, pcx.ctx_cm)
    end
    cache = build_lfix_base_cache_originzc(x_free0, pcx.ctx_cm, base, pcx.aug, νfull)
    g_econ, meta = composite_gradient_at_fast(x_free0, pcx.ctx_cm, pe; base = base, cache = cache, kwargs...)
    d_eta = d_delta_dual_d_eta_origin_vec(base.λstar, pcx.aug, νfull; mean_m = verify.m_mean)
    return vcat(g_econ, d_eta), meta
end

# ----------------------------------------------------------------------------
# Validation ground truth for the (g, A_od) gradient block -- no-CM analog of
# fixed_dual_L_meanzc / full_rebuild_gradient_fallback_meanzc.
# ----------------------------------------------------------------------------

"""
    fixed_dual_L_originzc(x_free, νfull, ctx_cm, base::BaseDualState) -> Float64

No-CM analog of `fixed_dual_L_meanzc`: the fixed-dual (NOT reoptimized)
divergence value at a perturbed `(x_free, νfull)`, holding `base.ζstar`/
`base.λstar` fixed. Rebuilds the FULL moment matrix via `obj.moments!` every
call.
"""
function fixed_dual_L_originzc(x_free::AbstractVector, νfull::AbstractVector{Float64}, ctx_cm, base::BaseDualState)
    obj = ctx_cm.obj
    θ_econ = CS.reconstruct_full(x_free, ctx_cm.m)
    θ_ext = vcat(θ_econ, νfull)
    W = size(obj.U, 1); d = obj.d
    K = zeros(eltype(θ_ext), W); G = zeros(eltype(θ_ext), W, d)
    obj.moments!(K, G, θ_ext, obj.U, obj)
    oci = obj.outer_constr_index
    q = [-base.ζstar - dot(base.λstar, @view(G[s, 1:oci-1])) for s in 1:W]
    Psi_q = similar(q)
    CS.Psi!(Psi_q, q)
    return -(sum(Psi_q) / W + base.ζstar)
end

"""
    d_delta_dual_d_eta_origin_fd(x_free0, νfull, ctx_cm, aug, base; h=1e-4) -> Vector{Float64}

Reoptimized (NOT fixed-dual) central finite difference of `Delta_dual`
w.r.t. every `eta_{o,k}` at fixed economic outer point -- the trusted ground
truth for validating `d_delta_dual_d_eta_origin_vec` (task brief Section 7:
"validate every analytic eta_{o,k} derivative ... against independently
reoptimized central finite differences"). Each probe re-solves the inner
dual from scratch at the perturbed `eta`; NOT a fixed-dual/full-rebuild
shortcut.
"""
function d_delta_dual_d_eta_origin_fd(x_free0::AbstractVector, νfull::AbstractVector{Float64}, ctx_cm; h::Float64 = 1e-4)
    n = length(νfull)
    g = Vector{Float64}(undef, n)
    η = log.(νfull)
    for j in 1:n
        ηp = copy(η); ηp[j] += h
        ηm = copy(η); ηm[j] -= h
        _, _, vp = cm_originzc_value_verified_from_eta(x_free0, ηp, ctx_cm)
        _, _, vm = cm_originzc_value_verified_from_eta(x_free0, ηm, ctx_cm)
        g[j] = (vp.Delta_dual - vm.Delta_dual) / (2h)
    end
    return g
end

"Helper: verified inner solve directly from an eta vector (exponentiates internally)."
function cm_originzc_value_verified_from_eta(x_free0::AbstractVector, η::AbstractVector{Float64}, ctx_cm)
    base, verify = archOZ_verified_state(x_free0, exp.(η), ctx_cm)
    return ctx_cm.obj.H_save, base, verify
end
