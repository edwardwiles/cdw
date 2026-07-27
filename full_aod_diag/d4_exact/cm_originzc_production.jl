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

# port/shared-inner-fg-operator-and-verification-2026-07-26: opt-in operator FG (`_originzc_fg_dispatch`,
# fg_backend=:operator on OriginZCCoreHessCtx) -- self-guarded include, this codebase's own convention.
isdefined(Main, :_originzc_fg_dispatch) || include(joinpath(@__DIR__, "cm_originzc_lookup_production.jl"))
isdefined(Main, :verify_inner_solution_operator_originzc!) || include(joinpath(@__DIR__, "operator_verification.jl"))   # verification-defaults task (2026-07-27): archOZ_verified_state's :operator backend below
# shared outer-A-gradient task (2026-07-27): shared_a_gradient.jl provides economic_A_gradient!/
# EconomicAGradientWorkspace, this arm's DEFAULT (g,A_od)-block gradient backend (see
# cm_originzc_production_gradient below).
isdefined(Main, :EconomicAGradientWorkspace) || include(joinpath(@__DIR__, "shared_a_gradient.jl"))

# get_or_build_econ_a_grad_ws (shared-FG-verification-and-A-gradient release, 2026-07-27): moved
# to shared_a_gradient.jl -- it was ZC-only-only in name only (nothing in its own body is
# ZC-specific), and every family wiring onto economic_A_gradient! needs the identical process-wide
# per-W cache, not a per-family copy. See that file's own docstring for the full rationale.

"""
    build_originzc_production_context(ctx, CS, layout) -> (ctx_cm, aug)

Analog of `build_cm_meanzc_production_context`, minus the CM-specific
`cctx`/`bins` (there is no CM block for this arm, hence nothing to
precompute for it). `ctx_cm.obj` is `aug.obj_cm`.
"""
function build_originzc_production_context(ctx, CS, layout::MeanZCTargetLayout; fg_backend::Symbol = ORIGINZC_FG_BACKEND_DEFAULT[])
    println(stdout, "cm_restriction_basis [origin-ZC] = none (no CM-grid block; origin-specific mean/pairwise-ZC targets only)")
    println(stdout, "cm_internal_feature_storage [origin-ZC] = none (no bin indices -- raw Zraw_all/Zpairraw_all power features only)")
    println(stdout, "origin_fg_backend [origin-ZC] = ", fg_backend, " (port/shared-inner-fg-operator-and-verification-2026-07-26)")
    flush(stdout)
    isdefined(Main, :record_cm_feature_context_build!) && record_cm_feature_context_build!()   # Phase 3 (2026-07-26): CM feature immutability counters
    aug = build_originzc_augmented_obj(ctx, CS, layout)
    # port/shared-winner-pair-core-hessian-production-2026-07-25 (task §4.4): `octx` rides on
    # `ctx_cm` itself (rather than as a new positional argument to
    # `archOZ_base_state`/`archOZ_verified_state`) so every EXISTING caller of those two
    # functions across the codebase (cm_screen_bridge.jl, cm_originzc_profile.jl,
    # cm_originzc_cplus.jl, and a dozen+ diagnostic/test scripts) needs zero signature-call
    # changes and automatically picks up the shared H_EE backend.
    octx = build_originzc_core_hess_ctx(aug; fg_backend = fg_backend)
    ctx_cm = merge(ctx, (obj = aug.obj_cm, octx = octx))
    return (ctx_cm = ctx_cm, aug = aug, octx = octx)
end

"port/shared-winner-pair-core-hessian-production-2026-07-25: resolves to the shared-H_EE partitioned callback when `ctx_cm` carries an `octx` (every current production caller does, via `build_originzc_production_context`), else the original unpartitioned dense Architecture A (a caller that built `ctx_cm` some other way, or a diagnostic script that never rebuilt it after this port)."
_originzc_hess_cb_builder(ctx_cm) = hasproperty(ctx_cm, :octx) ? archA_partitioned_hess_cb_builder(ctx_cm.octx) : archA_hess_cb_builder(ctx_cm.obj)

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
    K, x, nStatus, n_fg, n_hess = _originzc_fg_dispatch(ctx_cm, obj, θ_ext0)
    nStatus in (0, -100, -101, -103) || throw(CMExpectedSolveFailure("archOZ_base_state: inner solve failed, nStatus=$nStatus (x_free0=$x_free0, ν=$νfull)"))
    ζstar = x[1]; λstar = collect(x[2:end])
    return BaseDualState(collect(x_free0), θ_econ0, ζstar, λstar, copy(obj.arg1), nStatus)
end

"""
    archOZ_verified_state(x_free0, νfull, ctx_cm) -> (base::BaseDualState, verify::NamedTuple)

AUD-04-style verified analog of `archOZ_base_state`, mirroring
`archC_meanzc_verified_state`.
"""
function archOZ_verified_state(x_free0::AbstractVector, νfull::AbstractVector{Float64}, ctx_cm;
        dual_bank::Union{Nothing,RestrictedDualBank} = nothing, eval_id::Int = 0,
        verification_backend::Symbol = ORIGINZC_VERIFICATION_BACKEND_DEFAULT[])
    obj = ctx_cm.obj
    θ_econ0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    θ_ext0 = vcat(θ_econ0, νfull)
    warm_label = :unset
    if dual_bank !== nothing
        x0, warm_label, _ = select_warm_start_restricted(dual_bank, obj, vcat(collect(x_free0), νfull))
        obj.x = x0
        warm_label == :neutral ? (RESTRICTED_DUAL_BANK_COUNTERS[].cold_inner_solves += 1) :
                                  (RESTRICTED_DUAL_BANK_COUNTERS[].warm_inner_solves += 1)
    end
    K, inner_x, nStatus, n_fg, n_hess = _originzc_fg_dispatch(ctx_cm, obj, θ_ext0)
    if nStatus ∉ (0, -100, -101, -103)
        dual_bank !== nothing && warm_label != :neutral && (RESTRICTED_DUAL_BANK_COUNTERS[].warm_start_failures += 1)
        throw(CMExpectedSolveFailure("archOZ_verified_state: inner solve failed, nStatus=$nStatus (x_free0=$x_free0, ν=$νfull)"))
    end

    ζstar = inner_x[1]; λstar = collect(inner_x[2:end])
    W = size(obj.U, 1)

    local m_weights, verify
    if verification_backend === :operator
        # Verification-defaults task (2026-07-27): operator-based post-solve verification, G=[E|Z]
        # via the shared economic/ZC operators -- no dense obj.H read. Requires ctx_cm.octx to have
        # been built with fg_backend=:operator (ORIGINZC_FG_BACKEND_DEFAULT[] already defaults to
        # :operator in production, so octx.fg_zc_op/fg_layout are populated by default); a hard
        # error (no silent fallback) if that prerequisite isn't met.
        octx = ctx_cm.octx
        cf = octx.core_cf_ref[]
        cf isa CompressedFactual || error("archOZ_verified_state: verification_backend=:operator requires ctx_cm.octx.core_cf_ref[] to be a CompressedFactual (got $(typeof(cf))) -- prerequisite not met, refusing silent dense fallback")
        octx.fg_zc_op !== nothing || error("archOZ_verified_state: verification_backend=:operator requires ctx_cm.octx.fg_zc_op to be built (octx was built with fg_backend=:dense_reference) -- prerequisite not met, refusing silent dense fallback")
        ov = verify_inner_solution_operator_originzc!(ζstar, λstar, cf, octx.fg_zc_op, octx.fg_layout, νfull, obj, W)
        m_weights, verify = verify_namedtuple_from_operator(ov, obj, W, nStatus)
    elseif verification_backend === :dense_reference
        G = CS.select_G_from_H(obj, obj.H)

        ncon = obj.d - obj.outer_constr_index + 2
        cbuf = zeros(ncon)
        obj(inner_x, constr = @view(cbuf[1:ncon]))
        Delta_dual = cbuf[1] / 1e10
        m_weights = copy(obj.arg1)
        # Allocation fix (shared outer-A-gradient task, 2026-07-27, task §10): non-allocating
        # weight_norm_resid -- see cm_production_bundle.jl's identical fix for the full rationale and
        # the bit-identity verification.
        s_m_weights = sum(m_weights)
        Delta_primal = primal_divergence(m_weights)

        mean_m_resid = abs(sum(m_weights) / W - 1.0)
        nkkt = min(length(λstar), size(G, 2))
        max_abs_moment_kkt_resid = kkt_residual_blas(G, m_weights, nkkt, W)

        verify = (inner_status = nStatus, Delta_dual = Delta_dual, Delta_primal = Delta_primal,
                  primal_dual_gap = abs(Delta_dual - Delta_primal),
                  weight_norm_resid = abs(sum(x -> x / s_m_weights, m_weights) - 1.0),
                  mean_m_resid = mean_m_resid, max_abs_moment_kkt_resid = max_abs_moment_kkt_resid,
                  m_mean = sum(m_weights) / W, m_min = minimum(m_weights), m_max = maximum(m_weights))
        record_dense_reference_verification!()
    else
        error("archOZ_verified_state: unknown verification_backend=:$verification_backend (expected :operator or :dense_reference)")
    end

    base = BaseDualState(collect(x_free0), θ_econ0, ζstar, λstar, m_weights, nStatus)
    dual_bank !== nothing && record_success_restricted!(dual_bank, eval_id, vcat(collect(x_free0), νfull), inner_x)
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
    cm_originzc_production_gradient(x_free0, νfull, pcx, ctx, pe; base=nothing, verify=nothing,
                                     gradient_backend=:shared_inplace_pooled, econ_ws=nothing, kwargs...) -> (g_ext, meta)

Full outer gradient for this arm: the (g,A_od) block, PLUS the analytic
`d(Delta_dual)/d(eta_{o,k})` vector appended as the last `n_eta(layout)` components. `g_ext` has
length `D*Ddest + n_eta(layout)`.

`gradient_backend` (shared outer-A-gradient task, 2026-07-27): controls how the (g,A_od) block is
computed.
  - `:shared_inplace_pooled` (DEFAULT): the shared `economic_A_gradient!` entry point
    (shared_a_gradient.jl) -- writes directly into a preallocated buffer, uses the fixed-2-slot
    `TwoOriginScratch` (no Dict, no per-coordinate W-length allocation for the winner-flip/2-origin
    same-destination cases). Verified bit-for-bit identical to `:legacy_unbuffered` at D=4 and real
    D=20/W=80,000 (test_shared_a_gradient.jl / test_cm_originzc_shared_a_gradient_gate.jl).
  - `:legacy_unbuffered`: the ORIGINAL, fully-allocating `composite_gradient_at_fast` -- kept ONLY
    as an explicit reference/debug backend (task requirement: "no restricted-family production
    wrapper may call the original unbuffered composite_gradient_at_fast except through an explicit
    reference/debug backend"). Not used by any default call site after this port.

`econ_ws`: an `EconomicAGradientWorkspace` to reuse across calls (task §5: construct ONCE per live
outer-solver context, never per gradient call). If not supplied, a process-wide cache keyed by `W`
is used (`get_or_build_econ_a_grad_ws`, cm_originzc_production.jl) -- a pragmatic, disclosed
approximation of "one workspace per live outer-solver context" (see that function's own docstring
for the caveat: this shares one workspace across all concurrently-running contexts of the same W,
which is safe as long as this arm's own outer solver never calls two of ITS OWN gradients
concurrently -- true today, no threaded multi-context production driver exists for this arm).
"""
function cm_originzc_production_gradient(x_free0::AbstractVector, νfull::AbstractVector{Float64}, pcx, ctx, pe;
        base::Union{Nothing,BaseDualState} = nothing, verify = nothing,
        gradient_backend::Symbol = :shared_inplace_pooled,
        econ_ws::Union{Nothing,EconomicAGradientWorkspace} = nothing, kwargs...)
    if base === nothing || verify === nothing
        base, verify = archOZ_verified_state(x_free0, νfull, pcx.ctx_cm)
    end
    cache = build_lfix_base_cache_originzc(x_free0, pcx.ctx_cm, base, pcx.aug, νfull)
    if gradient_backend === :shared_inplace_pooled
        D = pcx.ctx_cm.D; Ddest = hasproperty(pcx.ctx_cm, :D_dest) ? pcx.ctx_cm.D_dest : pcx.ctx_cm.D
        ws = econ_ws === nothing ? get_or_build_econ_a_grad_ws(cache.W) : econ_ws
        g_econ = zeros(D * Ddest)
        meta = economic_A_gradient!(g_econ, base, pcx.ctx_cm, pe, ws; cache = cache, kwargs...)
    elseif gradient_backend === :legacy_unbuffered
        g_econ, meta = composite_gradient_at_fast(x_free0, pcx.ctx_cm, pe; base = base, cache = cache, kwargs...)
    else
        error("cm_originzc_production_gradient: gradient_backend must be :shared_inplace_pooled|:legacy_unbuffered, got $gradient_backend")
    end
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
