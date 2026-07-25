# ============================================================================
# Fixed-Fréchet-marginals production context dispatcher + base-state /
# gradient entry points (port-prep 2026-07-24, off production/fullA-exact
# @ c55e81e). Structural twin of `cm_production_bundle.jl`'s
# `build_cm_production_context`, dispatching on `CMFrechetConfig` instead of
# `CMConfig`, and extended with `frechet_feature_set` dispatch
# (:cdf_only -> the ported CDF-only Architecture-C kernel, cm_frechet_hessian.jl;
# :cdf_power -> the NEW combined structured kernel,
# cm_frechet_power_hessian_structured.jl).
#
# Consumes the SAME core-moment/winner-state interface flexible-CM does (see
# docs/CM_PRODUCTION_HOOK_INTERFACE_SPEC_2026-07-24.md): the plain `ctx`
# NamedTuple (`.obj::PsiObjectiveBundleImplicit`, `.D`/`.D_dest` for the
# rectangular layout), `wrap_moments_with_cm`-style composition, and
# `cm_production_gradient_cplus`'s C+ wiring pattern -- no separate
# fixed-Fréchet-specific winner scanner is introduced (task brief §4/§9).
#
# ---- Timeout / state-reuse discipline (task brief §8) ----
# Production's `knitro_status.jl` already provides a typed classifier
# (`decode_knitro_status`, `.category` in `:optimal`/`:feasible_approx`/
# `:limit_feasible`/`:limit_infeasible`/`:infeasible`/...). Every entry point
# below that used to hard-`nStatus in (0,-100,-101,-103)`-or-throw in the
# pre-omit-ROW archive now goes through `frechet_solve_outcome` (defined at
# the bottom of this file), which explicitly separates a TIME/ITER-LIMIT
# result with NO certificate (`:time_limit_no_certificate`,
# `.category===:limit_infeasible`, i.e. the solver ran out of budget WITHOUT
# ever finding a feasible point) from a genuine infeasibility/unboundedness
# certificate (`:infeasible_certificate`) and from an ordinary feasible
# result (`:feasible`, covering `:optimal`/`:feasible_approx`/
# `:limit_feasible` -- KNITRO hit a limit but DID return a feasible point,
# which is a normal, usable, cold-verifiable result, not a failure). NEVER
# `nStatus in (0,-100,-101,-103) || throw(...)` anywhere in this file --
# that pattern silently reclassifies a `:limit_feasible` timeout that
# returned a perfectly good feasible point as an unconditional hard error,
# which is exactly backwards for L=50-scale solves (see
# docs/FIXED_FRECHET_TIMEOUT_AND_STATE_REUSE_AUDIT_2026-07-24.md for the
# archive-branch incident this is designed to prevent from recurring here).
#
# Accepted-point base-state reuse: every gradient/Hessian entry point below
# accepts `base::Union{Nothing,BaseDualState}=nothing` and ONLY re-solves
# when `base===nothing` -- a caller (any future outer-driver built on this
# branch) that already holds a cold-verified `BaseDualState` for the EXACT
# point being queried must pass it through, never let it fall back to
# `nothing` after having computed it, and must NOT invent a materially
# shorter trial-solve time bound than the one used to build the checkpoint's
# own warm-start/cold-verify budgets -- no `*_trial_10s.opt`-style option
# file is introduced anywhere in this branch.
# ============================================================================

"""
    frechet_solve_outcome(nStatus) -> Symbol

Typed classification of an inner KNITRO solve's `nStatus`, task brief §8:
- `:feasible` -- a feasible point was returned (`:optimal`/`:feasible_approx`/
  `:limit_feasible`, i.e. `KnitroStatusInfo.is_feasible_result == true`).
  Includes ordinary time/iteration-limit-but-feasible results -- these are
  NOT failures.
- `:infeasible_certificate` -- `:infeasible`/`:unbounded` categories, a
  genuine structural certificate.
- `:time_limit_no_certificate` -- `:limit_infeasible` (time/iter/feval limit
  reached with NO feasible point ever found) -- numerical-unknown, MUST NOT
  be classified as infeasible or given Δ*=∞.
- `:solver_error` -- `:error`/`:unknown` categories.
- `:exact_screen_certificate` -- this repo's own pre-solve exact-screen
  sentinel codes (never reached KN_solve at all).
"""
function frechet_solve_outcome(nStatus::Integer)
    info = decode_knitro_status(nStatus)
    info.is_feasible_result && return :feasible
    info.category === :exact_screen_certificate && return :exact_screen_certificate
    info.category === :limit_infeasible && return :time_limit_no_certificate
    info.category in (:infeasible, :unbounded) && return :infeasible_certificate
    return :solver_error
end

"Throws iff `nStatus` is NOT a usable feasible result; carries the typed outcome symbol in the message so a caller can grep it rather than re-decoding."
function _frechet_require_feasible(nStatus::Integer, where::AbstractString, extra::AbstractString = "")
    outcome = frechet_solve_outcome(nStatus)
    outcome === :feasible && return outcome
    info = decode_knitro_status(nStatus)
    throw(CMExpectedSolveFailure("$where: inner solve did not return a feasible point, " *
        "nStatus=$nStatus (outcome=$outcome, category=$(info.category): $(info.meaning)) $extra"))
end

# ============================================================================
# CDF-only (frechet_feature_set=:cdf_only) base-state / verified-state entry points
# ============================================================================

"Architecture-C base-state solve for fixed Fréchet marginals, CDF-only -- structural twin of `archC_base_state`."
function archC_frechet_base_state(x_free0::AbstractVector, ctx_cm, fctx)
    obj = ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    K, x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_full0;
        hess_cb_builder = _obj -> archC_frechet_hess_cb_builder(fctx))
    _frechet_require_feasible(nStatus, "archC_frechet_base_state", "(x_free0=$x_free0)")
    ζstar = x[1]; λstar = collect(x[2:end])
    return BaseDualState(collect(x_free0), θ_full0, ζstar, λstar, copy(obj.arg1), nStatus)
end

"""
    archC_frechet_verified_state(x_free0, ctx_cm, fctx) -> (base, verify)

CDF-only cold-verify analog of `archC_verified_state` -- same independent-recompute
diagnostics (`Delta_dual`, `Delta_primal`, `primal_dual_gap`, `weight_norm_resid`,
`max_abs_moment_kkt_resid`, ...), reusing `primal_divergence`/`kkt_residual_blas` unchanged.
"""
function archC_frechet_verified_state(x_free0::AbstractVector, ctx_cm, fctx)
    obj = ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    K, inner_x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_full0;
        hess_cb_builder = _obj -> archC_frechet_hess_cb_builder(fctx))
    _frechet_require_feasible(nStatus, "archC_frechet_verified_state", "(x_free0=$x_free0)")

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

# ============================================================================
# CDF+POWER (frechet_feature_set=:cdf_power, DEFAULT) base-state / verified-state
# entry points -- NEW, use the combined structured Hessian (FrechetPowerBinHessCtx)
# ============================================================================

"Architecture-C base-state solve for fixed Fréchet marginals, CDF+POWER (default feature set)."
function archC_frechet_cdf_power_base_state(x_free0::AbstractVector, ctx_cm, fctx::FrechetPowerBinHessCtx)
    obj = ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    K, x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_full0;
        hess_cb_builder = _obj -> archC_frechet_cdf_power_hess_cb_builder(fctx))
    _frechet_require_feasible(nStatus, "archC_frechet_cdf_power_base_state", "(x_free0=$x_free0)")
    ζstar = x[1]; λstar = collect(x[2:end])
    return BaseDualState(collect(x_free0), θ_full0, ζstar, λstar, copy(obj.arg1), nStatus)
end

"CDF+POWER cold-verify analog, structural twin of `archC_frechet_verified_state`."
function archC_frechet_cdf_power_verified_state(x_free0::AbstractVector, ctx_cm, fctx::FrechetPowerBinHessCtx)
    obj = ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    K, inner_x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_full0;
        hess_cb_builder = _obj -> archC_frechet_cdf_power_hess_cb_builder(fctx))
    _frechet_require_feasible(nStatus, "archC_frechet_cdf_power_verified_state", "(x_free0=$x_free0)")

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

# ============================================================================
# Top-level dispatcher
# ============================================================================

"""
    build_cm_frechet_production_context(ctx, CS, cfg::CMFrechetConfig; L=cfg.cm.cm_grid_size) -> NamedTuple

`marginal_mode=:common_flexible` delegates to the EXISTING
`build_cm_production_context` unchanged (byte-identical results,
`targets=nothing`). `marginal_mode=:frechet_reference` builds the fixed-
Fréchet augmented objective, dispatching on `cfg.frechet_feature_set`:
  - `:cdf_only` -- Architecture B moments (fast per-call, ported unchanged
    from the pre-omit-ROW archive) + Architecture C structured Hessian
    (`cm_frechet_hessian.jl`).
  - `:cdf_power` (DEFAULT) -- Architecture A (dense) moment construction
    (`cm_frechet_bases.jl`'s `build_cm_frechet_augmented_obj_basis`, basis =
    `cfg.frechet_basis`) + the NEW Architecture C structured Hessian
    (`cm_frechet_power_hessian_structured.jl`). A fast Architecture-B moment
    path for the combined block (avoiding the persistent dense `W x 2DL`
    CM matrix) is NOT built in this port-prep pass -- see
    FIXED_FRECHET_POST_OMIT_ROW_PORT_READINESS_2026-07-24.md for the
    disclosed follow-up. The Hessian (the expensive per-callback object at
    L=50 scale) IS fast/structured in both cases.

Propagates `obj0.threshold_state` through every reconstructed
`PsiObjectiveBundleImplicit` (the 2026-07-24 release fix flexible-CM's own
`build_cm_production_context` already applies -- see
docs/CM_PRODUCTION_HOOK_INTERFACE_SPEC_2026-07-24.md §6) so the threshold-10
auto-reject gate never silently goes dark for the fixed-Fréchet path.
"""
function build_cm_frechet_production_context(ctx, CS, cfg::CMFrechetConfig; L::Int = cfg.cm.cm_grid_size)
    _cm_frechet_validate(cfg)
    if cfg.marginal_mode === :common_flexible
        pcx = build_cm_production_context(ctx, CS; L = L, contrasts = cfg.cm.contrasts)
        return (ctx_cm = pcx.ctx_cm, aug = pcx.aug, cctx = pcx.cctx, bins = pcx.bins,
                cfg = cfg, L = L, targets = nothing, mode = :common_flexible, fctx = nothing)
    end

    targets = build_frechet_reference_targets(ctx, cfg; L = L)
    report_frechet_targets(ctx, cfg, targets)

    if cfg.frechet_feature_set === :cdf_only
        aug = build_cm_frechet_augmented_obj_archB(ctx, CS, targets; contrasts = cfg.cm.contrasts)
        obj_cm = aug.obj_cm
        obj_cm2 = CS.PsiObjectiveBundleImplicit(δ = obj_cm.δ, find_smallest = obj_cm.find_smallest,
            γ = obj_cm.γ, (moments!) = obj_cm.moments!, moments_jacobian! = error,
            d = obj_cm.d, outer_constr_index = obj_cm.outer_constr_index,
            inequality_index = obj_cm.inequality_index, complement_index = obj_cm.complement_index,
            l = obj_cm.l, U = obj_cm.U, N = obj_cm.N, lower_limit = obj_cm.lower_limit,
            use_cached_x = obj_cm.use_cached_x,
            threshold_state = ctx.obj.threshold_state,   # propagate -- see docstring
            outer_loop_opt = obj_cm.outer_loop_opt, inner_loop_opt = obj_cm.inner_loop_opt,
            needs_outer_moment_jacobian = obj_cm.needs_outer_moment_jacobian)
        ctx_cm = merge(ctx, (obj = obj_cm2,))
        bins = Int.(compute_bin_indices(ctx.U, aug.z))
        fctx = build_cm_frechet_bin_ctx(ctx, aug)
        return (ctx_cm = ctx_cm, aug = aug, cctx = fctx.cctx, bins = bins, cfg = cfg, L = L,
                targets = targets, mode = :frechet_reference, fctx = fctx)
    end

    @assert cfg.frechet_feature_set === :cdf_power
    aug = build_cm_frechet_augmented_obj_basis(ctx, CS, targets;
        basis = cfg.frechet_basis, feature_set = :cdf_power, contrasts = cfg.cm.contrasts)
    obj_cm = aug.obj_cm
    obj_cm2 = CS.PsiObjectiveBundleImplicit(δ = obj_cm.δ, find_smallest = obj_cm.find_smallest,
        γ = obj_cm.γ, (moments!) = obj_cm.moments!, moments_jacobian! = error,
        d = obj_cm.d, outer_constr_index = obj_cm.outer_constr_index,
        inequality_index = obj_cm.inequality_index, complement_index = obj_cm.complement_index,
        l = obj_cm.l, U = obj_cm.U, N = obj_cm.N, lower_limit = obj_cm.lower_limit,
        use_cached_x = obj_cm.use_cached_x,
        threshold_state = ctx.obj.threshold_state,   # propagate -- see docstring
        outer_loop_opt = obj_cm.outer_loop_opt, inner_loop_opt = obj_cm.inner_loop_opt,
        needs_outer_moment_jacobian = obj_cm.needs_outer_moment_jacobian)
    ctx_cm = merge(ctx, (obj = obj_cm2,))
    # `aug_cdf` here is a CDF-only layout descriptor (same ncore/origins/refIndex1/z/contrasts as
    # `aug`) required by `build_frechet_power_bin_ctx` -- NOT a second, inconsistent moment block.
    # `frechet_basis=:cumulative` is required for the fast structured path (interval Q1/Q2 kernels
    # for the combined block are not built in this port-prep pass, see the readiness doc).
    cfg.frechet_basis === :cumulative ||
        error("build_cm_frechet_production_context: frechet_feature_set=:cdf_power currently " *
              "requires frechet_basis=:cumulative for the fast structured Hessian path (got " *
              ":$(cfg.frechet_basis)) -- see FIXED_FRECHET_POST_OMIT_ROW_PORT_READINESS_2026-07-24.md")
    aug_cdf = build_cm_frechet_augmented_obj_archB(ctx, CS, targets; contrasts = cfg.cm.contrasts)
    fctx = build_frechet_power_bin_ctx(ctx, aug_cdf, targets)
    bins = Int.(compute_bin_indices(ctx.U, aug.z))
    return (ctx_cm = ctx_cm, aug = aug, cctx = fctx.cctx, bins = bins, cfg = cfg, L = L,
            targets = targets, mode = :frechet_reference, fctx = fctx)
end

"""
    cm_frechet_base_state(x_free0, fpcx) -> BaseDualState

Dispatches on `fpcx.mode`/`fpcx.cfg.frechet_feature_set` to the correct
Architecture-C base-state solver. For `:common_flexible`, delegates to the
existing `archC_base_state` unchanged.
"""
function cm_frechet_base_state(x_free0::AbstractVector, fpcx)
    fpcx.mode === :common_flexible && return archC_base_state(x_free0, fpcx.ctx_cm, fpcx.cctx)
    fpcx.cfg.frechet_feature_set === :cdf_only && return archC_frechet_base_state(x_free0, fpcx.ctx_cm, fpcx.fctx)
    return archC_frechet_cdf_power_base_state(x_free0, fpcx.ctx_cm, fpcx.fctx)
end

"""
    cm_frechet_verified_state(x_free0, fpcx) -> (base, verify)

Cold-verify dispatcher, same contract as `cm_frechet_base_state`.
"""
function cm_frechet_verified_state(x_free0::AbstractVector, fpcx)
    fpcx.mode === :common_flexible && return archC_verified_state(x_free0, fpcx.ctx_cm, fpcx.cctx)
    fpcx.cfg.frechet_feature_set === :cdf_only && return archC_frechet_verified_state(x_free0, fpcx.ctx_cm, fpcx.fctx)
    return archC_frechet_cdf_power_verified_state(x_free0, fpcx.ctx_cm, fpcx.fctx)
end

# ============================================================================
# Screened wrappers (task brief §9) -- run the exact core screens
# (`cm_screen_precheck!`, cm_screen_bridge.jl) BEFORE any inner solve. Core
# infeasibility implies fixed-Fréchet infeasibility (feasibility is a
# STRICTER restriction), so a core certificate is valid here unmodified; core
# feasibility alone does NOT certify fixed-Fréchet feasibility (only the full
# inner solve/its own threshold-10 gate can do that). No duplicate winner
# logic -- `cm_screen_precheck!` is the same entry point flexible CM uses,
# rectangular/:exclude_row-safe by construction (see
# FIXED_FRECHET_CANONICAL_WINNER_INTEGRATION_MANIFEST_2026-07-24.md).
# ============================================================================

"Screened drop-in replacement for `cm_frechet_base_state`."
function cm_frechet_base_state_screened(x_free0::AbstractVector, fpcx;
                                         counters::Union{Nothing,CMScreenCounters} = nothing,
                                         use_witness::Bool = false)
    cm_screen_precheck!(x_free0, fpcx.ctx_cm; counters = counters, use_witness = use_witness)
    return cm_frechet_base_state(x_free0, fpcx)
end

"Screened drop-in replacement for `cm_frechet_verified_state`."
function cm_frechet_verified_state_screened(x_free0::AbstractVector, fpcx;
                                             counters::Union{Nothing,CMScreenCounters} = nothing,
                                             use_witness::Bool = false)
    cm_screen_precheck!(x_free0, fpcx.ctx_cm; counters = counters, use_witness = use_witness)
    return cm_frechet_verified_state(x_free0, fpcx)
end
