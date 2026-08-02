# ============================================================================
# Phase 13 (integration/phase12-13-runner-checkpoints-2026-08-02): real per-point evaluator
# wrappers for origin-ZC and CM+ZC, mirroring `evaluate_profiled_flexcm_point`/
# `evaluate_profiled_frechet_point` (profiled_restricted_family_adapters_2026-08-02.jl) EXACTLY
# in output shape (`result.inner_status`/`result.zeta`/`result.beta`/`obj`/`st`/`m_weights`/
# `theta_full`/`decoded`), so the production outer runner (profiled_production_outer_runner_2026-08-01.jl)
# can drive origin-ZC/CM+ZC through the SAME code path as flexible-CM/common-Frechet.
#
# No equivalent function existed anywhere in this tree before this file -- every existing
# ZC-lane gate (test_zc_lane_originzc_outer_gradient_zerodense_d4_2026-08-02.jl etc.) builds `ev`
# inline, ad hoc, once per test. This factors that exact same construction (reduced_originzc_
# base_state / reduced_meanzc_base_state, zero-dense throughout) into a reusable, named function
# so the production runner has one real call site per family, not five inline copies.
#
# `nu_full`/`nuvec` (the ZC mean/pair target vector) is NOT part of `w_profiled` -- the recovered
# 2026-08-01 outer-runner scaffold's own `w_start` shape is `[gp; r_free]` only, with no eta/nu
# slot at all. This file's evaluators therefore hold `nu_full` FIXED across the outer loop
# (passed once at construction, read-only thereafter) -- extending the runner to a genuine joint
# (gp, A, eta) KNITRO search is future work, out of this adaptation's scope (the task asks to
# adapt the recovered scaffold to the real family-adapter surface, not to invent a new outer
# coordinate axis the scaffold never had).
#
# ADDITIVE ONLY -- does not modify profiled_reduced_originzc_lookup_kernels_2026-08-02.jl,
# profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl, or either family adapter file.
# ============================================================================

isdefined(Main, :reduced_originzc_base_state) ||
    error("profiled_zc_lane_point_evaluators_2026-08-02.jl requires profiled_reduced_originzc_lookup_kernels_2026-08-02.jl to be included first.")
isdefined(Main, :reduced_meanzc_base_state) ||
    error("profiled_zc_lane_point_evaluators_2026-08-02.jl requires profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl to be included first.")
isdefined(Main, :OriginZCFamilyCtx) ||
    error("profiled_zc_lane_point_evaluators_2026-08-02.jl requires profiled_originzc_family_adapter_2026-08-02.jl to be included first.")
isdefined(Main, :CMZCFamilyCtx) ||
    error("profiled_zc_lane_point_evaluators_2026-08-02.jl requires profiled_cmzc_family_adapter_2026-08-02.jl to be included first.")

"""
    OriginZCPointEvalState

The extra state `evaluate_profiled_originzc_point` needs beyond `OriginZCFamilyCtx` itself:
`octx` (`OriginZCCoreHessCtx`, built from the SAME `aug_reduced`/`profiled_layout` the family ctx
wraps -- see `build_originzc_family_ctx`'s own docstring) and the fixed `nu_full` target vector.
Kept as a SEPARATE, small state object (not folded into `OriginZCFamilyCtx` itself) because
`octx` carries live, per-solve mutable state (`core_cf_ref`) the outer-gradient contract's own
five accessors never need to see -- exactly the same reasoning `OriginZCFamilyCtx`'s own
docstring gives for wrapping (not replacing) `OriginZCCoreHessCtx`.
"""
struct OriginZCPointEvalState
    octx::Any
    nu_full::Vector{Float64}
end

"""
    evaluate_profiled_originzc_point(w_profiled, fctx::OriginZCFamilyCtx, pes::OriginZCPointEvalState;
        maxit_override=nothing) -> NamedTuple

Real per-point evaluator for origin-ZC, SAME output shape as `evaluate_profiled_flexcm_point`/
`evaluate_profiled_frechet_point`. Delegates entirely to the EXISTING `reduced_originzc_base_state`
(profiled_reduced_originzc_lookup_kernels_2026-08-02.jl) -- the genuinely dense-G-free reduced FG
driver already verified against ForwardDiff and the q-decomposition gate -- rather than
re-deriving any of that here.
"""
function evaluate_profiled_originzc_point(w_profiled::AbstractVector{Float64}, fctx::OriginZCFamilyCtx,
        pes::OriginZCPointEvalState; maxit_override::Union{Nothing,Int} = nothing)
    ctx = fctx.ctx
    decoded = decode_outer_profiled(collect(Float64, w_profiled), ctx, fctx.pe)
    r = reduced_originzc_base_state(decoded.xf, ctx, fctx.layout, pes.octx, pes.nu_full)
    θ_full = CS.reconstruct_full(decoded.xf, ctx.m)

    m_weights = similar(r.obj.arg0)
    r.obj.dPsi!(m_weights, r.obj.arg0)

    # BUGFIX (found live 2026-08-02, this branch): `r.st` is a `ReducedOriginZCOperatorState`,
    # which carries the solved CompressedFactual as `core_cf_ref::Ref{Any}`, NOT a `.cf` field
    # directly -- unlike its sibling `ReducedCMMeanZCOperatorState`
    # (profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl), which overrides `Base.getproperty`
    # so `st.cf` already resolves to `core_cf_ref[]`. Passing `r.st` straight through as `ev.st`
    # (mirroring `evaluate_profiled_flexcm_point`'s own pattern, which works because FlexCM's
    # state type has this getproperty override too) made `shared_family_outer_gradient`'s
    # `cf = st.cf` throw `FieldError(ReducedOriginZCOperatorState, :cf)` inside the KNITRO
    # gradient callback (surfaces as KNITRO status -500, "Could not evaluate first
    # derivatives"). Fixed by constructing the SAME manually-built `(cf=..., layout=...)`
    # NamedTuple `test_zc_lane_originzc_outer_gradient_zerodense_d4_2026-08-02.jl`'s own already-
    # gated Step 2 uses, sourcing `cf` from `r.st.core_cf_ref[]` (equivalent to that test's
    # `octx_reduced.core_cf_ref[]` -- the SAME box, since `ReducedOriginZCOperatorState`'s own
    # constructor is handed the caller's `core_cf_ref` and `pes.octx`/`r.st` share it) rather than
    # `r.st` itself. This does NOT touch `profiled_reduced_originzc_lookup_kernels_2026-08-02.jl`
    # (not this file's job to add a getproperty override there) -- purely a fix in THIS adapter.
    st_for_gradient = (cf = r.st.core_cf_ref[], layout = fctx.layout)

    result = (inner_status = r.inner_status, zeta = r.ζstar, beta = r.λstar, n_fg_calls = r.n_fg, n_hess_calls = r.n_hess)
    return (result = result, obj = r.obj, st = st_for_gradient, m_weights = m_weights, theta_full = θ_full,
            decoded = decoded, nu_full = pes.nu_full)
end

"""
    CMZCPointEvalState

Analogous to `OriginZCPointEvalState`, for CM+ZC: wraps `cctx` (`build_cm_meanzc_bin_ctx`'s own
return value, the CM-grid+ZC Hessian context) and the fixed `nu_full` (`K_mean`-length mean-target
vector; `K_pair=0` in this gate's own configuration, matching every other ZC-lane gate's own
"one safe config" choice).
"""
struct CMZCPointEvalState
    cctx::Any
    nu_full::Vector{Float64}
end

"""
    evaluate_profiled_cmzc_point(w_profiled, fctx::CMZCFamilyCtx, pes::CMZCPointEvalState;
        maxit_override=nothing) -> NamedTuple

Real per-point evaluator for CM+ZC, SAME output shape convention. Delegates to the EXISTING
`reduced_meanzc_base_state` (profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl).
"""
function evaluate_profiled_cmzc_point(w_profiled::AbstractVector{Float64}, fctx::CMZCFamilyCtx,
        pes::CMZCPointEvalState; maxit_override::Union{Nothing,Int} = nothing)
    ctx = fctx.ctx
    decoded = decode_outer_profiled(collect(Float64, w_profiled), ctx, fctx.pe)
    r = reduced_meanzc_base_state(decoded.xf, pes.nu_full, ctx, fctx.layout, pes.cctx)
    θ_full = CS.reconstruct_full(decoded.xf, ctx.m)

    m_weights = similar(r.obj.arg0)
    r.obj.dPsi!(m_weights, r.obj.arg0)

    result = (inner_status = r.inner_status, zeta = r.ζstar, beta = r.λstar, n_fg_calls = r.n_fg, n_hess_calls = r.n_hess)
    return (result = result, obj = r.obj, st = r.st, m_weights = m_weights, theta_full = θ_full,
            decoded = decoded, nu_full = pes.nu_full)
end
