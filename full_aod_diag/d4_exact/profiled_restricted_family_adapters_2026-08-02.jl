# ============================================================================
# Integration continuation (2026-08-02), task Section 7-8: real Stage-I typed adapters for
# flexible CM and common Fréchet, wired to the GENUINE dense-G-free reduced operator FG
# (profiled_reduced_lookup_kernels_2026-08-02.jl / profiled_reduced_frechet_lookup_kernels_2026-08-02.jl)
# -- NOT the older :dense_reference contexts. Satisfies the outer bridge's own 8-accessor contract
# (profiled_outer_gradient_layout_contract_2026-08-01.jl) so `shared_family_outer_gradient`
# (profiled_shared_economic_gradient_engine_2026-08-01.jl) works UNCHANGED for these two families.
#
# Per task Section 8: flexible CM and common Fréchet's live production outer variables are only
# `gp`/profiled relative A -- CM marginal targets / Fréchet level targets are fixed campaign
# configuration, never a live KNITRO decision variable, so NO restriction-parameter gradient is
# appended; `shared_family_outer_gradient`'s own output (length `outer_dim_profiled(pe)`) is the
# family's complete outer gradient, unchanged from the unrestricted family's own shape.
# ============================================================================

isdefined(Main, :validate_family_layout_contract) || error("profiled_restricted_family_adapters_2026-08-02.jl requires profiled_outer_gradient_layout_contract_2026-08-01.jl to be included first.")
isdefined(Main, :build_reduced_cm_operator_bundle) || error("profiled_restricted_family_adapters_2026-08-02.jl requires profiled_reduced_lookup_kernels_2026-08-02.jl to be included first.")
isdefined(Main, :build_reduced_frechet_operator_bundle) || error("profiled_restricted_family_adapters_2026-08-02.jl requires profiled_reduced_frechet_lookup_kernels_2026-08-02.jl to be included first.")
isdefined(Main, :restriction_contrib0_flexcm!) || error("profiled_restricted_family_adapters_2026-08-02.jl requires profiled_restriction_contrib0_operators_2026-08-01.jl to be included first.")
isdefined(Main, :decode_outer_profiled) || error("profiled_restricted_family_adapters_2026-08-02.jl requires outer_coordinate_layout_profiled_2026-07-31.jl to be included first.")
# profiled-inner-readiness-2026-08-03, task §7: independent verification, auto-included (not an
# error-guard) so every existing caller of this file's evaluators picks it up transitively without
# needing its own include list edited.
isdefined(Main, :verify_inner_solution_reduced_cm!) || include(joinpath(@__DIR__, "reduced_restricted_family_verification_2026-08-03.jl"))

# ----------------------------------------------------------------------------
# Flexible CM
# ----------------------------------------------------------------------------

"""
    FlexCMFamilyCtx

Real (not mock) five/eight-accessor adapter for flexible CM, wired to the genuine dense-G-free
`ReducedCMLookupState`/`OperatorPsiBundle` reduced operator FG. `cctx` is the ALREADY-BUILT reduced
`CMBinHessCtx` (`build_cm_bin_ctx(...; profiled_layout=layout, ...)`) -- shared, campaign-lifetime,
same object every outer point. `rc0_ws`/`rc0_buf` are persistent (never reallocated per gradient
call, per task's own "do not allocate a fresh W-vector per gradient" instruction).
"""
mutable struct FlexCMFamilyCtx
    ctx::Any
    spec::AnchorSpec
    pe::PivotGravityElimOnRetained
    layout::ProfiledEconomicMomentLayout
    cctx::CMBinHessCtx
    rc0_ws::FlexCMRestrictionWorkspace
    rc0_buf::Vector{Float64}
end

function build_flexcm_family_ctx(ctx, spec::AnchorSpec, pe::PivotGravityElimOnRetained,
        layout::ProfiledEconomicMomentLayout, cctx::CMBinHessCtx)
    W = size(cctx.Bidx, 1)
    ws = FlexCMRestrictionWorkspace(length(cctx.origins), cctx.L, W)
    FlexCMFamilyCtx(ctx, spec, pe, layout, cctx, ws, zeros(W))
end

profiled_economic_layout(fctx::FlexCMFamilyCtx) = fctx.layout
economic_dual_range(fctx::FlexCMFamilyCtx) = 1:fctx.layout.total_reduced_economic_moments
function restriction_dual_ranges(fctx::FlexCMFamilyCtx)
    n_econ = fctx.layout.total_reduced_economic_moments
    return [RestrictionDualRange(:cm_marginals, (n_econ+1):(n_econ + fctx.cctx.ncm))]
end
profiled_anchor_spec(fctx::FlexCMFamilyCtx) = fctx.spec
profiled_outer_coordinate_layout(fctx::FlexCMFamilyCtx) = fctx.pe
family_kind(::FlexCMFamilyCtx) = :flexible_CM
layout_checksum(fctx::FlexCMFamilyCtx) = structural_checksum(profiled_economic_layout(fctx),
    profiled_anchor_spec(fctx), profiled_outer_coordinate_layout(fctx), restriction_dual_ranges(fctx))

"""
    restriction_contrib0(fctx::FlexCMFamilyCtx, ev) -> Vector{Float64}

Real operator-only restriction contribution: extracts the CM-grid dual slice from `ev.result.beta`
via `restriction_dual_ranges(fctx)` (contract's own beta-relative convention), calls the EXISTING
`restriction_contrib0_flexcm!` (profiled_restriction_contrib0_operators_2026-08-01.jl) into the
PERSISTENT `fctx.rc0_buf`/`fctx.rc0_ws` -- no winner recomputation, no dense G, no per-call
allocation.
"""
function restriction_contrib0(fctx::FlexCMFamilyCtx, ev)
    rr = restriction_dual_ranges(fctx)[1]
    λ_cm = @view ev.result.beta[rr.range]
    cctx = fctx.cctx
    bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
    restriction_contrib0_flexcm!(fctx.rc0_buf, λ_cm, bins_u, cctx.refIndex1, cctx.origins, cctx.R, ev.st.cf.SW, fctx.rc0_ws)
    return fctx.rc0_buf
end

"""
    evaluate_profiled_flexcm_point(w_profiled, fctx; maxit_override=nothing) -> NamedTuple

Real per-point evaluator for flexible CM, mirroring the unrestricted family's own
`evaluate_profiled_point` exactly in shape (`result.zeta`/`result.beta`/`m_weights`/`theta_full`/
`obj`/`st`/`decoded`). Delegates entirely to the EXISTING `reduced_cm_base_state` driver
(profiled_reduced_lookup_kernels_2026-08-02.jl) -- which already builds the genuine dense-G-free
operator bundle, primes it, publishes `cctx`'s state, and solves via KNITRO -- rather than
re-deriving any of that here. `m_weights = dPsi!(obj.arg0)` at the solved point -- the SAME
quantity `verify_namedtuple_from_operator` computes for the unrestricted family, computed directly
since `obj.arg0` already holds the true (economic+CM-grid) q_total by construction -- no separate
restricted-family verification function needed for this purpose.
"""
function evaluate_profiled_flexcm_point(w_profiled::AbstractVector{Float64}, fctx::FlexCMFamilyCtx;
        maxit_override::Union{Nothing,Int} = nothing)
    ctx = fctx.ctx
    decoded = decode_outer_profiled(collect(Float64, w_profiled), ctx, fctx.pe)
    r = reduced_cm_base_state(decoded.xf, ctx, fctx.layout, fctx.cctx)
    θ_full = CS.reconstruct_full(decoded.xf, ctx.m)

    # Independent verification (profiled-inner-readiness-2026-08-03, task §7): recomputes r/f/
    # g_lambda from fresh scratch via verify_inner_solution_reduced_cm!, never reading the live
    # FG callback's own st.arg0/cm_contrib/etc. m_weights below comes from the INDEPENDENTLY
    # recomputed dual residual (ov.r), not r.obj.arg0 -- the whole point of independent
    # verification is that a bug corrupting the live callback's own cached state would not be
    # silently reproduced here.
    cctx = fctx.cctx
    bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
    ov = verify_inner_solution_reduced_cm!(r.ζstar, r.λstar, r.st.cf, ctx, θ_full, fctx.layout,
        cctx.L, length(cctx.origins), cctx.origins, cctx.refIndex1, bins_u, cctx.R, r.obj, r.st.cf.W)
    m_weights, verify = verify_namedtuple_from_operator(ov, r.obj, r.st.cf.W, r.inner_status)

    result = merge(verify, (zeta = r.ζstar, beta = r.λstar, n_fg_calls = r.n_fg, n_hess_calls = r.n_hess))
    return (result = result, obj = r.obj, st = r.st, m_weights = m_weights, theta_full = θ_full, decoded = decoded)
end

# ----------------------------------------------------------------------------
# Common Fréchet
# ----------------------------------------------------------------------------

"FrechetFamilyCtx: real adapter for common Fréchet, same pattern as FlexCMFamilyCtx plus the level-block restriction (fixed level_targets)."
mutable struct FrechetFamilyCtx
    ctx::Any
    spec::AnchorSpec
    pe::PivotGravityElimOnRetained
    layout::ProfiledEconomicMomentLayout
    cctx::CMBinHessCtx
    level_targets::Vector{Float64}
    rc0_ws::FrechetRestrictionWorkspace
    rc0_buf::Vector{Float64}
end

function build_frechet_family_ctx(ctx, spec::AnchorSpec, pe::PivotGravityElimOnRetained,
        layout::ProfiledEconomicMomentLayout, cctx::CMBinHessCtx, level_targets::Vector{Float64})
    W = size(cctx.Bidx, 1)
    ws = FrechetRestrictionWorkspace(length(cctx.origins), cctx.L, W)
    FrechetFamilyCtx(ctx, spec, pe, layout, cctx, level_targets, ws, zeros(W))
end

profiled_economic_layout(fctx::FrechetFamilyCtx) = fctx.layout
economic_dual_range(fctx::FrechetFamilyCtx) = 1:fctx.layout.total_reduced_economic_moments
function restriction_dual_ranges(fctx::FrechetFamilyCtx)
    n_econ = fctx.layout.total_reduced_economic_moments
    ncm_level = fctx.cctx.L
    ncm_cm = fctx.cctx.ncm - ncm_level
    return [RestrictionDualRange(:cm_marginals, (n_econ+1):(n_econ+ncm_cm)),
            RestrictionDualRange(:frechet_level, (n_econ+ncm_cm+1):(n_econ+ncm_cm+ncm_level))]
end
profiled_anchor_spec(fctx::FrechetFamilyCtx) = fctx.spec
profiled_outer_coordinate_layout(fctx::FrechetFamilyCtx) = fctx.pe
family_kind(::FrechetFamilyCtx) = :common_Frechet
layout_checksum(fctx::FrechetFamilyCtx) = structural_checksum(profiled_economic_layout(fctx),
    profiled_anchor_spec(fctx), profiled_outer_coordinate_layout(fctx), restriction_dual_ranges(fctx))

"""
    restriction_contrib0(fctx::FrechetFamilyCtx, ev) -> Vector{Float64}

Calls the EXISTING `restriction_contrib0_frechet!` (CM-grid + level combined, single call) into the
persistent `fctx.rc0_buf`.
"""
function restriction_contrib0(fctx::FrechetFamilyCtx, ev)
    rr = restriction_dual_ranges(fctx)
    λ_cm = @view ev.result.beta[rr[1].range]
    λ_level = @view ev.result.beta[rr[2].range]
    cctx = fctx.cctx
    bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
    restriction_contrib0_frechet!(fctx.rc0_buf, λ_cm, λ_level, bins_u, cctx.refIndex1, cctx.origins,
        cctx.R, cctx.D, fctx.level_targets, ev.st.cf.SW, fctx.rc0_ws)
    return fctx.rc0_buf
end

"""
    evaluate_profiled_frechet_point(w_profiled, fctx; maxit_override=nothing) -> NamedTuple

Real per-point evaluator for common Fréchet, mirroring `evaluate_profiled_flexcm_point` exactly,
delegating entirely to the EXISTING `reduced_frechet_base_state` driver
(profiled_reduced_frechet_lookup_kernels_2026-08-02.jl).
"""
function evaluate_profiled_frechet_point(w_profiled::AbstractVector{Float64}, fctx::FrechetFamilyCtx;
        maxit_override::Union{Nothing,Int} = nothing)
    ctx = fctx.ctx
    decoded = decode_outer_profiled(collect(Float64, w_profiled), ctx, fctx.pe)
    r = reduced_frechet_base_state(decoded.xf, ctx, fctx.layout, fctx.cctx, fctx.level_targets)
    θ_full = CS.reconstruct_full(decoded.xf, ctx.m)

    # Independent verification (profiled-inner-readiness-2026-08-03, task §7), same rationale as
    # flexible_CM's evaluator above.
    cctx = fctx.cctx
    bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
    ov = verify_inner_solution_reduced_cm_frechet!(r.ζstar, r.λstar, r.st.cf, ctx, θ_full, fctx.layout,
        cctx.L, length(cctx.origins), cctx.origins, cctx.refIndex1, bins_u, cctx.R, fctx.level_targets,
        r.obj, r.st.cf.W)
    m_weights, verify = verify_namedtuple_from_operator(ov, r.obj, r.st.cf.W, r.inner_status)

    result = merge(verify, (zeta = r.ζstar, beta = r.λstar, n_fg_calls = r.n_fg, n_hess_calls = r.n_hess))
    return (result = result, obj = r.obj, st = r.st, m_weights = m_weights, theta_full = θ_full, decoded = decoded)
end
