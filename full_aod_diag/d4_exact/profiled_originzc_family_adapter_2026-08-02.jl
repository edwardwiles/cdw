# ZC lane task (2026-08-02): the REAL origin-ZC implementation of the five-accessor contract
# (profiled_outer_gradient_layout_contract_2026-08-01.jl), replacing the MOCK restricted-family
# adapter (profiled_family_adapters_2026-08-01.jl::MockRestrictedFamilyCtx) for this family --
# exactly what that file's own header said would happen "when the inner branch's live typed
# accessors land": nothing in the shared engine, the gradient formula, or the five-accessor
# contract itself changes; only this new concrete adapter type is added.
#
# NAMING NOTE: `economic_dual_range`/`profiled_economic_layout`/etc. here are DIFFERENT methods
# (dispatched on `OriginZCFamilyCtx`, a new adapter type) from the SAME-NAMED functions in
# profiled_restricted_accessors_2026-08-01.jl (dispatched on `octx::OriginZCCoreHessCtx` directly,
# a completely different convention -- that file's own `economic_dual_range` indexes into the FULL
# x=[zeta;beta] vector, e.g. `2:NCORE`; this file's indexes into beta ALONE, e.g.
# `1:total_reduced_economic_moments`, per the outer-gradient contract's own docstring). No
# ambiguity: Julia dispatches on the argument TYPE, and `OriginZCFamilyCtx` is never `octx` itself
# -- this adapter WRAPS an already-built, already-solved `OriginZCCoreHessCtx`, it is not one.
# ADDITIVE ONLY.

isdefined(Main, :validate_family_layout_contract) || error("profiled_originzc_family_adapter_2026-08-02.jl requires profiled_outer_gradient_layout_contract_2026-08-01.jl to be included first.")
isdefined(Main, :restriction_contrib0_originzc!) || error("profiled_originzc_family_adapter_2026-08-02.jl requires profiled_restriction_contrib0_operators_2026-08-01.jl to be included first.")

"""
    OriginZCFamilyCtx

Real origin-ZC adapter. Wraps the already-validated `(ctx, spec, pe, layout)` (SAME objects the
unrestricted family's own `UnrestrictedFamilyCtx` wraps -- the economic layout is genuinely
family-agnostic, per this whole session's own repeated finding) plus the ZC restriction operator
state (`op`, `ws`) needed for `restriction_contrib0`.
"""
struct OriginZCFamilyCtx
    ctx::Any
    spec::AnchorSpec
    pe::PivotGravityElimOnRetained
    layout::ProfiledEconomicMomentLayout
    n_eta::Int
    op::ZCRestrictionOperator
    ws::ZCRestrictionWorkspace
    zc_layout::Any   # MeanZCTargetLayout (OriginByPowerLayout/SharedByPowerLayout) -- for refresh_zc_targets!
end

"""
    build_originzc_family_ctx(ctx, spec, pe, layout, aug_reduced) -> OriginZCFamilyCtx

`aug_reduced` is `build_originzc_augmented_obj(...)`'s own return value (already built for the real
reduced solve) -- `aug_reduced.Zraw_all`/`Zpairraw_all`/`layout` (the MeanZCTargetLayout, NOT to be
confused with the `ProfiledEconomicMomentLayout` argument of the same generic name) are reused
unchanged to build the SAME kind of `ZCRestrictionOperator`/`ZCRestrictionWorkspace`
`build_originzc_core_hess_ctx`/`_fill_cm_HEE!`'s own `octx.hzz_zc_op`/`octx.hzz_zc_ws` already are
-- a fresh, independent pair here (task §7: never touch a live FG/Hessian state's own buffers).
"""
function build_originzc_family_ctx(ctx, spec::AnchorSpec, pe::PivotGravityElimOnRetained,
        layout::ProfiledEconomicMomentLayout, aug_reduced)
    zc_layout = aug_reduced.layout
    op = ZCRestrictionOperator(aug_reduced.Zraw_all, aug_reduced.Zpairraw_all, size(aug_reduced.Zraw_all[1], 2))
    ws = ZCRestrictionWorkspace(op)
    return OriginZCFamilyCtx(ctx, spec, pe, layout, n_eta(zc_layout), op, ws, zc_layout)
end

profiled_economic_layout(fctx::OriginZCFamilyCtx) = fctx.layout
economic_dual_range(fctx::OriginZCFamilyCtx) = 1:fctx.layout.total_reduced_economic_moments
function restriction_dual_ranges(fctx::OriginZCFamilyCtx)
    n_econ = fctx.layout.total_reduced_economic_moments
    return [RestrictionDualRange(:zc_targets, (n_econ + 1):(n_econ + fctx.n_eta))]
end
profiled_anchor_spec(fctx::OriginZCFamilyCtx) = fctx.spec
profiled_outer_coordinate_layout(fctx::OriginZCFamilyCtx) = fctx.pe
family_kind(::OriginZCFamilyCtx) = :ZC_only
layout_checksum(fctx::OriginZCFamilyCtx) =
    structural_checksum(fctx.layout, fctx.spec, fctx.pe, restriction_dual_ranges(fctx))

"""
    restriction_contrib0(fctx::OriginZCFamilyCtx, ev) -> Vector{Float64}

`ev.result.beta`'s restriction slice (`economic_dual_range(fctx)`'s complement, per the contract)
is `[λ_mean; λ_pair]`; `ev.nu_full` (this adapter's own extra field on top of the plain contract,
harmless -- `restriction_contrib0`'s signature is `(fctx, ev)`, free to read whatever `ev` fields
this family's own evaluator populates) supplies the CURRENT outer point's targets for
`refresh_zc_targets!`, exactly the same "caller refreshes ws for the current point" discipline
`restriction_contrib0_originzc!`'s own docstring requires.
"""
function restriction_contrib0(fctx::OriginZCFamilyCtx, ev)
    n_econ = fctx.layout.total_reduced_economic_moments
    K_mean = fctx.zc_layout.K_mean
    D = fctx.op.D
    β = ev.result.beta
    λ_mean = @view β[n_econ+1 : n_econ+K_mean*D]
    λ_pair = @view β[n_econ+K_mean*D+1 : n_econ+fctx.n_eta]
    refresh_zc_targets!(fctx.ws, fctx.op, fctx.zc_layout, ev.nu_full)
    dest = Vector{Float64}(undef, ev.st.cf.W)
    restriction_contrib0_originzc!(dest, λ_mean, λ_pair, fctx.op, fctx.ws, ev.st.cf.SW)
    return dest
end
