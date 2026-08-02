# ============================================================================
# Claude Code task 2026-08-01 (parallel outer-gradient workstream), §8: thin
# family adapters. ONE real adapter (unrestricted, wraps the already-gated
# ctx/spec/pe/layout unchanged) and FOUR MOCK adapters (flexible CM, common
# Frechet, ZC-only, CM+ZC) -- the inner branch's own restricted-family
# contexts are not ready yet (still in flight on
# architecture/profiled-restricted-inner-endtoend-2026-08-01), so per task
# §1/§7/§11 these are synthetic layouts sharing the SAME real economic dual
# slice as the unrestricted family, with an appended synthetic restriction
# dual range -- exactly task §11's "mock restricted layouts: the same
# economic dual slice, arbitrary restriction dual slices, arbitrary constant
# restriction contributions." When the inner branch's live typed accessors
# land, only the four `build_mock_*_family_ctx` constructors below need to be
# replaced by real ones over the inner branch's own context types; nothing in
# the shared engine, the gradient formula, or the tests that consume these
# adapters via the five-accessor contract needs to change.
#
# Each adapter may (task §8): validate family/layout checksum, extract the
# economic dual slice, extract restriction outer parameters, calculate a
# scalar restriction_contrib0, call the shared engine, append unchanged
# restriction-parameter gradients. It may NOT: recompute winners
# independently, materialize dense economic G, duplicate coordinate FD loops,
# or re-derive gravity-pivot/restriction derivatives -- none of the code
# below does any of that; `restriction_contrib0` for the mocks is a plain
# caller-supplied vector, and `restriction_param_gradient` below is an
# IDENTITY pass-through of whatever a family would supply (task §15: existing
# restriction-parameter gradients must be preserved unchanged, not rewritten
# here).
# ADDITIVE ONLY.
# ============================================================================

isdefined(Main, :validate_family_layout_contract) || error("profiled_family_adapters_2026-08-01.jl requires profiled_outer_gradient_layout_contract_2026-08-01.jl to be included first.")
isdefined(Main, :shared_family_outer_gradient) || error("profiled_family_adapters_2026-08-01.jl requires profiled_shared_economic_gradient_engine_2026-08-01.jl to be included first.")

# ----------------------------------------------------------------------------
# Unrestricted (real).
# ----------------------------------------------------------------------------

"""
    UnrestrictedFamilyCtx

The unrestricted family's implementation of the five-accessor contract.
Wraps the already-validated `(ctx, spec, pe, layout)` unchanged -- `layout`
is `ev.st.layout`, the SAME `ProfiledEconomicMomentLayout`
`build_profiled_operator_bundle` builds and the inner solve itself uses (not
rebuilt independently).
"""
struct UnrestrictedFamilyCtx
    ctx::Any
    spec::AnchorSpec
    pe::PivotGravityElimOnRetained
    layout::ProfiledEconomicMomentLayout
end

"build_unrestricted_family_ctx(ctx, spec, pe, ev) -> UnrestrictedFamilyCtx"
function build_unrestricted_family_ctx(ctx, spec::AnchorSpec, pe::PivotGravityElimOnRetained, ev)
    return UnrestrictedFamilyCtx(ctx, spec, pe, ev.st.layout)
end

profiled_economic_layout(fctx::UnrestrictedFamilyCtx) = fctx.layout
economic_dual_range(fctx::UnrestrictedFamilyCtx) = 1:fctx.layout.total_reduced_economic_moments
restriction_dual_ranges(::UnrestrictedFamilyCtx) = RestrictionDualRange[]
profiled_anchor_spec(fctx::UnrestrictedFamilyCtx) = fctx.spec
profiled_outer_coordinate_layout(fctx::UnrestrictedFamilyCtx) = fctx.pe
family_kind(::UnrestrictedFamilyCtx) = :unrestricted
layout_checksum(fctx::UnrestrictedFamilyCtx) =
    structural_checksum(fctx.layout, fctx.spec, fctx.pe, RestrictionDualRange[])
"restriction_contrib0 is identically zero for the unrestricted family -- there is no restriction block."
restriction_contrib0(fctx::UnrestrictedFamilyCtx, ev) = zeros(ev.st.cf.W)

# ----------------------------------------------------------------------------
# Mock restricted families.
# ----------------------------------------------------------------------------

"""
    MockRestrictedFamilyCtx

Synthetic restricted-family context (task §11) built by REUSING an
`UnrestrictedFamilyCtx`'s economic layout/spec/pe unchanged and appending a
synthetic restriction dual range placed immediately after
`economic_dual_range`. `rc0_fn(fctx, ev) -> Vector{Float64}` is caller-set so
tests can probe "arbitrary constant restriction contributions" without any
real restricted-family inner solve. `bad_checksum`, if true, deliberately
returns a wrong `layout_checksum` -- used ONLY by the negative
interface test to prove `validate_family_layout_contract` actually throws.
"""
struct MockRestrictedFamilyCtx
    base::UnrestrictedFamilyCtx
    family::Symbol
    n_restriction::Int
    rc0_fn::Function
    bad_checksum::Bool
end

function build_mock_restricted_family_ctx(base::UnrestrictedFamilyCtx, family::Symbol;
        n_restriction::Int = 5, rc0_fn::Function = (fctx, ev) -> zeros(ev.st.cf.W), bad_checksum::Bool = false)
    family in (:flexible_CM, :common_Frechet, :ZC_only, :CM_plus_ZC) ||
        error("build_mock_restricted_family_ctx: unknown family :$family")
    n_restriction > 0 || error("build_mock_restricted_family_ctx: n_restriction must be > 0 (empty restriction range is a contract violation)")
    return MockRestrictedFamilyCtx(base, family, n_restriction, rc0_fn, bad_checksum)
end

profiled_economic_layout(fctx::MockRestrictedFamilyCtx) = fctx.base.layout
economic_dual_range(fctx::MockRestrictedFamilyCtx) = economic_dual_range(fctx.base)
function restriction_dual_ranges(fctx::MockRestrictedFamilyCtx)
    n_econ = fctx.base.layout.total_reduced_economic_moments
    name = fctx.family == :flexible_CM ? :cm_marginals :
           fctx.family == :common_Frechet ? :frechet_shape :
           fctx.family == :ZC_only ? :zc_targets : :zc_targets_and_cm_marginals
    return [RestrictionDualRange(name, (n_econ + 1):(n_econ + fctx.n_restriction))]
end
profiled_anchor_spec(fctx::MockRestrictedFamilyCtx) = fctx.base.spec
profiled_outer_coordinate_layout(fctx::MockRestrictedFamilyCtx) = fctx.base.pe
family_kind(fctx::MockRestrictedFamilyCtx) = fctx.family
function layout_checksum(fctx::MockRestrictedFamilyCtx)
    ck = structural_checksum(profiled_economic_layout(fctx), profiled_anchor_spec(fctx),
        profiled_outer_coordinate_layout(fctx), restriction_dual_ranges(fctx))
    return fctx.bad_checksum ? ~ck : ck   # bitwise-complement: deterministic, deliberately WRONG
end
restriction_contrib0(fctx::MockRestrictedFamilyCtx, ev) = fctx.rc0_fn(fctx, ev)

"""
    mock_extended_beta(ev, n_restriction; scale=1.0) -> Vector{Float64}

Test helper: extends a REAL (unrestricted) solved dual vector `ev.result.beta`
with `n_restriction` synthetic restriction-dual entries, for feeding a
`MockRestrictedFamilyCtx`'s `economic_dual_range` (which points at the SAME
leading slice as the real beta, unchanged) plus a trailing synthetic slice the
shared engine's economic computation never reads directly (task §3: only
`economic_dual_range` and the caller-supplied `restriction_contrib0` are
consumed).
"""
function mock_extended_beta(ev, n_restriction::Int; scale::Float64 = 1.0)
    return vcat(ev.result.beta, scale .* randn(n_restriction))
end

"ev_with_beta(ev, beta_ext) -> NamedTuple -- shallow-copies ev with result.beta replaced."
function ev_with_beta(ev, beta_ext::Vector{Float64})
    return merge(ev, (result = merge(ev.result, (beta = beta_ext,)),))
end
