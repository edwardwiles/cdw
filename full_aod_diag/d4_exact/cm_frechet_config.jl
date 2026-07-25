# ============================================================================
# Fixed-Fréchet-marginals config surface (port-prep 2026-07-24, off
# production/fullA-exact @ c55e81e). Sits ALONGSIDE the existing CMConfig
# (cm_config.jl), same pattern CMMeanZCConfig uses -- `marginal_mode` is
# ORTHOGONAL to `cm_extension`. `marginal_mode = :common_flexible` (default)
# means: do not touch anything in this file or cm_frechet_*.jl -- every
# existing call site is reached exactly as before, byte-identical.
#
# Extends the pre-omit-ROW experimental design
# (experiment/fullA-fixed-frechet-basis-draft-reconciliation-2026-07-24) with
# two production-shaped additions:
#   - `frechet_feature_set`: :cdf_only (legacy/diagnostic, eq.37 alone) |
#     :cdf_power (DEFAULT -- eq.37+38, the full paper spec, task brief §1).
#   - `frechet_basis`: :cumulative (DEFAULT, matches the current flexible-CM
#     production default per docs/CURRENT_FLEXIBLE_CM_QUANTILE_IMPLEMENTATION_AUDIT_2026-07-24.md
#     Q1 -- interval is production-unreachable for flexible CM specifically
#     because of a documented D20/L=50 conditioning REGRESSION, 25-39x worse
#     than cumulative, per docs/fullA_common_marginals_production_integration.md)
#     | :interval (configurable, NOT yet promoted to default here -- task
#     brief §5 explicitly forbids a blanket "interval is best" claim in this
#     port; see FIXED_FRECHET_POST_OMIT_ROW_PORT_READINESS_2026-07-24.md for
#     the basis-default decision and the discrepancy with the OTHER,
#     Fréchet-specific pre-omit-ROW conditioning study that found interval
#     20-1020x BETTER for the FRÉCHET common-pin block specifically -- these
#     are two different Gram matrices (flexible-CM's vs. fixed-Fréchet's own
#     common/reference-pin block) and are not necessarily in tension, but are
#     not reconciled in this task).
# ============================================================================

"""
    CMFrechetConfig(; cm=CMConfig(), marginal_mode=:common_flexible,
                       frechet_feature_set=:cdf_power, frechet_basis=:cumulative)

- `cm::CMConfig`: the existing common-marginals configuration (grid,
  contrasts, Hessian backend) -- consulted regardless of `marginal_mode`.
- `marginal_mode::Symbol`: `:common_flexible` (default, off -- the existing
  CM restriction, unchanged) | `:frechet_reference` (new -- additionally
  pins the common level to the benchmark F* Fréchet marginal, see
  docs/FIXED_FRECHET_FULL_SPEC_MATH_AND_TARGETS_2026-07-24.md).
- `frechet_feature_set::Symbol`: `:cdf_only` (eq.37 alone, legacy/diagnostic)
  | `:cdf_power` (DEFAULT for `:frechet_reference` -- eq.37+38, full spec).
  Only consulted when `marginal_mode === :frechet_reference`.
- `frechet_basis::Symbol`: `:cumulative` (DEFAULT -- production-compatible,
  see header note) | `:interval`. Only consulted when
  `marginal_mode === :frechet_reference`. Independent of `cm.cm_basis`,
  which continues to govern `:common_flexible` mode unchanged.
"""
Base.@kwdef struct CMFrechetConfig
    cm::CMConfig = CMConfig()
    marginal_mode::Symbol = :common_flexible
    frechet_feature_set::Symbol = :cdf_power
    frechet_basis::Symbol = :cumulative
end

const CM_MARGINAL_MODES = (:common_flexible, :frechet_reference)
const CM_FRECHET_FEATURE_SETS = (:cdf_only, :cdf_power)
const CM_FRECHET_BASES = (:cumulative, :interval)

function _cm_frechet_validate(cfg::CMFrechetConfig)
    _cm_validate(cfg.cm)   # existing CMConfig validation, unchanged
    cfg.marginal_mode in CM_MARGINAL_MODES ||
        error("CMFrechetConfig: marginal_mode must be one of $(CM_MARGINAL_MODES), got $(cfg.marginal_mode)")
    cfg.frechet_feature_set in CM_FRECHET_FEATURE_SETS ||
        error("CMFrechetConfig: frechet_feature_set must be one of $(CM_FRECHET_FEATURE_SETS), got $(cfg.frechet_feature_set)")
    cfg.frechet_basis in CM_FRECHET_BASES ||
        error("CMFrechetConfig: frechet_basis must be one of $(CM_FRECHET_BASES), got $(cfg.frechet_basis)")
    # cm.cm_hessian_backend=:dense_reference remains legal (it is the D=4 correctness baseline
    # the structured Hessian is validated against, task brief §11/§7) but is NOT production-scale
    # viable at D=20/L=50 -- see FIXED_FRECHET_STRUCTURED_POWER_HESSIAN_2026-07-24.md.
    return nothing
end

"true iff this config activates the fixed-Fréchet restriction (i.e. any cm_frechet_*.jl machinery should be consulted at all)."
is_frechet_reference(cfg::CMFrechetConfig) = (_cm_frechet_validate(cfg); cfg.marginal_mode === :frechet_reference)
