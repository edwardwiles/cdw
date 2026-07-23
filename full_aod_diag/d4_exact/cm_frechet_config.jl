# ============================================================================
# Fixed-Fréchet-marginals config surface (2026-07-23). Sits ALONGSIDE the
# existing CMConfig (cm_config.jl), same pattern CMMeanZCConfig
# (cm_meanzc_config.jl) already uses to add cm_extension without touching
# CMConfig -- `marginal_mode` is ORTHOGONAL to `cm_extension`, per task brief
# §4 ("do not overload cm_extension"). `marginal_mode = :common_flexible`
# (default) means: do not touch anything in this file or cm_frechet_*.jl --
# every existing call site (build_cm_production_context_v2 and everything
# downstream of it) is reached exactly as before, byte-identical.
# ============================================================================

"""
    CMFrechetConfig(; cm=CMConfig(), marginal_mode=:common_flexible)

- `cm::CMConfig`: the existing common-marginals configuration (grid,
  contrasts, Hessian backend) -- consulted regardless of `marginal_mode`.
- `marginal_mode::Symbol`: `:common_flexible` (default, off -- the existing
  CM restriction, unchanged) | `:frechet_reference` (new -- additionally
  pins the common level to the benchmark F* Fréchet marginal, see
  docs/FIXED_FRECHET_MARGINALS_MATH_NOTE_2026-07-23.md).
"""
Base.@kwdef struct CMFrechetConfig
    cm::CMConfig = CMConfig()
    marginal_mode::Symbol = :common_flexible
end

const CM_MARGINAL_MODES = (:common_flexible, :frechet_reference)

function _cm_frechet_validate(cfg::CMFrechetConfig)
    _cm_validate(cfg.cm)   # existing CMConfig validation, unchanged
    cfg.marginal_mode in CM_MARGINAL_MODES ||
        error("CMFrechetConfig: marginal_mode must be one of $(CM_MARGINAL_MODES), got $(cfg.marginal_mode)")
    cfg.marginal_mode === :frechet_reference && cfg.cm.cm_basis !== :cumulative &&
        error("CMFrechetConfig: marginal_mode=:frechet_reference is only implemented for cm.cm_basis=:cumulative (got :$(cfg.cm.cm_basis)) -- the interval basis has no Architecture-B/C fast path for either mode, see cm_config.jl's own comment.")
    return nothing
end

"true iff this config activates the fixed-Fréchet restriction (i.e. any cm_frechet_*.jl machinery should be consulted at all)."
is_frechet_reference(cfg::CMFrechetConfig) = (_cm_frechet_validate(cfg); cfg.marginal_mode === :frechet_reference)
