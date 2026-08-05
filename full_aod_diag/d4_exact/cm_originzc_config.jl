# ============================================================================
# Explicit production configuration surface for the origin-specific
# pairwise-zero-covariance restriction (NO common marginals). Sits ALONGSIDE
# CMMeanZCConfig (cm_meanzc_config.jl) rather than modifying it, and
# deliberately does NOT reuse the `cm_extension` keyword name -- this arm has
# no common-marginals component at all, so overloading a "CM-something" name
# for it would misdescribe the restriction (task brief Section 5). The
# top-level selector here is `distribution_restriction`, orthogonal to
# `cm_extension`. A single run is either CM-family (`cm_extension`) or
# origin-family (`distribution_restriction`), never both -- see
# `originzc_resolve_K`'s docstring.
# ============================================================================

"""
    OriginZCConfig(; distribution_restriction=:unrestricted, K_mean=0, K_pair=0,
                     power_target_layout=:origin_by_power, meanzc_basis=:direct,
                     nu_bounds=nothing)

- `distribution_restriction::Symbol`: `:unrestricted` (default, off) |
  `:origin_specific_moments` (mean-defining moments only, K_pair forced to 0
  -- DEFINITIONAL, task brief Section 3: reproduces `:unrestricted`'s Delta
  when profiled over nu, not a genuine new restriction) |
  `:origin_specific_moments_zero_covariance` (mean-defining PLUS pairwise
  zero-covariance moments, K_pair>=1 -- the genuine new economic restriction).
- `K_mean`/`K_pair::Int`: power levels (`0 <= K_pair <= K_mean`).
- `power_target_layout::Symbol`: `:origin_by_power` (this arm's own new
  layout, one nu_{o,k} per origin) or `:shared_by_power` (the existing
  CM+meanzc convention, reused here WITHOUT any CM block -- a legitimate but
  separate diagnostic configuration, see the math note's `:unrestricted`
  reproduction test).
- `meanzc_basis::Symbol`: `:direct` only for `:origin_by_power` (no anchored
  analog, math note Section 3 -- hard error otherwise); `:direct` or
  `:anchored` for `:shared_by_power` (delegates to the existing meanzc
  validation).
- `nu_bounds::Union{Nothing,Vector{NTuple{2,Float64}}}`: per-eta-coordinate
  `(lo,hi)` KNITRO box, length `n_eta(layout)`. `nothing` (default) derives a
  wide box automatically (see `originzc_default_nu_bounds`).
"""
Base.@kwdef struct OriginZCConfig
    distribution_restriction::Symbol = :unrestricted
    K_mean::Int = 0
    K_pair::Int = 0
    power_target_layout::Symbol = :origin_by_power
    meanzc_basis::Symbol = :direct
    nu_bounds::Union{Nothing,Vector{NTuple{2,Float64}}} = nothing
end

const ORIGINZC_NAMED_RESTRICTIONS = (:unrestricted, :origin_specific_moments, :origin_specific_moments_zero_covariance)

"""
    originzc_resolve_K(cfg::OriginZCConfig) -> (K_mean::Int, K_pair::Int)

Resolves `cfg.distribution_restriction` (plus `K_mean`/`K_pair`) to concrete
`(K_mean, K_pair)`. `:unrestricted` resolves to `(0, 0)` -- callers must
branch on `cfg.distribution_restriction === :unrestricted` BEFORE calling
anything in cm_originzc_moments.jl/cm_originzc_production.jl, mirroring
`meanzc_resolve_K`'s own `:cm_only` convention.

A checkpoint/run is expected to use EITHER `cm_extension` (CMMeanZCConfig,
common-marginals family) OR `distribution_restriction` (this struct,
no-common-marginals family), never both non-trivially in the same run --
enforced at the checkpoint layer (`cm_originzc_checkpoint.jl`), not here
(this function only resolves ITS OWN config).
"""
function originzc_resolve_K(cfg::OriginZCConfig)
    if cfg.distribution_restriction === :unrestricted
        (cfg.K_mean == 0 && cfg.K_pair == 0) ||
            error("OriginZCConfig: distribution_restriction=:unrestricted requires K_mean=K_pair=0, got ($(cfg.K_mean),$(cfg.K_pair))")
        return (0, 0)
    elseif cfg.distribution_restriction === :origin_specific_moments
        cfg.K_mean >= 1 || error("OriginZCConfig: distribution_restriction=:origin_specific_moments requires K_mean >= 1, got $(cfg.K_mean)")
        cfg.K_pair == 0 ||
            error("OriginZCConfig: distribution_restriction=:origin_specific_moments requires K_pair=0 (no pairwise-ZC moments in this arm), got $(cfg.K_pair) -- use :origin_specific_moments_zero_covariance instead")
        return (cfg.K_mean, 0)
    elseif cfg.distribution_restriction === :origin_specific_moments_zero_covariance
        cfg.K_mean >= 1 || error("OriginZCConfig: distribution_restriction=:origin_specific_moments_zero_covariance requires K_mean >= 1, got $(cfg.K_mean)")
        1 <= cfg.K_pair <= cfg.K_mean ||
            error("OriginZCConfig: distribution_restriction=:origin_specific_moments_zero_covariance requires 1 <= K_pair <= K_mean, got K_pair=$(cfg.K_pair), K_mean=$(cfg.K_mean)")
        return (cfg.K_mean, cfg.K_pair)
    else
        error("OriginZCConfig: distribution_restriction must be one of $(ORIGINZC_NAMED_RESTRICTIONS), got $(cfg.distribution_restriction)")
    end
end

function _originzc_validate(cfg::OriginZCConfig)
    cfg.power_target_layout in (:shared_by_power, :origin_by_power) ||
        error("OriginZCConfig: power_target_layout must be :shared_by_power or :origin_by_power, got $(cfg.power_target_layout)")
    K_mean, K_pair = originzc_resolve_K(cfg)   # raises on any inconsistency
    if cfg.power_target_layout === :origin_by_power
        cfg.meanzc_basis === :direct ||
            error("OriginZCConfig: power_target_layout=:origin_by_power supports meanzc_basis=:direct only (no anchored analog -- math note Section 3), got $(cfg.meanzc_basis)")
    else
        cfg.meanzc_basis in (:direct, :anchored) ||
            error("OriginZCConfig: meanzc_basis must be :direct or :anchored, got $(cfg.meanzc_basis)")
    end
    return nothing
end

"""
    originzc_validate_bounds(cfg::OriginZCConfig, layout::MeanZCTargetLayout)

`cfg.nu_bounds` length depends on `D` (via `n_eta(layout)`), which is not
known to `_originzc_validate` alone -- called separately once `layout` has
been constructed (`originzc_make_layout`).
"""
function originzc_validate_bounds(cfg::OriginZCConfig, layout::MeanZCTargetLayout)
    cfg.nu_bounds === nothing && return nothing
    length(cfg.nu_bounds) == n_eta(layout) ||
        error("OriginZCConfig: nu_bounds must have length n_eta(layout)=$(n_eta(layout)), got $(length(cfg.nu_bounds))")
    for (j, (lo, hi)) in enumerate(cfg.nu_bounds)
        lo < hi || error("OriginZCConfig: nu_bounds[$j] = ($lo, $hi) is not a valid (lo<hi) interval")
    end
    return nothing
end

"""
    originzc_make_layout(cfg::OriginZCConfig, D::Int) -> Union{Nothing,MeanZCTargetLayout}

`nothing` for `:unrestricted` (no layout at all -- callers must branch on
this exactly as `meanzc_resolve_K`'s `:cm_only` convention). Otherwise
`make_target_layout(cfg.power_target_layout, D, K_mean, K_pair)`
(cm_originzc_target_layout.jl).
"""
function originzc_make_layout(cfg::OriginZCConfig, D::Int)
    _originzc_validate(cfg)
    K_mean, K_pair = originzc_resolve_K(cfg)
    K_mean == 0 && return nothing
    return make_target_layout(cfg.power_target_layout, D, K_mean, K_pair)
end

"""
    originzc_default_nu_bounds(ctx, layout) -> Vector{NTuple{2,Float64}}

Deliberately WIDE per-eta-coordinate box (log-nu units), length
`n_eta(layout)`, a 4x safety margin on EACH side of the hard finite-support
interval derived from the actual draws (same convention as
`meanzc_default_nu_bounds`, which this dispatches to unchanged for
`SharedByPowerLayout`). For `OriginByPowerLayout`, each origin gets its OWN
interval (no cross-origin intersection is needed or correct here, unlike
the shared case where one nu_k must lie within every origin's range
simultaneously).
"""
function originzc_default_nu_bounds(ctx, layout::SharedByPowerLayout)
    return meanzc_default_nu_bounds(ctx, layout.K_mean)
end
function originzc_default_nu_bounds(ctx, layout::OriginByPowerLayout)
    D = layout.D
    bounds = Vector{NTuple{2,Float64}}(undef, n_eta(layout))
    for k in 1:layout.K_mean
        Uk = frechet_power_feature(ctx.U, k, ctx.μHat)
        for o in 1:D
            lo_o = minimum(@view Uk[:, o]); hi_o = maximum(@view Uk[:, o])
            lo_o > 0 || error("originzc_default_nu_bounds: non-positive lower bound at origin=$o level=$k (lo=$lo_o) -- log(nu) undefined")
            bounds[target_index(layout, o, k)] = (log(lo_o / 4), log(hi_o * 4))
        end
    end
    return bounds
end

"""
    originzc_verify_box_not_binding(η_star, bounds; tol=1e-4) -> Vector{Bool}

Identical contract to `meanzc_verify_box_not_binding` (cm_meanzc_config.jl),
reused verbatim in spirit for the (generally much longer) origin-by-power
eta vector.
"""
originzc_verify_box_not_binding(η_star::AbstractVector{Float64}, bounds::Vector{NTuple{2,Float64}}; tol::Float64 = 1e-4) =
    meanzc_verify_box_not_binding(η_star, bounds; tol = tol)
