# ============================================================================
# Explicit restriction/configuration selector for the nested CM / CM+mean /
# CM+mean+ZC experiment (task brief Section 4). Mirrors cm_config.jl's
# CMConfig convention (Base.@kwdef struct, Symbol-typed dispatchers, a
# _validate function raising on any out-of-enum value) -- additive only, does
# not modify CMConfig.
# ============================================================================

"""
    MeanZCConfig(; cm_extension=:cm_only, nu_parameterization=:log, meanzc_basis=:direct)

- `cm_extension::Symbol`: `:cm_only` (default -- byte-identical to the
  pre-existing CM path, this file's machinery is never consulted),
  `:cm_plus_mean` (CM + exact common first moments, 1 outer scalar + D inner
  mean moments), or `:cm_plus_mean_zero_covariance` (CM + exact means +
  pairwise zero covariance, same 1 outer scalar + D mean + D(D-1)/2 pair inner
  moments). Cannot represent an invalid pair-only state by construction --
  there is no enum value for "pair restrictions without mean restrictions."
- `nu_parameterization::Symbol`: `:log` (only implemented value;
  `η_ν=log(ν)`, positivity-preserving, see math note Section 2).
- `meanzc_basis::Symbol`: `:direct` (production candidate, math note Section
  3) or `:anchored` (equivalent feasible set, conditioning/speed comparison
  candidate). Ignored when `cm_extension === :cm_only`.
"""
Base.@kwdef struct MeanZCConfig
    cm_extension::Symbol = :cm_only
    nu_parameterization::Symbol = :log
    meanzc_basis::Symbol = :direct
end

function _meanzc_validate(cfg::MeanZCConfig)
    cfg.cm_extension in (:cm_only, :cm_plus_mean, :cm_plus_mean_zero_covariance) ||
        error("MeanZCConfig: cm_extension must be :cm_only, :cm_plus_mean, or :cm_plus_mean_zero_covariance, got $(cfg.cm_extension)")
    cfg.nu_parameterization === :log ||
        error("MeanZCConfig: nu_parameterization must be :log (only implemented value), got $(cfg.nu_parameterization)")
    cfg.meanzc_basis in (:direct, :anchored) ||
        error("MeanZCConfig: meanzc_basis must be :direct or :anchored, got $(cfg.meanzc_basis)")
    return nothing
end

"`true` for the two arms this file's machinery actually builds anything for."
meanzc_active(cfg::MeanZCConfig) = cfg.cm_extension !== :cm_only

"`true` only for the arm that constructs/allocates/fingerprints pair-product columns."
meanzc_wants_pair(cfg::MeanZCConfig) = cfg.cm_extension === :cm_plus_mean_zero_covariance

"""
    meanzc_cache_key(cfg::MeanZCConfig, ν::Union{Nothing,Float64}) -> NamedTuple

Cache-partitioning key fragment (mirrors `cm_cache_key`, cm_config.jl):
`cm_extension=:cm_only` collapses to a single canonical key so unrestricted/
CM-only cached results are never partitioned by ν or basis. For the two active
arms, ν is a genuine cache-relevant coordinate (it changes the inner problem)
so it MUST be part of the key -- `ν=nothing` is only valid for a
not-yet-profiled context and should never be used to actually store/look up a
cache entry (enforced by the assert below, not a silent nothing-key).
"""
function meanzc_cache_key(cfg::MeanZCConfig, ν::Union{Nothing,Float64})
    _meanzc_validate(cfg)
    cfg.cm_extension === :cm_only && return (cm_extension = :cm_only,)
    ν !== nothing || error("meanzc_cache_key: ν=nothing is not a valid cache key for an active cm_extension arm ($(cfg.cm_extension))")
    return (cm_extension = cfg.cm_extension, nu_parameterization = cfg.nu_parameterization,
            meanzc_basis = cfg.meanzc_basis, nu = ν)
end
