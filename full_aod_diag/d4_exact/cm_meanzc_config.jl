# ============================================================================
# Explicit production configuration surface for the CM+moments(+ZC) extension,
# sitting ALONGSIDE the existing CMConfig (cm_config.jl) rather than modifying
# it -- matches this file family's own convention (cm_hessian_architectures.jl
# sits alongside cm_production_bundle.jl rather than editing it in place) and
# keeps the already-tested CMConfig/_cm_validate/cm_cache_key entirely
# untouched. `cm_extension = :cm_only` is the production default and means:
# don't touch this file's machinery at all -- the CM-only path
# (build_cm_production_context, cm_production_gradient, run_cm_upper,
# run_cm_upper_checkpointed) is reached exactly as it always was, unchanged.
# ============================================================================

"""
    CMMeanZCConfig(; cm=CMConfig(), cm_extension=:cm_only, meanzc_K_mean=0,
                      meanzc_K_pair=0, meanzc_basis=:direct, meanzc_nu_bounds=nothing)

- `cm::CMConfig`: the existing common-marginals configuration (grid, contrasts,
  Hessian backend) -- consulted regardless of `cm_extension`, since the
  extension is always CM-PLUS-something, never a replacement for CM.
- `cm_extension::Symbol`: `:cm_only` (default, off) | `:cm_plus_equal_means` |
  `:cm_plus_equal_means_zero_covariance` | `:cm_plus_moments` (the general
  K_mean/K_pair escape hatch -- the first three are named sugar for
  `(K_mean,K_pair) = (0,0)/(1,0)/(1,1)`, resolved via
  `meanzc_extension_to_K`/`meanzc_resolve_K` below).
- `meanzc_K_mean`/`meanzc_K_pair::Int`: ONLY consulted when
  `cm_extension = :cm_plus_moments` (error if set inconsistently with a named
  arm, e.g. `cm_extension=:cm_plus_equal_means` with `meanzc_K_mean=2`).
- `meanzc_basis::Symbol`: `:direct` (production default -- see
  docs/fullA_cm_meanzc_integration_note.md for the direct-vs-anchored
  conditioning comparison this default is based on) or `:anchored`.
- `meanzc_nu_bounds::Union{Nothing,Vector{NTuple{2,Float64}}}`: per-level
  `(lo,hi)` KNITRO box for every `ν_k`, `k=1:K_mean`. `nothing` (default)
  means "derive a wide box automatically" (see `meanzc_default_nu_bounds`) --
  callers should not hand-tune a narrow box without the widening check
  Section 6 of the task brief requires (`meanzc_verify_box_not_binding`).
"""
Base.@kwdef struct CMMeanZCConfig
    cm::CMConfig = CMConfig()
    cm_extension::Symbol = :cm_only
    meanzc_K_mean::Int = 0
    meanzc_K_pair::Int = 0
    meanzc_basis::Symbol = :direct
    meanzc_nu_bounds::Union{Nothing,Vector{NTuple{2,Float64}}} = nothing
end

const MEANZC_NAMED_ARMS = (:cm_only, :cm_plus_equal_means, :cm_plus_equal_means_zero_covariance)

"""
    meanzc_resolve_K(cfg::CMMeanZCConfig) -> (K_mean::Int, K_pair::Int)

Resolves `cfg.cm_extension` (plus `meanzc_K_mean`/`meanzc_K_pair` for the
`:cm_plus_moments` escape hatch) to concrete `(K_mean, K_pair)`. `:cm_only`
resolves to `(0, 0)` (meaning: no meanzc machinery in this file is consulted
at all -- callers must branch on `cfg.cm_extension === :cm_only` BEFORE
calling anything in cm_meanzc_moments.jl/cm_meanzc_production.jl, exactly
mirroring `common_marginals=false` in cm_config.jl).
"""
function meanzc_resolve_K(cfg::CMMeanZCConfig)
    if cfg.cm_extension === :cm_only
        (cfg.meanzc_K_mean == 0 && cfg.meanzc_K_pair == 0) ||
            error("CMMeanZCConfig: cm_extension=:cm_only requires meanzc_K_mean=meanzc_K_pair=0, got ($(cfg.meanzc_K_mean),$(cfg.meanzc_K_pair)) -- did you mean :cm_plus_moments?")
        return (0, 0)
    elseif cfg.cm_extension in (:cm_plus_equal_means, :cm_plus_equal_means_zero_covariance)
        K_mean, K_pair = meanzc_extension_to_K(cfg.cm_extension)
        (cfg.meanzc_K_mean == 0 || cfg.meanzc_K_mean == K_mean) &&
        (cfg.meanzc_K_pair == 0 || cfg.meanzc_K_pair == K_pair) ||
            error("CMMeanZCConfig: cm_extension=$(cfg.cm_extension) implies (K_mean,K_pair)=($K_mean,$K_pair); explicit meanzc_K_mean/meanzc_K_pair=($(cfg.meanzc_K_mean),$(cfg.meanzc_K_pair)) is inconsistent -- leave them at 0 (the default) or use :cm_plus_moments for a custom K.")
        return (K_mean, K_pair)
    elseif cfg.cm_extension === :cm_plus_moments
        cfg.meanzc_K_mean >= 1 || error("CMMeanZCConfig: cm_extension=:cm_plus_moments requires meanzc_K_mean >= 1, got $(cfg.meanzc_K_mean)")
        0 <= cfg.meanzc_K_pair <= cfg.meanzc_K_mean ||
            error("CMMeanZCConfig: meanzc_K_pair must satisfy 0 <= meanzc_K_pair <= meanzc_K_mean, got K_pair=$(cfg.meanzc_K_pair), K_mean=$(cfg.meanzc_K_mean)")
        return (cfg.meanzc_K_mean, cfg.meanzc_K_pair)
    else
        error("CMMeanZCConfig: cm_extension must be one of $(MEANZC_NAMED_ARMS) or :cm_plus_moments, got $(cfg.cm_extension)")
    end
end

function _meanzc_validate(cfg::CMMeanZCConfig)
    _cm_validate(cfg.cm)   # existing CMConfig validation, unchanged
    cfg.meanzc_basis in (:direct, :anchored) ||
        error("CMMeanZCConfig: meanzc_basis must be :direct or :anchored, got $(cfg.meanzc_basis)")
    K_mean, K_pair = meanzc_resolve_K(cfg)   # raises on any inconsistency
    if cfg.meanzc_nu_bounds !== nothing
        length(cfg.meanzc_nu_bounds) == K_mean ||
            error("CMMeanZCConfig: meanzc_nu_bounds must have length K_mean=$K_mean, got $(length(cfg.meanzc_nu_bounds))")
        for (k, (lo, hi)) in enumerate(cfg.meanzc_nu_bounds)
            lo < hi || error("CMMeanZCConfig: meanzc_nu_bounds[$k] = ($lo, $hi) is not a valid (lo<hi) interval")
        end
    end
    return nothing
end

"""
    meanzc_default_nu_bounds(ctx, K_mean) -> Vector{NTuple{2,Float64}}

Deliberately WIDE per-level box for η_ν (log ν): `(log(lo/4), log(hi*4))`
where `(lo,hi) = nu_feasible_interval(ctx.U, k)` -- a 4x safety margin on
EACH side of the hard finite-support interval derived from the actual draws,
not a narrow box centered on a single shakedown value (task brief Section 6:
"this is not proof a narrow box is safe"). Returned in η_ν-space directly
(natural log units) since that is what the KNITRO outer variable box uses.
"""
function meanzc_default_nu_bounds(ctx, K_mean::Int)
    return [begin
        lo, hi = nu_feasible_interval(ctx.U, k)
        (log(lo / 4), log(hi * 4))
    end for k in 1:K_mean]
end

"""
    meanzc_verify_box_not_binding(η_star::AbstractVector, bounds::Vector{NTuple{2,Float64}}; tol=1e-4) -> Vector{Bool}

Per-level check: is the solved `η_ν,k*` interior to its box (not within `tol`
of either bound)? Returns one `Bool` per level (`true` = interior/OK). A
production run should assert `all(meanzc_verify_box_not_binding(...))` and,
if any level fails, widen that level's bound and re-solve rather than
trusting a boundary solution (task brief Section 6).
"""
function meanzc_verify_box_not_binding(η_star::AbstractVector{Float64}, bounds::Vector{NTuple{2,Float64}}; tol::Float64 = 1e-4)
    return [!(η_star[k] <= bounds[k][1] + tol || η_star[k] >= bounds[k][2] - tol) for k in 1:length(η_star)]
end
