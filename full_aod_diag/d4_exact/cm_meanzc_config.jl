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
#
# CM+ZC-CROSS (2026-08-09): this file now also accepts
# `meanzc_target_layout=:shared_by_power_cross` (`SharedByPowerCrossLayout`,
# cm_meanzc_cross_target_layout.jl) -- the K_pair^2 ordered cross-power-grid
# pairwise-ZC restriction. Identical outer-parameter space to
# `:shared_by_power` (same `n_eta = K_mean`, same `target_index`/mean-block
# formulas), so it is threaded through wherever `:shared_by_power` is, with
# `SharedByPowerCrossLayout` substituted. Direct analog of origin-ZC's own
# `power_target_layout=:origin_by_power_cross`.
# ============================================================================

isdefined(Main, :SharedByPowerCrossLayout) || include(joinpath(@__DIR__, "cm_meanzc_cross_target_layout.jl"))

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
- `meanzc_target_layout::Symbol` (CM+ZC-CROSS, 2026-08-09):
  `:shared_by_power` (the pre-existing, diagonal-only pairwise-ZC block:
  `E[z_o^k z_p^k] = ν_k^2` for `k=1:K_pair`, `K_pair` restrictions per origin
  pair) or `:shared_by_power_cross` (CM+ZC-CROSS: the FULL ordered `K_pair^2`
  cross-power grid, `E[z_o^k1 z_p^k2] = ν_k1*ν_k2` for every
  `(k1,k2) ∈ {1..K_pair}^2`, `K_pair^2` restrictions per origin pair). SAME
  outer-parameter space either way (`n_eta = K_mean`, one shared `ν_k` per
  level -- the cross extension adds NO new outer parameters), so every
  `eta_nu`-shaped caller/checkpoint field keeps its length. Mirrors
  origin-ZC's own `power_target_layout` knob (cm_originzc_config.jl) exactly,
  including its `:origin_by_power` vs `:origin_by_power_cross` pairing.

  ON THE DEFAULT (deliberate, see CLAUDE.md's no-silent-scientific-defaults
  rule): `:shared_by_power` is not a placeholder -- it is the real, meaningful
  value for the pre-existing family, so an omitting caller gets EXACTLY the
  behavior they got before this field existed, never a silently-different
  economic problem. The genuine hazard a default could create -- running the
  CROSS restriction while a checkpoint/cache/campaign believes it is running
  the diagonal one -- is closed structurally, not by requiring the kwarg:
  (a) the checkpoint persists this field (`CMCheckpointV11`) and HARD-REFUSES
  a resume mismatch, exactly as `cm_extension`/`meanzc_K_mean` already do;
  (b) the driver's exact-cache `family_tag` is `:cm_meanzc_cross` rather than
  `:cm_meanzc`, so the two can never share a cache key (at identical
  `(K_mean,K_pair)` they give genuinely different Δ*, so a collision would be
  a silent wrong answer, not a perf nit); (c) the execution campaign and the
  reproducible-seed campaign both carry it as a distinct family identity.
  This is the same reasoning `meanzc_K_pair = 0` is already defaulted under.
"""
Base.@kwdef struct CMMeanZCConfig
    cm::CMConfig = CMConfig()
    cm_extension::Symbol = :cm_only
    meanzc_K_mean::Int = 0
    meanzc_K_pair::Int = 0
    meanzc_basis::Symbol = :direct
    meanzc_nu_bounds::Union{Nothing,Vector{NTuple{2,Float64}}} = nothing
    meanzc_target_layout::Symbol = :shared_by_power   # CM+ZC-CROSS (2026-08-09) -- see docstring above
end

const MEANZC_NAMED_ARMS = (:cm_only, :cm_plus_equal_means, :cm_plus_equal_means_zero_covariance)
const MEANZC_TARGET_LAYOUTS = (:shared_by_power, :shared_by_power_cross)   # CM+ZC-CROSS (2026-08-09)

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
    cfg.meanzc_target_layout in MEANZC_TARGET_LAYOUTS ||
        error("CMMeanZCConfig: meanzc_target_layout must be one of $(MEANZC_TARGET_LAYOUTS), got $(cfg.meanzc_target_layout)")
    K_mean, K_pair = meanzc_resolve_K(cfg)   # raises on any inconsistency
    # CM+ZC-CROSS (2026-08-09): the cross grid IS the pair block -- it is meaningless without one.
    # K_pair=0 means "mean moments only, no pairwise-ZC restriction at all", which the diagonal
    # layout expresses correctly and the cross layout cannot express at all (K_pair^2 = 0 blocks).
    # Refuse the combination rather than silently building an empty cross block.
    (cfg.meanzc_target_layout === :shared_by_power_cross && K_pair == 0) &&
        error("CMMeanZCConfig: meanzc_target_layout=:shared_by_power_cross requires K_pair >= 1 " *
              "(the cross grid IS the pair block; K_pair=0 means no pairwise-ZC restriction exists, " *
              "which is the :shared_by_power layout's own K_pair=0 case -- use that instead)")
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
    meanzc_make_layout(cfg::CMMeanZCConfig, D::Int) -> Union{Nothing,MeanZCTargetLayout}

CM+ZC-CROSS (2026-08-09). Config-facing layout constructor, the CM-family analog of
`originzc_make_layout` (cm_originzc_config.jl). `nothing` for `:cm_only` (no meanzc machinery is
consulted at all -- callers must branch on that BEFORE calling anything here, exactly as
`meanzc_resolve_K`'s own `:cm_only` convention requires). Otherwise
`SharedByPowerLayout(K_mean,K_pair)` or `SharedByPowerCrossLayout(K_mean,K_pair)` per
`cfg.meanzc_target_layout`. `D` is accepted but unused (both layouts are D-agnostic by
construction -- one shared target per level / level-pair regardless of D, see `layout_D`); it is
kept in the signature so callers need not branch before calling, matching `make_target_layout`'s
own convention.

Constructed here rather than inside `make_target_layout` (cm_originzc_target_layout.jl) for the
same reason `originzc_make_layout` handles `:origin_by_power_cross` itself: the cross layout types
live in files included AFTER that one (they include it, not the reverse), so putting a forward
reference there would be a load-order landmine.
"""
function meanzc_make_layout(cfg::CMMeanZCConfig, D::Int)
    _meanzc_validate(cfg)
    K_mean, K_pair = meanzc_resolve_K(cfg)
    K_mean == 0 && return nothing
    cfg.meanzc_target_layout === :shared_by_power_cross && return SharedByPowerCrossLayout(K_mean, K_pair)
    return SharedByPowerLayout(K_mean, K_pair)
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
        lo, hi = nu_feasible_interval(ctx.U, k; μ = ctx.μHat)
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
