# ================================================================================================
# Explicit production configuration surface for the pairwise-quantile-independence restriction
# (draft eq. 32), version B (fixed cutoffs + free bin masses, 2026-08-10). Mirrors
# `cm_originzc_config.jl::OriginZCConfig`'s shape/discipline: a `Base.@kwdef struct` with NO default
# values on any field that changes what economic/statistical problem is being solved (repo's own
# no-silent-defaults rule) -- every field below is a REQUIRED keyword argument.
# ================================================================================================

"""
    PairwiseQuantileConfig(; mode, cutoff_source, min_bin_count, mass_start)

- `mode::Symbol`: `:all_cross` (the scientific default -- all `(L-1)^2` cross-quantile cells per
  pair, per the math note's Section 3 equivalence proof) | `:draft_cumulative_diagonal`
  (replication-mode NAME, kept per the original task's explicit instruction to retain it --
  **NOT numerically implemented**, see the warning below).
- `cutoff_source::Symbol`: `:frechet_theoretical` | `:empirical_quantile` -- WHERE the now-fixed
  quantile cutoffs sit. Version B's central modelling choice; see
  `pairwise_quantile_fixed_cutoffs` (pairwise_quantile_bin_context.jl) for what each means.
- `min_bin_count::Int`: the non-degeneracy floor asserted on every marginal bin AND every joint
  cell (`assert_pairwise_quantile_bins_nondegenerate`). Must be read against the campaign's own `W`
  and `L` -- a joint cell holds only ~W/L^2 draws in expectation.
- `mass_start::Symbol`: `:uniform` (mu = 1/L, version A's fixed target) | `:empirical` (the
  unweighted draws' own bin frequencies under the fixed cutoffs) -- which starting point the
  restriction's outer mass coordinates take. See `pairwise_quantile_start_masses`.

Version A carried `min_crossed` here instead of the last three: the cutoff-gradient secant
bandwidth. It is gone with that gradient -- version B's outer gradient is closed-form and has no
bandwidth (`pairwise_quantile_mass_gradient.jl`).

!!! warning "`:draft_cumulative_diagonal` is a placeholder, not a working replication mode"
    The draft's own diagonal condition (`P(z_o<q_r,z_p<q_r)=p_r^2`, `r=1..L-1`) is a CUMULATIVE
    (block-sum) functional of the interval cells -- `F(r,r) = sum_{a<=r,b<=r} pi_{ab}` (math note
    Section 3), not simply "the single interval cell `(a,b)=(r,r)`". Replicating it correctly would
    require a genuinely DIFFERENT moment functional (a block-sum indicator) alongside the interval-
    cell one this codebase builds throughout (`pairwise_quantile_operator.jl`/`_hessian.jl`).
    `mode=:draft_cumulative_diagonal` is therefore accepted by the resolver below (so the config
    surface has the field the task asks for) but `resolve_pairwise_quantile_mode` errors if it is
    actually selected, rather than silently running `:all_cross`'s math under the draft's name.
"""
Base.@kwdef struct PairwiseQuantileConfig
    mode::Symbol
    cutoff_source::Symbol
    min_bin_count::Int
    mass_start::Symbol
end

const PAIRWISE_QUANTILE_NAMED_MODES = (:all_cross, :draft_cumulative_diagonal)
const PAIRWISE_QUANTILE_CUTOFF_SOURCES = (:frechet_theoretical, :empirical_quantile)
const PAIRWISE_QUANTILE_MASS_STARTS = (:uniform, :empirical)

"""
    resolve_pairwise_quantile_mode(cfg::PairwiseQuantileConfig) -> Symbol

Validates every field of `cfg` and returns `cfg.mode` (only ever `:all_cross` -- see the struct's
own docstring for why `:draft_cumulative_diagonal` errors here instead of silently degrading to
`:all_cross`'s math under a different name).
"""
function resolve_pairwise_quantile_mode(cfg::PairwiseQuantileConfig)
    cfg.mode in PAIRWISE_QUANTILE_NAMED_MODES ||
        error("PairwiseQuantileConfig: mode must be one of $PAIRWISE_QUANTILE_NAMED_MODES, got $(cfg.mode)")
    cfg.cutoff_source in PAIRWISE_QUANTILE_CUTOFF_SOURCES ||
        error("PairwiseQuantileConfig: cutoff_source must be one of $PAIRWISE_QUANTILE_CUTOFF_SOURCES, got $(cfg.cutoff_source)")
    cfg.min_bin_count >= 1 ||
        error("PairwiseQuantileConfig: min_bin_count must be >= 1, got $(cfg.min_bin_count)")
    cfg.mass_start in PAIRWISE_QUANTILE_MASS_STARTS ||
        error("PairwiseQuantileConfig: mass_start must be one of $PAIRWISE_QUANTILE_MASS_STARTS, got $(cfg.mass_start)")
    cfg.mode === :draft_cumulative_diagonal &&
        error("PairwiseQuantileConfig: mode=:draft_cumulative_diagonal is not numerically implemented " *
              "(it requires a distinct cumulative/block-sum moment functional, not the interval-cell one " *
              "this codebase builds) -- see this file's own struct docstring. Use mode=:all_cross, the " *
              "scientific default.")
    return cfg.mode
end
