# ================================================================================================
# Explicit production configuration surface for the pairwise-quantile-independence restriction
# (draft eq. 32), Section 9 of the implementation plan. Mirrors `cm_originzc_config.jl::
# OriginZCConfig`'s shape/discipline: a `Base.@kwdef struct` with NO default values on any field
# that changes what economic/statistical problem is being solved (repo's own no-silent-defaults
# rule) -- every field below is a REQUIRED keyword argument.
# ================================================================================================

"""
    PairwiseQuantileConfig(; mode, min_crossed)

- `mode::Symbol`: `:all_cross` (the new scientific default -- all 16 `r,s=1..4` cross-quantile
  combinations per pair, per the math note's Section 3 equivalence proof) |
  `:draft_cumulative_diagonal` (replication-mode NAME, kept per the task's explicit instruction to
  retain it -- **NOT numerically implemented in this prototype**, see the warning below).
- `min_crossed::Int`: the cutoff-gradient secant's minimum-crossed-draw floor
  (`pairwise_quantile_cutoff_gradient.jl::cutoff_secant_gradient!`'s own `min_crossed` kwarg) --
  required here too (no silent default), caller must choose based on the campaign's `W`.

!!! warning "`:draft_cumulative_diagonal` is a placeholder, not a working replication mode"
    The draft's own diagonal condition (`P(z_o<q_r,z_p<q_r)=p_r^2`, `r=1..4`) is a CUMULATIVE
    (block-sum) functional of the interval cells -- `F(r,r) = sum_{a<=r,b<=r} pi_{ab}` (math note
    Section 3), not simply "the single interval cell `(a,b)=(r,r)`". Replicating it correctly would
    require a genuinely DIFFERENT moment functional (a block-sum indicator) alongside the interval-
    cell one this prototype builds throughout (`pairwise_quantile_operator.jl`/`_hessian.jl`), which
    was out of scope for this task's primary ask (build `:all_cross`). `mode=:draft_cumulative_
    diagonal` is therefore accepted by the resolver below (so the config surface has the field the
    task asks for) but `resolve_pairwise_quantile_mode` errors if it is actually selected, rather
    than silently running `:all_cross`'s math under the draft's name. Implementing it for real is
    flagged as follow-up work in the session status doc, not attempted here.
"""
Base.@kwdef struct PairwiseQuantileConfig
    mode::Symbol
    min_crossed::Int
end

const PAIRWISE_QUANTILE_NAMED_MODES = (:all_cross, :draft_cumulative_diagonal)

"""
    resolve_pairwise_quantile_mode(cfg::PairwiseQuantileConfig) -> Symbol

Validates `cfg.mode` and `cfg.min_crossed`, returns `cfg.mode` (only ever `:all_cross` -- see the
struct's own docstring for why `:draft_cumulative_diagonal` errors here instead of silently
degrading to `:all_cross`'s math under a different name).
"""
function resolve_pairwise_quantile_mode(cfg::PairwiseQuantileConfig)
    cfg.mode in PAIRWISE_QUANTILE_NAMED_MODES ||
        error("PairwiseQuantileConfig: mode must be one of $PAIRWISE_QUANTILE_NAMED_MODES, got $(cfg.mode)")
    cfg.min_crossed >= 1 ||
        error("PairwiseQuantileConfig: min_crossed must be >= 1, got $(cfg.min_crossed)")
    cfg.mode === :draft_cumulative_diagonal &&
        error("PairwiseQuantileConfig: mode=:draft_cumulative_diagonal is not numerically implemented " *
              "in this prototype (it requires a distinct cumulative/block-sum moment functional, not the " *
              "interval-cell one this codebase builds) -- see this file's own struct docstring. Use " *
              "mode=:all_cross, the scientific default.")
    return cfg.mode
end
