# 2026-07-26/27 addendum session (governing prompt, "select the production outer-coordinate
# parameterization"): the TECHNOLOGY-coordinate axis, orthogonal to the existing
# `:logf`/`:logcutoff` PARTICIPATION-coordinate axis (`log_cutoff_param.jl`,
# `ctx.outer_parameterization`). Crossing this file's `technology_coordinate` with the
# existing `outer_parameterization` gives the full 3x2 factorial the addendum's Section D
# asks for. `melitz_unpower_theta_free`/`melitz_power_theta_free` (below) are called DIRECTLY
# from `log_cutoff_param.jl`'s `melitz_expand_theta`/`melitz_reduce_theta` dispatchers -- see
# those functions' own docstrings for why folding the technology rescale INTO the single
# expand/reduce dispatch point (rather than a separate post-hoc wrapper layer) makes every
# existing consumer -- every finite-difference gradient backend included -- technology-
# coordinate-correct automatically, with no separate chain-rule step needed anywhere else.
#
# Mathematical audit (addendum Section A, docs/melitz_outer_parameterization_comparison_2026-07-26.md):
# three natural scales for `log(A_od)` already coexist in the ACTIVE equations --
# `1x` (cutoff: `log(zhat_od) = ... - log(A_od) + ...`), `(sigma-1)x` (active-firm
# contribution `C_od ∝ A_od^(sigma-1)`), `theta_star x` (Pareto aggregate composite
# `chi = theta_star*a0 + beta*f0`).
#
# 2026-07-27 addendum Phase 5 CORRECTION (superseding the 2026-07-26 draft's own Section A.4):
# re-auditing the actual Ricardian OUTER-KNITRO-REGISTERED coordinate (not merely an internal
# ForwardDiff differentiation variable's name) in the CURRENT, most-heavily-used real-D20
# production driver (`full_aod_diag/d4_exact/c10_d20_production_driver.jl` +
# `full_aod_diag/d4_exact/gravity_elimination.jl`'s `build_pivot_elimination`/`pivot_expand`,
# read directly, Ricardian code NOT modified) finds the registered free variable `zfree` IS
# `log(Aod_theta)` -- UNSCALED (coefficient exactly 1 on `log(A_od)`, up to a KNOWN, data-only
# additive offset folded into `Aod_theta`'s own definition: `Aod = Aod_theta*cHat*(known
# wage/tariff/lambda ratios)`, i.e. `log(Aod_theta) = log(Aod) - [affine known-constant
# terms]`) -- matching `:logA`, NOT `:theta_logA`. `mu` (the trade elasticity) is FIXED
# (`ctx.fixed_vals[1]`/equal bounds) in every driver examined, never actually free-searched;
# `AodPow=(Aod/cHat)^(-mu)` is an internal reconstruction step inside the gravity-moment
# VALUE/GRADIENT formula (`gravity_tariff.jl`'s `gravity_grad_free!`), not the coordinate
# KNITRO itself receives (`cc_algo/outer_loop_cached.jl`'s own `KN_add_vars`/
# `KN_set_var_lobnds_all` registration confirms `x_free` IS `Aod_theta` directly, box-bounded
# `zfree_start .+/- 30.0` natural-log units, no separate KNITRO `var_scale`). A SEPARATE,
# less-current driver (`full_aod_diag/run_fullA_D4_production.jl`, self-described "ad_benchmark"
# reference path) instead searches over LINEAR (not log) `Aod_theta` with `mu` also fixed --
# a genuinely different, non-log convention, flagged as a real cross-driver inconsistency in
# this repo's own Ricardian code but NOT the coordinate that drives the bulk of this repo's
# real-D20 production work (confirmed by the volume of memory/doc references to
# `c10_d20_production_driver.jl` specifically). Since the dominant, current Ricardian
# coordinate is affinely equivalent to `:logA` (not genuinely nonlinear relative to any of the
# three candidates below), NO fourth candidate is added -- the governing prompt's own
# instruction ("add a candidate only if Phase 5 proves the actual coordinate is not affinely
# equivalent to one of these") is not triggered.

"""
    MELITZ_TECHNOLOGY_COORDINATES

The three technology-coordinate modes this file supports, all ONE-TO-ONE LINEAR rescalings
of `log(A_od)` -- they represent EXACTLY the same economic points, never a different
feasible set:

  - `:logA`                  -- `a_od = log(A_od)` (the cutoff equation's own natural scale;
                                also this codebase's PRE-EXISTING, only-ever-used scale for
                                both `:logf` and `:logcutoff` participation modes before this
                                file existed; also the dominant current Ricardian production
                                coordinate, up to sign/offset -- addendum Phase 5 above).
  - `:theta_logA`             -- `a_od = theta_star * log(A_od)` (the Pareto aggregate
                                composite's own natural scale).
  - `:sigma_minus_one_logA`   -- `a_od = (sigma-1) * log(A_od)` (the active-firm contribution
                                coefficient's own natural scale, `C_od ∝ A_od^(sigma-1)`).
"""
const MELITZ_TECHNOLOGY_COORDINATES = (:logA, :theta_logA, :sigma_minus_one_logA)

"""
    melitz_technology_coordinate_scale(technology_coordinate, ctx) -> Float64

The scalar `p_A` such that `a_scaled = p_A * log(A_od)` for every trade cell -- the SAME
constant for every cell (not cell-specific), since `theta_star`/`sigma` are global model
parameters, not bilateral.
"""
function melitz_technology_coordinate_scale(technology_coordinate::Symbol, ctx)
    technology_coordinate in MELITZ_TECHNOLOGY_COORDINATES || throw(ArgumentError(
        "melitz_technology_coordinate_scale: technology_coordinate must be one of " *
        "$MELITZ_TECHNOLOGY_COORDINATES, got $technology_coordinate"))
    technology_coordinate == :logA && return 1.0
    technology_coordinate == :theta_logA && return Float64(ctx.theta_star)
    return Float64(ctx.sigma - 1)   # :sigma_minus_one_logA
end

"""
    melitz_unpower_theta_free(theta_free, ctx) -> theta_free_plain

Un-scales `theta_free`'s A-block entries (`theta_free[2:1+nA]`, `nA=ctx.D^2-1` -- `theta_free[1]`
is always `g`, matching `melitz_nuisance_free_mask`'s own established layout convention,
UNCHANGED by any technology coordinate) from `get(ctx, :technology_coordinate, :logA)`'s own
units back to plain, unscaled `log(A_od)` -- the units `expand_free_theta`/
`expand_free_theta_logcutoff` require. `:logA` (absent `ctx.technology_coordinate`, or
`ctx.technology_coordinate == :logA`) is an EXACT no-op (returns a copy, not a view, matching
this codebase's own convention that `theta_free` is always freely mutable by the caller
afterward) -- every `ctx` built before this axis existed is byte-for-byte unaffected. Called
from `melitz_expand_theta` (`log_cutoff_param.jl`); see that function's own docstring.
"""
function melitz_unpower_theta_free(theta_free::AbstractVector, ctx)
    technology_coordinate = get(ctx, :technology_coordinate, :logA)
    p_A = melitz_technology_coordinate_scale(technology_coordinate, ctx)
    p_A == 1.0 && return copy(theta_free)
    nA = ctx.D^2 - 1
    theta_free_plain = copy(theta_free)
    theta_free_plain[2:1+nA] ./= p_A
    return theta_free_plain
end

"""
    melitz_power_theta_free(theta_free_plain, ctx) -> theta_free

Exact inverse of `melitz_unpower_theta_free`: re-scales a plain-`log(A_od)`-units `theta_free`
(as `reduce_to_free_theta`/`reduce_to_free_theta_logcutoff` produce) into
`get(ctx, :technology_coordinate, :logA)`'s own units. Called from `melitz_reduce_theta`
(`log_cutoff_param.jl`); see that function's own docstring.
"""
function melitz_power_theta_free(theta_free_plain::AbstractVector, ctx)
    technology_coordinate = get(ctx, :technology_coordinate, :logA)
    p_A = melitz_technology_coordinate_scale(technology_coordinate, ctx)
    p_A == 1.0 && return copy(theta_free_plain)
    nA = ctx.D^2 - 1
    theta_free = copy(theta_free_plain)
    theta_free[2:1+nA] .*= p_A
    return theta_free
end
