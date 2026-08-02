# ============================================================================
# Restricted-inner-endtoend task (2026-08-01), addendum: stable typed accessor
# surface for the SEPARATE outer-gradient/A-gp workstream. DIAGNOSTIC/SHARED-
# READ-ONLY -- wraps already-existing, already-validated objects
# (ProfiledEconomicMomentLayout, AnchorSpec, the reduced family contexts this
# session built/reused), adds no new numerical logic of its own. Does not
# touch anything the outer-gradient workstream owns (A/gp outer-gradient
# wrappers, ProfiledLFixCache, incremental winner-update gradient code,
# gravity-pivot chain rule, outer A/B harnesses) -- this file is purely
# READ access to reduced-layout structure, not gradient computation.
#
# Naming matches the addendum's own requested names exactly:
#   profiled_economic_layout(ctx)
#   economic_dual_range(cctx_or_octx)
#   restriction_dual_ranges(cctx_or_octx)
#   profiled_anchor_spec(ctx)
#   profiled_outer_coordinate_layout(ctx)
# ============================================================================

isdefined(Main, :ProfiledEconomicMomentLayout) || error("profiled_restricted_accessors_2026-08-01.jl requires profiled_economic_moment_layout_2026-08-01.jl to be included first.")
isdefined(Main, :build_profiled_ab_spec_pe) || error("profiled_restricted_accessors_2026-08-01.jl requires profiled_outer_evaluator_2026-08-01.jl to be included first (for build_profiled_ab_spec_pe).")

"""
    profiled_anchor_spec(ctx; global_overrides=Dict{Int,Int}()) -> AnchorSpec

The anchor MAP (one omitted destination-slot cell per active destination) shared by every layer of
the profiled architecture -- the outer relative-A coordinate (`relative_a_coordinate_2026-07-31.jl`),
the reduced economic-moment layout below, and (via `build_profiled_ab_spec_pe`) the outer coordinate
layout. Thin wrapper around `build_anchor_spec_from_ctx` (`profiled_economic_moment_layout_2026-08-01.jl`)
-- rectangular-safe (correct for a non-contiguous omitted destination too, unlike
`default_anchor_spec`'s square/contiguous-only own-cell rule).
"""
profiled_anchor_spec(ctx; global_overrides::Dict{Int,Int} = Dict{Int,Int}()) =
    build_anchor_spec_from_ctx(ctx; global_overrides = global_overrides)

"""
    profiled_economic_layout(ctx; global_overrides=Dict{Int,Int}(), has_france_ratio=nothing) -> ProfiledEconomicMomentLayout

The shared, family-agnostic reduced economic-moment layout (task §6): one retained bilateral
homogeneous-factual-moment column per (origin,active-destination) cell except the anchor cell, plus
at most one France counterfactual-ratio column. IDENTICAL across all five families -- built once from
`profiled_anchor_spec(ctx)` and a probe `CompressedFactual` (only used to read `cf.cf_col > 0`, a
fixed structural fact of `ctx`, not point-dependent -- any valid `cf` suffices). `has_france_ratio`
can be supplied directly (e.g. by a caller that already has a `cf` in hand) to skip the probe build.
"""
function profiled_economic_layout(ctx; global_overrides::Dict{Int,Int} = Dict{Int,Int}(),
        has_france_ratio::Union{Nothing,Bool} = nothing)
    spec = profiled_anchor_spec(ctx; global_overrides = global_overrides)
    hfr = has_france_ratio
    if hfr === nothing
        x_free_calib = ctx.θ0_up[ctx.free_idx]
        θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
        cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
        hfr = cf_probe.cf_col > 0
    end
    return build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = hfr)
end

"""
    profiled_outer_coordinate_layout(ctx; global_overrides=Dict{Int,Int}()) -> (spec, gauge, pe)

The OUTER relative-A coordinate layout (task §14's "profiled start must be produced by reducing the
exact full calibration point" convention) -- `(AnchorSpec, gauge, PivotGravityElimOnRetained)`, built
against the GENUINE calibration z-matrix (`ctx.θ0_up`'s own A_od block), never the gravity-elimination
pivot's `zfree=0` reference point (this repo's own standing CLAUDE.md warning: A_od==1 is not
calibration). Thin wrapper around `build_profiled_ab_spec_pe`
(`profiled_outer_evaluator_2026-08-01.jl`) -- the SAME function `evaluate_profiled_point`/
`reduce_calibration_to_w_profiled` already use, not a new construction.

NOTE for the outer-gradient workstream: this uses the SAME `spec` (`AnchorSpec`) as
`profiled_economic_layout(ctx)` above internally would if called with the same `global_overrides` --
both layers share one anchor MAP, per this session's own scope note. This function returns the
OUTER-coordinate triple (origin-fast `i=o+(d-1)*D` indexing, `relative_a_coordinate_2026-07-31.jl`
convention), which is NOT directly comparable index-for-index to `profiled_economic_layout`'s own
`retained_full_factual_j` (destination-fast `j=slot+(o-1)*Ddest` indexing,
`compressed_moments.jl`/`cc_algo/active_layout.jl` convention) -- see that struct's own docstring for
the full "never compare these two `j`/`i` numberings directly" warning.
"""
profiled_outer_coordinate_layout(ctx; global_overrides::Dict{Int,Int} = Dict{Int,Int}()) =
    build_profiled_ab_spec_pe(ctx; global_overrides = global_overrides)

# ----------------------------------------------------------------------------
# economic_dual_range / restriction_dual_ranges: per-built-context dual-vector
# index ranges. Dual index 1 is always the zeta/intercept column (every
# family's own x[1]=zeta, x[2:end]=beta convention) -- these accessors
# describe how x[2:end] (equivalently, obj.d's own column layout: index k in
# G/H corresponds to dual index k+1) is carved up for a GIVEN, already-built
# reduced family context. Multiple dispatch per family's own concrete context
# type -- each method reads ONLY fields that context type already carries
# (cctx.NCORE/cctx.ncm/cctx.frechet_ext_cache for CM-family contexts,
# octx.NCORE/octx.n_eta for origin-ZC), no new bookkeeping invented.
# ----------------------------------------------------------------------------

"""
    economic_dual_range(cctx::CMBinHessCtx) -> UnitRange{Int}
    economic_dual_range(octx::OriginZCCoreHessCtx) -> UnitRange{Int}

Dual-vector index range (1-based). Dual index 1 is `zeta`, a KNITRO dual variable entirely SEPARATE
from `G`'s own columns (`inner_loop_internal_archgeneric`'s own `obj.H[:,2] .= 1.0` "ones" column is
part of `G`/`H`'s column space, NOT the same thing as dual index 1) -- so `G`'s column `k` (`k=1:d`)
corresponds to dual index `k+1`, for every `k`. `cctx.NCORE`/`octx.NCORE` IS `1 + n_economic` and
spans `G`'s OWN columns `1:NCORE` (column 1 there is the "ones"/zeta-paired economic column, per
`winner_pair_cross_hessian_esum!`'s own docstring: "j=1 (the ones/zeta column, E[:,1]≡1)") -- so the
economic block occupies dual indices `2:(1+NCORE)`, NOT `2:NCORE` (confirmed by
`verify_dual_ranges_partition` below: `2:NCORE` left the LAST dual index uncovered, a genuine
off-by-one caught by that structural self-check before this was ever handed off).

CM+ZC (`cctx.ncore_core < cctx.NCORE`, the widened-core case, `CMZC_WIDENED_CORE_FINDING_2026-08-01.md`)
is EXPLICITLY NOT SUPPORTED here -- errors loudly rather than silently returning a range that
conflates the true economic block with the widened mean/pair-ZC columns.
"""
function economic_dual_range(cctx::CMBinHessCtx)
    cctx.ncore_core == cctx.NCORE ||
        error("economic_dual_range: cctx.ncore_core=$(cctx.ncore_core) != cctx.NCORE=$(cctx.NCORE) -- " *
              "this is a CM+ZC widened-core context (see CMZC_WIDENED_CORE_FINDING_2026-08-01.md); " *
              "economic_dual_range's simple 2:(1+NCORE) range is not well-defined for it (would " *
              "conflate the true economic block with the widened mean/pair-ZC columns). Not supported.")
    return 2:(1 + cctx.NCORE)
end
economic_dual_range(octx::OriginZCCoreHessCtx) = 2:(1 + octx.NCORE)

"""
    restriction_dual_ranges(cctx::CMBinHessCtx) -> NamedTuple
    restriction_dual_ranges(octx::OriginZCCoreHessCtx) -> NamedTuple

Dual-vector index ranges (1-based) for each restriction block, in COLUMN ORDER, starting immediately
after `economic_dual_range`'s own last index. Keys/count/order depend on which family `cctx`/`octx`
was built for:
  - flexible CM (`cctx.frechet_ext_cache === nothing`): `(C = <ncm columns>,)`.
  - common Fréchet (`cctx.frechet_ext_cache isa CMFrechetExtension`): `(C = <ncm_cm columns>,
    F = <ncm_level columns>)` -- CM-grid block then level block, matching
    `wrap_moments_with_cm_frechet_archB`'s own `[core | CM | level | gravity]` column layout.
  - ZC-only (`octx::OriginZCCoreHessCtx`): `(Z = <n_eta columns>,)`.
CM+ZC is not supported (see `economic_dual_range`'s identical guard) -- `cctx.frechet_ext_cache`
alone cannot distinguish CM+ZC from flexible CM (both have it `=== nothing`), so this dispatches on
`cctx.ncore_core == cctx.NCORE` FIRST, matching `economic_dual_range`'s own check.
"""
function restriction_dual_ranges(cctx::CMBinHessCtx)
    cctx.ncore_core == cctx.NCORE ||
        error("restriction_dual_ranges: cctx.ncore_core=$(cctx.ncore_core) != cctx.NCORE=$(cctx.NCORE) -- " *
              "this is a CM+ZC widened-core context; not supported (see economic_dual_range's identical guard).")
    start = cctx.NCORE + 2   # immediately after economic_dual_range's own last index, 1+NCORE
    ext = cctx.frechet_ext_cache
    if ext isa CMFrechetExtension
        ncm_level = cctx.L
        ncm_cm = cctx.ncm - ncm_level
        return (C = start:(start + ncm_cm - 1), F = (start + ncm_cm):(start + cctx.ncm - 1))
    else
        return (C = start:(start + cctx.ncm - 1),)
    end
end
function restriction_dual_ranges(octx::OriginZCCoreHessCtx)
    start = octx.NCORE + 2   # immediately after economic_dual_range's own last index, 1+NCORE
    return (Z = start:(start + octx.n_eta - 1),)
end

"""
    verify_dual_ranges_partition(cctx_or_octx, total_dual_dim::Int) -> Bool

Structural self-check (not a numerical gate): confirms `economic_dual_range`/`restriction_dual_ranges`
together, plus dual index 1 (zeta), partition `1:total_dual_dim` exactly -- no gaps, no overlaps, no
out-of-bounds. `total_dual_dim` should be `1 + obj.outer_constr_index` for the reduced `obj` this
context was built alongside (dual vector = [zeta; economic; restrictions...], length
`1 + outer_constr_index`).
"""
function verify_dual_ranges_partition(ctx_hess, total_dual_dim::Int)
    econ = economic_dual_range(ctx_hess)
    restr = restriction_dual_ranges(ctx_hess)
    all_ranges = vcat([econ], collect(values(restr)))
    covered = falses(total_dual_dim)
    covered[1] = true   # zeta, index 1, not part of either accessor's own range
    for r in all_ranges
        for i in r
            (1 <= i <= total_dual_dim) || return false
            covered[i] && return false   # overlap
            covered[i] = true
        end
    end
    return all(covered)
end
