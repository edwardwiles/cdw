# ============================================================================
# Claude Code task 2026-08-01 (parallel outer-gradient workstream), §7: the
# stable interface contract between this branch (profiled outer-coordinate
# decoding, shared economic A/gp gradient, family adapters, gravity-pivot
# chain rule, outer-gradient caches) and the separate, still-in-flight inner
# workstream (architecture/profiled-restricted-inner-endtoend-2026-08-01 or
# its descendant), which owns reduced economic layouts, restriction dual
# layouts, and inner FG/Hessian.
#
# This file defines FIVE typed accessors every restricted-family context must
# eventually provide, plus a validator that runs every structural check the
# shared engine (profiled_shared_economic_gradient_engine_2026-08-01.jl)
# relies on -- so that file can consume live restricted-family contexts with
# ZERO hard-coded offsets, family dimensions, or manual index arithmetic.
#
# Until the inner branch exposes live accessors, this file also provides:
#   - a REAL adapter for the unrestricted family (wraps the already-validated
#     ctx/spec/pe/layout from profiled_economic_moment_layout_2026-08-01.jl /
#     gravity_pivot_on_retained_2026-07-31.jl unchanged);
#   - MOCK adapters for the four restricted families (synthetic restriction
#     dual ranges appended after the SAME real economic layout/dual slice
#     the unrestricted family uses -- task §11's "mock restricted layouts").
# When the inner branch is ready, only the mock adapters in
# profiled_family_adapters_2026-08-01.jl need to be replaced; nothing in the
# shared engine or this contract file should need to change.
# ADDITIVE ONLY -- does not modify any existing file.
# ============================================================================

isdefined(Main, :ProfiledEconomicMomentLayout) || error("profiled_outer_gradient_layout_contract_2026-08-01.jl requires profiled_economic_moment_layout_2026-08-01.jl to be included first.")
isdefined(Main, :PivotGravityElimOnRetained) || error("profiled_outer_gradient_layout_contract_2026-08-01.jl requires gravity_pivot_on_retained_2026-07-31.jl to be included first.")
isdefined(Main, :AnchorSpec) || error("profiled_outer_gradient_layout_contract_2026-08-01.jl requires relative_a_coordinate_2026-07-31.jl to be included first.")

# ----------------------------------------------------------------------------
# The five required accessors (task §7). Each is a generic function with no
# fallback method -- calling it on an unsupported context type throws Julia's
# own MethodError, which is exactly the "no hard-coded family offsets" /
# "wrong dual length throws" behavior task §11 requires: there is no silent
# default to fall back to.
# ----------------------------------------------------------------------------

"""
    profiled_economic_layout(fctx) -> ProfiledEconomicMomentLayout

The shared economic-moment layout (task §3/§4: identical across all five
families by construction -- the restriction-family inner workstream's own
`build_reduced_base_obj_for_family` already reuses
`ProfiledEconomicMomentLayout` unchanged for this exact reason).
"""
function profiled_economic_layout end

"""
    economic_dual_range(fctx) -> UnitRange{Int}

Indices into the SOLVED DUAL VECTOR `β = x[2:end]` (KNITRO primal, not the
outer `w_profiled` vector) that belong to the economic block, in the SAME
order `profiled_economic_layout(fctx)`'s `retained_full_factual_j` /
`france_ratio_reduced_j` expect (`k`-th bilateral moment at `β[economic_dual_
range(fctx)][k]`, France ratio moment -- if present -- at
`β[economic_dual_range(fctx)][layout.france_ratio_reduced_j]`). Length MUST
equal `profiled_economic_layout(fctx).total_reduced_economic_moments`
(enforced by `validate_family_layout_contract`).
"""
function economic_dual_range end

"""
    RestrictionDualRange

One named, contiguous slice of `β` belonging to a family-specific restriction
block (CM marginals, ZC targets, Frechet shape parameters, ...). `name` is
free-form (`:cm_marginals`, `:zc_targets`, ...) -- used only for diagnostics
and the "restriction-parameter gradients unchanged" regression (task §15),
never for offset arithmetic.
"""
struct RestrictionDualRange
    name::Symbol
    range::UnitRange{Int}
end

"""
    restriction_dual_ranges(fctx) -> Vector{RestrictionDualRange}

Empty for the unrestricted family. For a restricted family, one entry per
restriction block, disjoint from `economic_dual_range(fctx)` and from each
other (enforced by `validate_family_layout_contract`).
"""
function restriction_dual_ranges end

"""
    profiled_anchor_spec(fctx) -> AnchorSpec

The same `AnchorSpec` used to build `profiled_economic_layout(fctx)` --
MUST satisfy `(spec.D, spec.Ddest) == (layout.D, layout.Ddest)`.
"""
function profiled_anchor_spec end

"""
    profiled_outer_coordinate_layout(fctx) -> PivotGravityElimOnRetained

The gravity-pivot-composed-with-anchor-reduction outer coordinate layout
(task §9) -- shared across families (gravity and the anchor reduction are
model-level, not family-level, restrictions). `pe.spec` MUST be
`profiled_anchor_spec(fctx)` (`===`, not just `==`, since the pivot's
`other_pos`/`pivot_pos`/`cr` are meaningless against a different spec
instance even if structurally equal).
"""
function profiled_outer_coordinate_layout end

"""
    family_kind(fctx) -> Symbol

One of `:unrestricted`, `:flexible_CM`, `:common_Frechet`, `:ZC_only`,
`:CM_plus_ZC` (task's own family names, mission preamble / task §8).
"""
function family_kind end

"""
    restriction_contrib0(fctx, ev) -> Vector{Float64}  (length cf.W)

Family-specific, PER-DRAW but A/gp-INDEPENDENT contribution the restriction
duals make to the fixed-dual `q[w] = -zeta - t[w]` functional (task §3's
economic-gradient theorem: restriction moments don't depend on relative-A or
gp at fixed restriction parameters, so this vector is evaluated ONCE per
`ev` and folded into the shared cache's `q0` exactly like the existing
`const_part`/`cf_raw_κcf` terms already are for the France ratio -- never
touched again during the A/gp coordinate loop). Zero vector for the
unrestricted family (`no restriction`). `ev` is whatever
`evaluate_profiled_point`-shaped NamedTuple the family's own evaluator
produces; this function may read `ev.result.beta`/`ev.st`/etc. but must NOT
call any winner-recomputation, must NOT materialize dense economic G, and
must NOT touch `economic_dual_range(fctx)`'s slice of beta (task §8: "It may
not perform winner recomputation independently / materialize economic G").
"""
function restriction_contrib0 end

# ----------------------------------------------------------------------------
# Structural fingerprint + validator (task §11's "incorrect layout checksum
# throws", "wrong dual length throws", "anchor coordinate cannot appear").
# ----------------------------------------------------------------------------

"""
    structural_checksum(layout, spec, pe, rranges) -> UInt64

A cheap, deterministic fingerprint of the pieces the shared engine trusts
NEVER to silently drift out of sync with each other within one `fctx`
(the economic layout's own D/Ddest/anchor map, the anchor spec, the pivot
outer-coordinate layout, and the family's restriction range names/extents).
Two `fctx` built from the same underlying model point must produce the same
checksum; any mismatch is a real structural bug (not a legitimate scientific
difference), so `validate_family_layout_contract` throws on mismatch rather
than warning.
"""
function structural_checksum(layout::ProfiledEconomicMomentLayout, spec::AnchorSpec,
        pe::PivotGravityElimOnRetained, rranges::Vector{RestrictionDualRange})
    h = hash((:profiled_outer_gradient_layout_contract, layout.D, layout.Ddest,
        layout.destination_ids, layout.anchor_origin_by_slot, layout.total_reduced_economic_moments,
        layout.france_ratio_reduced_j))
    h = hash((spec.D, spec.Ddest, spec.anchor_origin), h)
    h = hash((length(pe.cr), pe.pivot_pos, pe.other_pos), h)
    for r in rranges
        h = hash((r.name, r.range), h)
    end
    return h
end

"""
    layout_checksum(fctx) -> UInt64

Family-supplied claimed checksum (mock/real adapters compute this as
`structural_checksum(profiled_economic_layout(fctx), profiled_anchor_spec(fctx),
profiled_outer_coordinate_layout(fctx), restriction_dual_ranges(fctx))` when
honest -- `validate_family_layout_contract` recomputes the same quantity
independently and throws if they disagree, catching a stale/mismatched
adapter rather than silently using wrong offsets).
"""
function layout_checksum end

"""
    validate_family_layout_contract(fctx) -> NamedTuple

Runs every structural check `profiled_shared_economic_gradient_engine_2026-08-01.jl`
relies on. Throws (does not warn, does not silently coerce) on:
  - `economic_dual_range(fctx)` length mismatched against
    `profiled_economic_layout(fctx).total_reduced_economic_moments`;
  - any restriction range overlapping the economic range or another
    restriction range;
  - `profiled_anchor_spec(fctx)` dimension mismatch against the economic
    layout;
  - `profiled_outer_coordinate_layout(fctx).spec !== profiled_anchor_spec(fctx)`;
  - `layout_checksum(fctx)` disagreeing with the independently recomputed
    `structural_checksum`;
  - the economic layout itself failing `assert_no_factual_price_index_moment`
    (i.e. an anchor cell smuggled into the retained/economic block).
Returns a NamedTuple of the validated pieces so a caller that already paid
the validation cost can reuse them without re-fetching.
"""
function validate_family_layout_contract(fctx)
    layout = profiled_economic_layout(fctx)
    erange = economic_dual_range(fctx)
    rranges = restriction_dual_ranges(fctx)
    spec = profiled_anchor_spec(fctx)
    pe = profiled_outer_coordinate_layout(fctx)

    assert_no_factual_price_index_moment(layout)

    length(erange) == layout.total_reduced_economic_moments ||
        error("validate_family_layout_contract($(family_kind(fctx))): economic_dual_range length $(length(erange)) != layout.total_reduced_economic_moments $(layout.total_reduced_economic_moments)")

    all_ranges = Vector{UnitRange{Int}}()
    push!(all_ranges, erange)
    for r in rranges
        isempty(r.range) && error("validate_family_layout_contract($(family_kind(fctx))): restriction range :$(r.name) is empty")
        push!(all_ranges, r.range)
    end
    for i in eachindex(all_ranges), j in (i+1):length(all_ranges)
        isempty(intersect(all_ranges[i], all_ranges[j])) ||
            error("validate_family_layout_contract($(family_kind(fctx))): dual range $i overlaps dual range $j ($(all_ranges[i]) vs $(all_ranges[j]))")
    end

    (spec.D, spec.Ddest) == (layout.D, layout.Ddest) ||
        throw(DimensionMismatch("validate_family_layout_contract($(family_kind(fctx))): anchor spec (D=$(spec.D),Ddest=$(spec.Ddest)) != economic layout (D=$(layout.D),Ddest=$(layout.Ddest))"))

    pe.spec === spec ||
        error("validate_family_layout_contract($(family_kind(fctx))): profiled_outer_coordinate_layout(fctx).spec is not profiled_anchor_spec(fctx) (===) -- pivot other_pos/pivot_pos/cr would be meaningless against a mismatched spec instance")

    claimed = layout_checksum(fctx)
    actual = structural_checksum(layout, spec, pe, rranges)
    claimed == actual ||
        error("validate_family_layout_contract($(family_kind(fctx))): layout_checksum mismatch (claimed=$claimed, actual=$actual) -- a stale or mismatched adapter, not used")

    return (layout = layout, economic_dual_range = erange, restriction_dual_ranges = rranges,
        spec = spec, pe = pe)
end
