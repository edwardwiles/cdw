# ============================================================================
# Claude Code task 2026-08-01, §4/§5: the reduced ECONOMIC-MOMENT layout the
# prototype (architecture/profile-all-destination-scales-2026-07-31) never
# built. ADDITIVE ONLY -- does not modify any 2026-07-31 file; reuses
# AnchorSpec (relative_a_coordinate_2026-07-31.jl) for the anchor MAP itself
# (task instruction: "use the same AnchorSpec as the outer relative-A
# layout"), but builds it independently of `default_anchor_spec`'s own-cell
# rule, which silently assumes destination slot == global destination id
# (true only for a contiguous 1:Ddest active-destination set) -- exactly the
# square/contiguous-only limitation task §5 flags. This file's
# `build_anchor_spec_from_ctx` uses `global_destination(ctx, s)` for the
# own-cell default instead, so it is correct for a NON-contiguous omitted
# destination too (task §5's mandatory rectangular D=4 gate uses this).
#
# INDEX CONVENTION: this file operates ENTIRELY in the economic-moment
# (destination-fast) convention `j = slot + (o-1)*Ddest`, j in 1:D*Ddest,
# matching `CompressedFactual`/`compressed_moments.jl`/
# `cc_algo/active_layout.jl::active_cell_index` -- NOT the Aod-parameter-
# block (origin-fast) convention `i = o+(d-1)*D` that
# `relative_a_coordinate_2026-07-31.jl`/`gravity_pivot_on_retained_2026-07-31.jl`
# use for the OUTER A-coordinate. The two layers share the same anchor MAP
# (same (o,slot) cells declared anchors) but necessarily use different linear
# index numbers for the same cell, because they flatten a D x Ddest object
# with opposite strides. Never compare a `full_factual_j` value from this
# file to an `i` value from `relative_a_coordinate_2026-07-31.jl` directly.
# ============================================================================

isdefined(Main, :dest_slot) || error("profiled_economic_moment_layout_2026-08-01.jl requires cc_algo/active_layout.jl to be included first.")
isdefined(Main, :AnchorSpec) || error("profiled_economic_moment_layout_2026-08-01.jl requires relative_a_coordinate_2026-07-31.jl to be included first.")

"""
    build_anchor_spec_from_ctx(ctx; global_overrides=Dict{Int,Int}()) -> AnchorSpec

Rectangular-safe anchor-spec builder. `global_overrides` is keyed by GLOBAL
destination id (matching `DESTINATION_SCALE_ANCHOR_MANIFEST_2026-07-31.json`'s
own convention, e.g. `Dict(korea_id => brazil_id)`), not by slot. Own-cell
default for every other active destination is `global_destination(ctx,s)`
itself (the origin sharing that destination's global id) -- correct
regardless of whether the active-destination set is contiguous or which
destination (if any) is omitted, unlike `default_anchor_spec`'s
`anchor_origin[d]=d` rule (square/contiguous-only).
"""
function build_anchor_spec_from_ctx(ctx; global_overrides::Dict{Int,Int} = Dict{Int,Int}())
    D = ctx.D
    Ddest = length(active_destinations(ctx))
    anchor_origin = Vector{Int}(undef, Ddest)
    for s in 1:Ddest
        gd = global_destination(ctx, s)
        anchor_origin[s] = get(global_overrides, gd, gd)
    end
    return AnchorSpec(D, Ddest, anchor_origin)
end

"""
    ProfiledEconomicMomentLayout

The reduced economic-moment coordinate system (task §4): one retained
bilateral homogeneous-factual-moment column per (origin,active-destination)
cell EXCEPT the anchor cell (one omitted per destination, `Ddest` total
omissions), plus at most one France counterfactual-ratio moment column.
Structurally enforces (constructor asserts, not just documents):
  - exactly one omitted factual moment per active destination (from
    `AnchorSpec`'s own type-level guarantee, §2.4);
  - the omitted moment is the same cell used as the outer A anchor (both
    built from the identical `AnchorSpec`);
  - every other factual moment appears exactly once (`retained_full_factual_j`
    is `setdiff(1:D*Ddest, anchor set)`, so no duplicates, no gaps);
  - the France ratio moment appears at most once, never confused with a
    factual price-index moment (it is a SEPARATE field, `france_ratio_reduced_j`,
    never mixed into the bilateral `retained_*` arrays).
"""
struct ProfiledEconomicMomentLayout
    D::Int
    Ddest::Int

    destination_ids::Vector{Int}          # length Ddest: global destination id per slot
    anchor_origin_by_slot::Vector{Int}    # length Ddest: global origin id anchored per slot

    retained_full_factual_j::Vector{Int}  # length D*Ddest-Ddest, ascending full-space j
    retained_origin::Vector{Int}          # parallel: global origin o for retained_full_factual_j[k]
    retained_slot::Vector{Int}            # parallel: destination slot for retained_full_factual_j[k]

    full_factual_to_reduced::Vector{Int}  # length D*Ddest: reduced index, or 0 if this j is an anchor
    reduced_to_full_factual::Vector{Int}  # length D*Ddest-Ddest: inverse map

    france_ratio_reduced_j::Int           # 0 if this context has no France/cf_col ratio moment
    total_reduced_economic_moments::Int   # (D*Ddest-Ddest) + (france_ratio_reduced_j>0 ? 1 : 0)
end

"""
    build_profiled_economic_moment_layout(ctx, spec::AnchorSpec; has_france_ratio::Bool) -> ProfiledEconomicMomentLayout

`spec` must already be built against `ctx`'s live `(D,Ddest)`
(`build_anchor_spec_from_ctx(ctx; ...)`). `has_france_ratio` should be
`cf.cf_col > 0` for a genuine `CompressedFactual` built at any point in this
`ctx` (the property is a fixed layout fact of `ctx`, not point-dependent, so
any valid `cf` suffices to read it).
"""
function build_profiled_economic_moment_layout(ctx, spec::AnchorSpec; has_france_ratio::Bool)
    D = ctx.D
    Ddest = length(active_destinations(ctx))
    (spec.D, spec.Ddest) == (D, Ddest) ||
        throw(DimensionMismatch("build_profiled_economic_moment_layout: spec (D=$(spec.D),Ddest=$(spec.Ddest)) != ctx live (D=$D,Ddest=$Ddest)"))

    destination_ids = [global_destination(ctx, s) for s in 1:Ddest]
    anchor_origin_by_slot = copy(spec.anchor_origin)

    n_full = D * Ddest
    anchor_j = Set{Int}(active_cell_index(ctx, anchor_origin_by_slot[s], s) for s in 1:Ddest)
    length(anchor_j) == Ddest ||
        error("build_profiled_economic_moment_layout: anchor cells collide in economic-moment j-space (expected $Ddest distinct, got $(length(anchor_j))) -- this should be structurally impossible since each slot contributes exactly one j=slot+(o-1)*Ddest and slots are distinct")

    retained_full_factual_j = sort(collect(setdiff(1:n_full, anchor_j)))
    length(retained_full_factual_j) == n_full - Ddest ||
        error("build_profiled_economic_moment_layout: expected $(n_full-Ddest) retained factual moments, got $(length(retained_full_factual_j))")

    retained_origin = Vector{Int}(undef, length(retained_full_factual_j))
    retained_slot = Vector{Int}(undef, length(retained_full_factual_j))
    full_factual_to_reduced = zeros(Int, n_full)
    for (k, j) in enumerate(retained_full_factual_j)
        o, s = active_cell_from_index(ctx, j)
        retained_origin[k] = o
        retained_slot[k] = s
        full_factual_to_reduced[j] = k
    end
    reduced_to_full_factual = copy(retained_full_factual_j)

    n_retained = length(retained_full_factual_j)
    france_ratio_reduced_j = has_france_ratio ? n_retained + 1 : 0
    total = n_retained + (has_france_ratio ? 1 : 0)

    return ProfiledEconomicMomentLayout(D, Ddest, destination_ids, anchor_origin_by_slot,
        retained_full_factual_j, retained_origin, retained_slot,
        full_factual_to_reduced, reduced_to_full_factual,
        france_ratio_reduced_j, total)
end

"reduced_index(layout, o, slot) -> Int (0 if (o,slot) is the anchor cell for this slot)."
function reduced_index(layout::ProfiledEconomicMomentLayout, o::Int, slot::Int)
    j = slot + (o - 1) * layout.Ddest
    return layout.full_factual_to_reduced[j]
end

"is_anchor_cell(layout, o, slot) -> Bool"
is_anchor_cell(layout::ProfiledEconomicMomentLayout, o::Int, slot::Int) = layout.anchor_origin_by_slot[slot] == o

# ----------------------------------------------------------------------------
# Task §10 structural assertions: fail loudly if the reduced layout ever
# smuggles in a factual price-index moment, a full D-moment destination
# block, or any anchor factual moment.
# ----------------------------------------------------------------------------
"""
    assert_no_factual_price_index_moment(layout) -> nothing

Structural audit (task §10): confirms, from the layout's own bookkeeping
alone (no live evaluation needed), that:
  - every destination contributes exactly `D-1` retained factual moments
    (never all `D`, never a fixed `denom[d]`/`E[M_d]-1` normalization slot);
  - the France ratio moment (if present) is a single, separately-tracked
    field, never folded into `retained_full_factual_j`.
"""
function assert_no_factual_price_index_moment(layout::ProfiledEconomicMomentLayout)
    for s in 1:layout.Ddest
        n_s = count(==(s), layout.retained_slot)
        n_s == layout.D - 1 ||
            error("assert_no_factual_price_index_moment: destination slot $s has $n_s retained factual moments, expected D-1=$(layout.D-1) (a count of D would mean the anchor was NOT actually removed; a count that differs from D-1 for any other reason means an extra/missing moment column of some kind, e.g. a stray price-index slot)")
    end
    n_bilateral = length(layout.retained_full_factual_j)
    if layout.france_ratio_reduced_j > 0
        layout.france_ratio_reduced_j == n_bilateral + 1 ||
            error("assert_no_factual_price_index_moment: france_ratio_reduced_j=$(layout.france_ratio_reduced_j) must equal n_bilateral+1=$(n_bilateral+1) -- it must be a disjoint column appended after every bilateral reduced index, never collide with one")
    end
    expected_total = n_bilateral + (layout.france_ratio_reduced_j > 0 ? 1 : 0)
    layout.total_reduced_economic_moments == expected_total ||
        error("assert_no_factual_price_index_moment: total_reduced_economic_moments=$(layout.total_reduced_economic_moments) != bilateral($n_bilateral) + france($(layout.france_ratio_reduced_j>0)) = $expected_total")
    return nothing
end
