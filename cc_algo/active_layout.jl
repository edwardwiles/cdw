# Active-origin/active-destination layout accessors (2026-07-24 release addendum, Part B step 3).
#
# The current production context (`ctx`, built by context_real_d20.jl::d20_real_setup and the
# CM/origin-ZC context builders on top of it) is D-by-D square: every origin is also a
# destination. This file introduces the accessor layer the screen stack should query instead of
# assuming that squareness internally, so that a future rectangular/omit-ROW context (Part A,
# under separate development) can populate `ctx.active_origins`/`ctx.active_destinations` with a
# reduced destination set WITHOUT any change to the screen call sites here.
#
# This does NOT implement omit-ROW itself: `pairwise_certificate`/`screen_hard_winners`
# (infeasibility_screen.jl) still operate on the full D x D matrices internally, unchanged. Only
# the explicit iteration in cm_screen_bridge.jl's witness loop has been converted to use these
# accessors, per the task's minimal-scope instruction ("The key requirement is that screen logic
# uses the accessor/layout rather than assuming a square active sample internally").
#
# Under today's pre-omit-ROW production context, `ctx` carries no `active_origins`/
# `active_destinations` field, so every accessor here falls through to the full `1:ctx.D` range --
# zero behavior change from before this file existed.

"""
    active_origins(ctx) -> AbstractVector{Int}

Active origin indices for `ctx`. Falls through to `1:ctx.D` (every origin active) unless `ctx`
carries an explicit `active_origins` field (reserved for the future omit-ROW context).
"""
active_origins(ctx) = hasproperty(ctx, :active_origins) ? ctx.active_origins : Base.OneTo(ctx.D)

"""
    active_destinations(ctx) -> AbstractVector{Int}

Active destination indices for `ctx`. Falls through to `1:ctx.D` (every destination active)
unless `ctx` carries an explicit `active_destinations` field (reserved for the future omit-ROW
context, which will populate this with a strict subset of `1:ctx.D` when a destination such as
ROW is dropped).
"""
active_destinations(ctx) = hasproperty(ctx, :active_destinations) ? ctx.active_destinations : Base.OneTo(ctx.D)

"""
    active_od_cells(ctx) -> Iterator{Tuple{Int,Int}}

All (origin, destination) pairs over the active layout only -- `Iterators.product` of
`active_origins(ctx)` x `active_destinations(ctx)`, so an omitted destination never appears in a
loop built from this accessor even though today's default is the full square.
"""
active_od_cells(ctx) = Iterators.product(active_origins(ctx), active_destinations(ctx))
