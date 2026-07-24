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

# exclude-ROW-destination production release (2026-07-24 addendum): `ctx` now DOES carry
# active_origins/active_destinations under destination_sample=:exclude_row (see
# context_real_d20.jl's row_idx/D_dest wiring), so the accessors above are no longer a permanent
# no-op -- this banner is the "every startup banner must print..." requirement (task item 6):
# resolved destination_sample, origin/destination/active-cell counts, free reduced A dimension,
# and the gravity-sample/theta-calibration code versions (context_real_d20.jl).
"""
    print_active_layout_banner(ctx, mode_label::AbstractString)

Prints the resolved destination-sample provenance for `ctx` under `mode_label` (e.g.
"cm_flexible", "cm_plus_meanzc", "origin_zc", "unrestricted"). One line, `[active-layout]`
prefixed, flushed immediately -- same discipline as `print_screen_startup_banner`/
`[threshold-config]`.
"""
function print_active_layout_banner(ctx, mode_label::AbstractString)
    n_origin = length(active_origins(ctx))
    n_dest = length(active_destinations(ctx))
    n_cells = n_origin * n_dest
    free_dim = hasproperty(ctx, :m) ? n_free(ctx.m) : n_cells
    println("[active-layout] mode=", mode_label,
            " destination_sample=", hasproperty(ctx, :destination_sample) ? ctx.destination_sample : :all_legacy,
            " origins=", n_origin, " destinations=", n_dest, " active_A_cells=", n_cells,
            " free_reduced_A_dim=", free_dim,
            " gravity_sample_version=", hasproperty(ctx, :gravity_sample_version) ? ctx.gravity_sample_version : 1,
            " theta_calibration_version=", hasproperty(ctx, :theta_calibration_version) ? ctx.theta_calibration_version : 1)
    flush(stdout)
end
