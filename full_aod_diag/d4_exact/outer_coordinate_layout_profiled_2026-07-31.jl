# ============================================================================
# Task §6/§10 (Topic 1/10 generalization): profiled-destination-scales outer
# vector decode/encode, KNITRO-facing shape.
# ADDITIVE ONLY -- does NOT modify outer_coordinate_layout.jl or
# gravity_elimination.jl, both reused unchanged. This is the "generalize
# OuterCoordinateLayout" step from the master report's next-steps list,
# implemented as a parallel additive decoder (mirroring how
# gravity_pivot_on_retained_2026-07-31.jl layers on top of
# gravity_elimination.jl) rather than editing the trusted file in place, per
# this repo's own "new function, verify vs trusted path, then wire in"
# convention.
#
# Fixed-theta scope only (task §6's explicit instruction: "Implement and
# gate the fixed-theta production path first"). Produces the SAME `xf` shape
# `decode_outer_unified` does (`[gp; Aod_levels]`), from a GENUINELY SHORTER
# outer vector `w_profiled = [gp; r_free]` (length `1 + n_retained(spec)-1`
# instead of `1 + D*Ddest-1`), so downstream screened-evaluation/FG code that
# only reads `xf` needs no changes at all -- this is the concrete sense in
# which "only the economic coordinate/moment basis changes" (task §5).
# ============================================================================

isdefined(Main, :decode_full_z_on_retained) || error("outer_coordinate_layout_profiled_2026-07-31.jl requires gravity_pivot_on_retained_2026-07-31.jl to be included first.")

"""
    outer_dim_profiled(pe::PivotGravityElimOnRetained) -> Int

Length of the profiled outer vector `w_profiled = [gp; r_free]`:
`1 + (n_retained(spec)-1)`. Compare `outer_dim(layout,D,Ddest) = D*Ddest`
(fixed-theta) for the full formulation -- at real D=20 this is `1+360=361`
vs. the full formulation's `380`.
"""
outer_dim_profiled(pe::PivotGravityElimOnRetained) = 1 + length(pe.other_pos)

"""
    decode_outer_profiled(w_profiled, ctx, pe; μ=nothing) -> NamedTuple

`w_profiled = [gp; r_free]` -> `(xf, gp, Aod_levels, z_full)`, where `xf =
vcat(gp, Aod_levels)` is EXACTLY the shape `decode_outer_unified` produces
for a fixed-theta context (`screened_eval`/`evaluate_fullA_screened_ranged`'s
own expected input) -- so a production screened-evaluation entry point can
consume this `xf` with NO changes, only the (much shorter) `w_profiled` it
was built from differs from today's `w`.
"""
function decode_outer_profiled(w_profiled::AbstractVector{Float64}, ctx, pe::PivotGravityElimOnRetained; μ::Union{Nothing,Float64} = nothing)
    length(w_profiled) == outer_dim_profiled(pe) ||
        throw(DimensionMismatch("decode_outer_profiled: expected length $(outer_dim_profiled(pe)), got $(length(w_profiled))"))
    gp = w_profiled[1]
    r_free = @view w_profiled[2:end]
    z_full = decode_full_z_on_retained(r_free, pe)
    Aod_levels = vec(exp.(z_full))
    xf = vcat(gp, Aod_levels)
    return (xf = xf, gp = gp, Aod_levels = Aod_levels, z_full = z_full)
end

"""
    reduce_to_w_profiled(gp, z_full, pe) -> Vector{Float64}

Inverse direction: full `(gp, z_full)` -> `w_profiled`. Used to seed a
profiled-formulation start from a calibrated/recovered full-A point, mirroring
`reduce_to_w_unified`'s role for the full formulation.
"""
function reduce_to_w_profiled(gp::Float64, z_full::AbstractMatrix{Float64}, pe::PivotGravityElimOnRetained)
    r = encode_relative_A(z_full, pe.spec, pe.gauge)
    r_free = pivot_reduce_on_retained(r, pe)
    return vcat(gp, r_free)
end
