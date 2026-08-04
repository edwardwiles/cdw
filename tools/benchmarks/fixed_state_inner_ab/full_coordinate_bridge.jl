# Task "fixed-state FULL-vs-REDUCED inner A/B", derisking step before 5/6: converts a decoded
# economic state (gp, full log-A D x Ddest matrix -- exactly what `decode_outer_profiled` recovers
# from a REDUCED sentinel point) into the REAL FULL production `w0` vector for each family's own
# REAL production `A_coordinate_mode`, confirmed by directly reading `bin/run_profiled_model.jl`'s
# own dispatch (not assumed, not the family's full menu of SUPPORTED modes per FamilyRegistry.jl):
#   unrestricted / flexible_cm / common_frechet / cm_meanzc: :powered_aspace
#   origin_zc:                                               :legacy_z
#
# Read-only with respect to every full_aod_diag file (only `include`d, never edited). Does not
# duplicate any conversion math -- reuses `cm_w0_from_calibration`'s own primitives
# (`pivot_reduce`/`cm_a_from_z`/`precompute_cm_aspace_xy`/`cm_fixed_theta`, cm_aspace_coordinate.jl)
# for the 4 CM-family cases, and `reduce_to_w_unified`/`build_pivot_elimination_cheap`/
# `precompute_aspace_XY` (outer_coordinate_layout.jl / flexible_theta_aspace_production.jl /
# gravity_elimination.jl) for unrestricted -- just applied to an ARBITRARY decoded state instead
# of only `ctx.θ0_up`'s own calibration point, which is all `cm_w0_from_calibration` itself covers.

const D4X_BRIDGE = joinpath(dirname(dirname(dirname(@__DIR__))), "full_aod_diag", "d4_exact")

for f in ["cm_aspace_coordinate.jl", "flexible_theta_aspace_production.jl"]
    isdefined(Main, :cm_w0_from_calibration) || f != "cm_aspace_coordinate.jl" || include(joinpath(D4X_BRIDGE, f))
    isdefined(Main, :reduce_to_w_unified) || f != "flexible_theta_aspace_production.jl" || include(joinpath(D4X_BRIDGE, f))
end

const CM_FAMILY_A_COORDINATE_MODE = Dict(
    :unrestricted => :powered_aspace, :flexible_cm => :powered_aspace,
    :common_frechet => :powered_aspace, :cm_meanzc => :powered_aspace, :origin_zc => :legacy_z)

"""
    full_w0_from_state(family, gp, logA_full, ctx) -> Vector{Float64}

Builds the REAL FULL production cold-start `w0` (economic block only -- `eta_nu` for ZC families
is appended separately by the caller, orthogonal to the A-coordinate choice per
`cm_w0_from_calibration`'s own docstring) from an arbitrary decoded `(gp, logA_full)` state, using
each family's REAL confirmed `A_coordinate_mode` (see module docstring). `family===:unrestricted`
uses the separate unrestricted-only pivot/xy machinery; the other 4 share the CM machinery
(`build_pivot_elimination`, single-pivot, distinct from REDUCED's own
`PivotGravityElimOnRetained`).
"""
function full_w0_from_state(family::Symbol, gp::Float64, logA_full::AbstractMatrix{Float64}, ctx)
    mode = CM_FAMILY_A_COORDINATE_MODE[family]
    if family === :unrestricted
        pgc = build_pivot_elimination_cheap(ctx; mu_probe1 = 1.0, mu_probe2 = 2.0)
        xy = precompute_aspace_XY(ctx)
        theta = hasproperty(ctx, :theta_star) ? ctx.theta_star : 1.0 / ctx.μHat
        layout = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = mode, gp_coordinate_mode = :raw)
        return reduce_to_w_unified(theta, gp, logA_full, pgc, xy, layout)
    else
        pe = build_pivot_elimination(ctx)
        z_nonpivot = pivot_reduce(Matrix(logA_full), pe)
        if mode === :powered_aspace
            theta = cm_fixed_theta(ctx)
            xy = precompute_cm_aspace_xy(ctx)
            return vcat(gp, cm_a_from_z(z_nonpivot, theta, xy, pe))
        else
            return vcat(gp, z_nonpivot)
        end
    end
end

"""
    full_decode_w0(family, w0, ctx) -> (gp, Aod_levels)

Inverse direction, for the round-trip derisking check: decodes a FULL-native `w0` (economic block
only) back to `(gp, Aod_levels)` using the SAME machinery, so `full_decode_w0(family,
full_w0_from_state(family, gp0, logA0, ctx), ctx)` should reproduce `(gp0, vec(exp.(logA0)))`
to machine precision if the bridge above is correct.
"""
function full_decode_w0(family::Symbol, w0::AbstractVector{Float64}, ctx)
    mode = CM_FAMILY_A_COORDINATE_MODE[family]
    if family === :unrestricted
        pgc = build_pivot_elimination_cheap(ctx; mu_probe1 = 1.0, mu_probe2 = 2.0)
        xy = precompute_aspace_XY(ctx)
        theta = hasproperty(ctx, :theta_star) ? ctx.theta_star : 1.0 / ctx.μHat
        layout = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = mode, gp_coordinate_mode = :raw)
        dec = decode_outer_unified(w0, ctx, layout, pgc, xy)
        return (dec.gp, dec.xf[2:end])
    else
        pe = build_pivot_elimination(ctx)
        gp = w0[1]
        A_nonpivot_native = w0[2:end]
        if mode === :powered_aspace
            theta = cm_fixed_theta(ctx)
            xy = precompute_cm_aspace_xy(ctx)
            z_nonpivot = cm_z_from_a(A_nonpivot_native, theta, xy, pe)
        else
            z_nonpivot = A_nonpivot_native
        end
        z_full = pivot_expand(z_nonpivot, pe)
        return (gp, vec(exp.(z_full)))
    end
end
