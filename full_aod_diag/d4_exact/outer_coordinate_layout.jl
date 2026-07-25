# ============================================================================
# Shared outer-coordinate layout abstraction (addendum, 2026-07-25): unifies fixed-theta and
# flexible-theta outer drivers onto ONE coordinate-decoding/gradient-transform/checkpoint-
# fingerprint core, per the addendum's explicit "do not finalize production with separate,
# near-duplicate fixed and flexible outer drivers" instruction. Supersedes the original port's
# §4 architecture decision (a near-duplicate flexible-only driver), which was made under the
# ORIGINAL brief before this addendum arrived.
#
# Three independent axes:
#   trade_elasticity_mode :fixed | :flexible   -- whether theta is searched (eta_theta present
#                                                  in the outer vector) or held at ctx's own theta.
#   A_coordinate_mode     :legacy_z | :powered_aspace  -- whether the free A-block coordinates are
#                                                  z_nonpivot = log(Aod_theta) (current production's
#                                                  existing coordinate) or a_nonpivot = log(AodPow)
#                                                  (the theta-decoupled coordinate).
#   gp_coordinate_mode    :raw | :scaled_log   -- whether gp itself is searched directly, or via
#                                                  u_g = s_g*log(gp/gp_star) (task §7, experimental).
#
# CONSTRAINT (addendum §2): trade_elasticity_mode==:flexible REQUIRES A_coordinate_mode==
# :powered_aspace (enforced in the constructor below) -- the theta-decoupling that makes a-space
# valuable for flexible theta (see FLEXIBLE_THETA_ASPACE_MATHEMATICAL_PARAMETERIZATION_2026-07-25.md)
# has no counterpart in z-space (the old z-space flexible arm was kept ONLY as D=4 comparison
# scaffolding, never a production candidate).
#
# Outer vector shape (generic, matches every combination):
#   w = [eta_theta?; gp_coord; A_coord(D*Ddest-1)]
# `eta_theta` present iff trade_elasticity_mode==:flexible. Length is D*Ddest (:fixed) or
# D*Ddest+1 (:flexible) -- 380 / 381 at real D=20 post-omit-ROW.
#
# This file depends on gravity_elimination.jl (PivotGravityElimCache/pivot_expand_cheap/
# pivot_reduce_cheap) and flexible_theta_aspace_production.jl (APivotXY/precompute_aspace_XY/
# z_from_a/a_from_z) -- both already generic in theta (not flexible-mode-specific), reused
# verbatim, not duplicated.
# ============================================================================

isdefined(Main, :build_pivot_elimination_cheap) || error("outer_coordinate_layout.jl requires gravity_elimination.jl's cheap analytic path.")
isdefined(Main, :precompute_aspace_XY) || error("outer_coordinate_layout.jl requires flexible_theta_aspace_production.jl (APivotXY/z_from_a/a_from_z).")

"""
    OuterCoordinateLayout

Immutable descriptor of which outer-coordinate convention a driver call uses. Constructed via
`make_layout` (validates the flexible/:powered_aspace constraint) -- never construct directly.
"""
struct OuterCoordinateLayout
    trade_elasticity_mode::Symbol
    A_coordinate_mode::Symbol
    gp_coordinate_mode::Symbol
end

function make_layout(; trade_elasticity_mode::Symbol, A_coordinate_mode::Symbol, gp_coordinate_mode::Symbol = :raw)
    trade_elasticity_mode in (:fixed, :flexible) ||
        error("make_layout: trade_elasticity_mode must be :fixed or :flexible, got :$trade_elasticity_mode")
    A_coordinate_mode in (:legacy_z, :powered_aspace) ||
        error("make_layout: A_coordinate_mode must be :legacy_z or :powered_aspace, got :$A_coordinate_mode")
    gp_coordinate_mode in (:raw, :scaled_log) ||
        error("make_layout: gp_coordinate_mode must be :raw or :scaled_log, got :$gp_coordinate_mode")
    trade_elasticity_mode == :flexible && A_coordinate_mode != :powered_aspace &&
        error("make_layout: trade_elasticity_mode=:flexible REQUIRES A_coordinate_mode=:powered_aspace " *
              "(got :$A_coordinate_mode) -- the theta-decoupling a-space provides has no z-space " *
              "counterpart; z-space flexible theta is D=4 comparison scaffolding only, never a " *
              "production combination. See addendum §2.")
    return OuterCoordinateLayout(trade_elasticity_mode, A_coordinate_mode, gp_coordinate_mode)
end

"Outer vector length for a layout at a given (D, Ddest): D*Ddest (:fixed) or D*Ddest+1 (:flexible)."
outer_dim(layout::OuterCoordinateLayout, D::Int, Ddest::Int) = D * Ddest + (layout.trade_elasticity_mode == :flexible ? 1 : 0)

"""
    GpScale

Precomputed constants for `gp_coordinate_mode=:scaled_log`: `u_g = s_g*log(gp/gp_star)`,
inverse `gp = gp_star*exp(u_g/s_g)`, `d(gp)/d(u_g) = gp/s_g` (used by the gradient chain rule).
`gp_star` is the calibration-point gp (task §7's own notation); `s_g` a fixed positive scale
(default 1.0 -- task §7 says do not default to scaled-log at all yet, this struct exists so an
opt-in caller can supply a nondefault scale for its confirmation runs).
"""
struct GpScale
    gp_star::Float64
    s_g::Float64
end
GpScale(gp_star::Float64) = GpScale(gp_star, 1.0)

"gp_coord -> raw gp, per layout.gp_coordinate_mode."
function decode_gp(gp_coord::Float64, layout::OuterCoordinateLayout, gs::Union{Nothing,GpScale})
    if layout.gp_coordinate_mode == :raw
        return gp_coord
    else
        gs === nothing && error("decode_gp: gp_coordinate_mode=:scaled_log requires a GpScale")
        return gs.gp_star * exp(gp_coord / gs.s_g)
    end
end

"raw gp -> gp_coord, per layout.gp_coordinate_mode (inverse of decode_gp)."
function encode_gp(gp::Float64, layout::OuterCoordinateLayout, gs::Union{Nothing,GpScale})
    if layout.gp_coordinate_mode == :raw
        return gp
    else
        gs === nothing && error("encode_gp: gp_coordinate_mode=:scaled_log requires a GpScale")
        return gs.s_g * log(gp / gs.gp_star)
    end
end

"d(Delta)/d(gp_coord) from d(Delta)/d(gp_raw), per layout.gp_coordinate_mode chain rule."
function rescale_gp_gradient(dDelta_dgp_raw::Float64, gp_raw::Float64, layout::OuterCoordinateLayout, gs::Union{Nothing,GpScale})
    if layout.gp_coordinate_mode == :raw
        return dDelta_dgp_raw
    else
        gs === nothing && error("rescale_gp_gradient: gp_coordinate_mode=:scaled_log requires a GpScale")
        return dDelta_dgp_raw * (gp_raw / gs.s_g)   # d(gp_raw)/d(u_g) = gp_raw/s_g
    end
end

"""
    decode_outer_unified(w, ctx, layout, pgc, xy, gs) -> NamedTuple

Generic outer-vector decode covering all (trade_elasticity_mode, A_coordinate_mode,
gp_coordinate_mode) combinations. `ctx` may be a fixed-mode ctx (θ_full[1]=mu fixed, no
eta_theta in free_idx) or a flexible-mode ctx (make_flexible_theta'd, mu at free_idx[1]).
`pgc::PivotGravityElimCache` and `xy::APivotXY` are ctx-only precomputed constants (theta-
independent; reused across every call, built once per outer base point / ctx respectively).

Returns `(xf, theta, mu, gp, z_nonpivot, A_nonpivot_native, eta_theta)` -- `xf` is the exact
shape `screened_eval`/`evaluate_fullA_screened_ranged` expect (matching `ctx.m`'s own free_idx
convention: `[gp; Aod_levels]` for a fixed ctx, `[mu; gp; Aod_levels]` for a flexible ctx).
`A_nonpivot_native` is `z_nonpivot` (A_coordinate_mode=:legacy_z, identical object) or
`a_nonpivot` (A_coordinate_mode=:powered_aspace) -- whichever coordinate the OUTER vector
actually holds, for checkpoint/reduce round-trips.
"""
function decode_outer_unified(w::AbstractVector{Float64}, ctx, layout::OuterCoordinateLayout,
        pgc::PivotGravityElimCache, xy::APivotXY, gs::Union{Nothing,GpScale} = nothing)
    D = pgc.D; Ddest = pgc.Ddest
    if layout.trade_elasticity_mode == :flexible
        eta_theta = w[1]
        tol = 1e-9 * max(1.0, abs(ctx.θ_lo[1]), abs(ctx.θ_hi[1]))
        if !(ctx.θ_lo[1] - tol <= eta_theta <= ctx.θ_hi[1] + tol)
            reject_point(eta_theta, "decode_outer_unified: eta_theta=$eta_theta outside configured box [$(ctx.θ_lo[1]), $(ctx.θ_hi[1])]")
        end
        theta = exp(eta_theta)
        rest = @view w[2:end]
    else
        theta = hasproperty(ctx, :theta_star) ? ctx.theta_star : 1.0 / ctx.μHat
        eta_theta = log(theta)
        rest = @view w[1:end]
    end
    mu = 1.0 / theta
    gp_coord = rest[1]
    gp = decode_gp(gp_coord, layout, gs)
    A_nonpivot_native = rest[2:end]
    length(A_nonpivot_native) == D * Ddest - 1 ||
        throw(DimensionMismatch("decode_outer_unified: A-block has length $(length(A_nonpivot_native)), expected D*Ddest-1=$(D*Ddest-1)"))

    if layout.A_coordinate_mode == :powered_aspace
        logX_nonpivot = vec(xy.logX)[pgc.other_idx]
        logY_nonpivot = vec(xy.logY)[pgc.other_idx]
        z_nonpivot = @. -theta * (A_nonpivot_native + logX_nonpivot) - logY_nonpivot
    else
        z_nonpivot = A_nonpivot_native
    end

    Aod_levels = vec(exp.(pivot_expand_cheap(collect(z_nonpivot), pgc, mu)))
    xf = layout.trade_elasticity_mode == :flexible ? vcat(mu, gp, Aod_levels) : vcat(gp, Aod_levels)
    return (xf = xf, theta = theta, mu = mu, gp = gp, z_nonpivot = collect(z_nonpivot),
            A_nonpivot_native = collect(A_nonpivot_native), eta_theta = eta_theta)
end

"""
    reduce_to_w_unified(theta, gp, logA_full, pgc, xy, layout, gs) -> Vector{Float64}

Inverse direction: full (theta, gp, log-A matrix D x Ddest) -> the reduced outer vector `w`,
matching `layout`'s exact coordinate conventions. Used to seed a KNITRO start point from a
cold-verified incumbent's full A matrix / checkpoint, generic across every layout combination.
"""
function reduce_to_w_unified(theta::Float64, gp::Float64, logA_full::AbstractMatrix{Float64},
        pgc::PivotGravityElimCache, xy::APivotXY, layout::OuterCoordinateLayout, gs::Union{Nothing,GpScale} = nothing)
    z_nonpivot = pivot_reduce_cheap(logA_full, pgc)
    A_nonpivot_native = layout.A_coordinate_mode == :powered_aspace ? vec(a_from_z(logA_full, theta, xy))[pgc.other_idx] : z_nonpivot
    gp_coord = encode_gp(gp, layout, gs)
    base = vcat(gp_coord, A_nonpivot_native)
    return layout.trade_elasticity_mode == :flexible ? vcat(log(theta), base) : base
end

"""
    gradient_transform_unified(gfull_reduced_z, theta, gp_raw, layout, gs) -> Vector{Float64}

Rescales the EXISTING production z-space gradient `gfull_reduced_z = [d(Delta)/d(gp_raw);
d(Delta)/d(z_nonpivot)]` (from `composite_gradient_at_Cplus`, unmodified, always computed in raw
gp / z-space regardless of `layout` -- this is the one shared numerical kernel every combination
reuses) into the coordinates `layout` actually searches over: `[d(Delta)/d(gp_coord);
d(Delta)/d(A_nonpivot_native)]`, length `D*Ddest` (no eta_theta row -- the caller prepends the
theta-secant's own `d(Delta)/d(eta_theta)` separately in flexible mode, since that derivative
requires the fixed-dual secant machinery, not a coordinate rescale of this vector).
"""
function gradient_transform_unified(gfull_reduced_z::AbstractVector{Float64}, theta::Float64, gp_raw::Float64,
        layout::OuterCoordinateLayout, gs::Union{Nothing,GpScale} = nothing)
    g = copy(gfull_reduced_z)
    g[1] = rescale_gp_gradient(g[1], gp_raw, layout, gs)
    if layout.A_coordinate_mode == :powered_aspace
        g[2:end] .*= (-theta)
    end
    return g
end

"""
    layout_fingerprint(layout, gs) -> String

Short, stable string encoding a layout's identity for cache/checkpoint fingerprints -- e.g.
`"fixed|powered_aspace|scaled_log|gpstar=0.9878|sg=1.0"`. Distinguishes every combination this
port supports; two identical layouts (including gp-scale constants when relevant) fingerprint
identically.
"""
function layout_fingerprint(layout::OuterCoordinateLayout, gs::Union{Nothing,GpScale} = nothing)
    base = "$(layout.trade_elasticity_mode)|$(layout.A_coordinate_mode)|$(layout.gp_coordinate_mode)"
    layout.gp_coordinate_mode == :scaled_log && gs !== nothing && (base *= "|gpstar=$(gs.gp_star)|sg=$(gs.s_g)")
    return base
end

const AMAP_VERSION = 1   # "exact mapping version" (addendum §4) -- bump if z<->a formula changes

"""
    dual_bank_zfree(d, layout) -> Vector{Float64}

Reconciliation fix (transformed-A/flexible-theta production port task §12): production port
`port/flexible-theta-aspace-production-2026-07-25`'s `DualBank` warm-start selection was
theta-blind by construction -- `select_warm_start`'s scaled nearest-neighbor search
(`dual_bank.jl`) operates on whatever `zfree` vector it is handed, and every prior flexible-mode
call site handed it `d.z_nonpivot` alone (the A-block only), so two points differing only in
theta always distance-collapsed to `d==0` and could be selected as "nearest" regardless of how
far apart their thetas actually were.

Fix: in flexible mode, prepend `eta_theta=log(theta)` to the vector DualBank is keyed on, so its
existing per-coordinate scaled-distance metric (`dual_bank.jl`: `std` over bank history once
>=3 points exist, else unweighted) naturally extends to penalize theta distance on the same
footing as A-block distance. No change to `dual_bank.jl` itself -- it was already generic over
the length/content of `zfree`, only production's own call sites were passing an incomplete key.
`eta_theta=log(theta)` is O(1) (theta is O(1)-O(10) in this model), i.e. already roughly the same
order of magnitude as the log-A_od entries it's concatenated with, so the unweighted (<3-point)
fallback case is not badly scaled even before the std-normalization kicks in.

In fixed mode this is the identity (`d.z_nonpivot` unchanged) -- theta never varies within a
fixed-theta run, so there is nothing to distinguish.
"""
dual_bank_zfree(d, layout::OuterCoordinateLayout) =
    layout.trade_elasticity_mode == :flexible ? vcat(d.eta_theta, d.z_nonpivot) : d.z_nonpivot
