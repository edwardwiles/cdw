# ============================================================================
# Transformed-A ("powered a-space") coordinate port for the FIXED-THETA CM-family drivers
# (run_cm_upper_checkpointed, cm_checkpoint.jl; run_originzc_upper_checkpointed,
# cm_originzc_checkpoint.jl), 2026-07-26 production-audit task addendum.
#
# Context: the transformed-A production port (port/transformed-A-and-flexible-theta-production-
# 2026-07-25, merged cdw/production/fullA-exact@da62166) landed the shared theta/A_od
# decorrelation reparametrization (a<->z, flexible_theta_aspace_production.jl) ONLY for the
# unrestricted family's unified driver (c10_d20_production_driver_unified.jl). None of the four
# restricted-family drivers (flexible CM, CM+ZC, ZC-only, common-Frechet) had ANY transformed-A
# wiring -- confirmed by grep, not assumed (zero references to OuterCoordinateLayout/A_coordinate_
# mode/transformed_a/powered_aspace anywhere in cm_*.jl before this file).
#
# This file provides the MINIMAL a<->z conversion these restricted drivers need, WITHOUT pulling
# in flexible_theta_aspace_production.jl's own dependency chain (make_flexible_theta/
# screened_eval/c10_d20_production_driver.jl) -- unnecessary for these drivers, which are and
# remain FIXED THETA ONLY (no restricted family supports flexible theta; this file provides no
# flexible-theta machinery and error()s if asked to). Same formulas as
# flexible_theta_aspace_production.jl::APivotXY/precompute_aspace_XY/z_from_a/a_from_z
# (algebraically identical, re-derived here rather than reused via include -- see that file's own
# docstring for the X/Y-object equivalence proof against gravity_elimination.jl::gravity_from_logz,
# which this reuses verbatim).
#
# KEY SIMPLIFICATION (verified below, not assumed): because these drivers are fixed-theta, the
# a<->z map at the ALREADY-FIXED theta the driver's own `pe::PivotGravityElim` was built at
# (build_pivot_elimination(ctx), no μ kwarg -> defaults to ctx.fixed_vals[1]) is POINTWISE AFFINE
# with a CONSTANT slope -theta (no per-callback re-derivation needed, unlike the flexible-theta
# port's own PivotGravityElimCache/pivot_expand_cheap machinery, built specifically to handle
# theta VARYING within a run -- restricted-family fixed-theta drivers don't need that generality
# and keep using their existing, already-validated `pe::PivotGravityElim`/`pivot_expand`/
# `pivot_reduce` UNCHANGED; only the outer boundary (KNITRO <-> z_nonpivot) gains an extra
# a<->z conversion step). This means:
#   - decode:  a_nonpivot -> z_nonpivot = cm_z_from_a(a_nonpivot, theta, xy, pe)
#              -> EXISTING pivot_expand(z_nonpivot, pe), UNCHANGED
#   - encode:  logA_full -> EXISTING pivot_reduce(logA_full, pe), UNCHANGED -> z_nonpivot
#              -> a_nonpivot = cm_a_from_z(z_nonpivot, theta, xy, pe)
#   - gradient: d(Delta)/d(a_nonpivot) = d(Delta)/d(z_nonpivot) * (-theta), a SCALAR rescale of
#     the EXISTING z-space gradient (cm_production_gradient_cplus / cm_meanzc_production_gradient_
#     cplus / cm_frechet_production_gradient_cplus, ALL unchanged, computed exactly as before) --
#     no new gradient computation, exactly mirroring outer_coordinate_layout.jl's
#     gradient_transform_unified's own `g[2:end] .*= (-theta)` line for A_coordinate_mode=
#     :powered_aspace.
# ============================================================================

isdefined(Main, :build_pivot_elimination) || error("cm_aspace_coordinate.jl requires gravity_elimination.jl to already be included (needs build_pivot_elimination/PivotGravityElim/pivot_expand/pivot_reduce).")

"Ctx-only (theta/gp/A-independent) constants for the a<->z conversion, D x Ddest. Same formula as flexible_theta_aspace_production.jl::APivotXY (algebraic equivalence proven there); computed independently here to avoid that file's flexible-theta-only dependency chain."
struct CMAPivotXY
    logX::Matrix{Float64}
    logY::Matrix{Float64}
end

"""
    precompute_cm_aspace_xy(ctx) -> CMAPivotXY

Same construction as `flexible_theta_aspace_production.jl::precompute_aspace_XY`:
`lambda = reshape(ctx.γ.P, (Ddest,D))'`, `X = (wHat.*τ)/(wHat[1,1]*τ[1,:]')`, `Y = lambda/lambda[1,:]'`
-- the rectangularized (post-omit-ROW) D x Ddest convention, matching `gravity_elimination.jl`'s
own reshape convention exactly (verified there against the pre-omit-ROW square prototype).
"""
function precompute_cm_aspace_xy(ctx)
    D = ctx.D
    Ddest = _ctx_ddest(ctx)
    γo = ctx.γ
    lambda = reshape(γo.P, (Ddest, D))'
    X = (γo.wHat .* γo.τ) ./ (γo.wHat[1, 1] .* γo.τ[1, :]')
    Y = lambda ./ lambda[1, :]'
    return CMAPivotXY(log.(X), log.(Y))
end

"CM's own fixed theta, matching build_pivot_elimination(ctx)'s own implicit default (no μ kwarg -> ctx.fixed_vals[1]) EXACTLY -- so the a<->z conversion and the pivot machinery always agree on which theta they mean."
cm_fixed_theta(ctx) = 1.0 / ctx.fixed_vals[1]

"a_nonpivot (length D*Ddest-1, pe.other_idx order) + theta -> z_nonpivot (SAME order). Inverse of cm_a_from_z. `pe` supplies other_idx (same pivot choice the driver's own pivot_expand/pivot_reduce use)."
function cm_z_from_a(a_nonpivot::AbstractVector{Float64}, theta::Float64, xy::CMAPivotXY, pe)
    logX_nonpivot = vec(xy.logX)[pe.other_idx]
    logY_nonpivot = vec(xy.logY)[pe.other_idx]
    return @. -theta * (a_nonpivot + logX_nonpivot) - logY_nonpivot
end

"z_nonpivot + theta -> a_nonpivot. Inverse of cm_z_from_a."
function cm_a_from_z(z_nonpivot::AbstractVector{Float64}, theta::Float64, xy::CMAPivotXY, pe)
    logX_nonpivot = vec(xy.logX)[pe.other_idx]
    logY_nonpivot = vec(xy.logY)[pe.other_idx]
    return @. -(z_nonpivot + logY_nonpivot) / theta - logX_nonpivot
end

"""
    cm_w0_from_calibration(ctx, pe, A_coordinate_mode::Symbol) -> Vector{Float64}

Builds a fresh `w0 = [gp; A_nonpivot_native]` from `ctx`'s own genuine calibrated `θ0_up`, in
whichever coordinate `A_coordinate_mode` selects. `:legacy_z` reproduces the EXISTING
`vcat(x_free_calib[1], pivot_reduce(log.(reshape(x_free_calib[2:end],D,Ddest)), pe))` pattern
every restricted-family calling script already used before this port, byte-identical. Does NOT
append meanzc `eta_nu` (orthogonal to the A-coordinate choice) -- callers using `cm_extension !=
:cm_only` append that separately, exactly as they already do.
"""
function cm_w0_from_calibration(ctx, pe, A_coordinate_mode::Symbol)
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    gp = x_free_calib[1]
    logA_full = log.(reshape(x_free_calib[2:end], ctx.D, ctx.D_dest))
    z_nonpivot = pivot_reduce(logA_full, pe)
    if A_coordinate_mode == :powered_aspace
        theta = cm_fixed_theta(ctx)
        xy = precompute_cm_aspace_xy(ctx)
        return vcat(gp, cm_a_from_z(z_nonpivot, theta, xy, pe))
    end
    return vcat(gp, z_nonpivot)
end
