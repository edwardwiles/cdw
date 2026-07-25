# ============================================================================
# theta/A_od decorrelation reparametrization ("a-space"), production port, 2026-07-25.
#
# Ported from experiment/fullA-theta-aspace-reparam-2026-07-25's flexible_theta_aspace.jl
# onto current production's rectangular (D origins x D_dest active destinations,
# post-omit-ROW) layout and current screened_eval/ScreenCounters/DualBank/SafeExactCache/
# SafeNegativeCache stack (c10_d20_production_driver.jl) -- NOT copied verbatim: the
# experimental file assumed a square D x D layout (predates the omit-ROW release) and called
# its OWN screened_eval_flexible (a near-duplicate of the z-space driver's screened_eval).
# Here, `screened_eval_flexible_A` below delegates directly to c10_d20_production_driver.jl's
# EXISTING `screened_eval` (unchanged) once `decode_and_expand_flexible_A` has produced an
# `xf` in the exact shape `screened_eval`/`evaluate_fullA_screened_ranged` already expect from
# a ctx built via `make_flexible_theta` (mu free at free_idx[1]) -- no new screening code.
#
# See docs/FLEXIBLE_THETA_ASPACE_MATHEMATICAL_PARAMETERIZATION_2026-07-25.md for:
#   - the exact a<->z<->AodPow<->A_od mapping this file implements (cross-checked against
#     BOTH the experimental flexible_theta_aspace.jl's header derivation AND current
#     production's own gravity_elimination.jl::gravity_from_logz -- both give the identical
#     AodPow = (Aod_theta*Y)^(-mu)/X formula, confirming X,Y are exactly the same objects);
#   - why the pivot cell itself stays z-parametrized (gravity_elimination.jl's
#     PivotGravityElimCache/pivot_expand_cheap, theta-invariant pivot/slope, only the affine
#     intercept g0(mu) moves with theta -- rectangularized in this port, see
#     docs/FLEXIBLE_THETA_RECTANGULAR_GRAVITY_AUDIT_2026-07-25.md).
#
# DESIGN NOT CHANGED FROM THE EXPERIMENTAL FILE: because z = z_from_a(a,theta,xy) is pointwise
# affine at fixed theta (same scalar slope -theta for every cell), the whole file needs only:
# (1) the a<->z conversion, (2) a decode wrapper, (3) a gradient rescale (chain rule dz/da=-theta,
# a scalar, applied at driver call sites, not here), (4) a theta-secant wrapper holding
# a_nonpivot fixed. It does NOT touch composite_gradient_at_fast_buffered/composite_gradient_at_
# Cplus, build_lfix_base_cache, screens, caching, or the z-space pivot machinery -- all reused
# unchanged via freeze_theta_ctx (flexible_theta.jl) at driver call sites.
# ============================================================================

isdefined(Main, :make_flexible_theta) || error("flexible_theta_aspace_production.jl requires flexible_theta.jl to already be included (needs make_flexible_theta/decode_theta_full/freeze_theta_ctx).")
isdefined(Main, :build_pivot_elimination_cheap) || error("flexible_theta_aspace_production.jl requires gravity_elimination.jl's cheap analytic path to already be included.")
isdefined(Main, :reject_point) || error("flexible_theta_aspace_production.jl requires oracle.jl to already be included (needs reject_point).")
isdefined(Main, :screened_eval) || error("flexible_theta_aspace_production.jl requires c10_d20_production_driver.jl to already be included (needs screened_eval/ScreenCounters/DualBank/SafeExactCache).")

"""
    APivotXY

Ctx-only (theta/gp/A-independent) constants for the a<->z conversion, built once per ctx like
`pgc`: `logX[o,d] = log((wHat[o]*tau[o,d]) / (wHat[1,1]*tau[1,d]))`,
`logY[o,d] = log(lambda[o,d]/lambda[1,d])`, `lambda = reshape(gamma.P, (Ddest,D))'` -- the SAME
`lambda`/`X` convention `gravity_elimination.jl::gravity_from_logz` uses (rectangularized:
`lambda[1,d]` here means "origin 1's share of destination d", matching gravity_from_logz's own
`lambda_g[1,:]'` broadcast; the `cHat` factor cancels out of `AodPow` entirely, verified
algebraically -- see the mathematical-parameterization doc).
"""
struct APivotXY
    logX::Matrix{Float64}   # D x Ddest
    logY::Matrix{Float64}   # D x Ddest
end

"""
    precompute_aspace_XY(ctx) -> APivotXY

Rectangularized (D x Ddest) version of the experimental square-D<D=D=D> precompute_aspace_XY --
uses the SAME `lambda = reshape(ctx.γ.P, (Ddest,D))'` reshape convention
`gravity_elimination.jl::gravity_from_logz` and `fast_range_screen.jl`'s `Pmat` construction
both already use for the active-destination-sliced P vector, not the square `reshape(P,(D,D))`
the pre-omit-ROW experimental prototype used (that would silently misalign columns whenever
Ddest != D, i.e. under the current production default `destination_sample=:exclude_row`).
"""
function precompute_aspace_XY(ctx)
    D = ctx.D
    Ddest = _flex_ddest(ctx)
    γo = ctx.γ
    lambda = reshape(γo.P, (Ddest, D))'
    X = (γo.wHat .* γo.τ) ./ (γo.wHat[1, 1] .* γo.τ[1, :]')
    Y = lambda ./ lambda[1, :]'
    return APivotXY(log.(X), log.(Y))
end

"a (D x Ddest) + theta -> z (D x Ddest). z[o,d] = -theta*(a[o,d]+logX[o,d]) - logY[o,d]. Inverse of a_from_z."
z_from_a(a::AbstractMatrix, theta::Float64, xy::APivotXY) = @. -theta * (a + xy.logX) - xy.logY

"z (D x Ddest) + theta -> a (D x Ddest). a[o,d] = -(z[o,d]+logY[o,d])/theta - logX[o,d]. Inverse of z_from_a."
a_from_z(z::AbstractMatrix, theta::Float64, xy::APivotXY) = @. -(z + xy.logY) / theta - xy.logX

"""
    decode_and_expand_flexible_A(w_ext_a, ctx, xy) -> NamedTuple

`w_ext_a = [eta_theta; gp; a_nonpivot]` (length `D*Ddest+1`, same shape/length as the z-space
`w_ext` -- only the last `D*Ddest-1` entries' units differ: a-space instead of log(Aod_theta)).

The pivot cell itself stays z-parametrized -- its value is whatever nulls the gravity
constraint exactly, from the theta-dependent affine-offset machinery
(`pivot_expand_cheap`/`PivotGravityElimCache`), untouched. Only the `D*Ddest-1` NON-pivot cells
are converted a->z pointwise at the current theta before delegating to the existing z-space
expansion. Returns `xf = [mu; gp; Aod_levels...]`, the exact shape `screened_eval`/
`evaluate_fullA_screened_ranged` already expect from a ctx built via `make_flexible_theta`.
"""
function decode_and_expand_flexible_A(w_ext_a::AbstractVector{Float64}, ctx, xy::APivotXY)
    (hasproperty(ctx, :trade_elasticity_mode) && ctx.trade_elasticity_mode == :flexible) ||
        error("decode_and_expand_flexible_A: ctx is not in flexible-theta mode -- call make_flexible_theta first")
    eta_theta = w_ext_a[1]
    tol = 1e-9 * max(1.0, abs(ctx.θ_lo[1]), abs(ctx.θ_hi[1]))
    if !(ctx.θ_lo[1] - tol <= eta_theta <= ctx.θ_hi[1] + tol)
        reject_point(eta_theta, "decode_and_expand_flexible_A: eta_theta=$eta_theta (theta=$(exp(eta_theta))) outside " *
            "configured box [$(ctx.θ_lo[1]), $(ctx.θ_hi[1])] (log space; linear box [$(ctx.theta_lo), $(ctx.theta_hi)])")
    end
    theta = exp(eta_theta)
    mu = 1.0 / theta
    gp = w_ext_a[2]
    a_nonpivot = @view w_ext_a[3:end]
    Ddest = _flex_ddest(ctx)
    D = ctx.D
    length(a_nonpivot) == D * Ddest - 1 ||
        throw(DimensionMismatch("decode_and_expand_flexible_A: w_ext_a has length $(length(w_ext_a)), expected D*Ddest+1=$(D*Ddest+1)"))
    pgc = build_pivot_elimination_cheap(ctx; mu_probe1 = 1.0 / ctx.theta_lo, mu_probe2 = 1.0 / ctx.theta_hi)
    logX_nonpivot = vec(xy.logX)[pgc.other_idx]
    logY_nonpivot = vec(xy.logY)[pgc.other_idx]
    z_nonpivot = @. -theta * (a_nonpivot + logX_nonpivot) - logY_nonpivot
    Aod_levels = vec(exp.(pivot_expand_cheap(z_nonpivot, pgc, mu)))
    xf = vcat(mu, gp, Aod_levels)
    return (xf = xf, mu = mu, theta = theta, eta_theta = eta_theta, gp = gp, pgc = pgc,
            z_nonpivot = collect(z_nonpivot), a_nonpivot = collect(a_nonpivot))
end

"""
    reduce_to_w_ext_A(theta, gp, logA_full, pgc, xy) -> Vector{Float64}

Inverse direction: full (theta, gp, log-A matrix, D x Ddest) -> the reduced a-space outer
vector `w_ext_a = [log(theta); gp; a_nonpivot]`, dropping the pivot coordinate.
"""
function reduce_to_w_ext_A(theta::Float64, gp::Float64, logA_full::AbstractMatrix{Float64}, pgc::PivotGravityElimCache, xy::APivotXY)
    a_full = a_from_z(logA_full, theta, xy)
    a_nonpivot = vec(a_full)[pgc.other_idx]
    return vcat(log(theta), gp, a_nonpivot)
end

"""
    screened_eval_flexible_A(w_ext_a, ctx, rsc, sc, n_eval_ref, xy; warm=true, bank=nothing,
                              exact_cache=nothing, neg_cache=nothing) -> (result, screen_meta, decoded)

Delegates to the EXISTING, unmodified production `screened_eval` (c10_d20_production_driver.jl)
once `decode_and_expand_flexible_A` has produced `xf`. Returns the extra `decoded` NamedTuple
(theta/mu/gp/pgc/z_nonpivot/a_nonpivot) so callers (checkpoint/gradient/theta-secant code) don't
have to decode twice.
"""
function screened_eval_flexible_A(w_ext_a::AbstractVector{Float64}, ctx, rsc::RangedScreenContext, sc::ScreenCounters,
        n_eval_ref::Ref{Int}, xy::APivotXY; warm::Bool = true, bank::Union{Nothing,DualBank} = nothing,
        exact_cache::Union{Nothing,SafeExactCache,CrossDeltaExactCache} = nothing,
        neg_cache::Union{Nothing,SafeNegativeCache} = nothing)
    d = decode_and_expand_flexible_A(w_ext_a, ctx, xy)
    result, screen_meta = screened_eval(d.xf, ctx, rsc, sc, n_eval_ref; warm = warm, bank = bank,
        zfree = d.z_nonpivot, exact_cache = exact_cache, neg_cache = neg_cache)
    return result, screen_meta, d
end

"Typed 2-tuple cold-verify wrapper: (result, decoded). Discards screen_meta."
function screened_eval_flexible_A_verify(w_ext_a::AbstractVector{Float64}, ctx, rsc::RangedScreenContext, sc::ScreenCounters,
        n_eval_ref::Ref{Int}, xy::APivotXY; warm::Bool = false, exact_cache = nothing)
    result, _, d = screened_eval_flexible_A(w_ext_a, ctx, rsc, sc, n_eval_ref, xy; warm = warm, exact_cache = exact_cache)
    return result, d
end

"""
    theta_fixed_dual_delta_pivot_A(w_ext_a, inner_x_fixed, ctx, xy) -> Delta_dual

Evaluates the fixed-dual objective at the theta implied by `w_ext_a[1]`, holding `a_nonpivot`
(NOT `z_nonpivot`) fixed -- the entire point of the a-space reparametrization for the gradient:
a theta perturbation at fixed `a_nonpivot` lets `z_nonpivot` (hence `Aod_theta`) shift exactly
enough to keep `AodPow` fixed to leading order, isolating theta's remaining genuine
Frechet-dispersion effect (via `mu*log(U[s,o])` inside the winner-price comparison) instead of
re-triggering the dominant `mu*z` rescaling term the OLD z-space parametrization conflated it
with. Does NOT re-solve the inner KNITRO dual problem -- mirrors exactly how
`inner_loop_internal` itself calls `obj.moments!`, minus the solve step.
"""
function theta_fixed_dual_delta_pivot_A(w_ext_a::AbstractVector{Float64}, inner_x_fixed::AbstractVector{Float64}, ctx, xy::APivotXY)
    d = decode_and_expand_flexible_A(w_ext_a, ctx, xy)
    obj = ctx.obj
    θ_full = CS.reconstruct_full(d.xf, ctx.m)
    obj.moments!(@view(obj.H[:, 1]), CS.select_G_from_H(obj, obj.H), θ_full, obj.U, obj)
    obj.H[:, 2] .= 1.0
    fval = obj(inner_x_fixed)
    return -fval   # Delta_dual sign convention -- see oracle.jl's documented derivation
end
