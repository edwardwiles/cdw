# Direct Pareto F* benchmark parameter solver -- exact-sample finite correction solve.
# See docs/melitz_delta_star.md, main prompt Section 11, addendum Section 13.1.
#
# `fake_data.jl`'s fixture fixes `X_data = eq.trade_flow` (baseline "data", chosen
# independently of A/f during construction -- see that file's module docstring) and a
# gravity-feasible but NOT yet exact-sample-consistent outer point (`gamma_prime_target`
# from the ACR identity, `f[j,j]` derived from it, A/f gravity-projected). This file
# searches NEARBY (over the `2D^2-2` free gravity-pivoted coordinates, keeping `X_data`
# FIXED) for a point where the SAMPLE-MEAN (over the SAME `z_draws`, equal weights) of
# every one of the `D^2+1` active moments is ~0.
#
# METHOD: per-coordinate BISECTION (`Roots.jl`, canned), not a gradient-based minimizer.
# Found live: `melitz_firm`'s `active = profit > 0` participation gate makes the mean
# residual a function with small O(1/W) JUMPS (a firm crossing the cutoff for one draw) --
# `Optim.LBFGS` (both `autodiff=:forward` and `:finite`) either stalled far from zero
# (premature "convergence" on a biased/kink-blind gradient -- the same class of issue as
# this repo's own documented "winner-boundary derivative" bug in the related full-A
# codebase) or, with finite differences, took a wild line-search step into `exp(-800)`
# underflow. Bisection needs no derivative and each of the D^2+1 residuals is (to bisection
# resolution) MONOTONIC in one particular free coordinate: `price^(1-sigma)*active` is
# monotonic in `A_od` and in `f_od` (each shifts the cutoff/price in one direction), so
# every non-pivot free coordinate can be paired 1:1 with a target residual and solved by
# bisection, Gauss-Seidel sweeping over the (weak, single-shared-pivot-cell) coupling.

using Roots
using Optim

"""
    fstar_residual_sumsq(theta_free, ctx, z_draws) -> (sumsq, mean_residuals)

Sum of squared EQUAL-WEIGHT sample-mean moment residuals (`mean_residuals[k] =
mean(G[:,k])` over `z_draws`) at the free gravity-pivoted point `theta_free`, `ctx`
built the same way as `build_melitz_psi_bundle`'s (X_data FIXED at construction).
"""
function fstar_residual_sumsq(theta_free::AbstractVector{T}, ctx, z_draws::AbstractMatrix) where {T}
    d = ctx.moment_layout.num_moments
    W = size(z_draws, 1)
    K = zeros(T, W)
    G = zeros(T, W, d)
    melitz_moments_adapter!(K, G, theta_free, z_draws, (γ=ctx,))
    mean_residuals = vec(sum(G, dims=1)) ./ W
    return sum(abs2, mean_residuals), mean_residuals
end

"""
    bisect_coordinate!(theta, idx, target_col, ctx, z_draws; bracket0=0.25, max_bracket=6.0, xatol=1e-11)

Bisects `theta[idx]` (holding every other coordinate fixed) to zero
`fstar_residual_sumsq(theta,...)[2][target_col]`, expanding the search bracket by
doubling until a sign change is found, CAPPED at `max_bracket` (log-space -- `A`/`f`
ranging over `exp(-6)` to `exp(6)`, roughly 0.0025x to 400x their starting value, is
already generous; letting the bracket grow unbounded (an earlier version allowed up to
+-128) was found LIVE to occasionally reach `exp(*)` overflow/underflow at the bracket
ENDPOINTS used only to detect a sign change, corrupting that detection and causing
`Roots.find_zero` to converge to a numerically spurious root far from any economically
sensible point -- one coordinate's "correction" made its own residual 60x WORSE, not
better). No-op (leaves `theta` unchanged) if no sign change is found within
`max_bracket` -- reported via the returned `Bool`, never silently ignored.
"""
function bisect_coordinate!(theta::Vector{Float64}, idx::Int, target_col::Int, ctx,
                             z_draws::AbstractMatrix; bracket0::Float64=0.25,
                             max_bracket::Float64=6.0, xatol::Real=1e-11)
    f(x) = begin
        th = copy(theta)
        th[idx] = x
        fstar_residual_sumsq(th, ctx, z_draws)[2][target_col]
    end
    x0 = theta[idx]
    bracket = bracket0
    while bracket <= max_bracket
        lo, hi = x0 - bracket, x0 + bracket
        flo, fhi = f(lo), f(hi)
        if isfinite(flo) && isfinite(fhi) && sign(flo) != sign(fhi)
            theta[idx] = Roots.find_zero(f, (lo, hi), Roots.Bisection(); xatol=xatol)
            return true
        end
        bracket *= 2
    end
    return false
end

"""
    solve_pivot_pair!(theta, idx1, target1, idx2, target2, ctx, z_draws; fd_step=0.05,
                       iterations=60, damping=0.7, xtol=1e-10)

Joint 2D damped-Newton solve for coordinates `(idx1,idx2)` against their target residual
columns `(target1,target2)`, holding every other coordinate fixed. NEEDED because the
A-pivot cell and the f-pivot cell are genuinely mutually coupled: `A[A-pivot cell]`
depends on EVERY A_free coordinate (including the one dedicated to the f-pivot cell) via
the shared affine gravity constraint, and symmetrically `f[f-pivot cell]` depends on
every f_free coordinate (including the one paired with the A-pivot cell). Found LIVE,
reproducibly: naive alternating 1D bisection between these two converges geometrically
for a few sweeps then STALLS at a nonzero fixed point (confirmed not a bisection
artifact -- an independent full-30-dimension `Optim.LBFGS` polish landed at the exact
same stuck point). Uses a DELIBERATELY WIDE finite-difference step (`fd_step`, not
machine-epsilon scale) for the 2x2 Jacobian, to average over the `O(1/W)`
participation-jump noise in each residual rather than differentiate through it.
"""
function solve_pivot_pair!(theta::Vector{Float64}, idx1::Int, target1::Int, idx2::Int, target2::Int,
                            ctx, z_draws::AbstractMatrix; fd_step::Float64=0.05, iterations::Int=60,
                            damping::Float64=0.7, xtol::Real=1e-10)
    resid2(x1, x2) = begin
        th = copy(theta)
        th[idx1] = x1
        th[idx2] = x2
        _, r = fstar_residual_sumsq(th, ctx, z_draws)
        (r[target1], r[target2])
    end
    x1, x2 = theta[idx1], theta[idx2]
    for _ in 1:iterations
        f1, f2 = resid2(x1, x2)
        (abs(f1) < xtol && abs(f2) < xtol) && break
        f1_1, f2_1 = resid2(x1 + fd_step, x2)
        f1_2, f2_2 = resid2(x1, x2 + fd_step)
        J11, J21 = (f1_1 - f1) / fd_step, (f2_1 - f2) / fd_step
        J12, J22 = (f1_2 - f1) / fd_step, (f2_2 - f2) / fd_step
        detJ = J11 * J22 - J12 * J21
        abs(detJ) < 1e-10 && break
        dx1 = (f1 * J22 - f2 * J12) / detJ
        dx2 = (f2 * J11 - f1 * J21) / detJ
        x1 -= damping * dx1
        x2 -= damping * dx2
    end
    theta[idx1] = x1
    theta[idx2] = x2
    return theta
end

"""
    coordinate_descent_correction!(theta, ctx, z_draws; sweeps=6, kwargs...) -> theta

Gauss-Seidel coordinate-descent correction (addendum Section 13.1): each sweep bisects
every non-pivot free A coordinate against its OWN cell's trade-share residual, then
`gamma_prime_j` against the focal free-entry link residual; the ONE genuinely coupled
pair -- the A-pivot cell and the f-pivot cell (generically distinct cells) -- is solved
JOINTLY via `solve_pivot_pair!` (not alternating bisection, see that function's
docstring) once per sweep, after the independent coordinates have moved.
"""
function coordinate_descent_correction!(theta::Vector{Float64}, ctx, z_draws::AbstractMatrix,
                                         moment_layout::MelitzMomentLayout; sweeps::Int=6,
                                         bracket0::Float64=0.5, xatol::Real=1e-11)
    D = ctx.D
    nA = D^2 - 1
    avoid_f = f_pivot_avoid_index(ctx.A_pivot.pivot, ctx.f_free_lin)
    f_pivot_shape = build_gravity_pivot(ctx.c_full[ctx.f_free_lin], 0.0; avoid=avoid_f)
    f_cell_for_k(k) = lin2od(ctx.f_free_lin[f_pivot_shape.other[k]], D)
    po, pd = lin2od(ctx.A_pivot.pivot, D)
    qo, qd = lin2od(ctx.f_free_lin[f_pivot_shape.pivot], D)
    kmatch = findfirst(k -> f_cell_for_k(k) == (po, pd), 1:(D^2 - 2))
    kself = findfirst(==((qo, qd)), [lin2od(ctx.A_pivot.other[k], D) for k in 1:nA])

    for _ in 1:sweeps
        for k in 1:nA
            o, d = lin2od(ctx.A_pivot.other[k], D)
            (k == kself) && continue # handled jointly below, with the paired f coordinate
            bisect_coordinate!(theta, 1 + k, moment_layout.trade_index[o, d], ctx, z_draws;
                                bracket0=bracket0, xatol=xatol)
        end
        if kmatch !== nothing && kself !== nothing
            solve_pivot_pair!(theta, 1 + kself, moment_layout.trade_index[qo, qd],
                               1 + nA + kmatch, moment_layout.trade_index[po, pd], ctx, z_draws)
        end
        bisect_coordinate!(theta, 1, moment_layout.focal_link_index, ctx, z_draws;
                            bracket0=bracket0, xatol=xatol)
    end
    return theta
end

"""
    guarded_sumsq(theta_free, ctx, z_draws) -> Float64

`fstar_residual_sumsq`'s objective value, but catching exceptions/non-finite results and
returning a large finite penalty instead -- `Optim`'s line search can propose extreme
trial points (found live: `exp(-800)` underflow in `gamma_prime_j`, crashing
`MelitzPrimitives`'s `>0` check); this guard lets the line search back away from such
points instead of crashing the whole solve.
"""
function guarded_sumsq(theta_free::AbstractVector, ctx, z_draws::AbstractMatrix)
    val = try
        fstar_residual_sumsq(theta_free, ctx, z_draws)[1]
    catch
        NaN
    end
    return isfinite(val) ? val : 1e12
end

"""
    polish_local!(theta, ctx, z_draws; iterations=200, g_tol=1e-13) -> theta

Final joint LBFGS polish (`autodiff=:forward`, guarded via `guarded_sumsq`) over ALL free
coordinates, run AFTER `coordinate_descent_correction!`. Coordinate-wise bisection alone
was found live to converge geometrically but STALL at a small nonzero fixed point when
two cells share BOTH pivot dependencies (the A-pivot cell and the f-pivot cell, generically
distinct, each depend on every OTHER free A/f coordinate through the affine gravity
constraint -- alternating single-coordinate bisection between them does not in general
converge to their JOINT root). Once coordinate descent has removed the large,
kink-dominated part of the residual, the REMAINING small-residual neighborhood is smooth
enough that a gradient-based joint polish reliably finishes the job (ForwardDiff's
kink-blindness only matters when a step is large enough to cross a participation
boundary; near a small residual it rarely is).
"""
function polish_local!(theta::Vector{Float64}, ctx, z_draws::AbstractMatrix;
                        iterations::Int=200, g_tol::Real=1e-13)
    obj_fn(th) = guarded_sumsq(th, ctx, z_draws)
    result = Optim.optimize(obj_fn, theta, Optim.LBFGS(),
                             Optim.Options(iterations=iterations, g_tol=g_tol);
                             autodiff=:forward)
    theta .= Optim.minimizer(result)
    return theta
end

"""
    solve_fstar(data::MelitzSyntheticData; sweeps=6, bracket0=0.5, xatol=1e-11,
                polish_iterations=200) -> MelitzFStarResult

Addendum Section 13.1's exact-sample correction solve. Initializes at
`data.primitives`'s own (gravity-feasible) outer point (`reduce_to_free_theta`), runs
`coordinate_descent_correction!` to remove the large, kink-dominated part of the
residual, then `polish_local!` for a final joint gradient-based cleanup. Gravity
restrictions remain EXACT throughout (machine precision) regardless of the correction's
path -- they are structural, enforced by `expand_free_theta`'s pivot construction, never
something the correction could violate.
"""
function solve_fstar(data::MelitzSyntheticData; sweeps::Int=6, bracket0::Float64=0.5,
                      xatol::Real=1e-11, polish_iterations::Int=200)
    p = data.primitives
    D, j = p.D, p.target_country
    z_draws = data.z_draws
    eq = data.equilibrium

    moment_layout = MelitzMomentLayout(D)
    c_full, A_pivot = build_gravity_pivots(p.tau, j)
    outer_layout = melitz_outer_layout(D, j)

    ctx = (D=D, sigma=p.sigma, theta_star=p.theta_star, target_country=j, tau=p.tau, w=p.w,
           w_prime=data.counterfactual.w_prime, L=data.L, expenditure=eq.expenditure,
           cutoff=eq.cutoff, moment_layout=moment_layout, X_data=eq.trade_flow,
           c_full=c_full, A_pivot=A_pivot, jj_lin=outer_layout.jj_lin, f_free_lin=outer_layout.f_free_lin)

    theta = reduce_to_free_theta(p, ctx)
    coordinate_descent_correction!(theta, ctx, z_draws, moment_layout;
                                    sweeps=sweeps, bracket0=bracket0, xatol=xatol)
    polish_iterations > 0 && polish_local!(theta, ctx, z_draws; iterations=polish_iterations)
    _, mean_residuals = fstar_residual_sumsq(theta, ctx, z_draws)

    A_final, f_final, gamma_prime_final, f_jj_final = expand_free_theta(theta, ctx)
    primitives = MelitzPrimitives(D, p.sigma, p.theta_star, j, p.tau, p.w, A_final, f_final, gamma_prime_final)

    zhat_final = zeros(D, D)
    for o in 1:D, d in 1:D
        C_od = melitz_C(p.w[o], p.tau[o, d], A_final[o, d], p.sigma, eq.expenditure[d])
        zhat_final[o, d] = melitz_cutoff(p.w[o], f_final[o, d], p.sigma, C_od)
    end
    eq_final = MelitzEquilibrium(eq.expenditure, eq.price_power, zhat_final, eq.trade_flow)

    expenditure_prime = data.counterfactual.w_prime * data.L[j]
    counterfactual = MelitzCounterfactual(j, data.counterfactual.w_prime, expenditure_prime, one(Float64), expenditure_prime)

    max_trade_residual = maximum(abs.(mean_residuals[vec(moment_layout.trade_index)]))
    max_link_residual = abs(mean_residuals[moment_layout.focal_link_index])
    gravity_residual_A, gravity_residual_f = gravity_residuals(primitives)
    converged = maximum(abs.(mean_residuals)) < 1e-6
    status = converged ? :bisection_converged : :bisection_incomplete

    return MelitzFStarResult(primitives, eq_final, counterfactual, max_trade_residual,
                              max_link_residual, gravity_residual_A, gravity_residual_f,
                              converged, status)
end
