# Session prompt (2026-07-22) Section 3: direct F* feasibility solve, replacing the
# archived `solve_fstar`/`fstar_solver.jl` (which targeted the SUPERSEDED exact-sample-
# correction closure and is kept only as an optional debugging utility, docs Section 13.3)
# with a clean solver for the ACTIVE minimal (D^2+1)-moment / gravity-pivoted-free-theta
# system.
#
# Goal: find theta_free near the population-Pareto point at which the EQUAL-WEIGHT
# (reference-distribution) finite-sample mean of every one of the D^2+1 moments is ~0.
# Since equal weights are then (approximately) feasible for the CC inner minimum-
# divergence problem and divergence is nonnegative, this supplies a certificate that
# Delta_star(theta) = 0 to numerical tolerance AT this theta -- NOT a claim that A/f are
# uniquely recovered (the system is underidentified: 17 moments, 30 free coordinates).

using Optim
using Statistics: mean
using LinearAlgebra: norm

"""
    fstar_equal_weight_moments(theta_free, ctx, obj) -> Vector{Float64}

`m_Fstar(theta) = vec(mean(G(theta), dims=1))`, length `d = D^2+1` (session prompt
Section 3): the UNSCALED equal-(reference-)weight sample-mean moment vector, evaluated by
calling `obj.moments!` directly (no KNITRO dual solve -- this is a moment-matching
objective, not a divergence minimization).
"""
function fstar_equal_weight_moments(theta_free::AbstractVector{<:Real}, ctx, obj)
    W = size(obj.U, 1)
    K = zeros(Float64, W)
    G = zeros(Float64, W, obj.d)
    obj.moments!(K, G, theta_free, obj.U, obj)
    return vec(mean(G, dims=1))
end

"""
    fstar_default_scaling(ctx) -> Vector{Float64}

Session prompt Section 3: `S`, a documented diagonal scaling matrix (returned as a vector
of its diagonal, applied elementwise). Trade-share moments are scaled by `1.0` (already
O(1) trade-share units); the single focal free-entry link moment is scaled by `1.0` too --
both this closure's moments live in comparable (revenue-share-like / profit-share-like)
units at this fixture's calibration (verified empirically: population moment magnitudes at
the benchmark are all within about one order of magnitude of each other), so a uniform
scale is used rather than an artificial per-moment rescaling that would just as easily
distort as help. Kept as its own named function (not a bare `ones(d)`) so a future
calibration-specific rescaling has one obvious place to live.
"""
fstar_default_scaling(ctx) = ones(Float64, ctx.moment_layout.num_moments)

"""
    fstar_direct_objective(theta_free, ctx, obj, theta_population; S, rho, penalty) -> Float64

Session prompt Section 3's objective, `0.5*||S*m_Fstar(theta)||^2 + rho/2*||theta -
theta_population||^2`, PLUS a quadratic exterior penalty (`penalty * (sum(min.(g_d,0).^2)
+ sum(min.(g_e,0).^2))`) enforcing the Section 1.3 deterministic cutoff inequalities --
Optim.jl's unconstrained methods (used here, `Optim.LBFGS`, main prompt's own "canned
solver" preference) have no native constraint support, and the search is expected to stay
close to the ALREADY-feasible `theta_population` starting point, so an exterior penalty
(rather than a full constrained-optimization dependency) is a proportionate, documented
choice -- `penalty` defaults large enough (`1e8`) to make any constraint violation
dominate the objective immediately if the search strays.
"""
function fstar_direct_objective(theta_free::AbstractVector, ctx, obj,
                                 theta_population::AbstractVector;
                                 S::AbstractVector=fstar_default_scaling(ctx),
                                 rho::Real=1e-6, penalty::Real=1e8)
    m = fstar_equal_weight_moments(theta_free, ctx, obj)
    g_d, g_e = melitz_cutoff_constraints_at(theta_free, ctx)
    pen = sum(x -> min(x, 0.0)^2, g_d) + sum(x -> min(x, 0.0)^2, g_e)
    return 0.5 * sum(abs2, S .* m) + (rho / 2) * sum(abs2, theta_free .- theta_population) +
           penalty * pen
end

"""
    fstar_wide_bandwidth_gradient!(grad, theta_free, ctx, obj, theta_population; S, rho,
                                    penalty, h) -> grad

Central finite-difference gradient of `fstar_direct_objective`, using a DELIBERATELY WIDE
bandwidth `h` (default `1e-3`) -- NOT Optim's own default (tiny) step. This directly
follows the archived `solve_fstar`'s own documented finding (docs Section 11): the
per-cell trade-share moment mean has `O(1/W)` JUMPS as individual reference draws cross a
cutoff, so a naive small-`h` finite difference (or `ForwardDiff`) is locally blind to, or
badly noise-dominated by, that jump -- `Optim.LBFGS` with `autodiff=:forward`/`:finite`
was tried first there and found to stall far from zero. A wide `h` averages over several
draws' worth of jump noise instead of resolving (and being confused by) a single jump,
matching this session's own Section 5 Method B philosophy (fixed-bandwidth secant) applied
here to a moment-matching objective rather than the CC dual.
"""
function fstar_wide_bandwidth_gradient!(grad::AbstractVector, theta_free::AbstractVector, ctx, obj,
                                         theta_population::AbstractVector;
                                         S::AbstractVector=fstar_default_scaling(ctx),
                                         rho::Real=1e-6, penalty::Real=1e8, h::Real=1e-3)
    n = length(theta_free)
    @inbounds for k in 1:n
        tp = copy(theta_free); tp[k] += h
        tm = copy(theta_free); tm[k] -= h
        fp = fstar_direct_objective(tp, ctx, obj, theta_population; S=S, rho=rho, penalty=penalty)
        fm = fstar_direct_objective(tm, ctx, obj, theta_population; S=S, rho=rho, penalty=penalty)
        grad[k] = (fp - fm) / (2h)
    end
    return grad
end

"""
    MelitzFStarDirectResult

Session prompt Section 3's required report fields: initial/final equal-weight moments,
the regularizer's role, cutoff slacks, cold inner `Delta`, LFD deviation from equal
weights, and the numerically-zero `Delta_star` certificate.
"""
struct MelitzFStarDirectResult
    rho::Float64
    theta_free::Vector{Float64}
    theta_distance_from_population::Float64
    m_initial::Vector{Float64}
    m_final::Vector{Float64}
    max_abs_moment_initial::Float64
    max_abs_moment_final::Float64
    optim_result::Optim.OptimizationResults
    eval::MelitzDeltaEvalResult  # cold-verified real CC inner solve at theta_free
end

"""
    solve_fstar_direct(theta_population, ctx, obj; rho=1e-6, S=fstar_default_scaling(ctx),
                        penalty=1e8, h=1e-3, iterations=200, g_tol=1e-10) -> MelitzFStarDirectResult

Session prompt Section 3: solves `min_theta 0.5*||S*m_Fstar(theta)||^2 + rho/2*||theta -
theta_population||^2` s.t. the Section 1.3 cutoff inequalities (via exterior penalty),
starting from `theta_population` (the population-Pareto fixture's own reduced theta), via
`Optim.LBFGS` with the wide-bandwidth gradient above (canned OPTIMIZER, custom gradient --
main prompt's own "prefers canned solvers" guidance, and the archived `solve_fstar`'s own
documented reason a naive-gradient canned solve fails here). Then COLD-evaluates the real
CC inner problem at the result (`evaluate_melitz_delta(...; cold=true)`), certifying
`Delta_star approx 0` if `eval.Delta` is tiny and `eval.verified`.
"""
function solve_fstar_direct(theta_population::AbstractVector, ctx, obj;
                             rho::Real=1e-6, S::AbstractVector=fstar_default_scaling(ctx),
                             penalty::Real=1e4, h::Real=1e-3, iterations::Int=60,
                             g_tol::Real=1e-10, time_limit::Real=120.0)
    m_initial = fstar_equal_weight_moments(theta_population, ctx, obj)

    f(theta) = fstar_direct_objective(theta, ctx, obj, theta_population; S=S, rho=rho, penalty=penalty)
    g!(grad, theta) = fstar_wide_bandwidth_gradient!(grad, theta, ctx, obj, theta_population; S=S, rho=rho, penalty=penalty, h=h)

    result = Optim.optimize(f, g!, collect(theta_population), Optim.LBFGS(),
                             Optim.Options(iterations=iterations, g_tol=g_tol,
                                           allow_f_increases=true, time_limit=time_limit))
    theta_final = Optim.minimizer(result)

    m_final = fstar_equal_weight_moments(theta_final, ctx, obj)

    eval = evaluate_melitz_delta(theta_final, ctx, obj; cold=true)

    return MelitzFStarDirectResult(rho, theta_final, norm(theta_final .- theta_population),
        m_initial, m_final, maximum(abs.(m_initial)), maximum(abs.(m_final)), result, eval)
end
