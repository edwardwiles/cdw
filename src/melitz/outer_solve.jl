# RELABELED per the 2026-07-23 governing correction: `minimize_theta Delta(theta)` over
# all outer coordinates (this file's original "Section 4 nested Delta-star outer problem")
# is NOT an economic estimand -- the CC outer problem extremizes the counterfactual
# `g = log gamma_prime[j]` subject to a divergence BUDGET `Delta(theta) <= delta`, never
# unconstrained minimum divergence. This file is kept ONLY as a lightweight infrastructure
# regression test (does the nested inner-solve + Method-B-gradient + exterior-penalty
# machinery run at all, end to end, cold-verified) -- see `run_minimum_divergence_outer_
# smoke_test` below. Its result must never be reported as `Delta_star` or as an economic
# finding; the direct F* solve (fstar_direct.jl, Section 3) already gives a population-
# level Delta approx 0 certificate, so this smoke test's own terminal value is redundant
# with that as economics -- its only job is to exercise the outer-search code path that the
# real finite-delta upper/lower programs (Section 3/4 of the governing correction) reuse.
#
# Gradient: Method B (fixed-dual finite-bandwidth secant, gradient_lab.jl) -- the cheapest
# validated method (Section 5/6), matching production/fullA-exact's own L_fix construction
# (fullA_independent_assessor_brief_2026-07-22, Section 5.2). Optimizer: Optim.jl LBFGS
# (main prompt's own "prefers canned solvers" guidance) with an exterior quadratic penalty
# for the cutoff inequalities (same technique as fstar_direct.jl, and for the same reason:
# Optim's unconstrained methods have no native constraint support, and a full
# constrained-optimization dependency is not proportionate to an infrastructure smoke
# test). Each outer iteration re-solves the inner CC problem TWICE (once inside `f` for
# the current trial point, once inside `g!` to refresh Method B's base dual) -- simpler
# and more obviously correct than amortizing across a major/minor-iteration split, at the
# cost of extra inner solves; acceptable for D=4/W<=80,000's solve cost.

using Optim

"""
    melitz_outer_gradient_b!(grad, theta, x_base, ctx, obj; h=1e-3) -> grad

Fills `grad[k]` with Method B's central fixed-dual secant along the `k`-th COORDINATE
direction, for every `k` -- the outer gradient of `Delta(theta)` at fixed `x_base`.
"""
function melitz_outer_gradient_b!(grad::AbstractVector, theta::AbstractVector,
                                   x_base::AbstractVector, ctx, obj; h::Real=1e-3)
    n = length(theta)
    ei = zeros(n)
    @inbounds for k in 1:n
        ei[k] = 1.0
        r = method_b_fixed_dual_secant(theta, ei, h, x_base, ctx, obj)
        grad[k] = r.deriv
        ei[k] = 0.0
    end
    return grad
end

"""
    MinimumDivergenceSmokeTestResult

Result of the infrastructure-only `run_minimum_divergence_outer_smoke_test`. NOT an
economic result -- see the file-level note above. Fields otherwise as originally
documented: starting/final `Delta`, outer solver status, evaluation counts, and the
COLD-verified incumbent (re-solved from a cleared warm start, LFD independently
reconstructed, all Gate A residual checks rerun).
"""
struct MinimumDivergenceSmokeTestResult
    theta_init::Vector{Float64}
    theta_final::Vector{Float64}
    Delta_init::Float64
    Delta_final_warm::Float64      # Delta at theta_final from Optim's own last evaluation (warm)
    optim_result::Optim.OptimizationResults
    n_inner_solves::Int
    cold_verified::MelitzDeltaEvalResult   # evaluate_melitz_delta(theta_final; cold=true)
    wall_time::Float64
end

"""
    run_minimum_divergence_outer_smoke_test(theta_init, ctx, obj; h=1e-3, penalty=1e6,
                                             iterations=30, g_tol=1e-7)
        -> MinimumDivergenceSmokeTestResult

INFRASTRUCTURE DIAGNOSTIC ONLY (2026-07-23 governing correction) -- formerly named
`solve_melitz_delta_star_outer`, formerly (incorrectly) described as computing an economic
`Delta_star`. `minimize_theta Delta(theta)` over ALL outer coordinates is not an economic
estimand: the actual CC outer problem extremizes the counterfactual `g` subject to a
divergence budget `Delta(theta) <= delta` (see the finite-delta upper/lower programs).
This function is retained solely as a software regression test confirming the nested
outer-search machinery (Method B gradient, exterior-penalty cutoff constraints, cold
verification) runs end to end without crashing; do not read its terminal `Delta` as an
economic finding, and do not spend further effort tightening its convergence.

s.t. the Section 1.3 cutoff inequalities, from a given `theta_init`. Uses Method B for the
gradient (re-solving the inner problem once per `g!` call to refresh the base dual) and an
exterior penalty (weight `penalty`) on the deterministic cutoff constraints, whose gradient
is added EXACTLY via `melitz_cutoff_constraint_jacobian` (Section 1.3, ForwardDiff-exact,
smooth). Returns the run's own `Delta` at the last-evaluated (`_warm`) point AND a
`cold_verified` `MelitzDeltaEvalResult` at `theta_final` (fresh KNITRO solve from a cleared
warm start, independent LFD reconstruction, full ex-post equilibrium check via
`evaluate_melitz_delta`'s own machinery).
"""
function run_minimum_divergence_outer_smoke_test(theta_init::AbstractVector, ctx, obj;
                                        h::Real=1e-3, penalty::Real=1e4,
                                        iterations::Int=25, g_tol::Real=1e-7,
                                        time_limit::Real=180.0)
    t0 = time()
    n_inner_solves = Ref(0)

    function penalty_and_grad(theta)
        g_d, g_e = melitz_cutoff_constraints_at(theta, ctx)
        pen = sum(x -> min(x, 0.0)^2, g_d) + sum(x -> min(x, 0.0)^2, g_e)
        J_d, J_e = melitz_cutoff_constraint_jacobian(theta, ctx)
        grad_pen = 2 .* (J_d' * min.(g_d, 0.0) .+ J_e' * min.(g_e, 0.0))
        return pen, grad_pen
    end

    function f(theta)
        r = evaluate_melitz_delta(theta, ctx, obj; store_G=false)
        n_inner_solves[] += 1
        pen, _ = penalty_and_grad(theta)
        return r.Delta + penalty * pen
    end

    function g!(grad, theta)
        r = evaluate_melitz_delta(theta, ctx, obj; store_G=false)
        n_inner_solves[] += 1
        melitz_outer_gradient_b!(grad, theta, r.dual_x, ctx, obj; h=h)
        _, grad_pen = penalty_and_grad(theta)
        grad .+= penalty .* grad_pen
        return grad
    end

    r_init = evaluate_melitz_delta(collect(theta_init), ctx, obj; store_G=false)
    Delta_init = r_init.Delta
    n_inner_solves[] += 1

    result = Optim.optimize(f, g!, collect(theta_init), Optim.LBFGS(),
                             Optim.Options(iterations=iterations, g_tol=g_tol,
                                           allow_f_increases=true, time_limit=time_limit))
    theta_final = Optim.minimizer(result)

    r_final_warm = evaluate_melitz_delta(theta_final, ctx, obj; store_G=false)
    n_inner_solves[] += 1

    cold_verified = evaluate_melitz_delta(theta_final, ctx, obj; cold=true)
    n_inner_solves[] += 1

    return MinimumDivergenceSmokeTestResult(collect(theta_init), theta_final, Delta_init,
        r_final_warm.Delta, result, n_inner_solves[], cold_verified, time() - t0)
end
