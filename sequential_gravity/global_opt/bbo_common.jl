# ============================================================================
# Shared machinery for a gradient-free, population-based (BlackBoxOptim.jl) global
# search over the CC outer loop's free variables (gamma'_focal, A_od[1:D]).
#
# Motivation (see sequential_gravity/derivative_diagnostics/full_d2_correction_report.md
# section 7/7.1/7.2 and multistart_screening_d20.jl): the existing KNITRO outer search is
# local and gradient-based, always initialized at theta_r0 (A=A*), and at D=20 real data a
# severe gamma'-vs-Acol gradient-scale mismatch (~1e4-1e5x) leaves A essentially unexplored
# even with KNITRO-level variable scaling (which helps but doesn't converge cleanly and hurts
# D=4). multistart_screening_d20.jl found 20/20 random log-normal perturbations of A* (even to
# ~20x relative distance) were gravity-feasible with smoothly-scaling divergence cost -- strong
# evidence the local search is stuck in a basin, not that other basins don't exist. This file
# builds a fitness function ENTIRELY from `seq_gravcol`/`divergence_of`/`focal_bounds`
# (`run_profiled_production.jl`), with no gradients, no KNITRO outer-loop machinery, and no
# touching of any already-validated file.
#
# Must be included AFTER:
#   ENV["SKIP_BATCH_LOOP"] = "true"
#   include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
# which provides θr0, D, W, σ, KBOUNDS, seq_gravcol, divergence_of, focal_bounds, gp2kappa.
# ============================================================================

using LinearAlgebra, Printf

const Acol_star = θr0[4:3+D]
const θ_lo_full, θ_hi_full = focal_bounds(θr0)
const GP_LO, GP_HI = θ_lo_full[3], θ_hi_full[3]

# Free parameterization: x = [gamma'_focal, logratio(1:D)], logratio = log(Acol ./ Acol_star).
# Log-space (not focal_bounds' own raw-level +-1e4x box, and not raw Acol) because Acol is a
# multiplicative competitiveness parameter and multistart_screening_d20.jl found
# gravity-feasible points out to relΔA~20 under sigma=2 lognormal noise -- LOGBOUND=6.0
# (exp(6)=403x per-coordinate) gives real headroom past that evidence without the pathological
# 1e8x range focal_bounds' own box implies (which would swamp a population-based search).
const LOGBOUND = parse(Float64, get(ENV, "LOGBOUND", "6.0"))

function build_theta(x::AbstractVector)
    θ = copy(θr0)
    θ[3] = x[1]
    θ[4:3+D] .= Acol_star .* exp.(@view x[2:end])
    θ
end

x_from_theta(θ) = vcat(θ[3], log.(θ[4:3+D] ./ Acol_star))

search_range() = vcat([(GP_LO, GP_HI)], [(-LOGBOUND, LOGBOUND) for _ in 1:D])

# Big enough to dominate any feasible objective (gamma'_focal is bounded to (GP_LO,GP_HI) subset
# of (0,1) by construction), so every feasible candidate strictly beats every infeasible one; the
# remainder is a smooth measure of "how infeasible", giving the optimizer a descent direction
# back toward feasibility instead of a flat cliff at the boundary.
const INFEASIBLE_OFFSET = 100.0

function infeasibility_penalty(R, div_p, x, δbudget, tol)
    pen = 0.0
    pen += isfinite(R) ? max(0.0, abs(R) - tol) / tol : 1.0e4
    pen += isfinite(div_p) ? max(0.0, div_p - δbudget) / max(δbudget, 1.0e-6) : 1.0e4
    pen += 1.0e-3 * norm(@view x[2:end])   # pulls back toward logratio=0 (A*) when totally lost
    pen
end

"""
    make_fitness(; maximize_gp, δbudget, tol=5e-4, maxit=100)

Builds a scalar fitness (to be MINIMIZED by BlackBoxOptim) purely from `seq_gravcol` +
`divergence_of` -- no gradients, no KNITRO outer machinery.

`maximize_gp=true`  -> minimizing this fitness MAXIMIZES gamma'_focal (the LOWER kappa bound,
                       since kappa=gp2kappa(gp) is strictly decreasing in gp).
`maximize_gp=false` -> minimizing this fitness MINIMIZES gamma'_focal (the UPPER kappa bound).
Matches `run_one_bound`'s own `fs`/name convention in run_profiled_production.jl
(fs=true -> :upper -> gamma' MINIMIZED -> kappa maximized).
"""
function make_fitness(; maximize_gp::Bool, δbudget::Real, tol::Real = 5e-4, maxit::Int = 100)
    function fitness(x::AbstractVector)
        θ = build_theta(x)
        local R, div_p, ok
        try
            _, R, _, _, p, ok = seq_gravcol(θ; δ = Inf, maxit = maxit, tol = tol)
            div_p = divergence_of(p)
        catch e
            return INFEASIBLE_OFFSET + 1.0e4
        end
        feasible = ok && isfinite(div_p) && div_p <= δbudget * (1 + 1e-6) + 1e-10
        if feasible
            gp = x[1]
            return maximize_gp ? -gp : gp
        else
            return INFEASIBLE_OFFSET + infeasibility_penalty(R, div_p, x, δbudget, tol)
        end
    end
    fitness
end

"""
    eval_candidate(x; δbudget, tol=5e-4, maxit=100)

Re-evaluates a candidate `x` outside the fitness function's penalty machinery, returning the
full diagnostic tuple (θ, gp, kappa, R, div_p, feasible) -- used to verify/report the optimizer's
best point precisely once the search itself is done.
"""
function eval_candidate(x::AbstractVector; δbudget::Real, tol::Real = 5e-4, maxit::Int = 100)
    θ = build_theta(x)
    _, R, _, _, p, ok = seq_gravcol(θ; δ = Inf, maxit = maxit, tol = tol)
    div_p = divergence_of(p)
    feasible = ok && isfinite(div_p) && div_p <= δbudget * (1 + 1e-6) + 1e-10
    gp = x[1]
    κ = gp2kappa(gp)
    (θ = θ, gp = gp, κ = κ, R = R, div_p = div_p, feasible = feasible)
end
