# ============================================================================
# Task §22: external KKT stationarity check, independent of KNITRO's own
# reported optimality error. In the pivot-eliminated reduced coordinates
# (gravity already exactly satisfied, dropped as a constraint -- so no
# nu*grad_R term is needed here, unlike the task's general formula), with
# only ONE inequality constraint (Delta(w) <= delta) and box bounds:
#
#   grad_f + eta*grad_Delta + sum_i mu_i*e_i = 0,   eta >= 0,  eta*(Delta-delta)=0
#
# where mu_i are bound multipliers (nonzero only for coordinates AT a bound).
# For an interior point (no bounds active), this reduces to grad_f = -eta*grad_Delta,
# solved here by least squares (single scalar eta, since grad_f is the fixed
# unit vector e_1 up to sign).
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
using LinearAlgebra: dot, norm

"""
    external_stationarity_check(w, ctx, pe; find_smallest, h=0.01, w_lo=nothing, w_hi=nothing, bound_tol=1e-6)

Recomputes grad_f (trivial, unit vector) and grad_Delta (optimized-value
central FD, REUSING the same construction as run_d4_optimized_fd.jl's
eval_grad_central_fd but standalone here) at `w`, identifies any
near-active box bounds, solves the reduced KKT system (least squares over
eta and any active-bound multipliers), and reports the residual, eta, and
complementary slackness.
"""
function external_stationarity_check(w::AbstractVector, ctx, pe::PivotGravityElim;
        find_smallest::Bool, h::Float64 = 0.01,
        w_lo::Union{Nothing,AbstractVector} = nothing, w_hi::Union{Nothing,AbstractVector} = nothing,
        bound_tol::Float64 = 1e-4, δ::Float64 = 1.0)
    n = length(w)

    function x_free_from_w(ww)
        gp = ww[1]; zfree = ww[2:end]
        z = pivot_expand(zfree, pe)
        return vcat(gp, vec(exp.(z)))
    end
    function Delta_of_w(ww)
        r = evaluate_fullA(x_free_from_w(ww), ctx; cache = nothing, warm = true)
        return r.Delta_dual
    end

    Δ0 = Delta_of_w(w)
    grad_f = zeros(n); grad_f[1] = find_smallest ? 1.0 : -1.0
    grad_Delta = zeros(n)
    n_nonfinite = 0
    for i in 1:n
        wp = copy(w); wp[i] += h; wm = copy(w); wm[i] -= h
        Δp = Delta_of_w(wp); Δm = Delta_of_w(wm)
        if isfinite(Δp) && isfinite(Δm)
            grad_Delta[i] = (Δp - Δm) / (2h)
        else
            n_nonfinite += 1
            grad_Delta[i] = 0.0   # flagged, see n_nonfinite in return
        end
    end

    active_lo = w_lo === nothing ? falses(n) : (w .- w_lo .< bound_tol)
    active_hi = w_hi === nothing ? falses(n) : (w_hi .- w .< bound_tol)
    n_active_bounds = count(active_lo) + count(active_hi)

    # Interior-point reduced system (no active bounds): grad_f + eta*grad_Delta = 0 (least squares eta)
    eta = -dot(grad_f, grad_Delta) / max(dot(grad_Delta, grad_Delta), 1e-300)
    residual_vec = grad_f .+ eta .* grad_Delta
    residual_norm = norm(residual_vec)
    comp_slack = eta * (Δ0 - δ)

    return (Delta = Δ0, Delta_minus_delta = Δ0 - δ, eta = eta, eta_nonneg = eta >= -1e-8,
            residual_norm = residual_norm, residual_relative = residual_norm / max(norm(grad_f), 1e-12),
            complementary_slackness = comp_slack, n_active_bounds = n_active_bounds,
            n_nonfinite_probes = n_nonfinite, grad_Delta = grad_Delta)
end
