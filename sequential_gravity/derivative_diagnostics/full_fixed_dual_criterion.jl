# ============================================================================
# Full-(D+2)-moment fixed-dual criterion.
#
# Fixes the bug documented at the top of derivative_methods_report.md's
# follow-on task: the Part 1/2 machinery in fixed_dual_criterion.jl /
# fixed_dual_fd.jl / boundary_derivative.jl was validated and wired using only
# the REDUCED (D+1)-moment baseline (D trade shares + 1 price-index moment),
# dropping the gravity-linearized moment G[:,D+2] and its dual multiplier
# lambda_R entirely. But the actual production inner problem has D+2 moments
# (see run_profiled_production.jl::outer_solve_nested_cached, `d = D + 2`),
# and lambda_R*G_R enters the conjugate argument
#     arg0 = -zeta - lambda_b'*G_b - lambda_R*G_R
# as a common offset inside the NONLINEAR Psi(arg0) -- so it generally shifts
# the winner-boundary jump too, even though G_R itself has no winner/argmax
# dependence. Dropping it and patching the result on afterward (the OLD
# gradient_method_wiring.jl's "g_free_OLD_AD + reduced_FD_correction" design)
# is not generally valid. This file instead builds the FULL (D+2)-moment
# fixed-dual criterion directly, so it can be finite-differenced as one
# self-contained scalar function of theta.
#
# `build_fixed_dual_bundle` and `dual_criterion_fixed_x` (fixed_dual_criterion.jl,
# already included by the time this file loads) are REUSED UNCHANGED -- both
# are already generic in `d`/`moments_fn`, so no rewrite is needed there. The
# only new piece is the moments_fn itself: the frozen-gravity full-D+2
# moments function below.
# ============================================================================

"""
    make_frozen_gravity_moments(trade_moments_fn!, D, lastθ, lastRcol, gcol, dRdθ)

Returns a `moments!(K, G, θ, U, obj)` function computing the FULL (D+2)-moment
vector at arbitrary theta, with:
  - G[:,1:D+1] (the D focal trade shares + 1 price-index moment) recomputed
    FRESH at theta via `trade_moments_fn!` (normally
    `EK_moments_focal_norm_directgp!`) -- exact hard argmin winners, no
    smoothing, no caching: perturbing Acol genuinely changes winners here.
  - G[:,D+2] (the gravity-linearized moment) held at the FROZEN affine
    surrogate around the current sequential iterate (lastθ, lastRcol, gcol,
    dRdθ):
        Rlin(theta) = lastRcol + dot(dRdθ, theta - lastθ)
        G[ω,D+2]    = (gcol[ω] - lastRcol) + Rlin(theta)
    i.e. the per-draw INFLUENCE-FUNCTION SHAPE (gcol[ω]-lastRcol, a fixed
    per-draw offset) is frozen, and only the cross-draw MEAN LEVEL is moved
    by theta, via the linear total-derivative dRdθ. This is EXACTLY the
    formula run_profiled_production.jl::make_stateful_moments's `m!` already
    uses internally for ForwardDiff.Dual theta (the AD path) -- this function
    makes it available for Float64 theta too, which is what finite
    differences need (production's own Float64 branch instead calls
    `seq_gravcol` and genuinely RE-INVERTS the omitted destinations, which is
    exactly what the task instructs NOT to do inside a fixed-local-criterion
    evaluation: "not redo the nonlinear inversion merely because A is
    perturbed").
  - lastθ/lastRcol/gcol/dRdθ are captured as plain Float64 values at
    construction time (not Refs) so that every perturbation inside one
    finite-difference gradient evaluation shares IDENTICAL frozen
    linearization data (common random numbers style consistency) even if the
    caller's own stateful closure keeps moving in the meantime.

`gcol` must have length >= size(U,1) actually used (`obj.N` in production,
or W in the standalone validation drivers); only the first `nrow=size(G,1)`
entries are read, mirroring production's `gcol[][1:nrow]`.
"""
function make_frozen_gravity_moments(trade_moments_fn!::Function, D::Int,
        lastθ::Vector{Float64}, lastRcol::Float64, gcol::Vector{Float64}, dRdθ::Vector{Float64};
        CM_Moments::Union{Nothing,AbstractMatrix{Float64}} = nothing)
    lastθ_c = copy(lastθ); gcol_c = copy(gcol); dRdθ_c = copy(dRdθ)
    return function (K, G, θ, U, obj)
        trade_moments_fn!(K, @view(G[:, 1:D+1]), θ, U, obj)
        nrow = size(G, 1)
        Rlin = lastRcol + dot(dRdθ_c, θ .- lastθ_c)
        @inbounds @views @. G[:, D+2] = (gcol_c[1:nrow] - lastRcol) + Rlin
        # Common-marginals block (CDW eq. 35/36), if enabled: theta-INDEPENDENT, so no freezing/
        # linearization is needed (unlike the gravity column above) -- a plain copy is exact at
        # every perturbed theta the finite-difference gradient evaluates, matching
        # append_cm_moments!'s convention exactly (run_profiled_production.jl). Ported/extended
        # 2026-07-16 to make GRADIENT_METHOD=fixed_dual_fd_full usable with CM_ENABLED=true.
        if CM_Moments !== nothing
            ncm = size(CM_Moments, 2)
            @views G[:, D+3:D+2+ncm] .= CM_Moments[1:nrow, :]
        end
        return nothing
    end
end

"""
    freeze_gravity_linearization(θ, seq_gravcol_fn, grad_R_theta_fn; δ=Inf, tol=5e-4, warm=nothing, warm_p=nothing)

Convenience helper for the validation drivers (mirrors what
`make_stateful_moments`'s `m!` does internally at a fresh Float64 theta):
solves the exact D+1-moment blind CC problem, inverts the omitted
destinations, iterates the sequential gravity-linearization loop to
`seq_gravcol`'s own convergence, then computes dRdθ via `grad_R_theta`.
Returns a NamedTuple (θ, Rcol, gcol, dRdθ, umat, p, ok, gravity_ok, div_p, R)
suitable for `make_frozen_gravity_moments`. Runs the REAL production
sequential loop, not a re-implementation -- this is Part 4's "at a real
sequential iterate" input.

IMPORTANT (bug caught auditing a fixed-A endpoint that looked spuriously
"gravity-infeasible" -- see full_d2_correction_report.md): `seq_gravcol`'s own
`ok` return is `gravity_ok && δ_ok`, i.e. it ALSO checks whether the
divergence(p) needed to reach `θ` is within the `δ` BUDGET PASSED TO
`seq_gravcol` -- a check that is meaningful when `seq_gravcol` is being used
INSIDE an outer delta-constrained KNITRO search (make_stateful_moments's own
use), but is IRRELEVANT to "can we freeze a valid gravity linearization at
this theta at all" -- gravity feasibility (abs(R)<=tol) and budget
feasibility (div(p)<=δ) are two INDEPENDENT conditions, and conflating them
made the sequential loop's ordinary "this A needs MORE divergence budget to
reach this gamma'" outcome look like "the gravity equation itself is
violated at A*" -- a nonsensical/impossible claim for a variable (gravity)
that this code computes from an IMPUTED-fundamentals matrix, not literally a
population identity independent of the reweighting used to construct that
matrix. Default `δ=Inf` here makes `δ_ok` always true, so this function's own
`.ok`/`.gravity_ok` fields depend ONLY on the gravity-residual tolerance, as
intended for freezing a linearization or auditing the exact delta* at an
arbitrary theta. Pass a finite `δ` only if you specifically want
`seq_gravcol`'s OWN budget-gated behavior (e.g. reproducing what the outer
KNITRO search itself saw at that iterate).
"""
function freeze_gravity_linearization(θ::Vector{Float64}, seq_gravcol_fn::Function, grad_R_theta_fn::Function;
        δ::Real=Inf, tol::Real=5e-4, maxit::Int=100, warm=nothing, warm_p=nothing)
    # maxit=100 (vs seq_gravcol's own default 20): caught auditing a fixed-A point that looked
    # gravity-infeasible at maxit=20 (R_mean=-1.21e-3) but converges cleanly by iteration 50
    # (R_mean=-2.76e-4, stable through 200) -- i.e. the default budget was an ITERATION-LIMIT
    # artifact, not genuine infeasibility, understating what a fixed-A comparator can actually
    # reach. A generous maxit here costs little (only matters when NOT already converged) and
    # avoids silently mischaracterizing "needs more sequential iterations" as "cannot converge".
    col, R, Rcol, umat, p, ok_combined = seq_gravcol_fn(θ; δ=δ, maxit=maxit, warm=warm, warm_p=warm_p)
    gravity_ok = isfinite(R) && abs(R) <= tol
    div_p = isempty(p) ? NaN : divergence_of(p)
    if !gravity_ok
        # genuinely NOT gravity-feasible (or the blind solve / a destination inversion failed
        # outright) -- return gracefully (ok=false) rather than throwing, so callers auditing an
        # ARBITRARY (possibly KNITRO-endpoint) theta can check .ok themselves. A KNITRO outer solve
        # CAN converge to a gravity-infeasible endpoint (the `constr[1]` divergence value computed
        # during infeasible evaluations uses the make_stateful_moments placeholder
        # INFCOL column, not a real gravity residual -- see run_profiled_production.jl's own
        # best_θ/best_κ "best gravity-feasible point seen" tracker, which exists precisely because the
        # raw KNITRO endpoint is not always trustworthy). Callers that need a REAL frozen linearization
        # (e.g. Part 4's validation, which always calls this at an already-known-feasible theta) should
        # assert `.ok` themselves and error with task-specific context.
        return (θ=copy(θ), Rcol=NaN, gcol=Float64[], dRdθ=Float64[], umat=zeros(0,0), p=Float64[],
                ok=false, gravity_ok=false, div_p=div_p, budget_ok=ok_combined, R=R)
    end
    dRdθ = grad_R_theta_fn(θ, umat, p)
    return (θ=copy(θ), Rcol=Rcol, gcol=copy(col), dRdθ=dRdθ, umat=umat, p=p,
            ok=true, gravity_ok=true, div_p=div_p, budget_ok=ok_combined, R=R)
end

"""
    test_full_fixed_dual_identity(θ, frozen, γobj, U, D; l=length(θ))

Part 4's required first check: builds the frozen full-(D+2) moments function
from `frozen` (a `freeze_gravity_linearization` result), solves the exact
D+2-moment inner problem at theta (via `inner_loop`, reusing
`test_fixed_dual_identity` from fixed_dual_criterion.jl unchanged -- it is
already generic in d/moments_fn), and checks
`Q_full(θ,x*) == δ*(θ)`. STOP AND DIAGNOSE if this fails before computing any
derivative -- per the task's explicit instruction.
"""
function test_full_fixed_dual_identity(θ::Vector{Float64}, frozen::NamedTuple, γobj, U::Matrix{Float64}, D::Int;
        trade_moments_fn!::Function=EK_moments_focal_norm_directgp!, l::Int=length(θ), find_smallest::Bool=true,
        CM_Moments::Union{Nothing,AbstractMatrix{Float64}} = nothing)
    moments_fn = make_frozen_gravity_moments(trade_moments_fn!, D, frozen.θ, frozen.Rcol, frozen.gcol, frozen.dRdθ;
        CM_Moments = CM_Moments)
    nCM = CM_Moments === nothing ? 0 : size(CM_Moments, 2)
    return test_fixed_dual_identity(θ, moments_fn, D + 2 + nCM, γobj, U; l=l, find_smallest=find_smallest)
end
