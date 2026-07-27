using LinearAlgebra: dot, norm

# 2026-07-25 local-geometry/continuation session, Phases 6-7: an EXPERIMENTAL
# block-trust-region predictor-corrector continuation prototype -- a separate driver, does
# NOT modify the production KNITRO outer search (`finite_delta_outer.jl`) at all.
#
# Motivation (governing prompt's own central acceptance criterion): determine, using fully
# reoptimized inner solves, whether a sequence of small finite steps exists that lowers `g`
# while keeping DeltaStar finite and controlling it near the budget boundary. The prior
# sessions' large-step joint-KNITRO failures are not evidence such a path does not exist --
# this driver tests small, gradient-informed steps directly, one at a time, each fully
# reoptimized (never trusting an unresolved point).
#
# Design, following the governing prompt's Phases 6-7 exactly:
#   - block-norm trust radii (r_g on |dg|, r_eta on ||d_eta||_2 -- an L2 ball, NOT a
#     per-coordinate box divided evenly, since a per-coordinate box of radius r/sqrt(n)
#     under-uses the ball in every direction except the axes) enforced by rescaling a
#     proposed step to the radius if it would exceed it;
#   - predictor: minimum-norm first-order tangent correction,
#     d_eta = -(q_g*dg / dot(q_eta,q_eta)) * q_eta;
#   - evaluation: a REAL fully-reoptimized evaluate_melitz_delta cold solve (never a
#     proxy/certificate);
#   - corrector: if the predicted point is finite but over budget, hold g fixed and take
#     one additional pure nuisance-descent step (-q_eta direction) to reduce Delta;
#   - acceptance: finite solved DeltaStar, cutoff-feasible, gravity-feasible (structural,
#     checked anyway), verified diagnostics -- reject and restore the prior verified point
#     otherwise (never update the "current point" from an unresolved/failed trial);
#   - adaptive radii: halve on any cap-exceedance/numerical-failure/poor first-order
#     prediction, expand only after several well-predicted accepted steps in a row (never
#     from the outer model's own prediction alone).

"""
    MelitzBlockTrustRadii(r_g, r_eta)

Block-norm trust radii for the predictor-corrector driver: `|dg| <= r_g` and
`norm(d_eta) <= r_eta` (an L2 ball on the eta/nuisance block, not a per-coordinate box --
governing prompt Phase 6: "do not approximate an L2 block radius with identical
per-coordinate boxes unless the box size is divided by sqrt(block_dimension)").
"""
mutable struct MelitzBlockTrustRadii
    r_g::Float64
    r_eta::Float64
end

"""
    melitz_project_to_radius!(v, radius) -> v

Rescales `v` in place to have norm exactly `radius` if `norm(v) > radius`, else leaves it
unchanged. The block-norm-projection step Phase 6 asks for, applied to the eta block.
"""
function melitz_project_to_radius!(v::AbstractVector{Float64}, radius::Real)
    nv = norm(v)
    if nv > radius && nv > 0
        v .*= (radius / nv)
    end
    return v
end

"""
    MelitzPredictorCorrectorStep

One attempted (accepted or rejected) step of `melitz_predictor_corrector_continuation`.
`accepted` false means the driver restored the PRIOR verified point exactly (governing
prompt Phase 7's "never update the warm-start bank or incumbent using an unresolved
point"). `kind` is `:predictor` or `:corrector` (a corrector step is logged separately from
the predictor step it follows, so the full attempted path is legible even when a corrector
is needed).
"""
struct MelitzPredictorCorrectorStep
    kind::Symbol
    accepted::Bool
    dg::Float64
    deta_norm::Float64
    Delta::Float64
    nStatus::Int
    kappa_ratio::Float64   # 2026-07-26 closure (Phase 3): renamed from `kappa` -- see
                            # equilibrium.jl's MelitzWelfareMetrics/kappa_ratio_of_g for why
                            # this is NOT the gains-from-trade by itself.
    min_slack::Float64
    predicted_dDelta::Float64
    realized_dDelta::Union{Nothing,Float64}
    r_g::Float64
    r_eta::Float64
    wall::Float64
end

"""
    MelitzPredictorCorrectorResult

Full record of a `melitz_predictor_corrector_continuation` run: `path` is every VERIFIED
accepted point (theta_free), `steps` is every attempted step (accepted or not, predictor or
corrector) in order, `final_theta`/`final_Delta`/`final_kappa_ratio` describe the last
accepted point. `final_kappa_ratio` is the pre-`1-` ratio term, NOT the gains-from-trade --
construct `melitz_welfare_metrics_from_g(final_theta[1], ctx)` (equilibrium.jl) for the
actual `GT_j = 1 - kappa_ratio` if welfare is what's wanted.
"""
struct MelitzPredictorCorrectorResult
    path::Vector{Vector{Float64}}
    steps::Vector{MelitzPredictorCorrectorStep}
    final_theta::Vector{Float64}
    final_Delta::Float64
    final_kappa_ratio::Float64
    n_accepted::Int
    n_rejected::Int
end

"""
    melitz_predictor_corrector_continuation(ctx, obj_inner, theta0; delta, cap,
        radii::MelitzBlockTrustRadii, dg_init, n_steps, gradient_backend=:B_direct_argument_parallel,
        h=1e-4, inner_loop_opt, outer_loop_opt=<default>, min_slack_floor=0.0,
        prediction_error_tol=0.5, expand_after=3, shrink_factor=0.5, expand_factor=1.5)
        -> MelitzPredictorCorrectorResult

`theta0` MUST already be a genuine, verified `nStatus==0` finite point (asserted). At each
of `n_steps` iterations: recompute `(q_g, q_eta)` at the CURRENT accepted point via the
production direct fixed-dual gradient backend (never re-derived differently), propose
`dg = -sign-preserving min(dg_init, r_g)` (always a DECREASE in `g`, per the governing
prompt's own "lowers g" acceptance criterion), the minimum-norm tangent `d_eta`, project
`d_eta` to `radii.r_eta`, evaluate the FULL reoptimized point. Accept iff finite (`nStatus in
(0,-100,-101,-103)`) and `min_slack >= min_slack_floor` (cutoff-feasible; gravity feasibility
is structural by pivot construction and re-verified via `evaluate_melitz_delta`'s own
`gravity_residual_*` fields, asserted rather than silently trusted). On acceptance with
`Delta > delta` (over budget but finite): one corrector step, `d_eta` along `-q_eta`
(recomputed AT THE PREDICTOR POINT'S own converged dual) with a small radius
(`radii.r_eta/4`), `g` held fixed. On rejection (unresolved / above-cap / gravity or cutoff
violated) or on a first-order prediction error exceeding `prediction_error_tol` (relative):
halve BOTH radii and restore the prior accepted point (do not advance). After
`expand_after` consecutive well-predicted accepted steps, multiply both radii by
`expand_factor`.
"""
function melitz_predictor_corrector_continuation(ctx, obj_inner, theta0::AbstractVector;
        delta::Real, cap::Real, radii::MelitzBlockTrustRadii, dg_init::Real,
        n_steps::Int, gradient_backend::Symbol=:B_direct_argument_parallel, h::Real=1e-4,
        inner_loop_opt::AbstractString,
        outer_loop_opt::AbstractString=joinpath(@__DIR__, "..", "..", "melitz_outer_finite_delta.opt"),
        min_slack_floor::Real=0.0, prediction_error_tol::Real=0.5,
        expand_after::Int=3, shrink_factor::Real=0.5, expand_factor::Real=1.5)
    n = length(theta0)
    theta_cur = collect(Float64.(theta0))
    r0 = evaluate_melitz_delta(theta_cur, ctx, obj_inner; cold=true, store_G=false)
    @assert r0.nStatus == 0 "melitz_predictor_corrector_continuation: theta0 must already be a genuine converged finite point"
    Delta_cur = r0.Delta
    kappa_ratio_cur = kappa_ratio_of_g(theta_cur[1], ctx)
    x_cur = r0.dual_x

    path = [copy(theta_cur)]
    steps = MelitzPredictorCorrectorStep[]
    n_accepted = 0
    n_rejected = 0
    consecutive_good = 0
    direct_gradient_fn = gradient_backend == :B_direct_argument_parallel ?
        make_melitz_gradient_delta_direct_parallel(h) : make_melitz_gradient_delta_direct_serial(h)

    function fresh_gradient(theta, x)
        # 2026-07-26 production-port session: this function is NOT part of the main
        # production outer loop (solve_melitz_finite_delta_bound) -- it is Stage 2's own
        # alternative predictor-corrector continuation strategy (Phase 8 audit) -- and reads
        # `obj.H`/`obj.moments!` directly below, which only the dense/legacy bundle exposes.
        # Pinned to `backend=:dense_reference` explicitly rather than silently inheriting the
        # new matrix-free default (which would break this function with a field-not-found
        # error); porting this Stage 2 path to the matrix-free bundle is out of scope this
        # session (docs/melitz_production_fast_backend_2026-07-26.md).
        obj = build_melitz_implicit_bundle(ctx, obj_inner.U, theta; delta=Float64(delta),
            find_smallest=true, gradient_backend=gradient_backend, h=h,
            inner_loop_opt=inner_loop_opt, outer_loop_opt=outer_loop_opt,
            backend=:dense_reference)
        CS = CounterfactualSensitivity
        G_now = CS.select_G_from_H(obj, obj.H)
        obj.moments!(@view(obj.H[:, 1]), G_now, theta, obj.U, obj)
        obj.H[:, 2] .= 1.0
        g = zeros(n)
        direct_gradient_fn(g, theta, ctx, obj, x)
        g ./= 1e10
        return g
    end

    for step_idx in 1:n_steps
        grad = fresh_gradient(theta_cur, x_cur)
        q_g = grad[1]
        q_eta = @view grad[2:end]
        dg = -min(abs(dg_init), radii.r_g) * sign(q_g == 0 ? -1.0 : 1.0)   # always decrease g
        d_eta = -(q_g * dg / dot(q_eta, q_eta)) .* collect(q_eta)
        melitz_project_to_radius!(d_eta, radii.r_eta)

        theta_try = copy(theta_cur)
        theta_try[1] += dg
        theta_try[2:end] .+= d_eta
        pred_dDelta = q_g * dg + dot(q_eta, d_eta)

        t0 = time()
        r_try = evaluate_melitz_delta(theta_try, ctx, obj_inner; cold=true, store_G=false)
        wall = time() - t0
        state_try = melitz_outer_state(theta_try, ctx)
        finite_try = r_try.nStatus == 0
        cutoff_ok = state_try.min_slack >= min_slack_floor
        actual_dDelta = finite_try ? r_try.Delta - Delta_cur : nothing
        pred_err = (finite_try && abs(pred_dDelta) > 1e-12) ?
            abs(actual_dDelta - pred_dDelta) / abs(pred_dDelta) : Inf

        accept = finite_try && cutoff_ok && pred_err <= prediction_error_tol
        push!(steps, MelitzPredictorCorrectorStep(:predictor, accept, dg, norm(d_eta),
            finite_try ? r_try.Delta : NaN, r_try.nStatus, finite_try ? kappa_ratio_of_g(theta_try[1], ctx) : NaN,
            state_try.min_slack, pred_dDelta, actual_dDelta, radii.r_g, radii.r_eta, wall))

        if accept && r_try.Delta > delta
            # Corrector: hold g fixed, take a small nuisance-descent step at the PREDICTOR
            # point's own converged dual to pull Delta back under budget.
            grad_try = fresh_gradient(theta_try, r_try.dual_x)
            q_eta_try = @view grad_try[2:end]
            d_eta_corr = -collect(q_eta_try) ./ max(norm(q_eta_try), 1e-300)
            melitz_project_to_radius!(d_eta_corr, radii.r_eta / 4)
            theta_corr = copy(theta_try)
            theta_corr[2:end] .+= d_eta_corr
            t0c = time()
            r_corr = evaluate_melitz_delta(theta_corr, ctx, obj_inner; cold=true, store_G=false)
            wallc = time() - t0c
            state_corr = melitz_outer_state(theta_corr, ctx)
            corr_ok = r_corr.nStatus == 0 && state_corr.min_slack >= min_slack_floor
            push!(steps, MelitzPredictorCorrectorStep(:corrector, corr_ok, 0.0, norm(d_eta_corr),
                corr_ok ? r_corr.Delta : NaN, r_corr.nStatus, corr_ok ? kappa_ratio_of_g(theta_corr[1], ctx) : NaN,
                state_corr.min_slack, NaN, corr_ok ? r_corr.Delta - r_try.Delta : nothing,
                radii.r_g, radii.r_eta, wallc))
            if corr_ok
                theta_try, r_try = theta_corr, r_corr
            else
                accept = false   # corrector failed too -- reject the whole predictor step
            end
        end

        if accept
            theta_cur = theta_try
            Delta_cur = r_try.Delta
            kappa_ratio_cur = kappa_ratio_of_g(theta_cur[1], ctx)
            x_cur = r_try.dual_x
            push!(path, copy(theta_cur))
            n_accepted += 1
            consecutive_good += 1
            if consecutive_good >= expand_after
                radii.r_g *= expand_factor
                radii.r_eta *= expand_factor
                consecutive_good = 0
            end
        else
            n_rejected += 1
            consecutive_good = 0
            radii.r_g *= shrink_factor
            radii.r_eta *= shrink_factor
            # theta_cur/Delta_cur/x_cur are UNCHANGED -- the prior verified point is restored
            # exactly, per the governing prompt's own explicit requirement.
        end
    end

    return MelitzPredictorCorrectorResult(path, steps, theta_cur, Delta_cur, kappa_ratio_cur,
        n_accepted, n_rejected)
end
