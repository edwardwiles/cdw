# q-bandwidth convergence campaign (2026-07-29), Phase 11: ONE explicitly named experimental
# (A,q) outer-gradient backend, combining:
#   - the welfare (gamma) coordinate: a cheap fixed-dual central-difference secant (the
#     "already accurate welfare-coordinate derivative" the governing prompt refers to -- the
#     SAME construction every existing FD-based direct backend already uses for this
#     coordinate, not a new derivation);
#   - the A block: the EXACT analytical envelope-theorem gradient (`exact_a_gradient.jl`,
#     2026-07-29 prior-session commit) -- zero finite-difference probes;
#   - the q block: the CONFIGURED `MelitzQBandwidthPolicy` (`q_bandwidth_policy.jl`, this
#     session's Phase 1), via `melitz_q_coordinate_probe` in `:fixed_dual` mode -- the SAME
#     source-level evaluator the diagnostic scripts use (governing prompt Rule 10: no second,
#     script-only implementation).
#
# NOT wired as a new production default -- registered as an ADDITIONAL, explicitly named
# `gradient_backend` symbol (`:B_direct_argument_aq_experimental`) alongside the 5 existing
# direct backends; every existing symbol's dispatch branch is untouched (see the two additive
# edits in `finite_delta_outer.jl`, both pure `elseif`-style additions with zero changes to
# any pre-existing line).
#
# Disclosed limitation: not allocation-optimized (unlike the zero-allocation production direct
# backends) -- allocates fresh workspaces every call. Acceptable for an EXPERIMENTAL backend
# used only in the bounded Phase 12 smoke test, not for a production hot path.

"""
    make_melitz_gradient_delta_direct_aq_experimental(q_policy::MelitzQBandwidthPolicy; gamma_h=1e-6)

Factory (mirrors `make_melitz_gradient_delta_direct_sorted_serial(h)`'s own factory pattern)
returning a `direct_gradient_fn`-compatible closure `(g, theta, ctx, obj, x) -> g` for the
`:B_direct_argument_aq_experimental` backend. Requires `ctx.outer_parameterization ==
:logcutoff` -- throws `ArgumentError` otherwise (never silently falls back to `:logf`
semantics).
"""
function make_melitz_gradient_delta_direct_aq_experimental(q_policy::MelitzQBandwidthPolicy; gamma_h::Real=1e-6)
    function melitz_gradient_delta_direct_aq_experimental!(g::AbstractVector{Float64}, theta::AbstractVector{Float64},
                                                            ctx, obj, x::AbstractVector{Float64})
        get(ctx, :outer_parameterization, :logf) == :logcutoff || throw(ArgumentError(
            "melitz_gradient_delta_direct_aq_experimental!: requires ctx.outer_parameterization == " *
            ":logcutoff (got $(get(ctx, :outer_parameterization, :logf)))"))
        D = ctx.D
        nA = D^2 - 1
        nq = length(theta) - 1 - nA
        length(g) == length(theta) || throw(ArgumentError(
            "melitz_gradient_delta_direct_aq_experimental!: g must have length(theta)=$(length(theta)), got $(length(g))"))

        # 1. welfare (gamma) coordinate -- cheap fixed-dual central difference, SAME
        #    construction every existing FD direct backend uses for this coordinate.
        theta_p = copy(theta); theta_p[1] += gamma_h
        theta_m = copy(theta); theta_m[1] -= gamma_h
        melitz_update_operator_at_theta!(obj.op, theta_p, ctx); Dp = -obj(x)
        melitz_update_operator_at_theta!(obj.op, theta_m, ctx); Dm = -obj(x)
        melitz_update_operator_at_theta!(obj.op, theta, ctx)
        g[1] = (Dp - Dm) / (2 * gamma_h)

        # 2. exact A-block gradient (fixed q, envelope theorem, zero FD probes).
        A0, f0, gpj0, fjj0 = melitz_expand_theta(theta, ctx)
        state0 = MelitzExpandedState(D)
        state0.A .= A0; state0.f .= f0; state0.gamma_prime_j = gpj0; state0.f_jj = fjj0
        ws_a = MelitzExactAGradientWorkspace(obj.op)
        dDelta_da_full = zeros(D, D)
        melitz_exact_a_gradient_full!(dDelta_da_full, obj, x, state0, ctx, ws_a)
        g[2:1+nA] .= melitz_exact_a_gradient_free(dDelta_da_full, ctx)

        # 3. q-block gradient via the CONFIGURED bandwidth policy, one coordinate probe at a
        #    time -- the SAME `melitz_q_coordinate_probe` diagnostics call (Rule 10).
        for m in 1:nq
            r = melitz_q_coordinate_probe(theta, m, q_policy, obj, ctx; x0=x, mode=:fixed_dual)
            g[1+nA+m] = r.secant
        end
        return g
    end
    return melitz_gradient_delta_direct_aq_experimental!
end
