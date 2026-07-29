# q-bandwidth convergence campaign session (2026-07-29), Phase 2: the exact envelope-theorem
# DeltaStar derivative w.r.t. the q (baseline log-cutoff) block, AT FIXED ACTIVE SET -- the
# "smooth" component of the q derivative that survives even when no draw-level participation
# switch occurs, mirroring `exact_a_gradient.jl`'s own derivation exactly but for the q block.
#
# DERIVATION SUMMARY (see docs/melitz_aq_q_bandwidth_convergence_2026-07-29.md for the full
# writeup):
#
# 1. `melitz_log_f_from_q` (log_cutoff_param.jl): `log(f_od) = (sigma-1)*(q_od + a_od -
#    log(markup) - log(w_o) - log(tau_od)) + log(expenditure_d) - log(sigma) - log(w_o)`.
#    At FIXED a_od, `d(log f_od)/d(q_od) = (sigma-1)` exactly -- the SAME multiplicative
#    scale `exact_a_gradient.jl` uses for `d(log f_od)/d(a_od)`, by construction (q_od and
#    a_od enter this formula symmetrically, both scaled by `(sigma-1)` with a `+` sign).
#
# 2. Trade-share block (`R_od,s = coef_od*z_power[s,o]*active_od(s)`, `coef_od =
#    C_od/expenditure_d`, `C_od = expenditure_d*(markup*w_o*tau_od/A_od)^(1-sigma)`,
#    `firm_quantities.jl`): `coef_od` depends on `(w_o,tau_od,A_od,sigma,expenditure_d)`
#    ONLY -- q_od does not appear anywhere in `melitz_C`. So for any draw `s` that stays on
#    the SAME side of the active/inactive boundary (fixed active set), `d(R_od,s)/d(q_od) =
#    0` exactly. UNLIKE the A block, the q block's smooth trade-share contribution is
#    IDENTICALLY ZERO -- q's only trade-share effect is through switching (the crossing/
#    bandwidth machinery in `q_bandwidth_policy.jl`), never through a smooth intensive
#    channel.
#
# 3. Focal-link block: `melitz_firm`'s raw profit is `profit_od(z) = (C_od/price_power_d)*
#    z^(sigma-1)/sigma - w_o*f_od` (`firm_quantities.jl`). At fixed q's OWN active set,
#    `d(profit_od(z))/d(q_od) [fixed a_od] = -w_o*d(f_od)/d(q_od) = -w_o*(sigma-1)*f_od`
#    exactly -- ONLY the `-w_o*f_od` term contributes (the `C_od*z^(sigma-1)/sigma` term has
#    zero q-derivative, per point 2), unlike the A case where BOTH terms of `profit_od(z)`
#    contribute the SAME `(sigma-1)` factor. This is the ENTIRE smooth q channel: it enters
#    only through the origin-`j` focal-link row (`ell`), exactly the same `tail1`/`link_coef`
#    machinery `exact_a_gradient.jl` already builds for its own Step 2, with the `Rt_jd`/
#    `C_od`-sourced piece dropped (zero for q) and only the `f_od`/`tail1_d` piece retained.
#
# 4. Autarky sub-term (`derive_fjj_from_autarky_cutoff`, `equilibrium.jl`): `f_jj` is a
#    function of `(gamma_prime_j, A[j,j])` alone -- no `q` dependence anywhere (`q[j,j]` is
#    ITSELF derived from `g` alone, `derive_qjj_from_autarky_cutoff`, independent of every
#    free q coordinate). So the autarky term's smooth q-derivative is IDENTICALLY ZERO for
#    every free q coordinate -- no Step-3 analogue exists for q at all.
#
# 5. Chain rule to free q coordinates: EXACTLY the same pivot-adjoint structure as the A
#    block (`melitz_exact_a_gradient_free`), but over the q/f-pivot's own domain
#    (`ctx.f_free_lin`, length `D^2-1`, excluding `(j,j)`) rather than the full `D^2` A
#    domain -- `melitz_cached_f_pivot_parts(ctx)` gives the SAME `(c_free, pivot, other)`
#    the q-pivot uses (provably the same physical cell as the f-pivot, `log_cutoff_param.jl`'s
#    own docstring), with indices INTO the `f_free_lin`-ordered sub-vector, not the full D^2
#    linear index space.
#
# CONCLUSION (governing prompt Phase 2, Option A applies): the fixed-active-set q derivative
# is NONZERO (through the focal-link/fixed-cost channel alone) -- a genuine hybrid derivative
# (exact smooth + finite-bandwidth switching residual) is implemented, not a "prove it's
# zero" regression (Option B does not apply).

"""
    MelitzExactQSmoothGradientWorkspace(op)

Preallocated scratch for `melitz_exact_q_smooth_gradient_full!` -- zero allocation after
construction. Lighter than `MelitzExactAGradientWorkspace`: the trade-share block has zero
smooth q-sensitivity (derivation point 2 above), so no `mul_Gt!`/`g_dpsi` pass is needed at
all, only the origin-`j` active-tail machinery `exact_a_gradient.jl`'s own Step 2 already
uses.
"""
struct MelitzExactQSmoothGradientWorkspace
    u::Vector{Float64}
    dpsi::Vector{Float64}
    binsum1::Vector{Float64}
    tail1::Vector{Float64}
end
function MelitzExactQSmoothGradientWorkspace(op::MelitzMomentOperator)
    return MelitzExactQSmoothGradientWorkspace(zeros(op.W), zeros(op.W), zeros(op.D + 1), zeros(op.D + 1))
end

"""
    melitz_exact_q_smooth_gradient_full!(dDelta_dq::AbstractMatrix{Float64}, obj::MelitzCCBundle,
        x::AbstractVector{Float64}, state::MelitzExpandedState, ctx,
        ws::MelitzExactQSmoothGradientWorkspace) -> dDelta_dq

Fills `dDelta_dq[o,d] = d(DeltaStar)/d(q_od)` (fixed a, FIXED ACTIVE SET -- i.e. the smooth
component only, excluding any draw-level participation switch) for every cell in the q/f
domain (`(j,j)` is set to `0.0`, not part of that domain -- see file header point 4), in
`O(W+D)` (no `O(W*D)` pass at all, unlike the A gradient, since the trade-share block
contributes nothing here). Zero finite-difference probes, zero re-solves.

`obj.op`/`state` must reflect the SAME already-converged theta as `x` (identical contract to
`melitz_exact_a_gradient_full!`).
"""
function melitz_exact_q_smooth_gradient_full!(dDelta_dq::AbstractMatrix{Float64}, obj::MelitzCCBundle,
                                               x::AbstractVector{Float64}, state::MelitzExpandedState, ctx,
                                               ws::MelitzExactQSmoothGradientWorkspace)
    op = obj.op
    D = op.D
    W = op.W
    M = obj.M
    sigma = ctx.sigma
    j = ctx.target_country
    size(dDelta_dq) == (D, D) || throw(ArgumentError("melitz_exact_q_smooth_gradient_full!: dDelta_dq must be D x D"))
    fill!(dDelta_dq, 0.0)

    mu = @view x[2:end]
    mu_link = mu[op.layout.focal_link_index]
    mu_link == 0.0 && return dDelta_dq   # no focal link active at this dual -> smooth q term is exactly zero

    zeta = x[1]
    u = ws.u
    dpsi = ws.dpsi
    mul_G!(u, op, zeta, mu)
    melitz_cc_dPsi!(dpsi, u)

    binsum1 = ws.binsum1
    tail1 = ws.tail1
    @inbounds for k in 1:D+1
        binsum1[k] = 0.0
    end
    bin_j = @view op.bin[:, j]
    @inbounds for s in 1:W
        binsum1[bin_j[s]+1] += dpsi[s]
    end
    tail1[D+1] = binsum1[D+1]
    @inbounds for k in D:-1:1
        tail1[k] = tail1[k+1] + binsum1[k]
    end
    rank_j = @view op.rank[:, j]

    f = state.f
    link_coef = mu_link * (sigma - 1.0) / M
    @inbounds for d in 1:D
        d == j && continue   # (j,j) not part of the q free domain (q_jj derived from g alone)
        tail1_d = tail1[rank_j[d]+1]
        # smooth q contribution: d(profit_current_jd(z))/d(q_jd) [fixed a, fixed active set]
        #   = -w_j * d(f[j,d])/d(q_jd) = -w_j*(sigma-1)*f[j,d]  (point 1/3 above); the
        # C_od*z^(sigma-1) piece of profit_od(z) has zero q-derivative (point 2), so -- unlike
        # exact_a_gradient.jl's own link_term -- there is no Rt_jd/coef-sourced piece here.
        dDelta_dq[j, d] = -link_coef * f[j, d] * tail1_d
    end
    return dDelta_dq
end

"""
    melitz_exact_q_smooth_gradient_free(dDelta_dq_full::AbstractMatrix{Float64}, ctx) -> Vector{Float64}

Chain-rule map from the full-cell smooth `d(DeltaStar)/d(q_od)` gradient to the free q-block
gradient (length `D^2-2`, matching `q_free_free`'s own ordering), via the q/f-gravity pivot's
adjoint (`melitz_cached_f_pivot_parts(ctx)` -- provably the SAME physical pivot cell the
q-pivot uses, `log_cutoff_param.jl`). Indices are into the `ctx.f_free_lin`-ordered
`D^2-1`-length sub-vector, NOT the full `D^2` linear index space (unlike the A-block adjoint,
whose domain is the full `D^2` cells) -- `(j,j)` is excluded from `f_free_lin` entirely, so
`dDelta_dq_full[j,j]` (always `0.0`, filled above) never enters this reduction.
"""
function melitz_exact_q_smooth_gradient_free(dDelta_dq_full::AbstractMatrix{Float64}, ctx)
    D = ctx.D
    c_free, pivot_idx, other = melitz_cached_f_pivot_parts(ctx)
    f_free_lin = ctx.f_free_lin
    g_domain = Vector{Float64}(undef, length(f_free_lin))
    @inbounds for (k, lin) in enumerate(f_free_lin)
        o, d = lin2od(lin, D)
        g_domain[k] = dDelta_dq_full[o, d]
    end
    g_pivot = g_domain[pivot_idx]
    grad_free = Vector{Float64}(undef, length(other))
    @inbounds for (k, i) in enumerate(other)
        grad_free[k] = g_domain[i] - (c_free[i] / c_free[pivot_idx]) * g_pivot
    end
    return grad_free
end

"""
    melitz_exact_q_smooth_gradient(obj::MelitzCCBundle, x::AbstractVector{Float64},
        state::MelitzExpandedState, ctx) -> (grad_free, dDelta_dq_full)

Convenience, allocating wrapper (diagnostic/test use, not a hot-path function) -- mirrors
`melitz_exact_a_gradient`'s own signature exactly.
"""
function melitz_exact_q_smooth_gradient(obj::MelitzCCBundle, x::AbstractVector{Float64},
                                         state::MelitzExpandedState, ctx)
    ws = MelitzExactQSmoothGradientWorkspace(obj.op)
    dDelta_dq_full = zeros(ctx.D, ctx.D)
    melitz_exact_q_smooth_gradient_full!(dDelta_dq_full, obj, x, state, ctx, ws)
    return melitz_exact_q_smooth_gradient_free(dDelta_dq_full, ctx), dDelta_dq_full
end
