# A_q separation and gradient diagnostics session (2026-07-29), Phase 3-4: the exact
# envelope-theorem DeltaStar gradient w.r.t. the technology (A) block, at FIXED q
# (participation). Eliminates finite differences for the entire A block: no theta
# perturbation, no displaced economic state, no re-solved inner problem, no re-equilibration
# -- everything below is computed from ONE already-converged optimal dual `x=(zeta,mu)` at
# the CURRENT operator state, via the existing matrix-free `mul_G!`/`mul_Gt!` kernels.
#
# DERIVATION SUMMARY (full derivation in docs/melitz_A_q_separation_and_gradient_diagnostics_*.md):
#
# 1. Envelope theorem. The inner KNITRO objective functor (cc_bundle.jl) computes
#    `f(x;theta) = sum(Psi(u))/M + zeta`, `u = -zeta - G(theta)*mu`, minimized over `x`;
#    `DeltaStar(theta) = -f(x*(theta);theta)`. At the verified optimum `x*`, the envelope
#    theorem gives `d(DeltaStar)/d(theta) = -[partial f/partial theta](x*, theta)` -- the
#    TOTAL derivative equals the PARTIAL derivative at fixed x*, because x* is a stationary
#    point of f in x.
#
# 2. Trade-block A-dependence. `G_trade[s,(o,d)] = R_od,s - lambda_od`, `R_od,s =
#    coef_od*z_power[s,o]*active_od(s)`, `coef_od = C_od/expenditure_d`,
#    `C_od = expenditure_d*(markup*w_o*tau_od/A_od)^(1-sigma) ∝ A_od^(sigma-1)`
#    (`firm_quantities.jl`). At FIXED q (fixed active_od(s) -- the strict A/q block
#    separation this session's Phase 1-2 established: perturbing A_free never moves any q
#    value, hence never flips any participation indicator), `d(R_od,s)/d(a_od) =
#    (sigma-1)*R_od,s` exactly (`a_od = log(A_od)`; `lambda_od` is a pure data constant, no
#    A dependence). Summing `dPsi(u_s)*R_od,s` over `s` for EVERY (o,d) cell simultaneously
#    is exactly `mul_Gt!(_, op, dPsi(u))` plus the known `lambda*sum` correction (the same
#    algebra `moment_operator.jl`'s own `Rt_S` already uses for the Hessian) -- ONE
#    `O(W*D)` pass gives the complete full-cell trade-block gradient.
#
# 3. Focal-link-block A-dependence. `ell[s]` (the focal baseline-vs-autarky free-entry link
#    column, `moment_operator.jl`'s own construction) is built from raw per-draw firm
#    profit, `profit_od(z) = (C_od/price_power_d)*z^(sigma-1)/sigma - w_o*f_od` for z active
#    (`firm_quantities.jl`'s own `melitz_firm`) -- ONLY the focal-origin row `A[j,:]`
#    (through `C_od`) touches it, matching `MelitzCompactColumns.touches_link`. AT FIXED q,
#    `f_od` is reconstructed from `(a_od, q_od)` via `melitz_log_f_from_q`, giving
#    `d(f_od)/d(a_od) = (sigma-1)*f_od` -- so the TOTAL derivative of `profit_od(z)` w.r.t.
#    `a_od` at fixed q is `(sigma-1)*profit_od(z)` exactly, the SAME clean multiplicative
#    envelope structure as the trade block (verified algebraically both directly, via the
#    `C_od`+`f_od` chain rule, and via the zero-profit-cutoff-identity substitution -- both
#    give an identical result). The autarky counterfactual sub-term uses the SAME `A[j,j]`
#    cell with a DIFFERENT normalization: `derive_fjj_from_autarky_cutoff`
#    (`equilibrium.jl`) fixes the autarky cutoff at EXACTLY 1 for every `(A_jj,
#    gamma_prime_j)` (a hardwired normalization, not itself a free/varying q coordinate) --
#    this makes the autarky term's A-derivative equally clean, `w_prime*f_jj*(y_s-1)` for
#    active `s`, no genuinely different formula needed (the task's own explicitly-allowed
#    "one focal/autarky direction may need a different formula" case does NOT arise here --
#    a positive, reportable finding). Both current-equilibrium and autarky per-draw sums
#    reduce to already-known quantities (`Rt`, `f`, `w`) plus two cheap additional `O(W+D)`/
#    `O(W)` passes (an UNWEIGHTED origin-`j` tail sum, reusing `op.bin[:,j]`/`op.rank[:,j]`;
#    and a fixed `z_orig[:,j]>=1` masked sum) -- no new `O(W*D)` cost.
#
# 4. Chain rule to free A coordinates. `log(A_od)` is an EXACT LINEAR function of `A_free`
#    via the A-gravity pivot (`pivot_expand`, `g0=0` always) -- `affine_cutoff.jl`'s own
#    Section 3.1 `M_A` construction. The free-coordinate gradient is therefore the standard
#    pivot adjoint: `d/d(a_free[k]) = d/d(a_full[other[k]]) -
#    (c[other[k]]/c[pivot])*d/d(a_full[pivot])`.
#
# Nothing here perturbs `theta`, calls `melitz_update_operator_at_theta!` a second time, or
# re-solves the inner problem -- the caller is responsible for having already reached a
# verified optimal dual `x` at the theta being differentiated (exactly the state
# `solve_melitz_delta!`/`melitz_recover_lfd_from_solution` already leave the bundle in).

"""
    MelitzExactAGradientWorkspace(D)

Preallocated scratch for `melitz_exact_a_gradient_full!` -- zero allocation after
construction. `g_dpsi`/`u`/`dpsi` are `Vector{Float64}` sized `num_moments`/`W`/`W`;
`tail1`/`binsum1` are length `D+1`.
"""
struct MelitzExactAGradientWorkspace
    u::Vector{Float64}
    dpsi::Vector{Float64}
    g_dpsi::Vector{Float64}
    binsum1::Vector{Float64}
    tail1::Vector{Float64}
end
function MelitzExactAGradientWorkspace(op::MelitzMomentOperator)
    K = op.layout.num_moments
    return MelitzExactAGradientWorkspace(zeros(op.W), zeros(op.W), zeros(K), zeros(op.D + 1), zeros(op.D + 1))
end

"""
    melitz_exact_a_gradient_full!(dDelta_da::AbstractMatrix{Float64}, obj::MelitzCCBundle,
        x::AbstractVector{Float64}, state::MelitzExpandedState, ctx,
        ws::MelitzExactAGradientWorkspace) -> dDelta_da

Fills `dDelta_da[o,d] = d(DeltaStar)/d(log A_od)` (fixed q, envelope theorem, exact) for
EVERY full cell, in `O(W*D)` total (one `mul_Gt!` call plus two `O(W)`/`O(W+D)` extra
passes) -- zero finite-difference probes, zero re-solves, zero displaced states.

`obj.op` must already reflect the theta at which `x` is the verified optimal dual (the
caller's normal post-solve state -- this function does not call
`melitz_update_operator_at_theta!`). `state` must hold the SAME theta's expanded `(A,f,
gamma_prime_j,f_jj)` (e.g. from the `melitz_expand_theta!` call
`melitz_update_operator_at_theta!` itself performed, or an equivalent fresh expansion at the
identical theta -- never at a displaced point).
"""
function melitz_exact_a_gradient_full!(dDelta_da::AbstractMatrix{Float64}, obj::MelitzCCBundle,
                                        x::AbstractVector{Float64}, state::MelitzExpandedState, ctx,
                                        ws::MelitzExactAGradientWorkspace)
    op = obj.op
    D = op.D
    W = op.W
    M = obj.M
    sigma = ctx.sigma
    j = ctx.target_country
    size(dDelta_da) == (D, D) || throw(ArgumentError("melitz_exact_a_gradient_full!: dDelta_da must be D x D"))

    zeta = x[1]
    mu = @view x[2:end]
    u = ws.u
    dpsi = ws.dpsi
    mul_G!(u, op, zeta, mu)
    melitz_cc_dPsi!(dpsi, u)

    g_dpsi = ws.g_dpsi
    mul_Gt!(g_dpsi, op, dpsi)
    sum_dpsi = 0.0
    @inbounds for s in 1:W
        sum_dpsi += dpsi[s]
    end

    trade_index = op.layout.trade_index
    lambda = op.lambda
    coefsm1_over_M = (sigma - 1.0) / M

    # Step 1: pure trade-block contribution, every cell (T1).
    @inbounds for o in 1:D, d in 1:D
        Rt_od = g_dpsi[trade_index[o, d]] + lambda[o, d] * sum_dpsi
        mu_od = mu[trade_index[o, d]]
        dDelta_da[o, d] = mu_od * coefsm1_over_M * Rt_od
    end

    mu_link = mu[op.layout.focal_link_index]
    if mu_link != 0.0
        # Step 2: unweighted active-tail sum of dPsi(u), origin j only (reuses op.bin[:,j]/
        # op.rank[:,j] -- the SAME active-set indicators eq.cutoff[j,:] defines, i.e. exactly
        # the participation condition profit_current_j*(z) uses).
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

        expenditure = ctx.expenditure
        w_j = ctx.w[j]
        f = state.f
        link_coef = mu_link * coefsm1_over_M
        @inbounds for d in 1:D
            Rt_jd = g_dpsi[trade_index[j, d]] + lambda[j, d] * sum_dpsi
            tail1_d = tail1[rank_j[d]+1]
            # sum_s dPsi(u_s)*profit_current_jd(s) [active] = (expenditure_d/sigma)*Rt_jd -
            # w_j*f[j,d]*tail1_d  (raw melitz_firm profit formula, no substitution needed).
            link_term = (expenditure[d] / sigma) * Rt_jd - w_j * f[j, d] * tail1_d
            dDelta_da[j, d] += link_coef / w_j * link_term
        end

        # Step 3: autarky sub-term, cell (j,j) only. Autarky cutoff is EXACTLY 1 by
        # construction (derive_fjj_from_autarky_cutoff) -- a fixed mask on the RAW z_orig[:,j]
        # column, independent of theta.
        z_orig_j = @view op.sorted_ctx.z_original[:, j]
        z_power_j = @view op.sorted_ctx.z_power_original[:, j]
        Sya = 0.0
        S1a = 0.0
        @inbounds for s in 1:W
            if z_orig_j[s] >= 1.0
                Sya += dpsi[s] * z_power_j[s]
                S1a += dpsi[s]
            end
        end
        w_prime = ctx.w_prime
        f_jj = state.f_jj
        # sum_s dPsi(u_s)*profit_autarky(s) [active] = w_prime*f_jj*(Sya - S1a)
        # (zero-profit-at-cutoff-1 identity substitution -- see file header Section 3).
        autarky_term = w_prime * f_jj * (Sya - S1a)
        dDelta_da[j, j] -= link_coef / w_prime * autarky_term
    end

    return dDelta_da
end

"""
    melitz_exact_a_gradient_free(dDelta_da_full::AbstractMatrix{Float64}, ctx) -> Vector{Float64}

Chain-rule map from the full-cell `d(DeltaStar)/d(log A_od)` gradient (length `D^2`, ANY
D x D matrix -- typically `melitz_exact_a_gradient_full!`'s own output) to the free A-block
gradient (length `D^2-1`, matching `ctx.A_pivot.other`'s own ordering), via the A-gravity
pivot's exact linear reconstruction adjoint (`affine_cutoff.jl`'s own `M_A`, Section 3.1's
own step 1-2, applied as an adjoint here rather than rebuilt as a dense matrix). Does NOT
apply the technology-coordinate rescale (`ctx.technology_coordinate`) -- returns the gradient
w.r.t. PLAIN `log(A_od)` free coordinates; a caller under a non-`:logA` technology coordinate
must divide by `melitz_technology_coordinate_scale(ctx.technology_coordinate, ctx)`
(`melitz_reduce_theta`'s own inverse-scale convention, `technology_coordinate.jl`), exactly
mirroring how `melitz_expand_theta`'s own un-scale is applied before, not inside, the
A-pivot reconstruction.
"""
function melitz_exact_a_gradient_free(dDelta_da_full::AbstractMatrix{Float64}, ctx)
    A_pivot = ctx.A_pivot
    c = A_pivot.c
    pivot = A_pivot.pivot
    other = A_pivot.other
    g_full = vec(dDelta_da_full)   # column-major == od2lin(o,d,D) = o + (d-1)*D, confirmed
    g_pivot = g_full[pivot]
    grad_free = Vector{Float64}(undef, length(other))
    @inbounds for (k, i) in enumerate(other)
        grad_free[k] = g_full[i] - (c[i] / c[pivot]) * g_pivot
    end
    return grad_free
end

"""
    melitz_exact_a_gradient(obj::MelitzCCBundle, x::AbstractVector{Float64},
        state::MelitzExpandedState, ctx) -> Vector{Float64}

Convenience, allocating wrapper (diagnostic/test use, not a hot-path function): builds a
fresh workspace, computes the full-cell gradient, and returns the free-A-block gradient
(plain `log(A_od)` units -- see `melitz_exact_a_gradient_free`'s own docstring for the
technology-coordinate caveat).
"""
function melitz_exact_a_gradient(obj::MelitzCCBundle, x::AbstractVector{Float64},
                                  state::MelitzExpandedState, ctx)
    ws = MelitzExactAGradientWorkspace(obj.op)
    dDelta_da_full = zeros(ctx.D, ctx.D)
    melitz_exact_a_gradient_full!(dDelta_da_full, obj, x, state, ctx, ws)
    return melitz_exact_a_gradient_free(dDelta_da_full, ctx), dDelta_da_full
end
