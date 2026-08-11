# ================================================================================================
# Verifier extension for the pairwise-quantile-independence restriction (draft eq. 32), Section 8
# of the implementation plan.
#
# `verify_inner_solution_operator_pairwisequantile!` follows `operator_verification.jl::
# verify_inner_solution_operator_originzc!`'s template exactly (independent forward!/transpose!
# recompute from FRESH scratch, per-block KKT residuals) -- its output NamedTuple is meant to feed
# the SAME `classify_inner_result`/`verification_rejection_reasons` (`oracle.jl`) every other family
# already uses unchanged (both already read fields via `get(result,:field,default)`, so no edit is
# needed there). This file adds two things beyond the KKT-residual template, per the task's explicit
# instruction: (1) ALL `L` marginal-bin and all `LxL` joint-cell probabilities (not just the
# enforced `(L-1)`/`(L-1)^2` subset), (2) cumulative factorization residuals
# |P(z_o<q_r,z_p<q_s) - P_{o,r}*P_{p,s}| for every r,s = 1..L-1, computed from those probabilities
# via the telescoping map proved in
# docs/PAIRWISE_QUANTILE_INDEPENDENCE_MATH_NOTE_2026-08-09.md Section 4 -- no second dense recompute.
#
# VERSION B (free-mass reparameterization, 2026-08-10): the cumulative targets are the FREE masses'
# own cumulative sums `P_{o,r} = sum_{a<=r} mu_{o,a}` (`state.Pcum`) rather than the constants
# `r/L`, and the probability tables are reported UNDER THE LFD rather than under the unweighted
# draws -- see the block comment at the report itself for why the version-A form would actively
# mislead here.
#
# `lambda` flattening convention (documented once here, load-bearing for any caller assembling the
# flat outer-coordinate/dual vector): lambda_M (D x (L-1)) flattens via plain `vec` (a-major,
# o-minor, i.e. flat[(a-1)*D+o]); lambda_P ((L-1)x(L-1)xnpair) flattens via plain `vec` (a-fastest,
# then b, then pidx) -- `reshape` on the corresponding view recovers each exactly, since Julia's
# `reshape`/`vec` share one column-major convention.
#
# Requires pairwise_quantile_bin_context.jl, pairwise_quantile_operator.jl (forward!/transpose!,
# build_pairwise_quantile_tables_threaded!) to already be included. `economic_forward!`/
# `economic_transpose!`/`cf`/`econ_ws` are the SAME economic-block primitives every other family's
# verifier already calls (`operator_verification.jl`'s own imports) -- passed through here, not
# redefined.
# ================================================================================================

n_mean_flat(D::Int, L::Int) = (L - 1) * D
n_pair_flat(npair::Int, L::Int) = (L - 1)^2 * npair

"""
    verify_inner_solution_operator_pairwisequantile!(zeta, lambda, cf, op, state, W,
        economic_forward!, economic_transpose!, econ_ws, Psi!, dPsi!, ncore1) -> NamedTuple

Independently recomputes `r = -zeta*1 - E*lambda_E - G*lambda_restriction` via `economic_forward!`
(unchanged economic primitive) + `pairwise_quantile_forward!` (fresh call, not reading any FG-
callback-cached state), the objective `f`, and the full dual gradient via `economic_transpose!` +
`pairwise_quantile_transpose!` -- exactly `verify_inner_solution_operator_originzc!`'s own
structure, generalized to this restriction's block names. Returns block-level KKT residuals
(`kkt_resid_marginalbin`, `kkt_resid_pairindep`, plus the aggregate `kkt_resid`) PLUS the full
probability-table report and cumulative-residual report the task explicitly asks for.
"""
function verify_inner_solution_operator_pairwisequantile!(zeta::Float64, lambda::AbstractVector{Float64},
        cf, op::PairwiseQuantileOperator, state::PairwiseQuantileMassState, W::Int,
        economic_forward!::Function, economic_transpose!::Function, econ_ws,
        Psi!::Function, dPsi!::Function, ncore1::Int)
    D = op.D; npair = op.npair; L = op.L; nc = L - 1
    nM = n_mean_flat(D, L); nP = n_pair_flat(npair, L)
    length(lambda) == ncore1 + nM + nP ||
        error("verify_inner_solution_operator_pairwisequantile!: length(lambda)=$(length(lambda)) != ncore1+nM+nP=$(ncore1+nM+nP)")

    λ_E = @view lambda[1:ncore1]
    # Same o-major/a-major reshape convention fix as pairwise_quantile_production.jl's dual_index! --
    # marginal_row(o,a,L)=(o-1)*(L-1)+a is O-MAJOR; reshape(v,D,nc) is column-major (A-MAJOR). Must
    # match the ACTUAL KNITRO solution vector's layout (dual_index!'s own convention) for this
    # independent recompute to read the right lambda values.
    λ_M = reshape(@view(lambda[ncore1+1:ncore1+nM]), nc, D)'
    λ_P = reshape(@view(lambda[ncore1+nM+1:ncore1+nM+nP]), nc, nc, npair)

    r = fill(-zeta, W)
    econ_buf = zeros(W)
    economic_forward!(econ_buf, λ_E, cf, econ_ws)
    r .-= econ_buf
    pairwise_quantile_forward!(r, λ_M, λ_P, op, state)

    Psi_r = similar(r); Psi!(Psi_r, r)
    f = sum(Psi_r) / W + zeta

    dPsi_r = similar(r); dPsi!(dPsi_r, r)
    g_E = zeros(ncore1)
    economic_transpose!(g_E, dPsi_r, cf, econ_ws)
    g_E .*= -(1.0 / W)

    g_M = zeros(D, nc); g_P = zeros(nc, nc, npair)
    tls = build_pairwise_quantile_thread_scratch(D, npair, L)
    scratch = PairwiseQuantileTransposeScratch(D, npair, L)
    pairwise_quantile_transpose!(g_M, g_P, dPsi_r, op, state, tls, scratch)

    g_lambda = vcat(g_E, vec(g_M), vec(g_P))
    kkt_resid_E = isempty(g_E) ? 0.0 : maximum(abs, g_E)
    kkt_resid_marginalbin = maximum(abs, g_M)
    kkt_resid_pairindep = maximum(abs, g_P)

    # ---- full probability report, UNDER THE LFD (task: "even though only the nonredundant subset
    # is enforced") ----
    #
    # VERSION B CORRECTION, and it is not cosmetic. Version A built this report from the UNWEIGHTED
    # draws (`weight = ones(W)`) and compared it to the constants `r/L`. Under version B the targets
    # are the free masses `mu`, so an unweighted report would show a large "residual" at any point
    # where `mu` differs from the draws' own bin frequencies -- i.e. it would flag the search doing
    # exactly what it is supposed to do. The quantity the restriction actually constrains is the
    # probability under the LEAST-FAVOURABLE measure `m_w = Psi'(r_w)`, and reporting that also makes
    # this report the human-readable form of the KKT residuals computed just above: `g_M[o,a] = 0`
    # is exactly `Mraw[o,a]/S_m = mu[o,a]`.
    #
    # No extra work: `pairwise_quantile_transpose!` above already built the full `1:L` m-weighted
    # tables into `scratch.Mraw`/`scratch.Praw` on its way to `g_M`/`g_P`, so this reads them rather
    # than making a second O(W*(D+npair)) pass (which version A did).
    S_m = sum(dPsi_r)
    S_m > 0 || error("verify_inner_solution_operator_pairwisequantile!: sum of LFD weights is $S_m <= 0")
    marginal_prob = scratch.Mraw ./ S_m   # D x L, ALL L bins, under the LFD
    joint_prob = scratch.Praw ./ S_m      # L x L x npair, ALL L^2 cells per pair, under the LFD

    # ---- cumulative factorization residuals, telescoping map (math note Section 4), O(npair*nc^2) ----
    # Targets are now the free masses' own cumulative sums `Pcum[o,r] = sum_{a<=r} mu[o,a]`
    # (state.Pcum), not `r/L`. The math note's Sections 2-3 are stated in exactly this cumulative
    # form (`P(z_o<q_r)=p_r`, `F(r,s)=p_r*p_s`) with `p_r` a constant; version B frees `p_r` and
    # changes nothing else about the proofs.
    Pcum = state.Pcum
    cum_resid = zeros(nc, nc, npair)
    @inbounds for pidx in 1:npair
        (o, p) = op.pairs[pidx]
        for s in 1:nc, rr in 1:nc
            Frs = 0.0
            for b in 1:s, a in 1:rr
                Frs += joint_prob[a, b, pidx]
            end
            cum_resid[rr, s, pidx] = abs(Frs - Pcum[o, rr] * Pcum[p, s])
        end
    end
    marginal_cum_resid = zeros(nc, D)
    @inbounds for o in 1:D, rr in 1:nc
        marginal_cum_resid[rr, o] = abs(sum(@view marginal_prob[o, 1:rr]) - Pcum[o, rr])
    end

    record_operator_verification!()
    return (r = r, f = f, g_lambda = g_lambda, kkt_resid = maximum(abs, g_lambda),
            kkt_resid_E = kkt_resid_E, kkt_resid_marginalbin = kkt_resid_marginalbin,
            kkt_resid_pairindep = kkt_resid_pairindep,
            mu = copy(state.mu), mu_last = copy(state.mu_last), Pcum = copy(state.Pcum),
            marginal_prob = marginal_prob, joint_prob = joint_prob,
            cumulative_residual = cum_resid, marginal_cumulative_residual = marginal_cum_resid,
            max_cumulative_residual = isempty(cum_resid) ? 0.0 : maximum(cum_resid),
            max_marginal_cumulative_residual = isempty(marginal_cum_resid) ? 0.0 : maximum(marginal_cum_resid))
end

"""
    assert_pairwise_quantile_d20_counts(D::Int, L::Int)

Task's explicit final-count report (Section 12 of the plan), asserted (not just printed). Errors
loudly if `D != 20` is passed where D=20-specific numbers are expected by a caller. The task's own
draft used `L=5` (quintiles) -- when `L==5` this ALSO asserts the exact originally-published D=20
numbers (190/80/3040/3120) as a regression sentinel; other `L` values return the `L`-generic counts
without a hardcoded-literal comparison (there is no independently-published reference to check
them against).
"""
function assert_pairwise_quantile_d20_counts(D::Int, L::Int)
    D == 20 || error("assert_pairwise_quantile_d20_counts: expected D=20, got D=$D")
    npair = div(D * (D - 1), 2)
    nmarg = n_marginal_rows(D, L)
    npairrows = n_pair_rows(D, L)
    ntot = n_total_rows(D, L)
    npair == 190 || error("unordered_pairs assertion failed: got $npair, expected 190")
    if L == 5
        nmarg == 80 || error("marginal_rows assertion failed: got $nmarg, expected 80")
        npairrows == 3040 || error("pair_rows assertion failed: got $npairrows, expected 3040")
        ntot == 3120 || error("total_rows assertion failed: got $ntot, expected 3120")
    end
    return (n_bins = L, n_free_bins = L - 1, outer_mass_params = (L - 1) * D, unordered_pairs = npair,
            marginal_rows = nmarg, pair_rows = npairrows, total_rows = ntot, dense_G_production = false)
end
