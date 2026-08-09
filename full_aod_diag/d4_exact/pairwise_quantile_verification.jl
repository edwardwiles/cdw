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
# instruction: (1) all FIVE marginal-bin and all 5x5 joint-cell probabilities (not just the 4x4/4
# enforced subset), (2) cumulative factorization residuals |P(z_o<q_r,z_p<q_s)-p_r*p_s| for every
# r,s=1..4, computed from those probabilities via the telescoping map proved in
# docs/PAIRWISE_QUANTILE_INDEPENDENCE_MATH_NOTE_2026-08-09.md Section 4 -- no second dense recompute.
#
# `lambda` flattening convention (documented once here, load-bearing for any caller assembling the
# flat outer-coordinate/dual vector): lambda_M (D x 4) flattens via plain `vec` (a-major, o-minor,
# i.e. flat[(a-1)*D+o]); lambda_P (4x4xnpair) flattens via plain `vec` (a-fastest, then b, then
# pidx, i.e. flat[(pidx-1)*16+(b-1)*4+a]) -- `reshape` on the corresponding view recovers each
# exactly, since Julia's `reshape`/`vec` share one column-major convention.
#
# Requires pairwise_quantile_bin_context.jl, pairwise_quantile_operator.jl (forward!/transpose!,
# build_pairwise_quantile_tables_threaded!) to already be included. `economic_forward!`/
# `economic_transpose!`/`cf`/`econ_ws` are the SAME economic-block primitives every other family's
# verifier already calls (`operator_verification.jl`'s own imports) -- passed through here, not
# redefined.
# ================================================================================================

n_mean_flat(D::Int) = 4 * D
n_pair_flat(npair::Int) = 16 * npair

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
        cf, op::PairwiseQuantileOperator, state::PairwiseQuantileBinState, W::Int,
        economic_forward!::Function, economic_transpose!::Function, econ_ws,
        Psi!::Function, dPsi!::Function, ncore1::Int)
    D = op.D; npair = op.npair
    nM = n_mean_flat(D); nP = n_pair_flat(npair)
    length(lambda) == ncore1 + nM + nP ||
        error("verify_inner_solution_operator_pairwisequantile!: length(lambda)=$(length(lambda)) != ncore1+nM+nP=$(ncore1+nM+nP)")

    λ_E = @view lambda[1:ncore1]
    # Same o-major/a-major reshape convention fix as pairwise_quantile_production.jl's dual_index! --
    # marginal_row(o,a)=(o-1)*4+a is O-MAJOR; reshape(v,D,4) is column-major (A-MAJOR). Must match
    # the ACTUAL KNITRO solution vector's layout (dual_index!'s own convention) for this
    # independent recompute to read the right lambda values.
    λ_M = reshape(@view(lambda[ncore1+1:ncore1+nM]), 4, D)'
    λ_P = reshape(@view(lambda[ncore1+nM+1:ncore1+nM+nP]), 4, 4, npair)

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

    g_M = zeros(D, 4); g_P = zeros(4, 4, npair)
    tls = build_pairwise_quantile_thread_scratch(D, npair)
    scratch = PairwiseQuantileTransposeScratch(D, npair)
    pairwise_quantile_transpose!(g_M, g_P, dPsi_r, op, state, tls, scratch)

    g_lambda = vcat(g_E, vec(g_M), vec(g_P))
    kkt_resid_E = isempty(g_E) ? 0.0 : maximum(abs, g_E)
    kkt_resid_marginalbin = maximum(abs, g_M)
    kkt_resid_pairindep = maximum(abs, g_P)

    # ---- full probability report (task: "even though only the nonredundant subset is enforced") ----
    ones_w = ones(W)
    Mcount = zeros(D, 5); Pcount = zeros(5, 5, npair)
    build_pairwise_quantile_tables_threaded!(Mcount, Pcount, tls, op, state, ones_w)
    marginal_prob = Mcount ./ W          # D x 5, ALL five bins
    joint_prob = Pcount ./ W             # 5 x 5 x npair, ALL 25 cells per pair

    # ---- cumulative factorization residuals, telescoping map (math note Section 4), O(npair*16) ----
    p = (0.2, 0.4, 0.6, 0.8)
    cum_resid = zeros(4, 4, npair)
    @inbounds for pidx in 1:npair
        for s in 1:4, rr in 1:4
            Frs = 0.0
            for b in 1:s, a in 1:rr
                Frs += joint_prob[a, b, pidx]
            end
            cum_resid[rr, s, pidx] = abs(Frs - p[rr] * p[s])
        end
    end
    marginal_cum_resid = zeros(4, D)
    @inbounds for o in 1:D, rr in 1:4
        marginal_cum_resid[rr, o] = abs(sum(@view marginal_prob[o, 1:rr]) - p[rr])
    end

    record_operator_verification!()
    return (r = r, f = f, g_lambda = g_lambda, kkt_resid = maximum(abs, g_lambda),
            kkt_resid_E = kkt_resid_E, kkt_resid_marginalbin = kkt_resid_marginalbin,
            kkt_resid_pairindep = kkt_resid_pairindep,
            marginal_prob = marginal_prob, joint_prob = joint_prob,
            cumulative_residual = cum_resid, marginal_cumulative_residual = marginal_cum_resid,
            max_cumulative_residual = isempty(cum_resid) ? 0.0 : maximum(cum_resid),
            max_marginal_cumulative_residual = isempty(marginal_cum_resid) ? 0.0 : maximum(marginal_cum_resid))
end

"""
    assert_pairwise_quantile_d20_counts(D::Int)

Task's explicit final-count report (Section 12 of the plan), asserted (not just printed). Errors
loudly if `D != 20` is passed where D=20-specific numbers are expected by a caller.
"""
function assert_pairwise_quantile_d20_counts(D::Int)
    D == 20 || error("assert_pairwise_quantile_d20_counts: expected D=20, got D=$D")
    npair = div(D * (D - 1), 2)
    nmarg = n_marginal_rows(D)
    npairrows = n_pair_rows(D)
    ntot = n_total_rows(D)
    npair == 190 || error("unordered_pairs assertion failed: got $npair, expected 190")
    nmarg == 80 || error("marginal_rows assertion failed: got $nmarg, expected 80")
    npairrows == 3040 || error("pair_rows assertion failed: got $npairrows, expected 3040")
    ntot == 3120 || error("total_rows assertion failed: got $ntot, expected 3120")
    return (n_bins = 5, n_cutoffs = 4, outer_cutoff_params = 4 * D, unordered_pairs = npair,
            marginal_rows = nmarg, pair_rows = npairrows, total_rows = ntot, dense_G_production = false)
end
