# ================================================================================================
# Verifier for the CM + pairwise-quantile family (family #7, 2026-08-12).
#
# Follows `pairwise_quantile_verification.jl` (which follows `operator_verification.jl::
# verify_inner_solution_operator_originzc!`) exactly: an INDEPENDENT recompute of `r`, `f` and the
# full dual gradient from FRESH scratch, reading no FG- or Hessian-callback-cached state. Its output
# NamedTuple feeds the SAME family-agnostic `verify_namedtuple_from_operator` ->
# `classify_inner_result`/`is_verified_success` (oracle.jl) every other family uses, so no new
# acceptance predicate is invented here (memory `feedback-lfd-ok-verification-gate-required`:
# `FiniteSolved && within_budget` is NOT a verification gate).
#
# THREE BLOCKS, not two. The standalone family's verifier recomputes `[E | restriction]`. This one
# recomputes `[E | level+pair | CM-grid]`, so it carries a THIRD block KKT residual
# (`kkt_resid_cm`). Getting the CM block into `r` is not optional bookkeeping: it is a term of the
# per-draw dual index, and a verifier that omitted it would report a wrong `Delta_dual` while every
# residual it does compute still looked fine.
#
# WHAT "INDEPENDENT" MEANS HERE, precisely. It means fresh buffers and a fresh call path, not a
# second implementation of the moment algebra -- exactly the standard the other four families' own
# verifiers are held to (they call the same `*_forward!`/`*_transpose!` the inner solve calls). What
# it catches is stale cached state, a wrong dual slice, a layout drift between solve and report, and
# a `q0`/`r` inconsistency. What gates the algebra itself is the D=4 dense oracle.
#
# ONE GENUINELY INDEPENDENT CHECK IS ADDED on top, because this family's own rows are the new code:
# `level_prob`/`joint_prob` under the LFD are cross-checked against the KKT residuals they must
# reproduce, and the SHARED-mu cumulative factorization residuals are computed from them. Under a
# shared `mu` there is ONE reference marginal, so the marginal report is a length-L vector (the
# reference origin's), not a D x L matrix -- that collapse is the family's whole point and the report
# reflects it rather than hiding it behind a replicated matrix.
#
# Requires: cm_pairwise_quantile_config.jl, cm_pairwise_quantile_moments.jl,
# cm_pairwise_quantile_lookup_kernels.jl (for CM's own lookup kernels), pairwise_quantile_operator.jl.
# ================================================================================================

"""
    verify_inner_solution_operator_cmpairwisequantile!(zeta, lambda, cf, cmpq, state, W,
        economic_forward!, economic_transpose!, econ_ws, Psi!, dPsi!, ncore1) -> NamedTuple

`lambda` is the inner KNITRO solution WITHOUT `zeta`, i.e. `x[2:end]`, laid out
`[lambda_E(ncore1); lambda_L(L-1); lambda_P((L-1)^2*npair); lambda_CM(ncm)]`.

`cmpq` is this family's campaign context (`build_cm_pairwise_quantile_context`'s return value) --
`op`, `ncm`, `Lcm`, `nO`, `origins`, `refIndex1`, `Bidx`, `R`, `Pow` are all read off it, so the
verifier cannot drift from the context the solve actually ran against.
"""
function verify_inner_solution_operator_cmpairwisequantile!(zeta::Float64,
        lambda::AbstractVector{Float64}, cf, cmpq, state::CMPQMassState, W::Int,
        economic_forward!::Function, economic_transpose!::Function, econ_ws,
        Psi!::Function, dPsi!::Function, ncore1::Int)
    op = cmpq.op
    D = op.D; npair = op.npair; L = op.L; nc = L - 1
    ref = cmpq.refIndex1; Lcm = cmpq.Lcm; nO = cmpq.nO; ncm = cmpq.ncm
    origins = cmpq.origins; bins = cmpq.Bidx; R = cmpq.R; Pow = cmpq.Pow
    nL = n_cmpq_level_rows(L); nP = nc * nc * npair
    n_restr = n_cmpq_restr_rows(D, L)
    length(lambda) == ncore1 + n_restr + ncm ||
        error("verify_inner_solution_operator_cmpairwisequantile!: length(lambda)=$(length(lambda)) " *
              "!= ncore1+n_restr+ncm=$(ncore1 + n_restr + ncm)")
    nbins = Lcm + 1
    ncm_cdf = nO * Lcm
    (Pow === nothing) == (ncm == ncm_cdf) ||
        error("verify_inner_solution_operator_cmpairwisequantile!: Pow/ncm disagree -- Pow is " *
              "$(Pow === nothing ? "absent" : "present") but ncm=$ncm vs ncm_cdf=$ncm_cdf")

    λ_E = @view lambda[1:ncore1]
    # SAME slicing convention as `reshape_cmpq_duals` (cm_pairwise_quantile_moments.jl). With ONE
    # shared simplex there is no (o,a) matrix to transpose, so the standalone family's o-major /
    # a-major reshape hazard does not exist here -- see that function's own docstring.
    λ_L = @view lambda[ncore1+1 : ncore1+nL]
    λ_P = reshape(@view(lambda[ncore1+nL+1 : ncore1+nL+nP]), nc, nc, npair)
    λ_CM = @view lambda[ncore1+n_restr+1 : ncore1+n_restr+ncm]

    # ---- r, from fresh scratch, all three blocks ------------------------------------------------
    r = fill(-zeta, W)
    econ_buf = zeros(W)
    economic_forward!(econ_buf, λ_E, cf, econ_ws)
    r .-= econ_buf
    cm_pq_forward!(r, λ_L, λ_P, op, state, ref)
    # CM grid, via CM's own production lookup kernels on FRESH buffers (never the FG state's).
    λmat_block = zeros(nO, Lcm); λmat_ext = zeros(nO, Lcm + 1); cm_contrib = zeros(W)
    λ_cdf = Pow === nothing ? λ_CM : (@view λ_CM[1:ncm_cdf])
    apply_contrast!(λmat_block, reshape(λ_cdf, nO, Lcm), R)
    suffix_sums!(λmat_ext, λmat_block)
    cumulative_forward_contribution!(cm_contrib, bins, ref, origins, λmat_ext)
    r .-= cm_contrib
    if Pow !== nothing
        λmat_block2 = zeros(nO, Lcm); λmat_ext2 = zeros(nO, Lcm + 1); cm_contrib2 = zeros(W)
        λ_pow = @view λ_CM[ncm_cdf+1 : 2*ncm_cdf]
        apply_contrast!(λmat_block2, reshape(λ_pow, nO, Lcm), R)
        suffix_sums!(λmat_ext2, λmat_block2)
        cumulative_forward_contribution_pow!(cm_contrib2, bins, ref, origins, λmat_ext2, Pow)
        r .-= cm_contrib2
    end

    Psi_r = similar(r); Psi!(Psi_r, r)
    f = sum(Psi_r) / W + zeta
    dPsi_r = similar(r); dPsi!(dPsi_r, r)

    # ---- the full dual gradient, all three blocks -----------------------------------------------
    g_E = zeros(ncore1)
    economic_transpose!(g_E, dPsi_r, cf, econ_ws)
    g_E .*= -(1.0 / W)

    g_L = zeros(nc); g_P = zeros(nc, nc, npair)
    tls = build_pairwise_quantile_thread_scratch(D, npair, L)
    scratch = PairwiseQuantileTransposeScratch(D, npair, L)
    cm_pq_transpose!(g_L, g_P, dPsi_r, op, state, ref, tls, scratch)

    hist_partials = [zeros(D, nbins) for _ in 1:max(1, Threads.nthreads())]
    hist_h = zeros(D, nbins); Hpre = zeros(D, Lcm)
    g_block = zeros(nO, Lcm); g_stored = zeros(nO, Lcm)
    build_weighted_histogram!(hist_h, hist_partials, bins, dPsi_r, D, nbins)
    prefix_sums!(Hpre, hist_h, Lcm)
    cumulative_backward_gradient_from_prefix!(g_block, Hpre, ref, origins, Lcm, W)
    apply_contrast!(g_stored, g_block, R)
    g_CM = Vector{Float64}(undef, ncm)
    g_CM[1:ncm_cdf] .= vec(g_stored)
    if Pow !== nothing
        hist_partials2 = [zeros(D, nbins) for _ in 1:max(1, Threads.nthreads())]
        hist_h2 = zeros(D, nbins); Hpre2 = zeros(D, Lcm)
        g_block2 = zeros(nO, Lcm); g_stored2 = zeros(nO, Lcm)
        build_weighted_histogram_pow!(hist_h2, hist_partials2, bins, dPsi_r, Pow, D, nbins)
        prefix_sums!(Hpre2, hist_h2, Lcm)
        # eq.36's indicator is `1{U>c}`: reflect the prefix table, exactly as the FG functor does.
        @inbounds for o in 1:D
            total_o = sum(@view hist_h2[o, :])
            for l in 1:Lcm
                Hpre2[o, l] = total_o - Hpre2[o, l]
            end
        end
        cumulative_backward_gradient_from_prefix!(g_block2, Hpre2, ref, origins, Lcm, W)
        apply_contrast!(g_stored2, g_block2, R)
        g_CM[ncm_cdf+1:ncm] .= vec(g_stored2)
    end

    g_lambda = vcat(g_E, g_L, vec(g_P), g_CM)
    kkt_resid_E = isempty(g_E) ? 0.0 : maximum(abs, g_E)
    kkt_resid_level = maximum(abs, g_L)
    kkt_resid_pairindep = maximum(abs, g_P)
    kkt_resid_cm = maximum(abs, g_CM)

    # ---- probability report UNDER THE LFD --------------------------------------------------------
    # Same reasoning as the standalone family's version-B correction: the targets are the free masses
    # `mu`, so an UNWEIGHTED report would flag the search doing exactly what it is supposed to do.
    # The quantity the restriction constrains is the probability under the least-favourable measure
    # `m_w = Psi'(r_w)`, and reporting that makes this the human-readable form of the KKT residuals
    # just computed (`g_L[a] = 0` is exactly `Mraw[ref,a]/S_m = mu[a]`).
    #
    # `cm_pq_transpose!` already built the full `1:L` m-weighted tables into `scratch.Mraw`/`Praw`
    # on its way to `g_L`/`g_P`, so this reads them rather than making a second O(W*(D+npair)) pass.
    #
    # ONE reference marginal, not D of them: under a shared `mu` the level rows constrain the
    # reference origin's marginal only, and the other origins' marginals are implied by CM. Both are
    # reported -- `level_prob` (the constrained one) and `origin_prob` (all D, for the campaign log),
    # so a reader can see the implication holding rather than take it on faith.
    S_m = sum(dPsi_r)
    S_m > 0 || error("verify_inner_solution_operator_cmpairwisequantile!: sum of LFD weights is $S_m <= 0")
    origin_prob = scratch.Mraw ./ S_m           # D x L, all origins, all bins, under the LFD
    level_prob = origin_prob[ref, :]            # L, the reference origin's -- what is enforced
    joint_prob = scratch.Praw ./ S_m            # L x L x npair

    # ---- cumulative factorization residuals against the SHARED Pcum ------------------------------
    Pcum = state.Pcum
    cum_resid = zeros(nc, nc, npair)
    @inbounds for pidx in 1:npair
        for s in 1:nc, rr in 1:nc
            Frs = 0.0
            for b in 1:s, a in 1:rr
                Frs += joint_prob[a, b, pidx]
            end
            cum_resid[rr, s, pidx] = abs(Frs - Pcum[rr] * Pcum[s])
        end
    end
    level_cum_resid = zeros(nc)
    @inbounds for rr in 1:nc
        level_cum_resid[rr] = abs(sum(@view level_prob[1:rr]) - Pcum[rr])
    end
    # The CM implication, measured rather than assumed: every NON-reference origin's cumulative
    # marginal must also match the shared `Pcum`, because CM ties the origins' marginals to each
    # other and the level rows pin the level. This is the family's defining claim, so the verifier
    # reports how well it actually holds at the solved point rather than asserting it structurally.
    implied_cum_resid = zeros(nc, D)
    @inbounds for o in 1:D, rr in 1:nc
        implied_cum_resid[rr, o] = abs(sum(@view origin_prob[o, 1:rr]) - Pcum[rr])
    end

    record_operator_verification!()
    return (r = r, f = f, g_lambda = g_lambda, kkt_resid = maximum(abs, g_lambda),
            kkt_resid_E = kkt_resid_E, kkt_resid_level = kkt_resid_level,
            kkt_resid_pairindep = kkt_resid_pairindep, kkt_resid_cm = kkt_resid_cm,
            mu = copy(state.mu), mu_last = state.mu_last, Pcum = copy(Pcum),
            level_prob = level_prob, origin_prob = origin_prob, joint_prob = joint_prob,
            cumulative_residual = cum_resid, level_cumulative_residual = level_cum_resid,
            implied_cumulative_residual = implied_cum_resid,
            max_cumulative_residual = isempty(cum_resid) ? 0.0 : maximum(cum_resid),
            max_level_cumulative_residual = isempty(level_cum_resid) ? 0.0 : maximum(level_cum_resid),
            max_implied_cumulative_residual = isempty(implied_cum_resid) ? 0.0 : maximum(implied_cum_resid))
end

"""
    assert_cm_pairwise_quantile_d20_counts(D, L, G, n_families) -> NamedTuple

The family's own final-count report, asserted rather than printed, mirroring
`assert_pairwise_quantile_d20_counts`. At the production configuration (D=20, L=5, G=50, two
families) it additionally asserts the exact published numbers as a regression sentinel:
`npair=190`, `n_restr=3044`, `ncm=1862`, and the outer collapse `L-1=4` against the standalone
family's `D*(L-1)=80`.
"""
function assert_cm_pairwise_quantile_d20_counts(D::Int, L::Int, G::Int, n_families::Int)
    D == 20 || error("assert_cm_pairwise_quantile_d20_counts: expected D=20, got D=$D")
    L >= 2 || error("assert_cm_pairwise_quantile_d20_counts: L must be >= 2, got $L")
    G % L == 0 || error("assert_cm_pairwise_quantile_d20_counts: L=$L does not divide G=$G")
    npair = div(D * (D - 1), 2)
    nO = D - 1
    Lcm = G - 1
    n_restr = n_cmpq_restr_rows(D, L)
    ncm = n_families * nO * Lcm
    npair == 190 || error("unordered_pairs assertion failed: got $npair, expected 190")
    if L == 5 && G == 50 && n_families == 2
        n_restr == 3044 || error("n_restr assertion failed: got $n_restr, expected 3044")
        ncm == 1862 || error("ncm assertion failed: got $ncm, expected 1862")
        n_cmpq_raw(L) == 4 || error("outer mass coordinate assertion failed: got $(n_cmpq_raw(L)), expected 4")
    end
    return (n_bins = L, n_free_bins = L - 1, outer_mass_params = n_cmpq_raw(L),
            standalone_outer_mass_params = (L - 1) * D, unordered_pairs = npair,
            level_rows = n_cmpq_level_rows(L), pair_rows = n_cmpq_pair_rows(D, L),
            restriction_rows = n_restr, cm_moments = ncm, n_cm_levels = Lcm,
            inner_rows = n_restr + ncm, dense_G_production = false)
end
