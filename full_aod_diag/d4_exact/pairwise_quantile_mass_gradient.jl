# ================================================================================================
# EXACT closed-form outer gradient of `Delta*` w.r.t. this restriction's free bin masses `mu_{o,a}`
# (version B, free-mass reparameterization, 2026-08-10).
#
# THIS IS A MIRROR, NOT A NEW GRADIENT ENGINE. The formula below is structurally identical to
# origin-ZC's long-validated `d_delta_dual_d_eta_origin_vec` (cm_originzc_moments.jl) -- same
# envelope argument, same `-mean_m *(own dual + sum over partners of partner-target x pair dual)`
# shape, same product-rule term. Read that function alongside this one; the only real difference is
# the index the PAIR block runs over (origin-ZC: the power `k`; here: the bin pair `(a,b)`), which
# is called out explicitly below because it is the one place a silent indexing bug could live.
#
# DERIVATION (do it yourself before trusting this; it is four lines).
# The inner solve minimizes `f = (1/W) sum_w Psi(r_w) + zeta`, the reported divergence is
# `Delta* = -f*` (`verify_namedtuple_from_operator`), and `mu` enters `r` ONLY through the per-draw
# constant `C_lambda` of `pairwise_quantile_forward!`:
#
#   C_lambda      = sum_{o,a} lambda^M_{o,a} mu_{o,a} + sum_{(o,p),a,b} lambda^P_{op,ab} mu_{o,a} mu_{p,b}
#   dC_lambda/dmu_{o,a} = lambda^M_{o,a} + sum_{p != o} sum_b lambda^P_{op,ab} mu_{p,b}   =:  A_{o,a}
#   dr_w/dmu_{o,a}      = +A_{o,a}          (draw-INDEPENDENT: r = ... - (G_R lambda_R)_w, and
#                                            forward! SUBTRACTS, so -d(-C_lambda) = +A)
#   df/dmu_{o,a}        = (1/W) sum_w Psi'(r_w) A_{o,a} = A_{o,a} * mean_m
#
# and by the envelope theorem at the converged `(zeta*, lambda*)`:
#
#   d(Delta*)/dmu_{o,a} = -mean_m * ( lambda^M_{o,a} + sum_{p != o} sum_b lambda^P_{op,ab} mu_{p,b} )
#
# compared with origin-ZC's, which this file deliberately reproduces line for line:
#
#   d(Delta*)/dnu_{o,k} = -mean_m * ( lambda_mean,o,k  + sum_{p != o} nu_{p,k} lambda_pair,op,k )
#
# `mean_m = (1/W) sum_w Psi'(r_w)` is `verify.m_mean`, exactly the scalar origin-ZC passes.
#
# WHY THERE IS NO BANDWIDTH/SECANT MACHINERY HERE (and why version A needed some). Version A's
# outer coordinates were the quantile CUTOFFS, which sit inside indicator functions: moving one did
# nothing until it crossed a draw, so `Delta*` was a genuine step function of every outer coordinate
# and its exact derivative was zero between crossings and undefined at them. Version B's outer
# coordinates shift a moment TARGET smoothly and never reassign a draw between bins -- precisely
# origin-ZC's situation -- so the envelope derivative above is exact, and
# `pairwise_quantile_cutoff_gradient.jl` (bandwidth_target/crossed_draw_range/fixed_dual_delta_f/
# cutoff_probe_points/cutoff_secant_gradient!) was DELETED rather than kept as a fallback. Removing
# that tuning burden is the point of the reparameterization.
#
# Requires pairwise_quantile_bin_context.jl (PairwiseQuantileOperator/PairwiseQuantileMassState) and
# pairwise_quantile_mass_transform.jl (mass_jacobian_block!, raw_index) to already be included.
# ================================================================================================

"""
    d_delta_dual_d_mu(lambda_M, lambda_P, mu, op; mean_m::Float64) -> Matrix{Float64}

`d(Delta_dual)/d(mu_{o,a})` for every origin/free-bin pair, returned as a `D x (L-1)` matrix
indexed exactly like `lambda_M` and `state.mu`.

`lambda_M`/`lambda_P` are the converged restriction duals in this family's own block shapes
(`reshape_pq_duals`), `mu` the free masses those duals were solved at, `mean_m` the LFD weight mean
(`verify.m_mean`).

INDEXING NOTE -- the one genuine difference from `d_delta_dual_d_eta_origin_vec`. Origin-ZC's pair
dual is indexed by POWER `k`, so its product term pairs `nu_{p,k}` with `lambda_pair,op,k` at the
SAME `k`. This family's pair dual is indexed by the BIN PAIR `(a,b)`: row `(op,a,b)` constrains
"origin `o` in bin `a` AND origin `p` in bin `b`", so the mass that multiplies `lambda^P_{op,ab}` in
`dC/dmu_{o,a}` is `mu_{p,b}` -- the PARTNER's mass at the PARTNER's own bin index, summed over `b`.
Reading it as `mu_{p,a}` (same bin index) would be the natural transcription error, and would be
invisible at `mu = 1/L` (where every bin's mass is equal). The FD gate is run at a NON-uniform `mu`
for exactly this reason.
"""
function d_delta_dual_d_mu(lambda_M::AbstractMatrix{Float64}, lambda_P::AbstractArray{Float64,3},
                            mu::AbstractMatrix{Float64}, op::PairwiseQuantileOperator; mean_m::Float64)
    D = op.D; npair = op.npair; L = op.L; nc = L - 1
    size(lambda_M) == (D, nc) || error("d_delta_dual_d_mu: size(lambda_M)=$(size(lambda_M)) != ($D,$nc)")
    size(lambda_P) == (nc, nc, npair) || error("d_delta_dual_d_mu: size(lambda_P)=$(size(lambda_P)) != ($nc,$nc,$npair)")
    size(mu) == (D, nc) || error("d_delta_dual_d_mu: size(mu)=$(size(mu)) != ($D,$nc)")

    d_mu = zeros(D, nc)
    @inbounds for a in 1:nc, o in 1:D
        d_mu[o, a] -= lambda_M[o, a]
    end
    @inbounds for pidx in 1:npair
        (o, p) = op.pairs[pidx]
        for b in 1:nc, a in 1:nc
            lp = lambda_P[a, b, pidx]
            lp == 0.0 && continue
            d_mu[o, a] -= mu[p, b] * lp      # partner's mass at the PARTNER's bin index -- see docstring
            d_mu[p, b] -= mu[o, a] * lp
        end
    end
    d_mu .*= mean_m
    return d_mu
end

"""
    chain_mass_gradient_to_raw(d_mu, raw, mu, layout) -> Vector{Float64}

Chain-rules `d(Delta_dual)/d(mu_{o,a})` through the stick-breaking transform to the raw KNITRO
coordinates: `g[raw_index(o,k)] = sum_a d_mu[o,a] * (d mu_{o,a} / d raw_{o,k})`, with the per-origin
Jacobian from `mass_jacobian_block!`.

Block-diagonal across origins by construction (the transform is), so this never mixes origins --
the same structure `cutoff_secant_gradient!`'s own chain-rule step had, and the same reason the
outer coordinate layout is origin-major.
"""
function chain_mass_gradient_to_raw(d_mu::AbstractMatrix{Float64}, raw::AbstractVector{Float64},
                                     mu::AbstractMatrix{Float64}, layout::PairwiseQuantileMassLayout)
    D = layout.D; nb = n_free_bins(layout)
    size(d_mu) == (D, nb) || error("chain_mass_gradient_to_raw: size(d_mu)=$(size(d_mu)) != ($D,$nb)")
    size(mu) == (D, nb) || error("chain_mass_gradient_to_raw: size(mu)=$(size(mu)) != ($D,$nb)")
    length(raw) == n_raw(layout) ||
        error("chain_mass_gradient_to_raw: length(raw)=$(length(raw)) != n_raw(layout)=$(n_raw(layout))")
    g = zeros(n_raw(layout))
    J = zeros(nb, nb)
    mu_row = Vector{Float64}(undef, nb)
    @inbounds for o in 1:D
        base = raw_index(layout, o, 1)
        for a in 1:nb
            mu_row[a] = mu[o, a]
        end
        mass_jacobian_block!(J, @view(raw[base:base+nb-1]), mu_row)
        for k in 1:nb
            acc = 0.0
            for a in 1:nb
                acc += d_mu[o, a] * J[a, k]
            end
            g[base+k-1] = acc
        end
    end
    return g
end
