# Sorted fixed-dual scalar/dual-argument construction (2026-07-26 continuation, Phase 8 of
# docs/melitz_sorted_tail_optimization_2026-07-25.md). Pure Melitz-side prototype -- does
# NOT touch cc_algo/inner_loop_functions.jl (the shared KNITRO-facing inner-loop hot path)
# at all, and does not modify direct_gradient.jl's own `_base_arg0!` either. This is a
# standalone, independently-validated function a Melitz-specific caller can use, matching
# the governing prompt's own Phase 8 scoping ("implement first as an experimental exact
# backend... do not integrate it into the production inner solver until exact equivalence
# passes"). Not wired into any production driver this session.
#
# THE ALGORITHM (per the governing prompt's own outline):
#
# The fixed-dual scalar argument is `u[w] = -zeta - dot(G[w,:], mu)` (`mu` the dual weight on
# every moment column, `zeta` the K-column multiplier -- matches `direct_gradient.jl`'s own
# `_base_arg0!` construction exactly, `PsiObjectiveBundleImplicit`'s non-gradient functor
# branch).
#
# For a fixed origin `o`, its own `D` trade-share columns contribute (Section A.1's algebraic
# identity, `melitz_moments_sorted_tail!`'s own derivation):
#
#     sum_d mu_od * G[w, trade_index[o,d]]
#       = sum_d mu_od*(-lambda_od)                                    [a draw-INDEPENDENT
#                                                                        constant, per origin]
#       + z_{w,o}^(sigma-1) * sum_{d: cutoff_od < z_{w,o}} mu_od*coef_od   [the active-tail sum]
#
# The active-tail sum, for a GIVEN `z`, is a sum over the destinations whose cutoff is BELOW
# `z` -- i.e., in cutoff-ASCENDING order, a PREFIX. Sorting the `D` (not `W`!) cutoffs once
# per origin and forming a prefix cumulative sum turns "which destinations are active, and
# what do they sum to" into ONE binary search (into a length-`D` array) plus one array
# lookup, for EVERY draw -- no `D`-length inner loop per draw. Sweeping origin `o`'s own
# ALREADY-SORTED `W` draws (the same `sorted_ctx` Phases 1-7 already build) then costs
# `O(W log D)` per origin instead of `O(W D)`, and `O(D log D)` to sort/cumsum the `D`
# cutoffs once -- versus the dense reference's `O(W D)` per origin (`O(W D^2)` total across
# `D` origins). At real `D=20`, this is a genuine, `D`-independent-of-active-fraction
# reduction (`D/log2(D) ~ 4.6x` fewer per-draw operations), COMPLEMENTARY to (not the same
# mechanism as) Phase 3's active-fraction-exploiting trick.

using LinearAlgebra: dot

"""
    melitz_sorted_dual_argument(zeta, mu, p, eq, cf, sorted_ctx, layout; X_data=eq.trade_flow) -> u::Vector{Float64}

Phase 8 prototype: builds `u[w] = -zeta - dot(G[w,:], mu)` directly from the sorted-tail
context, WITHOUT ever materializing the dense `W x (D^2+1)` moment matrix `G`. `mu` must
have length `layout.num_moments` (one dual weight per moment column, trade-share cells
first then the focal link column, matching `layout`'s own convention).

The trade-share block (`D^2` columns) is computed via the sorted cumulative-sum sweep
described in this file's own header, one pass per origin. The single focal link column is
computed DENSELY (the same Section C.1 scope decision as `melitz_moments_sorted_tail!`) and
folded in with one more `O(D*W)` pass -- skipped entirely if `mu[layout.focal_link_index] ==
0` (a common case when a caller is probing only the trade-share block's own contribution).
"""
function melitz_sorted_dual_argument(zeta::Real, mu::AbstractVector{Float64}, p::MelitzPrimitives,
                                      eq::MelitzEquilibrium, cf::MelitzCounterfactual,
                                      sorted_ctx::MelitzSortedTailContext, layout::MelitzMomentLayout;
                                      X_data::AbstractMatrix=eq.trade_flow)
    D = p.D
    W = sorted_ctx.W
    D == sorted_ctx.D || throw(ArgumentError(
        "melitz_sorted_dual_argument: sorted_ctx.D=$(sorted_ctx.D) != p.D=$D"))
    p.sigma == sorted_ctx.sigma || throw(ArgumentError(
        "melitz_sorted_dual_argument: sorted_ctx.sigma=$(sorted_ctx.sigma) != p.sigma=$(p.sigma) " *
        "-- stale sorted context (built under a different sigma)"))
    length(mu) == layout.num_moments || throw(ArgumentError(
        "melitz_sorted_dual_argument: mu must have length layout.num_moments=$(layout.num_moments), got $(length(mu))"))

    sigma = p.sigma
    u = fill(-Float64(zeta), W)

    cutoffs_o = Vector{Float64}(undef, D)
    coefmu_o = Vector{Float64}(undef, D)
    prefix_o = Vector{Float64}(undef, D + 1)

    @melitz_profile :sorted_dual_argument_trade begin
        @inbounds for o in 1:D
            const_o = 0.0
            for d in 1:D
                cutoffs_o[d] = eq.cutoff[o, d]
                C_od = melitz_C(p.w[o], p.tau[o, d], p.A[o, d], sigma, eq.expenditure[d])
                coef_od = C_od / eq.expenditure[d]
                lambda_od = X_data[o, d] / eq.expenditure[d]
                mu_od = mu[layout.trade_index[o, d]]
                coefmu_o[d] = mu_od * coef_od
                const_o -= mu_od * lambda_od
            end
            u .-= const_o   # u = -zeta - dot(G,mu); applies to every draw regardless of origin o's own z value

            perm_d = sortperm(cutoffs_o)
            sorted_cutoffs = @view cutoffs_o[perm_d]
            prefix_o[1] = 0.0
            for m in 1:D
                prefix_o[m+1] = prefix_o[m] + coefmu_o[perm_d[m]]
            end

            sorted_z_o = @view sorted_ctx.sorted_z[:, o]
            zpow_o = @view sorted_ctx.sorted_z_power[:, o]
            perm_w = @view sorted_ctx.permutation[:, o]
            for pos in 1:W
                z = sorted_z_o[pos]
                m = searchsortedfirst(sorted_cutoffs, z) - 1   # count of cutoffs STRICTLY < z
                m == 0 && continue    # nothing active for this origin at this draw
                s = perm_w[pos]
                u[s] -= zpow_o[pos] * prefix_o[m+1]
            end
        end
    end

    j = p.target_country
    mu_link = mu[layout.focal_link_index]
    if mu_link != 0.0
        @melitz_profile :sorted_dual_argument_link begin
            profit_j = zeros(Float64, W)
            z_orig = sorted_ctx.z_original
            @inbounds for d in 1:D
                for w in 1:W
                    z = z_orig[w, j]
                    firm = melitz_firm(p.w[j], p.tau[j, d], p.A[j, d], p.f[j, d], sigma,
                                        eq.expenditure[d], 1.0, z)
                    profit_j[w] += firm.realized_operating_profit
                end
            end
            price_power_autarky = p.gamma_prime_target
            @inbounds for w in 1:W
                z_j = z_orig[w, j]
                firm_autarky = melitz_firm(cf.w_prime, 1.0, p.A[j, j], p.f[j, j], sigma,
                                            cf.expenditure_prime, price_power_autarky, z_j)
                link_val = profit_j[w] / p.w[j] - firm_autarky.realized_operating_profit / cf.w_prime
                u[w] -= mu_link * link_val
            end
        end
    end

    return u
end

"""
    melitz_dense_dual_argument(zeta, mu, K, G) -> u::Vector{Float64}

Reference (dense) construction of the SAME quantity `melitz_sorted_dual_argument` computes:
`u[w] = -zeta - dot(G[w,:], mu)`, given an already-built moment matrix `G` (e.g. from
`melitz_moments!`). `K` is accepted for call-site symmetry with `melitz_moments!`'s own
`(K, G)` output pair but is NOT used (matches `direct_gradient.jl`'s own `_base_arg0!`,
which reads only `obj.H`'s `[ones(M) G]` block, never the `K` column -- `moments.jl`'s own
documented placeholder). Used as the correctness reference for
`melitz_sorted_dual_argument` (test suite); not itself a production hot path.
"""
function melitz_dense_dual_argument(zeta::Real, mu::AbstractVector{Float64}, K::AbstractVector, G::AbstractMatrix)
    W = size(G, 1)
    u = Vector{Float64}(undef, W)
    @inbounds for w in 1:W
        u[w] = -Float64(zeta) - dot(@view(G[w, :]), mu)
    end
    return u
end
