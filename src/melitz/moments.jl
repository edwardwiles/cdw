# Moment matrix (G) construction for the full-D Melitz Christensen-Connault benchmark.
# See docs/melitz_delta_star.md Section 4 for the moment definitions.
#
# All D^2 bilateral trade-flow moments and D free-entry moments are computed here, using
# the SAME shared per-draw firm routine (firm_quantities.jl) for every (o,d) cell and
# every draw -- never gated on a focal country, never evaluated at a single draw index.
# The two gravity restrictions are NOT columns of G (docs Section 4C) -- they are
# F-independent and are evaluated directly from (A, f, tau) by `gravity_residuals`.

"""
    melitz_moments!(K, G, p::MelitzPrimitives, eq::MelitzEquilibrium,
                     cf::MelitzCounterfactual, z_draws, layout::MelitzMomentLayout;
                     X_data=eq.trade_flow, entry_target=p.w .* p.f_entry, scale_trade=true)

Fills `K` (`W`-vector) and `G` (`W x layout.num_moments`) in place.

- `G[:, layout.trade_index[o,d]] = entrant_mass[o]*realized_revenue_od(z) - X_data[o,d]`
  (optionally divided by `X_data[o,d]` for conditioning when `scale_trade=true` -- the
  economic zero set is unchanged either way).
- `G[:, layout.entry_index[o]] = sum_d realized_operating_profit_od(z) - entry_target[o]`
  (never multiplied by `entrant_mass[o]`, docs Section 4B/7).
- `K[:] .= price_power_prime[target] - 1` (the raw counterfactual scalar; the GT
  transform `1-(.)^(1/(1-sigma))` is applied downstream, matching the Ricardian repo's
  own `counterVal = gamma[baseIndex]/gamma_prime[baseIndex] - 1` convention, docs Section
  2) -- constant across draws because our counterfactual is deterministic given
  `(p, eq)`, exactly as production's `counterExplicit==0` branch.

`X_data` defaults to `eq.trade_flow` and `entry_target` defaults to the analytical
`w.*f_entry` (the model's *population* values -- appropriate for Mode 2, evaluating a
finite-`W` sample against the population target, docs "Mode 2"). For the Mode 1
exact-sample smoke test, both `X_data` AND `entry_target` must be recomputed as SAMPLE
AVERAGES over the SAME `z_draws` (see `scripts/run_melitz_delta_star_fake.jl`) -- passing
the analytical population values in Mode 1 mixes a finite-sample realization against its
own (slightly different) population moment and would show spurious O(1/sqrt(W)) residuals
that are pure Monte Carlo noise, not a modeling error.
"""
function melitz_moments!(K::AbstractVector, G::AbstractMatrix, p::MelitzPrimitives,
                          eq::MelitzEquilibrium, cf::MelitzCounterfactual,
                          z_draws::AbstractMatrix, layout::MelitzMomentLayout;
                          X_data::AbstractMatrix=eq.trade_flow,
                          entry_target::AbstractVector=p.w .* p.f_entry,
                          scale_trade::Bool=true)
    D = p.D
    W = size(z_draws, 1)
    size(z_draws, 2) == D || throw(ArgumentError("z_draws must be W x D (one draw per origin)"))
    size(G, 1) == W || throw(ArgumentError("G must have W rows matching z_draws"))
    size(G, 2) == layout.num_moments || throw(ArgumentError("G must have layout.num_moments columns"))
    length(K) == W || throw(ArgumentError("K must have length W"))

    @views G[:, :] .= 0
    price_power_d = 1.0 # baseline normalization (docs Section 1.2) -- price_power_d == 1 for every d

    @inbounds for o in 1:D
        entry_col = layout.entry_index[o]
        for d in 1:D
            trade_col = layout.trade_index[o, d]
            for w in 1:W
                z = z_draws[w, o]
                firm = melitz_firm(p.w[o], p.tau[o, d], p.A[o, d], p.f[o, d], p.sigma,
                                    eq.expenditure[d], price_power_d, z)
                g_trade = eq.entrant_mass[o] * firm.realized_revenue - X_data[o, d]
                if scale_trade
                    g_trade /= X_data[o, d]
                end
                G[w, trade_col] = g_trade
                G[w, entry_col] += firm.realized_operating_profit
            end
        end
        @views G[:, entry_col] .-= entry_target[o]
    end

    K .= cf.price_power_prime - 1
    return nothing
end

"""
    gravity_residuals(p::MelitzPrimitives) -> (residual_A, residual_f)

The two F-independent gravity restrictions (docs Section 4C), evaluated directly from
`(A, f, tau)` -- never a column of `G`, never duplicated as both a moment and an outer
constraint.
"""
function gravity_residuals(p::MelitzPrimitives)
    T = doubleDiff(p.tau)
    residual_A = sum(T .* doubleDiff(p.A))
    residual_f = sum(T .* doubleDiff(p.f))
    return residual_A, residual_f
end

"""
    trade_flow_residuals(p, eq, z_draws; X_data=eq.trade_flow) -> D x D matrix

Diagnostic: the mean (over draws) of the UNSCALED economic trade-flow residual for every
(o,d) cell -- always reported alongside the (possibly scaled) `G` columns, per the brief.
"""
function trade_flow_residuals(p::MelitzPrimitives, eq::MelitzEquilibrium,
                               z_draws::AbstractMatrix; X_data::AbstractMatrix=eq.trade_flow)
    D = p.D
    W = size(z_draws, 1)
    resid = zeros(D, D)
    for o in 1:D, d in 1:D
        s = 0.0
        for w in 1:W
            z = z_draws[w, o]
            firm = melitz_firm(p.w[o], p.tau[o, d], p.A[o, d], p.f[o, d], p.sigma,
                                eq.expenditure[d], 1.0, z)
            s += eq.entrant_mass[o] * firm.realized_revenue
        end
        resid[o, d] = s / W - X_data[o, d]
    end
    return resid
end

"""
    entry_residuals(p, eq, z_draws) -> D-vector

Diagnostic: the mean (over draws) free-entry residual for every origin (economic units).
"""
function entry_residuals(p::MelitzPrimitives, eq::MelitzEquilibrium, z_draws::AbstractMatrix)
    D = p.D
    W = size(z_draws, 1)
    resid = zeros(D)
    for o in 1:D
        s = 0.0
        for w in 1:W
            profit_sum = 0.0
            for d in 1:D
                z = z_draws[w, o]
                firm = melitz_firm(p.w[o], p.tau[o, d], p.A[o, d], p.f[o, d], p.sigma,
                                    eq.expenditure[d], 1.0, z)
                profit_sum += firm.realized_operating_profit
            end
            s += profit_sum
        end
        resid[o] = s / W - p.w[o] * p.f_entry[o]
    end
    return resid
end
