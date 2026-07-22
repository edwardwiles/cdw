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
                     X_data=eq.trade_flow, entry_target=p.w .* p.f_entry)

Fills `K` (`W`-vector) and `G` (`W x layout.num_moments`) in place.

- `G[:, layout.trade_index[o,d]] = entrant_mass[o]*realized_revenue_od(z)/expenditure_d -
  lambda_od`, where `lambda_od = X_data[o,d]/expenditure_d` -- i.e. the trade moment
  matches SHARES, not flow levels. This mirrors the Ricardian repo's own convention:
  `moments/hFunction.jl`'s trade-share moment (`G[ω,d1] = pricesTemp[o] -
  P[d1]*denom[d]`) compares a simulated per-draw object against `P[d1]` (`lambda_od`,
  confirmed by reading `prepare_cc/buildObjectsForMoments.jl` -- `P` is literally the
  vectorized trade-SHARE data) times a destination-level rescaling constant `denom[d]` --
  the fundamental data object matched is the share, and `setup/createFakeData.jl`
  constructs exactly that (`lambda`) via the closed-form Frechet gravity equation, never a
  Monte-Carlo sample average. Matching shares (dividing by `expenditure_d`, the natural
  destination-wide normalization) also puts every cell's residual on a comparable ~O(1)
  scale, unlike raw flow levels which can span orders of magnitude across cells.
- `G[:, layout.entry_index[o]] = sum_d realized_operating_profit_od(z) - entry_target[o]`
  (never multiplied by `entrant_mass[o]`, docs Section 4B/7).
- `K[:] .= price_power_prime[target] - 1` (the raw counterfactual scalar; the GT
  transform `1-(.)^(1/(1-sigma))` is applied downstream, matching the Ricardian repo's
  own `counterVal = gamma[baseIndex]/gamma_prime[baseIndex] - 1` convention, docs Section
  2) -- constant across draws because our counterfactual is deterministic given
  `(p, eq)`, exactly as production's `counterExplicit==0` branch.

`X_data` defaults to `eq.trade_flow` and `entry_target` defaults to the analytical
`w.*f_entry` (the model's *population*, CLOSED-FORM values -- matching how
`setup/createFakeData.jl` builds its own "data" via closed form, not Monte Carlo
simulation; this is therefore the PRIMARY validation target, not a sample average over
`z_draws`). Under this target, `Delta(theta*)` is generally small but NOT exactly zero
even at the true parameters -- machine-precision zero would only be expected if `X_data`
were itself defined as the sample mean over the SAME `z_draws` used to evaluate the
moments (a tautological construction the brief also describes as "Mode 1", useful only as
a code-correctness check, not a believable validation with real or even closed-form data).
"""
function melitz_moments!(K::AbstractVector, G::AbstractMatrix, p::MelitzPrimitives,
                          eq::MelitzEquilibrium, cf::MelitzCounterfactual,
                          z_draws::AbstractMatrix, layout::MelitzMomentLayout;
                          X_data::AbstractMatrix=eq.trade_flow,
                          entry_target::AbstractVector=p.w .* p.f_entry)
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
            lambda_od = X_data[o, d] / eq.expenditure[d]
            for w in 1:W
                z = z_draws[w, o]
                firm = melitz_firm(p.w[o], p.tau[o, d], p.A[o, d], p.f[o, d], p.sigma,
                                    eq.expenditure[d], price_power_d, z)
                model_share = eq.entrant_mass[o] * firm.realized_revenue / eq.expenditure[d]
                G[w, trade_col] = model_share - lambda_od
                G[w, entry_col] += firm.realized_operating_profit
            end
        end
        @views G[:, entry_col] .-= entry_target[o]
    end

    K .= cf.price_power_prime - 1
    return nothing
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

"""
    min_active_draw_count(p::MelitzPrimitives, eq::MelitzEquilibrium, z_draws) -> (count, cell)

Diagnostic: the minimum, across all `D^2` cells, of the number of draws in `z_draws` with
`z_draws[w,o] > eq.cutoff[o,d]` (i.e. an ACTIVE/participating firm for that cell), and
which cell attains it.

Matters because a cell with **zero** active draws has a perfectly constant (never-zero)
share/level residual across every single draw -- no reweighting of a Psi-divergence dual
problem can bring that to zero, so the CC inner KNITRO solve genuinely diverges (dual
variables blow up, not a graceful "large but finite" answer). This is not a bug: bilateral
participation probability under Pareto is `Pr(active) = zhat_od^(-theta_star)`, which can
be very small for high-cutoff (especially export) cells, so a MUCH larger `W` than one
might naively expect can be required before every cell has even a handful of active draws
-- discovered live while validating `Delta(theta*)` against a closed-form (non-tautological)
population target: `W` in the low thousands to tens of thousands reliably left at least one
cell with zero active draws for the D=4 benchmark fixture; `W>=100_000` did not.
"""
function min_active_draw_count(p::MelitzPrimitives, eq::MelitzEquilibrium, z_draws::AbstractMatrix)
    D = p.D
    min_count = size(z_draws, 1)
    worst_cell = (0, 0)
    for o in 1:D, d in 1:D
        n_active = count(>(eq.cutoff[o, d]), @view(z_draws[:, o]))
        if n_active < min_count
            min_count = n_active
            worst_cell = (o, d)
        end
    end
    return min_count, worst_cell
end
