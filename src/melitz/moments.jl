# Moment matrix (G) construction for the full-D Melitz Christensen-Connault benchmark.
# See docs/melitz_delta_star.md Section 4 for the moment definitions.
#
# ACTIVE MOMENT SYSTEM (minimal, D^2+1 -- supersedes the D^2+D closure): D^2 baseline
# bilateral trade-SHARE moments (main prompt Section 4.1) + exactly ONE focal
# baseline-vs-autarky free-entry LINK moment (Section 4.2). Baseline N_o=1 and
# gamma_d=1 are universal, so the trade-share moment collapses to
# `price_od(z_o)^(1-sigma)*active_od(z_o) - lambda_od` with NO entrant-mass multiplier
# (contrast the superseded `entrant_mass[o]*realized_revenue/expenditure_d`). All D^2
# bilateral trade-flow moments and the single link moment are computed here, using the
# SAME shared per-draw firm routine (firm_quantities.jl) for every cell/draw -- never
# gated on a focal country except where the model itself is asymmetric (the link moment
# is inherently focal-country-specific: it is THE mechanism tying baseline and autarky
# free entry together for `target_country`, per main prompt Section 4.2).
# The two gravity restrictions are NOT columns of G -- F-independent, enforced exactly by
# the gravity pivots in delta_star.jl, evaluated directly via `gravity_residuals`.

"""
    melitz_moments!(K, G, p::MelitzPrimitives, eq::MelitzEquilibrium,
                     cf::MelitzCounterfactual, z_draws, layout::MelitzMomentLayout;
                     X_data=eq.trade_flow)

Fills `K` (`W`-vector) and `G` (`W x layout.num_moments`, `num_moments = D^2+1`) in place.

- `G[:, layout.trade_index[o,d]] = realized_revenue_od(z)/expenditure_d - lambda_od`,
  `lambda_od = X_data[o,d]/expenditure_d`. Since baseline `N_o=1` (`entrant_mass` is not
  even a field of `MelitzEquilibrium` anymore) and `gamma_d=1`, this IS exactly
  `price_od(z_o)^(1-sigma)*active_od(z_o) - lambda_od` (main prompt Section 4.1) -- no
  double markup, no entrant-mass rescaling.
- `G[:, layout.focal_link_index] = Pi_baseline_j(z)/w[j] - Pi_autarky_j(z)/w_prime[j]`
  (main prompt Section 4.2), `Pi_baseline_j(z) = sum_d realized_operating_profit[j,d](z)`,
  `Pi_autarky_j(z) = realized_operating_profit'_jj(z)`. This single column replaces the
  superseded closure's D per-origin free-entry moments AND ties baseline/autarky free
  entry together without ever introducing `f_entry[j]` as a parameter.
- `K .= gamma_prime_target - 1` -- a documented, harmless placeholder (docs Section 10):
  grepping `cc_algo/inner_loop_functions.jl` confirms `K`/`obj.H[:,1]` is never read by a
  fixed-theta `inner_loop` call (only by the full outer delta-search, out of scope this
  milestone), so this is not consumed by `Delta(theta)` itself.

`X_data` defaults to `eq.trade_flow` (the closed-form population value at construction);
callers doing the exact-sample validation (main prompt Section 11) pass a sample-mean
`X_data` built from the SAME `z_draws`.
"""
function melitz_moments!(K::AbstractVector, G::AbstractMatrix, p::MelitzPrimitives,
                          eq::MelitzEquilibrium, cf::MelitzCounterfactual,
                          z_draws::AbstractMatrix, layout::MelitzMomentLayout;
                          X_data::AbstractMatrix=eq.trade_flow)
    D = p.D
    W = size(z_draws, 1)
    size(z_draws, 2) == D || throw(ArgumentError("z_draws must be W x D (one draw per origin)"))
    size(G, 1) == W || throw(ArgumentError("G must have W rows matching z_draws"))
    size(G, 2) == layout.num_moments || throw(ArgumentError("G must have layout.num_moments columns"))
    length(K) == W || throw(ArgumentError("K must have length W"))

    @views G[:, :] .= 0
    price_power_d = 1.0 # baseline normalization -- gamma_d == 1 for every d
    price_power_autarky = p.gamma_prime_target # autarky price power is the SEARCHED parameter, not 1
    j = p.target_country
    profit_j = zeros(eltype(G), W)

    @melitz_profile :moments_trade_share begin
        @inbounds for o in 1:D
            for d in 1:D
                trade_col = layout.trade_index[o, d]
                lambda_od = X_data[o, d] / eq.expenditure[d]
                for w in 1:W
                    z = z_draws[w, o]
                    firm = melitz_firm(p.w[o], p.tau[o, d], p.A[o, d], p.f[o, d], p.sigma,
                                        eq.expenditure[d], price_power_d, z)
                    G[w, trade_col] = firm.realized_revenue / eq.expenditure[d] - lambda_od
                    o == j && (profit_j[w] += firm.realized_operating_profit)
                end
            end
        end
    end

    link_col = layout.focal_link_index
    @melitz_profile :moments_focal_link begin
        @inbounds for w in 1:W
            z_j = z_draws[w, j]
            firm_autarky = melitz_firm(cf.w_prime, 1.0, p.A[j, j], p.f[j, j], p.sigma,
                                        cf.expenditure_prime, price_power_autarky, z_j)
            G[w, link_col] = profit_j[w] / p.w[j] - firm_autarky.realized_operating_profit / cf.w_prime
        end
    end

    K .= p.gamma_prime_target - 1
    return nothing
end

"""
    trade_flow_residuals(p, eq, z_draws; X_data=eq.trade_flow) -> D x D matrix

Diagnostic: the mean (over draws) of the UNSCALED economic trade-flow residual for every
(o,d) cell (baseline `N_o=1`, so no entrant-mass factor).
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
            s += firm.realized_revenue
        end
        resid[o, d] = s / W - X_data[o, d]
    end
    return resid
end

"""
    focal_link_residual(p, eq, cf, z_draws) -> Float64

Diagnostic: the mean (over draws) UNSCALED focal free-entry link residual (economic
units, `Pi_baseline_j/w[j] - Pi_autarky_j/w_prime[j]`).
"""
function focal_link_residual(p::MelitzPrimitives, eq::MelitzEquilibrium,
                              cf::MelitzCounterfactual, z_draws::AbstractMatrix)
    D = p.D
    W = size(z_draws, 1)
    j = p.target_country
    s = 0.0
    for w in 1:W
        z_j = z_draws[w, j]
        profit_baseline = 0.0
        for d in 1:D
            firm = melitz_firm(p.w[j], p.tau[j, d], p.A[j, d], p.f[j, d], p.sigma,
                                eq.expenditure[d], 1.0, z_j)
            profit_baseline += firm.realized_operating_profit
        end
        firm_autarky = melitz_firm(cf.w_prime, 1.0, p.A[j, j], p.f[j, j], p.sigma,
                                    cf.expenditure_prime, p.gamma_prime_target, z_j)
        s += profit_baseline / p.w[j] - firm_autarky.realized_operating_profit / cf.w_prime
    end
    return s / W
end

"""
    min_active_draw_count(p::MelitzPrimitives, eq::MelitzEquilibrium, z_draws) -> (count, cell)

Diagnostic: the minimum, across all `D^2` cells, of the number of draws in `z_draws` with
`z_draws[w,o] > eq.cutoff[o,d]` (an ACTIVE/participating firm for that cell), and which
cell attains it. A cell with zero active draws has a constant (never-zero) share residual
across every draw -- no reweighting of the CC dual problem can zero it, so the inner
KNITRO solve genuinely diverges. Bilateral participation probability under the reference
Pareto is `pareto_tail_prob(zhat_od, theta_star) = zhat_od^(-theta_star)`, which can be
very small for high-cutoff cells.
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

"""
    cell_participation_diagnostics(p, eq, z_draws; weights=nothing) -> NamedTuple

Main prompt Section 1.6 extension: for every `(o,d)` cell, reports `cutoff`, the
reference PARETO participation probability `pareto_tail_prob(zhat_od, theta_star)`, the
raw active-draw `count`, and (if `weights` given) the EFFECTIVE active count under the
LFD, `sum(weights[w] for w active) * W` (the LFD-reweighted analogue of `count` -- how
many "effective" reference draws the recovered distribution places on the active region,
useful for spotting cells where the LFD has pushed weight almost entirely off the
participation region even though the raw draw count looked fine).
"""
function cell_participation_diagnostics(p::MelitzPrimitives, eq::MelitzEquilibrium,
                                         z_draws::AbstractMatrix;
                                         weights::Union{Nothing,AbstractVector}=nothing)
    D = p.D
    W = size(z_draws, 1)
    cutoff = eq.cutoff
    ref_prob = [pareto_tail_prob(cutoff[o, d], p.theta_star) for o in 1:D, d in 1:D]
    count_active = zeros(Int, D, D)
    effective_active = weights === nothing ? nothing : zeros(Float64, D, D)
    for o in 1:D, d in 1:D
        active_mask = @view(z_draws[:, o]) .> cutoff[o, d]
        count_active[o, d] = count(active_mask)
        if weights !== nothing
            effective_active[o, d] = sum(weights[w] for w in 1:W if active_mask[w]) * W
        end
    end
    return (cutoff=cutoff, reference_probability=ref_prob, count_active=count_active,
            effective_active=effective_active)
end
