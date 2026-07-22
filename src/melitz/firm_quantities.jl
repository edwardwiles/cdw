# Shared firm-level Melitz calculations. One routine, used identically for baseline and
# counterfactual, evaluated at every reference draw -- never a separate manual
# counterfactual formula and never evaluated at a single draw index (see
# docs/melitz_legacy_audit.md "Reject" section).
#
# Convention: markup = sigma/(sigma-1) always; no rho, no hybrid branch, no
# perfect-competition fallback. sigma > 1 is enforced by MelitzPrimitives's constructor.

"""
    melitz_markup(sigma) -> sigma/(sigma-1)
"""
melitz_markup(sigma::Real) = sigma / (sigma - 1)

"""
    marginal_cost(w_o, tau_od, A_od, z) -> w_o * tau_od / (A_od * z)

Paper convention: higher `A_od` => lower marginal cost (docs Section 2).
"""
marginal_cost(w_o::Real, tau_od::Real, A_od::Real, z::Real) = w_o * tau_od / (A_od * z)

"""
    melitz_price(w_o, tau_od, A_od, sigma, z) -> markup * marginal_cost
"""
function melitz_price(w_o::Real, tau_od::Real, A_od::Real, sigma::Real, z::Real)
    return melitz_markup(sigma) * marginal_cost(w_o, tau_od, A_od, z)
end

"""
    unconstrained_revenue(price, sigma, expenditure_d, price_power_d)
        -> expenditure_d * price^(1-sigma) / price_power_d

Revenue a firm at this price *would* earn ignoring the fixed-cost participation decision.
"""
function unconstrained_revenue(price::Real, sigma::Real, expenditure_d::Real, price_power_d::Real)
    return expenditure_d * price^(1 - sigma) / price_power_d
end

"""
    MelitzFirmResult

Per-draw firm outcome for one (o,d,z) triple: `active` is the entry/participation
decision (`operating_profit > 0`, the convention used consistently everywhere -- the
equality case has probability zero under the continuous Pareto); `realized_revenue` and
`realized_operating_profit` are zero when `!active`.
"""
struct MelitzFirmResult{T<:Real}
    price::T
    unconstrained_revenue::T
    operating_profit::T
    active::Bool
    realized_revenue::T
    realized_operating_profit::T
end

"""
    melitz_firm(w_o, tau_od, A_od, f_od, sigma, expenditure_d, price_power_d, z)
        -> MelitzFirmResult

The single shared per-draw firm routine (docs Section 6). Computes price, unconstrained
revenue, operating profit, the active/participation decision, and realized (post-decision)
revenue/profit -- used identically for every (o,d) cell, every draw, in both the baseline
and the counterfactual.
"""
function melitz_firm(w_o::Real, tau_od::Real, A_od::Real, f_od::Real, sigma::Real,
                      expenditure_d::Real, price_power_d::Real, z::Real)
    price = melitz_price(w_o, tau_od, A_od, sigma, z)
    rev = unconstrained_revenue(price, sigma, expenditure_d, price_power_d)
    profit = rev / sigma - w_o * f_od
    active = profit > 0
    realized_revenue = active ? rev : zero(rev)
    realized_profit = active ? profit : zero(profit)
    return MelitzFirmResult(price, rev, profit, active, realized_revenue, realized_profit)
end

"""
    melitz_cutoff(w_o, f_od, sigma, C_od) -> zhat_od

Zero-profit productivity cutoff solving `C_od * zhat^(sigma-1) = sigma * w_o * f_od`
(docs Section 1.3), where `C_od = expenditure_d * (markup*w_o*tau_od/A_od)^(1-sigma)`.
Provided as a direct formula (not implied by simulation) for use in the equilibrium solver
and the synthetic-data generator.
"""
function melitz_cutoff(w_o::Real, f_od::Real, sigma::Real, C_od::Real)
    return (sigma * w_o * f_od / C_od)^(1 / (sigma - 1))
end

"""
    melitz_C(w_o, tau_od, A_od, sigma, expenditure_d) -> C_od

`C_od = expenditure_d * (markup*w_o*tau_od/A_od)^(1-sigma)`, so that
`realized_revenue_od(z) = C_od * z^(sigma-1) * active` under the `price_power_d == 1`
normalization (docs Section 1.2-1.3).
"""
function melitz_C(w_o::Real, tau_od::Real, A_od::Real, sigma::Real, expenditure_d::Real)
    return expenditure_d * (melitz_markup(sigma) * w_o * tau_od / A_od)^(1 - sigma)
end
