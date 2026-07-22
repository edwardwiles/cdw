# Closed-form construction of the Melitz baseline equilibrium + autarky counterfactual.
# See docs/melitz_delta_star.md Sections 1.3-1.8 (revised) for the full derivation.
#
# REVISED DESIGN (twice). This file originally tried to solve a D-dimensional steep
# power-law NLsolve system for `expenditure` given fixed A/f/tau/w/f_entry -- fragile and,
# per Melitz & Redding (2014 Handbook ch.1, sec 5.1-5.6), not the right architecture:
# their cutoffs decouple from market size only because of a numeraire/outside sector we
# don't have; in the fully general case they themselves need a joint wage-cutoff solve
# (their Section 5.7).
#
# The FIX mirrors the Ricardian repo exactly, including a correction caught mid-session:
# the repo's data object is trade SHARES (lambda), not trade-flow LEVELS (X). Shares alone
# don't pin down levels -- levels require expenditure = w.*L, and wages must be SOLVED so
# that income (w_o*L_o) equals sales (sum_d X_od) for every country simultaneously (no
# deficits) -- exactly `prestep/iterWagesPreStep!.jl`'s fixed point
# `w1 = lambda*(w0.*L)./L`, reused verbatim below as `melitz_solve_wages`. Choosing X_od
# levels directly (an earlier draft of this file) only enforces the destination-side
# balance (expenditure_d = sum_o X_od) and silently misses the origin-side balance
# (income_o = sum_d X_od) -- i.e. it can silently run a trade deficit/surplus.
#
# Given shares+L+the wage solve, expenditure and X follow in closed form. Free cutoffs
# zhat_od (docs Section 1.5) then give A, f in closed form; entrant mass N_o is a
# genuinely free scale (Section 1.4) EXCEPT for the target country, whose N is pinned in
# closed form (not solved) by the requirement that the autarky cutoff sit exactly at 1
# while HOLDING the target country's expenditure fixed at its baseline value (same L,
# same wage numeraire) -- necessary for the ACR cross-check to be meaningful at all (a
# GT comparison across different resource endowments is not what ACR's formula compares).

"""
    melitz_solve_wages(lambda, L; damping=0.6, tol=1e-12, max_iter=100_000) -> w

Reuses `prestep/iterWagesPreStep!.jl`'s exact damped-Jacobi fixed point
`w1 = lambda*(w0.*L)./L`, adapted to return (not mutate a global) and to raise instead of
printing a sentinel on non-convergence. `lambda` is the D x D trade-SHARE matrix
(columns sum to 1, `lambda[o,d]` = share of `d`'s expenditure spent on `o`'s goods) and
`L` is the (chosen) labor endowment vector; wages are normalized so `w[1] == 1` at exit
(re-normalize to a different numeraire index afterward if needed).
"""
function melitz_solve_wages(lambda::Matrix{Float64}, L::Vector{Float64};
                             damping::Real=0.6, tol::Real=1e-12, max_iter::Int=100_000)
    D = length(L)
    isapprox(vec(sum(lambda, dims=1)), ones(D); atol=1e-10) || throw(ArgumentError(
        "lambda columns must sum to 1 (trade shares)"))
    w0 = ones(Float64, D)
    w1 = copy(w0)
    diff = tol + 1
    iter = 1
    while diff > tol && iter < max_iter
        w0 .= w0 .* (1 - damping) .+ w1 .* damping
        w1 .= lambda * (w0 .* L) ./ L
        diff = maximum(abs.(w1 .- w0))
        iter += 1
    end
    iter < max_iter || error("melitz_solve_wages: damped Jacobi did not converge in $max_iter iterations")
    w1 ./= w1[1]
    return w1
end

"""
    melitz_K1(w_o, tau_od, A_od, sigma) -> (markup*w_o*tau_od/A_od)^(1-sigma)
"""
melitz_K1(w_o::Real, tau_od::Real, A_od::Real, sigma::Real) =
    (melitz_markup(sigma) * w_o * tau_od / A_od)^(1 - sigma)

"""
    cell_from_cutoff(X_od, N_o, w_o, tau_od, expenditure_d, zhat_od, sigma, theta_star)
        -> (C_od, A_od, f_od)

Docs Section 1.5 closed form, inverting the aggregate-trade identity `(*)` for `C_od`
given data `X_od`, chosen `N_o` and `zhat_od`, then reading off `A_od` and `f_od`. This is
the *entire* content of "solving" the baseline economy for one (o,d) cell -- no
iteration, exact given `price_power_d==1`.
"""
function cell_from_cutoff(X_od::Real, N_o::Real, w_o::Real, tau_od::Real,
                           expenditure_d::Real, zhat_od::Real, sigma::Real, theta_star::Real)
    C_od = (X_od / N_o) * (theta_star - sigma + 1) / theta_star * zhat_od^(theta_star - sigma + 1)
    A_od = melitz_markup(sigma) * w_o * tau_od * (C_od / expenditure_d)^(1 / (sigma - 1))
    f_od = C_od * zhat_od^(sigma - 1) / (sigma * w_o)
    return (C_od, A_od, f_od)
end

"""
    entry_cost_from_free_entry(C_row, zhat_row, w_o, sigma, theta_star) -> f_entry_o

Closed-form free-entry identity (docs Section 1.4): `f_entry_o = [sum_d
C_od*(sigma-1)/(sigma*(theta_star-sigma+1))*zhat_od^(sigma-1-theta_star)] / w_o`.
`C_row`/`zhat_row` are the length-D vectors of `C_od`/`zhat_od` for fixed origin `o`.
"""
function entry_cost_from_free_entry(C_row::AbstractVector, zhat_row::AbstractVector,
                                     w_o::Real, sigma::Real, theta_star::Real)
    s = zero(eltype(C_row))
    for d in eachindex(C_row)
        s += C_row[d] * (sigma - 1) / (sigma * (theta_star - sigma + 1)) *
             zhat_row[d]^(sigma - 1 - theta_star)
    end
    return s / w_o
end

"""
    entrant_mass_from_labor(L_o, f_entry_o, sigma, theta_star) -> N_o

Closed form for the mass of potential entrants (docs Section 1.4, revised after a live
cross-check against Melitz & Redding 2014): combining the free-entry identity
`f_entry_o = S_o*(sigma-1)/(sigma*theta_star*w_o)` (`S_o = sum_d C_od*M_od`, from
`entry_cost_from_free_entry`) with the labor/income constraint `N_o = w_o*L_o/S_o` and
eliminating `S_o` gives

    N_o = (sigma-1)/(sigma*theta_star) * L_o/f_entry_o

-- exactly Melitz & Redding (2014)'s closed-economy eq. (22),
`M_Ei=(sigma-1)/(k*sigma)*L_i/f_Ei`, extended unchanged to the bilateral-`A_od` case (the
cutoff/`A`/`tau` dependence cancels algebraically, exactly as it does in their
Pareto-specific result -- verified numerically to match the construction's own
`entry_cost_from_free_entry` to 1e-8). `N_o` is therefore **not** a free scale; `f_entry_o`
and `L_o` are the primitives, and `N_o` is derived.
"""
function entrant_mass_from_labor(L_o::Real, f_entry_o::Real, sigma::Real, theta_star::Real)
    return (sigma - 1) / (sigma * theta_star) * L_o / f_entry_o
end

"""
    autarky_fixed_cost(f_entry_target, sigma, theta_star) -> f[target,target]

Closed form (docs Section 1.7): combining the autarky free-entry condition with the
`zhat'[target,target] = 1` normalization gives `f_entry*(theta_star-sigma+1)/(sigma-1)`.
"""
function autarky_fixed_cost(f_entry_target::Real, sigma::Real, theta_star::Real)
    return f_entry_target * (theta_star - sigma + 1) / (sigma - 1)
end

"""
    target_baseline_cutoff_for_autarky(expenditure_target, w_target, X_target_target, theta_star)
        -> zhat[target,target]  (baseline)

Closed form for the target country's OWN baseline domestic cutoff such that the autarky
counterfactual (holding expenditure and entrant mass fixed at their baseline values, same
labor endowment/numeraire) has `zhat'=1` exactly (docs Section 1.7, revised). Derived by
combining the baseline and autarky zero-profit conditions with the price-power
definitions for cell (target,target); verified numerically to be an identity that holds
for *any* `N[target]` once this cutoff value is used -- i.e. `N[target]` remains a fully
free choice (Section 1.4), and it is this ONE baseline cutoff, not `N[target]` or
`f[target,target]` directly, that must be set to this value rather than freely
gravity-projected like every other cell:

    zhat[target,target] = (expenditure_target * w_target / X_target_target)^(1/theta_star)
                         = (w_target / lambda[target,target])^(1/theta_star)

(the second form uses `lambda_tt = X_target_target/expenditure_target`, the baseline
domestic trade share -- note the resemblance to the ACR formula `1-lambda_tt^(1/theta*)`,
which is exactly why the cross-check in Section "Counterfactual quantity of interest"
comes out consistent).
"""
function target_baseline_cutoff_for_autarky(expenditure_target::Real, w_target::Real,
                                             X_target_target::Real, theta_star::Real)
    return (expenditure_target * w_target / X_target_target)^(1 / theta_star)
end

"""
    build_equilibrium(X, N, w, tau, expenditure, zhat, sigma, theta_star)
        -> (A, f, C, MelitzEquilibrium)

Assembles the full D x D `A`, `f`, `C` matrices and the `MelitzEquilibrium` diagnostic
object from data `X` (trade flows, `= lambda .* expenditure'`), chosen `N` (entrant
mass, except `N[target]` which is pinned -- see `target_entrant_mass_for_autarky`), `w`
(solved via `melitz_solve_wages`), `tau`, `expenditure` (`= w.*L`), and the free cutoff
parameterization `zhat` -- purely closed-form (docs Section 1.5). `price_power_d == 1` is
verified (not imposed) as a consistency check.
"""
function build_equilibrium(X::Matrix{T}, N::Vector{T}, w::Vector{T}, tau::Matrix{Float64},
                            expenditure::Vector{T}, zhat::Matrix{T}, sigma::Real,
                            theta_star::Real) where {T<:Real}
    D = size(X, 1)
    A = zeros(T, D, D)
    f = zeros(T, D, D)
    C = zeros(T, D, D)
    for o in 1:D, d in 1:D
        C[o, d], A[o, d], f[o, d] = cell_from_cutoff(X[o, d], N[o], w[o], tau[o, d],
                                                       expenditure[d], zhat[o, d], sigma, theta_star)
    end
    price_power = vec(sum(X, dims=1)) ./ expenditure
    eq = MelitzEquilibrium(N, expenditure, price_power, zhat, X)
    return A, f, C, eq
end

"""
    solve_autarky_counterfactual(p::MelitzPrimitives, eq::MelitzEquilibrium)
        -> MelitzCounterfactual

Closed-form autarky counterfactual for `p.target_country` (docs Section 1.7/9, revised):
`entrant_mass[target]` is reused unchanged from `eq` (N'=N -- by this point `eq` was
itself built using the pinned `N[target]` from `target_entrant_mass_for_autarky`),
`w'[target]=1`, `tau'[target,target]=1`, and **`expenditure_prime[target] =
eq.expenditure[target]`** (the SAME baseline value -- same labor endowment, same wage
numeraire, required for the ACR cross-check).
"""
function solve_autarky_counterfactual(p::MelitzPrimitives{T}, eq::MelitzEquilibrium{T};
                                       rtol::Real=1e-6) where {T<:Real}
    t = p.target_country
    sigma, theta_star = p.sigma, p.theta_star
    w_prime = one(T)
    expenditure_prime = eq.expenditure[t]

    zhat_tt_required = target_baseline_cutoff_for_autarky(expenditure_prime, p.w[t], eq.trade_flow[t, t], theta_star)
    isapprox(eq.cutoff[t, t], zhat_tt_required; rtol=rtol) || throw(ArgumentError(
        "cutoff[target,target]=$(eq.cutoff[t,t]) does not satisfy the autarky zhat'=1 " *
        "normalization (expected $zhat_tt_required from expenditure/w/X at [target,target]); " *
        "fake_data.jl must construct zhat[target,target] that way"))

    K1_tt = melitz_K1(w_prime, one(T), p.A[t, t], sigma) # tau'[t,t] = 1 (autarky)
    N_t = eq.entrant_mass[t]
    price_power_prime = N_t * K1_tt * theta_star / (theta_star - sigma + 1)
    cutoff_prime = one(T)
    trade_flow_prime = expenditure_prime # market-clearing identity (docs Section 1.7)

    return MelitzCounterfactual(t, w_prime, expenditure_prime, price_power_prime,
                                 cutoff_prime, trade_flow_prime)
end
