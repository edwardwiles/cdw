# Direct Pareto F* benchmark parameter solver.
# See docs/melitz_delta_star.md Section "Solving the F* benchmark parameters".
#
# Mirrors the Ricardian repo's prestep/master_prestep.jl role: given OBSERVED data
# (bilateral trade flows X_data, labor endowments L, trade costs tau, entry costs
# f_entry), solve directly (closed form, no KNITRO, no Monte Carlo) for the structural
# parameter vector under F*. Unlike the forward construction in fake_data.jl (which also
# CHOOSES/generates the data), this function treats X_data as given: wages and entrant
# mass follow immediately from the accounting identities (no damped-Jacobi fixed point
# needed once trade-flow LEVELS, not just shares, are already observed), and A/f follow
# from the same closed-form cell_from_cutoff machinery used everywhere else in this
# module. A choice of cutoff parameterization `zhat` must be supplied (or generated) for
# the non-identified direction (docs Section 1.6/1.7) -- solving from DIFFERENT `zhat`
# choices on the SAME data is the intended way to demonstrate observational equivalence
# (docs Section 9, addendum Section 1.3), not a solver failure.

"""
    solve_fstar(D, sigma, theta_star, target_country, tau, L, X_data, f_entry, zhat)
        -> MelitzFStarResult

Direct F* solver. Given observed/calibrated `(tau, L, X_data, f_entry)` and a chosen
cutoff parameterization `zhat` (`zhat[target,target]` is OVERRIDDEN to the value required
by the autarky `zhat'=1` normalization, docs Section 1.8 -- solving "enforces the focal
autarky conditions" per the brief, it does not merely check them):

1. `w_o = (Σ_d X_data[o,d]) / L_o` -- wages follow directly from observed trade-flow
   levels and labor endowments (income = sales, no fixed point needed once levels, not
   just shares, are already observed).
2. `expenditure_d = Σ_o X_data[o,d]`.
3. `N_o = entrant_mass_from_labor(L_o, f_entry_o, ...)` (closed form, docs Section 1.4).
4. `zhat[target,target]` is set to `target_baseline_cutoff_for_autarky(...)`.
5. `A, f, C` from `build_equilibrium` (closed form).
6. Residuals, gravity restrictions, and solver status are computed and packaged.

Returns a `MelitzFStarResult` (never claims elementwise recovery of a unique "true" `A`/
`f` -- see docs Section 1.7/9).
"""
function solve_fstar(D::Int, sigma::Float64, theta_star::Float64, target_country::Int,
                      tau::Matrix{Float64}, L::Vector{Float64}, X_data::Matrix{Float64},
                      f_entry::Vector{Float64}, zhat::Matrix{Float64})
    size(tau) == (D, D) && size(X_data) == (D, D) && size(zhat) == (D, D) ||
        throw(ArgumentError("tau, X_data, zhat must all be D x D"))
    length(L) == D && length(f_entry) == D || throw(ArgumentError("L, f_entry must have length D"))

    w = vec(sum(X_data, dims=2)) ./ L
    w ./= w[target_country]
    # income was defined via X_data BEFORE renormalizing w; renormalizing w (a common
    # rescaling) leaves relative prices, shares, and every real quantity unchanged, but
    # requires expenditure to be recomputed as w.*L under the NEW numeraire for internal
    # consistency with build_equilibrium's convention (expenditure_d = sum_o X_data[o,d]
    # is invariant to the wage numeraire choice; kept as-is).
    expenditure = vec(sum(X_data, dims=1))

    N = [entrant_mass_from_labor(L[o], f_entry[o], sigma, theta_star) for o in 1:D]

    zhat_local = copy(zhat)
    zhat_tt_required = target_baseline_cutoff_for_autarky(expenditure[target_country],
                                                            w[target_country],
                                                            X_data[target_country, target_country],
                                                            theta_star)
    zhat_local[target_country, target_country] = zhat_tt_required

    A, f, C, eq = build_equilibrium(X_data, N, w, tau, expenditure, zhat_local, sigma, theta_star)

    f_entry_check = [entry_cost_from_free_entry(C[o, :], zhat_local[o, :], w[o], sigma, theta_star) for o in 1:D]
    max_entry_residual = maximum(abs.(f_entry_check .- f_entry))

    primitives = MelitzPrimitives(D, sigma, theta_star, target_country, tau, w, A, f, f_entry)
    converged = true
    status = :closed_form
    counterfactual = try
        solve_autarky_counterfactual(primitives, eq)
    catch e
        converged = false
        status = :autarky_inconsistent
        MelitzCounterfactual(target_country, one(eltype(w)), expenditure[target_country],
                              NaN, NaN, NaN)
    end

    trade_flow_model = [N[o] * C[o, d] * pareto_tail_power_mean(zhat_local[o, d], sigma, theta_star)
                         for o in 1:D, d in 1:D]
    max_trade_residual = maximum(abs.(trade_flow_model .- X_data))

    gravity_residual_A, gravity_residual_f = gravity_residuals(primitives)

    return MelitzFStarResult(primitives, eq, counterfactual, max_trade_residual,
                              max_entry_residual, gravity_residual_A, gravity_residual_f,
                              converged, status)
end
