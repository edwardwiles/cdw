# Closed-form construction helpers, the origin-scale normalization, gravity-pivot
# elimination, and general-F LFD-based recovery/ex-post checks for the full-D Melitz
# minimal-moment benchmark. See docs/melitz_delta_star.md for the full derivation.
#
# Three groups of functions live in this file, in this order:
#   1. General-purpose closed-form helpers usable both for benchmark construction AND
#      (where noted) inside the active general-F path: melitz_solve_wages, melitz_K1,
#      gravity_residuals, cell_from_cutoff, build_equilibrium,
#      normalize_baseline_entrant_mass, derive_fjj_from_autarky_cutoff, the gravity-pivot
#      machinery.
#   2. General-F LFD-based recovery (main prompt Section 7) and the ex-post equilibrium
#      check (Section 9): recover_entry_costs_from_lfd, recover_focal_autarky_entry_cost,
#      recover_N_prime_market_clearing, recover_N_prime_price_index,
#      check_profiled_melitz_equilibrium.
#   3. PARETO-ONLY closed-form benchmark diagnostics (addendum Section 4), prefixed
#      `pareto_`, NEVER called from the active moment/delta_star path: they encode the
#      superseded f_entry-primitive / N-derived / N'=N closure and exist only so the new
#      formulation can be cross-checked against the old one at the exact Pareto benchmark
#      (addendum Section 15).

using LinearAlgebra: dot

"""
    melitz_solve_wages(lambda, L; damping=0.6, tol=1e-12, max_iter=100_000) -> w

Reuses `prestep/iterWagesPreStep!.jl`'s exact damped-Jacobi fixed point
`w1 = lambda*(w0.*L)./L`, adapted to return (not mutate a global) and to raise instead of
printing a sentinel on non-convergence. `lambda` is the D x D trade-SHARE matrix
(columns sum to 1) and `L` is the labor endowment vector; wages are normalized so
`w[1] == 1` at exit (re-normalize to a different numeraire index afterward if needed).
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
    population_X(w, L, tau, A, f, sigma, theta_star) -> (X, q, E)

Closed-form POPULATION (not finite-sample) Melitz-Pareto bilateral trade flows, addendum
Section 2: `E_d = w_d*L_d` (balanced trade, no deficit, iceberg closure -- addendum
Section 6, no `1/tau_od` tariff-revenue term); `C_od = E_d*melitz_K1(w_o,tau_od,A_od,sigma)`;
`q_od = melitz_cutoff(w_o,f_od,sigma,C_od)` (zero-profit cutoff); `X_od = C_od *
pareto_tail_power_mean(q_od,sigma,theta_star)` (the population aggregate-revenue integral,
`melitz_C`'s per-firm formula times the Pareto tail expectation of `z^(sigma-1)`, exactly
`theta_star/(theta_star-sigma+1)*q_od^(sigma-1-theta_star)`). `X`, not shares -- callers
needing shares divide by `sum(X,dims=1)`.
"""
function population_X(w::AbstractVector, L::AbstractVector, tau::AbstractMatrix,
                       A::AbstractMatrix, f::AbstractMatrix, sigma::Real, theta_star::Real)
    D = length(w)
    E = w .* L
    X = zeros(eltype(w), D, D)
    q = zeros(eltype(w), D, D)
    @inbounds for o in 1:D, d in 1:D
        K1 = melitz_K1(w[o], tau[o, d], A[o, d], sigma)
        C = E[d] * K1
        q[o, d] = melitz_cutoff(w[o], f[o, d], sigma, C)
        X[o, d] = C * pareto_tail_power_mean(q[o, d], sigma, theta_star)
    end
    return X, q, E
end

"""
    melitz_solve_wages_ge(L, tau, A0, f, sigma, theta_star; damping=0.1, tol=1e-13, max_iter=20_000)
        -> (w, A, iters)

General-equilibrium wage solve for the POPULATION-Pareto construction (addendum Section 2),
replacing the superseded exact-sample-correction pathway's data-driven `melitz_solve_wages`.
Trivial extension of the SAME damped-Jacobi income-redistribution iteration (`w1 =
lambda*(w0.*L)./L`) already used there and in the mature Ricardian implementation -- the
only difference is `lambda`/`X` is RECOMPUTED from the Melitz-Pareto closed form
(`population_X`) at the current wage guess each iterate, rather than taken as fixed input
data. No NLsolve/autodiff needed (tried first, unnecessarily complex for what is, at heart,
the same trivial fixed-point iteration as the Ricardian model -- just with an extra
per-destination rescaling step, below).

At every iterate, `A`'s columns are rescaled by `s_d = (E_d/colsum_d(X))^(1/theta_star)` to
force the baseline price-index normalization `gamma_d==1` EXACTLY (not just as a diagnostic
check): `withinTransform` (Gate A5's gravity transform, `gravity_residuals`) is invariant
to per-destination (column) additive shifts in log-space by construction (it explicitly
subtracts the column/destination mean, then adds back the grand mean -- any constant added
to an entire column of `log z` shifts that column's mean by the same constant, which then
cancels exactly), so this column rescaling NEVER disturbs the A-gravity restriction,
however it is chosen (this invariance held equally well for the ORIGINAL `doubleDiff`
choice -- doubleDiff's own anchored-difference formula also cancels a per-column shift --
so this part of the construction did not need to change when the gravity transform did).
`s_d` is
closed-form because rescaling A's column `d` by `s_d` scales `X[:,d]` (every origin) by the
SAME factor `s_d^theta_star` (verified algebraically: `K1 ~ A^(sigma-1)`, `q ~ s_d^-1`, `X =
C*tailmean(q) ~ s_d^(sigma-1) * s_d^(theta_star-sigma+1) = s_d^theta_star`), so no separate
solve is needed for `s_d` -- just the closed-form ratio.

Requires damping around 0.1 (NOT the data-driven solver's 0.6 default) to converge: the
Melitz-Pareto share elasticity in wages is effectively `theta_star` (~6.8 in this repo's
benchmarks), much steeper than a plain CES gravity elasticity `sigma-1`, so the naive
damped-Jacobi map is not a contraction at large damping (found live: `damping=0.5` diverges/
oscillates; `damping=0.1` converges geometrically in ~150-250 iterations).

IMPORTANT: unlike `melitz_solve_wages` (which solves against EXOGENOUS DATA shares and has a
genuine nominal wage-scale indeterminacy, resolved by normalizing `w[1]=1`), this GE solve
ties absolute wage levels to the REAL primitives (`f_od`/`A_od` are labor-value/productivity
quantities, not nominal ones) -- there is NO free wage-scale normalization here, and the
returned `w` must NOT be renormalized post-hoc (verified live: forcing `w[j]=1` after
convergence broke the fixed point, factor-market residual jumping from ~1e-12 to ~1).
"""
function melitz_solve_wages_ge(L::AbstractVector, tau::AbstractMatrix, A0::AbstractMatrix,
                                f::AbstractMatrix, sigma::Real, theta_star::Real;
                                damping::Real=0.1, tol::Real=1e-13, max_iter::Int=20_000)
    D = length(L)
    w0 = ones(Float64, D)
    w1 = copy(w0)
    A = copy(A0)
    diff = tol + 1
    iter = 1
    while diff > tol && iter < max_iter
        w0 .= w0 .* (1 - damping) .+ w1 .* damping
        X, _, E = population_X(w0, L, tau, A0, f, sigma, theta_star)
        s = (E ./ vec(sum(X, dims=1))) .^ (1 / theta_star)
        for d in 1:D
            A[:, d] .= A0[:, d] .* s[d]
        end
        X, _, _ = population_X(w0, L, tau, A, f, sigma, theta_star)
        w1 .= vec(sum(X, dims=2)) ./ L
        diff = maximum(abs.(w1 .- w0))
        iter += 1
    end
    iter < max_iter || error("melitz_solve_wages_ge: damped Jacobi did not converge in $max_iter iterations (diff=$diff)")
    return w1, A, iter
end

"""
    gravity_residuals(p::MelitzPrimitives) -> (residual_A, residual_f)

The two F-independent gravity restrictions, evaluated directly from `(A, f, tau)` --
never a column of `G`, never duplicated as both a moment and an outer constraint.

Gate A5 (docs/melitz_delta_star.md): uses `misc/doubleDiff.jl`'s `withinTransform`
(the symmetric two-way origin+destination fixed-effects "within" residual,
`z_od - mean_o(z) - mean_d(z) + grand_mean(z)`), NOT `doubleDiff` (the ORIGINAL choice
here, an asymmetric "anchored" contrast differencing against a FIXED row-1/column-2
reference cell). By Frisch-Waugh-Lovell, `withinTransform` reproduces EXACTLY the
coefficient of an OLS gravity regression with origin+destination fixed effects -- this is
the SAME restriction `moments/newGravityMoment!.jl`'s `UoModel==1` branch enforces (the
UNIVERSAL branch: every production run config in this repo sets `UoModel=1`,
`GravityMomentFirstApproach=0`), and matches `full_aod_diag/d4_exact/gravity_elimination.jl`'s
`gravity_value` (also `withinTransform`-based, independently cross-validated against
`newGravityMoment!` in `full_aod_diag/test_free_param_and_gravity.jl`). `doubleDiff` was
verified LIVE (this repo's own pre-existing `gravity_check.jl`, re-run for Gate A5) to
NOT reproduce this coefficient: `max|b_DDcell - b_FE| = 2.31` over 2000 random trials
(machine-precision for `withinTransform`, `1.8e-15`), and the two transforms' restriction
values can even have OPPOSITE SIGNS on the same data -- `moments/newGravityMoment!.jl`'s
own comment independently confirms this ("the old cell double-difference did not
[reproduce the OLS-two-way-FE coefficient]"). `doubleDiff` and `withinTransform` DO share
the same null space (both vanish exactly on any log-additive-FE `A`/`f`, verified), which
is why using `doubleDiff` never triggered a visible failure in this module's own
self-constructed fixtures/tests -- but as a general RESTRICTION on genuine (non-additive)
bilateral structure, it is a different, non-canonical choice. `withinTransform`, like
`doubleDiff`, already takes `log()` of its argument internally, so it is applied directly
to the LEVELS `tau`/`A`/`f` here (no separate `log.()` wrapping).
"""
function gravity_residuals(p::MelitzPrimitives)
    T = withinTransform(p.tau)
    residual_A = sum(T .* withinTransform(p.A))
    residual_f = sum(T .* withinTransform(p.f))
    return residual_A, residual_f
end

"""
    cell_from_cutoff(X_od, N_o, w_o, tau_od, expenditure_d, zhat_od, sigma, theta_star)
        -> (C_od, A_od, f_od)

Closed-form Pareto inversion of the aggregate-trade identity for `C_od` given data `X_od`,
a chosen `N_o` and `zhat_od`, then reading off `A_od` and `f_od`. PARETO-BENCHMARK
CONSTRUCTION HELPER ONLY (used by `fake_data.jl`/`fstar_solver.jl` to build/solve a
fixture) -- never called from the active general-F moment/delta_star path.
"""
function cell_from_cutoff(X_od::Real, N_o::Real, w_o::Real, tau_od::Real,
                           expenditure_d::Real, zhat_od::Real, sigma::Real, theta_star::Real)
    C_od = (X_od / N_o) * (theta_star - sigma + 1) / theta_star * zhat_od^(theta_star - sigma + 1)
    A_od = melitz_markup(sigma) * w_o * tau_od * (C_od / expenditure_d)^(1 / (sigma - 1))
    f_od = C_od * zhat_od^(sigma - 1) / (sigma * w_o)
    return (C_od, A_od, f_od)
end

"""
    build_equilibrium(X, N, w, tau, expenditure, zhat, sigma, theta_star)
        -> (A, f, C, MelitzEquilibrium)

Assembles the full D x D `A`, `f`, `C` matrices and the `MelitzEquilibrium` diagnostic
object from data `X` (`= lambda .* expenditure'`), chosen `N` (PARETO-BENCHMARK
construction only -- pass `ones(D)` for the active `N_o=1` normalization), `w`, `tau`,
`expenditure`, and the free cutoff parameterization `zhat`. `price_power_d == 1` is
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
    eq = MelitzEquilibrium(expenditure, price_power, zhat, X)
    return A, f, C, eq
end

"""
    melitz_baseline_cutoff(A, f, w, tau, expenditure, sigma) -> D x D matrix

Session prompt Section 1.2/1.3: the deterministic (data/parameter-only, NO Monte Carlo)
zero-profit baseline cutoff `zhat_od = melitz_cutoff(w_o, f_od, sigma, C_od)`, `C_od =
melitz_C(w_o, tau_od, A_od, sigma, expenditure_d)`. Recomputable at ANY `(A, f)` -- NEVER
read from a fixed benchmark object at a displaced outer point (`melitz_outer_state`,
delta_star.jl, is the one caller that matters for this: it calls this function fresh on
every evaluation instead of reusing a stored `ctx.benchmark_cutoff`). `w`, `tau`,
`expenditure` are treated as fixed DATA throughout the outer search (never re-solved from
`theta`), matching how the trade-share moments themselves treat `expenditure` as fixed
(`moments.jl`'s `lambda_od = X_data[o,d]/eq.expenditure[d]`) -- only `A`/`f` (and, through
`f[j,j]`, `gamma_prime_target`) vary with the outer point. Fully type-generic (parametric
`T = promote_type(eltype(A), eltype(f))`) so it composes with ForwardDiff Duals for the
Section 1.3 Jacobian.
"""
function melitz_baseline_cutoff(A::AbstractMatrix, f::AbstractMatrix, w::AbstractVector,
                                 tau::AbstractMatrix, expenditure::AbstractVector, sigma::Real)
    D = size(A, 1)
    T = promote_type(eltype(A), eltype(f))
    zhat = zeros(T, D, D)
    @inbounds for o in 1:D, d in 1:D
        C_od = melitz_C(w[o], tau[o, d], A[o, d], sigma, expenditure[d])
        zhat[o, d] = melitz_cutoff(w[o], f[o, d], sigma, C_od)
    end
    return zhat
end

"""
    melitz_deterministic_cutoff_constraints(zhat) -> (g_domestic, g_export)

Session prompt Section 1.3: the MINIMAL deterministic feasibility inequality system --
`log zhat[o,o] >= 0` for every `o` (`g_domestic`, length `D`) and `log zhat[o,d] - log
zhat[o,o] >= 0` for every `o` and `d != o` (`g_export`, length `D*(D-1)`, export-selection).
Together these imply every bilateral cutoff is `>= 1` (`log zhat[o,d] = log zhat[o,o] +
(log zhat[o,d]-log zhat[o,o]) >= 0 + 0 = 0`), so imposing a SEPARATE, redundant `D^2`
raw-cutoff-`>=1` system on top would add nothing (session prompt's own instruction).
Feasible iff every entry of both returned vectors is `>= 0`; the more negative a coordinate,
the further into infeasibility.
"""
function melitz_deterministic_cutoff_constraints(zhat::AbstractMatrix)
    D = size(zhat, 1)
    T = eltype(zhat)
    g_domestic = zeros(T, D)
    g_export = zeros(T, D * (D - 1))
    k = 0
    @inbounds for o in 1:D
        g_domestic[o] = log(zhat[o, o])
        for d in 1:D
            d == o && continue
            k += 1
            g_export[k] = log(zhat[o, d]) - log(zhat[o, o])
        end
    end
    return g_domestic, g_export
end

"""
    normalize_baseline_entrant_mass(N, A, f, f_entry, sigma; N_prime=nothing)
        -> (A_new, f_new, f_entry_new, N_prime_new)

Origin-specific scale normalization (addendum Section 3): converts an arbitrary feasible
baseline economy with entrant mass `N_o` (old) to the active `N_o == 1` normalization,
transforming `A_od -> N_o^(1/(sigma-1)) * A_od`, `f_od -> N_o * f_od`,
`f_entry_o -> N_o * f_entry_o` (diagnostic only post-refactor, but transformed for
consistency/comparison), and any per-origin counterfactual mass `N_prime_o -> N_prime_o /
N_o`. Leaves every baseline trade share, cutoff, aggregate bilateral flow, baseline
`gamma_d`, both gravity restrictions, free-entry feasibility, and gains-from-trade
UNCHANGED -- verified algebraically (X_od = N_o*C_od*M_od and the zero-profit cutoff
equation `C_od*zhat^(sigma-1) = sigma*w_o*f_od` are both invariant to this rescaling) and
covered by an explicit invariance test.
"""
function normalize_baseline_entrant_mass(N::AbstractVector{T}, A::AbstractMatrix{T},
                                          f::AbstractMatrix{T}, f_entry::AbstractVector{T},
                                          sigma::Real;
                                          N_prime::Union{Nothing,AbstractVector{T}}=nothing) where {T<:Real}
    D = length(N)
    size(A) == (D, D) && size(f) == (D, D) || throw(ArgumentError("A, f must be D x D"))
    A_new = similar(A)
    f_new = similar(f)
    @inbounds for o in 1:D, d in 1:D
        A_new[o, d] = N[o]^(1 / (sigma - 1)) * A[o, d]
        f_new[o, d] = N[o] * f[o, d]
    end
    f_entry_new = N .* f_entry
    N_prime_new = N_prime === nothing ? nothing : N_prime ./ N
    return A_new, f_new, f_entry_new, N_prime_new
end

"""
    derive_fjj_from_autarky_cutoff(gamma_prime_j, w_prime_j, tau_prime_jj, A_jj,
                                    expenditure_prime_j, sigma) -> f_jj

Main prompt Section 3: the exact parameter transformation (not a moment) deriving the
focal domestic fixed cost from the autarky cutoff-at-one normalization. At
`zhat_prime[j,j]=1`, the productivity-one firm must earn exactly zero operating profit:
`expenditure_prime_j * (markup*w_prime_j*tau_prime_jj/A_jj)^(1-sigma) / (sigma*gamma_prime_j)
== w_prime_j * f_jj`. `tau_prime_jj` is always 1 (autarky, no iceberg cost against
oneself) -- kept as an explicit argument only for testability.
"""
function derive_fjj_from_autarky_cutoff(gamma_prime_j::Real, w_prime_j::Real,
                                         tau_prime_jj::Real, A_jj::Real,
                                         expenditure_prime_j::Real, sigma::Real)
    markup = melitz_markup(sigma)
    p_prime_at_one = markup * w_prime_j * tau_prime_jj / A_jj
    return expenditure_prime_j * p_prime_at_one^(1 - sigma) / (sigma * w_prime_j * gamma_prime_j)
end

"""
    population_autarky_profit_integral(f_jj, sigma, theta_star) -> E[Pi_autarky_j]

Closed-form POPULATION expectation of the focal country's autarky operating profit,
`E[revenue'_jj(z)*1{z>=1}]/sigma - w_prime_j*f_jj*Pr(z>=1)`, using that the autarky cutoff
is EXACTLY 1 (`derive_fjj_from_autarky_cutoff`'s own construction, so `Pr(active)=1`
exactly and every reference draw participates) and `w_prime_j=1` (numeraire). Reduces
algebraically to `f_jj*(pareto_tail_power_mean(1,sigma,theta_star) - 1)` -- verified via
`C'_jj = sigma*w_prime_j*f_jj` (immediate from `derive_fjj_from_autarky_cutoff`'s own
defining equation), so `E[Pi_autarky_j] = C'_jj*tailmean(1)/sigma - w_prime_j*f_jj =
w_prime_j*f_jj*(tailmean(1)-1)`.
"""
population_autarky_profit_integral(f_jj::Real, sigma::Real, theta_star::Real) =
    f_jj * (pareto_tail_power_mean(1.0, sigma, theta_star) - 1)

"""
    population_baseline_focal_profit_integral(p, X, q) -> E[Pi_baseline_j]

Closed-form POPULATION expectation of the focal country's TOTAL baseline operating
profit (summed over all destinations `d`), `sum_d [X[j,d]/sigma -
w[j]*f[j,d]*pareto_tail_prob(q[j,d],theta_star)]`, using the already-computed population
trade flows `X` and cutoffs `q` (`population_X`) -- `X[j,d]/sigma` is the population
revenue integral (`melitz_C(...)*pareto_tail_power_mean(...)`, already folded into `X`),
`pareto_tail_prob(q[j,d],theta_star)` is the exact participation probability.
"""
function population_baseline_focal_profit_integral(p::MelitzPrimitives, X::AbstractMatrix, q::AbstractMatrix)
    j = p.target_country
    return sum(X[j, d] / p.sigma - p.w[j] * p.f[j, d] * pareto_tail_prob(q[j, d], p.theta_star) for d in 1:p.D)
end

"""
    melitz_gains_from_trade(p::MelitzPrimitives, cf::MelitzCounterfactual) -> GT_j

Main prompt Section A1: the exact gains-from-trade formula, valid under ARBITRARY
(not-necessarily-equal) baseline/autarky nominal wage normalizations,
`GT_j = 1 - (w_prime_j/w_j) * gamma_prime_j^(1/(sigma-1))`.

Derivation: with baseline `gamma_j==1` (universal normalization, `price_power_d==1`),
`price_power_d` is `P_d^{-(sigma-1)}` for the CES price index `P_d` (`unconstrained_revenue
= expenditure_d*price^(1-sigma)/price_power_d`, i.e. `price_power_d` scales like
`P_d^{-(sigma-1)}`). So baseline `P_j = 1` in these units, and autarky `P'_j =
gamma_prime_j^{-1/(sigma-1)}`. Real wage is `w/P`, so
`GT_j = 1 - (w'_j/P'_j)/(w_j/P_j) = 1 - (w'_j/w_j)*(P_j/P'_j) = 1 - (w'_j/w_j) *
gamma_prime_j^(1/(sigma-1))`.

The SIMPLIFIED formula `1 - gamma_prime_j^(1/(sigma-1))` (previously reported in
docs/melitz_delta_star.md Section 13.4 and scripts/run_melitz_delta_star_fake.jl) silently
assumes `w_prime_j/w_j == 1`. This does NOT hold under the active population-Pareto
construction: `w[target_country]` is a genuine GENERAL-EQUILIBRIUM output
(`melitz_solve_wages_ge`, NOT renormalized to 1 -- see that function's docstring), while
`w_prime_j == 1` always (the autarky numeraire). Omitting the ratio does not merely
rescale `GT_j` by a constant -- it silently redefines what "gains from trade" means
whenever `w_j != 1`, and a negative value produced by the simplified formula is NOT by
itself evidence of negative gains from trade (the sign confound this function fixes).

This module deliberately does NOT renormalize the fixture to a common `w_j==1` numeraire
(the alternative route main prompt Section A1 offers): `melitz_solve_wages_ge`'s own
docstring documents, from a LIVE finding, that forcing `w[j]=1` post-hoc after that GE
solve breaks its fixed point (factor-market residual jumps from ~1e-12 to ~1) --
re-deriving a numeraire-invariant rescaling of the ENTIRE population-Pareto GE
construction (A, w, expenditure, X all together) to avoid that live failure is a separate,
higher-risk undertaking than fixing the welfare formula itself, and is not required for
correctness: the wage-ratio formula above is valid for ANY `w_j`, so keeping unequal
wages and using it everywhere (main prompt Section A1's explicitly offered alternative)
is both sufficient and lower-risk given `melitz_solve_wages_ge` is freshly bug-fixed
([[melitz-population-pareto-bugfix-2026-07-22]]).
"""
melitz_gains_from_trade(p::MelitzPrimitives, cf::MelitzCounterfactual) =
    1 - (cf.w_prime / p.w[p.target_country]) * p.gamma_prime_target^(1 / (p.sigma - 1))

"""
    acr_gains_from_trade(p::MelitzPrimitives, eq::MelitzEquilibrium) -> (lambda_jj, GT_ACR)

Main prompt Section A2: the ACR/Chaney sufficient-statistic cross-check,
`lambda_jj = X[j,j]/expenditure[j]`, `GT_ACR = 1 - lambda_jj^(1/theta_star)`. This is a
REAL (nominal-normalization-invariant) quantity -- it uses only the domestic trade share,
never wages directly -- so, unlike `melitz_gains_from_trade`, it needs no wage-ratio
correction. Population-level agreement between `GT_ACR` and `melitz_gains_from_trade`
(to approximately machine precision, since both are population closed forms at the same
benchmark) is the required cross-check (main prompt Section A2); a mismatch is a blocking
economic error, not a documented discrepancy.
"""
function acr_gains_from_trade(p::MelitzPrimitives, eq::MelitzEquilibrium)
    j = p.target_country
    lambda_jj = eq.trade_flow[j, j] / eq.expenditure[j]
    GT_ACR = 1 - lambda_jj^(1 / p.theta_star)
    return lambda_jj, GT_ACR
end

# ============================================================================
# 2026-07-26 production-closure session (governing prompt Phase 3): authoritative,
# explicitly-named welfare metrics. A prior session's own reporting scripts printed the raw
# outer-loop coordinate `g = theta_free[1] = log(gamma_prime_j)` under a field/label implying
# it was the gains-from-trade `kappa`/`GT_j` (docs/melitz_production_fast_backend_2026-07-26.md
# Section 5.5, caught live by direct user questioning: "kappa is meant to live between 0 and
# 0.113" -- correctly flagged as impossible for the raw `g` value, which is unbounded and can
# be negative with no welfare interpretation on its own). This was a REPORTING error, not an
# economics error (the underlying `melitz_gains_from_trade`/`acr_gains_from_trade` formulas
# above were always correct) -- but the ambiguity is unacceptable in production results, so
# every quantity in the g -> kappa_ratio -> GT chain now has its own explicit name, bundled
# into one struct so a caller/report never has to reconstruct or guess which one it has.
# ============================================================================

"""
    MelitzWelfareMetrics

Authoritative, explicitly-named bundle of the g -> kappa_ratio -> GT welfare chain. NEVER
store or print the raw coordinate `g` under a variable/field/CSV-column name containing
`kappa` or `GT` -- construct this struct (via `melitz_welfare_metrics_from_g`/
`melitz_welfare_metrics`) and read the correctly-named field instead.

Fields:
  - `g::Float64`                -- the raw outer-loop free coordinate,
    `theta_free[1] = log(gamma_prime_target)`. Unbounded, sign has no welfare
    interpretation by itself.
  - `gamma_prime::Float64`      -- `exp(g)`, the autarky price-power level.
  - `wage_ratio::Float64`       -- `w_prime / w[target_country]`.
  - `kappa_ratio::Float64`      -- `wage_ratio * gamma_prime^(1/(sigma-1))`, the "pre-`1-`"
    term (main prompt Section A1's derivation, `melitz_gains_from_trade`'s docstring above).
    This is NOT the gains-from-trade by itself -- this is exactly the quantity the OLD
    `kappa_of_g` computed under a name that collided with `kappa`/`GT`.
  - `gains_from_trade::Float64` -- `GT_j = 1 - kappa_ratio`, the actual gains-from-trade
    (`melitz_gains_from_trade`'s return value, reconstructed here field-by-field).
"""
struct MelitzWelfareMetrics
    g::Float64
    gamma_prime::Float64
    wage_ratio::Float64
    kappa_ratio::Float64
    gains_from_trade::Float64
end

"""
    melitz_welfare_metrics_from_g(g, wage_ratio, sigma) -> MelitzWelfareMetrics
    melitz_welfare_metrics_from_g(g, calib_or_ctx) -> MelitzWelfareMetrics

Constructs `MelitzWelfareMetrics` from the raw outer-loop coordinate `g` (as returned by
`solve_melitz_finite_delta_bound`'s `theta_free[1]`/`MelitzOuterCandidate.objective`'s own
underlying coordinate). The second method accepts any object exposing `.sigma`, `.w_prime`,
`.w`, `.target_country` (a `MelitzPrimitives`+`MelitzCounterfactual`-merged `ctx`, or any
calibration context with the same field names -- matches the OLD `kappa_of_g`'s calling
convention exactly, so existing call sites only need the name/return-type updated, not their
arguments).
"""
function melitz_welfare_metrics_from_g(g::Real, wage_ratio::Real, sigma::Real)
    gamma_prime = exp(g)
    kappa_ratio = Float64(wage_ratio) * gamma_prime^(1 / (sigma - 1))
    return MelitzWelfareMetrics(Float64(g), gamma_prime, Float64(wage_ratio), kappa_ratio, 1 - kappa_ratio)
end
function melitz_welfare_metrics_from_g(g::Real, calib_or_ctx)
    wage_ratio = calib_or_ctx.w_prime / calib_or_ctx.w[calib_or_ctx.target_country]
    return melitz_welfare_metrics_from_g(g, wage_ratio, calib_or_ctx.sigma)
end

"""
    melitz_welfare_metrics(p::MelitzPrimitives, cf::MelitzCounterfactual) -> MelitzWelfareMetrics

Population-level construction directly from primitives + counterfactual -- the closed-form
reference point (wraps the SAME arithmetic as `melitz_gains_from_trade`, never a live
outer-search trial). `gains_from_trade` here agrees with `melitz_gains_from_trade(p, cf)`
exactly (tested).
"""
function melitz_welfare_metrics(p::MelitzPrimitives, cf::MelitzCounterfactual)
    g = log(p.gamma_prime_target)
    wage_ratio = cf.w_prime / p.w[p.target_country]
    return melitz_welfare_metrics_from_g(g, wage_ratio, p.sigma)
end

"""
    kappa_ratio_of_g(g, calib_or_ctx) -> Float64

Renamed from the prior session's `kappa_of_g` (2026-07-26 closure session, governing prompt
Phase 3) -- the OLD name was the direct cause of the live mislabeling incident this section's
header describes: it has NEVER computed the gains-from-trade `kappa`/`GT_j`, only the
pre-`1-` ratio term (`MelitzWelfareMetrics.kappa_ratio`). No external call sites existed for
the old name (grep-confirmed: `predictor_corrector.jl`-internal only), so it is renamed
outright rather than kept as a deprecated alias. Equivalent to
`melitz_welfare_metrics_from_g(g, calib_or_ctx).kappa_ratio`.
"""
kappa_ratio_of_g(g::Real, calib_or_ctx) = melitz_welfare_metrics_from_g(g, calib_or_ctx).kappa_ratio

"""
    population_focal_link_residual(p, X, q, f_jj, cf) -> Float64

The POPULATION-level (not finite-sample) focal free-entry link residual, main prompt
Section 4.2 evaluated at the reference Pareto distribution exactly (closed form, no Monte
Carlo): `E[Pi_baseline_j]/w[j] - E[Pi_autarky_j]/w_prime_j`. This is the equation that
PINS DOWN `gamma_prime_target` (via `f_jj`) given an already-fixed baseline (A,f,w) --
addendum Section 8: the population closed-form equilibrium equations must hold at
numerical precision, not merely shrink with `W` (only the FINITE-SAMPLE gap around this
population value is expected to shrink with `W`).
"""
function population_focal_link_residual(p::MelitzPrimitives, X::AbstractMatrix, q::AbstractMatrix,
                                         f_jj::Real, cf::MelitzCounterfactual)
    pi_baseline = population_baseline_focal_profit_integral(p, X, q)
    pi_autarky = population_autarky_profit_integral(f_jj, p.sigma, p.theta_star)
    return pi_baseline / p.w[p.target_country] - pi_autarky / cf.w_prime
end

# ----------------------------------------------------------------------------
# Gravity-pivot elimination (main prompt Section 2, addendum Section 6).
#
# Cov(within(log x), within(log tau)) = sum(T .* withinTransform(x)), T =
# withinTransform(tau), is linear and HOMOGENEOUS in log(x) (verified: at x=ones (log
# x=0), withinTransform(x)=0, so the restriction is 0 with no A/f-independent intercept).
# GATE A5 CORRECTION (docs/melitz_delta_star.md, was `doubleDiff` before): unlike
# `doubleDiff` (an asymmetric "anchored" contrast against a FIXED reference row/column),
# `withinTransform` IS the symmetric two-way-FE orthogonal projection, so it is
# self-adjoint (`P=P^T=P^2` for its underlying linear map `vec(withinTransform(x)) = P @
# vec(log x)`) -- meaning the coefficient vector `c` in `sum(T.*withinTransform(x)) ==
# dot(c, vec(log x))` IS simply `vec(T)` here (`dot(vec(T), P@v) = dot(P@vec(T), v) =
# dot(vec(T), v)` since `T` is already in `P`'s image, `P@vec(T)=vec(T)`). This is a
# genuine simplification available now but NOT taken -- `gravity_coefficient_vector` keeps
# computing `c` via unit-log perturbations (robust, transform-agnostic, verified to agree
# with `vec(T)` by the existing test suite's own cross-check) rather than hand-asserting
# self-adjointness, so this comment stays accurate even if the transform choice changes
# again. For a length-n vector of free LOG-coordinates `z_free` (with one pivot cell
# solved out to satisfy the restriction exactly), the affine offset `g0` is the
# contribution of any FIXED (non-free) cells in that domain, evaluated in LOG-space
# directly since `c` multiplies log-coordinates (zero for the A-restriction, since every A
# cell is free; nonzero for the f-restriction, since `f[j,j]` is fixed given
# `gamma_prime_j` -- see delta_star.jl). Reuses the exact struct/expand/reduce shape
# already validated in full_aod_diag/d4_exact/gravity_elimination.jl's PivotGravityElim,
# reimplemented directly against Melitz's simpler (no extra Ricardian rescaling)
# linear-in-logs restriction.
# ----------------------------------------------------------------------------

"""
    GravityPivot(n, pivot, other, c, g0)

`z_free` (length `n-1`) -> full `z` (length `n`), gravity-feasible EXACTLY: `z[other] =
z_free`, `z[pivot]` solved so that `dot(c, z) + g0 == 0`.
"""
struct GravityPivot{T<:Real}
    n::Int
    pivot::Int
    other::Vector{Int}
    c::Vector{Float64}
    g0::T
end

"""
    gravity_coefficient_vector(D, tau) -> c (length D^2)

`c` such that `sum(withinTransform(tau) .* withinTransform(X)) == dot(c, vec(log.(X)))`
for ANY D x D positive matrix `X` (verified linear+homogeneous in `log(X)`; see the
module-level note above on why `c` is NOT simply `vec(withinTransform(tau))`). Computed
robustly via unit-log perturbations rather than hand-derived: `c[k]` is the
gravity-restriction value at `log(X) = e_k` (`X` = all-ones except cell `k` set to `e^1`,
so `log(X)=e_k` exactly, and linearity gives `sum(T.*withinTransform(X)) = c[k]` directly).

Gate A5 (docs/melitz_delta_star.md): uses `withinTransform` (the symmetric two-way FE
"within" residual), NOT `doubleDiff` (this function's ORIGINAL implementation) -- see
`gravity_residuals`'s docstring for the full derivation of why: `doubleDiff` does not
reproduce production/fullA-exact's own canonical OLS-two-way-FE gravity coefficient
(`moments/newGravityMoment!.jl`'s `UoModel==1` branch, universal in every production run
config), verified live via this repo's pre-existing `gravity_check.jl`.
"""
function gravity_coefficient_vector(D::Int, tau::Matrix{Float64})
    T = withinTransform(tau)
    c = zeros(Float64, D^2)
    for k in 1:D^2
        X = ones(Float64, D, D)
        X[k] = exp(1.0)
        c[k] = sum(T .* withinTransform(X))
    end
    return c
end

"""
    build_gravity_pivot(c, g0; avoid=nothing) -> GravityPivot

Chooses the entry with the LARGEST `|c|` as the pivot (not near zero, for numerical
stability -- matches the existing full_aod pivot-selection rule). `avoid`, if given, is a
LOCAL index (or collection of indices) excluded from pivot candidacy (still a free
coordinate, just not eligible to be solved-out) -- used so the f-pivot doesn't land on the
SAME physical cell as the A-pivot when their domains overlap (both restrictions covary
against the same `c` direction, so without this the two pivots frequently coincide,
over-concentrating both corrections on one cell and driving its participation probability
to ~0 -- caught live while building `fake_data.jl`'s fixture), and (Gate A5, post
`withinTransform` switch) so the f-pivot avoids DOMESTIC (`o==o`) cells -- see
`f_pivot_domestic_avoid_indices`.
"""
function build_gravity_pivot(c::Vector{Float64}, g0::Real;
                              avoid::Union{Nothing,Int,AbstractVector{Int}}=nothing)
    n = length(c)
    avoid_set = avoid === nothing ? Int[] : (avoid isa Int ? [avoid] : avoid)
    candidates = setdiff(1:n, avoid_set)
    pivot = candidates[argmax(abs.(c[candidates]))]
    other = setdiff(1:n, pivot)
    return GravityPivot(n, pivot, other, c, g0)
end

"""
    f_pivot_domestic_avoid_indices(D, f_free_lin) -> Vector{Int}

Gate A5 finding: under `withinTransform` (the corrected gravity transform), the
coefficient vector `c` is systematically LARGEST in magnitude on the DIAGONAL
(`o==d`, domestic) cells (verified live: for `tau` with narrow-range off-diagonal
iceberg costs, `withinTransform(tau)`'s diagonal entries -- `log tau[o,o]=0`, below every
row/column's mean once off-diagonal `tau>1` pulls those means up -- dominate `|c|`, for
EVERY origin, not just one). Letting the f-pivot land on a domestic cell is risky: solving
that cell out to satisfy the gravity constraint can push a domestic fixed cost `f[o,o]`
above its own export cells' `f[o,d]`, silently breaking export-selection
(`zhat[o,d]>=zhat[o,o]`) -- reproduced live (100% of 60 tried seeds failed
`generate_fake_melitz_data`'s export-selection check immediately after the `doubleDiff` ->
`withinTransform` switch, and retuning the domestic/export cost GAP alone did not fix it
even at extreme settings, confirming the failure mode is the PIVOT CHOICE, not fixture
noise scale). Returns every position within `f_free_lin` (the `D^2-1` free f-cells,
excluding `f[j,j]`) whose `(o,d)` is domestic (`o==d`), for exclusion from f-pivot
candidacy -- routing the f-pivot to an off-diagonal (export) cell instead, which has much
more feasibility slack (export cutoffs only need to stay ABOVE the domestic cutoff, not
below any other specific value).
"""
function f_pivot_domestic_avoid_indices(D::Int, f_free_lin::Vector{Int})
    return [k for (k, i) in enumerate(f_free_lin) if (lin2od(i, D)[1] == lin2od(i, D)[2])]
end

"""
    f_pivot_avoid_index(A_pivot_global, f_free_lin) -> Union{Nothing,Int}

Maps the A-pivot's GLOBAL (1:D^2) linear index to its position WITHIN `f_free_lin` (the
D^2-1 free f-cells, excluding `(j,j)`), if present -- `nothing` if the A-pivot landed
exactly on `(j,j)` (which isn't in `f_free_lin` at all) or is otherwise not a candidate.
"""
f_pivot_avoid_index(A_pivot_global::Int, f_free_lin::Vector{Int}) =
    findfirst(==(A_pivot_global), f_free_lin)

"""
    f_gravity_pivot_avoid_indices(D, f_free_lin, A_pivot_global) -> Vector{Int}

Session prompt Section 1.1: the combined LOCAL (within `f_free_lin`) avoid-set for
choosing the f-gravity pivot, replacing the previous production path's incomplete
`f_pivot_avoid_index`-only exclusion (which excluded only the A-pivot's physical cell,
NOT domestic cells -- so the f-pivot could, and empirically did, land on a domestic
`(o,o)` cell). Combines every domestic (`o==d`) cell (`f_pivot_domestic_avoid_indices`,
the Gate A5 diagonal-dominance finding -- landing the f-pivot there risks pushing
`f[o,o]` above its own export cells and breaking export-selection) with the A-pivot's
own physical cell (`f_pivot_avoid_index`, so the two pivots cannot coincide). Guarantees
(session prompt Section 1.1's four requirements, given `build_gravity_pivots` also
restricts the A-pivot to off-diagonal): `o != d` for both pivots, the two pivot cells
differ, and neither pivot is `(j,j)` (never a candidate for either domain in the first
place -- `(j,j)` is diagonal, excluded from `f_free_lin` entirely and from A's
diagonal-avoid set).
"""
function f_gravity_pivot_avoid_indices(D::Int, f_free_lin::Vector{Int}, A_pivot_global::Int)
    domestic = f_pivot_domestic_avoid_indices(D, f_free_lin)
    a_local = f_pivot_avoid_index(A_pivot_global, f_free_lin)
    return a_local === nothing ? domestic : vcat(domestic, a_local)
end

"""
    pivot_conditioning_diagnostics(tau, target_country) -> NamedTuple

Session prompt Section 1.1: a CONDITIONING-ONLY comparison of the active off-diagonal
gravity-pivot elimination against (a) the UNRESTRICTED max-`|c|` pivot (whichever cell
that is, diagonal or not -- the choice `build_gravity_pivots` made before this session's
fix) and (b) an orthonormal/null-space parameterization of the SAME affine constraint
(equivalent in spirit to `project_to_gravity_manifold`'s minimum-L2 correction, which
spreads a unit perturbation evenly across every free coordinate instead of concentrating
it in one). Reports each choice's worst-case LEVERAGE: for a single-cell pivot
elimination `z[pivot] = -(g0 + sum_k c[other[k]]*z_free[k]) / c[pivot]`, a unit
perturbation of free coordinate `k` moves the pivot coordinate by `|c[other[k]]/c[pivot]|`;
`max_leverage` is the worst case over every free coordinate. An orthogonal-projection
(orthonormal/null-space) parameterization is NON-EXPANSIVE in L2 by construction --
leverage `<=1` always -- so `orthonormal_leverage` is reported as exactly `1.0` rather
than separately constructed. Diagnostic only: does NOT choose the production pivot
(`build_gravity_pivots` does that, using the off-diagonal-restricted choice).
"""
function pivot_conditioning_diagnostics(tau::Matrix{Float64}, target_country::Int)
    D = size(tau, 1)
    c_full = gravity_coefficient_vector(D, tau)

    unrestricted = build_gravity_pivot(c_full, 0.0)
    A_diag_avoid = [od2lin(o, o, D) for o in 1:D]
    restricted = build_gravity_pivot(c_full, 0.0; avoid=A_diag_avoid)

    leverage(gp) = maximum(abs.(gp.c[gp.other] ./ gp.c[gp.pivot]))

    unrestricted_od = lin2od(unrestricted.pivot, D)
    restricted_od = lin2od(restricted.pivot, D)
    return (
        unrestricted_pivot_od=unrestricted_od,
        unrestricted_pivot_is_diagonal=(unrestricted_od[1] == unrestricted_od[2]),
        unrestricted_max_leverage=leverage(unrestricted),
        restricted_pivot_od=restricted_od,
        restricted_max_leverage=leverage(restricted),
        orthonormal_leverage=1.0,
    )
end

"z_free (length n-1) -> full z (length n), gravity-feasible EXACTLY."
function pivot_expand(z_free::AbstractVector{T}, gp::GravityPivot) where {T}
    z = zeros(T, gp.n)
    @inbounds for (k, i) in enumerate(gp.other)
        z[i] = z_free[k]
    end
    rhs = -gp.g0 - sum(gp.c[gp.other[k]] * z_free[k] for k in eachindex(z_free))
    z[gp.pivot] = rhs / gp.c[gp.pivot]
    return z
end

"full z (length n) -> z_free (length n-1), dropping the pivot coordinate."
pivot_reduce(z::AbstractVector, gp::GravityPivot) = z[gp.other]

"""
    project_to_gravity_manifold(z_raw, c, g0) -> z_proj

Orthogonal (minimum-L2-perturbation) projection of `z_raw` onto the affine constraint
`dot(c, z) + g0 == 0`: `z_proj = z_raw - c*(dot(c,z_raw)+g0)/dot(c,c)`. Spreads the
correction evenly across every coordinate -- contrast the single-cell `GravityPivot`
elimination (delta_star.jl's ACTUAL free/economic outer coordinate system). Used only for
constructing a well-behaved synthetic fixture (fake_data.jl): dumping an entire
correction onto one pivot cell can produce economically extreme (even cutoff<1
infeasible) values when the two gravity restrictions' pivots happen to require a large
single-cell adjustment -- found live while building the D=4 fixture.
"""
function project_to_gravity_manifold(z_raw::AbstractVector, c::AbstractVector, g0::Real)
    return z_raw .- c .* ((dot(c, z_raw) + g0) / dot(c, c))
end

"""
    project_to_gravity_manifold_weighted(z_raw, c, g0, weights) -> z_proj

Weighted generalization of `project_to_gravity_manifold`: the minimum-`sum(weights .*
(z_proj-z_raw).^2)` point on the SAME affine constraint `dot(c,z)+g0==0` (closed form,
Lagrangian: `z_proj[k] = z_raw[k] - (c[k]/weights[k]) * (dot(c,z_raw)+g0) /
sum(c.^2 ./ weights)`; reduces to `project_to_gravity_manifold` at `weights.==1`). Gate A5:
needed because, under the corrected `withinTransform` gravity coefficient vector, `|c|` is
systematically LARGEST on DOMESTIC (`o==d`) cells (verified live -- see
`f_pivot_domestic_avoid_indices`), so the UNWEIGHTED min-L2 projection used by
`fake_data.jl` disproportionately dumps the f-gravity correction onto domestic fixed
costs, pushing them above their own export costs and breaking export-selection (reproduced
live: 100% of 60 tried seeds failed immediately after the `doubleDiff` ->
`withinTransform` switch, and retuning noise scale/level gaps alone did not fix it -- the
failure tracks the PROJECTION's implicit weighting, not the fixture's raw noise). Passing
a large weight on domestic cells and weight 1 on export cells routes nearly all of the
correction onto export costs instead, which have far more feasibility slack.
"""
function project_to_gravity_manifold_weighted(z_raw::AbstractVector, c::AbstractVector, g0::Real,
                                               weights::AbstractVector)
    scale = c ./ weights
    return z_raw .- scale .* ((dot(c, z_raw) + g0) / dot(c, scale))
end

# ----------------------------------------------------------------------------
# General-F LFD-based recovery (main prompt Section 7) and ex-post checks (Section 9).
# All use the shared `melitz_firm` routine and the recovered LFD `weights` -- never the
# reference (equal/Dirac) weights, and never a Pareto closed form.
# ----------------------------------------------------------------------------

"""
    recover_entry_costs_from_lfd(p, eq, z_draws, weights) -> f_entry (D-vector)

Main prompt Section 7.1: `f_entry[o] = E_LFD[Pi_baseline_o] / w[o]`,
`Pi_baseline_o(z) = sum_d operating_profit[o,d](z_o)` (REALIZED, i.e. gated by the
participation decision -- matches `melitz_firm`'s `realized_operating_profit`, never the
unconstrained value, consistent with how `operating_profit>0` is used everywhere else as
the participation convention).
"""
function recover_entry_costs_from_lfd(p::MelitzPrimitives, eq::MelitzEquilibrium,
                                       z_draws::AbstractMatrix, weights::AbstractVector)
    D = p.D
    f_entry = zeros(D)
    @inbounds for o in 1:D
        s = 0.0
        for w in eachindex(weights)
            z = z_draws[w, o]
            profit_sum = 0.0
            for d in 1:D
                firm = melitz_firm(p.w[o], p.tau[o, d], p.A[o, d], p.f[o, d], p.sigma,
                                    eq.expenditure[d], 1.0, z)
                profit_sum += firm.realized_operating_profit
            end
            s += weights[w] * profit_sum
        end
        f_entry[o] = s / p.w[o]
    end
    return f_entry
end

"""
    recover_focal_autarky_entry_cost(p, cf, z_draws, weights) -> f_entry_autarky_j

Main prompt Section 7.1 (autarky): `f_entry_autarky_j = E_LFD[Pi_autarky_j] / w_prime[j]`.
"""
function recover_focal_autarky_entry_cost(p::MelitzPrimitives, cf::MelitzCounterfactual,
                                           z_draws::AbstractMatrix, weights::AbstractVector)
    j = p.target_country
    s = 0.0
    @inbounds for w in eachindex(weights)
        z = z_draws[w, j]
        firm = melitz_firm(cf.w_prime, 1.0, p.A[j, j], p.f[j, j], p.sigma,
                            cf.expenditure_prime, p.gamma_prime_target, z)
        s += weights[w] * firm.realized_operating_profit
    end
    return s / cf.w_prime
end

"""
    recover_N_prime_market_clearing(cf, f_jj, sigma, f_entry_j) -> N_prime_j

Main prompt Section 7.2: autarky market clearing, `N'_j = expenditure'_j /
(sigma*w'_j*(f_jj + f_entry_j))`.
"""
recover_N_prime_market_clearing(cf::MelitzCounterfactual, f_jj::Real, sigma::Real, f_entry_j::Real) =
    cf.expenditure_prime / (sigma * cf.w_prime * (f_jj + f_entry_j))

"""
    recover_N_prime_price_index(p, cf, z_draws, weights) -> N_prime_j

Main prompt Section 9.5 (independent cross-check formula): `N'_j = gamma'_j /
E_LFD[price_prime_jj(z)^(1-sigma) * active'_jj(z)]`. Gated by the participation decision
via `melitz_firm` (a no-op here since the reference Pareto support is `[1,infinity)` and
`zhat_prime[j,j]==1` exactly, so every reference draw is active in autarky -- gating kept
for correctness under any candidate LFD/derived cutoff, not assumed away).
"""
function recover_N_prime_price_index(p::MelitzPrimitives, cf::MelitzCounterfactual,
                                      z_draws::AbstractMatrix, weights::AbstractVector)
    j = p.target_country
    s = 0.0
    @inbounds for w in eachindex(weights)
        z = z_draws[w, j]
        firm = melitz_firm(cf.w_prime, 1.0, p.A[j, j], p.f[j, j], p.sigma,
                            cf.expenditure_prime, p.gamma_prime_target, z)
        s += weights[w] * (firm.active ? firm.price^(1 - p.sigma) : 0.0)
    end
    return p.gamma_prime_target / s
end

"""
    check_profiled_melitz_equilibrium(p, eq, cf, z_draws, weights; full=true) -> MelitzEquilibriumCheck

Main prompt Section 9 / addendum Section 9: every omitted-equilibrium-equation residual,
evaluated under the recovered LFD `weights` (never reference/equal weights). See
`MelitzEquilibriumCheck`'s docstring for the field-by-field mapping to Section 9.1-9.9.

Governing prompt Phase 2 (2026-07-XX outer-search session): `full=false` skips every `O(D^2*W)`
loop above (Sections 9.1-9.6, 9.9 -- confirmed live to be the ENTIRE cost of this function,
`docs/melitz_outer_search_scaling_and_profile_2026-07-XX.md` Phase 2: ~99.8% of one real-D20
finite FC's own wall time) and computes ONLY `gravity_residual_A`/`gravity_residual_f`
(Section 9.8, `gravity_residuals(p)` -- a function of `p` ALONE, `O(D^2)`, no dependence on
`z_draws`/`weights` at all) -- the ONLY two fields any production code path actually reads
(confirmed by grep: `melitz_classify_outer_feasibility`'s own `gravity_feasible` line is the
SOLE consumer of `.equilibrium_check` anywhere in `src/melitz/`). Every other field is filled
with `NaN` (or the appropriately NaN-filled vector), never a stale/zero value that could be
silently mistaken for a real diagnostic -- callers that need the full diagnostic detail (the
eventual cold-verified final answer, which always calls `evaluate_melitz_delta(...; cold=true)`
fresh, never reusing a live registration's own possibly-cheap check) must pass `full=true`
(the default, byte-identical to this function's pre-existing behavior for every existing
caller).
"""
function check_profiled_melitz_equilibrium(p::MelitzPrimitives, eq::MelitzEquilibrium,
                                            cf::MelitzCounterfactual,
                                            z_draws::AbstractMatrix, weights::AbstractVector;
                                            full::Bool=true)
    D = p.D
    if !full
        gravity_residual_A, gravity_residual_f = gravity_residuals(p)
        Tg = typeof(gravity_residual_A)
        nanD = fill(Tg(NaN), D)
        nanS = Tg(NaN)
        return MelitzEquilibriumCheck(nanD, nanD, nanD, nanS, nanS, nanS, nanS, nanS, nanS, nanS, nanS,
            nanS, nanS, nanS, nanS, gravity_residual_A, gravity_residual_f)
    end
    j = p.target_country

    # 9.1 baseline price-index identities
    residual_gamma_baseline = zeros(D)
    for d in 1:D
        s = 0.0
        for o in 1:D, w in eachindex(weights)
            z = z_draws[w, o]
            firm = melitz_firm(p.w[o], p.tau[o, d], p.A[o, d], p.f[o, d], p.sigma,
                                eq.expenditure[d], 1.0, z)
            s += weights[w] * (firm.active ? firm.price^(1 - p.sigma) : 0.0)
        end
        residual_gamma_baseline[d] = s - 1.0
    end

    # 7.1 / 9.2 baseline entry costs and free-entry residuals
    f_entry = recover_entry_costs_from_lfd(p, eq, z_draws, weights)
    residual_free_entry_baseline = zeros(D)
    for o in 1:D
        s = 0.0
        for w in eachindex(weights)
            z = z_draws[w, o]
            profit_sum = 0.0
            for d in 1:D
                firm = melitz_firm(p.w[o], p.tau[o, d], p.A[o, d], p.f[o, d], p.sigma,
                                    eq.expenditure[d], 1.0, z)
                profit_sum += firm.realized_operating_profit
            end
            s += weights[w] * profit_sum
        end
        residual_free_entry_baseline[o] = s - p.w[o] * f_entry[o]
    end

    # 7.1 / 9.3 focal autarky entry cost and free-entry residual
    f_entry_autarky_j = recover_focal_autarky_entry_cost(p, cf, z_draws, weights)
    s_autarky = 0.0
    for w in eachindex(weights)
        z = z_draws[w, j]
        firm = melitz_firm(cf.w_prime, 1.0, p.A[j, j], p.f[j, j], p.sigma,
                            cf.expenditure_prime, p.gamma_prime_target, z)
        s_autarky += weights[w] * firm.realized_operating_profit
    end
    residual_free_entry_autarky = s_autarky - cf.w_prime * f_entry_autarky_j

    # 7.2 / 9.4 autarky market clearing (using the BASELINE-recovered f_entry[j], per the
    # focal link moment's requirement that the SAME f_entry[j] rationalize both)
    N_prime_mc = recover_N_prime_market_clearing(cf, p.f[j, j], p.sigma, f_entry[j])
    mean_revenue_prime_jj = 0.0
    for w in eachindex(weights)
        z = z_draws[w, j]
        firm = melitz_firm(cf.w_prime, 1.0, p.A[j, j], p.f[j, j], p.sigma,
                            cf.expenditure_prime, p.gamma_prime_target, z)
        mean_revenue_prime_jj += weights[w] * firm.realized_revenue
    end
    residual_market_clearing_autarky = N_prime_mc * mean_revenue_prime_jj - cf.expenditure_prime

    # 9.5 autarky price-index identity (the KEY omitted moment) + two-formula N'_j cross-check
    N_prime_gamma = recover_N_prime_price_index(p, cf, z_draws, weights)
    residual_gamma_autarky = N_prime_mc * (p.gamma_prime_target / N_prime_gamma) - p.gamma_prime_target
    N_prime_diff_abs = abs(N_prime_gamma - N_prime_mc)
    N_prime_diff_rel = N_prime_diff_abs / abs(N_prime_mc)

    # 9.6 autarky cutoff: revenue_prime_jj(z=1)/sigma - w'*f_jj (== operating_profit at z=1)
    firm_at_one = melitz_firm(cf.w_prime, 1.0, p.A[j, j], p.f[j, j], p.sigma,
                               cf.expenditure_prime, p.gamma_prime_target, 1.0)
    residual_autarky_cutoff = firm_at_one.operating_profit

    # 9.9 cutoff inequalities
    min_baseline_cutoff = minimum(eq.cutoff)
    min_cutoff_minus_one = min_baseline_cutoff - 1.0
    min_export_minus_domestic = minimum(eq.cutoff[o, d] - eq.cutoff[o, o] for o in 1:D, d in 1:D if d != o)

    # 9.8 gravity
    gravity_residual_A, gravity_residual_f = gravity_residuals(p)

    return MelitzEquilibriumCheck(residual_gamma_baseline, f_entry, residual_free_entry_baseline,
        f_entry_autarky_j, residual_free_entry_autarky, N_prime_mc, residual_market_clearing_autarky,
        N_prime_gamma, N_prime_diff_abs, N_prime_diff_rel, residual_gamma_autarky,
        residual_autarky_cutoff, min_baseline_cutoff, min_cutoff_minus_one,
        min_export_minus_domestic, gravity_residual_A, gravity_residual_f)
end

# ----------------------------------------------------------------------------
# PARETO-ONLY closed-form benchmark diagnostics (addendum Section 4). Encode the
# SUPERSEDED f_entry-as-primitive / N-derived / N'=N closure. Kept only for the
# addendum Section 15 cross-check at the exact Pareto benchmark -- NEVER called from
# moments.jl, delta_star.jl, or fake_data.jl's active construction path.
# ----------------------------------------------------------------------------

"""
    pareto_entry_cost_from_free_entry(C_row, zhat_row, w_o, sigma, theta_star) -> f_entry_o

Closed-form free-entry identity under the reference PARETO distribution (not a general-F
recovery -- compare `recover_entry_costs_from_lfd`, which works under any weights).
"""
function pareto_entry_cost_from_free_entry(C_row::AbstractVector, zhat_row::AbstractVector,
                                            w_o::Real, sigma::Real, theta_star::Real)
    s = zero(eltype(C_row))
    for d in eachindex(C_row)
        s += C_row[d] * (sigma - 1) / (sigma * (theta_star - sigma + 1)) *
             zhat_row[d]^(sigma - 1 - theta_star)
    end
    return s / w_o
end

"""
    pareto_entrant_mass_from_labor(L_o, f_entry_o, sigma, theta_star) -> N_o

SUPERSEDED closed form (Melitz & Redding 2014 eq. 22 analog): `N_o = (sigma-1)/(sigma*
theta_star) * L_o/f_entry_o`. Kept only as a Pareto-benchmark cross-check -- the active
closure normalizes `N_o == 1` instead (`normalize_baseline_entrant_mass`).
"""
pareto_entrant_mass_from_labor(L_o::Real, f_entry_o::Real, sigma::Real, theta_star::Real) =
    (sigma - 1) / (sigma * theta_star) * L_o / f_entry_o

"""
    pareto_autarky_fixed_cost_fixed_mass(f_entry_target, sigma, theta_star) -> f[target,target]

SUPERSEDED closed form combining the autarky free-entry condition with `zhat'=1` UNDER
THE FIXED-MASS (`N'=N`) CLOSURE. The active closure instead derives `f[j,j]` from
`gamma_prime_j` directly (`derive_fjj_from_autarky_cutoff`), with no dependence on any
entry-cost primitive.
"""
pareto_autarky_fixed_cost_fixed_mass(f_entry_target::Real, sigma::Real, theta_star::Real) =
    f_entry_target * (theta_star - sigma + 1) / (sigma - 1)

"""
    pareto_target_cutoff_fixed_mass(expenditure_target, w_target, X_target_target, theta_star)
        -> zhat[target,target]  (baseline)

SUPERSEDED closed form for the target country's baseline domestic cutoff under the
FIXED-MASS (`N'=N`) closure (`zhat[t,t] = (w_target/lambda_tt)^(1/theta_star)`). The
active closure does not need this -- `derive_fjj_from_autarky_cutoff` derives `f[j,j]`
directly from `gamma_prime_j`, and the baseline cutoff `zhat[j,j]` is whatever the
(searched) `A[j,j]`/(derived) `f[j,j]` imply, not mechanically pinned.
"""
pareto_target_cutoff_fixed_mass(expenditure_target::Real, w_target::Real,
                                 X_target_target::Real, theta_star::Real) =
    (expenditure_target * w_target / X_target_target)^(1 / theta_star)

"""
    pareto_solve_autarky_counterfactual_fixed_mass(p_with_f_entry, eq_with_N; rtol=1e-6)
        -> MelitzCounterfactual

SUPERSEDED closed-form autarky counterfactual under `N'[target]=N[target]` (reuses
`eq.entrant_mass[target]` unchanged) and a precomputed `price_power_prime` -- kept ONLY
for the addendum Section 15 cross-check against the active LFD-recovered
`gamma_prime_target`/`N'_j` at the exact Pareto benchmark. `p_with_f_entry`/`eq_with_N`
here are plain NamedTuples (not `MelitzPrimitives`/`MelitzEquilibrium`, which no longer
carry `f_entry`/`entrant_mass` fields) with fields
`(D, sigma, theta_star, target_country, tau, w, A, f, f_entry)` and
`(expenditure, cutoff, trade_flow, entrant_mass)` respectively.
"""
function pareto_solve_autarky_counterfactual_fixed_mass(p, eq; rtol::Real=1e-6)
    t = p.target_country
    sigma, theta_star = p.sigma, p.theta_star
    T = eltype(p.A)
    w_prime = one(T)
    expenditure_prime = eq.expenditure[t]

    zhat_tt_required = pareto_target_cutoff_fixed_mass(expenditure_prime, p.w[t], eq.trade_flow[t, t], theta_star)
    isapprox(eq.cutoff[t, t], zhat_tt_required; rtol=rtol) || throw(ArgumentError(
        "cutoff[target,target]=$(eq.cutoff[t,t]) does not satisfy the fixed-mass autarky " *
        "zhat'=1 normalization (expected $zhat_tt_required)"))

    K1_tt = melitz_K1(w_prime, one(T), p.A[t, t], sigma)
    N_t = eq.entrant_mass[t]
    price_power_prime = N_t * K1_tt * theta_star / (theta_star - sigma + 1)
    cutoff_prime = one(T)
    trade_flow_prime = expenditure_prime

    cf = MelitzCounterfactual(t, w_prime, expenditure_prime, cutoff_prime, trade_flow_prime)
    cf.entrant_mass_prime = N_t
    return cf, price_power_prime
end
