# Data-only Pareto calibration pipeline for the full-D Melitz Christensen-Connault
# benchmark. See docs/melitz_pareto_data_calibration_2026-07-24.md for the full
# derivation. This file closes the methodological gap documented in
# docs/melitz_wage_calibration_gap_2026-07-24.md: every existing Melitz estimation
# context (`build_melitz_psi_bundle` applied to a `MelitzSyntheticData`) takes baseline
# wages `p.w` from the hidden synthetic DGP (`melitz_solve_wages_ge`, which requires the
# TRUE `A`/`f`). This file builds a `MelitzParetoCalibration` from OBSERVABLES ONLY
# (bilateral trade shares/flows, labor endowments, iceberg costs, sigma, theta_star,
# focal country) -- no hidden DGP object is ever read here.
#
# Type-level separation (main prompt Section 1):
#   MelitzObservedData     -- what an estimator/calibration is allowed to see.
#   MelitzParetoCalibration -- the calibrated reference model built from observables only.
#   MelitzSyntheticTruth    -- hidden DGP primitives, used ONLY after estimation/
#                              calibration, for recovery comparison (Monte Carlo studies).
#
# Section-by-section map to the governing prompt:
#   Section 3  -> MelitzObservedData (construction + validation)
#   Section 4  -> calibrate_melitz_wages (reuses melitz_solve_wages, adds diagnostics +
#                 an independent Perron/eigenvector cross-check)
#   Section 5  -> melitz_gravity_theta_check (theta_star/gravity compatibility)
#   Section 6  -> melitz_pareto_composite (the identified chi_od composite)
#   Section 7/8-> melitz_ad_from_cutoffs (cell-by-cell A/f inversion, affine-in-u structure)
#   Section 9  -> melitz_cutoff_target, calibrate_melitz_cutoffs (the convex cutoff QP)
#   Section 10 -> melitz_conditioning_diagnostics
#   Section 11 -> melitz_profile_entry_costs
#   Section 12 -> melitz_verify_baseline_equilibrium
#   Section 13 -> melitz_autarky_reference
#   Section 14 -> MelitzParetoCalibration, calibrate_melitz_pareto (top-level API)
#   Section 15 -> split_melitz_synthetic_truth (observed/truth separation for the
#                 EXISTING `generate_fake_melitz_data` fixture, kept as the legacy
#                 leaked-DGP generator per Section 15's explicit permission)

using LinearAlgebra: dot, eigen, norm, svd, I
using JuMP
using HiGHS
using Roots: find_zero, Bisection

# ============================================================================
# Section 1/3: observed data, synthetic truth (type-level separation)
# ============================================================================

"""
    MelitzShareDiagnostics

Raw-data diagnostics computed BEFORE any share transformation (addendum Section 1):
`min_cell`/`max_cell`, `column_sums`, `max_abs_column_sum_deviation`, and counts of
zero/negative/nonfinite cells. Reported unconditionally so a caller can see exactly how
close the SUPPLIED matrix already is to a valid share matrix before any policy is applied.
"""
struct MelitzShareDiagnostics
    min_cell::Float64
    max_cell::Float64
    column_sums::Vector{Float64}
    max_abs_column_sum_deviation::Float64
    n_zero_or_negative::Int
    n_nonfinite::Int
end

function melitz_share_diagnostics(lam::AbstractMatrix{<:Real})
    csums = vec(sum(lam, dims=1))
    finite_cells = filter(isfinite, vec(lam))
    return MelitzShareDiagnostics(
        isempty(finite_cells) ? NaN : minimum(finite_cells),
        isempty(finite_cells) ? NaN : maximum(finite_cells),
        csums, maximum(abs.(csums .- 1)),
        count(x -> isfinite(x) && x <= 0, lam), count(!isfinite, lam))
end

"""
    MelitzObservedData

Everything a Melitz calibration/estimation step is allowed to see: bilateral trade
SHARES `lambda` (D x D, columns sum to 1), labor endowments `L`, and iceberg trade costs
`tau`. No `A`, `f`, `w`, cutoff, entry cost, or `gamma_prime_target` may ever be a field
here -- those are either calibrated (`MelitzParetoCalibration`) or hidden truth
(`MelitzSyntheticTruth`).

CRITICAL ADDENDUM (2026-07-24) compliance: this constructor no longer silently mutates
supplied data by default. `share_policy`/`tau_diagonal_policy` (below) are explicit,
recorded, and used identically by every downstream consumer (theta estimation and
calibration alike) -- see `raw_share_diagnostics`/`tau_diagonal_raw` for exactly what was
supplied, and `share_policy`/`tau_diagonal_policy` for exactly what was done to it, if
anything.
"""
struct MelitzObservedData
    D::Int
    countries::Vector{String}
    lambda::Matrix{Float64}
    L::Vector{Float64}
    tau::Matrix{Float64}
    share_policy::Symbol
    raw_share_diagnostics::MelitzShareDiagnostics
    tau_diagonal_policy::Symbol
    tau_diagonal_raw::Vector{Float64}
end

"""
    MelitzObservedData(; lambda=nothing, X=nothing, L, tau, countries=String[],
                          atol=1e-6, zero_policy=:error, floor_value=NaN,
                          share_policy=:as_supplied, tau_diagonal_policy=:normalize_to_one)

Constructs and VALIDATES observed data (main prompt Section 3, addendum Sections 1-3).
Exactly one of `lambda` (trade shares) or `X` (trade flows, from which `lambda_od =
X_od/sum_o X_od` is built, "the repository's established origin/destination orientation"
-- rows are origin `o`, columns are destination `d`, confirmed against `pi.csv`'s own
column-sum-to-1 convention and `prestep/master_prestep.jl`'s `lambda * (w0.*L)./L` usage)
must be supplied.

Validates: dimensions, finiteness, nonnegativity, strictly positive cells (required by
log-gravity -- `zero_policy=:error`, the default, THROWS if any zero/negative cell is
found rather than silently flooring it; pass `zero_policy=:floor, floor_value=eps` for an
explicit, non-default, documented floor), column sums within `atol` of 1, positive
domestic shares, and `tau` diagonal within `atol` of 1.

**`share_policy`** (addendum Section 1, default `:as_supplied`): the matrix returned in
`.lambda` is the SUPPLIED matrix, UNCHANGED, as long as its columns already sum to 1
within `atol` (true of `real_data/noah_D20/pi.csv` to machine precision -- see
`raw_share_diagnostics`). Pass `share_policy=:renormalize` to explicitly force `lam ./=
sum(lam,dims=1)` (a no-op to machine precision when columns already sum to ~1, but
documented/visible/opt-in rather than silent). Any policy other than `:as_supplied`/
`:renormalize` throws.

**`tau_diagonal_policy`** (addendum Section 2, default `:normalize_to_one`): `MelitzPrimitives`
(`types.jl`) hard-requires `diag(tau) == 1.0` EXACTLY (a genuine, repo-wide structural
requirement of this model -- not a preference of this calibration file, confirmed by
reading `types.jl`'s own inner-constructor assertion) -- so per the addendum's Section 2.2
branch ("if the maintained model requires every diagonal tau to equal one"), this is the
ONE explicit, documented preprocessing step, applied here (before any theta estimation or
calibration reads `tau`), with the ORIGINAL diagonal preserved in `.tau_diagonal_raw` for
the record. Pass `tau_diagonal_policy=:as_supplied` to skip this (the raw diagonal is kept
exactly, which will make any later `MelitzPrimitives` construction throw unless the raw
diagonal already happens to be exactly 1.0 -- appropriate only for data already known to
satisfy the model's own invariant, or for diagnostic-only use of this struct).
"""
function MelitzObservedData(; lambda::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                             X::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                             L::AbstractVector{<:Real},
                             tau::AbstractMatrix{<:Real},
                             countries::Vector{String}=String[],
                             atol::Real=1e-6,
                             zero_policy::Symbol=:error,
                             floor_value::Real=NaN,
                             share_policy::Symbol=:as_supplied,
                             tau_diagonal_policy::Symbol=:normalize_to_one)
    (lambda === nothing) != (X === nothing) || throw(ArgumentError(
        "MelitzObservedData: supply exactly one of `lambda` (shares) or `X` (flows)"))
    D = length(L)
    size(tau) == (D, D) || throw(ArgumentError("tau must be D x D matching length(L)=$D"))
    lam = lambda === nothing ? Matrix{Float64}(X) ./ sum(Matrix{Float64}(X), dims=1) :
                                Matrix{Float64}(lambda)
    size(lam) == (D, D) || throw(ArgumentError("lambda/X must be D x D matching length(L)=$D"))

    all(isfinite, lam) || throw(ArgumentError("MelitzObservedData: non-finite trade-share cell(s)"))
    all(isfinite, tau) || throw(ArgumentError("MelitzObservedData: non-finite tau cell(s)"))
    all(isfinite, L) && all(>(0), L) || throw(ArgumentError("MelitzObservedData: L must be finite and strictly positive"))
    all(>=(0), lam) || throw(ArgumentError("MelitzObservedData: negative trade-share cell(s) present"))
    all(>(0), tau) || throw(ArgumentError("MelitzObservedData: nonpositive tau cell(s) present"))

    # Addendum Section 1: report diagnostics on the RAW supplied matrix BEFORE any policy
    # (zero-flooring or renormalization) is applied.
    raw_diag = melitz_share_diagnostics(lam)

    zero_cells = [(o, d) for o in 1:D, d in 1:D if lam[o, d] <= 0]
    if !isempty(zero_cells)
        if zero_policy == :error
            throw(ArgumentError(
                "MelitzObservedData: zero/negative trade-share cells at $(zero_cells) -- " *
                "log-gravity requires strictly positive shares; no zero-handling policy was " *
                "supplied. Pass zero_policy=:floor, floor_value=<eps> to explicitly floor them."))
        elseif zero_policy == :floor
            isfinite(floor_value) && floor_value > 0 || throw(ArgumentError(
                "zero_policy=:floor requires a finite, positive floor_value"))
            for (o, d) in zero_cells
                lam[o, d] = floor_value
            end
            lam ./= sum(lam, dims=1)  # re-normalize columns after flooring (unavoidable: the
                                       # floor itself perturbs adding-up, independent of share_policy)
        else
            throw(ArgumentError("MelitzObservedData: unknown zero_policy $zero_policy"))
        end
    end

    isapprox(vec(sum(lam, dims=1)), ones(D); atol=atol) || throw(ArgumentError(
        "MelitzObservedData: trade-share columns must sum to 1 (destination price-index " *
        "normalization); max deviation = $(maximum(abs.(vec(sum(lam,dims=1)) .- 1))). This is " *
        "a genuine data-convention mismatch (addendum Section 1) -- diagnose the source data, " *
        "do not pass a laxer atol to silence it."))
    all(o -> lam[o, o] > 0, 1:D) || throw(ArgumentError("MelitzObservedData: domestic shares must be positive"))
    isapprox([tau[o, o] for o in 1:D], ones(D); atol=atol) || throw(ArgumentError(
        "MelitzObservedData: tau diagonal must be within atol=$atol of 1 (a diagonal entry " *
        "wildly different from 1 is a genuine data problem, not something any tau_diagonal_policy " *
        "here is meant to paper over)"))
    isempty(countries) || length(countries) == D || throw(ArgumentError(
        "MelitzObservedData: countries must have length D=$D if supplied"))
    share_policy in (:as_supplied, :renormalize) || throw(ArgumentError(
        "MelitzObservedData: share_policy must be :as_supplied or :renormalize, got $share_policy"))
    tau_diagonal_policy in (:as_supplied, :normalize_to_one) || throw(ArgumentError(
        "MelitzObservedData: tau_diagonal_policy must be :as_supplied or :normalize_to_one, got $tau_diagonal_policy"))

    # Addendum Section 1: `:as_supplied` (default) leaves `lam` byte-for-byte as supplied
    # (post zero-policy only) whenever it already clears the `atol` adding-up check above --
    # no forced division. `:renormalize` is the OLD unconditional behavior, now opt-in/visible.
    if share_policy == :renormalize
        lam = lam ./ sum(lam, dims=1)
    end

    # Addendum Section 2.2: ONE explicit, documented preprocessing step for the tau diagonal,
    # applied here (before theta estimation or calibration ever sees `tau`), with the ORIGINAL
    # diagonal preserved for the record (`real_data/noah_D20/tau.csv`'s "row" aggregate category
    # is ~1.001122, a measurement artifact -- `MelitzPrimitives` requires exact diag==1).
    tau_raw_diag = [tau[o, o] for o in 1:D]
    tau_out = Matrix{Float64}(tau)
    if tau_diagonal_policy == :normalize_to_one
        for o in 1:D
            tau_out[o, o] = 1.0
        end
    end

    return MelitzObservedData(D, countries, lam, Vector{Float64}(L), tau_out,
                               share_policy, raw_diag, tau_diagonal_policy, tau_raw_diag)
end

"""
    MelitzSyntheticTruth

Hidden DGP primitives from a synthetic fixture (`generate_fake_melitz_data`). Used ONLY
after `calibrate_melitz_pareto` has produced an observable-only calibration, to evaluate
recovery -- never as an input to calibration/estimation (main prompt Section 15/16).
"""
struct MelitzSyntheticTruth
    A::Matrix{Float64}
    f::Matrix{Float64}
    w::Vector{Float64}
    gamma_prime_target::Float64
    cutoff::Matrix{Float64}
    f_jj::Float64
    seed::Int
end

"""
    split_melitz_synthetic_truth(data::MelitzSyntheticData) -> (observed, truth)

Splits an EXISTING synthetic fixture (built by the legacy `generate_fake_melitz_data`
leaked-DGP generator, kept per main prompt Section 15's explicit permission to retain a
"clearly labeled legacy leaked-DGP path... for regression comparison") into
`MelitzObservedData` (trade shares implied by the population equilibrium, `L`, `tau` --
exactly what a real-data caller would have) and `MelitzSyntheticTruth` (everything else).
`data.equilibrium.trade_flow ./ sum(data.equilibrium.trade_flow, dims=1)` reconstructs the
observed shares from the population-Pareto trade flow -- this is what "observing the
economy's trade data" means for a synthetic fixture (a real dataset supplies these shares
directly, e.g. `pi.csv`). The normal Monte-Carlo recovery test (main prompt Section 15)
must discard/never read the returned `truth` until AFTER `calibrate_melitz_pareto` has run.
"""
function split_melitz_synthetic_truth(data::MelitzSyntheticData)
    p = data.primitives
    lam = data.equilibrium.trade_flow ./ sum(data.equilibrium.trade_flow, dims=1)
    observed = MelitzObservedData(; lambda=lam, L=data.L, tau=p.tau)
    truth = MelitzSyntheticTruth(p.A, p.f, p.w, p.gamma_prime_target, data.equilibrium.cutoff,
                                  p.f[p.target_country, p.target_country], data.seed)
    return observed, truth
end

# ============================================================================
# Section 4: data-only wage calibration
# ============================================================================

"""
    MelitzWageCalibration

Diagnostics for a data-only wage solve (main prompt Section 4, tightened by Phase 2 of the
2026-07-24 continuation session). The AUTHORITATIVE `w`/`E` come from a direct rank-D
linear solve (`melitz_solve_wages_linear`, machine-precision market clearing by
construction), NOT from the damped-Jacobi iteration, which real D=20 data was found to
floor around `diff~1.7e-7` regardless of `max_iter` (a genuine round-off floor of that
iterative scheme at this scale, `docs/melitz_pareto_data_calibration_2026-07-24.md` Section
H) -- an authoritative direct solve exists (this is a dense `D x D` linear system, trivial
at D=20), so the damped-Jacobi result is retained only as a secondary cross-check
(`w_damped`, `damped_residual`, `damped_iterations`), never as the calibration's own `w`.
An INDEPENDENT dense-eigendecomposition Perron-vector solve is a THIRD cross-check
(`perron_residual`) of the same fixed point (`lambda` is column-stochastic and nonnegative,
so by Perron-Frobenius its spectral radius is exactly 1 with a nonnegative right
eigenvector -- generically unique/positive if `lambda` is irreducible, true whenever every
country trades, directly or indirectly, with every other).
"""
struct MelitzWageCalibration
    w::Vector{Float64}
    E::Vector{Float64}
    numeraire::Int
    iterations::Int
    market_clearing_residual::Float64
    perron_eigenvalue::Float64
    perron_residual::Float64
    wage_range::Tuple{Float64,Float64}
    w_damped::Vector{Float64}
    damped_residual::Float64
    damped_iterations::Int
    damped_vs_direct_residual::Float64
end

"""
    melitz_solve_wages_linear(lambda, L; numeraire=1) -> (w, E)

Phase 2 of the 2026-07-24 continuation session: an AUTHORITATIVE direct solution of `E =
lambda*E` (equivalently `w_o*L_o = sum_d lambda_od*w_d*L_d`), replacing the damped-Jacobi
iteration's own convergence floor with an exact dense linear solve. `lambda` is column-
stochastic, so `(I - lambda)` has rank exactly `D-1` (a `Perron-Frobenius` null space of
dimension 1, generic irreducibility); the row corresponding to `numeraire` is therefore
REDUNDANT given the other `D-1` market-clearing equations (Walras' law: the `D` equations
`E = lambda*E` sum to an identity, `sum(E) = sum(lambda*E) = sum(E)`, so any one is implied
by the rest) and is replaced with the numeraire condition `E[numeraire] = L[numeraire]`
(equivalently `w[numeraire] = 1`) to pin the scale, giving a genuinely rank-D system solved
by ordinary Gaussian elimination -- no iteration, no convergence floor, market-clearing
residual at machine precision by construction (limited only by the linear solve's own
conditioning, not by any tolerance/max_iter choice).
"""
function melitz_solve_wages_linear(lambda::AbstractMatrix{<:Real}, L::AbstractVector{<:Real};
                                    numeraire::Int=1)
    D = length(L)
    1 <= numeraire <= D || throw(ArgumentError("numeraire out of range"))
    lam = Matrix{Float64}(lambda)
    Lv = Vector{Float64}(L)
    M = Matrix{Float64}(I, D, D) .- lam
    rhs = zeros(Float64, D)
    M[numeraire, :] .= 0.0
    M[numeraire, numeraire] = 1.0
    rhs[numeraire] = Lv[numeraire]
    E = M \ rhs
    w = E ./ Lv
    return w, E
end

"""
    calibrate_melitz_wages(lambda, L; numeraire=1, damping=0.6, tol=1e-8, max_iter=100_000)
        -> MelitzWageCalibration

Data-only baseline wage calibration (main prompt Section 4, Phase 2 tightened): `w_o*L_o =
sum_d lambda_od*w_d*L_d`, i.e. `E = lambda*E` for expenditure `E = w.*L`. The RETURNED
`.w`/`.E` come from `melitz_solve_wages_linear` (the authoritative direct solve, machine
precision by construction). `melitz_solve_wages` (`equilibrium.jl`, the damped-Jacobi
fixed point previously wired as THE calibration in an earlier session -- see
`docs/melitz_wage_calibration_gap_2026-07-24.md` for why it was dead code before that, and
`docs/melitz_pareto_data_calibration_2026-07-24.md` Section H for its real-D20 convergence
floor) and an INDEPENDENT dense-eigendecomposition Perron-vector solve are both retained as
CROSS-CHECKS ONLY -- `damped_residual`/`perron_residual`/`damped_vs_direct_residual` should
all be small; a large disagreement is a genuine red flag (e.g. `lambda` reducibility), not
something this function papers over.
"""
function calibrate_melitz_wages(lambda::AbstractMatrix{<:Real}, L::AbstractVector{<:Real};
                                 numeraire::Int=1, damping::Real=0.6, tol::Real=1e-8,
                                 max_iter::Int=100_000)
    D = length(L)
    1 <= numeraire <= D || throw(ArgumentError("numeraire out of range"))
    lam = Matrix{Float64}(lambda)
    Lv = Vector{Float64}(L)

    # AUTHORITATIVE: direct rank-D linear solve (Phase 2). Machine-precision market
    # clearing by construction -- no iteration, no tolerance/max_iter to tune.
    w, E = melitz_solve_wages_linear(lam, Lv; numeraire=numeraire)
    all(>(0), w) || error("calibrate_melitz_wages: direct linear solve gave a nonpositive wage -- infeasible/reducible lambda")
    mc_resid = maximum(abs.(E .- lam * E)) / maximum(abs.(E))

    # damped-Jacobi CROSS-CHECK ONLY (melitz_solve_wages internally normalizes w[1]=1; track
    # iterations by re-implementing the loop here so we can report `iterations` without
    # touching that function's signature).
    w0 = ones(Float64, D)
    w1 = copy(w0)
    diff = tol + 1
    iter = 1
    while diff > tol && iter < max_iter
        w0 .= w0 .* (1 - damping) .+ w1 .* damping
        w1 .= lam * (w0 .* Lv) ./ Lv
        diff = maximum(abs.(w1 .- w0))
        iter += 1
    end
    w_damped = w1 ./ w1[numeraire]
    E_damped = w_damped .* Lv
    damped_resid = maximum(abs.(E_damped .- lam * E_damped)) / maximum(abs.(E_damped))
    damped_vs_direct = maximum(abs.(w_damped .- w)) / maximum(abs.(w))

    # independent Perron/eigenvector cross-check (of the AUTHORITATIVE direct solve).
    F = eigen(lam)
    idx = argmin(abs.(F.values .- 1))
    perron_eigenvalue = real(F.values[idx])
    vec1 = real.(F.vectors[:, idx])
    sum(vec1) < 0 && (vec1 = -vec1)
    all(>=(-1e-8), vec1) || @warn "calibrate_melitz_wages: Perron eigenvector has a negative component (lambda may be reducible)"
    vec1_scaled = vec1 .* (E[numeraire] / vec1[numeraire])
    perron_residual = maximum(abs.(vec1_scaled .- E)) / maximum(abs.(E))

    return MelitzWageCalibration(w, E, numeraire, iter, mc_resid, perron_eigenvalue,
                                  perron_residual, (minimum(w), maximum(w)),
                                  w_damped, damped_resid, iter, damped_vs_direct)
end

# ============================================================================
# Section 5: theta_star/gravity compatibility
# ============================================================================

"""
    MelitzGravityDiagnostics

`theta_hat`: the data's own two-way-FE OLS gravity coefficient (`prestep/master_prestep.jl`'s
own formula, reused verbatim: `-sum(within(log lambda).*within(log tau)) /
sum(within(log tau).^2)`). `theta_star`: the value actually used. `compatible`: whether
they agree within `theta_tol`. `composite_gravity_residual`: `dot(c, vec(chi))` for the
identified composite `chi` (Section 6) -- algebraically `(theta_star-theta_hat)*dot(c,c)`,
reported as an independent cross-check (should be ~0 iff `compatible`).
"""
struct MelitzGravityDiagnostics
    theta_hat::Float64
    theta_star::Float64
    compatible::Bool
    theta_gap::Float64
    composite_gravity_residual::Float64
    c::Vector{Float64}
    cc::Float64
end

"""
    melitz_estimate_theta_hat(lambda, tau) -> theta_hat

The SINGLE canonical two-way-FE OLS gravity-coefficient estimator this repo uses, reused
verbatim (not re-derived) from `prestep/master_prestep.jl`'s `thetaIn==0` branch (Phase 1.2
audit: traced line-by-line, `moments/newGravityMoment!.jl`'s `UoModel==1` branch --
confirmed the universal production setting via `grep` across every run config -- computes
this SAME `withinTransform`-based bilinear form for the F-independent gravity/orthogonality
moment). `withinTransform` logs internally (`misc/doubleDiff.jl`), so `lambda`/`tau` are
passed in LEVELS here, matching `master_prestep.jl`'s own call convention exactly. No mask,
no weights, no ROW exclusion, no domestic-cell exclusion, flows-vs-shares: SHARES (not raw
flows) throughout -- the full `D x D` matrix (every origin-destination pair, including
domestic `o==d` and any ROW aggregate row/column) enters this one bilinear form. Called on
a FROZEN `MelitzObservedData`'s own `.lambda`/`.tau` (post whatever `share_policy`/
`tau_diagonal_policy` was applied), never on a raw pre-policy CSV read directly -- see
`calibrate_melitz_pareto`'s `theta_star=:estimate` path, which is the fix for the
main-prompt's Phase 1.1 finding (theta previously estimated on a DIFFERENT tau than the one
calibration used).
"""
function melitz_estimate_theta_hat(lambda::AbstractMatrix{<:Real}, tau::AbstractMatrix{<:Real})
    Wlambda = withinTransform(lambda)
    Wtau = withinTransform(tau)
    return -sum(Wlambda .* Wtau) / sum(Wtau .* Wtau)
end

"""
    melitz_gravity_theta_check(lambda, tau, theta_star; sigma, theta_tol=nothing,
                                w=nothing, E=nothing) -> MelitzGravityDiagnostics

Main prompt Section 5: checks whether an externally-supplied `theta_star` is compatible
with `(lambda, tau)`'s own two-way-FE gravity coefficient (`melitz_estimate_theta_hat`,
the SAME `withinTransform` convention `production/fullA-exact`'s own canonical gravity
moment uses -- re-derived here, not re-cited, and algebraically PROVEN (see module header /
accompanying report) to be EXACTLY the condition under which the two SEPARATE Melitz
gravity restrictions (`Cov(within(logtau),within(logA))=0` and
`Cov(within(logtau),within(logf))=0`) are jointly satisfiable for SOME cutoff-decomposition
`u`: `dot(c,vec(chi)) == (theta_star-theta_hat)*dot(c,c)`, `chi` the theta_star/beta
composite (Section 6), `c = gravity_coefficient_vector(D,tau)`. If `w`/`E` are supplied,
also computes `composite_gravity_residual` directly from the composite (an independent
numerical cross-check of the same algebraic identity, not merely re-stating
`theta_star-theta_hat`). Default `theta_tol = 1e-3*max(1,|theta_hat|)` (loose enough for
real-data gravity-estimation noise, tight enough to catch a genuinely incompatible
externally-supplied `theta_star`).
"""
function melitz_gravity_theta_check(lambda::AbstractMatrix{<:Real}, tau::AbstractMatrix{<:Real},
                                     theta_star::Real; sigma::Real,
                                     theta_tol::Union{Nothing,Real}=nothing,
                                     w::Union{Nothing,AbstractVector{<:Real}}=nothing,
                                     E::Union{Nothing,AbstractVector{<:Real}}=nothing)
    D = size(lambda, 1)
    theta_hat = melitz_estimate_theta_hat(lambda, tau)
    c = gravity_coefficient_vector(D, Matrix{Float64}(tau))
    cc = dot(c, c)
    tol = theta_tol === nothing ? 1e-3 * max(1.0, abs(theta_hat)) : theta_tol
    theta_gap = theta_star - theta_hat
    compatible = abs(theta_gap) < tol

    composite_residual = if w !== nothing && E !== nothing
        chi, _, _, _ = melitz_pareto_composite(lambda, w, E, tau, sigma, theta_star)
        dot(c, vec(chi))
    else
        theta_gap * cc  # algebraic identity, used when (w,E) not yet available
    end

    return MelitzGravityDiagnostics(theta_hat, theta_star, compatible, theta_gap,
                                     composite_residual, c, cc)
end

# ============================================================================
# Section 6: the identified Pareto composite chi_od = theta_star*logA + beta*logf
# ============================================================================

"""
    melitz_pareto_composite(lambda, w, E, tau, sigma, theta_star) -> (chi, a0, f0, beta)

Main prompt Section 6: derives the FULL identified bilateral composite from observables
alone (baseline `N_o=1`, `gamma_d=1`, `E_d=w_d*L_d`). From `B_od = M(q_od)/lambda_od`
(Section 7) and `A_od = mu_sigma*w_o*tau_od*B_od^{-1/(sigma-1)}`,
`f_od = E_d*lambda_od*q_od^{sigma-1}/(sigma*w_o*M(q_od))` (verified against
`melitz_firm`/`population_X`'s own convention in the accompanying report), `log A_od` and
`log f_od` are each AFFINE in `u_od = log q_od` (Section 8):
`log A_od = a0_od + (theta_star/(sigma-1)-1)*u_od`, `log f_od = f0_od + theta_star*u_od`.
The composite `chi_od = theta_star*log A_od + beta*log f_od` (`beta = 1 -
theta_star/(sigma-1)`) is therefore EXACTLY `u`-INVARIANT (the `u_od` coefficients cancel:
`theta_star*(theta_star/(sigma-1)-1) + beta*theta_star = 0`), hence computable directly
from `(lambda, w, E, tau, sigma, theta_star)` alone, with NO cutoff choice required:
`chi_od = theta_star*a0_od + beta*f0_od`.
"""
function melitz_pareto_composite(lambda::AbstractMatrix{<:Real}, w::AbstractVector{<:Real},
                                  E::AbstractVector{<:Real}, tau::AbstractMatrix{<:Real},
                                  sigma::Real, theta_star::Real)
    D = size(lambda, 1)
    mu = melitz_markup(sigma)
    beta = 1 - theta_star / (sigma - 1)
    log_M_ratio = log(theta_star / (theta_star - sigma + 1))
    a0 = zeros(D, D)
    f0 = zeros(D, D)
    chi = zeros(D, D)
    @inbounds for o in 1:D, d in 1:D
        a0[o, d] = log(mu * w[o] * tau[o, d]) - log_M_ratio / (sigma - 1) + log(lambda[o, d]) / (sigma - 1)
        f0[o, d] = log(E[d] * lambda[o, d] / (sigma * w[o])) - log_M_ratio
        chi[o, d] = theta_star * a0[o, d] + beta * f0[o, d]
    end
    return chi, a0, f0, beta
end

# ============================================================================
# Section 7/8: cell-by-cell Pareto inversion given a cutoff matrix
# ============================================================================

"""
    melitz_ad_from_cutoffs(lambda, w, E, tau, sigma, theta_star, u) -> (A, f)

Main prompt Section 7/8: for ANY admissible `u = log q` (cutoff matrix), inverts the
Pareto trade-share equation cell by cell: `B_od = M(q_od)/lambda_od`, `A_od =
mu_sigma*w_o*tau_od*B_od^{-1/(sigma-1)}`, `f_od = E_d*lambda_od*q_od^{sigma-1}/
(sigma*w_o*M(q_od))`. By construction (verified in the test suite), `population_X(w, L,
tau, A, f, sigma, theta_star)` applied to the RETURNED `(A,f)` reproduces `lambda_od`
EXACTLY and the resulting cutoff EXACTLY equals `q_od = exp(u_od)`, for every choice of
admissible `u`.
"""
function melitz_ad_from_cutoffs(lambda::AbstractMatrix{<:Real}, w::AbstractVector{<:Real},
                                 E::AbstractVector{<:Real}, tau::AbstractMatrix{<:Real},
                                 sigma::Real, theta_star::Real, u::AbstractMatrix{<:Real})
    D = size(lambda, 1)
    mu = melitz_markup(sigma)
    A = zeros(D, D)
    f = zeros(D, D)
    @inbounds for o in 1:D, d in 1:D
        q_od = exp(u[o, d])
        Mq = pareto_tail_power_mean(q_od, sigma, theta_star)
        B = Mq / lambda[o, d]
        A[o, d] = mu * w[o] * tau[o, d] * B^(-1 / (sigma - 1))
        f[o, d] = E[d] * lambda[o, d] * q_od^(sigma - 1) / (sigma * w[o] * Mq)
    end
    return A, f
end

# ============================================================================
# Section 9: transparent, configurable cutoff calibration (convex QP)
# ============================================================================

"""
    melitz_cutoff_target(D, theta_star; policy=:uniform, p_domestic=0.3, p_export=0.1,
                          lambda=nothing) -> u_target (D x D)

Main prompt Section 9.2: an explicit, NON-identified target cutoff matrix used only as the
QP's proximity objective (never claimed to be identified from aggregate trade data).
`policy=:uniform`: a common target domestic participation probability `p_domestic` and a
lower common target export probability `p_export` (`u_od = -log(p)/theta_star`, from
`Pr(z>=q)=q^{-theta_star}`). `policy=:share_informed`: the SAME domestic target, but the
export target probability is nudged upward for cells with a large OBSERVED share relative
to their destination's other origins (larger observed share -> lower implied cutoff target
-> more numerical headroom) -- purely a numerical-conditioning choice (Section 10), not an
economic restriction.
"""
function melitz_cutoff_target(D::Int, theta_star::Real; policy::Symbol=:uniform,
                               p_domestic::Real=0.3, p_export::Real=0.1,
                               lambda::Union{Nothing,AbstractMatrix{<:Real}}=nothing)
    u = zeros(D, D)
    if policy == :uniform
        for o in 1:D, d in 1:D
            p = (o == d) ? p_domestic : p_export
            u[o, d] = -log(p) / theta_star
        end
    elseif policy == :share_informed
        lambda === nothing && throw(ArgumentError("policy=:share_informed requires `lambda`"))
        for o in 1:D, d in 1:D
            if o == d
                u[o, d] = -log(p_domestic) / theta_star
            else
                rel = lambda[o, d] / maximum(@view lambda[:, d])
                p = clamp(p_export * (1 + rel), 1e-6, 0.99)
                u[o, d] = -log(p) / theta_star
            end
        end
    else
        throw(ArgumentError("melitz_cutoff_target: unknown policy $policy"))
    end
    return u
end

"""
    MelitzCutoffCalibration

`u`: the calibrated D x D log-cutoff matrix. `objective`: the QP's optimal value.
`gravity_rhs_A`/`gravity_rhs_f`: the two target levels for `dot(c,vec(u))` implied
separately by the A- and f-gravity restrictions (should agree to `gravity_tol` --
`melitz_gravity_theta_check`'s `compatible` flag is the pre-check for this); `rhs_used`
is what was actually imposed (their average). `active_domestic`/`active_export`: indices
where the corresponding inequality is active (within `active_tol`) at the solution.
`p_min_used`: the participation-probability upper bound, if any.
"""
struct MelitzCutoffCalibration
    u::Matrix{Float64}
    objective::Float64
    gravity_rhs_A::Float64
    gravity_rhs_f::Float64
    rhs_used::Float64
    gravity_rank::Int
    gravity_singular_values::Vector{Float64}
    active_domestic::Vector{Int}
    active_export::Vector{Tuple{Int,Int}}
    p_min_used::Union{Nothing,Float64}
    termination_status::Symbol
end

"""
    calibrate_melitz_cutoffs(lambda, tau, w, E, sigma, theta_star; u_target,
        weights=ones(D,D), eps_support=1e-6, eps_export=1e-6, p_min=nothing,
        gravity_tol=1e-6, a0=nothing, f0=nothing) -> MelitzCutoffCalibration

Main prompt Section 9/9.1/9.2: solves
`min 0.5*sum(weights.*(u-u_target).^2)` subject to `u_od >= eps_support`,
`u_od - u_oo >= eps_export` (`d != o`), the gravity-restriction linear constraint(s), and
(if `p_min` given) `u_od <= -log(p_min)/theta_star`.

Rank-revealing check (Section 9.1): both gravity restrictions are linear in `u` with the
SAME coefficient vector `c = gravity_coefficient_vector(D,tau)` (verified algebraically --
`log A(u)`/`log f(u)` are both affine in `u` with SCALAR coefficients, so
`Cov(within(logtau),within(logA(u)))` and `Cov(within(logtau),within(logf(u)))` are both
`(data term) + (scalar)*dot(c,vec(u))`); the 2xD^2 "constraint matrix" `[kappa_A*c';
theta_star*c']` is therefore rank <= 1 by construction, confirmed here via `svd`. If the
two implied RHS targets (`gravity_rhs_A`, `gravity_rhs_f`) disagree by more than
`gravity_tol` (relative to `dot(c,c)`), this is exactly the "externally supplied theta_star
incompatible with both restrictions" failure mode (Section 5) -- this function THROWS
rather than silently averaging incompatible targets away.
"""
function calibrate_melitz_cutoffs(lambda::AbstractMatrix{<:Real}, tau::AbstractMatrix{<:Real},
                                   w::AbstractVector{<:Real}, E::AbstractVector{<:Real},
                                   sigma::Real, theta_star::Real;
                                   u_target::AbstractMatrix{<:Real},
                                   weights::AbstractMatrix{<:Real}=ones(size(lambda)),
                                   eps_support::Real=1e-6, eps_export::Real=1e-6,
                                   p_min::Union{Nothing,Real}=nothing,
                                   gravity_tol::Real=1e-6,
                                   a0::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                                   f0::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                                   fixed::Union{Nothing,Dict{Tuple{Int,Int},Float64}}=nothing)
    D = size(lambda, 1)
    c = gravity_coefficient_vector(D, Matrix{Float64}(tau))
    kappa_A = theta_star / (sigma - 1) - 1

    if a0 === nothing || f0 === nothing
        _, a0c, f0c, _ = melitz_pareto_composite(lambda, w, E, tau, sigma, theta_star)
        a0 = a0 === nothing ? a0c : a0
        f0 = f0 === nothing ? f0c : f0
    end

    rhs_A = -dot(c, vec(a0)) / kappa_A
    rhs_f = -dot(c, vec(f0)) / theta_star
    cc = dot(c, c)
    rel_gap = abs(rhs_A - rhs_f) * cc / max(1.0, abs(rhs_A), abs(rhs_f))
    rel_gap < gravity_tol || throw(ArgumentError(
        "calibrate_melitz_cutoffs: A-gravity and f-gravity restrictions imply DIFFERENT " *
        "target levels for dot(c,vec(u)) (rhs_A=$rhs_A, rhs_f=$rhs_f, scaled gap=$rel_gap > " *
        "gravity_tol=$gravity_tol) -- theta_star is not compatible with both restrictions " *
        "holding simultaneously (main prompt Section 5). Re-estimate theta_star from the " *
        "same gravity moment (melitz_gravity_theta_check) or supply a compatible value."))
    rhs_used = (rhs_A + rhs_f) / 2

    M = vcat((kappa_A .* c)', (theta_star .* c)')
    sv = svd(M).S
    grank = count(>(1e-10 * max(sv[1], 1.0)), sv)

    fixed_set = fixed === nothing ? Dict{Tuple{Int,Int},Float64}() : fixed
    fixed_lin = Dict{Int,Float64}(od2lin(o, d, D) => val for ((o, d), val) in fixed_set)

    # Eliminate the single gravity equality constraint ANALYTICALLY (a `GravityPivot`-style
    # substitution, exactly the mechanism `equilibrium.jl`/`delta_star.jl` already use for the
    # SAME kind of restriction elsewhere in this repo) rather than passing a dense D^2-length
    # equality row to the QP solver directly. This is both more consistent with established
    # practice AND, found live at D=20 (400 variables), materially more robust: HiGHS's QP
    # solver reliably returns `OTHER_ERROR` on this problem's dense single-row equality
    # constraint at that scale (reproduced with a MINIMAL isolated example -- box constraints
    # alone solve fine, the dense equality row alone triggers the failure -- not a conditioning
    # artifact of `c` itself, whose condition number here is a modest ~1.2e4), while the
    # SAME restriction expressed as one variable eliminated in closed form (leaving only
    # inequality constraints for the QP) solves instantly at every `D` tried, D=4 through D=20.
    offdiag_candidates = [k for k in 1:D^2 if !haskey(fixed_lin, k) && lin2od(k, D)[1] != lin2od(k, D)[2]]
    all_candidates = [k for k in 1:D^2 if !haskey(fixed_lin, k)]
    isempty(all_candidates) && throw(ArgumentError("calibrate_melitz_cutoffs: every cell is fixed, nothing to calibrate"))
    pivot_candidates = isempty(offdiag_candidates) ? all_candidates : offdiag_candidates
    pivot = pivot_candidates[argmax(abs.(c[pivot_candidates]))]
    free = [k for k in 1:D^2 if k != pivot && !haskey(fixed_lin, k)]
    fixed_contrib = isempty(fixed_lin) ? 0.0 : sum(c[k] * val for (k, val) in fixed_lin)

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, uf[free])
    pivot_expr = @expression(model, (rhs_used - fixed_contrib - sum(c[k] * uf[k] for k in free)) / c[pivot])

    u_expr(k::Int) = haskey(fixed_lin, k) ? fixed_lin[k] : (k == pivot ? pivot_expr : uf[k])

    @objective(model, Min, 0.5 * sum(weights[lin2od(k, D)...] * (u_expr(k) - u_target[lin2od(k, D)...])^2 for k in 1:D^2))
    for o in 1:D
        kd = od2lin(o, o, D)
        haskey(fixed_lin, kd) && continue
        @constraint(model, u_expr(kd) >= eps_support)
    end
    for o in 1:D, d in 1:D
        d == o && continue
        kd, ko = od2lin(o, d, D), od2lin(o, o, D)
        haskey(fixed_lin, kd) && continue
        @constraint(model, u_expr(kd) - u_expr(ko) >= eps_export)
    end
    if p_min !== nothing
        u_max = -log(p_min) / theta_star
        for k in 1:D^2
            haskey(fixed_lin, k) && continue
            @constraint(model, u_expr(k) <= u_max)
        end
    end
    optimize!(model)
    status = termination_status(model)
    status in (OPTIMAL, LOCALLY_SOLVED) || error("calibrate_melitz_cutoffs: QP did not solve to optimality (status=$status)")

    u_sol = zeros(D, D)
    for k in 1:D^2
        o, d = lin2od(k, D)
        u_sol[o, d] = haskey(fixed_lin, k) ? fixed_lin[k] : (k == pivot ? value(pivot_expr) : value(uf[k]))
    end
    active_domestic = [o for o in 1:D if u_sol[o, o] - eps_support < 1e-7]
    active_export = [(o, d) for o in 1:D, d in 1:D if d != o && (u_sol[o, d] - u_sol[o, o] - eps_export) < 1e-7]

    return MelitzCutoffCalibration(u_sol, objective_value(model), rhs_A, rhs_f, rhs_used,
                                    grank, sv, active_domestic, active_export,
                                    p_min === nothing ? nothing : Float64(p_min), :OPTIMAL)
end

# ============================================================================
# Section 10: conditioning diagnostics
# ============================================================================

"""
    melitz_conditioning_diagnostics(p::MelitzPrimitives, eq::MelitzEquilibrium, cf, W;
        seed=1, draw_mode=:halton) -> NamedTuple

Main prompt Section 10: cutoff spread within each origin, participation probabilities,
active draw counts at the requested `W`, and the QMC moment matrix's rank / singular
values / condition number (built via `melitz_moments!`, the SAME machinery the real inner
solve uses -- never a separate ad hoc moment construction).
"""
function melitz_conditioning_diagnostics(p::MelitzPrimitives, eq::MelitzEquilibrium,
                                          cf::MelitzCounterfactual, W::Int;
                                          seed::Int=1, draw_mode::Symbol=:halton)
    D = p.D
    z_draws = pareto_draws(W, D, p.theta_star; seed=seed, mode=draw_mode)
    layout = MelitzMomentLayout(D)
    K = zeros(W)
    G = zeros(W, layout.num_moments)
    melitz_moments!(K, G, p, eq, cf, z_draws, layout)

    sv = svd(G).S
    grank = count(>(1e-10 * max(sv[1], 1.0)), sv)
    cond_number = sv[1] / sv[end]

    cutoff_spread = [(maximum(@view eq.cutoff[o, :]) - minimum(@view eq.cutoff[o, :])) / (sum(@view eq.cutoff[o, :]) / D) for o in 1:D]
    ref_prob = [pareto_tail_prob(eq.cutoff[o, d], p.theta_star) for o in 1:D, d in 1:D]
    min_count, worst_cell = min_active_draw_count(p, eq, z_draws)

    return (W=W, cutoff_spread_by_origin=cutoff_spread, reference_probability=ref_prob,
            min_active_count=min_count, worst_cell=worst_cell,
            moment_matrix_rank=grank, moment_matrix_size=size(G),
            singular_values=sv, condition_number=cond_number, z_draws=z_draws, G=G)
end

# ============================================================================
# Section 11: entry-cost profiling
# ============================================================================

"""
    melitz_profile_entry_costs(w, tau, A, f, sigma, theta_star, X, q) -> f_E (D-vector)

Main prompt Section 11: profiles every origin's entry cost from the baseline free-entry
condition `w_o*f_Eo = E[operating profit of an entrant from o]`, using the SAME
closed-form population integral `equilibrium.jl`'s `population_baseline_focal_profit_integral`
uses for the focal country, generalized to every origin. `X`, `q` are `population_X`'s
already-computed baseline trade flow / cutoff matrices at the calibrated `(w,A,f)`.
Requires `f_E[o] > 0` for every `o` (checked by the caller, main prompt Section 20).
"""
function melitz_profile_entry_costs(w::AbstractVector{<:Real}, sigma::Real,
                                     f::AbstractMatrix{<:Real}, theta_star::Real,
                                     X::AbstractMatrix{<:Real}, q::AbstractMatrix{<:Real})
    D = length(w)
    f_E = zeros(D)
    for o in 1:D
        s = sum(X[o, d] / sigma - w[o] * f[o, d] * pareto_tail_prob(q[o, d], theta_star) for d in 1:D)
        f_E[o] = s / w[o]
    end
    return f_E
end

# ============================================================================
# Section 13: autarky reference point (data-only, no gamma_prime_target leak)
# ============================================================================

"""
    melitz_autarky_reference(A, f, w, L, sigma, target_country) -> (gamma_prime_j, cf)

Main prompt Section 13: given the ALREADY-CALIBRATED baseline `(A, f, w)` (Sections 6-9 --
in particular `f[j,j]` is the value the general cell-by-cell inversion (Section 7) gives at
whatever `u_jj` the cutoff-calibration QP chose for the DOMESTIC cell of the focal country,
matching the OBSERVED domestic trade share `lambda_jj` exactly, main prompt Section 12
point 1), solves the focal country's autarky price power `gamma_prime_j` from the
`zhat'_jj=1` cutoff-at-one normalization (`derive_fjj_from_autarky_cutoff`'s OWN defining
equation, algebraically inverted for `gamma_prime_j` given `f_jj` instead of the other way
around -- CLOSED FORM, not a search, since `f_jj` is TREATED AS FIXED by this function --
callers that also need the free-entry LINK condition
(`population_focal_link_residual == 0`, required for `GT_model == GT_ACR`, main prompt
Section 13's non-negotiable check) must use `calibrate_melitz_cutoffs_and_autarky`, which
calls this function repeatedly inside a 1-D search over the focal domestic cutoff `u_jj`
(hence over `f_jj`) rather than treating `f_jj` as fixed data. This function on its own is
the CLOSED-FORM half of that search's inner step -- not, by itself, a caller-facing
guarantee that the link condition holds.
"""
function melitz_autarky_reference(A::AbstractMatrix{<:Real}, f::AbstractMatrix{<:Real},
                                   w::AbstractVector{<:Real}, L::AbstractVector{<:Real},
                                   sigma::Real, target_country::Int)
    j = target_country
    A_jj = A[j, j]
    f_jj = f[j, j]
    w_prime_j = 1.0
    expenditure_prime_j = w_prime_j * L[j]
    mu = melitz_markup(sigma)
    p_prime_at_one = mu * w_prime_j * 1.0 / A_jj
    gamma_prime_j = expenditure_prime_j * p_prime_at_one^(1 - sigma) / (sigma * w_prime_j * f_jj)
    gamma_prime_j > 0 || error("melitz_autarky_reference: derived gamma_prime_j <= 0 -- infeasible calibration")
    cf = MelitzCounterfactual(j, w_prime_j, expenditure_prime_j, 1.0, expenditure_prime_j)
    return gamma_prime_j, cf
end

"""
    calibrate_melitz_cutoffs_and_autarky(lambda, tau, w, E, L, sigma, theta_star,
        target_country; u_target, weights, eps_support, eps_export, p_min, gravity_tol,
        u_jj_bracket=nothing) -> (u, A, f, gamma_prime_j, cf, link_residual, cutoffcal)

Joint calibration of the cutoff matrix AND the autarky reference point, needed because the
FOCAL country's domestic cell `(j,j)` is doubly constrained under a data-only calibration:
(1) it must reproduce the OBSERVED domestic share `lambda_jj` exactly (Section 12 point 1,
via the Section-7 inversion, for WHATEVER `u_jj` is chosen), and (2) the classic
Arkolakis-Costinot-Rodriguez-Clare sufficient-statistic identity `GT_model == GT_ACR`
(Section 13's "non-negotiable end-to-end validation") requires the SAME entry cost to
rationalize both the baseline and the autarky free-entry conditions for country `j`
(`population_focal_link_residual == 0`), NOT merely the `zhat'_jj=1` normalization alone.
With `f_jj` pinned by (1), `gamma_prime_j`'s closed-form solve from cutoff-at-one
(`melitz_autarky_reference`) is therefore generally INCONSISTENT with (2) unless `u_jj`
itself is chosen to make it so -- this mirrors the coupled A-pivot/f-pivot cell pattern
already documented in this repo's legacy `fstar_solver.jl` (`docs/melitz_delta_star.md`
Section 11), here resolved via a 1-D bisection (`Roots.jl`) on `u_jj` (NOT a hand-tuned
pivot): at each trial `u_jj`, the OTHER `D^2-1` free cells are recalibrated via
`calibrate_melitz_cutoffs` with `u_jj` held fixed (so the SAME single gravity constraint
holds at every trial), `A`/`f` are inverted, `gamma_prime_j` solved via
`melitz_autarky_reference`, and the resulting `population_focal_link_residual` is the
bisection's root function. This still matches data exactly at EVERY cell (the QP always
reproduces `lambda` exactly via the Section-7 inversion, for any `u_jj`) while achieving
`GT_model == GT_ACR` to the tolerance the bisection converges to.
"""
function calibrate_melitz_cutoffs_and_autarky(lambda::AbstractMatrix{<:Real}, tau::AbstractMatrix{<:Real},
        w::AbstractVector{<:Real}, E::AbstractVector{<:Real}, L::AbstractVector{<:Real},
        sigma::Real, theta_star::Real, target_country::Int;
        u_target::AbstractMatrix{<:Real}, weights::AbstractMatrix{<:Real}=ones(size(lambda)),
        eps_support::Real=1e-6, eps_export::Real=1e-6, p_min::Union{Nothing,Real}=nothing,
        gravity_tol::Real=1e-6, u_jj_bracket::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
        xatol::Real=1e-10)
    D = size(lambda, 1)
    j = target_country

    function solve_given_ujj(u_jj::Real)
        fixed = Dict{Tuple{Int,Int},Float64}((j, j) => u_jj)
        cutoffcal = calibrate_melitz_cutoffs(lambda, tau, w, E, sigma, theta_star;
            u_target=u_target, weights=weights, eps_support=eps_support, eps_export=eps_export,
            p_min=p_min, gravity_tol=gravity_tol, fixed=fixed)
        A, f = melitz_ad_from_cutoffs(lambda, w, E, tau, sigma, theta_star, cutoffcal.u)
        gamma_prime_j, cf = melitz_autarky_reference(A, f, w, L, sigma, j)
        X, q, _ = population_X(w, L, tau, A, f, sigma, theta_star)
        p_tmp = MelitzPrimitives(D, sigma, theta_star, j, tau, w, A, f, gamma_prime_j)
        link_resid = population_focal_link_residual(p_tmp, X, q, f[j, j], cf)
        return (cutoffcal=cutoffcal, A=A, f=f, gamma_prime_j=gamma_prime_j, cf=cf, link_resid=link_resid)
    end

    # Default bracket kept safely INSIDE the p_min upper bound (if any): the domestic cell's
    # own upper bound is u_max_true = -log(p_min)/theta_star, but the focal country's EXPORT
    # cells also need `u[j,d] >= u_jj + eps_export` while ALSO respecting `u[j,d] <= u_max_true`
    # -- so u_jj itself must stay well below u_max_true, not just below it, or the inner QP at
    # the bracket's upper endpoint is infeasible (export-selection vs. participation-bound
    # conflict). Halving leaves ample room for any economically reasonable trade elasticity.
    u_max_bound = p_min === nothing ? 6.0 : 0.5 * (-log(p_min) / theta_star)
    bracket = u_jj_bracket === nothing ? (max(1000 * eps_support, 1e-3), u_max_bound) : u_jj_bracket
    u_jj_star = find_zero(u_jj -> solve_given_ujj(u_jj).link_resid, bracket, Bisection(); xatol=xatol)

    result = solve_given_ujj(u_jj_star)
    return result.cutoffcal.u, result.A, result.f, result.gamma_prime_j, result.cf, result.link_resid, result.cutoffcal
end

# ============================================================================
# Section 12: population baseline-equilibrium verification
# ============================================================================

"""
    MelitzBaselineEquilibriumCheck

Population-level (closed-form, no Monte Carlo) residuals for every restriction main prompt
Section 12 lists. `residual_shares`: `model lambda - data lambda` (should be ~0 to machine
precision -- Section 7's inversion is EXACT by construction). `residual_gamma`: `sum_o
model_lambda_od - 1` for every `d`. `residual_market_clearing`: `w.*L -
rowsum(X)`. `residual_cutoff_reconstruction`: reconstructed cutoff (via `melitz_baseline_cutoff`
on the calibrated `A,f`) minus the calibrated `u`'s own `q = exp(u)`. `min_support`/
`min_export_minus_domestic`: the two feasibility inequalities (Section 12 points 5/6).
`f_E`: profiled entry costs (Section 11, point 7 -- by construction `w_o*f_Eo ==
E[profit_o]` exactly, so `residual_free_entry` is ~0 machine-precision by definition, not
an independent check). `gravity_residual_A`/`gravity_residual_f`: point 8.
"""
struct MelitzBaselineEquilibriumCheck
    residual_shares::Matrix{Float64}
    residual_gamma::Vector{Float64}
    residual_market_clearing::Vector{Float64}
    residual_cutoff_reconstruction::Matrix{Float64}
    min_support::Float64
    min_export_minus_domestic::Float64
    f_E::Vector{Float64}
    residual_free_entry::Vector{Float64}
    gravity_residual_A::Float64
    gravity_residual_f::Float64
end

"""
    melitz_verify_baseline_equilibrium(lambda, L, tau, w, A, f, sigma, theta_star, u)
        -> (X, q, check::MelitzBaselineEquilibriumCheck)

Main prompt Section 12: verifies every population baseline-equilibrium equation at the
calibrated primitives, using ONLY analytical/closed-form population formulas
(`population_X`, `gravity_residuals`, `melitz_baseline_cutoff`,
`melitz_deterministic_cutoff_constraints`, `melitz_profile_entry_costs`) -- no Monte Carlo
here (the finite-QMC Delta-vs-W check is a separate step, `melitz_conditioning_diagnostics`
+ a real inner KNITRO solve, main prompt Section 12's second paragraph).
"""
function melitz_verify_baseline_equilibrium(lambda::AbstractMatrix{<:Real}, L::AbstractVector{<:Real},
                                             tau::AbstractMatrix{<:Real}, w::AbstractVector{<:Real},
                                             A::AbstractMatrix{<:Real}, f::AbstractMatrix{<:Real},
                                             sigma::Real, theta_star::Real, u::AbstractMatrix{<:Real})
    D = length(w)
    X, q, E = population_X(w, L, tau, A, f, sigma, theta_star)
    model_lambda = X ./ sum(X, dims=1)
    residual_shares = model_lambda .- lambda
    residual_gamma = vec(sum(model_lambda, dims=1)) .- 1.0
    residual_market_clearing = w .* L .- vec(sum(X, dims=2))

    zhat_reconstructed = melitz_baseline_cutoff(A, f, w, tau, E, sigma)
    residual_cutoff_reconstruction = zhat_reconstructed .- exp.(u)

    g_domestic, g_export = melitz_deterministic_cutoff_constraints(zhat_reconstructed)
    min_support = minimum(g_domestic)
    min_export_minus_domestic = minimum(g_export)

    f_E = melitz_profile_entry_costs(w, sigma, f, theta_star, X, q)
    residual_free_entry = zeros(D)  # definitional, exactly 0 by construction of f_E

    p_tmp = MelitzPrimitives(D, sigma, theta_star, 1, tau, w, A, f, 1.0)
    gravity_residual_A, gravity_residual_f = gravity_residuals(p_tmp)

    check = MelitzBaselineEquilibriumCheck(residual_shares, residual_gamma, residual_market_clearing,
        residual_cutoff_reconstruction, min_support, min_export_minus_domestic, f_E,
        residual_free_entry, gravity_residual_A, gravity_residual_f)
    return X, q, check
end

# ============================================================================
# Section 14: MelitzParetoCalibration + calibrate_melitz_pareto (top-level API)
# ============================================================================

"""
    MelitzParetoCalibration

The single authoritative, observable-only calibrated Pareto reference model (main prompt
Section 14). Every field is either directly OBSERVED (`lambda`, `L`, `tau`) or CALIBRATED
from observables (everything else) -- never a hidden-truth object.
"""
struct MelitzParetoCalibration
    D::Int
    target_country::Int
    sigma::Float64
    theta_star::Float64
    lambda::Matrix{Float64}
    L::Vector{Float64}
    tau::Matrix{Float64}
    w::Vector{Float64}
    E::Vector{Float64}
    u::Matrix{Float64}
    q::Matrix{Float64}
    A::Matrix{Float64}
    f::Matrix{Float64}
    f_jj::Float64
    f_entry::Vector{Float64}
    gamma_prime_target::Float64
    w_prime::Float64
    X::Matrix{Float64}
    wage::MelitzWageCalibration
    gravity::MelitzGravityDiagnostics
    cutoff_calibration::MelitzCutoffCalibration
    equilibrium_check::MelitzBaselineEquilibriumCheck
    free_entry_link_residual::Float64
    seed::Int
end

"""
    calibrate_melitz_pareto(observed::MelitzObservedData; sigma, theta_star, focal_country,
        wage_numeraire=1, cutoff_policy=:uniform, cutoff_target_kwargs=(;),
        cutoff_weights=nothing, p_min=nothing, eps_support=1e-6, eps_export=1e-6,
        gravity_tol=1e-6, theta_tol=nothing, seed=1) -> MelitzParetoCalibration

Main prompt Section 14 (Phase 1.1 of the 2026-07-24 continuation session fixes the
ORDERING bug documented in that session's governing prompt): runs, in order:
construct/freeze `observed` (caller's responsibility, done before this function is ever
called) -> wage calibration (Section 4) -> theta estimation, IF `theta_star=:estimate`,
from `observed.lambda`/`observed.tau` THEMSELVES (never from a pre-policy raw CSV read --
see `melitz_estimate_theta_hat`'s docstring) -> theta/gravity compatibility check
(Section 5) -> the identified composite (Section 6, folded into the cutoff QP's gravity
constraint, Section 9) -> the cutoff-calibration QP (Section 9) -> cell-by-cell A/f
inversion (Section 7/8) -> entry-cost profiling (Section 11) -> the autarky reference point
(Section 13) -> population equilibrium verification (Section 12). THROWS with a clear
diagnostic (not a silent fallback) if theta_star is gravity-incompatible
(`calibrate_melitz_cutoffs`) or the calibration is economically infeasible
(negative `gamma_prime_j`, `f_E`, or violated support/export-selection constraints).

`theta_star=:estimate` (recommended for any real-data caller, main prompt Phase 1.1's own
preferred API): estimates `theta_hat` from `observed.lambda`/`observed.tau` -- the SAME
frozen object every other step of this function reads -- so the two gravity restrictions
(Section 9.1) are compatible with `theta_star` BY CONSTRUCTION, to numerical precision, not
merely within a loosened `gravity_tol`. Passing an explicit real `theta_star` still works
unchanged (e.g. a synthetic fixture's own known/assumed elasticity) and is still checked
for compatibility against `observed`'s own gravity coefficient exactly as before.
"""
function calibrate_melitz_pareto(observed::MelitzObservedData; sigma::Real,
                                  theta_star::Union{Real,Symbol},
                                  focal_country::Int, wage_numeraire::Int=1,
                                  wage_tol::Real=1e-8, wage_max_iter::Int=100_000,
                                  cutoff_policy::Symbol=:uniform,
                                  cutoff_target_kwargs::NamedTuple=NamedTuple(),
                                  cutoff_weights::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                                  p_min::Union{Nothing,Real}=nothing,
                                  eps_support::Real=1e-6, eps_export::Real=1e-6,
                                  gravity_tol::Real=1e-6, theta_tol::Union{Nothing,Real}=nothing,
                                  seed::Int=1)
    sigma > 1 || throw(ArgumentError("sigma must be > 1"))
    D = observed.D
    1 <= focal_country <= D || throw(ArgumentError("focal_country out of range"))
    lambda, L, tau = observed.lambda, observed.L, observed.tau

    # Phase 1.1 fix: theta_star=:estimate is computed HERE, from the FROZEN observed.lambda/
    # observed.tau (post whatever share_policy/tau_diagonal_policy was applied at construction)
    # -- never from a raw pre-policy CSV read passed in by the caller. This is the exact
    # ordering the main prompt's governing addendum requires (construct observed -> estimate
    # theta from observed -> calibrate from the SAME observed).
    theta_star_val = theta_star === :estimate ? melitz_estimate_theta_hat(lambda, tau) : Float64(theta_star)
    theta_star_val > sigma - 1 || throw(ArgumentError("theta_star must be > sigma-1, got theta_star=$theta_star_val"))

    wage = calibrate_melitz_wages(lambda, L; numeraire=wage_numeraire, tol=wage_tol, max_iter=wage_max_iter)
    w, E = wage.w, wage.E

    grav0 = melitz_gravity_theta_check(lambda, tau, theta_star_val; sigma=sigma, theta_tol=theta_tol, w=w, E=E)
    grav0.compatible || throw(ArgumentError(
        "calibrate_melitz_pareto: theta_star=$theta_star_val is NOT compatible with the data's " *
        "own two-way-FE gravity coefficient theta_hat=$(grav0.theta_hat) (gap=$(grav0.theta_gap)) " *
        "-- main prompt Section 5. Pass theta_star=:estimate, or investigate the mismatch " *
        "before proceeding."))

    u_target = melitz_cutoff_target(D, theta_star_val; policy=cutoff_policy, lambda=lambda, cutoff_target_kwargs...)
    weights = cutoff_weights === nothing ? ones(D, D) : cutoff_weights
    u, A, f, gamma_prime_j, cf, link_resid, cutoffcal = calibrate_melitz_cutoffs_and_autarky(
        lambda, tau, w, E, L, sigma, theta_star_val, focal_country; u_target=u_target,
        weights=weights, eps_support=eps_support, eps_export=eps_export, p_min=p_min, gravity_tol=gravity_tol)

    X, q, check = melitz_verify_baseline_equilibrium(lambda, L, tau, w, A, f, sigma, theta_star_val, u)
    check.min_support >= -1e-6 || error("calibrate_melitz_pareto: calibrated cutoffs violate zhat>=1 (min=$(check.min_support))")
    check.min_export_minus_domestic >= -1e-6 || error("calibrate_melitz_pareto: calibrated cutoffs violate export-selection")
    all(>(0), check.f_E) || error("calibrate_melitz_pareto: profiled entry cost f_E has a nonpositive entry")

    return MelitzParetoCalibration(D, focal_country, sigma, theta_star_val, lambda, L, tau, w, E,
        u, q, A, f, f[focal_country, focal_country], check.f_E, gamma_prime_j, cf.w_prime, X,
        wage, grav0, cutoffcal, check, link_resid, seed)
end

# ============================================================================
# Wiring into the CC inner-loop machinery (mirrors build_melitz_psi_bundle, but
# consumes a MelitzParetoCalibration -- never a MelitzSyntheticData/hidden truth).
# ============================================================================

"""
    melitz_calibration_outer_ctx(calib::MelitzParetoCalibration; outer_parameterization=:logf,
        inner_loop_opt=..., outer_loop_opt=...) -> (p, eq, cf, ctx)

Shared builder (factored out so `build_melitz_psi_bundle_from_calibration` and the Phase
1.4 roundtrip check, `melitz_calibration_roundtrip_check`, cannot silently drift apart)
for the `MelitzPrimitives`/outer `ctx` pair from a calibration, with NO Monte Carlo draws
and NO `PsiObjectiveBundleDelta` construction -- everything `melitz_reduce_theta`/
`melitz_expand_theta` need and nothing more.
"""
function melitz_calibration_outer_ctx(calib::MelitzParetoCalibration;
        outer_parameterization::Symbol=:logf,
        inner_loop_opt::String=joinpath(dirname(dirname(@__DIR__)), "ek_inner_loop_options.opt"),
        outer_loop_opt::String=joinpath(dirname(dirname(@__DIR__)), "ek_outer_loop_options.opt"),
        moment_backend::Symbol=:dense_reference, z_draws::Union{Nothing,AbstractMatrix}=nothing)
    moment_backend in (:dense_reference, :sorted_tail_serial) || throw(ArgumentError(
        "melitz_calibration_outer_ctx: moment_backend must be :dense_reference or " *
        ":sorted_tail_serial, got $moment_backend"))
    D = calib.D
    j = calib.target_country
    p = MelitzPrimitives(D, calib.sigma, calib.theta_star, j, calib.tau, calib.w, calib.A, calib.f, calib.gamma_prime_target)
    eq = MelitzEquilibrium(calib.E, ones(Float64, D), calib.q, calib.X)
    cf = MelitzCounterfactual(j, calib.w_prime, calib.w_prime * calib.L[j], 1.0, calib.w_prime * calib.L[j])

    moment_layout = MelitzMomentLayout(D)
    c_full, A_pivot = build_gravity_pivots(p.tau, j)
    outer_layout = melitz_outer_layout(D, j)

    if moment_backend == :sorted_tail_serial
        z_draws === nothing && throw(ArgumentError(
            "melitz_calibration_outer_ctx: moment_backend=:sorted_tail_serial requires " *
            "the SAME z_draws the bundle will use, passed via the z_draws kwarg"))
        sorted_tail_ctx = build_melitz_sorted_tail_context(z_draws, p.sigma; theta_star=p.theta_star)
    else
        sorted_tail_ctx = nothing
    end

    ctx = (D=D, sigma=p.sigma, theta_star=p.theta_star, target_country=j, tau=p.tau, w=p.w,
           w_prime=cf.w_prime, L=calib.L, expenditure=eq.expenditure, benchmark_cutoff=eq.cutoff,
           moment_layout=moment_layout, X_data=eq.trade_flow, c_full=c_full, A_pivot=A_pivot,
           jj_lin=outer_layout.jj_lin, f_free_lin=outer_layout.f_free_lin,
           outer_parameterization=outer_parameterization, inner_loop_opt=inner_loop_opt, outer_loop_opt=outer_loop_opt,
           moment_backend=moment_backend, sorted_tail_ctx=sorted_tail_ctx)
    return p, eq, cf, ctx
end

"""
    MelitzRoundtripCheck

Phase 1.4 of the 2026-07-24 continuation session: the calibrated `(A, f, gamma_prime_target,
f_jj)` reduced through the ACTIVE outer coordinate system's gravity pivots
(`melitz_reduce_theta`) and expanded back (`melitz_expand_theta`), compared to the
calibration's own values. Both gravity restrictions are imposed EXACTLY (machine precision)
by the pivot construction at BOTH ends of this roundtrip, so a calibration whose OWN gravity
residuals are not already at machine precision (Phase 1.3) would show up here as a
DISAGREEING roundtrip -- the outer parameterization silently re-derives the pivot cells from
the imposed restriction, which can differ from whatever the calibration itself computed at
those same cells if the calibration's restriction only held approximately.
"""
struct MelitzRoundtripCheck
    theta_free::Vector{Float64}
    max_abs_A_diff::Float64
    max_rel_A_diff::Float64
    max_abs_f_diff::Float64
    max_rel_f_diff::Float64
    gamma_prime_diff::Float64
    f_jj_diff::Float64
    max_abs_cutoff_diff::Float64
    max_abs_share_diff::Float64
    focal_link_residual_reexpanded::Float64
end

"""
    melitz_calibration_roundtrip_check(calib::MelitzParetoCalibration; outer_parameterization=:logf)
        -> MelitzRoundtripCheck

Main prompt Phase 1.4: builds `MelitzPrimitives` from the calibrated `(A,f)`, reduces to the
active outer free coordinates (`melitz_reduce_theta`), expands back through the gravity
pivots (`melitz_expand_theta`), and compares reconstructed `A`, `f`, cutoffs, shares, and
the focal free-entry link against the original calibration -- essential because the outer
parameterization imposes BOTH gravity restrictions EXACTLY, so a calibration with nonzero
gravity residuals could otherwise be silently changed at its pivot cells when reduced and
re-expanded (documented risk this function makes concrete rather than assuming away).
"""
function melitz_calibration_roundtrip_check(calib::MelitzParetoCalibration; outer_parameterization::Symbol=:logf)
    p, eq, cf, ctx = melitz_calibration_outer_ctx(calib; outer_parameterization=outer_parameterization)
    theta_free = melitz_reduce_theta(p, ctx)
    A2, f2, gamma_prime_j2, f_jj2 = melitz_expand_theta(theta_free, ctx)

    cutoff2 = melitz_baseline_cutoff(A2, f2, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma)
    X2, q2, _ = population_X(calib.w, calib.L, calib.tau, A2, f2, calib.sigma, calib.theta_star)
    share2 = X2 ./ sum(X2, dims=1)
    share1 = calib.X ./ sum(calib.X, dims=1)

    p2 = MelitzPrimitives(calib.D, calib.sigma, calib.theta_star, calib.target_country, calib.tau, calib.w, A2, f2, gamma_prime_j2)
    link_resid2 = population_focal_link_residual(p2, X2, q2, f_jj2, cf)

    return MelitzRoundtripCheck(theta_free,
        maximum(abs.(A2 .- calib.A)), maximum(abs.(A2 .- calib.A) ./ max.(abs.(calib.A), 1e-300)),
        maximum(abs.(f2 .- calib.f)), maximum(abs.(f2 .- calib.f) ./ max.(abs.(calib.f), 1e-300)),
        abs(gamma_prime_j2 - calib.gamma_prime_target), abs(f_jj2 - calib.f_jj),
        maximum(abs.(cutoff2 .- calib.q)), maximum(abs.(share2 .- share1)), link_resid2)
end

"""
    build_melitz_psi_bundle_from_calibration(calib::MelitzParetoCalibration; W=20_000,
        seed=calib.seed, draw_mode=:halton, inner_loop_opt=..., outer_loop_opt=...,
        needs_outer_moment_jacobian=false) -> (obj, theta_free)

Main prompt Section 4/14: the observable-only analogue of `build_melitz_psi_bundle`. Builds
a `PsiObjectiveBundleDelta` wired to the calibrated `(A, f, w, gamma_prime_target)` -- NEVER
reads a `MelitzSyntheticTruth`. Reference draws are generated fresh (Pareto, `theta_star`)
since `MelitzParetoCalibration` does not itself carry a Monte-Carlo sample.
"""
function build_melitz_psi_bundle_from_calibration(calib::MelitzParetoCalibration;
        W::Int=20_000, seed::Int=calib.seed, draw_mode::Symbol=:halton,
        inner_loop_opt::String=joinpath(dirname(dirname(@__DIR__)), "ek_inner_loop_options.opt"),
        outer_loop_opt::String=joinpath(dirname(dirname(@__DIR__)), "ek_outer_loop_options.opt"),
        needs_outer_moment_jacobian::Bool=false,
        inner_solve_config::Union{Nothing,MelitzInnerSolveConfig}=nothing,
        moment_backend::Symbol=:dense_reference)
    D = calib.D
    j = calib.target_country
    z_draws = pareto_draws(W, D, calib.theta_star; seed=seed, mode=draw_mode)

    p, eq, cf, ctx = melitz_calibration_outer_ctx(calib; outer_parameterization=:logf,
        inner_loop_opt=inner_loop_opt, outer_loop_opt=outer_loop_opt,
        moment_backend=moment_backend, z_draws=z_draws)

    theta_free = melitz_reduce_theta(p, ctx)

    obj = PsiObjectiveBundleDelta(
        γ=ctx, (moments!)=melitz_moments_adapter!, d=ctx.moment_layout.num_moments,
        l=length(theta_free), inequality_index=Int64[], U=z_draws,
        inner_loop_opt=inner_loop_opt, outer_loop_opt=outer_loop_opt,
        needs_outer_moment_jacobian=needs_outer_moment_jacobian,
        # See build_melitz_psi_bundle's identical kwarg (delta_star.jl) for why this is
        # optional here (preserves every existing call site's exact uncapped behavior) and
        # where the cap is made MANDATORY instead (solve_melitz_nuisance_min_delta).
        lower_limit=(inner_solve_config === nothing ? -KNITRO.KN_INFINITY : inner_solve_config.lower_limit))

    return obj, theta_free
end
