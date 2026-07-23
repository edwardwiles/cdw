using LinearAlgebra: diag

# Structured types for the full-D Melitz Christensen-Connault benchmark.
# See docs/melitz_delta_star.md for the full derivation and notation map.
#
# Convention (matches production/fullA-exact's own A-productivity convention, no adapter
# needed -- docs/melitz_delta_star.md Section 2):
#   marginal_cost_od(z) = w_o * tau_od / (A_od * z)
# i.e. higher A_od => lower marginal cost.
#
# REVISED (minimal-moment / LFD-recovery formulation, superseding the original
# f_entry-as-primitive / N-derived-from-f_entry / N'=N closure -- see docs Section 16
# "Superseded statements"):
#   - baseline entrant mass N_o is normalized to 1 for every origin (a scale
#     normalization, not searched, not stored -- see `normalize_baseline_entrant_mass`
#     in equilibrium.jl);
#   - gamma_prime[target] is now a SEARCHED outer parameter, stored on
#     `MelitzPrimitives`, not precomputed;
#   - f[target,target] is DERIVED from the autarky cutoff-at-one normalization
#     (`derive_fjj_from_autarky_cutoff`), not packed independently;
#   - f_entry is no longer a primitive -- it is recovered from the LFD (see
#     `recover_entry_costs_from_lfd`) and is NOT a field of `MelitzPrimitives`.

"""
Fixed structural primitives of the D-country Melitz economy under the active
(minimal-moment) closure. `A`, `f`, `tau` are primitives; `w` is the baseline wage vector
(solved from data, target_country's baseline wage need not be 1 -- only the AUTARKY
numeraire w_prime[target]=1 is normalized, handled separately in `MelitzCounterfactual`);
`gamma_prime_target` is the searched autarky price-power outer parameter for
`target_country`. Baseline `gamma[d]=1` and `N[o]=1` for every d/o are UNIVERSAL
normalizations, not stored fields (docs Section 16).
"""
struct MelitzPrimitives{T<:Real,W<:Real}
    D::Int
    sigma::Float64                  # CES elasticity, fixed/calibrated, sigma > 1
    theta_star::Float64             # Pareto shape, fixed/calibrated (benchmark-fixture only)
    target_country::Int             # focal country index for the autarky counterfactual
    tau::Matrix{Float64}            # D x D, tau[o,d] = 1+t_od, diag == 1
    w::Vector{W}                    # D, baseline wages (solved from data -- NOT part of theta,
                                     # so kept independently typed from A/f/gamma_prime_target:
                                     # a ForwardDiff-differentiated outer search over (A,f,gamma')
                                     # must not force plain Float64 `w` into a Dual-typed struct)
    A::Matrix{T}                    # D x D, efficiency shifter (higher => lower cost)
    f::Matrix{T}                    # D x D, bilateral fixed market-access cost (f[j,j] derived)
    gamma_prime_target::T           # searched outer parameter: autarky price-power, target country

    function MelitzPrimitives(D::Int, sigma::Float64, theta_star::Float64,
                               target_country::Int, tau::Matrix{Float64},
                               w::Vector{W}, A::Matrix{T}, f::Matrix{T},
                               gamma_prime_target::T) where {T<:Real,W<:Real}
        sigma > 1 || throw(ArgumentError("sigma must be > 1, got $sigma"))
        theta_star > sigma - 1 || throw(ArgumentError(
            "theta_star must be > sigma-1 for finite Pareto moments, got theta_star=$theta_star, sigma=$sigma"))
        size(tau) == (D, D) || throw(ArgumentError("tau must be D x D"))
        size(A) == (D, D) || throw(ArgumentError("A must be D x D"))
        size(f) == (D, D) || throw(ArgumentError("f must be D x D"))
        length(w) == D || throw(ArgumentError("w must have length D"))
        all(==(1.0), diag(tau)) || throw(ArgumentError("tau diagonal must be 1"))
        1 <= target_country <= D || throw(ArgumentError("target_country out of range"))
        gamma_prime_target > 0 || throw(ArgumentError("gamma_prime_target must be > 0"))
        new{T,W}(D, sigma, theta_star, target_country, tau, w, A, f, gamma_prime_target)
    end
end

"""
Baseline equilibrium objects, derived (not primitive): destination expenditure
(= sum_o X_od by construction), price_power_d (~= 1 by construction, since baseline
gamma[d]=1 and N[o]=1 are universal normalizations), the D x D cutoff matrix, and the
resulting D x D trade-flow matrix. `entrant_mass` is NOT a field -- it is identically 1
for every origin under the active normalization (docs Section 16), carrying no
information.
"""
struct MelitzEquilibrium{T<:Real}
    expenditure::Vector{T}          # D, expenditure_d = sum_o X_od
    price_power::Vector{T}          # D, should be ~1 for every d by construction
    cutoff::Matrix{T}               # D x D, zhat_od >= 1, zhat_od >= zhat_oo for d != o
    trade_flow::Matrix{T}           # D x D, X_od implied by the model at these primitives
end

"""
Autarky counterfactual bookkeeping for the single focal `target_country`.
`price_power_prime` is NOT a field here (it lives on `MelitzPrimitives` as the searched
`gamma_prime_target` -- storing it twice would let the two silently diverge).
`entrant_mass_prime` is `NaN` until populated by `recover_N_prime_market_clearing` /
`recover_N_prime_price_index` post-LFD (docs Section 7.2, addendum Section 5.3) -- it is
never imposed equal to the baseline entrant mass.
"""
mutable struct MelitzCounterfactual{T<:Real}
    target_country::Int
    w_prime::T                      # counterfactual wage of target_country, normalized to 1
    expenditure_prime::T            # autarky expenditure of target_country (= w_prime*L[target])
    cutoff_prime::T                 # zhat'[target,target], normalized to 1 by construction
    trade_flow_prime::T             # X'[target,target] (= expenditure_prime in autarky)
    entrant_mass_prime::T           # N'[target], recovered post-LFD (NaN until then)
end

MelitzCounterfactual(target_country::Int, w_prime::T, expenditure_prime::T,
                      cutoff_prime::T, trade_flow_prime::T) where {T<:Real} =
    MelitzCounterfactual(target_country, w_prime, expenditure_prime, cutoff_prime,
                          trade_flow_prime, T(NaN))

"""
Bundle of primitives + baseline equilibrium + autarky counterfactual + reference draws,
used as the ground-truth synthetic fixture (Section "Synthetic D=4 economy"). Also carries
`L` (baseline labor endowments, `= expenditure ./ w`) since the active closure needs
`L[target]` directly (autarky `expenditure_prime[target] = w_prime[target]*L[target]`) and
no longer has a free `entrant_mass` field to derive it from implicitly.
"""
struct MelitzSyntheticData{T<:Real}
    primitives::MelitzPrimitives     # untyped: MelitzPrimitives has independent A/f/gamma_prime_target
                                      # vs. w type params (see its docstring) -- not worth threading here
    equilibrium::MelitzEquilibrium{T}
    counterfactual::MelitzCounterfactual{T}
    L::Vector{T}                    # D, baseline labor endowments
    z_draws::Matrix{Float64}        # W x D reference Pareto(1, theta_star) draws, seeded/fixed
    seed::Int
end

"""
Named column layout of the moment matrix G. Trade-share moments occupy columns
`trade_index[o,d]` (all D^2 cells, each appearing exactly once); the single focal
baseline-vs-autarky free-entry LINK moment occupies `focal_link_index` (main prompt
Section 4.2 -- replaces the D per-origin free-entry moments of the superseded closure).
The two gravity restrictions are NOT columns of G -- they are F-independent outer-loop
equality constraints, enforced exactly via the gravity pivots in delta_star.jl, never
duplicated as G columns.
"""
struct MelitzMomentLayout
    D::Int
    trade_index::Matrix{Int}        # D x D, column index of g_trade[o,d] in G
    focal_link_index::Int           # column index of g_free_entry_link in G
    num_moments::Int                # total inner-loop (G) columns = D^2 + 1
    names::Vector{String}           # length num_moments
end

function MelitzMomentLayout(D::Int)
    trade_index = zeros(Int, D, D)
    names = String[]
    col = 0
    for o in 1:D, d in 1:D
        col += 1
        trade_index[o, d] = col
        push!(names, "trade[o=$o,d=$d]")
    end
    col += 1
    focal_link_index = col
    push!(names, "focal_free_entry_link")
    MelitzMomentLayout(D, trade_index, focal_link_index, col, names)
end

"""
Result of the direct Pareto F* benchmark solver (`fstar_solver.jl`): the outer point found
by the finite-sample correction solve, its residuals (economic units), gravity residuals,
and solver status. Deliberately does not claim elementwise recovery of a unique "true"
`A`/`f` (docs Section 1.6/9, addendum Section 1.3) -- residuals/feasibility/composite
gravity match are the acceptance criteria.
"""
struct MelitzFStarResult{T<:Real}
    primitives::MelitzPrimitives     # untyped, see MelitzSyntheticData's field comment
    equilibrium::MelitzEquilibrium{T}
    counterfactual::MelitzCounterfactual{T}
    max_trade_residual::Float64      # max |sample-mean share residual| across all D^2 cells
    max_link_residual::Float64       # |focal free-entry link residual| at equal weights
    gravity_residual_A::Float64      # Cov(withinTransform(log tau), withinTransform(log A))
    gravity_residual_f::Float64      # Cov(withinTransform(log tau), withinTransform(log f))
    converged::Bool
    solver_status::Symbol
end

"""
Result of a fixed-parameter CC inner minimum-divergence solve (main prompt Section 10),
including the recovered LFD -- NOT just the dual objective/status (the superseded
closure's `run_melitz_inner_delta` returned only `(val, x, nStatus)`).

`weights` are the normalized (sum to 1) LFD probabilities over the `W` reference draws.
`moment_residuals` is `E_LFD[G[:,k]]` for every one of the `num_moments` columns
(should be ~0 whenever `lfd_ok`). `normalization_residual = sum(raw LFD weights)/W - 1`
before normalization (numerical sanity check, not the same as `sum(weights)-1` which is
~0 by construction once normalized).
"""
struct MelitzLFDResult
    Delta::Float64                  # Delta(theta) -- the primal/dual minimum-divergence value
    dual_x::Vector{Float64}         # KNITRO-optimal dual vector (zeta, lambda...)
    nStatus::Int                    # raw KNITRO status code (0 = optimal)
    weights::Vector{Float64}        # W, normalized LFD probabilities (sum to 1)
    lfd_ok::Bool                    # false if the LFD reconstruction failed (non-finite/negative)
    moment_residuals::Vector{Float64}  # num_moments, E_LFD[G[:,k]]
    normalization_residual::Float64    # sum(raw weights)/W - 1, pre-normalization
    min_weight::Float64
    max_weight::Float64
    max_density_deviation::Float64  # max |W*weight - 1|
    # Gate A4 (docs/melitz_delta_star.md): primal-dual diagnostics stored explicitly, not
    # merely used internally by `lfd_ok`'s pass/fail gate.
    primal_divergence::Float64             # melitz_primal_divergence(weights, W) -- the PRIMAL
                                            # phi-divergence at the recovered LFD
    dual_divergence::Float64               # == Delta -- the DUAL objective KNITRO returned
                                            # (kept as its own field for symmetry with primal_divergence)
    primal_dual_gap::Float64                # abs(primal_divergence - dual_divergence)
    maximum_weighted_moment_residual::Float64  # maximum(abs.(moment_residuals))
    probability_normalization_residual::Float64  # == normalization_residual (explicit alias
                                                  # per Gate A4's exact naming)
    kkt_opt_error::Float64                  # KNITRO KN_get_abs_opt_error at the inner solution
                                             # (NaN if not captured, e.g. nStatus != 0)
    kkt_feas_error::Float64                 # KNITRO KN_get_abs_feas_error at the inner solution
end

"""
Every ex-post equilibrium-identity residual computed under a recovered LFD (main prompt
Section 9 / addendum Section 9), for the profiled/omitted equations that are NOT active
inner moments this milestone: baseline price-index identities, baseline/autarky free
entry, autarky market clearing, autarky price-index, autarky cutoff, cutoff inequalities,
gravity, and the two independent formulas for N'[target].
"""
struct MelitzEquilibriumCheck{T<:Real}
    residual_gamma_baseline::Vector{T}       # D, Section 9.1
    f_entry_recovered::Vector{T}             # D, Section 7.1 (baseline)
    residual_free_entry_baseline::Vector{T}  # D, Section 9.2
    f_entry_autarky_recovered::T             # Section 7.1 (autarky)
    residual_free_entry_autarky::T           # Section 9.3
    N_prime_market_clearing::T               # Section 7.2 / 9.4
    residual_market_clearing_autarky::T      # Section 9.4
    N_prime_from_gamma::T                    # Section 9.5
    N_prime_diff_abs::T                      # Section 9.5
    N_prime_diff_rel::T                      # Section 9.5
    residual_gamma_autarky::T                # Section 9.5 (key omitted moment)
    residual_autarky_cutoff::T               # Section 9.6
    min_baseline_cutoff::T                   # Section 9.9
    min_cutoff_minus_one::T                  # Section 9.9
    min_export_minus_domestic::T             # Section 9.9
    gravity_residual_A::T                    # Section 9.8 (Cov(withinTransform(log tau), withinTransform(log A)))
    gravity_residual_f::T                    # Section 9.8 (Cov(withinTransform(log tau), withinTransform(log f)))
end
