using LinearAlgebra: diag

# Structured types for the full-D Melitz Christensen-Connault benchmark.
# See docs/melitz_delta_star.md for the full derivation and notation map.
#
# Convention (matches production/fullA-exact's own A-productivity convention, no adapter
# needed -- docs/melitz_delta_star.md Section 2):
#   marginal_cost_od(z) = w_o * tau_od / (A_od * z)
# i.e. higher A_od => lower marginal cost.

"""
Fixed structural primitives of the D-country Melitz economy: everything that does not
change between the baseline and the autarky counterfactual (A, f, tau, f_entry are
primitives of the model; w is the baseline wage vector, w'[target] is a separate
counterfactual numeraire handled in `MelitzCounterfactual`).
"""
struct MelitzPrimitives{T<:Real}
    D::Int
    sigma::Float64                  # CES elasticity, fixed/calibrated, sigma > 1
    theta_star::Float64             # Pareto shape, fixed/calibrated, theta_star > sigma-1
    target_country::Int             # focal country index for the autarky counterfactual
    tau::Matrix{Float64}            # D x D, tau[o,d] = 1+t_od, diag == 1
    w::Vector{T}                    # D, baseline wages, w[target_country] == 1 (numeraire)
    A::Matrix{T}                    # D x D, efficiency shifter (higher => lower cost)
    f::Matrix{T}                    # D x D, bilateral fixed market-access cost
    f_entry::Vector{T}               # D, entry cost (per potential entrant)

    function MelitzPrimitives(D::Int, sigma::Float64, theta_star::Float64,
                               target_country::Int, tau::Matrix{Float64},
                               w::Vector{T}, A::Matrix{T}, f::Matrix{T},
                               f_entry::Vector{T}) where {T<:Real}
        sigma > 1 || throw(ArgumentError("sigma must be > 1, got $sigma"))
        theta_star > sigma - 1 || throw(ArgumentError(
            "theta_star must be > sigma-1 for finite Pareto moments, got theta_star=$theta_star, sigma=$sigma"))
        size(tau) == (D, D) || throw(ArgumentError("tau must be D x D"))
        size(A) == (D, D) || throw(ArgumentError("A must be D x D"))
        size(f) == (D, D) || throw(ArgumentError("f must be D x D"))
        length(w) == D || throw(ArgumentError("w must have length D"))
        length(f_entry) == D || throw(ArgumentError("f_entry must have length D"))
        all(==(1.0), diag(tau)) || throw(ArgumentError("tau diagonal must be 1"))
        1 <= target_country <= D || throw(ArgumentError("target_country out of range"))
        new{T}(D, sigma, theta_star, target_country, tau, w, A, f, f_entry)
    end
end

"""
Baseline equilibrium objects, derived (not primitive): entrant mass N_o (closed form,
Section 1.4), destination expenditure (= sum_o X_od by construction, Section 4),
price_power_d (~= 1 by construction, Section 1.2/4), the D x D cutoff matrix, and the
resulting D x D trade-flow matrix.
"""
struct MelitzEquilibrium{T<:Real}
    entrant_mass::Vector{T}         # D, N_o -- economic mass of potential entrants
    expenditure::Vector{T}          # D, expenditure_d = sum_o X_od
    price_power::Vector{T}          # D, should be ~1 for every d by construction
    cutoff::Matrix{T}               # D x D, zhat_od >= 1, zhat_od >= zhat_oo for d != o
    trade_flow::Matrix{T}           # D x D, X_od implied by the model at these primitives
end

"""
Autarky counterfactual objects for the single focal `target_country`. Only that country's
objects are populated; `entrant_mass` is NOT independently searched -- it is the *same*
vector as the baseline (N'_o = N_o), passed through unchanged (see docs Section 9).
"""
struct MelitzCounterfactual{T<:Real}
    target_country::Int
    w_prime::T                      # counterfactual wage of target_country, normalized to 1
    expenditure_prime::T            # autarky expenditure of target_country
    price_power_prime::T            # autarky price-power aggregate (endogenous, NOT normalized)
    cutoff_prime::T                 # zhat'[target,target], normalized to 1
    trade_flow_prime::T             # X'[target,target] (= expenditure_prime in autarky)
end

"""
Bundle of primitives + baseline equilibrium + autarky counterfactual + reference draws,
used as the ground-truth synthetic fixture (Section "Synthetic D=4 economy").
"""
struct MelitzSyntheticData{T<:Real}
    primitives::MelitzPrimitives{T}
    equilibrium::MelitzEquilibrium{T}
    counterfactual::MelitzCounterfactual{T}
    z_draws::Matrix{Float64}        # W x D reference Pareto(1, theta_star) draws, seeded/fixed
    seed::Int
end

"""
Named column layout of the moment matrix G. Trade-flow moments occupy columns
`trade_index[o,d]` (all D^2 cells, each appearing exactly once); free-entry moments
occupy `entry_index[o]`. The two gravity restrictions are NOT columns of G -- they are
F-independent outer-loop equality constraints (Section 4C), evaluated directly from
(A, f, tau), never duplicated as G columns.
"""
struct MelitzMomentLayout
    D::Int
    trade_index::Matrix{Int}        # D x D, column index of g_trade[o,d] in G
    entry_index::Vector{Int}        # D, column index of g_entry[o] in G
    num_moments::Int                # total inner-loop (G) columns = D^2 + D
    names::Vector{String}           # length num_moments, e.g. "trade[o=1,d=2]", "entry[o=1]"
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
    entry_index = zeros(Int, D)
    for o in 1:D
        col += 1
        entry_index[o] = col
        push!(names, "entry[o=$o]")
    end
    MelitzMomentLayout(D, trade_index, entry_index, col, names)
end

"""
Result of the direct Pareto F* benchmark solver: solved parameters, equilibrium objects,
cutoffs, residuals (economic units), and solver status. Deliberately does not claim
elementwise recovery of A/f (see docs Section 1.6, 9) -- residuals/feasibility/composite
gravity match are the acceptance criteria, not distance to any single "true" A/f.
"""
struct MelitzFStarResult{T<:Real}
    primitives::MelitzPrimitives{T}
    equilibrium::MelitzEquilibrium{T}
    counterfactual::MelitzCounterfactual{T}
    max_trade_residual::Float64      # max |X_model - X_data| across all D^2 cells
    max_entry_residual::Float64      # max |free-entry residual| across D origins
    gravity_residual_A::Float64      # <doubleDiff(tau), doubleDiff(A)>
    gravity_residual_f::Float64      # <doubleDiff(tau), doubleDiff(f)>
    converged::Bool
    solver_status::Symbol
end
