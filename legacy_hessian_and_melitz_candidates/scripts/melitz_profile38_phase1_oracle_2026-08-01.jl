# Phase 1: diagnostic completion oracle for the 38 nonfocal, non-gravity-included cells.
# DIAGNOSTIC ONLY -- never called from the inner/middle hot path.
#
# Given any LFD weight vector p (W-length, sums to 1) and the production z-draw matrix
# (W x D, one column per ORIGIN -- see sorted_tail.jl), completes each profiled cell (o,d) by:
#   1. fixing k_od = 1 (cutoff level 1, i.e. q_od = log(k_od) = 0 in :logcutoff units --
#      the WLOG "no selection beyond the Pareto floor" convention), so 1{z_os >= k_od} == 1
#      identically (z_os >= 1 always, firm_quantities.jl/pareto.jl support), hence
#      T_o(1) = sum_s p_s * Y_os, a pure per-origin expectation (no destination index).
#   2. inverting the SAME bilateral-trade-flow/zero-profit pair every production cell
#      satisfies, using production's own melitz_C/melitz_K1/melitz_cutoff conventions,
#      adapted from the closed-form `cell_from_cutoff` (equilibrium.jl) to a FINITE-SAMPLE
#      tail expectation under the given p (cell_from_cutoff itself is population-closed-form
#      only, T_o(k) there is the Pareto-population tail integral `pareto_tail_power_mean`,
#      never called from the active moment/delta_star path per its own docstring).
REPO2 = "/bbkinghome/edav/gravity_robustness/worktrees/melitz-profile-38-nongravity-nonfocal-2026-08-01"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using DelimitedFiles, Printf, Statistics

"""
    melitz_profile38_oracle(profiled_od, w, tau, expenditure, lambda, sigma, z, p; N=ones(D))
        -> (A_completed, f_completed, T_o, diag)

`profiled_od :: Vector{Tuple{Int,Int}}` the (o,d) cells to complete. `z :: W x D` matrix
(one column per origin, production convention). `p :: W`-vector, sums to 1. `lambda :: D x D`
empirical trade-SHARE matrix (columns sum to 1). `expenditure :: D`-vector (`E_d = w_d*L_d`).
`N :: D`-vector entrant mass (factual N_o == 1 in the current maintained closure).
Returns `A_completed`, `f_completed` as `Dict{Tuple{Int,Int},Float64}` (profiled cells only)
and `T_o :: D`-vector (`T_o[o] = sum_s p[s]*z[s,o]^(sigma-1)`, well-defined for EVERY origin --
independent of destination, per z_os's own support: z_os >= 1 identically).
"""
function melitz_profile38_oracle(profiled_od::Vector{Tuple{Int,Int}},
                                  w::Vector{Float64}, tau::Matrix{Float64},
                                  expenditure::Vector{Float64}, lambda::Matrix{Float64},
                                  sigma::Float64, z::Matrix{Float64}, p::Vector{Float64};
                                  N::Vector{Float64}=ones(length(w)))
    W, D = size(z)
    length(p) == W || throw(ArgumentError("melitz_profile38_oracle: length(p)=$(length(p)) != W=$W"))
    isapprox(sum(p), 1.0; atol=1e-8) || throw(ArgumentError("melitz_profile38_oracle: p must sum to 1, got $(sum(p))"))
    all(>=(1.0 - 1e-12), z) || throw(ArgumentError("melitz_profile38_oracle: found a draw with z < 1 -- k_od=1 completion invalid"))

    markup = melitz_markup(sigma)
    Y = z .^ (sigma - 1)                    # W x D, Y_os
    T_o = vec(sum(p .* Y, dims=1))          # D-vector, T_o[o] = sum_s p_s*Y_os (indicator==1 identically)

    A_completed = Dict{Tuple{Int,Int},Float64}()
    f_completed = Dict{Tuple{Int,Int},Float64}()
    for (o, d) in profiled_od
        lambda[o, d] > 0 || throw(ArgumentError("melitz_profile38_oracle: lambda[$o,$d] <= 0, cellwise completion undefined"))
        T_o[o] > 0 || throw(ArgumentError("melitz_profile38_oracle: T_o[$o] <= 0, cellwise completion undefined"))
        B_od = N[o] * T_o[o] / lambda[o, d]
        A_od = markup * w[o] * tau[o, d] / B_od^(1 / (sigma - 1))
        f_od = 1.0 * lambda[o, d] * expenditure[d] / (sigma * w[o] * N[o] * T_o[o])
        A_completed[(o, d)] = A_od
        f_completed[(o, d)] = f_od
    end
    return A_completed, f_completed, T_o
end

"""
    melitz_profile38_verify(profiled_od, A_completed, f_completed, w, tau, expenditure,
                             lambda, sigma, z, p) -> NamedTuple of max residuals

Independent re-derivation check (NOT reusing internal state from the oracle call): for each
completed cell, recomputes the finite-sample predicted revenue share under (A_od,f_od,p) via
the SAME melitz_C/melitz_cutoff production primitives moments.jl/moment_operator.jl use, and
compares against the empirical share; also re-derives the zero-profit cutoff and checks it
equals 1 (k_od=1) to machine precision.
"""
function melitz_profile38_verify(profiled_od::Vector{Tuple{Int,Int}},
                                  A_completed::Dict{Tuple{Int,Int},Float64},
                                  f_completed::Dict{Tuple{Int,Int},Float64},
                                  w::Vector{Float64}, tau::Matrix{Float64},
                                  expenditure::Vector{Float64}, lambda::Matrix{Float64},
                                  sigma::Float64, z::Matrix{Float64}, p::Vector{Float64})
    W, D = size(z)
    max_share_resid = 0.0
    max_cutoff_resid = 0.0
    worst_share_cell = (0, 0)
    for (o, d) in profiled_od
        A_od = A_completed[(o, d)]
        f_od = f_completed[(o, d)]
        C_od = melitz_C(w[o], tau[o, d], A_od, sigma, expenditure[d])
        zhat_od = melitz_cutoff(w[o], f_od, sigma, C_od)
        cutoff_resid = abs(zhat_od - 1.0)
        max_cutoff_resid = max(max_cutoff_resid, cutoff_resid)

        # predicted revenue share under p, active firms only (z >= zhat_od == 1, i.e. all draws)
        pred_rev = 0.0
        @inbounds for s in 1:W
            zs = z[s, o]
            if zs >= zhat_od
                price = melitz_markup(sigma) * w[o] * tau[o, d] / (A_od * zs)
                pred_rev += p[s] * expenditure[d] * price^(1 - sigma)
            end
        end
        pred_share = pred_rev / expenditure[d]
        share_resid = abs(pred_share - lambda[o, d])
        if share_resid > max_share_resid
            max_share_resid = share_resid
            worst_share_cell = (o, d)
        end
    end
    return (max_share_resid=max_share_resid, worst_share_cell=worst_share_cell,
            max_cutoff_resid=max_cutoff_resid)
end

function main()
    real_dir = joinpath(REPO2, "real_data", "noah_D20")
    lambda = readdlm(joinpath(real_dir, "pi.csv"), ',')
    L = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
    tau = readdlm(joinpath(real_dir, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
    D = length(countries)
    focal = resolve_country_index(countries, "fra")

    gs = melitz_gravity_sample(countries)
    profiled_od = Tuple{Int,Int}[]
    for d in 1:D, o in 1:D
        if !gs.mask[o, d] && o != focal
            push!(profiled_od, (o, d))
        end
    end
    @assert length(profiled_od) == 38

    observed = MelitzObservedData(; lambda=lambda, L=L, tau=tau, countries=countries, atol=2e-3)
    calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal,
        p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
    w = calib.w
    sigma = calib.sigma
    expenditure = calib.E
    println("theta_star = ", calib.theta_star, "  sigma = ", sigma, "  D = ", D)
    flush(stdout)

    # Finite-QMC draw pool + a first, fast sanity LFD: UNIFORM p (always valid; tests the
    # ORACLE FORMULA's own correctness -- self-consistency should hold for ANY valid p, not
    # only an optimized one -- per the cellwise-completion theorem argued in Phase 0/1).
    W = 200_000
    z = pareto_draws(W, D, calib.theta_star; seed=1, mode=:halton)
    p_uniform = fill(1.0 / W, W)

    A_completed, f_completed, T_o = melitz_profile38_oracle(
        profiled_od, w, tau, expenditure, lambda, sigma, z, p_uniform)

    res = melitz_profile38_verify(profiled_od, A_completed, f_completed, w, tau, expenditure,
                                   lambda, sigma, z, p_uniform)
    @printf("\n[uniform p, W=%d] max|predicted_share - data_share| over 38 profiled cells = %.3e (worst cell: %s -> %s)\n",
        W, res.max_share_resid, countries[res.worst_share_cell[1]], countries[res.worst_share_cell[2]])
    @printf("[uniform p, W=%d] max|zhat_od - 1| over 38 profiled cells (zero-profit/cutoff consistency) = %.3e\n",
        W, res.max_cutoff_resid)

    println("\nPer-cell completed (A_od, f_od) at uniform p:")
    for (o, d) in profiled_od
        @printf("  %s -> %s : A=%.6g  f=%.6g  T_o=%.6g\n", countries[o], countries[d],
            A_completed[(o, d)], f_completed[(o, d)], T_o[o])
    end

    allpos_A = all(v -> v > 0 && isfinite(v), values(A_completed))
    allpos_f = all(v -> v > 0 && isfinite(v), values(f_completed))
    println("\nALL completed A_od finite & positive: ", allpos_A)
    println("ALL completed f_od finite & positive: ", allpos_f)

    pass = res.max_share_resid < 1e-6 && res.max_cutoff_resid < 1e-8 && allpos_A && allpos_f
    println("\nPHASE 1 ORACLE (uniform-p sanity check): ", pass ? "PASS" : "FAIL")

    return calib, profiled_od, z, w, tau, expenditure, lambda, sigma
end

result = main()
