# Phase 2 (D4 leg): reduced/full equivalence check on the pre-existing, UNMODIFIED synthetic
# D4 fixture (fake_data.jl/build_melitz_psi_bundle, mask=nothing throughout -- never touched by
# this task), adapted here (diagnostically, NOT by editing fake_data.jl) to contain a
# profiled-cell analog: focal=country 1, treat destination 4 as "ROW", diagonal cells (2,2) and
# (3,3), and bilateral outlier (2,3) -- 6 profiled cells out of 16, the same three-category
# structure as the real-D20 38-cell set, at a scale where a real KNITRO solve is fast.
REPO2 = "/bbkinghome/edav/gravity_robustness/worktrees/melitz-profile-38-nongravity-nonfocal-2026-08-01"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, LinearAlgebra

function melitz_profile38_oracle(profiled_od::Vector{Tuple{Int,Int}},
                                  w::Vector{Float64}, tau::Matrix{Float64},
                                  expenditure::Vector{Float64}, lambda::Matrix{Float64},
                                  sigma::Float64, z::Matrix{Float64}, p::Vector{Float64};
                                  N::Vector{Float64}=ones(length(w)))
    Y = z .^ (sigma - 1)
    T_o = vec(sum(p .* Y, dims=1))
    markup = melitz_markup(sigma)
    A_completed = Dict{Tuple{Int,Int},Float64}()
    f_completed = Dict{Tuple{Int,Int},Float64}()
    for (o, d) in profiled_od
        B_od = N[o] * T_o[o] / lambda[o, d]
        A_completed[(o, d)] = markup * w[o] * tau[o, d] / B_od^(1 / (sigma - 1))
        f_completed[(o, d)] = lambda[o, d] * expenditure[d] / (sigma * w[o] * N[o] * T_o[o])
    end
    return A_completed, f_completed, T_o
end

data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj, theta0 = build_melitz_psi_bundle(data; outer_parameterization=:logcutoff,
    policy=CappedEvaluation(10.0), forbid_dense_fallback=true)
ctx = obj.γ
D = ctx.D
j = ctx.target_country
@assert j == 1

focal = 1
profiled_od = [(2, 2), (3, 3), (2, 4), (3, 4), (4, 4), (2, 3)]
pivot_cells = [ctx.A_pivot.pivot]  # linear index of the A-pivot cell
for lin in pivot_cells
    o, d = lin2od(lin, D)
    @assert (o, d) ∉ profiled_od "A-pivot cell ($o,$d) collides with the chosen D4 profiled set -- pick a different diagnostic mask"
end

obj.use_cached_x = false; obj.x .= NaN
lfd0 = melitz_recover_lfd(obj, theta0)
@printf("[FULL, D4]  Delta0=%.8e  nStatus=%d  lfd_ok=%s  max|moment_resid|=%.3e\n",
    lfd0.Delta, lfd0.nStatus, lfd0.lfd_ok, lfd0.maximum_weighted_moment_residual)
@assert lfd0.lfd_ok "D4 base point failed to verify"

A0, f0, gpj0, fjj0, q0 = expand_free_theta_logcutoff(theta0, ctx)
z = obj.U
lambda = ctx.X_data ./ ctx.expenditure'
A_c, f_c, T_o = melitz_profile38_oracle(profiled_od, ctx.w, ctx.tau, ctx.expenditure, lambda,
    ctx.sigma, z, lfd0.weights)

A1 = copy(A0); f1 = copy(f0)
for (o, d) in profiled_od
    A1[o, d] = A_c[(o, d)]
    f1[o, d] = f_c[(o, d)]
end
theta1 = reduce_to_free_theta_logcutoff(A1, f1, gpj0, ctx)
_, _, _, _, q1_check = expand_free_theta_logcutoff(theta1, ctx)
max_q_dev = maximum(abs(q1_check[o, d]) for (o, d) in profiled_od)
@printf("max|q_od| over D4 profiled cells after embedding theta1 (should be ~0) = %.3e\n", max_q_dev)

obj.x = copy(lfd0.dual_x); obj.use_cached_x = true
lfd1 = melitz_recover_lfd(obj, theta1)
@printf("[COMPLETED, D4]  Delta1=%.8e  nStatus=%d  lfd_ok=%s  max|moment_resid|=%.3e\n",
    lfd1.Delta, lfd1.nStatus, lfd1.lfd_ok, lfd1.maximum_weighted_moment_residual)
@assert lfd1.lfd_ok "D4 completed-state solve failed to verify"

delta_gap = abs(lfd1.Delta - lfd0.Delta)
lfd_gap = maximum(abs.(lfd1.weights .- lfd0.weights))
@printf("\n|Delta1 - Delta0| = %.3e\n", delta_gap)
@printf("max|weights1 - weights0| = %.3e\n", lfd_gap)

function moment_resid_by_group(lfd, D, profiled_od)
    mr = lfd.moment_residuals
    active_max = 0.0; profiled_max = 0.0
    for d in 1:D, o in 1:D
        lin = o + (d - 1) * D
        r = abs(mr[lin])
        if (o, d) in profiled_od
            profiled_max = max(profiled_max, r)
        else
            active_max = max(active_max, r)
        end
    end
    return active_max, profiled_max
end
act0, prof0 = moment_resid_by_group(lfd0, D, profiled_od)
act1, prof1 = moment_resid_by_group(lfd1, D, profiled_od)
@printf("[FULL, D4]      max active resid=%.3e  max PROFILED resid=%.3e\n", act0, prof0)
@printf("[COMPLETED, D4] max active resid=%.3e  max PROFILED resid=%.3e\n", act1, prof1)

pass = lfd1.lfd_ok && delta_gap < 1e-4 && lfd_gap < 1e-3 && prof1 < 1e-3 && act1 < 1e-3
println("\nPHASE 2 (D4 leg) REDUCED/FULL EQUIVALENCE: ", pass ? "PASS" : "FAIL")
