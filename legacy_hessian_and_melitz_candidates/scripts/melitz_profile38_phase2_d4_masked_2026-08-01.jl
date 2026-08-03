# Phase 2 (D4 leg, take 2): a GENUINELY masked D4 fixture, built via the same production
# calibration pipeline real-D20 uses (calibrate_melitz_pareto + an explicit MelitzGravitySample)
# rather than the unmasked synthetic-fixture generator (fake_data.jl). This guarantees the
# hand-picked "profiled" cells actually have zero A/f-gravity coefficient (the premise the
# first D4 attempt, scripts/melitz_profile38_phase2_d4_2026-08-01.jl, was found to violate).
#
# Countries: foc(=focal, index1), nf1(2), row(3, ROW-analog destination), nf2(4).
# Gravity sample: destination==row excluded; diagonal (o==d, o!=row) excluded; outlier (nf1,nf2)
# excluded. Profiled (nonfocal & excluded): (nf1,row)=(2,3), (row,row)=(3,3), (nf2,row)=(4,3),
# (nf1,nf1)=(2,2), (nf2,nf2)=(4,4), (nf1,nf2)=(2,4) -- 6 cells. Active-but-excluded (focal
# add-backs): (foc,foc)=(1,1), (foc,row)=(1,3).
REPO2 = "/bbkinghome/edav/gravity_robustness/worktrees/melitz-profile-38-nongravity-nonfocal-2026-08-01"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, LinearAlgebra, Random

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

# Reuse the pre-existing D4 synthetic generator ONLY as a source of a plausible, positive,
# adding-up-to-1 trade-share matrix and moderate iceberg costs -- not its (unmasked) gravity
# construction, which is discarded entirely.
seed_data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
tau_fake = seed_data.primitives.tau
L_fake = seed_data.L
X_fake = seed_data.equilibrium.trade_flow
E_fake = seed_data.equilibrium.expenditure
lambda_fake = X_fake ./ E_fake'
@assert all(>(0), lambda_fake)
@assert isapprox(vec(sum(lambda_fake, dims=1)), ones(4); atol=1e-8)

countries4 = ["foc", "nf1", "row", "nf2"]
observed = MelitzObservedData(; lambda=lambda_fake, L=L_fake, tau=tau_fake, countries=countries4, atol=2e-3)
gs4 = melitz_gravity_sample(countries4; row_label="row", outlier_pairs=[("nf1", "nf2")])
@printf("D4 gravity sample: included=%d excluded_row=%d excluded_diag=%d excluded_outlier=%d\n",
    length(gs4.included_lin), length(gs4.excluded_row_lin), length(gs4.excluded_diag_lin), length(gs4.excluded_outlier_lin))

calib4 = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=6.8, focal_country=1,
    p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6, gravity_sample=gs4)
@printf("D4 masked calibration: theta_star=%.6f  composite_gravity_residual=%.3e\n",
    calib4.theta_star, calib4.gravity.composite_gravity_residual)

focal = 1
profiled_od = Tuple{Int,Int}[]
for d in 1:4, o in 1:4
    if !gs4.mask[o, d] && o != focal
        push!(profiled_od, (o, d))
    end
end
println("profiled_od = ", profiled_od)
@assert length(profiled_od) == 6

obj, theta0 = build_melitz_psi_bundle_from_calibration(calib4; W=20_000, seed=1,
    outer_parameterization=:logcutoff,
    inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=CappedEvaluation(10.0))
ctx = obj.γ
D = ctx.D
@assert D == 4

# gravity-zero-coefficient check for the chosen mask (Phase 0 premise, D4 instance)
c4 = gravity_coefficient_vector(D, ctx.tau; mask=gs4.mask)
max_c_excluded = maximum(abs(c4[o+(d-1)*D]) for (o, d) in profiled_od)
@printf("max|A/f-gravity coefficient| at the 6 D4 profiled cells = %.3e (should be ~0)\n", max_c_excluded)
@assert max_c_excluded < 1e-10

obj.use_cached_x = false; obj.x .= NaN
lfd0 = melitz_recover_lfd(obj, theta0)
@printf("[FULL, D4-masked]  Delta0=%.8e  nStatus=%d  lfd_ok=%s  max|moment_resid|=%.3e\n",
    lfd0.Delta, lfd0.nStatus, lfd0.lfd_ok, lfd0.maximum_weighted_moment_residual)
@assert lfd0.lfd_ok "D4-masked base point failed to verify"

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
@printf("max|q_od| over D4-masked profiled cells after embedding theta1 (should be ~0) = %.3e\n", max_q_dev)

# with the mask premise now genuinely satisfied, the A-gravity restriction is UNCHANGED by
# swapping only zero-coefficient cells -- so the pivot cell should be UNCHANGED too (unlike the
# first D4 attempt). Verify directly before re-solving.
pivot = ctx.A_pivot.pivot
po, pd = lin2od(pivot, D)
@printf("A-pivot cell = (%s,%s): A0=%.6f A1=%.6f (should match)\n", countries4[po], countries4[pd], A0[po,pd], A1[po,pd])

obj.x = copy(lfd0.dual_x); obj.use_cached_x = true
lfd1 = melitz_recover_lfd(obj, theta1)
@printf("[COMPLETED, D4-masked]  Delta1=%.8e  nStatus=%d  lfd_ok=%s  max|moment_resid|=%.3e\n",
    lfd1.Delta, lfd1.nStatus, lfd1.lfd_ok, lfd1.maximum_weighted_moment_residual)

if lfd1.lfd_ok
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
    @printf("[FULL, D4-masked]      max active resid=%.3e  max PROFILED resid=%.3e\n", act0, prof0)
    @printf("[COMPLETED, D4-masked] max active resid=%.3e  max PROFILED resid=%.3e\n", act1, prof1)

    pass = delta_gap < 1e-4 && lfd_gap < 1e-3 && prof1 < 1e-3 && act1 < 1e-3
    println("\nPHASE 2 (D4-masked leg) REDUCED/FULL EQUIVALENCE: ", pass ? "PASS" : "FAIL")
else
    println("\nPHASE 2 (D4-masked leg): state COMPLETED did not verify -- see nStatus above.")
end
