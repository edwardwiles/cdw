# Phase 2 (real-D20 leg): reduced/full equivalence check using a REAL KNITRO solver-derived
# LFD (not the Phase 1 uniform-p sanity check). Diagnostic only -- reuses the EXISTING,
# UNMODIFIED production inner solver (melitz_recover_lfd / inner_loop) at two embeddings of
# the SAME outer point theta0:
#
#   state FULL (baseline): theta0 exactly as calibrated -- all D^2 cells' (A,f) as produced
#     by calibrate_melitz_pareto's own baseline construction.
#   state COMPLETED: theta0 with the 38 profiled cells' (A,f) REPLACED by the Phase 1 oracle's
#     completion (evaluated at the FULL state's own recovered LFD weights p0 -- an
#     internally-consistent embedding), q_od forced to 0 there (cutoff level 1) via
#     reduce_to_free_theta_logcutoff's own re-derivation of q from (A,f).
#
# If the profiled cells truly impose no restriction on the LFD (Phase 0's cellwise-completion
# argument), the EXISTING full inner solver run fresh at state COMPLETED must reproduce
# state FULL's Delta/LFD/active-moment-residuals to numerical precision, with the 38 profiled
# cells' OWN moment residuals ALSO ~0.
REPO2 = "/bbkinghome/edav/gravity_robustness/worktrees/melitz-profile-38-nongravity-nonfocal-2026-08-01"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using DelimitedFiles, Printf, LinearAlgebra

melitz_thread_startup_report()
LinearAlgebra.BLAS.set_num_threads(1)

function melitz_profile38_oracle(profiled_od::Vector{Tuple{Int,Int}},
                                  w::Vector{Float64}, tau::Matrix{Float64},
                                  expenditure::Vector{Float64}, lambda::Matrix{Float64},
                                  sigma::Float64, z::Matrix{Float64}, p::Vector{Float64};
                                  N::Vector{Float64}=ones(length(w)))
    W, D = size(z)
    markup = melitz_markup(sigma)
    Y = z .^ (sigma - 1)
    T_o = vec(sum(p .* Y, dims=1))
    A_completed = Dict{Tuple{Int,Int},Float64}()
    f_completed = Dict{Tuple{Int,Int},Float64}()
    for (o, d) in profiled_od
        B_od = N[o] * T_o[o] / lambda[o, d]
        A_od = markup * w[o] * tau[o, d] / B_od^(1 / (sigma - 1))
        f_od = 1.0 * lambda[o, d] * expenditure[d] / (sigma * w[o] * N[o] * T_o[o])
        A_completed[(o, d)] = A_od
        f_completed[(o, d)] = f_od
    end
    return A_completed, f_completed, T_o
end

function main()
    real_dir = joinpath(REPO2, "real_data", "noah_D20")
    lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
    focal = resolve_country_index(countries, "fra")
    D = length(countries)

    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal,
        p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)

    gs = calib.gravity_sample
    profiled_od = Tuple{Int,Int}[]
    for d in 1:D, o in 1:D
        if !gs.mask[o, d] && o != focal
            push!(profiled_od, (o, d))
        end
    end
    @assert length(profiled_od) == 38

    obj, theta0 = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
        outer_parameterization=:logcutoff,
        inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
        forbid_dense_fallback=true, policy=CappedEvaluation(10.0))
    ctx = obj.γ
    println("D=$D  target_country=$(ctx.target_country)  W=$(size(obj.U,1))  n_theta=$(length(theta0))")
    flush(stdout)

    # ---- state FULL: real KNITRO solve at the calibration's own theta0 ----
    obj.use_cached_x = false; obj.x .= NaN
    t0 = time()
    lfd0 = melitz_recover_lfd(obj, theta0)
    @printf("[FULL]  solved in %.1fs  Delta0=%.8e  nStatus=%d  lfd_ok=%s  max|moment_resid|=%.3e\n",
        time() - t0, lfd0.Delta, lfd0.nStatus, lfd0.lfd_ok, lfd0.maximum_weighted_moment_residual)
    @assert lfd0.lfd_ok "real-D20 base point (state FULL) failed to verify"
    flush(stdout)

    A0, f0, gpj0, fjj0, q0 = expand_free_theta_logcutoff(theta0, ctx)
    z = obj.U
    A_c, f_c, T_o = melitz_profile38_oracle(profiled_od, ctx.w, ctx.tau, ctx.expenditure,
        lambdaData, ctx.sigma, z, lfd0.weights)

    A1 = copy(A0); f1 = copy(f0)
    for (o, d) in profiled_od
        A1[o, d] = A_c[(o, d)]
        f1[o, d] = f_c[(o, d)]
    end
    theta1 = reduce_to_free_theta_logcutoff(A1, f1, gpj0, ctx)

    # confirm the 38 cells really do land at q=0 (cutoff level 1) after round-tripping
    _, _, _, _, q1_check = expand_free_theta_logcutoff(theta1, ctx)
    max_q_dev = maximum(abs(q1_check[o, d]) for (o, d) in profiled_od)
    @printf("max|q_od| over profiled cells after embedding theta1 (should be ~0) = %.3e\n", max_q_dev)

    # ---- state COMPLETED: fresh real KNITRO solve at theta1 (existing, unmodified full solver) ----
    obj.use_cached_x = false; obj.x .= NaN
    t1 = time()
    lfd1 = melitz_recover_lfd(obj, theta1)
    @printf("[COMPLETED]  solved in %.1fs  Delta1=%.8e  nStatus=%d  lfd_ok=%s  max|moment_resid|=%.3e\n",
        time() - t1, lfd1.Delta, lfd1.nStatus, lfd1.lfd_ok, lfd1.maximum_weighted_moment_residual)
    @assert lfd1.lfd_ok "state COMPLETED failed to verify"
    flush(stdout)

    delta_gap = abs(lfd1.Delta - lfd0.Delta)
    lfd_gap = maximum(abs.(lfd1.weights .- lfd0.weights))
    @printf("\n|Delta1 - Delta0| = %.3e\n", delta_gap)
    @printf("max|weights1 - weights0| (LFD agreement) = %.3e\n", lfd_gap)

    # per-cell moment residuals, split active vs profiled, at BOTH states
    D2 = D * D
    function moment_resid_by_group(lfd)
        # moment_residuals is ordered by melitz_outer_layout's own moment index; first D^2
        # entries are the bilateral trade moments in od2lin (column-major) order, per
        # moments.jl/delta_star.jl convention used throughout this repo.
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
    act0, prof0 = moment_resid_by_group(lfd0)
    act1, prof1 = moment_resid_by_group(lfd1)
    @printf("\n[FULL]      max active-cell moment resid=%.3e   max PROFILED-cell moment resid=%.3e\n", act0, prof0)
    @printf("[COMPLETED] max active-cell moment resid=%.3e   max PROFILED-cell moment resid=%.3e\n", act1, prof1)

    pass = lfd1.lfd_ok && delta_gap < 1e-5 && lfd_gap < 1e-4 && prof1 < 1e-4 && act1 < 1e-4
    println("\nPHASE 2 (real-D20 leg) REDUCED/FULL EQUIVALENCE: ", pass ? "PASS" : "FAIL")
    return (; lfd0, lfd1, theta0, theta1, delta_gap, lfd_gap, act0, prof0, act1, prof1, calib, profiled_od)
end

result = main()
