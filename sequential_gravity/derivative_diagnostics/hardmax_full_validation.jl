# Full 3-step hard-max validation (focal check, hard-max non-focal inversion via
# rho-continuation, gravity-moment check) on the 3 example points from
# HARDMAX_INVERSION_HANDOFF_PROMPT.md. Needs one KNITRO inner solve per point (recover_lfd);
# everything else (the actual hard-max inversion machinery) is KNITRO-free.
#
# v2: flushes output after every line (v1 had none -- silent for 40+ min with no visibility),
# threads the destination loop (v1 ran it serially despite -t 19; PARALLEL_INVERSION-style,
# matching run_profiled_production.jl's own invert_all convention), and picks up the new
# NewtonTrustRegion default for invert_destination's rho>0 solves for free (no code change
# needed here -- that default lives in profiled_gravity.jl now).
#
#   FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true julia -t 19 --project=. \
#     sequential_gravity/derivative_diagnostics/hardmax_full_validation.jl
ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
include(joinpath(@__DIR__, "hardmax_inversion.jl"))
using Printf, JLD2, LinearAlgebra, Base.Threads

BLAS.set_num_threads(1)  # avoid oversubscription vs the Threads.@threads destination loop below
@printf("[julia threads = %d]\n", Threads.nthreads()); flush(stdout)

function build_theta_point2()
    d = JLD2.load(joinpath(@__DIR__, "..", "head_to_head", "out_lu", "lu_T2_Astar.jld2"))
    @assert d["done"] == true
    vcat(0.11432339572008998, 2.5, d["gammap_target"], d["best_feasible_Aod"])
end
function build_theta_point3()
    d = JLD2.load(joinpath(@__DIR__, "..", "head_to_head", "out_lu_multistart50", "lu_ms_T3_pt40.jld2"))
    @assert d["done"] == true
    vcat(0.11432339572008998, 2.5, d["gammap_target"], d["best_feasible_Aod"])
end

POINTS = [
    ("Point1_LC_T1", JLD2.load(joinpath(@__DIR__, "..", "head_to_head", "out_lc", "lc_T1_Astar.jld2"))["best_feasible_theta"]),
    ("Point2_LU_T2", build_theta_point2()),
    ("Point3_LUms_T3", build_theta_point3()),
]

function run_point(name, θ)
    println("\n" * "="^78); println(">>> ", name); println("="^78); flush(stdout)
    @assert θ[2] == σ "sigma mismatch for $name"
    @printf("mu=%.6f sigma=%.4f gammap_focal=%.6f\n", θ[1], θ[2], θ[3]); flush(stdout)

    log_x = build_log_x(Uσ, θ[1])
    uf = focal_u(θ)
    t0 = time()
    p, ok = recover_lfd(θ, EK_moments_focal_norm_directgp!, D + 1)
    @printf("recover_lfd: ok=%s wall=%.1fs\n", ok, time() - t0); flush(stdout)
    @assert ok "$name: recover_lfd failed"
    logp = log.(p)

    # Step 1: focal check (rho=0 hard-argmin, no inversion -- closed form)
    focal_shares, _ = dest_share(log_x, logp, uf; ρ = 0.0)
    focal_err = maximum(abs.(focal_shares .- λData[:, focal]))
    @printf("STEP 1 focal check: max|model-empirical|=%.3e\n", focal_err); flush(stdout)

    umat_smooth = zeros(D, D); umat_hard = zeros(D, D)
    umat_smooth[:, focal] = uf; umat_hard[:, focal] = uf
    hard_errs = zeros(D); smooth_hard_errs = zeros(D); u_diffs = zeros(D)
    homotopy_ok_vec = trues(D); rho0_converged_vec = trues(D); smooth_ok_vec = trues(D)
    t0 = time()
    prog = Threads.Atomic{Int}(0)
    Threads.@threads for d in omitted
        inv_smooth = invert_destination(log_x, p, λData[:, d]; ref = ref, ρ = ρ, tol = 1e-6, maxit = 150, ls_iters = 100)
        fallback_used = false
        if !inv_smooth.converged
            # trustregion (default) failed to converge on this destination -- fall back to the
            # hand-rolled solver before giving up, rather than crashing the whole run. Report
            # BOTH outcomes so a "trustregion struggles here" pattern is visible, not hidden.
            inv_hr = invert_destination(log_x, p, λData[:, d]; ref = ref, ρ = ρ, tol = 1e-6, maxit = 150,
                ls_iters = 100, method = :handrolled)
            @printf("  [%s] destination %2d: trustregion did NOT converge (share_err=%.3e) -- falling back to handrolled (converged=%s, share_err=%.3e)\n",
                name, d, inv_smooth.max_abs_share_error, inv_hr.converged, inv_hr.max_abs_share_error); flush(stdout)
            fallback_used = true
            inv_smooth = inv_hr
        end
        smooth_ok_vec[d] = inv_smooth.converged
        umat_smooth[:, d] = inv_smooth.u_full
        smooth_hard_shares, _ = dest_share(log_x, logp, inv_smooth.u_full; ρ = 0.0)
        smooth_hard_errs[d] = maximum(abs.(smooth_hard_shares .- λData[:, d]))

        hm = hardmax_invert_destination(log_x, p, λData[:, d]; ref = ref, u_init = inv_smooth.u_full)
        umat_hard[:, d] = hm.u
        hard_errs[d] = hm.hard_err
        u_diffs[d] = norm(hm.u .- inv_smooth.u_full)
        homotopy_ok_vec[d] = hm.homotopy_ok
        rho0_converged_vec[d] = hm.rho0_converged
        n = Threads.atomic_add!(prog, 1) + 1
        @printf("  [%s] destination %2d done (%2d/%d)  smooth_ok=%s%s hard_err=%.3e\n",
            name, d, n, D - 1, inv_smooth.converged, fallback_used ? "(via handrolled)" : "", hm.hard_err); flush(stdout)
    end
    if !all(smooth_ok_vec[omitted])
        bad = [d for d in omitted if !smooth_ok_vec[d]]
        @printf("WARNING: %s: destinations %s did NOT converge even after the handrolled fallback -- results for these are unreliable, proceeding anyway\n",
            name, string(bad)); flush(stdout)
    end
    homotopy_all_ok = all(homotopy_ok_vec[omitted]); rho0_all_converged = all(rho0_converged_vec[omitted])
    @printf("STEP 2 hard-max inversion (all %d omitted destinations): wall=%.1fs  homotopy_all_ok=%s  rho0_all_converged=%s\n",
        D - 1, time() - t0, homotopy_all_ok, rho0_all_converged); flush(stdout)
    @printf("  hard_err: max=%.3e mean=%.3e median=%.3e  (this is the honest achieved gap, may be a genuine nonzero floor)\n",
        maximum(hard_errs[omitted]), sum(hard_errs[omitted]) / (D - 1), sort(hard_errs[omitted])[cld(D - 1, 2)]); flush(stdout)
    @printf("  [informational] rho=%.4g solution's OWN rho=0 gap (no hard-max inversion, just re-evaluated): max=%.3e mean=%.3e\n",
        ρ, maximum(smooth_hard_errs[omitted]), sum(smooth_hard_errs[omitted]) / (D - 1)); flush(stdout)
    @printf("  ||u_hardmax - u_rho2e-3||: max=%.3e mean=%.3e\n", maximum(u_diffs[omitted]), sum(u_diffs[omitted]) / (D - 1)); flush(stdout)

    gr_smooth = gravity_residual(umat_smooth, logτ, logw, σ)
    gr_hard = gravity_residual(umat_hard, logτ, logw, σ)
    @printf("STEP 3 gravity moment: R_mean(smoothed umat)=%.4e   R_mean(hardmax umat)=%.4e\n", gr_smooth.R_mean, gr_hard.R_mean); flush(stdout)

    return (name = name, focal_err = focal_err, hard_errs = copy(hard_errs), smooth_hard_errs = copy(smooth_hard_errs),
        u_diffs = copy(u_diffs), homotopy_all_ok = homotopy_all_ok, rho0_all_converged = rho0_all_converged,
        R_mean_smooth = gr_smooth.R_mean, R_mean_hard = gr_hard.R_mean, umat_smooth = umat_smooth, umat_hard = umat_hard)
end

results = NamedTuple[]
for (name, θ) in POINTS
    push!(results, run_point(name, θ))
end

println("\n" * "="^78); println(">>> SUMMARY"); println("="^78); flush(stdout)
@printf("%-16s %10s %14s %14s %14s %14s\n", "point", "focal_err", "hardmax_maxerr", "smoothgap_maxerr", "R_mean(smooth)", "R_mean(hard)"); flush(stdout)
for r in results
    @printf("%-16s %10.2e %14.2e %14.2e %14.2e %14.2e\n", r.name, r.focal_err,
        maximum(r.hard_errs[omitted]), maximum(r.smooth_hard_errs[omitted]), r.R_mean_smooth, r.R_mean_hard)
    flush(stdout)
end

JLD2.save(joinpath(@__DIR__, "hardmax_full_validation_results.jld2"), "results", results)
println("\nFULL VALIDATION DONE"); flush(stdout)
