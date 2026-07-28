# Governing prompt (2026-07-28 outer-search gamma-profile session), Phase 8: short real-D20
# diagnosis -- user-directed refinement: initialize the outer search AT the Phase 2 profile
# point where DeltaStar(g) ~= delta (the fixed-A/f boundary), which is KNOWN FEASIBLE and
# already near the budget, rather than at the Pareto calibration point (governing prompt's
# own "start near DeltaStar~0.7-0.95" instruction, sharpened by the user mid-session to the
# EXACT known root g_fixed=-0.49783321 from the 2026-07-24 companion report / this session's
# own Phase 2 profile, Delta(g_fixed)=0.9969 -- as close to the delta=1 budget as a verified
# point gets without crossing it).
#
# Joint (full A/f + g) search from this point, Active Set, var_scale + Phase 1's
# objective_scale fix BOTH applied, a MODERATE block-scaled box (g_radius=0.05, A/f
# radius=0.15 log-units -- the SAME box the 2026-07-24 companion campaign used successfully,
# chosen here specifically to bound the overshoot Phase 3 just found live at box radius=0.15
# gamma-only) -- external_incumbent=the boundary starting point itself, so the reported
# answer can never be worse than the known-feasible floor.
#
# Usage: julia --project=. -t 20 scripts/melitz_phase8_realD20_boundary_start_2026-07-28.jl

using Pkg
Pkg.activate(dirname(@__DIR__))
using Printf, DelimitedFiles, LinearAlgebra
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

const OUTDIR = joinpath(dirname(@__DIR__), "docs", "key_results")
mkpath(OUTDIR)
const REPO = dirname(@__DIR__)
const S_G, S_A, S_F = 1e-4, 1e-5, 1e-5
const G_FIXED_BOUNDARY = -0.49783321   # 2026-07-24 companion report's own bisected Delta(g)=1 root

function block_scaled_box(n, D; g_radius, A_radius=0.15, f_radius=0.15)
    nA = D^2 - 1
    box = zeros(n)
    box[1] = g_radius
    box[2:1+nA] .= A_radius
    box[2+nA:end] .= f_radius
    return box
end

function main()
    real_dir = joinpath(REPO, "real_data", "noah_D20")
    lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
    BLAS.set_num_threads(20)
    inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2024-07-24.opt")
    inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
    obj, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
        inner_loop_opt=inner_opt, forbid_dense_fallback=true)
    ctx = obj.γ
    n = length(theta_calib); D = ctx.D

    theta_boundary = copy(theta_calib); theta_boundary[1] = G_FIXED_BOUNDARY
    r_boundary = evaluate_melitz_delta(theta_boundary, ctx, obj; cold=true, store_G=false)
    @assert r_boundary.verified "boundary starting point failed to verify"
    @printf("Boundary starting point: g=%.8f  Delta=%.6e  nStatus=%d\n", G_FIXED_BOUNDARY, r_boundary.Delta, r_boundary.nStatus)

    var_scale = ones(n); var_scale[1] = S_G
    nA = D^2 - 1
    var_scale[2:1+nA] .= S_A
    var_scale[2+nA:end] .= S_F
    var_center = copy(theta_boundary)
    box = block_scaled_box(n, D; g_radius=0.05, A_radius=0.15, f_radius=0.15)
    outer_active = joinpath(REPO, "melitz_outer_finite_delta_alg_active_2026-07-27.opt")

    melitz_profile_reset!(); MELITZ_PROFILE[] = true
    local res
    wall = @elapsed begin
        res = solve_melitz_finite_delta_bound(ctx, obj, theta_boundary; delta=1.0, direction=:upper,
            delta_evaluation_cap=10.0, gradient_backend=:B_direct_argument_parallel, h=1e-4,
            theta_box=box, cutoff_constraint_backend=:linear,
            inner_loop_opt=inner_opt, outer_loop_opt=outer_active,
            var_scale=var_scale, var_center=var_center,
            backend=:matrix_free, forbid_dense_fallback=true,
            objective_scale=S_G, external_incumbent=theta_boundary)
    end
    MELITZ_PROFILE[] = false

    println("\n" * "="^100); println("PHASE 8 RESULT (boundary-start, joint, objective_scale-fixed)"); println("="^100)
    @printf("wall=%.2fs  nStatus=%d  n_fc=%d  n_ga=%d  n_inner_solved=%d\n",
        wall, res.nStatus, res.n_fc_calls, res.n_ga_calls, res.n_inner_solved)
    @printf("n_infinite_delta_reject=%d  n_above_cap_reject=%d  n_numerical_failure_reject=%d\n",
        res.n_infinite_delta_reject, res.n_above_cap_reject, res.n_numerical_failure_reject)
    for (label, cand) in (("initial_incumbent", res.initial_incumbent),
                          ("best_live_incumbent", res.best_live_incumbent),
                          ("cold_verified_incumbent (THE ANSWER)", res.cold_verified_incumbent))
        if cand === nothing
            @printf("  %-38s: nothing\n", label)
        else
            g = cand.eval.theta_free[1]
            @printf("  %-38s: g=%.6f  Delta=%.6e  source=%s\n", label, g, cand.eval.Delta, cand.source)
        end
    end
    dg = res.terminal_eval.theta_free[1] - G_FIXED_BOUNDARY
    dlogA = norm(res.terminal_eval.theta_free[2:1+nA] .- theta_boundary[2:1+nA])
    dlogf = norm(res.terminal_eval.theta_free[2+nA:end] .- theta_boundary[2+nA:end])
    @printf("\nMovement from boundary start: dg=%.6e  norm(dlogA)=%.6e  norm(dlogf)=%.6e\n", dg, dlogA, dlogf)

    println("\n" * "="^100); println("WALL-CLOCK DECOMPOSITION"); println("="^100)
    melitz_profile_report(stdout; trajectory_total_s=wall)

    outfile = joinpath(OUTDIR, "melitz_phase8_realD20_boundary_start_2026-07-28.csv")
    open(outfile, "w") do io
        println(io, "label,g,Delta,source")
        for (label, cand) in (("initial_incumbent", res.initial_incumbent),
                              ("best_live_incumbent", res.best_live_incumbent),
                              ("cold_verified_incumbent", res.cold_verified_incumbent))
            if cand !== nothing
                println(io, "$label,$(cand.eval.theta_free[1]),$(cand.eval.Delta),$(cand.source)")
            end
        end
    end
    BLAS.set_num_threads(1)
    println("\nDONE. CSV written to ", outfile)
    return res
end

main()
