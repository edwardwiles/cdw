# Reduced-q-subspace outer-search backend (2026-07-29 continuation session), Phase 11: D4
# derivative and stage shakedown at the mandatory development W=320,000, with one
# representative W=1,280,000 spot check.
#
# Governing-prompt targets are Delta*~0.5 and Delta*~2 -- D4's own fixed-A/f gamma-profile
# corridor tops out at Delta~0.572 (established, not re-derived, see
# docs/melitz_aq_q_bandwidth_convergence_2026-07-29.md Phase 3's own table: "D4's fixed-A/f
# corridor tops out at Delta~0.572 -- Delta~1/2 are NOT reachable there"), so this script uses
# the two REACHABLE D4 fixtures -- target=0.5 (the closest available to the first requested
# point) and the "pareto" calibration point (Delta~0.572, the closest available to the second
# requested Delta~2 point) -- and discloses this substitution explicitly rather than silently
# reporting against an unreachable target.

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random
println("Julia threads: ", Threads.nthreads()); flush(stdout)

const OUTDIR = joinpath(REPO, "docs", "key_results")
CAP = 10.0
policy_cap = CappedEvaluation(CAP)

function load_theta_q_rows(path)
    rows = Dict{Tuple{String,Float64},Vector{Float64}}()
    for line in eachline(path)
        parts = split(line, ",")
        rows[(parts[1], parse(Float64, parts[2]))] = parse.(Float64, parts[5:end])
    end
    return rows
end
theta_q_rows = load_theta_q_rows(joinpath(OUTDIR, "melitz_qbw_phase3_theta_q_2026-07-29.csv"))

function build_bundle_at_W(W::Int; base_seed::Int=29)
    data0 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=base_seed, W=20_000)
    z = pareto_draws(W, data0.primitives.D, data0.primitives.theta_star; seed=base_seed, mode=:halton)
    data = MelitzSyntheticData(data0.primitives, data0.equilibrium, data0.counterfactual, data0.L, z, base_seed)
    obj, _ = build_melitz_psi_bundle(data; outer_parameterization=:logcutoff,
        policy=policy_cap, backend=:matrix_free, forbid_dense_fallback=true)
    return obj, obj.γ
end

results = NamedTuple[]

function run_point(label::String, target::Float64, W::Int)
    println("\n" * "="^100); @printf("label=%s target=%.3f W=%d\n", label, target, W); println("="^100); flush(stdout)
    theta0 = theta_q_rows[("D4_seed29_W20000", target)]
    obj, ctx = build_bundle_at_W(W)
    D = ctx.D; nA = D^2-1; nq = D^2-2
    obj.use_cached_x = false; obj.x .= NaN
    lfd0 = melitz_recover_lfd(obj, theta0)
    if !lfd0.lfd_ok
        @printf("[SKIP] base point failed to verify at W=%d\n", W)
        return
    end
    x0 = copy(lfd0.dual_x)
    @printf("Delta0=%.6f\n", lfd0.Delta); flush(stdout)

    t_build0 = time()
    stage = melitz_build_reduced_q_stage(collect(theta0), x0, ctx, obj, 1;
        bandwidth_policy=PowerScaledQBandwidth(1e-3, 80_000, 0.5), target_switches=100)
    t_build = time() - t_build0
    if stage === nothing
        @printf("[NO DIRECTION FOUND] label=%s target=%.3f W=%d\n", label, target, W)
        push!(results, (label=label, target=target, W=W, Delta0=lfd0.Delta, direction_found=false,
                         h_s=NaN, secant_fd_central=NaN, secant_reopt_central=NaN, relerr_central=NaN,
                         s_lo=NaN, s_hi=NaN, build_time_s=t_build,
                         onesided_s=NaN, onesided_pred=NaN, onesided_actual=NaN, onesided_relerr=NaN))
        return
    end
    @printf("stage built: |q_basis_free|=%.6e  s_lo=%.4f  s_hi=%.4f  (build_time=%.2fs)\n",
            norm(stage.q_basis_free), stage.s_lo, stage.s_hi, t_build); flush(stdout)

    # Transformed constraints correctness check (cheap, always run).
    C_r, b_r, sys = melitz_reduced_affine_cutoff_system(stage, ctx)
    rng = MersenneTwister(hash((label, target, W)))
    x_reduced0 = vcat(theta0[1], theta0[2:1+nA], 0.0)
    maxdiff = 0.0
    for _ in 1:10
        xr = x_reduced0 .+ 0.02 .* randn(rng, length(x_reduced0))
        theta_full = melitz_reduced_full_theta(xr, stage, ctx)
        maxdiff = max(maxdiff, maximum(abs.((C_r*xr .+ b_r) .- (sys.C*theta_full .+ sys.b))))
    end
    @printf("transformed-constraint max discrepancy vs production (10 random points): %.3e\n", maxdiff); flush(stdout)

    # Scalar-s central derivative: direct fixed-dual vs fully reoptimized central secant AT THE SAME h_s.
    g_reduced = zeros(2+nA)
    t_grad0 = time()
    info = melitz_reduced_q_gradient!(g_reduced, x_reduced0, stage, ctx, obj, x0)
    t_grad = time() - t_grad0
    h_s = info[2].h_s
    secant_reopt, ok_p, ok_m = melitz_q_direct_block_secant(collect(theta0), stage.q_basis_free, h_s, obj, ctx, x0; mode=:reoptimized)
    relerr_central = (isfinite(secant_reopt) && secant_reopt != 0.0) ? abs(g_reduced[end]-secant_reopt)/abs(secant_reopt) : NaN
    @printf("scalar-s: h_s=%.4e (one_sided=%s)  fixed_dual_central=%.6e  reoptimized_central=%.6e (ok=%s)  relerr=%.4f  (grad_time=%.2fs)\n",
            h_s, info[2].one_sided, g_reduced[end], secant_reopt, ok_p&&ok_m, relerr_central, t_grad); flush(stdout)

    # One-sided prediction at a stage-relevant s value (s = 0.5 * min(1,s_hi), a genuine
    # within-stage-range amplitude, not the tiny crossing-calibrated h_s).
    s_test = 0.5 * min(1.0, stage.s_hi)
    pred_onesided = g_reduced[end] * s_test   # linear extrapolation from the central derivative at s=0
    theta_full_test = melitz_reduced_full_theta(vcat(x_reduced0[1:end-1], s_test), stage, ctx)
    obj.use_cached_x = false; obj.x .= NaN
    lfd_test = melitz_recover_lfd(obj, theta_full_test)
    actual_onesided = lfd_test.lfd_ok ? (lfd_test.Delta - lfd0.Delta) : NaN
    onesided_relerr = (lfd_test.lfd_ok && actual_onesided != 0.0) ? abs(pred_onesided-actual_onesided)/abs(actual_onesided) : NaN
    @printf("one-sided stage check: s=%.4f  pred(linear from central deriv)=%.6e  actual(reoptimized)=%.6e (ok=%s)  relerr=%.4f\n",
            s_test, pred_onesided, actual_onesided, lfd_test.lfd_ok, onesided_relerr); flush(stdout)
    melitz_update_operator_at_theta!(obj.op, theta0, ctx)

    push!(results, (label=label, target=target, W=W, Delta0=lfd0.Delta, direction_found=true,
                     h_s=h_s, secant_fd_central=g_reduced[end], secant_reopt_central=secant_reopt,
                     relerr_central=relerr_central, s_lo=stage.s_lo, s_hi=stage.s_hi, build_time_s=t_build,
                     onesided_s=s_test, onesided_pred=pred_onesided, onesided_actual=actual_onesided,
                     onesided_relerr=onesided_relerr))
end

for (label, target) in [("target_0.5", 0.5), ("pareto_0.572", 0.0)]
    run_point(label, target, 320_000)
end

# One representative W=1,280,000 spot check (governing prompt: "if the reduced method looks
# promising, repeat one representative D4 derivative and stage test at W=1,280,000" -- run
# unconditionally here since this IS that one representative check, not a full grid).
run_point("target_0.5_bigW", 0.5, 1_280_000)

open(joinpath(OUTDIR, "melitz_reducedq_phase11_d4_shakedown_2026-07-29.csv"), "w") do io
    println(io, "label,target,W,Delta0,direction_found,h_s,secant_fd_central,secant_reopt_central,relerr_central,s_lo,s_hi,build_time_s,onesided_s,onesided_pred,onesided_actual,onesided_relerr")
    for r in results
        println(io, join([r.label, r.target, r.W, r.Delta0, r.direction_found, r.h_s, r.secant_fd_central,
                           r.secant_reopt_central, r.relerr_central, r.s_lo, r.s_hi, r.build_time_s,
                           r.onesided_s, r.onesided_pred, r.onesided_actual, r.onesided_relerr], ","))
    end
end
println("\nPhase 11 shakedown complete. Rows: ", length(results))
flush(stdout)
