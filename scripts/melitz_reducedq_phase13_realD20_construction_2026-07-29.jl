# Reduced-q-subspace outer-search backend (2026-07-29 continuation session), Phase 13:
# LIMITED real-D20 construction/callback shakedown. NOT a D20 outer campaign (governing
# prompt's own explicit rule) -- one verified real-D20 point at Delta*~0.5 (seed=1, W=80,000,
# the SAME fixture Phase 10 of the q-bandwidth campaign used, `realD20_seed1_W80000`/target=0.5
# from `melitz_qbw_phase3_theta_q_2026-07-29.csv`), construction/callback correctness only.

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random
melitz_thread_startup_report()
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
theta0 = theta_q_rows[("realD20_seed1_W80000", 0.5)]

function load_realD20_calib()
    real_dir = joinpath(REPO, "real_data", "noah_D20")
    lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    return calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal,
        p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6), focal
end
calib, focal = load_realD20_calib()

obj, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
    inner_loop_opt=joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
ctx = obj.γ
D = ctx.D; nA = D^2-1; nq = D^2-2
println("D=", D, "  nA=", nA, "  nq=", nq); flush(stdout)

obj.use_cached_x = false; obj.x .= NaN
t0 = time()
lfd0 = melitz_recover_lfd(obj, theta0)
println("base-point solve: ", time()-t0, "s  Delta0=", lfd0.Delta, "  lfd_ok=", lfd0.lfd_ok)
@assert lfd0.lfd_ok
x0 = copy(lfd0.dual_x)
flush(stdout)

# 1. reduced state reconstructed correctly.
t_build0 = time()
stage = melitz_build_reduced_q_stage(collect(theta0), x0, ctx, obj, 1;
    bandwidth_policy=PowerScaledQBandwidth(1e-3, 80_000, 0.5), target_switches=100)
t_build = time() - t_build0
@assert stage !== nothing
println("stage built (", t_build, "s): |q_basis_free|=", norm(stage.q_basis_free),
        "  s_lo=", stage.s_lo, "  s_hi=", stage.s_hi)
x_reduced0 = vcat(theta0[1], theta0[2:1+nA], 0.0)
theta_full_reconstructed = melitz_reduced_full_theta(x_reduced0, stage, ctx)
println("s=0 reproduces anchor bit-for-bit: ", theta_full_reconstructed == collect(theta0))
flush(stdout)

# 2. transformed constraints correct.
C_r, b_r, sys = melitz_reduced_affine_cutoff_system(stage, ctx)
rng = MersenneTwister(1)
maxdiff = 0.0
for _ in 1:10
    global maxdiff
    xr = x_reduced0 .+ 0.01 .* randn(rng, length(x_reduced0))
    theta_f = melitz_reduced_full_theta(xr, stage, ctx)
    maxdiff = max(maxdiff, maximum(abs.((C_r*xr .+ b_r) .- (sys.C*theta_f .+ sys.b))))
end
println("transformed-constraint max discrepancy vs production (10 random points, D=20): ", maxdiff)
flush(stdout)

# 3. direct scalar derivative -- no dense G, bounded memory, timing.
before_dense = MELITZ_DENSE_G_MATERIALIZATIONS[]
mem_before = Base.gc_live_bytes()
g_reduced = zeros(2+nA)
t_grad0 = time()
info = melitz_reduced_q_gradient!(g_reduced, x_reduced0, stage, ctx, obj, x0)
t_grad = time() - t_grad0
mem_after = Base.gc_live_bytes()
after_dense = MELITZ_DENSE_G_MATERIALIZATIONS[]
println("reduced gradient computed in ", t_grad, "s.  dense-G materializations: before=", before_dense, " after=", after_dense,
        " (delta=", after_dense-before_dense, ")")
println("live heap bytes: before=", mem_before, " after=", mem_after, " (delta_MB=", (mem_after-mem_before)/1e6, ")")
println("g_reduced[1] (welfare)=", g_reduced[1], "  g_reduced[end] (s, direct block secant)=", g_reduced[end],
        "  h_s=", info[2].h_s, " one_sided=", info[2].one_sided)
flush(stdout)

# 4. one direct scalar fixed-dual secant vs one fully reoptimized central secant (the ONE
# comparison this phase asks for -- not a sweep).
h_s = info[2].h_s
t_reopt0 = time()
secant_reopt, ok_p, ok_m = melitz_q_direct_block_secant(collect(theta0), stage.q_basis_free, h_s, obj, ctx, x0; mode=:reoptimized)
t_reopt = time() - t_reopt0
relerr = (isfinite(secant_reopt) && secant_reopt != 0.0) ? abs(g_reduced[end]-secant_reopt)/abs(secant_reopt) : NaN
println("\nFINAL COMPARISON (D=20, W=80000, Delta0=", lfd0.Delta, "):")
println("  direct block fixed-dual central secant = ", g_reduced[end])
println("  fully reoptimized central secant        = ", secant_reopt, " (ok=", ok_p&&ok_m, ", took ", t_reopt, "s)")
println("  relative error                          = ", relerr)
melitz_update_operator_at_theta!(obj.op, theta0, ctx)

open(joinpath(OUTDIR, "melitz_reducedq_phase13_realD20_construction_2026-07-29.csv"), "w") do io
    println(io, "D,W,Delta0,stage_build_s,constraint_maxdiff,grad_time_s,dense_G_delta,live_heap_delta_MB,h_s,one_sided,secant_fixed_dual,secant_reopt,relerr,reopt_time_s")
    println(io, join([D, 80_000, lfd0.Delta, t_build, maxdiff, t_grad, after_dense-before_dense,
                       (mem_after-mem_before)/1e6, h_s, info[2].one_sided, g_reduced[end], secant_reopt, relerr, t_reopt], ","))
end
println("\nPhase 13 construction test complete.")
flush(stdout)
