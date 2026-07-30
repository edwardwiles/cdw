# Phase 0: recover exact real-D20 reduced-q anchor state + direction, verify tri-backend
# reproduction. Governing prompt: melitz_d20_negative_switch_geometry_audit_2026-07-30.
const REPO = joinpath(@__DIR__, "..", "..", "..", "..", "..", "..", "bbkinghome", "edav",
                       "gravity_robustness", "trade_robustness_modular")
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random, Serialization
melitz_thread_startup_report()
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

const OUTDIR = joinpath(REPO2, "docs", "key_results")
const SCRATCH = @__DIR__
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
    real_dir = joinpath(REPO2, "real_data", "noah_D20")
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
println("focal country index = ", focal); flush(stdout)

function build_bundle()
    obj, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
        inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
        forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
    return obj
end

obj = build_bundle()
ctx = obj.γ
D = ctx.D; nA = D^2-1; nq = D^2-2
println("D=", D, "  nA=", nA, "  nq=", nq, "  obj.mode=", obj.mode, "  obj.lower_limit=", obj.lower_limit); flush(stdout)

obj.use_cached_x = false; obj.x .= NaN
t0 = time()
lfd0 = melitz_recover_lfd(obj, theta0)
@printf("base-point solve: %.2fs  Delta0=%.10f  lfd_ok=%s  nStatus=%d\n", time()-t0, lfd0.Delta, lfd0.lfd_ok, lfd0.nStatus)
@assert lfd0.lfd_ok
@assert isapprox(lfd0.Delta, 0.483276; atol=1e-4) "base Delta0 mismatch vs documented 0.483276"
x0 = copy(lfd0.dual_x)
flush(stdout)

# cross-check via the typed classifier too (production-facing entry point)
session0 = MelitzInnerSession(obj, ctx, policy_cap)
r0 = solve_melitz_delta!(session0, theta0, policy_cap)
println("typed classifier at anchor: ", typeof(r0), "  Delta=", hasproperty(r0,:Delta_theta) ? r0.Delta_theta : "n/a")
flush(stdout)

# ================================================================================================
# direction / basis reconstruction (identical to Gate 3B / forensic session)
# ================================================================================================
bwpolicy = PowerScaledQBandwidth(1e-3, 80_000, 0.5)
sorted_ctx = ctx.sorted_tail_ctx
stage = melitz_build_reduced_q_stage(theta0, x0, ctx, obj, 1; bandwidth_policy=bwpolicy, target_switches=100)
@assert stage !== nothing
b_q = stage.q_basis_free
r_basis = norm(b_q)
@printf("basis: r_basis=%.6f  |b_q|=%.6f  s_lo=%.6f  s_hi=%.6f\n", r_basis, norm(b_q), stage.s_lo, stage.s_hi)
tp1, tm1 = melitz_q_direction_two_sided_crossings(theta0, b_q, 1.0, ctx, sorted_ctx)
@printf("crossings at s=+-1: (+%d,-%d)  [documented (+117,-100)]\n", tp1, tm1)
flush(stdout)

# ================================================================================================
# Tri-backend reproduction check: (1) reduced-q state map, (2) direct full-state evaluator
# (melitz_recover_lfd on theta0 directly -- already done above as backend (2) baseline),
# (3) production (A,f) evaluator after converting the SAME state into (A,f) via :logf reduce.
# ================================================================================================
println("\n" * "="^100); println("Tri-backend anchor reproduction"); println("="^100); flush(stdout)

# (1) reduced-q state map: x_reduced = (g, A_free..., s=0) must reproduce theta0 bit-for-bit.
x_reduced0 = vcat(theta0[1], theta0[2:1+nA], 0.0)
theta_via_reduced = melitz_reduced_full_theta(x_reduced0, stage, ctx)
diff_reduced = maximum(abs.(theta_via_reduced .- theta0))
println("max|theta_via_reduced_map - theta0| = ", diff_reduced, "  (must be 0.0 exactly, s=0 bit-identical)")
@assert diff_reduced == 0.0

# (2) direct full-state evaluator: already lfd0 above (melitz_recover_lfd(obj, theta0)).
println("(2) direct full-state Delta0 = ", lfd0.Delta)

# (3) production (A,f) evaluator: convert theta0 (:logcutoff free vector) into (A,f,gamma'_j),
# then into a :logf theta_free via reduce_to_free_theta_logcutoff's OWN sibling
# (delta_star.jl's expand_free_theta / reduce_to_free_theta for :logf), then evaluate via
# build_melitz_implicit_bundle / solve_melitz_finite_delta_bound's own bundle construction path
# (mode=:implicit, matching production exactly) but WITHOUT running an outer KNITRO search --
# just a single classified inner solve at the converted point.
A0, f0, gpj0, fjj0 = melitz_expand_theta(theta0, ctx)
ctx_logf = merge(ctx, (outer_parameterization=:logf,))
p0 = MelitzPrimitives(D, ctx.sigma, ctx.theta_star, ctx.target_country, ctx.tau, ctx.w, A0, f0, gpj0)
theta0_logf = melitz_reduce_theta(p0, ctx_logf)
A0b, f0b, gpj0b, fjj0b = melitz_expand_theta(theta0_logf, ctx_logf)
println("round-trip (A,f) via :logf reduce/expand max|A diff|=", maximum(abs.(A0 .- A0b)),
        "  max|f diff|=", maximum(abs.(f0 .- f0b)), "  |gpj diff|=", abs(gpj0-gpj0b))

obj_logf, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
    inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logf)
obj_logf.use_cached_x = false; obj_logf.x .= NaN
lfd0_logf = melitz_recover_lfd(obj_logf, theta0_logf)
@printf("(3) production (:logf, mode=%s) Delta0 = %.10f  lfd_ok=%s  nStatus=%d\n",
        obj_logf.mode, lfd0_logf.Delta, lfd0_logf.lfd_ok, lfd0_logf.nStatus)
diff_delta_13 = abs(lfd0.Delta - lfd0_logf.Delta)
println("max|Delta(reduced/logcutoff) - Delta(:logf production-equivalent)| = ", diff_delta_13)

# session via mode=:implicit build_melitz_implicit_bundle (the actual production bundle
# constructor path solve_melitz_finite_delta_bound itself uses)
obj_impl = build_melitz_implicit_bundle(ctx_logf, obj.U, theta0_logf; delta=10.0, find_smallest=true,
    inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    outer_loop_opt=joinpath(REPO2, "melitz_outer_finite_delta_alg_direct_2026-07-27.opt"),
    policy=policy_cap, forbid_dense_fallback=true)
session_impl = MelitzInnerSession(obj_impl, ctx_logf, policy_cap)
r0_impl = solve_melitz_delta!(session_impl, theta0_logf, policy_cap)
println("production mode=:implicit typed classifier @ converted anchor: ", typeof(r0_impl))
if hasproperty(r0_impl, :Delta_theta)
    println("  Delta_theta=", r0_impl.Delta_theta)
end

# Persist state.
serialize(joinpath(SCRATCH, "phase0_state.jls"),
    (theta0=theta0, x0=x0, calib=calib, focal=focal, D=D, nA=nA, nq=nq,
     b_q=b_q, r_basis=r_basis, stage=stage, Delta0=lfd0.Delta,
     tp1=tp1, tm1=tm1, W=80_000, qmc_seed=1))
println("\nPersisted phase0 state to phase0_state.jls")
flush(stdout)
println("DONE PHASE 0")
