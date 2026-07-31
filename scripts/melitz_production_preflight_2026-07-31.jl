# Preflight for the D20 profiled-A production campaign at delta=0.5
# (docs/melitz_d20_profiledA_production_delta0p5_2026-07-31.md governing prompt).
#
# 1. Cold-reverify the stored headline point (GT~7.0969%, Delta*~0.499019) under CURRENT code.
# 2. Confirm FiniteSolved, no dense G (forbid_dense_fallback=true), BLAS threads=1.
# 3. Run one ordinary profiled point from current_calibration and verify the returned incumbent
#    is no worse than the verified start, unique inner solves <= unique A points, no
#    NumericalFailure.
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Serialization
melitz_thread_startup_report()
LinearAlgebra.BLAS.set_num_threads(1)
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

const OUTDIR = joinpath(REPO2, "docs", "key_results")
CAP = 10.0
policy_cap = CappedEvaluation(CAP)
MIDDLE_OPT_D20 = joinpath(REPO2, "melitz_middle_loop_opt_2026-07-30.opt")
const D20_MIDDLE_BOX = 0.1
const MAX_MIDDLE_EVALS = 120
const CAP_HANDLING = :barrier
const CAP_BARRIER_MULTIPLE = 5.0

println("\n", "="^100); println("STEP 0: build real-D20 fixture fresh"); println("="^100); flush(stdout)

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
@assert calib !== nothing
println("focal country index = ", focal); flush(stdout)

t_build0 = time()
obj_d20, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
    inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
@printf("fixture build: %.2fs\n", time() - t_build0); flush(stdout)
ctx_d20 = obj_d20.γ
D20 = ctx_d20.D; nA20 = D20^2 - 1
session_d20 = MelitzInnerSession(obj_d20, ctx_d20, policy_cap)
println("D=", D20, "  nA=", nA20, "  forbid_dense_fallback=true (no dense G)"); flush(stdout)

function load_theta_q_rows(path)
    rows = Dict{Tuple{String,Float64},Vector{Float64}}()
    for line in eachline(path)
        parts = split(line, ",")
        rows[(parts[1], parse(Float64, parts[2]))] = parse.(Float64, parts[5:end])
    end
    return rows
end
theta_q_rows = load_theta_q_rows(joinpath(OUTDIR, "melitz_qbw_phase3_theta_q_2026-07-29.csv"))
theta0_d20 = theta_q_rows[("realD20_seed1_W80000", 0.5)]

println("\n", "="^100); println("STEP 1: cold-reverify anchor Delta0 = 0.483276"); println("="^100); flush(stdout)
obj_d20.use_cached_x = false; obj_d20.x .= NaN
t0 = time()
lfd0 = melitz_recover_lfd(obj_d20, theta0_d20)
@printf("anchor solve: %.2fs  Delta0=%.10f  lfd_ok=%s  nStatus=%d\n", time() - t0, lfd0.Delta, lfd0.lfd_ok, lfd0.nStatus)
@assert lfd0.lfd_ok
@assert isapprox(lfd0.Delta, 0.483276; atol=1e-4)
println("PASS: anchor Delta0 reproduces 0.483276 fresh."); flush(stdout)

println("\n", "="^100); println("STEP 2: cold-reverify the STORED headline point (GT~7.0969%, Delta*~0.499019)"); println("="^100); flush(stdout)
# Matches scripts/melitz_d20_profiled_A_welfare_continuation_2026-07-30.jl's own struct
# definition exactly -- required in scope for `deserialize` to reconstruct the stored state.
mutable struct ContinuationPoint
    idx::Int
    g::Float64
    GT::Float64
    classification::Symbol
    Delta::Float64
    accepted::Bool
    best_start::Symbol
    A_free::Vector{Float64}
    theta_free::Vector{Float64}
end
state_path = joinpath(REPO2, "scripts", "melitz_d20_profiledA_continuation_state_2026-07-30.jls")
@assert isfile(state_path) "stored continuation state file not found: $state_path"
stored = deserialize(state_path)
ext = stored.result_upper.most_extreme
@printf("stored extreme point: GT=%.10f%%  Delta_stored=%.10f  classification=%s\n", ext.GT, ext.Delta, ext.classification)
@assert ext.classification == :FiniteSolved
@assert isapprox(ext.GT, 7.0969; atol=0.01)
@assert isapprox(ext.Delta, 0.499019; atol=0.001)

obj_d20.use_cached_x = false; obj_d20.x .= NaN
t0 = time()
r_headline = solve_melitz_delta!(session_d20, ext.theta_free, policy_cap; warm_start_source=:neutral)
wall_headline = time() - t0
@assert r_headline isa FiniteSolved "cold reverification did not return FiniteSolved: $(typeof(r_headline))"
@printf("LIVE cold re-verification: %.2fs  Delta=%.10f  nStatus=%d  (stored=%.10f)\n",
    wall_headline, r_headline.Delta, r_headline.nStatus, ext.Delta)
@assert isapprox(r_headline.Delta, ext.Delta; atol=1e-6) "live Delta does not match stored Delta to 1e-6"
@assert isapprox(r_headline.Delta, 0.499019; atol=1e-4)
GT0_headline = 100 * melitz_welfare_metrics_from_g(ext.g, ctx_d20.w_prime / ctx_d20.w[ctx_d20.target_country], ctx_d20.sigma).gains_from_trade
@printf("GT at headline point (recomputed): %.6f%%\n", GT0_headline)
@assert isapprox(GT0_headline, 7.0969; atol=0.01)
println("PASS: stored headline point reproduces GT~7.0969%%, Delta*~0.499019 fresh, FiniteSolved."); flush(stdout)

println("\n", "="^100); println("STEP 3: full primal/dual/LFD verification residuals at the headline point"); println("="^100); flush(stdout)
lfd_headline = melitz_recover_lfd(obj_d20, ext.theta_free)
@assert lfd_headline.lfd_ok
@printf("lfd_ok=%s  Delta(lfd)=%.10f  nStatus=%d\n", lfd_headline.lfd_ok, lfd_headline.Delta, lfd_headline.nStatus)
@assert isapprox(lfd_headline.Delta, ext.Delta; atol=1e-6)
println("PASS: LFD recovery agrees with the cold-reverified Delta."); flush(stdout)

println("\n", "="^100); println("STEP 4: one ordinary profiled point from current_calibration (anchor g0)"); println("="^100); flush(stdout)
theta_plain0_d20 = melitz_unpower_theta_free(theta0_d20, ctx_d20)
A_free0_d20 = theta_plain0_d20[2:1+nA20]
g0 = theta_plain0_d20[1]
gpj0 = exp(g0)
_, f0_full, _, _, q0_d20 = expand_free_theta_logcutoff(theta_plain0_d20, ctx_d20)
sys0 = melitz_fixed_q_middle_constraint_system(theta_plain0_d20, ctx_d20, obj_d20)
A_start0 = melitz_project_start_to_middle_constraints(copy(A_free0_d20), sys0, ctx_d20)

session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
t0 = time()
r_profile0 = solve_melitz_fixed_q_A_profile_v2(session_d20, q0_d20, gpj0, A_start0, ctx_d20;
    coordinate=:logA, max_evals=MAX_MIDDLE_EVALS, box=D20_MIDDLE_BOX, outer_loop_opt=MIDDLE_OPT_D20, sys=sys0,
    cap_handling=CAP_HANDLING, cap_barrier_multiple=CAP_BARRIER_MULTIPLE)
wall_profile0 = time() - t0
@printf("profiled point: %.2fs  incumbent_source=%s  Delta_incumbent=%.6g  Delta_start_verified=%.6g\n",
    wall_profile0, r_profile0.incumbent_source, r_profile0.Delta_incumbent, r_profile0.Delta_start_verified)
@printf("  unique_A_points=%d  unique_inner_solves=%d  cache_hits=%d  n_fc=%d  n_ga=%d\n",
    r_profile0.unique_A_points, r_profile0.unique_inner_solves, r_profile0.cache_hits,
    r_profile0.n_fc_calls, r_profile0.n_ga_calls)
@assert r_profile0.r_incumbent isa FiniteSolved "profiled anchor point did not return FiniteSolved"
@assert r_profile0.Delta_incumbent <= r_profile0.Delta_start_verified + 1e-6 "incumbent worse than verified start"
@assert r_profile0.unique_inner_solves <= r_profile0.unique_A_points "unique_inner_solves exceeded unique_A_points"
println("PASS: profiled anchor point -- incumbent no worse than verified start, dedup invariant holds, FiniteSolved throughout."); flush(stdout)

println("\n", "="^100); println("PREFLIGHT SUMMARY"); println("="^100)
println("  BLAS threads = ", LinearAlgebra.BLAS.get_num_threads(), " (must be 1)")
@assert LinearAlgebra.BLAS.get_num_threads() == 1
println("  forbid_dense_fallback = true (no dense G)")
println("  cap_handling = :barrier, cap_barrier_multiple = 5.0 (repaired v2 driver, callback cache active)")
println("  focal-link fast path: wired into production unconditionally as of commit 169f77d (no opt-out flag)")
@printf("  anchor Delta0 = %.10f (target 0.483276)\n", lfd0.Delta)
@printf("  headline point: GT=%.6f%%  Delta*=%.10f (target GT~7.0969%%, Delta*~0.499019)\n", GT0_headline, r_headline.Delta)
@printf("  ordinary profiled point at anchor: Delta_incumbent=%.6g FiniteSolved, dedup OK\n", r_profile0.Delta_incumbent)
println("\nALL PREFLIGHT CHECKS PASSED -- cleared to launch the production campaign.")
