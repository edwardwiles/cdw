# Phase 5 (profiledA_parallel_speed_and_cutoff_portfolio governing prompt, 2026-07-30, MANDATORY):
# whole-solve thread-scaling benchmark. Launched as a separate clean Julia process per thread
# count (see melitz_phase5_launch_all_2026-07-30.sh) -- this script measures COMPLETE
# profiled-A middle solves and a short welfare-continuation segment, not merely a Hessian
# kernel microbenchmark. BLAS threads pinned to 1 always (Julia threads vary, the axis under
# test).
#
# Benchmarks:
#   A. one representative fixed-q middle solve from the anchor (row_states idx=1 input state)
#   B. one near-boundary middle solve (row_states idx=6 input state, GT=7.0969%/Delta*=0.499)
#   C. a fixed three-point welfare-continuation segment (idx=1,2,3 in sequence, each using its
#      own precomputed continuation-start input state -- exercises the SAME inter-point warm
#      state carryover the real welfare continuation uses, not three isolated solves)
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Serialization
LinearAlgebra.BLAS.set_num_threads(1)
melitz_thread_startup_report()
const NTHREADS = Threads.nthreads()
println("Julia threads: ", NTHREADS, "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

const OUTDIR = joinpath(REPO2, "docs", "key_results")
CAP = 10.0
policy_cap = CappedEvaluation(CAP)
MIDDLE_OPT_D20 = joinpath(REPO2, "melitz_middle_loop_opt_2026-07-30.opt")
const D20_MIDDLE_BOX = 0.1
const MAX_MIDDLE_EVALS = 120
const CAP_HANDLING = :barrier
const CAP_BARRIER_MULTIPLE = 5.0

function proc_rss_kb()
    for line in eachline("/proc/self/status")
        startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
    end
    return -1
end
function proc_cpu_seconds()
    fields = split(read("/proc/self/stat", String))
    utime = parse(Int, fields[14]); stime = parse(Int, fields[15])
    return (utime + stime) / 100.0   # CLK_TCK=100 on this host
end

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
t_load0 = time()
calib, focal = load_realD20_calib()
obj_d20, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
    inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
ctx_d20 = obj_d20.γ
D20 = ctx_d20.D; nA20 = D20^2 - 1
session_d20 = MelitzInnerSession(obj_d20, ctx_d20, policy_cap)
t_load = time() - t_load0
println("Fixture load/calibration wall: ", round(t_load; digits=2), "s"); flush(stdout)

function q_full_at_g_factory(theta_plain0_d20)
    return g_target -> begin
        th = copy(theta_plain0_d20)
        th[1] = g_target
        _, _, _, _, q = expand_free_theta_logcutoff(th, ctx_d20)
        q
    end
end
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
theta_plain0_d20 = melitz_unpower_theta_free(theta0_d20, ctx_d20)
q_full_at_g = q_full_at_g_factory(theta_plain0_d20)

struct RowInputState
    idx::Int; g_target::Float64; prev_A_free::Vector{Float64}; prev_q::Matrix{Float64}
    prev_p_star::Vector{Float64}; stored_GT::Float64; stored_Delta::Float64
    stored_classification::Symbol; stored_accepted::Bool; stored_best_start::Symbol
end
loaded = deserialize(joinpath(@__DIR__, "melitz_phase2_row_states_2026-07-30.jls"))
row_states = Dict(rs.idx => rs for rs in loaded.row_states)

function one_middle_solve(rs::RowInputState)
    gpj_target = exp(rs.g_target)
    q_target = q_full_at_g(rs.g_target)
    theta_for_constraints = melitz_fixed_q_state_theta(rs.prev_A_free, q_target, gpj_target, ctx_d20)
    sys_t = melitz_fixed_q_middle_constraint_system(theta_for_constraints, ctx_d20, obj_d20)
    A_cont = melitz_project_start_to_middle_constraints(copy(rs.prev_A_free), sys_t, ctx_d20)
    session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
    t0 = time()
    res = solve_melitz_fixed_q_A_profile_v2(session_d20, q_target, gpj_target, A_cont, ctx_d20;
        coordinate=:logA, max_evals=MAX_MIDDLE_EVALS, box=D20_MIDDLE_BOX, outer_loop_opt=MIDDLE_OPT_D20, sys=sys_t,
        cap_handling=CAP_HANDLING, cap_barrier_multiple=CAP_BARRIER_MULTIPLE)
    wall = time() - t0
    return res, wall
end

results = Dict{String,Any}()

println("\n== Benchmark A: representative fixed-q middle solve (idx=1, anchor-adjacent) =="); flush(stdout)
melitz_profile_reset!(); MELITZ_PROFILE[] = true
resA, wallA = one_middle_solve(row_states[1])
MELITZ_PROFILE[] = false
resultsA_profile = melitz_profile_summary()
@printf("  Delta=%.10f classification=%s wall=%.2fs unique_solves=%d fc+ga=%d\n",
    resA.Delta_incumbent, nameof(typeof(resA.r_incumbent)), wallA, resA.unique_inner_solves,
    resA.n_fc_calls + resA.n_ga_calls)
results["A"] = (Delta=resA.Delta_incumbent, classification=nameof(typeof(resA.r_incumbent)), wall_s=wallA,
    unique_A_points=resA.unique_A_points, unique_inner_solves=resA.unique_inner_solves,
    fc_calls=resA.n_fc_calls, ga_calls=resA.n_ga_calls, cache_hits=resA.cache_hits,
    obj_s=sum((r.total_s for r in resultsA_profile if r.category==:fc_inner_obj_eval), init=0.0),
    grad_s=sum((r.total_s for r in resultsA_profile if r.category==:fc_inner_grad_eval), init=0.0),
    hess_s=sum((r.total_s for r in resultsA_profile if r.category==:fc_inner_hess_eval), init=0.0),
    operator_s=sum((r.total_s for r in resultsA_profile if r.category==:moment_operator_link_update), init=0.0))

println("\n== Benchmark B: near-boundary fixed-q middle solve (idx=6, GT=7.0969%) =="); flush(stdout)
melitz_profile_reset!(); MELITZ_PROFILE[] = true
resB, wallB = one_middle_solve(row_states[6])
MELITZ_PROFILE[] = false
resultsB_profile = melitz_profile_summary()
@printf("  Delta=%.10f classification=%s wall=%.2fs unique_solves=%d fc+ga=%d\n",
    resB.Delta_incumbent, nameof(typeof(resB.r_incumbent)), wallB, resB.unique_inner_solves,
    resB.n_fc_calls + resB.n_ga_calls)
results["B"] = (Delta=resB.Delta_incumbent, classification=nameof(typeof(resB.r_incumbent)), wall_s=wallB,
    unique_A_points=resB.unique_A_points, unique_inner_solves=resB.unique_inner_solves,
    fc_calls=resB.n_fc_calls, ga_calls=resB.n_ga_calls, cache_hits=resB.cache_hits,
    obj_s=sum((r.total_s for r in resultsB_profile if r.category==:fc_inner_obj_eval), init=0.0),
    grad_s=sum((r.total_s for r in resultsB_profile if r.category==:fc_inner_grad_eval), init=0.0),
    hess_s=sum((r.total_s for r in resultsB_profile if r.category==:fc_inner_hess_eval), init=0.0),
    operator_s=sum((r.total_s for r in resultsB_profile if r.category==:moment_operator_link_update), init=0.0))

println("\n== Benchmark C: three-point welfare-continuation segment (idx=1,2,3 in sequence) =="); flush(stdout)
melitz_profile_reset!(); MELITZ_PROFILE[] = true
t0C = time()
seg_deltas = Float64[]
seg_unique_solves = 0
for i in (1, 2, 3)
    global seg_unique_solves
    resi, walli = one_middle_solve(row_states[i])
    push!(seg_deltas, resi.Delta_incumbent)
    seg_unique_solves += resi.unique_inner_solves
    @printf("  segment idx=%d Delta=%.10f wall=%.2fs\n", i, resi.Delta_incumbent, walli)
    flush(stdout)
end
wallC = time() - t0C
MELITZ_PROFILE[] = false
results["C"] = (Deltas=seg_deltas, wall_s=wallC, unique_inner_solves=seg_unique_solves)

peak_rss_mb = proc_rss_kb() / 1024
cpu_seconds = proc_cpu_seconds()
total_wall = wallA + wallB + wallC + t_load
cpu_util_pct = total_wall > 0 ? 100 * cpu_seconds / total_wall : NaN

outpath = joinpath(OUTDIR, "melitz_phase5_thread_scaling_T$(NTHREADS)_2026-07-30.csv")
open(outpath, "w") do io
    println(io, "nthreads,benchmark,Delta,classification,wall_s,unique_A_points,unique_inner_solves,fc_calls,ga_calls,cache_hits,obj_s,grad_s,hess_s,operator_s,peak_rss_mb,cpu_seconds,cpu_util_pct,fixture_load_s")
    a = results["A"]; b = results["B"]
    println(io, join([NTHREADS, "A", a.Delta, a.classification, a.wall_s, a.unique_A_points, a.unique_inner_solves,
        a.fc_calls, a.ga_calls, a.cache_hits, a.obj_s, a.grad_s, a.hess_s, a.operator_s, peak_rss_mb, cpu_seconds, cpu_util_pct, t_load], ","))
    println(io, join([NTHREADS, "B", b.Delta, b.classification, b.wall_s, b.unique_A_points, b.unique_inner_solves,
        b.fc_calls, b.ga_calls, b.cache_hits, b.obj_s, b.grad_s, b.hess_s, b.operator_s, peak_rss_mb, cpu_seconds, cpu_util_pct, t_load], ","))
    c = results["C"]
    println(io, join([NTHREADS, "C", join(c.Deltas, ";"), "segment", c.wall_s, "", c.unique_inner_solves,
        "", "", "", "", "", "", "", peak_rss_mb, cpu_seconds, cpu_util_pct, t_load], ","))
end
@printf("\npeak_rss_mb=%.1f  cpu_seconds=%.1f  cpu_util_pct=%.1f  total_wall_s=%.1f\n",
    peak_rss_mb, cpu_seconds, cpu_util_pct, total_wall)
println("Wrote ", outpath)
println("\nPHASE 5 THREAD-SCALING (T=", NTHREADS, ") COMPLETE")
