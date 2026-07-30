# Phase 6 batch worker: processes a contiguous slice of the fixed job batch
# (melitz_phase6_job_batch_2026-07-30.jls, built by melitz_phase6_build_batch_2026-07-30.jl)
# sequentially in ONE OS process (separate KNITRO session per job, never concurrent KN_solve
# calls within a process -- this project's own established rule). Used by
# melitz_phase6_launch_config_2026-07-30.sh to realize a given (n_processes x threads_per_process)
# allocation: the dispatcher launches one of these per process, each with a disjoint job slice
# and `-t threads_per_process`.
#
# Usage: julia --project=. -t <T> melitz_phase6_batch_worker_2026-07-30.jl <job_start> <job_end> <out_csv> <config_label>
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Serialization
LinearAlgebra.BLAS.set_num_threads(1)
melitz_thread_startup_report()
const NTHREADS = Threads.nthreads()
println("Julia threads: ", NTHREADS); flush(stdout)

job_start = parse(Int, ARGS[1])
job_end = parse(Int, ARGS[2])
out_csv = ARGS[3]
config_label = ARGS[4]
println("Phase6 worker: jobs ", job_start, "..", job_end, "  config=", config_label, "  T=", NTHREADS); flush(stdout)

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
    (parse(Int, fields[14]) + parse(Int, fields[15])) / 100.0
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
session_d20 = MelitzInnerSession(obj_d20, ctx_d20, policy_cap)
t_load = time() - t_load0

struct Phase6Job
    job_id::Int
    direction::Symbol
    idx::Int
    start_kind::Symbol
    q_target::Matrix{Float64}
    gpj_target::Float64
    A_start::Vector{Float64}
    sys::MelitzFixedQMiddleConstraintSystem
    stored_GT::Float64
end
all_jobs = deserialize(joinpath(@__DIR__, "melitz_phase6_job_batch_2026-07-30.jls"))
my_jobs = [j for j in all_jobs if job_start <= j.job_id <= job_end]
println("Assigned ", length(my_jobs), " jobs (fixture load ", round(t_load; digits=2), "s)"); flush(stdout)

results = NamedTuple[]
t_batch0 = time()
for j in my_jobs
    session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
    t0 = time()
    res = solve_melitz_fixed_q_A_profile_v2(session_d20, j.q_target, j.gpj_target, j.A_start, ctx_d20;
        coordinate=:logA, max_evals=MAX_MIDDLE_EVALS, box=D20_MIDDLE_BOX, outer_loop_opt=MIDDLE_OPT_D20, sys=j.sys,
        cap_handling=CAP_HANDLING, cap_barrier_multiple=CAP_BARRIER_MULTIPLE)
    wall = time() - t0
    finite = res.r_incumbent isa FiniteSolved
    @printf("  job=%d dir=%s idx=%d start=%s GT=%.4f%% Delta=%.6g finite=%s wall=%.1fs unique_solves=%d\n",
        j.job_id, j.direction, j.idx, j.start_kind, j.stored_GT, res.Delta_incumbent, finite, wall, res.unique_inner_solves)
    flush(stdout)
    push!(results, (job_id=j.job_id, direction=String(j.direction), idx=j.idx, start_kind=String(j.start_kind),
        GT=j.stored_GT, Delta=res.Delta_incumbent, finite=finite, wall_s=wall,
        unique_A_points=res.unique_A_points, unique_inner_solves=res.unique_inner_solves,
        fc_calls=res.n_fc_calls, ga_calls=res.n_ga_calls, cache_hits=res.cache_hits))
end
t_batch = time() - t_batch0
peak_rss_mb = proc_rss_kb() / 1024
cpu_seconds = proc_cpu_seconds()

open(out_csv, "w") do io
    println(io, "config_label,nthreads,job_id,direction,idx,start_kind,GT,Delta,finite,wall_s,unique_A_points,unique_inner_solves,fc_calls,ga_calls,cache_hits,fixture_load_s,worker_batch_wall_s,peak_rss_mb,cpu_seconds")
    for r in results
        println(io, join([config_label, NTHREADS, r.job_id, r.direction, r.idx, r.start_kind, r.GT, r.Delta, r.finite,
            r.wall_s, r.unique_A_points, r.unique_inner_solves, r.fc_calls, r.ga_calls, r.cache_hits,
            t_load, t_batch, peak_rss_mb, cpu_seconds], ","))
    end
end
@printf("\nworker done: n_jobs=%d worker_batch_wall_s=%.1f peak_rss_mb=%.1f cpu_seconds=%.1f\n",
    length(my_jobs), t_batch, peak_rss_mb, cpu_seconds)
println("Wrote ", out_csv)
println("PHASE6 WORKER (", config_label, ", jobs ", job_start, "-", job_end, ") COMPLETE")
