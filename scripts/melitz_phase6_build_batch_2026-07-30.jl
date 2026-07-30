# Phase 6 (profiledA_parallel_speed_and_cutoff_portfolio governing prompt, 2026-07-30):
# builds the fixed batch of independent jobs used by the fixed-total-core throughput
# benchmark. Per the governing prompt: "at least 20 representative jobs: several
# continuation-start middle solves; several compensated-start middle solves; upper- and
# lower-direction welfare points". Uses the 13 stored upper-direction row states
# (melitz_phase2_row_states_2026-07-30.jls) plus their own lower-direction counterparts
# (rebuilt fresh here the same way, from the stored lower-direction CSV/state), giving
# BOTH starts (continuation + cellwise-compensated) at every row -- comfortably >=20 jobs,
# with genuine upper AND lower coverage.
#
# Each job is fully self-contained (q_target, gpj_target, A_start, sys, coordinate) so a
# worker process can run it directly via solve_melitz_fixed_q_A_profile_v2 without any of
# this script's own state-threading logic.
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Serialization
LinearAlgebra.BLAS.set_num_threads(1)
melitz_thread_startup_report()
println("Julia threads: ", Threads.nthreads()); flush(stdout)

const OUTDIR = joinpath(REPO2, "docs", "key_results")
CAP = 10.0
policy_cap = CappedEvaluation(CAP)

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
obj_d20, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
    inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
ctx_d20 = obj_d20.γ
D20 = ctx_d20.D; nA20 = D20^2 - 1
sorted_ctx_d20 = ctx_d20.sorted_tail_ctx
sigma_d20 = ctx_d20.sigma

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
function q_full_at_g(g_target::Real)
    th = copy(theta_plain0_d20)
    th[1] = g_target
    _, _, _, _, q = expand_free_theta_logcutoff(th, ctx_d20)
    return q
end

struct RowInputState
    idx::Int; g_target::Float64; prev_A_free::Vector{Float64}; prev_q::Matrix{Float64}
    prev_p_star::Vector{Float64}; stored_GT::Float64; stored_Delta::Float64
    stored_classification::Symbol; stored_accepted::Bool; stored_best_start::Symbol
end
loaded_upper = deserialize(joinpath(@__DIR__, "melitz_phase2_row_states_2026-07-30.jls"))
upper_row_states = [(direction=:upper, rs=rs) for rs in loaded_upper.row_states]

# Lower-direction row states: build with the SAME precompute recipe as Phase 2's own script,
# inline here (small -- 15 rows, cheap melitz_recover_lfd calls only at accepted rows).
mutable struct ContinuationPoint
    idx::Int; g::Float64; GT::Float64; classification::Symbol; Delta::Float64
    accepted::Bool; best_start::Symbol; A_free::Vector{Float64}; theta_free::Vector{Float64}
end
st = deserialize(joinpath(REPO2, "scripts", "melitz_d20_profiledA_continuation_state_2026-07-30.jls"))
lower_points = st.result_lower.points
A_free0_d20 = loaded_upper.A_free0_d20
q0_d20 = loaded_upper.q0_d20
p_star_d20 = loaded_upper.p_star_d20
CAP_l = 10.0
session_d20 = MelitzInnerSession(obj_d20, ctx_d20, policy_cap)

cur_A_free = copy(A_free0_d20); cur_q = copy(q0_d20); cur_p_star = copy(p_star_d20)
lower_row_states = NamedTuple[]
for p in lower_points
    global cur_A_free, cur_q, cur_p_star
    p.idx == 0 && continue
    push!(lower_row_states, (direction=:lower, rs=RowInputState(p.idx, p.g, copy(cur_A_free), copy(cur_q),
        copy(cur_p_star), p.GT, p.Delta, p.classification, p.accepted, p.best_start)))
    if p.accepted
        cur_A_free = p.A_free
        _, _, _, _, cur_q = expand_free_theta_logcutoff(p.theta_free, ctx_d20)
        session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
        lfd_r = melitz_recover_lfd(obj_d20, p.theta_free)
        @assert lfd_r.lfd_ok
        cur_p_star = copy(lfd_r.weights)
    end
end
println("Built ", length(lower_row_states), " lower-direction row states"); flush(stdout)

all_rows = vcat(upper_row_states, lower_row_states)
println("Total rows available (upper+lower): ", length(all_rows)); flush(stdout)

struct Phase6Job
    job_id::Int
    direction::Symbol
    idx::Int
    start_kind::Symbol   # :continuation or :cellwise_compensated
    q_target::Matrix{Float64}
    gpj_target::Float64
    A_start::Vector{Float64}
    sys::MelitzFixedQMiddleConstraintSystem
    stored_GT::Float64
end

jobs = Phase6Job[]
job_id = 0
for row in all_rows
    rs = row.rs
    gpj_target = exp(rs.g_target)
    q_target = q_full_at_g(rs.g_target)
    theta_for_constraints = melitz_fixed_q_state_theta(rs.prev_A_free, q_target, gpj_target, ctx_d20)
    sys_t = melitz_fixed_q_middle_constraint_system(theta_for_constraints, ctx_d20, obj_d20)
    A_cont = melitz_project_start_to_middle_constraints(copy(rs.prev_A_free), sys_t, ctx_d20)
    global job_id
    job_id += 1
    push!(jobs, Phase6Job(job_id, row.direction, rs.idx, :continuation, q_target, gpj_target, A_cont, sys_t, rs.stored_GT))

    prev_A_full = exp.(reshape(pivot_expand(rs.prev_A_free, ctx_d20.A_pivot), D20, D20))
    A_cellwise, status_cellwise = melitz_cellwise_A_from_moments(prev_A_full, rs.prev_q, q_target, rs.prev_p_star,
                                                                   sorted_ctx_d20, sigma_d20)
    n_bad = count(!=(:ok), status_cellwise)
    if n_bad > 0
        A_cellwise[status_cellwise .!= :ok] .= prev_A_full[status_cellwise .!= :ok]
    end
    A_cellwise_free = pivot_reduce(vec(log.(A_cellwise)), ctx_d20.A_pivot)
    A_comp = melitz_project_start_to_middle_constraints(A_cellwise_free, sys_t, ctx_d20)
    job_id += 1
    push!(jobs, Phase6Job(job_id, row.direction, rs.idx, :cellwise_compensated, q_target, gpj_target, A_comp, sys_t, rs.stored_GT))
end

println("Built ", length(jobs), " total jobs (", count(j -> j.direction == :upper, jobs), " upper, ",
    count(j -> j.direction == :lower, jobs), " lower; ",
    count(j -> j.start_kind == :continuation, jobs), " continuation-start, ",
    count(j -> j.start_kind == :cellwise_compensated, jobs), " compensated-start)")
@assert length(jobs) >= 20 "governing prompt requires >=20 jobs"

serialize(joinpath(@__DIR__, "melitz_phase6_job_batch_2026-07-30.jls"), jobs)
println("\nWrote scripts/melitz_phase6_job_batch_2026-07-30.jls (", length(jobs), " jobs)")
