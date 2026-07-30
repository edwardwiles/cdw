# Phase 2 (profiledA_parallel_speed_and_cutoff_portfolio governing prompt, 2026-07-30):
# precomputes, for every row of the STORED upper-direction continuation path
# (docs/key_results/melitz_d20_profiledA_continuation_points_2026-07-30.csv), the exact
# (g_target, prev_A_free, prev_q, prev_p_star) inputs that profile_phi_at_g would have received
# at that row in the ORIGINAL always-two-start run -- i.e. replays the ORIGINAL script's own
# state-threading rule ("update cur_A_free/cur_q/cur_p_star only on accepted==true rows, in
# stored order") using the stored per-row theta_free, WITHOUT re-running any middle-loop
# optimization (only cheap melitz_recover_lfd single inner solves at the handful of accepted
# rows, matching the original script's own cur_p_star update). Writes a small .jls that the
# parallel row-worker script(s) load, so the actual (expensive) two-start/adaptive comparison
# can be split across independent OS processes without re-deriving this shared prefix state
# sequentially inside each one.
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
session_d20 = MelitzInnerSession(obj_d20, ctx_d20, policy_cap)

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
A_free0_d20 = theta_plain0_d20[2:1+nA20]
g0 = theta_plain0_d20[1]

session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
lfd0 = melitz_recover_lfd(obj_d20, theta0_d20)
@assert lfd0.lfd_ok
p_star_d20 = copy(lfd0.weights)
_, _, _, _, q0_d20 = expand_free_theta_logcutoff(theta_plain0_d20, ctx_d20)

# --- Load the STORED upper-direction points (CSV, ground truth for g/theta reconstruction). ---
struct StoredRow
    idx::Int
    g::Float64
    GT::Float64
    classification::Symbol
    Delta::Float64
    accepted::Bool
    best_start::Symbol
end
rows = StoredRow[]
open(joinpath(OUTDIR, "melitz_d20_profiledA_continuation_points_2026-07-30.csv")) do io
    readline(io)  # header
    for line in eachline(io)
        p = split(line, ",")
        p[1] == "upper" || continue
        push!(rows, StoredRow(parse(Int, p[2]), parse(Float64, p[3]), parse(Float64, p[4]),
            Symbol(p[5]), parse(Float64, p[6]), p[7] == "true", Symbol(p[8])))
    end
end
sort!(rows, by=r -> r.idx)
println("Loaded ", length(rows), " stored upper-direction rows (idx 0..", rows[end].idx, ")"); flush(stdout)

# theta_free per row is NOT in the CSV (only in the .jls state) -- load it from there.
mutable struct ContinuationPoint
    idx::Int; g::Float64; GT::Float64; classification::Symbol; Delta::Float64
    accepted::Bool; best_start::Symbol; A_free::Vector{Float64}; theta_free::Vector{Float64}
end
st = deserialize(joinpath(REPO2, "scripts", "melitz_d20_profiledA_continuation_state_2026-07-30.jls"))
stored_points = st.result_upper.points  # includes idx=0 (anchor) through idx=13
@assert length(stored_points) == length(rows)
theta_free_by_idx = Dict(p.idx => p.theta_free for p in stored_points)
A_free_by_idx = Dict(p.idx => p.A_free for p in stored_points)

# --- Replay state-threading EXACTLY as run_continuation_direction did: update
# cur_A_free/cur_q/cur_p_star only on accepted==true rows, in stored idx order. Precompute the
# (g_target, prev_A_free, prev_q, prev_p_star) INPUT state for every row (idx 1..13; idx 0 is
# the anchor itself, already profiled as phi0 in the original script). ---
struct RowInputState
    idx::Int
    g_target::Float64
    prev_A_free::Vector{Float64}
    prev_q::Matrix{Float64}
    prev_p_star::Vector{Float64}
    stored_GT::Float64
    stored_Delta::Float64
    stored_classification::Symbol
    stored_accepted::Bool
    stored_best_start::Symbol
end

cur_A_free = copy(A_free0_d20)
cur_q = copy(q0_d20)
cur_p_star = copy(p_star_d20)
row_states = RowInputState[]
for r in rows
    r.idx == 0 && continue  # anchor itself, not a middle-loop trial
    global cur_A_free, cur_q, cur_p_star
    push!(row_states, RowInputState(r.idx, r.g, copy(cur_A_free), copy(cur_q), copy(cur_p_star),
        r.GT, r.Delta, r.classification, r.accepted, r.best_start))
    if r.accepted
        cur_A_free = A_free_by_idx[r.idx]
        theta_f = theta_free_by_idx[r.idx]
        _, _, _, _, cur_q = expand_free_theta_logcutoff(theta_f, ctx_d20)
        session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
        lfd_r = melitz_recover_lfd(obj_d20, theta_f)
        @assert lfd_r.lfd_ok
        cur_p_star = copy(lfd_r.weights)
    end
end
println("Precomputed ", length(row_states), " row input states (idx ", row_states[1].idx, "..", row_states[end].idx, ")")
for rs in row_states
    @printf("  idx=%2d  GT_target=%.6f%%  stored_accepted=%-5s  stored_best_start=%-20s  stored_Delta=%.6g\n",
        rs.idx, rs.stored_GT, rs.stored_accepted, rs.stored_best_start, rs.stored_Delta)
end
flush(stdout)

serialize(joinpath(@__DIR__, "melitz_phase2_row_states_2026-07-30.jls"),
    (row_states=row_states, A_free0_d20=A_free0_d20, q0_d20=q0_d20, p_star_d20=p_star_d20, g0=g0))
println("\nWrote scripts/melitz_phase2_row_states_2026-07-30.jls")
