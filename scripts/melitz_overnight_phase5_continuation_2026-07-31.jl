# Overnight continuation, Phase 5: full scalar welfare continuation from a retained cutoff
# candidate's best verified Phase 3/4 point, holding q(m) fixed throughout.
#
# ARGS = [start_jls_path, chain_label, direction]
#   start_jls_path: a docs/key_results/overnight_qpoll_delta0p5_2026-07-31/phase3_phase4_points/
#                    *.jls dump (must have lfd_ok=true -- refuses an unverified start)
#   chain_label:     free-form id for this chain's own checkpoint/output files
#   direction:       "upper" or "lower"
#
# Combines: the ORIGINAL production driver's wide-expansion-then-bracket discovery (search widely
# first, matching the governing prompt's "search widely first; polish only once a local frontier
# is identified") with the Phase-1 resume driver's safeguarded-interpolation / bracket-subdivision
# polish logic AND its lfd_ok incumbent gate (a point primal-FiniteSolved-but-lfd_ok=false does
# NOT count as a usable/most-extreme incumbent -- confirmed a genuine near-cliff dual degeneracy
# in this campaign, not a bug). q_free is held FIXED at the candidate's own value throughout; only
# g and A/f vary.
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Dates, Serialization
melitz_thread_startup_report()
LinearAlgebra.BLAS.set_num_threads(1)
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

length(ARGS) >= 3 || error("usage: julia melitz_overnight_phase5_continuation_2026-07-31.jl <start_jls_path> <chain_label> <direction>")
const START_JLS = ARGS[1]
const CHAINID = ARGS[2]
const DIRECTION = Symbol(ARGS[3])
DIRECTION in (:upper, :lower) || error("direction must be upper or lower")
const SIGN = DIRECTION == :upper ? 1 : -1

const OUTDIR = joinpath(REPO2, "docs", "key_results")
const OVERDIR = joinpath(OUTDIR, "overnight_qpoll_delta0p5_2026-07-31")
const P5DIR = joinpath(OVERDIR, "phase5")
const CKPT_PATH = joinpath(P5DIR, "checkpoints", "$(CHAINID).jls")
const POINTS_CSV = joinpath(P5DIR, "points", "$(CHAINID)_points.csv")
mkpath(dirname(CKPT_PATH)); mkpath(dirname(POINTS_CSV))

isfile(START_JLS) || error("start point not found: $START_JLS")
start_pt = deserialize(START_JLS)
start_pt.lfd_ok || error("[$CHAINID] refusing to start Phase 5 from an lfd_ok=false point: $START_JLS")
println("[$CHAINID] start point: GT=$(start_pt.GT_target) Delta=$(start_pt.Delta) m=$(start_pt.m) (verified)")
flush(stdout)

CAP = 10.0
policy_cap = CappedEvaluation(CAP)
const MIDDLE_OPT_D20 = joinpath(REPO2, "melitz_middle_loop_opt_2026-07-30.opt")
const D20_MIDDLE_BOX = 0.1
const MAX_MIDDLE_EVALS = 180
const CAP_HANDLING = :barrier
const CAP_BARRIER_MULTIPLE = 5.0
const DELTA_BUDGET = 0.5

const GT_STEP0 = 0.5
const GT_STEP_GROWTH = 1.5
const GT_STEP_MAX = 2.0
const GT_BRACKET_STOP_TOL = 0.005
const DELTA_STOP_TOL = 0.001
const INTERP_CLAMP = (0.2, 0.8)
const MAX_WELFARE_POINTS = 30
const MAX_WALL_S = parse(Float64, get(ENV, "MELITZ_MAX_WALL_S", "21600"))
const MAX_NONIMPROVING = 6
const PERIODIC_K = 4

# ============================================================================
# Real-D20 fixture.
# ============================================================================
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
sigma_d20 = ctx_d20.sigma
sorted_ctx_d20 = ctx_d20.sorted_tail_ctx
session_d20 = MelitzInnerSession(obj_d20, ctx_d20, policy_cap)
wage_ratio_d20 = ctx_d20.w_prime / ctx_d20.w[ctx_d20.target_country]
gt_of_g(g::Real) = 100 * melitz_welfare_metrics_from_g(g, wage_ratio_d20, sigma_d20).gains_from_trade
g_of_gt(gt_pct::Real) = (sigma_d20 - 1) * log((1 - gt_pct / 100) / wage_ratio_d20)

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
nA20check = nA20
q_free_fixed = start_pt.q_free   # THE retained cutoff candidate, held fixed throughout.

function q_full_at_g_anchor(g_target::Real)
    th = copy(theta_plain0_d20)
    th[1] = g_target
    th[2+nA20:end] .= q_free_fixed
    _, _, _, _, q = expand_free_theta_logcutoff(th, ctx_d20)
    return q
end

mutable struct ChainPoint
    idx::Int
    phase::Symbol
    g::Float64
    GT::Float64
    classification::Symbol
    Delta::Float64
    within_budget::Bool
    lfd_ok::Bool
    lfd_Delta::Float64
    A_free::Vector{Float64}
    theta_free::Vector{Float64}
    f_full::Matrix{Float64}
    q_full::Matrix{Float64}
    lfd_weights::Vector{Float64}
    nStatus::Int
    unique_inner_solves::Int
    wall_s::Float64
    timestamp::String
end
mutable struct ChainState
    chain_label::String
    direction::Symbol
    points::Vector{ChainPoint}
    bracket::Union{Nothing,NTuple{4,Float64}}
    n_evaluated::Int
    most_extreme_idx::Int
    n_nonimproving::Int
    elapsed_wall_s::Float64
    status::Symbol
    cur_A_free::Vector{Float64}
    cur_q::Matrix{Float64}
    cur_p_star::Vector{Float64}
    cur_g::Float64
    cur_GT::Float64
    step_gt::Float64
    n_accepted::Int
end

function save_checkpoint(state::ChainState)
    tmp = CKPT_PATH * ".tmp"
    serialize(tmp, state)
    mv(tmp, CKPT_PATH; force=true)
end
function append_point_csv(pt::ChainPoint)
    is_new = !isfile(POINTS_CSV)
    open(POINTS_CSV, "a") do io
        is_new && println(io, "idx,phase,g,GT,classification,Delta,within_budget,lfd_ok,lfd_Delta,nStatus,unique_inner_solves,wall_s,timestamp")
        println(io, join([pt.idx, pt.phase, pt.g, pt.GT, pt.classification, pt.Delta, pt.within_budget,
            pt.lfd_ok, pt.lfd_Delta, pt.nStatus, pt.unique_inner_solves, pt.wall_s, pt.timestamp], ","))
    end
end

function evaluate_point!(state::ChainState, GT_target::Float64, phase::Symbol)
    g_target = g_of_gt(GT_target)
    q_target = q_full_at_g_anchor(g_target)
    gpj_target = exp(g_target)

    theta_for_constraints = melitz_fixed_q_state_theta(state.cur_A_free, q_target, gpj_target, ctx_d20)
    sys_t = melitz_fixed_q_middle_constraint_system(theta_for_constraints, ctx_d20, obj_d20)

    A_cont = melitz_project_start_to_middle_constraints(copy(state.cur_A_free), sys_t, ctx_d20)
    prev_A_full = exp.(reshape(pivot_expand(state.cur_A_free, ctx_d20.A_pivot), D20, D20))
    A_cellwise, status_cellwise = melitz_cellwise_A_from_moments(prev_A_full, state.cur_q, q_target,
        state.cur_p_star, sorted_ctx_d20, sigma_d20)
    n_bad = count(!=(:ok), status_cellwise)
    if n_bad > 0
        A_cellwise[status_cellwise .!= :ok] .= prev_A_full[status_cellwise .!= :ok]
    end
    A_cellwise_free = pivot_reduce(vec(log.(A_cellwise)), ctx_d20.A_pivot)
    A_comp = melitz_project_start_to_middle_constraints(A_cellwise_free, sys_t, ctx_d20)

    idx = state.n_evaluated + 1
    t0 = time()
    local adapt, ok
    ok = true
    try
        session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
        adapt = melitz_middle_two_start_adaptive!(session_d20, q_target, gpj_target, A_cont, A_comp, ctx_d20;
            delta_budget=DELTA_BUDGET, policy=MelitzAdaptiveStartPolicy(), periodic_safeguard_due=(idx % PERIODIC_K == 0),
            sys=sys_t, prev_sys=nothing, coordinate=:logA, max_evals=MAX_MIDDLE_EVALS,
            box=D20_MIDDLE_BOX, outer_loop_opt=MIDDLE_OPT_D20, cap_handling=CAP_HANDLING,
            cap_barrier_multiple=CAP_BARRIER_MULTIPLE)
    catch e
        ok = false
        @warn "[$CHAINID] evaluate_point! threw at GT_target=$GT_target" exception=(e, catch_backtrace())
    end
    wall = time() - t0

    if !ok
        return ChainPoint(idx, phase, g_target, GT_target, :error, NaN, false, false, NaN,
            Float64[], Float64[], zeros(0,0), zeros(0,0), Float64[], -999, 0, wall, string(Dates.now()))
    end

    finite = adapt.r_incumbent isa FiniteSolved
    within_budget = finite && adapt.Delta <= DELTA_BUDGET
    cls = finite ? :FiniteSolved :
          adapt.r_incumbent isa AboveEvaluationCap ? :AboveEvaluationCap :
          adapt.r_incumbent isa InfiniteDeltaCertified ? :InfiniteDeltaCertified : :NumericalFailure
    n_solves = adapt.r_continuation.unique_inner_solves + (adapt.ran_compensated ? adapt.r_compensated.unique_inner_solves : 0)

    lfd_ok = false; lfd_Delta = NaN; lfd_weights = Float64[]
    _, f_full_t, _, _, q_full_t = expand_free_theta_logcutoff(adapt.theta_free, ctx_d20)
    if finite
        session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
        lfd_r = melitz_recover_lfd(obj_d20, adapt.theta_free)
        lfd_ok = lfd_r.lfd_ok
        lfd_Delta = lfd_r.Delta
        lfd_weights = copy(lfd_r.weights)
    end

    return ChainPoint(idx, phase, g_target, gt_of_g(g_target), cls, adapt.Delta, within_budget, lfd_ok, lfd_Delta,
        copy(adapt.A_free), copy(adapt.theta_free), f_full_t, q_full_t, lfd_weights,
        finite ? adapt.r_incumbent.nStatus : -999, n_solves, wall, string(Dates.now()))
end

function update_most_extreme!(state::ChainState, pt::ChainPoint)
    if pt.classification == :FiniteSolved && pt.within_budget && pt.lfd_ok
        cur_best = state.points[state.most_extreme_idx]
        if SIGN * (pt.GT - cur_best.GT) > 0
            state.most_extreme_idx = length(state.points)
            return true
        end
    end
    return false
end

function record_point!(state::ChainState, pt::ChainPoint)
    push!(state.points, pt)
    append_point_csv(pt)
    state.n_evaluated += 1
    if pt.classification == :FiniteSolved && pt.within_budget && pt.lfd_ok
        state.n_accepted += 1
        state.cur_A_free = copy(pt.A_free)
        state.cur_q = copy(pt.q_full)
        state.cur_p_star = copy(pt.lfd_weights)
        state.cur_g = pt.g
        state.cur_GT = pt.GT
    end
    improved = update_most_extreme!(state, pt)
    if improved
        state.n_nonimproving = 0
    else
        state.n_nonimproving += 1
    end
    save_checkpoint(state)
    return improved
end

function propose_next(state::ChainState)
    if state.bracket === nothing
        return state.cur_GT + SIGN * state.step_gt, :expand
    end
    GT_feas, Delta_feas, GT_infeas, Delta_infeas = state.bracket
    width = abs(GT_infeas - GT_feas)
    if width <= GT_BRACKET_STOP_TOL
        return nothing, :done
    end
    t = 0.5
    if isfinite(Delta_feas) && isfinite(Delta_infeas) && Delta_infeas != Delta_feas
        t_raw = (DELTA_BUDGET - Delta_feas) / (Delta_infeas - Delta_feas)
        if isfinite(t_raw) && 0.0 <= t_raw <= 1.0
            t = clamp(t_raw, INTERP_CLAMP[1], INTERP_CLAMP[2])
        end
    end
    return GT_feas + t * (GT_infeas - GT_feas), :polish
end

# ============================================================================
# Initialize state from the retained Phase3/4 point (fresh start -- no prior checkpoint expected).
# ============================================================================
t_loop_start = time()
if isfile(CKPT_PATH)
    state = deserialize(CKPT_PATH)
    println("[$CHAINID] RESUMING from checkpoint: $(length(state.points)) points, status=$(state.status)")
else
    obj_d20.use_cached_x = false; obj_d20.x .= NaN
    lfd0 = melitz_recover_lfd(obj_d20, start_pt.theta_free)
    @assert lfd0.lfd_ok && isapprox(lfd0.Delta, start_pt.Delta; atol=1e-6)
    _, f_full0, _, _, q_full0 = expand_free_theta_logcutoff(start_pt.theta_free, ctx_d20)
    pt0 = ChainPoint(1, :anchor, start_pt.g, start_pt.GT_target, :FiniteSolved, start_pt.Delta, true, true,
        lfd0.Delta, copy(start_pt.A_free), copy(start_pt.theta_free), f_full0, q_full0, copy(lfd0.weights),
        0, 0, 0.0, string(Dates.now()))
    state = ChainState(CHAINID, DIRECTION, ChainPoint[], nothing, 1, 1, 0, 0.0, :running,
        copy(pt0.A_free), copy(pt0.q_full), copy(pt0.lfd_weights), pt0.g, pt0.GT, GT_STEP0, 0)
    push!(state.points, pt0)
    append_point_csv(pt0)
    save_checkpoint(state)
    println("[$CHAINID] FRESH START from verified point GT=$(pt0.GT) Delta=$(pt0.Delta)")
end
flush(stdout)

while state.status == :running
    elapsed_total = state.elapsed_wall_s + (time() - t_loop_start)
    if elapsed_total > MAX_WALL_S
        state.status = :max_wall; break
    end
    if state.n_evaluated >= MAX_WELFARE_POINTS
        state.status = :max_points; break
    end
    if state.n_nonimproving >= MAX_NONIMPROVING
        state.status = :stalled; break
    end

    GT_target, phase = propose_next(state)
    if phase == :done
        GT_feas, Delta_feas, GT_infeas, Delta_infeas = state.bracket
        gap = DELTA_BUDGET - Delta_feas
        state.status = (isfinite(gap) && 0.0 <= gap <= DELTA_STOP_TOL) ? :converged :
                        (abs(GT_infeas - GT_feas) <= GT_BRACKET_STOP_TOL) ? :converged : :bracket_narrow
        break
    end

    @printf("\n[%s] [%s] attempt #%d  GT_target=%.6f%%\n", CHAINID, phase, state.n_evaluated + 1, GT_target)
    flush(stdout)
    pt = evaluate_point!(state, GT_target, phase)
    improved = record_point!(state, pt)
    @printf("[%s] [%s] result: classification=%s Delta=%s within_budget=%s lfd_ok=%s wall=%.1fs improved=%s\n",
        CHAINID, phase, pt.classification, string(pt.Delta), pt.within_budget, pt.lfd_ok, pt.wall_s, improved)
    flush(stdout)

    is_verified_feasible = pt.classification == :FiniteSolved && pt.within_budget && pt.lfd_ok
    if phase == :expand
        if is_verified_feasible
            state.step_gt = min(state.step_gt * GT_STEP_GROWTH, GT_STEP_MAX)
        else
            prev_feas = state.points[end-1]
            state.bracket = (prev_feas.GT, prev_feas.Delta, pt.GT, pt.Delta)
            println("[$CHAINID] BRACKET FOUND: feasible GT=$(prev_feas.GT) | infeasible/unverified GT=$(pt.GT)")
        end
    else
        GT_feas, Delta_feas, GT_infeas, Delta_infeas = state.bracket
        if is_verified_feasible
            state.bracket = (pt.GT, pt.Delta, GT_infeas, Delta_infeas)
        else
            state.bracket = (GT_feas, Delta_feas, pt.GT, pt.Delta)
        end
    end
end
state.elapsed_wall_s += (time() - t_loop_start)
save_checkpoint(state)

best = state.points[state.most_extreme_idx]
println("\n", "="^100)
@printf("[%s] PHASE5 COMPLETE: status=%s  n_evaluated=%d  elapsed_wall_s=%.1f\n", CHAINID, state.status, state.n_evaluated, state.elapsed_wall_s)
@printf("[%s] BEST VERIFIED INCUMBENT: GT=%.6f%%  Delta*=%.6g  classification=%s\n", CHAINID, best.GT, best.Delta, best.classification)
println("="^100)
println("\nDONE PHASE5 $CHAINID")
