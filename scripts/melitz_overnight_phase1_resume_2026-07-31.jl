# Overnight continuation, Phase 1: resume the six production_delta0p5_2026-07-31 chains,
# remove the old hard 3-boundary-polish cap, and apply the new stop rules / proposal hierarchy
# from docs/melitz_d20_profiledA_overnight_qpoll_delta0p5_2026-07-31 governing prompt.
#
# ARGS = [anchor_label, direction]. Requires an existing checkpoint from the completed
# production_delta0p5_2026-07-31 campaign (scripts/melitz_production_chain_2026-07-31.jl) --
# refuses to fresh-start. Does NOT discard any previously evaluated point: the checkpoint's
# `points` vector is only ever appended to.
#
# Reuses (does not modify): solve_melitz_fixed_q_A_profile_v2 / melitz_middle_two_start_adaptive!
# (src/melitz/fixed_q_a_middle_loop.jl), the reduced-q anchor construction, the exact same
# fixture/fingerprint/evaluate_point! machinery as the original production chain driver. Does not
# modify the Ricardian implementation or place the free-q vector into KNITRO.
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Dates, Serialization
melitz_thread_startup_report()
LinearAlgebra.BLAS.set_num_threads(1)
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

length(ARGS) >= 2 || error("usage: julia melitz_overnight_phase1_resume_2026-07-31.jl <anchor_label> <direction>")
const ANCHOR_LABEL = ARGS[1]
const DIRECTION = Symbol(ARGS[2])
ANCHOR_LABEL in ("current_calibration", "reduced_q_pre_switch", "reduced_q_post_switch") ||
    error("unknown anchor_label=$ANCHOR_LABEL")
DIRECTION in (:upper, :lower) || error("direction must be upper or lower, got $DIRECTION")
const CHAINID = "$(ANCHOR_LABEL)_$(DIRECTION)"
const SIGN = DIRECTION == :upper ? 1 : -1

const OUTDIR = joinpath(REPO2, "docs", "key_results")
const PRODDIR = joinpath(OUTDIR, "production_delta0p5_2026-07-31")
const CKPT_PATH = joinpath(PRODDIR, "checkpoints", "$(CHAINID).jls")
const POINTS_CSV = joinpath(PRODDIR, "points", "$(CHAINID)_points.csv")

const OVERNIGHT_DIR = joinpath(OUTDIR, "overnight_qpoll_delta0p5_2026-07-31")
const PH1_CKPT_PATH = joinpath(OVERNIGHT_DIR, "phase1_checkpoints", "$(CHAINID).jls")
mkpath(dirname(PH1_CKPT_PATH))

isfile(CKPT_PATH) || error("[$CHAINID] no existing production_delta0p5_2026-07-31 checkpoint at $CKPT_PATH -- refusing to fresh-start")

CAP = 10.0
policy_cap = CappedEvaluation(CAP)
const MIDDLE_OPT_D20 = joinpath(REPO2, "melitz_middle_loop_opt_2026-07-30.opt")
const D20_MIDDLE_BOX = 0.1
const MAX_MIDDLE_EVALS = 120
const CAP_HANDLING = :barrier
const CAP_BARRIER_MULTIPLE = 5.0

const DELTA_BUDGET = 0.5
const INTERP_CLAMP = (0.2, 0.8)
const MAX_WALL_S = parse(Float64, get(ENV, "MELITZ_MAX_WALL_S", "21600"))   # 6h, fresh clock this process

# New Phase-1 stop/proposal parameters (governing prompt).
const GT_BRACKET_STOP_TOL_NEW = 0.005     # pp, both directions
const DELTA_SLACK_STOP_TOL = 0.001        # upper only: 0 <= 0.5-Delta* <= 0.001
const MAX_ADDITIONAL_EVALS_UPPER = 20
const MAX_CONSEC_NONIMPROVE_UPPER = 4
const MAX_ADDITIONAL_EVALS_LOWER = 16
const PERIODIC_K = 4
const EXPLORE_STEP_PP = 0.25

# ============================================================================
# Real-D20 fixture (identical recipe to melitz_production_chain_2026-07-31.jl).
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
println("[$CHAINID] focal country index = ", focal); flush(stdout)

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
A_free0_d20 = theta_plain0_d20[2:1+nA20]
q_free0_d20 = theta_plain0_d20[2+nA20:end]
g0 = theta_plain0_d20[1]
GT0 = gt_of_g(g0)

obj_d20.use_cached_x = false; obj_d20.x .= NaN
lfd0 = melitz_recover_lfd(obj_d20, theta0_d20)
@assert lfd0.lfd_ok && isapprox(lfd0.Delta, 0.483276; atol=1e-4)
x0_d20 = copy(lfd0.dual_x)
p_star0_d20 = copy(lfd0.weights)

bwpolicy = PowerScaledQBandwidth(1e-3, 80_000, 0.5)
stage_d20 = melitz_build_reduced_q_stage(theta0_d20, x0_d20, ctx_d20, obj_d20, 1;
    bandwidth_policy=bwpolicy, target_switches=100)
b_q_d20 = stage_d20.q_basis_free

const ANCHOR_Q_FREE = Dict(
    "current_calibration" => copy(q_free0_d20),
    "reduced_q_pre_switch" => q_free0_d20 .+ (-1.0) .* 6.31e-3 .* b_q_d20,
    "reduced_q_post_switch" => q_free0_d20 .+ (-1.0) .* 1.26e-2 .* b_q_d20,
)
q_free_fixed = ANCHOR_Q_FREE[ANCHOR_LABEL]

function q_full_at_g_anchor(g_target::Real)
    th = copy(theta_plain0_d20)
    th[1] = g_target
    th[2+nA20:end] .= q_free_fixed
    _, _, _, _, q = expand_free_theta_logcutoff(th, ctx_d20)
    return q
end

GIT_COMMIT = try
    strip(read(`git -C $REPO2 rev-parse HEAD`, String))
catch
    "unknown"
end
FINGERPRINT = (D=D20, W=80_000, seed=1, sigma=sigma_d20, focal=focal, anchor_label=ANCHOR_LABEL,
    direction=String(DIRECTION), q_free_hash=hash(round.(q_free_fixed; digits=10)),
    theta0_hash=hash(round.(theta0_d20; digits=10)), git_commit=GIT_COMMIT)
println("[$CHAINID] fingerprint = ", FINGERPRINT); flush(stdout)

# ============================================================================
# Checkpoint types -- MUST match melitz_production_chain_2026-07-31.jl exactly (same field
# layout) so the existing production checkpoint deserializes.
# ============================================================================
mutable struct ChainPoint
    idx::Int
    phase::Symbol
    g::Float64
    GT::Float64
    classification::Symbol
    Delta::Float64
    within_budget::Bool
    best_start::Symbol
    trigger_reason::Symbol
    A_free::Vector{Float64}
    theta_free::Vector{Float64}
    f_full::Matrix{Float64}
    q_full::Matrix{Float64}
    dual_x::Vector{Float64}
    lfd_weights::Vector{Float64}
    lfd_ok::Bool
    lfd_Delta::Float64
    nStatus::Int
    unique_A_points::Int
    unique_inner_solves::Int
    n_fc_calls::Int
    n_ga_calls::Int
    wall_s::Float64
    timestamp::String
end

mutable struct ChainState
    anchor_label::String
    direction::Symbol
    fingerprint::NamedTuple
    points::Vector{ChainPoint}
    phase::Symbol
    step_gt::Float64
    bracket::Union{Nothing,NTuple{4,Float64}}
    n_evaluated::Int
    n_polish::Int
    n_accepted::Int
    most_extreme_idx::Int
    n_nonimproving::Int
    elapsed_wall_s::Float64
    status::Symbol
    cur_A_free::Vector{Float64}
    cur_q::Matrix{Float64}
    cur_p_star::Vector{Float64}
    cur_g::Float64
    cur_GT::Float64
    prev_sys::Any
end

function save_checkpoint(state::ChainState)
    tmp = CKPT_PATH * ".tmp"
    serialize(tmp, state)
    mv(tmp, CKPT_PATH; force=true)
    tmp2 = PH1_CKPT_PATH * ".tmp"
    serialize(tmp2, state)
    mv(tmp2, PH1_CKPT_PATH; force=true)
end

function append_point_csv(pt::ChainPoint)
    is_new = !isfile(POINTS_CSV)
    open(POINTS_CSV, "a") do io
        is_new && println(io, "idx,phase,g,GT,classification,Delta,within_budget,best_start,trigger_reason,lfd_ok,lfd_Delta,nStatus,unique_A_points,unique_inner_solves,n_fc_calls,n_ga_calls,wall_s,timestamp")
        println(io, join([pt.idx, pt.phase, pt.g, pt.GT, pt.classification, pt.Delta, pt.within_budget,
            pt.best_start, pt.trigger_reason, pt.lfd_ok, pt.lfd_Delta, pt.nStatus, pt.unique_A_points,
            pt.unique_inner_solves, pt.n_fc_calls, pt.n_ga_calls, pt.wall_s, pt.timestamp], ","))
    end
end

# ============================================================================
# One classified welfare-point evaluation (identical machinery to the production driver).
# ============================================================================
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

    periodic_due = state.n_accepted > 0 && state.n_accepted % 4 == 0

    t0 = time()
    local adapt, ok
    ok = true
    try
        session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
        adapt = melitz_middle_two_start_adaptive!(session_d20, q_target, gpj_target, A_cont, A_comp, ctx_d20;
            delta_budget=DELTA_BUDGET, policy=MelitzAdaptiveStartPolicy(), periodic_safeguard_due=periodic_due,
            sys=sys_t, prev_sys=state.prev_sys, coordinate=:logA, max_evals=MAX_MIDDLE_EVALS,
            box=D20_MIDDLE_BOX, outer_loop_opt=MIDDLE_OPT_D20, cap_handling=CAP_HANDLING,
            cap_barrier_multiple=CAP_BARRIER_MULTIPLE)
    catch e
        ok = false
        @warn "[$CHAINID] evaluate_point! threw at GT_target=$GT_target (ordinary exception, not a crash)" exception=(e, catch_backtrace())
    end
    wall = time() - t0

    idx = state.n_evaluated + 1
    if !ok
        pt = ChainPoint(idx, phase, g_target, GT_target, :error, NaN, false, :none, :exception,
            Float64[], Float64[], zeros(0,0), zeros(0,0), Float64[], Float64[], false, NaN, -999,
            0, 0, 0, 0, wall, string(Dates.now()))
        return pt, sys_t
    end

    finite = adapt.r_incumbent isa FiniteSolved
    within_budget = finite && adapt.Delta <= DELTA_BUDGET
    cls = adapt.r_incumbent isa FiniteSolved ? :FiniteSolved :
          adapt.r_incumbent isa AboveEvaluationCap ? :AboveEvaluationCap :
          adapt.r_incumbent isa InfiniteDeltaCertified ? :InfiniteDeltaCertified : :NumericalFailure
    n_unique = adapt.r_continuation.unique_A_points + (adapt.ran_compensated ? adapt.r_compensated.unique_A_points : 0)
    n_solves = adapt.r_continuation.unique_inner_solves + (adapt.ran_compensated ? adapt.r_compensated.unique_inner_solves : 0)
    n_fc = adapt.r_continuation.n_fc_calls + (adapt.ran_compensated ? adapt.r_compensated.n_fc_calls : 0)
    n_ga = adapt.r_continuation.n_ga_calls + (adapt.ran_compensated ? adapt.r_compensated.n_ga_calls : 0)

    dual_x = finite ? copy(adapt.r_incumbent.x) : Float64[]
    lfd_ok = false; lfd_Delta = NaN; lfd_weights = Float64[]
    _, f_full_t, _, _, q_full_t = expand_free_theta_logcutoff(adapt.theta_free, ctx_d20)
    if finite
        session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
        lfd_r = melitz_recover_lfd(obj_d20, adapt.theta_free)
        lfd_ok = lfd_r.lfd_ok
        lfd_Delta = lfd_r.Delta
        lfd_weights = copy(lfd_r.weights)
    end

    pt = ChainPoint(idx, phase, g_target, gt_of_g(g_target), cls, adapt.Delta, within_budget, adapt.best_source,
        adapt.trigger_reason, copy(adapt.A_free), copy(adapt.theta_free), f_full_t, q_full_t, dual_x,
        lfd_weights, lfd_ok, lfd_Delta, adapt.r_incumbent isa FiniteSolved ? adapt.r_incumbent.nStatus : -999,
        n_unique, n_solves, n_fc, n_ga, wall, string(Dates.now()))
    return pt, sys_t
end

# NOTE (caught live, 2026-07-31): a point can be primal :FiniteSolved && within_budget while its
# LFD/dual recovery fails (lfd_ok=false) -- confirmed a genuine near-cliff dual degeneracy, not a
# caching artifact (reproduces on an independent fresh-process re-check). The governing prompt
# requires "fully verify the inner LFD" at every candidate, so lfd_ok is part of the incumbent
# gate, not just classification+within_budget.
function update_most_extreme!(state::ChainState, pt::ChainPoint)
    if pt.classification == :FiniteSolved && pt.within_budget && pt.lfd_ok
        cur_best = state.points[state.most_extreme_idx]
        if SIGN * (pt.GT - GT0) > SIGN * (cur_best.GT - GT0)
            state.most_extreme_idx = length(state.points)
            return true
        end
    end
    return false
end

function accept_point!(state::ChainState, pt::ChainPoint)
    if pt.classification == :FiniteSolved && pt.within_budget && pt.lfd_ok
        state.n_accepted += 1
        state.cur_A_free = copy(pt.A_free)
        state.cur_q = copy(pt.q_full)
        state.cur_p_star = pt.lfd_ok ? copy(pt.lfd_weights) : state.cur_p_star
        state.cur_g = pt.g
        state.cur_GT = pt.GT
    end
end

function record_point!(state::ChainState, pt::ChainPoint, sys_t)
    push!(state.points, pt)
    append_point_csv(pt)
    state.n_evaluated += 1
    state.prev_sys = sys_t
    accept_point!(state, pt)
    improved = update_most_extreme!(state, pt)
    save_checkpoint(state)
    return improved
end

function find_point_by_GT(state::ChainState, GT::Float64)
    for pt in Iterators.reverse(state.points)
        pt.GT == GT && return pt
    end
    return nothing
end

mutable struct BracketInfo
    GT_feas::Float64
    Delta_feas::Float64
    cls_feas::Symbol
    GT_infeas::Float64
    Delta_infeas::Float64
    cls_infeas::Symbol
end

function recent_profile_monotone(state::ChainState; k::Int=3)
    recent = ChainPoint[]
    for pt in Iterators.reverse(state.points)
        pt.phase in (:polish, :expand, :subdivide) || continue
        pt.classification == :FiniteSolved || continue
        push!(recent, pt)
        length(recent) >= k && break
    end
    length(recent) < 2 && return true
    reverse!(recent)
    for i in 2:length(recent)
        d_GT = recent[i].GT - recent[i-1].GT
        d_Delta = recent[i].Delta - recent[i-1].Delta
        if d_GT * SIGN > 0 && d_Delta * SIGN < 0
            return false
        end
    end
    return true
end

# ============================================================================
# Load state, reconstruct BracketInfo from the historical checkpoint.
# ============================================================================
state = deserialize(CKPT_PATH)
println("[$CHAINID] LOADED production checkpoint: $(length(state.points)) historical points, status=$(state.status)")
fp = state.fingerprint
@assert fp.D == FINGERPRINT.D && fp.W == FINGERPRINT.W && fp.seed == FINGERPRINT.seed &&
        isapprox(fp.sigma, FINGERPRINT.sigma) && fp.focal == FINGERPRINT.focal &&
        fp.anchor_label == FINGERPRINT.anchor_label && fp.direction == FINGERPRINT.direction &&
        fp.q_free_hash == FINGERPRINT.q_free_hash && fp.theta0_hash == FINGERPRINT.theta0_hash (
    "[$CHAINID] FINGERPRINT MISMATCH on resume -- refusing to silently continue. stored=$fp current=$FINGERPRINT")

state.bracket !== nothing || error("[$CHAINID] checkpoint has no established bracket -- cannot apply Phase-1 subdivision/polish logic")
GT_feas0, Delta_feas0, GT_infeas0, Delta_infeas0 = state.bracket
pt_feas0 = find_point_by_GT(state, GT_feas0)
pt_infeas0 = find_point_by_GT(state, GT_infeas0)
cls_feas0 = pt_feas0 === nothing ? :FiniteSolved : pt_feas0.classification
cls_infeas0 = pt_infeas0 === nothing ? :FiniteSolved : pt_infeas0.classification
bracket = BracketInfo(GT_feas0, Delta_feas0, cls_feas0, GT_infeas0, Delta_infeas0, cls_infeas0)
const GT_FIRST_CLIFF = GT_infeas0   # immutable reference point for lower-direction "below the first apparent cliff" exploration
println("[$CHAINID] initial bracket: feasible GT=$(bracket.GT_feas) Delta=$(bracket.Delta_feas) cls=$(bracket.cls_feas) | ",
        "infeasible GT=$(bracket.GT_infeas) Delta=$(bracket.Delta_infeas) cls=$(bracket.cls_infeas)")
flush(stdout)

# cold-reverify the stored incumbent before resuming
best_pt = state.points[state.most_extreme_idx]
session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
r_cold = solve_melitz_delta!(session_d20, best_pt.theta_free, policy_cap; warm_start_source=:neutral)
@assert r_cold isa FiniteSolved "[$CHAINID] stored incumbent failed cold reverification: $(typeof(r_cold))"
@assert isapprox(r_cold.Delta, best_pt.Delta; atol=1e-6) "[$CHAINID] stored incumbent Delta drifted on cold reverification: stored=$(best_pt.Delta) live=$(r_cold.Delta)"
println("[$CHAINID] cold-reverified incumbent OK: GT=$(best_pt.GT) Delta=$(r_cold.Delta)")
flush(stdout)

state.status = :running

# ============================================================================
# Proposal functions (new Phase-1 rules).
# ============================================================================
function propose_upper(state::ChainState, bracket::BracketInfo, n_additional::Int)
    if (n_additional + 1) % PERIODIC_K == 0
        return bracket.GT_infeas + EXPLORE_STEP_PP, :explore
    end
    finite_both = bracket.cls_feas == :FiniteSolved && bracket.cls_infeas == :FiniteSolved
    mono = recent_profile_monotone(state)
    if finite_both && mono && bracket.Delta_infeas != bracket.Delta_feas
        t_raw = (DELTA_BUDGET - bracket.Delta_feas) / (bracket.Delta_infeas - bracket.Delta_feas)
        t = (isfinite(t_raw) && 0.0 <= t_raw <= 1.0) ? clamp(t_raw, INTERP_CLAMP[1], INTERP_CLAMP[2]) : 0.5
        return bracket.GT_feas + t * (bracket.GT_infeas - bracket.GT_feas), :polish
    else
        return (bracket.GT_feas + bracket.GT_infeas) / 2, :polish
    end
end

function propose_lower(bracket::BracketInfo, n_subdiv_since_explore::Int)
    if n_subdiv_since_explore >= PERIODIC_K
        return GT_FIRST_CLIFF - EXPLORE_STEP_PP, :explore
    end
    return (bracket.GT_feas + bracket.GT_infeas) / 2, :subdivide
end

function update_bracket_upper!(bracket::BracketInfo, pt::ChainPoint)
    if pt.classification == :FiniteSolved && pt.within_budget
        if SIGN * (pt.GT - bracket.GT_feas) > 0
            bracket.GT_feas = pt.GT; bracket.Delta_feas = pt.Delta; bracket.cls_feas = pt.classification
        end
    else
        if SIGN * (pt.GT - bracket.GT_infeas) < 0
            bracket.GT_infeas = pt.GT; bracket.Delta_infeas = pt.Delta; bracket.cls_infeas = pt.classification
        end
    end
end

function update_bracket_lower_subdivide!(bracket::BracketInfo, pt::ChainPoint)
    if pt.classification == :FiniteSolved && pt.within_budget && pt.lfd_ok
        bracket.GT_feas = pt.GT; bracket.Delta_feas = pt.Delta; bracket.cls_feas = pt.classification
    else
        # A point that is primal FiniteSolved && within_budget but lfd_ok=false is NOT usable
        # feasible ground (dual/LFD recovery is unverified there -- confirmed a genuine near-cliff
        # dual degeneracy, not a caching artifact). Conservatively treat it like the infeasible
        # edge: don't stand on unverifiable ground, and don't keep re-exploring the same
        # degenerate neighborhood via naive bisection.
        bracket.GT_infeas = pt.GT; bracket.Delta_infeas = pt.Delta
        bracket.cls_infeas = (pt.classification == :FiniteSolved && pt.within_budget) ? :LfdUnverified : pt.classification
    end
end

# ============================================================================
# Main loop (wrapped in a function so Julia's ordinary local scoping applies --
# top-level script scoping requires explicit `global` on loop-body reassignment, which is easy
# to miss; see docs/../CLAUDE.md-adjacent memory on the top-level catch-scoping gotcha).
# ============================================================================
function run_phase1!(state::ChainState, bracket::BracketInfo)
t_loop_start = time()
n_additional = 0
n_consec_nonimprove = 0
n_subdiv_since_explore = 0
disconnected_hits = ChainPoint[]

stop_reason = :none
while true
    elapsed_this_run = time() - t_loop_start
    if elapsed_this_run > MAX_WALL_S
        stop_reason = :max_wall_additional; break
    end

    if DIRECTION == :upper
        slack = DELTA_BUDGET - bracket.Delta_feas
        if bracket.cls_feas == :FiniteSolved && isfinite(slack) && 0.0 <= slack <= DELTA_SLACK_STOP_TOL
            stop_reason = :budget_slack_converged; break
        end
        if abs(bracket.GT_infeas - bracket.GT_feas) <= GT_BRACKET_STOP_TOL_NEW
            stop_reason = :bracket_width_converged; break
        end
        if n_additional >= MAX_ADDITIONAL_EVALS_UPPER
            stop_reason = :max_additional_evals; break
        end
        if n_consec_nonimprove >= MAX_CONSEC_NONIMPROVE_UPPER
            stop_reason = :max_consec_nonimprove; break
        end
        GT_target, tag = propose_upper(state, bracket, n_additional)
    else
        if abs(bracket.GT_infeas - bracket.GT_feas) <= GT_BRACKET_STOP_TOL_NEW
            stop_reason = :bracket_width_converged; break
        end
        if n_additional >= MAX_ADDITIONAL_EVALS_LOWER
            stop_reason = :max_additional_evals; break
        end
        GT_target, tag = propose_lower(bracket, n_subdiv_since_explore)
    end

    n_additional += 1
    @printf("\n[%s] [phase1 #%d/%s] GT_target=%.6f%%  tag=%s\n", CHAINID, n_additional,
        DIRECTION == :upper ? MAX_ADDITIONAL_EVALS_UPPER : MAX_ADDITIONAL_EVALS_LOWER, GT_target, tag)
    flush(stdout)

    pt, sys_t = evaluate_point!(state, GT_target, tag)
    improved = record_point!(state, pt, sys_t)

    @printf("[%s] result: classification=%s Delta=%s within_budget=%s wall=%.1fs improved_extreme=%s\n",
        CHAINID, pt.classification, string(pt.Delta), pt.within_budget, pt.wall_s, improved)
    flush(stdout)

    if tag == :explore
        n_subdiv_since_explore = 0
        if pt.classification == :FiniteSolved && pt.within_budget && SIGN * (pt.GT - bracket.GT_feas) > 0 &&
           SIGN * (pt.GT - GT_FIRST_CLIFF) > 0
            push!(disconnected_hits, pt)
            println("[$CHAINID] DISCONNECTED FEASIBLE REGION HIT beyond the known cliff at GT=$(pt.GT) (kept as separate incumbent candidate; primary bracket left unperturbed)")
            flush(stdout)
            # most_extreme_idx already updated by update_most_extreme! inside record_point! since
            # this is unconditionally the most extreme feasible point found so far in that case.
        end
    elseif DIRECTION == :upper
        update_bracket_upper!(bracket, pt)
    else
        update_bracket_lower_subdivide!(bracket, pt)
        n_subdiv_since_explore += 1
    end

    # Persist the working BracketInfo back into the checkpointed state.bracket field so a
    # subsequent resume of THIS script continues from the narrowed bracket instead of
    # re-deriving the stale pre-Phase1 bracket (caught live, 2026-07-31 -- state.bracket was
    # never written back before this fix).
    state.bracket = (bracket.GT_feas, bracket.Delta_feas, bracket.GT_infeas, bracket.Delta_infeas)
    save_checkpoint(state)

    if improved
        n_consec_nonimprove = 0
    else
        n_consec_nonimprove += 1
    end
end

state.elapsed_wall_s += (time() - t_loop_start)
state.status = stop_reason
save_checkpoint(state)

best = state.points[state.most_extreme_idx]
println("\n", "="^100)
@printf("[%s] PHASE1 RESUME COMPLETE: stop_reason=%s  n_additional=%d  n_total_evaluated=%d  elapsed_wall_s=%.1f\n",
    CHAINID, stop_reason, n_additional, state.n_evaluated, state.elapsed_wall_s)
@printf("[%s] BEST VERIFIED INCUMBENT: GT=%.6f%%  Delta*=%.6g  classification=%s\n",
    CHAINID, best.GT, best.Delta, best.classification)
@printf("[%s] final bracket: feasible GT=%.6f Delta=%.6g cls=%s | infeasible GT=%.6f Delta=%.6g cls=%s\n",
    CHAINID, bracket.GT_feas, bracket.Delta_feas, bracket.cls_feas, bracket.GT_infeas, bracket.Delta_infeas, bracket.cls_infeas)
if !isempty(disconnected_hits)
    println("[$CHAINID] DISCONNECTED-REGION HITS: ", length(disconnected_hits))
    for h in disconnected_hits
        @printf("  GT=%.6f%% Delta=%.6g\n", h.GT, h.Delta)
    end
end
println("="^100)
println("\nDONE $CHAINID PHASE1")
return stop_reason
end

run_phase1!(state, bracket)
