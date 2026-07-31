# D20 profiled-A production campaign, delta=0.5 (docs/melitz_d20_profiledA_production_delta0p5_2026-07-31.md).
# ONE anchor x direction chain, process-isolated. ARGS = [anchor_label, direction].
#
#   anchor_label in ("current_calibration", "reduced_q_pre_switch", "reduced_q_post_switch")
#   direction    in ("upper", "lower")
#
# Reuses (does not modify): solve_melitz_fixed_q_A_profile_v2 / melitz_middle_two_start_adaptive!
# (src/melitz/fixed_q_a_middle_loop.jl), the reduced-q anchor construction
# (scripts/melitz_phase7_cutoff_portfolio_2026-07-30.jl), the wide-expansion/bisection skeleton
# (scripts/melitz_d20_profiled_A_welfare_continuation_2026-07-30.jl). Does not modify the
# Ricardian implementation, build another cutoff-gradient method, or continuously optimize the
# free-cutoff coordinates (held FIXED at the anchor's own values throughout; only g and A/f vary).
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Dates, Serialization
melitz_thread_startup_report()
LinearAlgebra.BLAS.set_num_threads(1)
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

length(ARGS) >= 2 || error("usage: julia melitz_production_chain_2026-07-31.jl <anchor_label> <direction>")
const ANCHOR_LABEL = ARGS[1]
const DIRECTION = Symbol(ARGS[2])
ANCHOR_LABEL in ("current_calibration", "reduced_q_pre_switch", "reduced_q_post_switch") ||
    error("unknown anchor_label=$ANCHOR_LABEL")
DIRECTION in (:upper, :lower) || error("direction must be upper or lower, got $DIRECTION")
const CHAIN_SUFFIX = get(ENV, "MELITZ_CHAIN_SUFFIX", "")
const CHAINID = "$(ANCHOR_LABEL)_$(DIRECTION)$(CHAIN_SUFFIX)"
const SIGN = DIRECTION == :upper ? 1 : -1

const OUTDIR = joinpath(REPO2, "docs", "key_results")
const PRODDIR = joinpath(OUTDIR, "production_delta0p5_2026-07-31")
const CKPT_PATH = joinpath(PRODDIR, "checkpoints", "$(CHAINID).jls")
const PENDING_PATH = joinpath(PRODDIR, "checkpoints", "$(CHAINID).pending")
const POINTS_CSV = joinpath(PRODDIR, "points", "$(CHAINID)_points.csv")
mkpath(dirname(CKPT_PATH)); mkpath(dirname(POINTS_CSV))

CAP = 10.0
policy_cap = CappedEvaluation(CAP)
const MIDDLE_OPT_D20 = joinpath(REPO2, "melitz_middle_loop_opt_2026-07-30.opt")
const D20_MIDDLE_BOX = 0.1
const MAX_MIDDLE_EVALS = 120
const CAP_HANDLING = :barrier
const CAP_BARRIER_MULTIPLE = 5.0

const DELTA_BUDGET = 0.5
const GT_STEP0 = 0.5            # governing prompt Step 2: initial movement, GT percentage points
const GT_STEP_GROWTH = 1.5      # up to 1.5x on acceptance
const GT_STEP_MAX = 2.0         # maximum step
const GT_BRACKET_STOP_TOL = 0.01    # Step 4 stop rule: GT_hi - GT_lo <= 0.01pp
const DELTA_STOP_TOL = 0.002        # Step 4 stop rule: 0 <= delta - Delta* <= 0.002
const MAX_POLISH_EVALS = 3          # Step 4: at most 3 boundary-polish evaluations after bracketing
const INTERP_CLAMP = (0.2, 0.8)     # clamp interpolation proposal to interior 20-80% of bracket
const MAX_WELFARE_POINTS = parse(Int, get(ENV, "MELITZ_MAX_POINTS", "40"))
const MAX_WALL_S = parse(Float64, get(ENV, "MELITZ_MAX_WALL_S", "21600"))
const MAX_NONIMPROVING = 6
const PERIODIC_K = 4
const ADAPTIVE_POLICY = MelitzAdaptiveStartPolicy()   # governing-prompt defaults

# ============================================================================
# Real-D20 fixture (identical recipe to the prior sessions' own scripts).
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

# Reduced-q direction: EXACT same basis/thresholds as Phase 7/8 (negative-switch audit).
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
q0_anchor = q_full_at_g_anchor(g0)

# Deterministic fingerprint (validated on every resume).
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
# Checkpoint types.
# ============================================================================
mutable struct ChainPoint
    idx::Int
    phase::Symbol             # :anchor, :expand, :polish
    g::Float64
    GT::Float64
    classification::Symbol    # :FiniteSolved / :AboveEvaluationCap / :InfiniteDeltaCertified / :error
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
    phase::Symbol                          # :expand or :polish
    step_gt::Float64
    bracket::Union{Nothing,NTuple{4,Float64}}   # (GT_feas, Delta_feas, GT_infeas, Delta_infeas)
    n_evaluated::Int
    n_polish::Int
    n_accepted::Int
    most_extreme_idx::Int
    n_nonimproving::Int
    elapsed_wall_s::Float64                # accumulated across resumes
    status::Symbol                         # :running, :converged, :budget_exhausted, :max_points, :stalled, :max_wall
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
end

function write_pending(GT_target::Float64, phase::Symbol)
    open(PENDING_PATH, "w") do io
        println(io, GT_target, ",", phase, ",", Dates.now())
    end
end
clear_pending() = isfile(PENDING_PATH) && rm(PENDING_PATH)

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
# One classified welfare-point evaluation (adaptive two-start), with full checkpoint payload.
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

    periodic_due = state.n_accepted > 0 && state.n_accepted % PERIODIC_K == 0

    write_pending(GT_target, phase)   # crash-detection marker: what we're about to attempt

    t0 = time()
    local adapt, ok
    ok = true
    try
        session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
        adapt = melitz_middle_two_start_adaptive!(session_d20, q_target, gpj_target, A_cont, A_comp, ctx_d20;
            delta_budget=DELTA_BUDGET, policy=ADAPTIVE_POLICY, periodic_safeguard_due=periodic_due,
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
        clear_pending()
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
    clear_pending()
    return pt, sys_t
end

function update_most_extreme!(state::ChainState, pt::ChainPoint)
    if pt.classification == :FiniteSolved && pt.within_budget
        cur_best = state.points[state.most_extreme_idx]
        if SIGN * (pt.GT - GT0) > SIGN * (cur_best.GT - GT0)
            state.most_extreme_idx = length(state.points)
            state.n_nonimproving = 0
            return true
        end
    end
    state.n_nonimproving += 1
    return false
end

function propose_next(state::ChainState)
    if state.bracket === nothing
        GT_target = state.cur_GT + SIGN * state.step_gt
        return GT_target, :expand
    end
    GT_feas, Delta_feas, GT_infeas, Delta_infeas = state.bracket
    width = abs(GT_infeas - GT_feas)
    if width <= GT_BRACKET_STOP_TOL || state.n_polish >= MAX_POLISH_EVALS
        return nothing, :done
    end
    t = 0.5
    if isfinite(Delta_feas) && isfinite(Delta_infeas) && Delta_infeas != Delta_feas
        t_raw = (DELTA_BUDGET - Delta_feas) / (Delta_infeas - Delta_feas)
        if isfinite(t_raw) && 0.0 <= t_raw <= 1.0
            t = clamp(t_raw, INTERP_CLAMP[1], INTERP_CLAMP[2])
        end
    end
    GT_prop = GT_feas + t * (GT_infeas - GT_feas)
    return GT_prop, :polish
end

# ============================================================================
# Initialize or resume.
# ============================================================================
t_loop_start = time()

if isfile(CKPT_PATH)
    state = deserialize(CKPT_PATH)
    println("[$CHAINID] RESUMING from checkpoint: $(length(state.points)) points completed, status=$(state.status)")
    fp = state.fingerprint
    @assert fp.D == FINGERPRINT.D && fp.W == FINGERPRINT.W && fp.seed == FINGERPRINT.seed &&
            isapprox(fp.sigma, FINGERPRINT.sigma) && fp.focal == FINGERPRINT.focal &&
            fp.anchor_label == FINGERPRINT.anchor_label && fp.direction == FINGERPRINT.direction &&
            fp.q_free_hash == FINGERPRINT.q_free_hash && fp.theta0_hash == FINGERPRINT.theta0_hash (
        "[$CHAINID] FINGERPRINT MISMATCH on resume -- refusing to silently continue. stored=$fp current=$FINGERPRINT")
    if fp.git_commit != FINGERPRINT.git_commit
        @warn "[$CHAINID] resuming under a different git commit (stored=$(fp.git_commit), current=$(FINGERPRINT.git_commit)) -- code may have changed; continuing since state-defining fingerprints (D/W/seed/sigma/anchor/theta0) match."
    end
    # cold-reverify the stored incumbent
    best_pt = state.points[state.most_extreme_idx]
    session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
    r_cold = solve_melitz_delta!(session_d20, best_pt.theta_free, policy_cap; warm_start_source=:neutral)
    @assert r_cold isa FiniteSolved "[$CHAINID] stored incumbent failed cold reverification: $(typeof(r_cold))"
    @assert isapprox(r_cold.Delta, best_pt.Delta; atol=1e-6) "[$CHAINID] stored incumbent Delta drifted on cold reverification: stored=$(best_pt.Delta) live=$(r_cold.Delta)"
    println("[$CHAINID] cold-reverified incumbent OK: GT=$(best_pt.GT) Delta=$(r_cold.Delta)")
else
    println("[$CHAINID] FRESH START -- establishing profiled anchor (Step 1)")
    state = ChainState(ANCHOR_LABEL, DIRECTION, FINGERPRINT, ChainPoint[], :expand, GT_STEP0, nothing,
        0, 0, 0, 1, 0, 0.0, :running, copy(A_free0_d20), copy(q0_anchor), copy(p_star0_d20), g0, GT0, nothing)
    sys0 = melitz_fixed_q_middle_constraint_system(theta_plain0_d20, ctx_d20, obj_d20)
    A_start0 = melitz_project_start_to_middle_constraints(copy(A_free0_d20), sys0, ctx_d20)
    write_pending(GT0, :anchor)
    session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
    r0 = solve_melitz_fixed_q_A_profile_v2(session_d20, q0_anchor, exp(g0), A_start0, ctx_d20;
        coordinate=:logA, max_evals=MAX_MIDDLE_EVALS, box=D20_MIDDLE_BOX, outer_loop_opt=MIDDLE_OPT_D20, sys=sys0,
        cap_handling=CAP_HANDLING, cap_barrier_multiple=CAP_BARRIER_MULTIPLE)
    clear_pending()
    @assert r0.r_incumbent isa FiniteSolved "[$CHAINID] anchor profiled point did not reach FiniteSolved"
    @assert r0.Delta_incumbent <= r0.Delta_start_verified + 1e-6
    lfd_r0 = melitz_recover_lfd(obj_d20, r0.theta_free_incumbent)
    _, f_full0, _, _, q_full0 = expand_free_theta_logcutoff(r0.theta_free_incumbent, ctx_d20)
    pt0 = ChainPoint(1, :anchor, g0, GT0, :FiniteSolved, r0.Delta_incumbent, r0.Delta_incumbent <= DELTA_BUDGET,
        r0.incumbent_source == :input_start ? :continuation : Symbol(r0.incumbent_source), :anchor_establish,
        copy(r0.A_free_incumbent), copy(r0.theta_free_incumbent), f_full0, q_full0,
        copy(r0.r_incumbent.x), copy(lfd_r0.weights), lfd_r0.lfd_ok, lfd_r0.Delta, r0.r_incumbent.nStatus,
        r0.unique_A_points, r0.unique_inner_solves, r0.n_fc_calls, r0.n_ga_calls, r0.wall_s, string(Dates.now()))
    push!(state.points, pt0)
    append_point_csv(pt0)
    state.n_evaluated = 1
    state.cur_A_free = copy(pt0.A_free)
    state.cur_q = copy(pt0.q_full)
    state.cur_p_star = copy(pt0.lfd_weights)
    state.cur_g = pt0.g
    state.cur_GT = pt0.GT
    @printf("[%s] anchor established: GT=%.6f%% Delta=%.6g FiniteSolved\n", CHAINID, pt0.GT, pt0.Delta)
    flush(stdout)
    save_checkpoint(state)
end

# ============================================================================
# Main loop.
# ============================================================================
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
        # Check the convergence tolerance explicitly (Step 4 stop rule).
        GT_feas, Delta_feas, GT_infeas, Delta_infeas = state.bracket
        gap = DELTA_BUDGET - Delta_feas
        if isfinite(gap) && 0.0 <= gap <= DELTA_STOP_TOL
            state.status = :converged
        elseif abs(GT_infeas - GT_feas) <= GT_BRACKET_STOP_TOL
            state.status = :converged   # bracket width converged even if not exactly at delta-tol
        else
            state.status = :max_points   # polish budget exhausted without full convergence
        end
        break
    end

    @printf("\n[%s] [%s] attempt #%d  GT_target=%.6f%%  step_gt=%.5f\n", CHAINID, phase, state.n_evaluated + 1, GT_target, state.step_gt)
    flush(stdout)
    pt, sys_t = evaluate_point!(state, GT_target, phase)
    push!(state.points, pt)
    append_point_csv(pt)
    state.n_evaluated += 1
    if phase == :expand
        state.prev_sys = sys_t
    end
    improved = update_most_extreme!(state, pt)

    @printf("[%s] [%s] result: classification=%s Delta=%s within_budget=%s best_start=%s trigger=%s wall=%.1fs improved_extreme=%s\n",
        CHAINID, phase, pt.classification, string(pt.Delta), pt.within_budget, pt.best_start, pt.trigger_reason, pt.wall_s, improved)
    flush(stdout)

    if phase == :expand
        if pt.within_budget
            state.n_accepted += 1
            state.cur_A_free = copy(pt.A_free)
            state.cur_q = copy(pt.q_full)
            state.cur_p_star = pt.lfd_ok ? copy(pt.lfd_weights) : state.cur_p_star
            state.cur_g = pt.g
            state.cur_GT = pt.GT
            state.step_gt = min(state.step_gt * GT_STEP_GROWTH, GT_STEP_MAX)
        else
            state.bracket = (state.cur_GT, state.points[end-1].Delta, pt.GT, pt.Delta)
            println("[$CHAINID] BRACKET FOUND: feasible GT=$(state.cur_GT) Delta=$(state.points[end-1].Delta) | infeasible GT=$(pt.GT) Delta=$(pt.Delta)")
        end
    else   # :polish
        GT_feas, Delta_feas, GT_infeas, Delta_infeas = state.bracket
        state.n_polish += 1
        if pt.within_budget
            state.n_accepted += 1
            state.bracket = (pt.GT, pt.Delta, GT_infeas, Delta_infeas)
            state.cur_A_free = copy(pt.A_free)
            state.cur_q = copy(pt.q_full)
            state.cur_p_star = pt.lfd_ok ? copy(pt.lfd_weights) : state.cur_p_star
            state.cur_g = pt.g
            state.cur_GT = pt.GT
        else
            state.bracket = (GT_feas, Delta_feas, pt.GT, pt.Delta)
        end
    end

    save_checkpoint(state)
end
state.elapsed_wall_s += (time() - t_loop_start)
save_checkpoint(state)

best = state.points[state.most_extreme_idx]
println("\n", "="^100)
@printf("[%s] CHAIN COMPLETE: status=%s  n_evaluated=%d  n_polish=%d  elapsed_wall_s=%.1f\n",
    CHAINID, state.status, state.n_evaluated, state.n_polish, state.elapsed_wall_s)
@printf("[%s] BEST VERIFIED INCUMBENT: GT=%.6f%%  Delta*=%.6g  classification=%s\n",
    CHAINID, best.GT, best.Delta, best.classification)
println("="^100)
println("\nDONE $CHAINID")
