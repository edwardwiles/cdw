# Overnight continuation, Phase 3/4: finite derivative-free cutoff-anchor grid search along the
# existing reduced-q direction, profiling A at a seed's own welfare level (Phase 3) or its
# challenge welfare level (Phase 4, seed GT +/- 0.05pp).
#
# ARGS = [seed_chain_id, m, gt_mode]
#   seed_chain_id in ("reduced_q_pre_switch_upper", "current_calibration_upper",
#                      "reduced_q_post_switch_lower", "reduced_q_pre_switch_lower")
#   m             integer grid multiple, q_free(m) = q_free0 + m*h*b_q, h=6.31e-3
#   gt_mode       "seed" (Phase 3: profile at the seed's own GT) or "challenge" (Phase 4: seed GT
#                 +0.05pp for an upper seed, -0.05pp for a lower seed)
#
# Reuses (does not modify): solve_melitz_fixed_q_A_profile_v2 / melitz_middle_two_start_adaptive!,
# melitz_cellwise_A_from_moments, melitz_project_start_to_middle_constraints,
# melitz_fixed_q_middle_constraint_system -- the exact same middle-loop/screening machinery as the
# production driver and the Phase-1 resume driver. Does not modify the Ricardian implementation,
# does not place q into KNITRO, does not build a new cutoff-gradient method.
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Dates, Serialization
melitz_thread_startup_report()
LinearAlgebra.BLAS.set_num_threads(1)
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

length(ARGS) >= 3 || error("usage: julia melitz_overnight_phase3_cutoff_grid_2026-07-31.jl <seed_chain_id> <m> <gt_mode>")
const SEED_CHAIN_ID = ARGS[1]
const MGRID = parse(Int, ARGS[2])
const GT_MODE = ARGS[3]
GT_MODE in ("seed", "challenge") || error("gt_mode must be seed or challenge")

const OUTDIR = joinpath(REPO2, "docs", "key_results")
const PRODDIR = joinpath(OUTDIR, "production_delta0p5_2026-07-31")
const OVERDIR = joinpath(OUTDIR, "overnight_qpoll_delta0p5_2026-07-31")
const GRIDCSV = joinpath(OVERDIR, "phase3_phase4_grid_points.csv")
mkpath(OVERDIR)

CAP = 10.0
policy_cap = CappedEvaluation(CAP)
const MIDDLE_OPT_D20 = joinpath(REPO2, "melitz_middle_loop_opt_2026-07-30.opt")
const D20_MIDDLE_BOX = 0.1
const MAX_MIDDLE_EVALS = 180   # governing prompt Phase 3: reject if >180 unique middle-A evals without improving on the verified start
const CAP_HANDLING = :barrier
const CAP_BARRIER_MULTIPLE = 5.0
const DELTA_BUDGET = 0.5
const H_STEP = 6.31e-3
const CHALLENGE_PP = 0.05

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

ckpt_path = joinpath(PRODDIR, "checkpoints", "$(SEED_CHAIN_ID).jls")
isfile(ckpt_path) || error("seed checkpoint not found: $ckpt_path")
seed_state = deserialize(ckpt_path)
seed_pt = seed_state.points[seed_state.most_extreme_idx]
seed_direction = seed_state.direction   # :upper or :lower
println("[$SEED_CHAIN_ID m=$MGRID $GT_MODE] seed: GT=$(seed_pt.GT) Delta=$(seed_pt.Delta) lfd_ok=$(seed_pt.lfd_ok) direction=$seed_direction")
flush(stdout)

# ============================================================================
# Real-D20 fixture (identical recipe to the production/resume drivers).
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
A_free0_d20 = theta_plain0_d20[2:1+nA20]
q_free0_d20 = theta_plain0_d20[2+nA20:end]

obj_d20.use_cached_x = false; obj_d20.x .= NaN
lfd0 = melitz_recover_lfd(obj_d20, theta0_d20)
@assert lfd0.lfd_ok && isapprox(lfd0.Delta, 0.483276; atol=1e-4)
x0_d20 = copy(lfd0.dual_x)

bwpolicy = PowerScaledQBandwidth(1e-3, 80_000, 0.5)
stage_d20 = melitz_build_reduced_q_stage(theta0_d20, x0_d20, ctx_d20, obj_d20, 1;
    bandwidth_policy=bwpolicy, target_switches=100)
b_q_d20 = stage_d20.q_basis_free

q_free_m = q_free0_d20 .+ (Float64(MGRID) * H_STEP) .* b_q_d20

function q_full_at_g(g_target::Real, q_free_target::Vector{Float64})
    th = copy(theta_plain0_d20)
    th[1] = g_target
    th[2+nA20:end] .= q_free_target
    _, _, _, _, q = expand_free_theta_logcutoff(th, ctx_d20)
    return q
end

# Target welfare point.
GT_target = GT_MODE == "seed" ? seed_pt.GT :
            (seed_direction == :upper ? seed_pt.GT + CHALLENGE_PP : seed_pt.GT - CHALLENGE_PP)
g_target = g_of_gt(GT_target)
gpj_target = exp(g_target)

println("[$SEED_CHAIN_ID m=$MGRID $GT_MODE] target GT=$GT_target%  g=$g_target"); flush(stdout)

# ============================================================================
# Screen + profile A.
# ============================================================================
t0 = time()
function run_grid_point(t0::Float64)
    try
        q_target = q_full_at_g(g_target, q_free_m)
        theta_for_constraints = melitz_fixed_q_state_theta(seed_pt.A_free, q_target, gpj_target, ctx_d20)
        local sys_t
        try
            sys_t = melitz_fixed_q_middle_constraint_system(theta_for_constraints, ctx_d20, obj_d20)
        catch e
            println("[$SEED_CHAIN_ID m=$MGRID $GT_MODE] REJECTED at constraint-system construction (native structural restriction violated): ", e)
            return (outcome=:ScreenRejected, cls=:NumericalFailure, Delta=NaN, lfd_ok=false, lfd_Delta=NaN,
                nStatus=-999, n_unique=0, n_solves=0, theta_free=Float64[], A_free=Float64[], wall_s=time() - t0)
        end

        A_cont = melitz_project_start_to_middle_constraints(copy(seed_pt.A_free), sys_t, ctx_d20)
        seed_A_full = exp.(reshape(pivot_expand(seed_pt.A_free, ctx_d20.A_pivot), D20, D20))
        A_cellwise, status_cellwise = melitz_cellwise_A_from_moments(seed_A_full, seed_pt.q_full, q_target,
            seed_pt.lfd_weights, sorted_ctx_d20, sigma_d20)
        n_bad = count(!=(:ok), status_cellwise)
        if n_bad > 0
            A_cellwise[status_cellwise .!= :ok] .= seed_A_full[status_cellwise .!= :ok]
        end
        A_cellwise_free = pivot_reduce(vec(log.(A_cellwise)), ctx_d20.A_pivot)
        A_comp = melitz_project_start_to_middle_constraints(A_cellwise_free, sys_t, ctx_d20)

        session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
        adapt = melitz_middle_two_start_adaptive!(session_d20, q_target, gpj_target, A_cont, A_comp, ctx_d20;
            delta_budget=DELTA_BUDGET, policy=MelitzAdaptiveStartPolicy(), periodic_safeguard_due=false,
            sys=sys_t, prev_sys=nothing, coordinate=:logA, max_evals=MAX_MIDDLE_EVALS,
            box=D20_MIDDLE_BOX, outer_loop_opt=MIDDLE_OPT_D20, cap_handling=CAP_HANDLING,
            cap_barrier_multiple=CAP_BARRIER_MULTIPLE)

        finite = adapt.r_incumbent isa FiniteSolved
        cls_out = finite ? :FiniteSolved :
                  adapt.r_incumbent isa AboveEvaluationCap ? :AboveEvaluationCap :
                  adapt.r_incumbent isa InfiniteDeltaCertified ? :InfiniteDeltaCertified : :NumericalFailure
        n_unique_out = adapt.r_continuation.unique_A_points + (adapt.ran_compensated ? adapt.r_compensated.unique_A_points : 0)
        n_solves_out = adapt.r_continuation.unique_inner_solves + (adapt.ran_compensated ? adapt.r_compensated.unique_inner_solves : 0)
        nStatus_out = finite ? adapt.r_incumbent.nStatus : -999

        lfd_ok_out = false; lfd_Delta_out = NaN; outcome = cls_out
        if finite
            session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
            lfd_r = melitz_recover_lfd(obj_d20, adapt.theta_free)
            lfd_ok_out = lfd_r.lfd_ok
            lfd_Delta_out = lfd_r.Delta
            outcome = lfd_ok_out ? :FiniteSolvedVerified : :FiniteSolvedLfdUnverified
        end
        return (outcome=outcome, cls=cls_out, Delta=adapt.Delta, lfd_ok=lfd_ok_out, lfd_Delta=lfd_Delta_out,
            nStatus=nStatus_out, n_unique=n_unique_out, n_solves=n_solves_out,
            theta_free=copy(adapt.theta_free), A_free=copy(adapt.A_free), wall_s=time() - t0)
    catch e
        @warn "[$SEED_CHAIN_ID m=$MGRID $GT_MODE] threw" exception=(e, catch_backtrace())
        return (outcome=:error, cls=:NumericalFailure, Delta=NaN, lfd_ok=false, lfd_Delta=NaN,
            nStatus=-999, n_unique=0, n_solves=0, theta_free=Float64[], A_free=Float64[], wall_s=time() - t0)
    end
end

result = run_grid_point(time())
outcome, cls_out, Delta_out, lfd_ok_out, lfd_Delta_out, nStatus_out, n_unique_out, n_solves_out, theta_free_out, A_free_out, wall_s =
    result.outcome, result.cls, result.Delta, result.lfd_ok, result.lfd_Delta, result.nStatus, result.n_unique, result.n_solves, result.theta_free, result.A_free, result.wall_s

@printf("[%s m=%d %s] outcome=%s classification=%s Delta=%s lfd_ok=%s wall=%.1fs\n",
    SEED_CHAIN_ID, MGRID, GT_MODE, outcome, cls_out, string(Delta_out), lfd_ok_out, wall_s)
flush(stdout)

is_new = !isfile(GRIDCSV)
open(GRIDCSV, "a") do io
    is_new && println(io, "seed_chain_id,seed_direction,m,gt_mode,GT_target,outcome,classification,Delta,lfd_ok,lfd_Delta,nStatus,unique_A_points,unique_inner_solves,wall_s,timestamp")
    println(io, join([SEED_CHAIN_ID, seed_direction, MGRID, GT_MODE, GT_target, outcome, cls_out, Delta_out,
        lfd_ok_out, lfd_Delta_out, nStatus_out, n_unique_out, n_solves_out, wall_s, string(Dates.now())], ","))
end

# Persist the full point (A_free/theta_free) for any FiniteSolved (verified or not) result, keyed
# by (seed, m, gt_mode), so Phase 5 can pick up a challenge-point win as a warm continuation start.
if cls_out == :FiniteSolved
    pdir = joinpath(OVERDIR, "phase3_phase4_points")
    mkpath(pdir)
    outpath = joinpath(pdir, "$(SEED_CHAIN_ID)_m$(MGRID)_$(GT_MODE).jls")
    serialize(outpath, (seed_chain_id=SEED_CHAIN_ID, m=MGRID, gt_mode=GT_MODE, GT_target=GT_target,
        g=g_target, Delta=Delta_out, classification=cls_out, lfd_ok=lfd_ok_out, lfd_Delta=lfd_Delta_out,
        theta_free=theta_free_out, A_free=A_free_out, q_free=q_free_m))
end

println("\nDONE PHASE3GRID $SEED_CHAIN_ID m=$MGRID $GT_MODE")
