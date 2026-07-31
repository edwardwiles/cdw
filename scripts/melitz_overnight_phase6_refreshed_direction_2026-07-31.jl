# Overnight continuation, Phase 6 (optional, run at most once per bound direction): build ONE new
# reduced-q direction at the Phase-5-improved incumbent, and directly evaluate a finite line of
# cutoff candidates along it (no gradient reported to KNITRO).
#
# ARGS = [phase5_chain_label, direction]
#   phase5_chain_label: a docs/key_results/overnight_qpoll_delta0p5_2026-07-31/phase5/checkpoints/
#                        <label>.jls checkpoint (its most_extreme_idx point is the new anchor)
#   direction: "upper" or "lower"
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Dates, Serialization
melitz_thread_startup_report()
LinearAlgebra.BLAS.set_num_threads(1)
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

length(ARGS) >= 2 || error("usage: julia melitz_overnight_phase6_refreshed_direction_2026-07-31.jl <phase5_chain_label> <direction>")
const P5LABEL = ARGS[1]
const DIRECTION = ARGS[2]

const OUTDIR = joinpath(REPO2, "docs", "key_results")
const OVERDIR = joinpath(OUTDIR, "overnight_qpoll_delta0p5_2026-07-31")
const P5CKPT = joinpath(OVERDIR, "phase5", "checkpoints", "$(P5LABEL).jls")
const OUTCSV = joinpath(OVERDIR, "phase6_refreshed_direction_points.csv")
const PDIR = joinpath(OVERDIR, "phase6_points")
mkpath(PDIR)

CAP = 10.0
policy_cap = CappedEvaluation(CAP)
const MIDDLE_OPT_D20 = joinpath(REPO2, "melitz_middle_loop_opt_2026-07-30.opt")
const D20_MIDDLE_BOX = 0.1
const MAX_MIDDLE_EVALS = 180
const CAP_HANDLING = :barrier
const CAP_BARRIER_MULTIPLE = 5.0
const DELTA_BUDGET = 0.5
const H_NEW = 6.31e-3   # same numeric convention as the existing direction's own h (see governing
                         # prompt Phase 6 step 5: "choose h_new using the existing target-switch
                         # scaling with target_switches=100" -- reusing the SAME h magnitude applied
                         # to a freshly-built target_switches=100 direction preserves the same
                         # qualitative "fraction of a switch" meaning as the original h=6.31e-3).

mutable struct ChainPoint
    idx::Int; phase::Symbol; g::Float64; GT::Float64; classification::Symbol; Delta::Float64
    within_budget::Bool; lfd_ok::Bool; lfd_Delta::Float64; A_free::Vector{Float64}; theta_free::Vector{Float64}
    f_full::Matrix{Float64}; q_full::Matrix{Float64}; lfd_weights::Vector{Float64}; nStatus::Int
    unique_inner_solves::Int; wall_s::Float64; timestamp::String
end
mutable struct ChainState
    chain_label::String; direction::Symbol; points::Vector{ChainPoint}
    bracket::Union{Nothing,NTuple{4,Float64}}; n_evaluated::Int; most_extreme_idx::Int
    n_nonimproving::Int; elapsed_wall_s::Float64; status::Symbol
    cur_A_free::Vector{Float64}; cur_q::Matrix{Float64}; cur_p_star::Vector{Float64}
    cur_g::Float64; cur_GT::Float64; step_gt::Float64; n_accepted::Int
end

isfile(P5CKPT) || error("Phase5 checkpoint not found: $P5CKPT")
p5state = deserialize(P5CKPT)
anchor_pt = p5state.points[p5state.most_extreme_idx]
println("[phase6 $DIRECTION] new anchor from $P5LABEL: GT=$(anchor_pt.GT) Delta=$(anchor_pt.Delta) lfd_ok=$(anchor_pt.lfd_ok)")
anchor_pt.lfd_ok || error("refusing to build Phase 6 direction from an unverified anchor")
flush(stdout)

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

# Build the new direction at the anchor point.
obj_d20.use_cached_x = false; obj_d20.x .= NaN
lfd_anchor = melitz_recover_lfd(obj_d20, anchor_pt.theta_free)
@assert lfd_anchor.lfd_ok && isapprox(lfd_anchor.Delta, anchor_pt.Delta; atol=1e-6)
x0_anchor = copy(lfd_anchor.dual_x)

bwpolicy = PowerScaledQBandwidth(1e-3, 80_000, 0.5)
stage_new = melitz_build_reduced_q_stage(anchor_pt.theta_free, x0_anchor, ctx_d20, obj_d20, 2;
    bandwidth_policy=bwpolicy, target_switches=100)
@assert stage_new !== nothing "Phase 6: could not build a refreshed reduced-q direction at the improved anchor"
b_q_new = stage_new.q_basis_free
q_free_anchor = anchor_pt.theta_free[2+nA20:end]
println("[phase6 $DIRECTION] refreshed direction built, |b_q_new|=", norm(b_q_new)); flush(stdout)

function q_full_at(g_target::Real, q_free_target::Vector{Float64})
    th = copy(anchor_pt.theta_free)
    th[1] = g_target
    th[2+nA20:end] .= q_free_target
    _, _, _, _, q = expand_free_theta_logcutoff(th, ctx_d20)
    return q
end

function run_one(m::Int)
    q_free_m = q_free_anchor .+ (Float64(m) * H_NEW) .* b_q_new
    g_target = anchor_pt.g
    gpj_target = exp(g_target)
    t0 = time()
    try
        q_target = q_full_at(g_target, q_free_m)
        theta_for_constraints = melitz_fixed_q_state_theta(anchor_pt.A_free, q_target, gpj_target, ctx_d20)
        local sys_t
        try
            sys_t = melitz_fixed_q_middle_constraint_system(theta_for_constraints, ctx_d20, obj_d20)
        catch e
            return (m=m, outcome=:ScreenRejected, cls=:NumericalFailure, Delta=NaN, lfd_ok=false,
                theta_free=Float64[], A_free=Float64[], q_free=q_free_m, wall_s=time() - t0)
        end
        A_cont = melitz_project_start_to_middle_constraints(copy(anchor_pt.A_free), sys_t, ctx_d20)
        anchor_A_full = exp.(reshape(pivot_expand(anchor_pt.A_free, ctx_d20.A_pivot), D20, D20))
        A_cellwise, status_cellwise = melitz_cellwise_A_from_moments(anchor_A_full, anchor_pt.q_full, q_target,
            anchor_pt.lfd_weights, sorted_ctx_d20, sigma_d20)
        n_bad = count(!=(:ok), status_cellwise)
        if n_bad > 0
            A_cellwise[status_cellwise .!= :ok] .= anchor_A_full[status_cellwise .!= :ok]
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
        lfd_ok_out = false
        if finite
            session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
            lfd_r = melitz_recover_lfd(obj_d20, adapt.theta_free)
            lfd_ok_out = lfd_r.lfd_ok
        end
        outcome = finite ? (lfd_ok_out ? :FiniteSolvedVerified : :FiniteSolvedLfdUnverified) : cls_out
        return (m=m, outcome=outcome, cls=cls_out, Delta=adapt.Delta, lfd_ok=lfd_ok_out,
            theta_free=copy(adapt.theta_free), A_free=copy(adapt.A_free), q_free=q_free_m, wall_s=time() - t0)
    catch e
        @warn "[phase6 $DIRECTION] m=$m threw" exception=(e, catch_backtrace())
        return (m=m, outcome=:error, cls=:NumericalFailure, Delta=NaN, lfd_ok=false,
            theta_free=Float64[], A_free=Float64[], q_free=q_free_anchor, wall_s=time() - t0)
    end
end

const MS = [-4, -2, -1, 1, 2, 4]
function run_grid()
    is_new = !isfile(OUTCSV)
    for m in MS
        r = run_one(m)
        @printf("[phase6 %s] m=%d outcome=%s classification=%s Delta=%s lfd_ok=%s wall=%.1fs\n",
            DIRECTION, m, r.outcome, r.cls, string(r.Delta), r.lfd_ok, r.wall_s)
        flush(stdout)
        open(OUTCSV, "a") do io
            is_new && println(io, "direction,phase5_source,m,outcome,classification,Delta,lfd_ok,wall_s,timestamp")
            println(io, join([DIRECTION, P5LABEL, m, r.outcome, r.cls, r.Delta, r.lfd_ok, r.wall_s, string(Dates.now())], ","))
        end
        is_new = false
        if r.cls == :FiniteSolved
            serialize(joinpath(PDIR, "$(DIRECTION)_m$(m).jls"),
                (direction=DIRECTION, m=m, GT_target=gt_of_g(anchor_pt.g), g=anchor_pt.g, Delta=r.Delta,
                 classification=r.cls, lfd_ok=r.lfd_ok, theta_free=r.theta_free, A_free=r.A_free, q_free=r.q_free))
        end
    end
end
run_grid()
println("\nDONE PHASE6 $DIRECTION")
