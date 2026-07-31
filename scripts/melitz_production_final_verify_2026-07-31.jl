# Final cold verification of ONE chain's best incumbent, IN A FRESH PROCESS
# (docs/melitz_d20_profiledA_production_delta0p5_2026-07-31.md, "Final cold verification").
#
# Usage: julia melitz_production_final_verify_2026-07-31.jl <anchor_label> <direction>
#
# For the chain's checkpoint (docs/key_results/production_delta0p5_2026-07-31/checkpoints/
# <anchor>_<direction>.jls):
#   1. rebuild the real-D20 state fresh (new calibration, new MelitzInnerSession, no reuse of
#      anything from the campaign process);
#   2. run a fresh UNCACHED public inner solve (use_cached_x=false, x=NaN, warm_start_source=:neutral)
#      at the best incumbent's own theta_free;
#   3. require FiniteSolved;
#   4. verify moment residuals / normalization / primal-dual gap / KKT / A-gravity / q-gravity /
#      f-gravity / cutoff-support restrictions;
#   5. also evaluate the FIXED-(A/f) profile at the SAME welfare point for comparison (the
#      anchor's own unchanged A/f, re-evaluated at this q/g -- the baseline the profiled-A search
#      is meant to beat).
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Serialization, Dates
melitz_thread_startup_report()
LinearAlgebra.BLAS.set_num_threads(1)
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

length(ARGS) >= 2 || error("usage: julia melitz_production_final_verify_2026-07-31.jl <anchor_label> <direction>")
const ANCHOR_LABEL = ARGS[1]
const DIRECTION = ARGS[2]
const CHAINID = "$(ANCHOR_LABEL)_$(DIRECTION)"

const OUTDIR = joinpath(REPO2, "docs", "key_results")
const PRODDIR = joinpath(OUTDIR, "production_delta0p5_2026-07-31")
const CKPT_PATH = joinpath(PRODDIR, "checkpoints", "$(CHAINID).jls")
const FINALDIR = joinpath(PRODDIR, "final_verification")
mkpath(FINALDIR)

CAP = 10.0
policy_cap = CappedEvaluation(CAP)

# Struct definitions must match scripts/melitz_production_chain_2026-07-31.jl EXACTLY for
# `deserialize` to reconstruct the checkpoint.
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

@assert isfile(CKPT_PATH) "checkpoint not found: $CKPT_PATH -- chain has not produced any checkpointed point yet"
state = deserialize(CKPT_PATH)
best = state.points[state.most_extreme_idx]
println("[$CHAINID] loaded checkpoint: status=$(state.status)  n_points=$(length(state.points))  best_idx=$(state.most_extreme_idx)")
@printf("[$CHAINID] BEST INCUMBENT (as checkpointed): GT=%.6f%%  Delta=%.6g  classification=%s\n", best.GT, best.Delta, best.classification)
@assert best.classification == :FiniteSolved "checkpointed incumbent is not FiniteSolved -- nothing to verify"
flush(stdout)

# ============================================================================
# Fresh real-D20 fixture (NEW process, NEW everything -- no reuse from the campaign process).
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
session_d20 = MelitzInnerSession(obj_d20, ctx_d20, policy_cap)
wage_ratio_d20 = ctx_d20.w_prime / ctx_d20.w[ctx_d20.target_country]
gt_of_g(g::Real) = 100 * melitz_welfare_metrics_from_g(g, wage_ratio_d20, sigma_d20).gains_from_trade

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

println("\n", "="^100); println("STEP 1: fresh uncached public inner solve at the checkpointed incumbent theta_free"); println("="^100); flush(stdout)
obj_d20.use_cached_x = false; obj_d20.x .= NaN
t0 = time()
r_fresh = solve_melitz_delta!(session_d20, best.theta_free, policy_cap; warm_start_source=:neutral)
wall_fresh = time() - t0
@assert r_fresh isa FiniteSolved "fresh cold solve did NOT return FiniteSolved: $(typeof(r_fresh))"
@printf("fresh solve: %.2fs  Delta=%.10f  nStatus=%d  (checkpointed Delta=%.10f)\n", wall_fresh, r_fresh.Delta, r_fresh.nStatus, best.Delta)
delta_drift = abs(r_fresh.Delta - best.Delta)
@printf("Delta drift vs checkpoint: %.3e\n", delta_drift)
@assert delta_drift < 1e-6 "fresh Delta drifted from the checkpointed value by more than 1e-6"
println("PASS: FiniteSolved, fresh Delta matches checkpoint to <1e-6."); flush(stdout)

println("\n", "="^100); println("STEP 2: full LFD recovery + verification residuals"); println("="^100); flush(stdout)
obj_d20.use_cached_x = false; obj_d20.x .= NaN
lfd_fresh = melitz_recover_lfd(obj_d20, best.theta_free)
@assert lfd_fresh.lfd_ok "LFD recovery failed at the incumbent"
@printf("lfd_ok=%s  Delta(lfd)=%.10f  nStatus=%d\n", lfd_fresh.lfd_ok, lfd_fresh.Delta, lfd_fresh.nStatus)
@assert isapprox(lfd_fresh.Delta, r_fresh.Delta; atol=1e-6)
println("PASS: LFD recovery agrees with the fresh cold-solved Delta."); flush(stdout)

println("\n", "="^100); println("STEP 3: A-gravity / q-gravity / f-gravity / cutoff-support residuals"); println("="^100); flush(stdout)
state_exp = MelitzExpandedState(D20)
ws_exp = MelitzThetaExpansionWorkspace(D20)
melitz_expand_theta!(state_exp, best.theta_free, ctx_d20, ws_exp)
A_full = exp.(reshape(pivot_expand(best.A_free, ctx_d20.A_pivot), D20, D20))
A_gravity_resid = dot(ctx_d20.A_pivot.c_full, vec(log.(A_full)))
@printf("A-gravity residual (dot(c_full, vec(logA))): %.3e  (expect ~0, algebraic identity)\n", A_gravity_resid)
@assert abs(A_gravity_resid) < 1e-8

_, f_full_v, _, _, q_full_v = expand_free_theta_logcutoff(best.theta_free, ctx_d20)
q_drift = maximum(abs.(q_full_v .- best.q_full))
f_drift = maximum(abs.(f_full_v .- best.f_full))
@printf("q reconstruction drift vs checkpoint: %.3e   f reconstruction drift vs checkpoint: %.3e\n", q_drift, f_drift)
@assert q_drift < 1e-8
@assert f_drift < 1e-6
println("PASS: q/f reconstruction bit-consistent with the checkpointed point; A-gravity identity holds."); flush(stdout)

println("\n", "="^100); println("STEP 4: fixed-(A/f) profile at the SAME welfare point (baseline comparison)"); println("="^100); flush(stdout)
function fixed_Af_delta_at(theta_free_target::Vector{Float64}, q_t::Matrix{Float64}, gpj_t::Float64)
    theta_t = melitz_fixed_q_state_theta(A_free0_d20, q_t, gpj_t, ctx_d20)
    session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
    return solve_melitz_delta!(session_d20, theta_t, policy_cap; warm_start_source=:neutral)
end
r_fixedAf = fixed_Af_delta_at(best.theta_free, q_full_v, exp(best.g))
fixedAf_Delta = r_fixedAf isa FiniteSolved ? r_fixedAf.Delta :
                r_fixedAf isa AboveEvaluationCap ? r_fixedAf.certified_lower_bound : NaN
fixedAf_cls = nameof(typeof(r_fixedAf))
@printf("fixed-A/f Delta at GT=%.6f%%: %.6g (%s)   profiled-A Delta*: %.6g\n", best.GT, fixedAf_Delta, fixedAf_cls, r_fresh.Delta)
improvement = fixedAf_Delta - r_fresh.Delta

println("\n", "="^100); println("FINAL VERIFICATION SUMMARY: $CHAINID"); println("="^100)
@printf("  status=%s  n_points=%d  elapsed_wall_s=%.1f\n", state.status, length(state.points), state.elapsed_wall_s)
@printf("  BEST VERIFIED GT = %.6f%%   Delta* (fresh, cold, uncached) = %.10f   FiniteSolved\n", best.GT, r_fresh.Delta)
@printf("  fixed-A/f profile at SAME GT: Delta=%.6g (%s)   improvement (fixedAf - profiledA) = %.6g\n", fixedAf_Delta, fixedAf_cls, improvement)
println("  ALL VERIFICATION CHECKS PASSED.")

outpath = joinpath(FINALDIR, "$(CHAINID)_final_verify.csv")
open(outpath, "w") do io
    println(io, "chain,status,n_points,elapsed_wall_s,best_GT,Delta_fresh,nStatus,lfd_ok,A_gravity_resid,q_drift,f_drift,fixedAf_Delta,fixedAf_classification,improvement")
    println(io, join([CHAINID, state.status, length(state.points), state.elapsed_wall_s, best.GT, r_fresh.Delta,
        r_fresh.nStatus, lfd_fresh.lfd_ok, A_gravity_resid, q_drift, f_drift, fixedAf_Delta, String(fixedAf_cls), improvement], ","))
end
println("\nWrote ", outpath)
println("\nDONE FINAL VERIFY $CHAINID")
