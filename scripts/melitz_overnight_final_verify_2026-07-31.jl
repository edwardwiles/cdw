# Overnight campaign final selection and verification (fresh process).
#
# ARGS = [phase5_chain_label, direction, nearest_anchor_label]
#   phase5_chain_label: docs/key_results/overnight_qpoll_delta0p5_2026-07-31/phase5/checkpoints/
#                        <label>.jls (its most_extreme_idx point is the final incumbent)
#   direction:          "upper" or "lower"
#   nearest_anchor_label: one of the three ORIGINAL production anchors
#                        ("current_calibration","reduced_q_pre_switch","reduced_q_post_switch")
#                        whose OWN A_free (profiled at that anchor's own state) is evaluated at
#                        this incumbent's (q,g) for the "profiled A from the original nearest
#                        anchor" comparison required by the governing prompt's Final Selection.
#
# Verifies, in this fresh process:
#   1. rebuild all state (fresh calibration, fresh MelitzInnerSession);
#   2. fresh uncached public inner solve at the incumbent's theta_free;
#   3. require FiniteSolved;
#   4. moments/normalization/primal-dual/KKT (via melitz_recover_lfd + solver's own nStatus),
#      A-gravity identity, q/f-gravity reconstruction, cutoff/support restrictions (native checks
#      inside expand_free_theta_logcutoff/melitz_fixed_q_middle_constraint_system, which throw on
#      violation);
#   5. at the SAME (q,g): fixed original-calibration A/f; the nearest original anchor's OWN
#      profiled A/f; and this incumbent's own final profiled A/f.
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Serialization, Dates
melitz_thread_startup_report()
LinearAlgebra.BLAS.set_num_threads(1)
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

length(ARGS) >= 3 || error("usage: julia melitz_overnight_final_verify_2026-07-31.jl <phase5_chain_label> <direction> <nearest_anchor_label>")
const P5LABEL = ARGS[1]
const DIRECTION = ARGS[2]
const NEAREST_ANCHOR = ARGS[3]

const OUTDIR = joinpath(REPO2, "docs", "key_results")
const PRODDIR = joinpath(OUTDIR, "production_delta0p5_2026-07-31")
const OVERDIR = joinpath(OUTDIR, "overnight_qpoll_delta0p5_2026-07-31")
const P5CKPT = joinpath(OVERDIR, "phase5", "checkpoints", "$(P5LABEL).jls")
const FINALDIR = joinpath(OVERDIR, "final_verification")
mkpath(FINALDIR)

CAP = 10.0
policy_cap = CappedEvaluation(CAP)

# Struct definitions must match melitz_overnight_phase5_continuation_2026-07-31.jl EXACTLY.
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

isfile(P5CKPT) || error("checkpoint not found: $P5CKPT")
state = deserialize(P5CKPT)
best = state.points[state.most_extreme_idx]
println("[$P5LABEL] loaded checkpoint: status=$(state.status) n_points=$(length(state.points)) best_idx=$(state.most_extreme_idx)")
@printf("[%s] INCUMBENT (as checkpointed): GT=%.6f%%  Delta=%.6g  classification=%s  lfd_ok=%s\n",
    P5LABEL, best.GT, best.Delta, best.classification, best.lfd_ok)
@assert best.classification == :FiniteSolved && best.lfd_ok "checkpointed incumbent is not a verified FiniteSolved point"
flush(stdout)

# ============================================================================
# Fresh real-D20 fixture (NEW process, no reuse from any campaign process).
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
A_free0_d20 = theta_plain0_d20[2:1+nA20]         # fixed original-calibration A/f baseline
q_free0_d20 = theta_plain0_d20[2+nA20:end]

obj_d20.use_cached_x = false; obj_d20.x .= NaN
lfd0 = melitz_recover_lfd(obj_d20, theta0_d20)
@assert lfd0.lfd_ok && isapprox(lfd0.Delta, 0.483276; atol=1e-4)
x0_d20 = copy(lfd0.dual_x)
bwpolicy = PowerScaledQBandwidth(1e-3, 80_000, 0.5)
stage_d20 = melitz_build_reduced_q_stage(theta0_d20, x0_d20, ctx_d20, obj_d20, 1;
    bandwidth_policy=bwpolicy, target_switches=100)
b_q_d20 = stage_d20.q_basis_free
const ANCHOR_Q_FREE = Dict(
    "current_calibration" => copy(q_free0_d20),
    "reduced_q_pre_switch" => q_free0_d20 .+ (-1.0) .* 6.31e-3 .* b_q_d20,
    "reduced_q_post_switch" => q_free0_d20 .+ (-1.0) .* 1.26e-2 .* b_q_d20,
)
const ANCHOR_AFREE_PATH = joinpath(OVERDIR, "anchor_afree_extract", "$(NEAREST_ANCHOR)_$(DIRECTION)_afree.jls")
isfile(ANCHOR_AFREE_PATH) || error("pre-extracted anchor A_free not found: $ANCHOR_AFREE_PATH -- run melitz_extract_anchor_afree_2026-07-31.jl first")
anchor_A_free = deserialize(ANCHOR_AFREE_PATH)::Vector{Float64}

println("\n", "="^100); println("STEP 1: fresh uncached public inner solve at the incumbent theta_free"); println("="^100); flush(stdout)
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

println("\n", "="^100); println("STEP 2: full LFD recovery + primal-dual/KKT verification"); println("="^100); flush(stdout)
obj_d20.use_cached_x = false; obj_d20.x .= NaN
lfd_fresh = melitz_recover_lfd(obj_d20, best.theta_free)
@assert lfd_fresh.lfd_ok "LFD recovery failed at the incumbent"
@printf("lfd_ok=%s  Delta(lfd)=%.10f  nStatus=%d\n", lfd_fresh.lfd_ok, lfd_fresh.Delta, lfd_fresh.nStatus)
@assert isapprox(lfd_fresh.Delta, r_fresh.Delta; atol=1e-6)
println("PASS: LFD recovery (primal-dual agreement / KKT) agrees with the fresh cold-solved Delta."); flush(stdout)

println("\n", "="^100); println("STEP 3: A-gravity / q-gravity / f-gravity / cutoff-support residuals"); println("="^100); flush(stdout)
A_full = exp.(reshape(pivot_expand(best.A_free, ctx_d20.A_pivot), D20, D20))
A_gravity_resid = dot(ctx_d20.A_pivot.c, vec(log.(A_full))) + ctx_d20.A_pivot.g0
@printf("A-gravity residual: %.3e (expect ~0)\n", A_gravity_resid)
@assert abs(A_gravity_resid) < 1e-8
_, f_full_v, _, _, q_full_v = expand_free_theta_logcutoff(best.theta_free, ctx_d20)
q_drift = maximum(abs.(q_full_v .- best.q_full))
f_drift = maximum(abs.(f_full_v .- best.f_full))
@printf("q reconstruction drift: %.3e   f reconstruction drift: %.3e\n", q_drift, f_drift)
@assert q_drift < 1e-8
@assert f_drift < 1e-6
println("PASS: q/f reconstruction bit-consistent; A-gravity identity holds; cutoff/support restrictions satisfied (native checks did not throw)."); flush(stdout)

println("\n", "="^100); println("STEP 4: three-way A/f comparison at the SAME (q,g)"); println("="^100); flush(stdout)
function delta_at_fixed_A(A_free_ref::Vector{Float64})
    theta_t = melitz_fixed_q_state_theta(A_free_ref, q_full_v, exp(best.g), ctx_d20)
    session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
    return solve_melitz_delta!(session_d20, theta_t, policy_cap; warm_start_source=:neutral)
end
r_fixed_calib = delta_at_fixed_A(A_free0_d20)
r_nearest_anchor = delta_at_fixed_A(anchor_A_free)
class_of(r) = nameof(typeof(r))
delta_of(r) = r isa FiniteSolved ? r.Delta : r isa AboveEvaluationCap ? r.certified_lower_bound : NaN
@printf("fixed original-calibration A/f  : Delta=%.6g (%s)\n", delta_of(r_fixed_calib), class_of(r_fixed_calib))
@printf("nearest-anchor (%s) profiled A  : Delta=%.6g (%s)\n", NEAREST_ANCHOR, delta_of(r_nearest_anchor), class_of(r_nearest_anchor))
@printf("final profiled A (this incumbent): Delta=%.10f (FiniteSolved)\n", r_fresh.Delta)

println("\n", "="^100); println("FINAL VERIFICATION SUMMARY: $P5LABEL ($DIRECTION)"); println("="^100)
@printf("  status=%s  n_points=%d  elapsed_wall_s=%.1f\n", state.status, length(state.points), state.elapsed_wall_s)
@printf("  BEST VERIFIED GT = %.6f%%   Delta* (fresh, cold, uncached) = %.10f   FiniteSolved\n", best.GT, r_fresh.Delta)
println("  ALL VERIFICATION CHECKS PASSED.")

outpath = joinpath(FINALDIR, "$(P5LABEL)_final_verify.csv")
open(outpath, "w") do io
    println(io, "chain,direction,status,n_points,elapsed_wall_s,best_GT,Delta_fresh,nStatus,lfd_ok,A_gravity_resid,q_drift,f_drift,fixed_calib_Delta,fixed_calib_class,nearest_anchor_label,nearest_anchor_Delta,nearest_anchor_class")
    println(io, join([P5LABEL, DIRECTION, state.status, length(state.points), state.elapsed_wall_s, best.GT, r_fresh.Delta,
        r_fresh.nStatus, lfd_fresh.lfd_ok, A_gravity_resid, q_drift, f_drift, delta_of(r_fixed_calib), String(class_of(r_fixed_calib)),
        NEAREST_ANCHOR, delta_of(r_nearest_anchor), String(class_of(r_nearest_anchor))], ","))
end
println("\nWrote ", outpath)
println("\nDONE FINAL VERIFY $P5LABEL")
