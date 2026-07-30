# Phase 8 (profiledA_parallel_speed_and_cutoff_portfolio governing prompt, 2026-07-30): bounded
# campaign-readiness comparison. NOT the final frontier campaign (governing prompt's own
# explicit caveat). Uses the adaptive second-start policy (Phase 2) throughout.
#
# Anchors: Phase 7's OWN portfolio run (melitz_phase7_cutoff_portfolio_2026-07-30.csv) found
# only 3/6 candidate anchors survive its feasibility screen at the calibration welfare point --
# current_calibration, reduced_q_pre_switch, reduced_q_post_switch (all FiniteSolved); the other
# three (rank_spaced, origin_block_korea, destination_block_focal) were REJECTED there
# (AboveEvaluationCap on the very first, continuation-projected evaluation -- an immediate
# support cliff, exactly the governing prompt's own explicit screening criterion, not an
# implementation defect: disclosed, not silently substituted). Feeding an already-disqualified
# anchor into this readiness comparison would trivially fail again for the same reason and waste
# the bounded compute budget, so this script uses the THREE anchors that actually survived
# Phase 7's screen, not the originally-planned four-anchor wishlist.
#   1. current_calibration
#   2. reduced_q_pre_switch     (structured perturbation #1 -- before the audited minus-side cliff)
#   3. reduced_q_post_switch    (structured perturbation #2 -- past the audited cliff)
# Directions: upper (increasing GT), lower (decreasing GT). delta=0.5. At most 5 profiled
# welfare points per anchor/direction, FIXED GT-percentage-point step offsets (0.10, 0.20, 0.30,
# 0.40, 0.50 pp from that anchor's own g0), not the full safeguarded-bisection search (out of
# this bounded readiness check's own scope -- a simple, deterministic, reproducible probe of
# whether independent cutoff anchors find meaningfully different incumbents, not a new frontier
# search).
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
MIDDLE_OPT_D20 = joinpath(REPO2, "melitz_middle_loop_opt_2026-07-30.opt")
const D20_MIDDLE_BOX = 0.1
const MAX_MIDDLE_EVALS = 120
const CAP_HANDLING = :barrier
const CAP_BARRIER_MULTIPLE = 5.0
const DELTA_BUDGET = 0.5
const GT_OFFSETS = (0.10, 0.20, 0.30, 0.40, 0.50)   # percentage points, up to 5 points

function load_realD20_calib()
    real_dir = joinpath(REPO2, "real_data", "noah_D20")
    lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    return calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal,
        p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6), focal, countries
end
calib, focal, countries = load_realD20_calib()
korea_idx = findfirst(==("kor"), countries)
korea_idx === nothing && (korea_idx = 14)

obj_d20, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
    inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
ctx_d20 = obj_d20.γ
D20 = ctx_d20.D; nA20 = D20^2 - 1
sigma_d20 = ctx_d20.sigma
sorted_ctx_d20 = ctx_d20.sorted_tail_ctx
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
q_free0_d20 = theta_plain0_d20[2+nA20:end]
g0 = theta_plain0_d20[1]
wage_ratio_d20 = ctx_d20.w_prime / ctx_d20.w[ctx_d20.target_country]
gt_of_g(g::Real) = 100 * melitz_welfare_metrics_from_g(g, wage_ratio_d20, sigma_d20).gains_from_trade
g_of_gt(gt_pct::Real) = (sigma_d20 - 1) * log((1 - gt_pct / 100) / wage_ratio_d20)
GT0 = gt_of_g(g0)

session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
lfd0 = melitz_recover_lfd(obj_d20, theta0_d20)
@assert lfd0.lfd_ok
p_star0_d20 = copy(lfd0.weights)

# Reduced-q direction: EXACT same basis/thresholds as Phase 7 (and the underlying negative-
# switch audit) -- not re-derived independently.
session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
x0_d20 = copy(lfd0.dual_x)
bwpolicy = PowerScaledQBandwidth(1e-3, 80_000, 0.5)
stage_d20 = melitz_build_reduced_q_stage(theta0_d20, x0_d20, ctx_d20, obj_d20, 1;
    bandwidth_policy=bwpolicy, target_switches=100)
b_q_d20 = stage_d20.q_basis_free

anchors = [
    (label="current_calibration", q_free_fixed=copy(q_free0_d20)),
    (label="reduced_q_pre_switch", q_free_fixed=q_free0_d20 .+ (-1.0) .* 6.31e-3 .* b_q_d20),
    (label="reduced_q_post_switch", q_free_fixed=q_free0_d20 .+ (-1.0) .* 1.26e-2 .* b_q_d20),
]

function q_full_at_g_anchor(g_target::Real, q_free_fixed::Vector{Float64})
    th = copy(theta_plain0_d20)
    th[1] = g_target
    th[2+nA20:end] .= q_free_fixed
    _, _, _, _, q = expand_free_theta_logcutoff(th, ctx_d20)
    return q
end

all_rows = NamedTuple[]
for anc in anchors
    q0_anchor = q_full_at_g_anchor(g0, anc.q_free_fixed)
    for direction in (:upper, :lower)
        sign = direction == :upper ? 1 : -1
        println("\n", "="^100); println("ANCHOR=", anc.label, "  DIRECTION=", direction); flush(stdout)
        cur_A_free = copy(A_free0_d20)
        cur_q = copy(q0_anchor)
        cur_p_star = copy(p_star0_d20)
        n_points = 0
        for offset in GT_OFFSETS
            n_points += 1
            GT_target = GT0 + sign * offset
            g_target = g_of_gt(GT_target)
            q_target = q_full_at_g_anchor(g_target, anc.q_free_fixed)
            gpj_target = exp(g_target)

            theta_for_constraints = melitz_fixed_q_state_theta(cur_A_free, q_target, gpj_target, ctx_d20)
            sys_t = melitz_fixed_q_middle_constraint_system(theta_for_constraints, ctx_d20, obj_d20)
            A_cont = melitz_project_start_to_middle_constraints(copy(cur_A_free), sys_t, ctx_d20)
            prev_A_full = exp.(reshape(pivot_expand(cur_A_free, ctx_d20.A_pivot), D20, D20))
            A_cellwise, status_cellwise = melitz_cellwise_A_from_moments(prev_A_full, cur_q, q_target, cur_p_star,
                                                                           sorted_ctx_d20, sigma_d20)
            n_bad = count(!=(:ok), status_cellwise)
            if n_bad > 0
                A_cellwise[status_cellwise .!= :ok] .= prev_A_full[status_cellwise .!= :ok]
            end
            A_cellwise_free = pivot_reduce(vec(log.(A_cellwise)), ctx_d20.A_pivot)
            A_comp = melitz_project_start_to_middle_constraints(A_cellwise_free, sys_t, ctx_d20)

            local adapt
            local ok = true
            t0 = time()
            try
                adapt = melitz_middle_two_start_adaptive!(session_d20, q_target, gpj_target, A_cont, A_comp, ctx_d20;
                    delta_budget=DELTA_BUDGET, sys=sys_t, coordinate=:logA, max_evals=MAX_MIDDLE_EVALS,
                    box=D20_MIDDLE_BOX, outer_loop_opt=MIDDLE_OPT_D20, cap_handling=CAP_HANDLING,
                    cap_barrier_multiple=CAP_BARRIER_MULTIPLE)
            catch e
                ok = false
                @warn "anchor=$(anc.label) dir=$direction GT_target=$GT_target: threw" exception=(e, catch_backtrace())
            end
            wall = time() - t0

            if !ok
                push!(all_rows, (anchor=anc.label, direction=String(direction), point=n_points, GT_target=GT_target,
                    finite=false, Delta=NaN, within_budget=false, ran_compensated=false, trigger="error",
                    unique_inner_solves=0, wall_s=wall))
                break
            end
            finite = adapt.r_incumbent isa FiniteSolved
            within_budget = finite && adapt.Delta <= DELTA_BUDGET
            n_unique = adapt.r_continuation.unique_inner_solves + (adapt.ran_compensated ? adapt.r_compensated.unique_inner_solves : 0)
            @printf("  [pt %d] GT_target=%.4f%% finite=%s Delta=%.6g within_budget=%s trigger=%s wall=%.1fs\n",
                n_points, GT_target, finite, adapt.Delta, within_budget, adapt.trigger_reason, wall)
            flush(stdout)
            push!(all_rows, (anchor=anc.label, direction=String(direction), point=n_points, GT_target=GT_target,
                finite=finite, Delta=adapt.Delta, within_budget=within_budget, ran_compensated=adapt.ran_compensated,
                trigger=String(adapt.trigger_reason), unique_inner_solves=n_unique, wall_s=wall))

            if within_budget
                cur_A_free = adapt.A_free
                _, _, _, _, cur_q = expand_free_theta_logcutoff(adapt.theta_free, ctx_d20)
                session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
                lfd_r = melitz_recover_lfd(obj_d20, adapt.theta_free)
                cur_p_star = lfd_r.lfd_ok ? copy(lfd_r.weights) : cur_p_star
            else
                break   # budget exceeded -- stop expanding this anchor/direction (bounded probe)
            end
        end
    end
end

outpath = joinpath(OUTDIR, "melitz_phase8_readiness_2026-07-30.csv")
open(outpath, "w") do io
    println(io, "anchor,direction,point,GT_target,finite,Delta,within_budget,ran_compensated,trigger,unique_inner_solves,wall_s")
    for r in all_rows
        println(io, join([r.anchor, r.direction, r.point, r.GT_target, r.finite, r.Delta, r.within_budget,
            r.ran_compensated, r.trigger, r.unique_inner_solves, r.wall_s], ","))
    end
end
println("\nWrote ", outpath)

println("\n== Best verified GT per anchor/direction ==")
for anc in anchors, direction in ("upper", "lower")
    rows = filter(r -> r.anchor == anc.label && r.direction == direction && r.within_budget, all_rows)
    if isempty(rows)
        println("  ", anc.label, " / ", direction, ": no verified within-budget point reached")
    else
        best = direction == "upper" ? rows[argmax([r.GT_target for r in rows])] : rows[argmin([r.GT_target for r in rows])]
        @printf("  %-26s / %-6s: best_GT=%.4f%%  Delta=%.6g\n", anc.label, direction, best.GT_target, best.Delta)
    end
end
println("\nPHASE 8 READINESS COMPARISON COMPLETE")
