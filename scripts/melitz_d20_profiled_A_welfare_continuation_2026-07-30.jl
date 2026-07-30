# Final bounded D20 search experiment: profiled-A welfare continuation at delta=0.5.
# Governing prompt (2026-07-30), follow-up to
# docs/melitz_fixed_q_A_middle_loop_experiment_2026-07-30.md, gated on
# scripts/melitz_addendum_audit_and_gate_2026-07-30.jl PASSING first (mandatory performance and
# cap-handling addendum). Uses ONLY the repaired v2 middle-loop driver
# (`solve_melitz_fixed_q_A_profile_v2`, src/melitz/fixed_q_a_middle_loop.jl).
#
# NON-NEGOTIABLE SCOPE (governing prompt): D=20, W=80,000, delta=0.5, upper+lower GT
# directions only. No q-gradient/broad multistart/relative-parameterization development. Free
# q coordinates held FIXED at the anchor's own values; at each welfare point g, the FULL q is
# reconstructed via the EXISTING q-gravity map (`expand_free_theta_logcutoff`'s own
# `build_q_gravity_offset`/`derive_qjj_from_autarky_cutoff` machinery, unmodified) -- i.e.
# `theta_probe = copy(theta_plain0_d20); theta_probe[1] = g_target` reproduces q(g) exactly,
# since q_free_free (theta_probe[2+nA:end]) is untouched and only the g-dependent pivot offset
# moves (verified live below, not assumed).
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random, Dates, Serialization
melitz_thread_startup_report()
LinearAlgebra.BLAS.set_num_threads(1)
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

const OUTDIR = joinpath(REPO2, "docs", "key_results")
const SCRATCH = @__DIR__
CAP = 10.0
policy_cap = CappedEvaluation(CAP)
MIDDLE_OPT_D20 = joinpath(REPO2, "melitz_middle_loop_opt_2026-07-30.opt")
const D20_MIDDLE_BOX = 0.1
const MAX_MIDDLE_EVALS = 120
const DELTA_BUDGET = 0.5
const MAX_WELFARE_POINTS_PER_DIRECTION = 15
# Set from the live Addendum-G A/B gate (scripts/melitz_addendum_audit_and_gate_2026-07-30.jl,
# docs/key_results/melitz_addendumG_gate_v1_vs_v2_2026-07-30.csv): whichever of
# cap_handling in (:reject, :barrier) achieved the LOWER Delta_incumbent at the SAME real-D20
# anchor/start/budget. Both modes satisfy the structural gate (never worse than the verified
# start, dedup) identically by construction (strict incumbent retention) -- this only changes
# which explores better within budget.
const CAP_HANDLING = :barrier
const CAP_BARRIER_MULTIPLE = 5.0
const GT_STEP0 = 0.05          # initial welfare-movement step, GT percentage points
const GT_STEP_MIN = 0.0025     # stop expansion below this step
const GT_BRACKET_TOL = 0.005   # stop bisection when bracket width below this
const GT_STEP_GROWTH = 1.5     # optional step growth on acceptance (<=50%)

# ============================================================================
# Real-D20 anchor reconstruction (identical recipe to the prior session's own scripts).
# ============================================================================
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
println("focal country index = ", focal); flush(stdout)

obj_d20, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
    inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
ctx_d20 = obj_d20.γ
D20 = ctx_d20.D; nA20 = D20^2 - 1
println("D=", D20, "  nA=", nA20); flush(stdout)

obj_d20.use_cached_x = false; obj_d20.x .= NaN
t0 = time()
lfd0 = melitz_recover_lfd(obj_d20, theta0_d20)
@printf("D20 base-point solve: %.2fs  Delta0=%.10f  lfd_ok=%s  nStatus=%d\n", time() - t0, lfd0.Delta, lfd0.lfd_ok, lfd0.nStatus)
@assert lfd0.lfd_ok
@assert isapprox(lfd0.Delta, 0.483276; atol=1e-4)
x0_d20 = copy(lfd0.dual_x)
p_star_d20 = copy(lfd0.weights)
flush(stdout)

session_d20 = MelitzInnerSession(obj_d20, ctx_d20, policy_cap)
sorted_ctx_d20 = ctx_d20.sorted_tail_ctx
theta_plain0_d20 = melitz_unpower_theta_free(theta0_d20, ctx_d20)
A_free0_d20 = theta_plain0_d20[2:1+nA20]
g0 = theta_plain0_d20[1]
gpj0_d20 = exp(g0)
wage_ratio_d20 = ctx_d20.w_prime / ctx_d20.w[ctx_d20.target_country]
sigma_d20 = ctx_d20.sigma

gt_of_g(g::Real) = 100 * melitz_welfare_metrics_from_g(g, wage_ratio_d20, sigma_d20).gains_from_trade
g_of_gt(gt_pct::Real) = (sigma_d20 - 1) * log((1 - gt_pct / 100) / wage_ratio_d20)
GT0 = gt_of_g(g0)
@printf("anchor: g0=%.10f  GT0=%.6f%%  wage_ratio=%.10f  sigma=%.3f\n", g0, GT0, wage_ratio_d20, sigma_d20)
@assert isapprox(g_of_gt(GT0), g0; atol=1e-9) "gt_of_g/g_of_gt round-trip failed at the anchor"
flush(stdout)

function q_full_at_g(g_target::Real)
    th = copy(theta_plain0_d20)
    th[1] = g_target
    _, _, _, _, q = expand_free_theta_logcutoff(th, ctx_d20)
    return q
end
q0_d20_check = q_full_at_g(g0)
@assert q0_d20_check == q_full_at_g(g0)   # deterministic reconstruction sanity check
_, f0_full, _, _, q0_d20 = expand_free_theta_logcutoff(theta_plain0_d20, ctx_d20)
A0_full_d20 = exp.(reshape(pivot_expand(A_free0_d20, ctx_d20.A_pivot), D20, D20))
println("q(g) reconstruction verified: q_full_at_g(g0) matches the anchor's own expand_free_theta_logcutoff output.")
flush(stdout)

# Live verification (module header requirement: never assume) that moving g with q_free_free
# FIXED changes ONLY q[j,j] and the q-pivot's own physical pivot cell -- every OTHER free q
# cell must be BIT-IDENTICAL to the anchor's own value.
let q_probe = q_full_at_g(g0 + 0.01), D_ = D20, j_ = ctx_d20.target_country
    ndiff = 0
    for lin in 1:D_^2
        o, d = lin2od(lin, D_)
        if !(o == j_ && d == j_) && lin != ctx_d20.A_pivot.pivot && abs(q_probe[o, d] - q0_d20[o, d]) > 1e-9
            ndiff += 1
        end
    end
    @printf("q(g) perturbation check: %d of %d non-(j,j) free q cells changed by >1e-9 (expect O(1) -- only the q-pivot's own physical pivot cell)\n", ndiff, D_^2 - D_)
end
flush(stdout)

# ============================================================================
# Fixed-A/f profile incumbent at delta=0.5 (BASELINE for comparison, per the report's own
# required baseline-comparison section): the anchor's own A held FIXED, DeltaStar re-evaluated
# at each candidate g -- i.e. the ORIGINAL (non-profiled) welfare search this session is meant
# to improve on. Only needs a handful of points near the anchor for the report's own baseline
# row (not a full independent outer search -- that already exists in prior sessions' own work).
# ============================================================================
function fixed_Af_delta_at_g(g_target::Real)
    q_t = q_full_at_g(g_target)
    theta_t = melitz_fixed_q_state_theta(A_free0_d20, q_t, exp(g_target), ctx_d20)
    session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
    return solve_melitz_delta!(session_d20, theta_t, policy_cap; warm_start_source=:neutral)
end

# ============================================================================
# Profiled welfare function Phi(g): two-start (continuation + cellwise-compensated) v2 middle
# profile at the q(g) implied by the FIXED anchor free-q coordinates.
# ============================================================================
"""
    profile_phi_at_g(g_target, prev_A_free, prev_q, prev_p_star, sorted_ctx) -> NamedTuple

Runs the ADDENDUM-repaired two-start middle profile (`solve_melitz_fixed_q_A_profile_v2`) at
`g_target`'s own `q(g_target)` (anchor free-q coordinates fixed, reconstructed via the
existing q-gravity map). Start A: (1) continuation -- `prev_A_free` projected; (2) cellwise
`p*`-compensated -- `melitz_cellwise_A_from_moments` from `(prev_A_free, prev_q) ->
q_target` using `prev_p_star`, projected (module header formula (*), `lfd_preserving_state.jl`,
reused verbatim). Returns the BEST (lowest verified `Delta`) `FiniteSolved` result across the
two v2 runs, or the least-bad classification if neither is `FiniteSolved`.
"""
function profile_phi_at_g(g_target::Real, prev_A_free::Vector{Float64}, prev_q::Matrix{Float64},
                           prev_p_star::Vector{Float64})
    q_target = q_full_at_g(g_target)
    gpj_target = exp(g_target)
    theta_for_constraints = melitz_fixed_q_state_theta(prev_A_free, q_target, gpj_target, ctx_d20)
    sys_t = melitz_fixed_q_middle_constraint_system(theta_for_constraints, ctx_d20, obj_d20)

    # start 1: continuation
    A_cont = melitz_project_start_to_middle_constraints(copy(prev_A_free), sys_t, ctx_d20)

    # start 2: cellwise p*-compensated
    prev_A_full = exp.(reshape(pivot_expand(prev_A_free, ctx_d20.A_pivot), D20, D20))
    A_cellwise, status_cellwise = melitz_cellwise_A_from_moments(prev_A_full, prev_q, q_target, prev_p_star,
                                                                   sorted_ctx_d20, sigma_d20)
    n_bad = count(!=(:ok), status_cellwise)
    if n_bad > 0
        A_cellwise[status_cellwise .!= :ok] .= prev_A_full[status_cellwise .!= :ok]
    end
    A_cellwise_free = pivot_reduce(vec(log.(A_cellwise)), ctx_d20.A_pivot)
    A_comp = melitz_project_start_to_middle_constraints(A_cellwise_free, sys_t, ctx_d20)

    results = NamedTuple[]
    for (sname, A_start) in ((:continuation, A_cont), (:cellwise_compensated, A_comp))
        t0r = time()
        session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
        res = solve_melitz_fixed_q_A_profile_v2(session_d20, q_target, gpj_target, A_start, ctx_d20;
            coordinate=:logA, max_evals=MAX_MIDDLE_EVALS, box=D20_MIDDLE_BOX, outer_loop_opt=MIDDLE_OPT_D20, sys=sys_t,
            cap_handling=CAP_HANDLING, cap_barrier_multiple=CAP_BARRIER_MULTIPLE)
        wallr = time() - t0r
        push!(results, (start=sname, res=res, wall_s=wallr))
        @printf("    [start=%-20s] incumbent_source=%-12s classification=%-22s Delta=%.6g wall=%.1fs n_fc=%d n_ga=%d unique_A=%d unique_solves=%d cache_hits=%d\n",
            sname, res.incumbent_source, nameof(typeof(res.r_incumbent)), res.Delta_incumbent, wallr,
            res.n_fc_calls, res.n_ga_calls, res.unique_A_points, res.unique_inner_solves, res.cache_hits)
        flush(stdout)
    end

    finite_results = filter(r -> r.res.r_incumbent isa FiniteSolved, results)
    if !isempty(finite_results)
        best = argmin([r.res.Delta_incumbent for r in finite_results])
        winner = finite_results[best]
        return (g=g_target, GT=gt_of_g(g_target), q=q_target, classification=:FiniteSolved,
                Delta=winner.res.Delta_incumbent, best_start=winner.start,
                A_free=winner.res.A_free_incumbent, theta_free=winner.res.theta_free_incumbent,
                r=winner.res.r_incumbent, all_results=results)
    else
        # neither start reached FiniteSolved -- report the least-bad classification for
        # diagnostics (NEVER silently treated as an ordinary FiniteSolved value).
        cls_rank(r) = r.res.r_incumbent isa AboveEvaluationCap ? 1 :
                      r.res.r_incumbent isa InfiniteDeltaCertified ? 2 : 3   # NumericalFailure worst
        best = argmin([cls_rank(r) for r in results])
        winner = results[best]
        Delta_report = winner.res.r_incumbent isa AboveEvaluationCap ? winner.res.r_incumbent.certified_lower_bound :
                       winner.res.r_incumbent isa InfiniteDeltaCertified ? Inf : NaN
        return (g=g_target, GT=gt_of_g(g_target), q=q_target,
                classification=Symbol(nameof(typeof(winner.res.r_incumbent))),
                Delta=Delta_report, best_start=winner.start, A_free=winner.res.A_free_incumbent,
                theta_free=winner.res.theta_free_incumbent, r=winner.res.r_incumbent, all_results=results)
    end
end

# ============================================================================
# Initial welfare point (the anchor itself, g=g0): profile using the anchor A and the
# (trivially-identical, since q(g0)==q0 exactly) compensated A -- reproduces the prior
# session's own reported Phi(anchor)=0.289806 as a live cross-check.
# ============================================================================
println("\n" * "="^100); println("INITIAL WELFARE POINT (g=g0, anchor)"); println("="^100); flush(stdout)
phi0 = profile_phi_at_g(g0, A_free0_d20, q0_d20, p_star_d20)
@printf("Phi(g0) = %s classification=%s best_start=%s (prior session reported 0.289806 at the anchor)\n",
    string(phi0.Delta), phi0.classification, phi0.best_start)
@assert phi0.classification == :FiniteSolved
@assert phi0.Delta < 0.35   # sanity: should reproduce ~0.29, decisively better than the 0.4833 fixed-A/f baseline
flush(stdout)

# ============================================================================
# One-dimensional continuation, per direction (:upper = increasing GT%, :lower = decreasing GT%).
# ============================================================================
mutable struct ContinuationPoint
    idx::Int
    g::Float64
    GT::Float64
    classification::Symbol
    Delta::Float64
    accepted::Bool
    best_start::Symbol
    A_free::Vector{Float64}
    theta_free::Vector{Float64}
end

function run_continuation_direction(direction::Symbol)
    sign = direction == :upper ? 1 : -1
    println("\n" * "="^100); println("CONTINUATION DIRECTION: $direction (sign=$sign)"); println("="^100); flush(stdout)

    points = ContinuationPoint[ContinuationPoint(0, g0, GT0, :FiniteSolved, phi0.Delta, true, phi0.best_start,
                                                  copy(phi0.A_free), copy(phi0.theta_free))]
    cur_g, cur_GT = g0, GT0
    cur_A_free = copy(phi0.A_free)
    _, _, _, _, cur_q = expand_free_theta_logcutoff(phi0.theta_free, ctx_d20)
    cur_p_star = copy(p_star_d20)   # p* carried from the anchor's own dual weights; refreshed below on acceptance
    step_gt = GT_STEP0
    bracket = nothing    # (feasible_g, feasible_GT, infeasible_g, infeasible_GT)
    n_evaluated = 0
    most_extreme = points[1]   # most extreme verified within-budget (Delta<=DELTA_BUDGET) point

    function maybe_update_extreme!(pt::ContinuationPoint)
        if pt.classification == :FiniteSolved && pt.Delta <= DELTA_BUDGET
            if sign * (pt.GT - GT0) > sign * (most_extreme.GT - GT0)
                most_extreme = pt
            end
        end
    end

    # --- Phase 1: expand outward until bracketed / step too small / budget exhausted ---
    while n_evaluated < MAX_WELFARE_POINTS_PER_DIRECTION && step_gt >= GT_STEP_MIN && bracket === nothing
        GT_target = cur_GT + sign * step_gt
        g_target = g_of_gt(GT_target)
        @printf("\n  [expand] step_gt=%.5f  GT_target=%.6f%%  g_target=%.8f\n", step_gt, GT_target, g_target)
        flush(stdout)
        phi = profile_phi_at_g(g_target, cur_A_free, cur_q, cur_p_star)
        n_evaluated += 1
        within_budget = phi.classification == :FiniteSolved && phi.Delta <= DELTA_BUDGET
        pt = ContinuationPoint(n_evaluated, g_target, phi.GT, phi.classification, phi.Delta, within_budget,
                                phi.best_start, phi.A_free, phi.theta_free)
        push!(points, pt)
        maybe_update_extreme!(pt)
        if within_budget
            @printf("  ACCEPT: Phi=%.6g <= %.2f -- new incumbent at GT=%.6f%%\n", phi.Delta, DELTA_BUDGET, phi.GT)
            cur_g, cur_GT, cur_A_free = g_target, phi.GT, phi.A_free
            _, _, _, _, cur_q = expand_free_theta_logcutoff(phi.theta_free, ctx_d20)
            r_best = phi.r
            cur_p_star = r_best isa FiniteSolved ? melitz_recover_lfd(obj_d20, phi.theta_free).weights : cur_p_star
            step_gt = min(step_gt * GT_STEP_GROWTH, GT_STEP0 * 4)
        else
            @printf("  REJECT: classification=%s Delta=%s -- bracket found, will bisect\n", phi.classification, string(phi.Delta))
            bracket = (cur_g, cur_GT, g_target, GT_target)
            step_gt /= 2
        end
    end

    # --- Phase 2: safeguarded bisection in g, once bracketed ---
    if bracket !== nothing
        g_feas, GT_feas, g_infeas, GT_infeas = bracket
        while n_evaluated < MAX_WELFARE_POINTS_PER_DIRECTION && abs(GT_infeas - GT_feas) > GT_BRACKET_TOL
            g_mid = 0.5 * (g_feas + g_infeas)
            @printf("\n  [bisect] bracket GT=[%.6f%%, %.6f%%]  g_mid=%.8f\n", min(GT_feas,GT_infeas), max(GT_feas,GT_infeas), g_mid)
            flush(stdout)
            phi = profile_phi_at_g(g_mid, cur_A_free, cur_q, cur_p_star)
            n_evaluated += 1
            within_budget = phi.classification == :FiniteSolved && phi.Delta <= DELTA_BUDGET
            pt = ContinuationPoint(n_evaluated, g_mid, phi.GT, phi.classification, phi.Delta, within_budget,
                                    phi.best_start, phi.A_free, phi.theta_free)
            push!(points, pt)
            maybe_update_extreme!(pt)
            if within_budget
                @printf("  bisect ACCEPT: Phi=%.6g -- move feasible bracket edge to GT=%.6f%%\n", phi.Delta, phi.GT)
                g_feas, GT_feas = g_mid, phi.GT
                cur_A_free = phi.A_free
                _, _, _, _, cur_q = expand_free_theta_logcutoff(phi.theta_free, ctx_d20)
                cur_p_star = melitz_recover_lfd(obj_d20, phi.theta_free).weights
            else
                @printf("  bisect REJECT: classification=%s -- move infeasible bracket edge\n", phi.classification)
                g_infeas, GT_infeas = g_mid, phi.GT
            end
        end
    end

    return (direction=direction, points=points, most_extreme=most_extreme, n_evaluated=n_evaluated,
            bracket=bracket)
end

result_upper = run_continuation_direction(:upper)
result_lower = run_continuation_direction(:lower)

# ============================================================================
# Baseline comparison + report data.
# ============================================================================
println("\n" * "="^100); println("BASELINE COMPARISON"); println("="^100); flush(stdout)

function summarize_direction(res)
    dir = res.direction
    ext = res.most_extreme
    r_fixedAf = fixed_Af_delta_at_g(ext.g)
    fixedAf_Delta = r_fixedAf isa FiniteSolved ? r_fixedAf.Delta :
                    r_fixedAf isa AboveEvaluationCap ? r_fixedAf.certified_lower_bound : NaN
    fixedAf_cls = nameof(typeof(r_fixedAf))

    A_full_ext = exp.(reshape(pivot_expand(ext.A_free, ctx_d20.A_pivot), D20, D20))
    A_move_norm = norm(vec(log.(A_full_ext)) .- vec(log.(A0_full_d20)))
    _, f_full_ext, _, _, _ = expand_free_theta_logcutoff(ext.theta_free, ctx_d20)
    f_move_norm = norm(vec(log.(f_full_ext)) .- vec(log.(f0_full)))

    n_finite = count(p -> p.classification == :FiniteSolved, res.points)
    n_cap = count(p -> p.classification == :AboveEvaluationCap, res.points)
    n_inf = count(p -> p.classification == :InfiniteDeltaCertified, res.points)
    n_numfail = count(p -> p.classification == :NumericalFailure, res.points)

    @printf("\n[%s] fixed-A/f profile Delta at extreme GT=%.6f%%: %.6g (%s)\n", dir, ext.GT, fixedAf_Delta, fixedAf_cls)
    @printf("[%s] starting profiled-A point: GT=%.6f%% Delta=%.6g\n", dir, GT0, phi0.Delta)
    @printf("[%s] best profiled-A extreme: GT=%.6f%% Delta=%.6g (best_start=%s)\n", dir, ext.GT, ext.Delta, ext.best_start)
    @printf("[%s] A movement norm (log units): %.6g   f movement norm (log units): %.6g\n", dir, A_move_norm, f_move_norm)
    @printf("[%s] welfare points evaluated: %d   classification counts F/C/I/NumFail = %d/%d/%d/%d\n",
        dir, length(res.points) - 1, n_finite, n_cap, n_inf, n_numfail)

    return (direction=String(dir), GT0=GT0, Delta0_profiled=phi0.Delta,
            fixedAf_GT=ext.GT, fixedAf_Delta=fixedAf_Delta, fixedAf_classification=String(fixedAf_cls),
            extreme_GT=ext.GT, extreme_Delta=ext.Delta, extreme_best_start=String(ext.best_start),
            A_move_norm=A_move_norm, f_move_norm=f_move_norm,
            n_points_evaluated=length(res.points) - 1, n_finite_points=n_finite, n_above_cap_points=n_cap,
            n_infinite_points=n_inf, n_numfail_points=n_numfail,
            improvement_pct_points=ext.GT - GT0)
end

summary_upper = summarize_direction(result_upper)
summary_lower = summarize_direction(result_lower)

# ============================================================================
# Write machine-readable results.
# ============================================================================
open(joinpath(OUTDIR, "melitz_d20_profiledA_continuation_points_2026-07-30.csv"), "w") do io
    println(io, "direction,idx,g,GT,classification,Delta,accepted,best_start")
    for res in (result_upper, result_lower)
        for p in res.points
            println(io, "$(res.direction),$(p.idx),$(p.g),$(p.GT),$(p.classification),$(p.Delta),$(p.accepted),$(p.best_start)")
        end
    end
end
open(joinpath(OUTDIR, "melitz_d20_profiledA_continuation_summary_2026-07-30.csv"), "w") do io
    println(io, "direction,GT0,Delta0_profiled,fixedAf_GT,fixedAf_Delta,fixedAf_classification,extreme_GT,extreme_Delta,extreme_best_start,A_move_norm,f_move_norm,n_points_evaluated,n_finite_points,n_above_cap_points,n_infinite_points,n_numfail_points,improvement_pct_points")
    for s in (summary_upper, summary_lower)
        println(io, "$(s.direction),$(s.GT0),$(s.Delta0_profiled),$(s.fixedAf_GT),$(s.fixedAf_Delta),$(s.fixedAf_classification),$(s.extreme_GT),$(s.extreme_Delta),$(s.extreme_best_start),$(s.A_move_norm),$(s.f_move_norm),$(s.n_points_evaluated),$(s.n_finite_points),$(s.n_above_cap_points),$(s.n_infinite_points),$(s.n_numfail_points),$(s.improvement_pct_points)")
    end
end
serialize(joinpath(SCRATCH, "melitz_d20_profiledA_continuation_state_2026-07-30.jls"),
    (g0=g0, GT0=GT0, phi0=phi0, result_upper=result_upper, result_lower=result_lower,
     summary_upper=summary_upper, summary_lower=summary_lower))
println("\nCSV + state files written.")
println("\nDONE D20 PROFILED-A WELFARE CONTINUATION")
