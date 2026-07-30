# Phase 2 row-worker: for a contiguous slice of the precomputed upper-direction row states
# (scripts/melitz_phase2_row_states_2026-07-30.jls, written by
# melitz_phase2_precompute_states_2026-07-30.jl), runs BOTH (a) the ORIGINAL always-two-start
# policy (mirroring profile_phi_at_g's own two solve_melitz_fixed_q_A_profile_v2 calls exactly)
# and (b) the new adaptive policy (melitz_middle_two_start_adaptive!) at the SAME
# (g_target, prev_A_free, prev_q, prev_p_star) input state for every row, and records a
# head-to-head comparison (Delta, middle-solve count, unique inner solves, wall time) for each.
#
# Usage: julia --project=. -t <threads> melitz_phase2_row_worker_2026-07-30.jl <row_start> <row_end> <out_suffix>
# (row_start/row_end are inclusive `idx` values into the stored upper-direction CSV, 1..13.)
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Serialization
LinearAlgebra.BLAS.set_num_threads(1)
melitz_thread_startup_report()
println("Julia threads: ", Threads.nthreads()); flush(stdout)

row_start = parse(Int, ARGS[1])
row_end = parse(Int, ARGS[2])
out_suffix = ARGS[3]
println("Row worker: idx ", row_start, "..", row_end, "  out_suffix=", out_suffix); flush(stdout)

const OUTDIR = joinpath(REPO2, "docs", "key_results")
CAP = 10.0
policy_cap = CappedEvaluation(CAP)
MIDDLE_OPT_D20 = joinpath(REPO2, "melitz_middle_loop_opt_2026-07-30.opt")
const D20_MIDDLE_BOX = 0.1
const MAX_MIDDLE_EVALS = 120
const DELTA_BUDGET = 0.5
const CAP_HANDLING = :barrier
const CAP_BARRIER_MULTIPLE = 5.0

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
sorted_ctx_d20 = ctx_d20.sorted_tail_ctx
sigma_d20 = ctx_d20.sigma

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
loaded = deserialize(joinpath(@__DIR__, "melitz_phase2_row_states_2026-07-30.jls"))
row_states = loaded.row_states

function build_starts(rs::RowInputState)
    gpj_target = exp(rs.g_target)
    q_target = q_full_at_g(rs.g_target)
    theta_for_constraints = melitz_fixed_q_state_theta(rs.prev_A_free, q_target, gpj_target, ctx_d20)
    sys_t = melitz_fixed_q_middle_constraint_system(theta_for_constraints, ctx_d20, obj_d20)
    A_cont = melitz_project_start_to_middle_constraints(copy(rs.prev_A_free), sys_t, ctx_d20)
    prev_A_full = exp.(reshape(pivot_expand(rs.prev_A_free, ctx_d20.A_pivot), D20, D20))
    A_cellwise, status_cellwise = melitz_cellwise_A_from_moments(prev_A_full, rs.prev_q, q_target, rs.prev_p_star,
                                                                   sorted_ctx_d20, sigma_d20)
    n_bad = count(!=(:ok), status_cellwise)
    if n_bad > 0
        A_cellwise[status_cellwise .!= :ok] .= prev_A_full[status_cellwise .!= :ok]
    end
    A_cellwise_free = pivot_reduce(vec(log.(A_cellwise)), ctx_d20.A_pivot)
    A_comp = melitz_project_start_to_middle_constraints(A_cellwise_free, sys_t, ctx_d20)
    return q_target, gpj_target, A_cont, A_comp, sys_t
end

# q(g) reconstruction: identical recipe to the continuation script -- moving g with the anchor's
# own free-q coordinates fixed changes only q[j,j] and the q-pivot's own physical pivot cell.
theta_plain0_d20 = melitz_unpower_theta_free(
    let d = Dict{Tuple{String,Float64},Vector{Float64}}();
        for line in eachline(joinpath(OUTDIR, "melitz_qbw_phase3_theta_q_2026-07-29.csv"))
            p = split(line, ","); d[(p[1], parse(Float64, p[2]))] = parse.(Float64, p[5:end])
        end
        d[("realD20_seed1_W80000", 0.5)]
    end, ctx_d20)
function q_full_at_g(g_target::Real)
    th = copy(theta_plain0_d20)
    th[1] = g_target
    _, _, _, _, q = expand_free_theta_logcutoff(th, ctx_d20)
    return q
end

results = NamedTuple[]
for rs in row_states
    rs.idx < row_start && continue
    rs.idx > row_end && continue
    println("\n", "="^80); println("ROW idx=", rs.idx, "  GT_target=", rs.stored_GT); flush(stdout)
    q_target, gpj_target, A_cont, A_comp, sys_t = build_starts(rs)

    # --- (a) ORIGINAL always-two-start policy ---
    t0 = time()
    session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
    r_cont_base = solve_melitz_fixed_q_A_profile_v2(session_d20, q_target, gpj_target, A_cont, ctx_d20;
        coordinate=:logA, max_evals=MAX_MIDDLE_EVALS, box=D20_MIDDLE_BOX, outer_loop_opt=MIDDLE_OPT_D20, sys=sys_t,
        cap_handling=CAP_HANDLING, cap_barrier_multiple=CAP_BARRIER_MULTIPLE)
    session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
    r_comp_base = solve_melitz_fixed_q_A_profile_v2(session_d20, q_target, gpj_target, A_comp, ctx_d20;
        coordinate=:logA, max_evals=MAX_MIDDLE_EVALS, box=D20_MIDDLE_BOX, outer_loop_opt=MIDDLE_OPT_D20, sys=sys_t,
        cap_handling=CAP_HANDLING, cap_barrier_multiple=CAP_BARRIER_MULTIPLE)
    wall_base = time() - t0
    cont_fin = r_cont_base.r_incumbent isa FiniteSolved
    comp_fin = r_comp_base.r_incumbent isa FiniteSolved
    if cont_fin && comp_fin
        Delta_base = min(r_cont_base.Delta_incumbent, r_comp_base.Delta_incumbent)
        base_best = r_cont_base.Delta_incumbent <= r_comp_base.Delta_incumbent ? :continuation : :cellwise_compensated
    elseif cont_fin
        Delta_base, base_best = r_cont_base.Delta_incumbent, :continuation
    elseif comp_fin
        Delta_base, base_best = r_comp_base.Delta_incumbent, :cellwise_compensated
    else
        Delta_base, base_best = NaN, :none
    end
    base_finite = cont_fin || comp_fin
    base_middle_solves = 2
    base_unique_inner_solves = r_cont_base.unique_inner_solves + r_comp_base.unique_inner_solves
    base_fc_ga = (r_cont_base.n_fc_calls + r_cont_base.n_ga_calls) + (r_comp_base.n_fc_calls + r_comp_base.n_ga_calls)

    @printf("  BASELINE (always two-start): Delta=%.6g best=%s wall=%.1fs unique_solves=%d fc+ga=%d\n",
        Delta_base, base_best, wall_base, base_unique_inner_solves, base_fc_ga)
    flush(stdout)

    # --- (b) ADAPTIVE policy ---
    t1 = time()
    adapt = melitz_middle_two_start_adaptive!(session_d20, q_target, gpj_target, A_cont, A_comp, ctx_d20;
        delta_budget=DELTA_BUDGET, sys=sys_t, coordinate=:logA, max_evals=MAX_MIDDLE_EVALS, box=D20_MIDDLE_BOX,
        outer_loop_opt=MIDDLE_OPT_D20, cap_handling=CAP_HANDLING, cap_barrier_multiple=CAP_BARRIER_MULTIPLE)
    wall_adapt = time() - t1
    adapt_finite = adapt.r_incumbent isa FiniteSolved
    adapt_middle_solves = adapt.ran_compensated ? 2 : 1
    adapt_unique_inner_solves = adapt.r_continuation.unique_inner_solves +
        (adapt.ran_compensated ? adapt.r_compensated.unique_inner_solves : 0)
    adapt_fc_ga = (adapt.r_continuation.n_fc_calls + adapt.r_continuation.n_ga_calls) +
        (adapt.ran_compensated ? (adapt.r_compensated.n_fc_calls + adapt.r_compensated.n_ga_calls) : 0)

    @printf("  ADAPTIVE: Delta=%.6g best=%s trigger=%s ran_compensated=%s wall=%.1fs unique_solves=%d fc+ga=%d\n",
        adapt.Delta, adapt.best_source, adapt.trigger_reason, adapt.ran_compensated, wall_adapt,
        adapt_unique_inner_solves, adapt_fc_ga)
    flush(stdout)

    within_budget_base = base_finite && Delta_base <= DELTA_BUDGET
    within_budget_adapt = adapt_finite && adapt.Delta <= DELTA_BUDGET
    same_budget_status = within_budget_base == within_budget_adapt
    delta_agree = base_finite && adapt_finite ? abs(Delta_base - adapt.Delta) : NaN

    push!(results, (idx=rs.idx, GT=rs.stored_GT, stored_Delta=rs.stored_Delta, stored_best_start=rs.stored_best_start,
        Delta_base=Delta_base, base_best=String(base_best), base_finite=base_finite, wall_base=wall_base,
        base_middle_solves=base_middle_solves, base_unique_inner_solves=base_unique_inner_solves, base_fc_ga=base_fc_ga,
        Delta_adapt=adapt.Delta, adapt_best=String(adapt.best_source), adapt_finite=adapt_finite,
        trigger_reason=String(adapt.trigger_reason), ran_compensated=adapt.ran_compensated, wall_adapt=wall_adapt,
        adapt_middle_solves=adapt_middle_solves, adapt_unique_inner_solves=adapt_unique_inner_solves, adapt_fc_ga=adapt_fc_ga,
        within_budget_base=within_budget_base, within_budget_adapt=within_budget_adapt,
        same_budget_status=same_budget_status, delta_agree_abs=delta_agree))
end

outpath = joinpath(OUTDIR, "melitz_phase2_adaptive_vs_twostart_$(out_suffix)_2026-07-30.csv")
open(outpath, "w") do io
    println(io, "idx,GT,stored_Delta,stored_best_start,Delta_base,base_best,base_finite,wall_base,base_middle_solves,base_unique_inner_solves,base_fc_ga,Delta_adapt,adapt_best,adapt_finite,trigger_reason,ran_compensated,wall_adapt,adapt_middle_solves,adapt_unique_inner_solves,adapt_fc_ga,within_budget_base,within_budget_adapt,same_budget_status,delta_agree_abs")
    for r in results
        println(io, join([r.idx, r.GT, r.stored_Delta, r.stored_best_start, r.Delta_base, r.base_best, r.base_finite,
            r.wall_base, r.base_middle_solves, r.base_unique_inner_solves, r.base_fc_ga, r.Delta_adapt, r.adapt_best,
            r.adapt_finite, r.trigger_reason, r.ran_compensated, r.wall_adapt, r.adapt_middle_solves,
            r.adapt_unique_inner_solves, r.adapt_fc_ga, r.within_budget_base, r.within_budget_adapt,
            r.same_budget_status, r.delta_agree_abs], ","))
    end
end
println("\nWrote ", outpath)
println("ROW WORKER (", out_suffix, ") COMPLETE")
