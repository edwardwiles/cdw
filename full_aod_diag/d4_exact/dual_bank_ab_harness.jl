# ============================================================================
# Successful-dual-bank A/B harness (task §12).
#
# Two things distinguish this from a black-box outcome-only comparison:
#  1. `DualBankABStats`/`record_ab_selection!` capture the PER-CALL selection label
#     (`:actual`/`:last_accepted`/`:nearest`/`:neutral`) and candidate count that
#     `screened_eval` (c10_d20_production_driver.jl, NOT a frozen file -- this driver
#     owns it) already computes via `select_warm_start` and previously discarded. This
#     is additive instrumentation of driver code, NOT a change to dual_bank.jl's own
#     scoring/selection logic (frozen, task §2/§13) -- `select_warm_start` itself is
#     called exactly as before, with the exact same arguments and return value; only the
#     already-computed return value is now also recorded.
#  2. Fixed-trajectory replay (task §12's own stated preference, "removes outer-path
#     noise"): both arms replay the IDENTICAL sequence of outer w-points through
#     `screened_eval` directly, rather than letting two separate live KNITRO outer
#     solves potentially diverge onto different trajectories because of a warm-start
#     difference feeding back into the (quasi-Newton) outer Hessian approximation --
#     exactly the hybrid-gradient-source-switching hazard this repo's own memory notes
#     warn about for cross-iterate solver-state mixing.
# ============================================================================

mutable struct DualBankABStats
    n_calls::Int
    n_candidates::Vector{Int}
    labels::Vector{Symbol}
    scoring_wall::Vector{Float64}
end
DualBankABStats() = DualBankABStats(0, Int[], Symbol[], Float64[])

function record_ab_selection!(stats::DualBankABStats, label::Symbol, n_candidates::Int, scoring_wall::Float64)
    stats.n_calls += 1
    push!(stats.n_candidates, n_candidates)
    push!(stats.labels, label)
    push!(stats.scoring_wall, scoring_wall)
    return stats
end

function summarize_ab_stats(stats::DualBankABStats)
    n = stats.n_calls
    n == 0 && return (n_calls = 0, label_counts = Dict{Symbol,Int}(), mean_n_candidates = NaN, total_scoring_wall = 0.0)
    counts = Dict{Symbol,Int}()
    for l in stats.labels
        counts[l] = get(counts, l, 0) + 1
    end
    return (n_calls = n, label_counts = counts, mean_n_candidates = sum(stats.n_candidates) / n,
            total_scoring_wall = sum(stats.scoring_wall))
end

"""
    dual_bank_ab_trajectory(ctx, pe, rsc, w_trajectory; dual_bank_size=8) -> NamedTuple

Fixed-trajectory A/B (task §12): replays the identical sequence of `w = [gp; zfree]`
outer points in `w_trajectory` through `screened_eval` twice on the SAME immutable
(ctx, pe, rsc) -- once per the task's Arm A (exact-point cache on, dual bank OFF), once
per Arm B (exact-point cache on, dual bank ON, production scoring rule via
`select_warm_start`). Each arm gets its OWN fresh `SafeExactCache` and starts from the
SAME cold (`ctx.obj.x .= NaN`, forcing the first point's own neutral/cold start in both
arms) dual state, so results reflect each arm's own mechanism rather than one arm
benefiting from cache/warm-state the other arm already built up.
"""
function dual_bank_ab_trajectory(ctx, pe, rsc, w_trajectory::Vector{<:AbstractVector{<:Real}};
        dual_bank_size::Int = 8)
    function run_arm(use_bank::Bool)
        ctx.obj.x .= NaN   # force a cold/neutral start for the first point, same in both arms
        sc = ScreenCounters(); n_eval = Ref(0)
        bank = use_bank ? DualBank(dual_bank_size) : nothing
        exact_cache = SafeExactCache()
        ab_stats = use_bank ? DualBankABStats() : nothing
        statuses = Int[]; wall_per_point = Float64[]
        t0 = time()
        for w in w_trajectory
            xf = x_free_from_w(w, pe)
            t_pt0 = time()
            r, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = true, bank = bank, zfree = w[2:end],
                exact_cache = exact_cache, ab_stats = ab_stats)
            push!(wall_per_point, time() - t_pt0)
            push!(statuses, r.inner_status)
        end
        t_total = time() - t0
        return (t_total = t_total, wall_per_point = wall_per_point, statuses = statuses,
                n_feasible = count(s -> s in FEASIBLE_CODES, statuses),
                n_organic_failure = count(is_organic_failure, statuses),
                exact_cache_final_size = length(exact_cache),
                bank_final_size = bank === nothing ? 0 : length(bank.history),
                screen_counts = as_namedtuple(sc),
                ab_summary = ab_stats === nothing ? nothing : summarize_ab_stats(ab_stats))
    end
    arm_A = run_arm(false)   # exact cache ON (always fresh per arm), dual bank OFF
    arm_B = run_arm(true)    # exact cache ON, dual bank ON

    same_statuses = arm_A.statuses == arm_B.statuses
    same_n_feasible = arm_A.n_feasible == arm_B.n_feasible
    harmful_points = same_statuses ? Int[] :
        [i for i in 1:min(length(arm_A.statuses), length(arm_B.statuses))
             if arm_A.statuses[i] in FEASIBLE_CODES && !(arm_B.statuses[i] in FEASIBLE_CODES)]
    beneficial_points = same_statuses ? Int[] :
        [i for i in 1:min(length(arm_A.statuses), length(arm_B.statuses))
             if !(arm_A.statuses[i] in FEASIBLE_CODES) && arm_B.statuses[i] in FEASIBLE_CODES]

    return (arm_A_bank_off = arm_A, arm_B_bank_on = arm_B,
            n_points = length(w_trajectory),
            wall_savings = arm_A.t_total - arm_B.t_total,
            wall_savings_pct = arm_A.t_total == 0 ? NaN : 100 * (arm_A.t_total - arm_B.t_total) / arm_A.t_total,
            same_statuses_at_every_point = same_statuses, same_n_feasible = same_n_feasible,
            harmful_points = harmful_points, beneficial_points = beneficial_points)
end
