# delta_star_path_bisection.jl -- safeguarded 1-D bisection along a linear path in the live
# powered-A outer coordinates between two ALREADY-VERIFIED family points, searching for a point
# whose own Delta* is close to a target value. Built for TARGETED_K3_EXTENSIONS_2026-08-04 Section
# 5.2 (unrestricted upper delta=0.01 alternate boundary seed) and Section 6 S3 (common-Frechet
# delta=1 backtracked-from-delta=2 candidate).
#
# Delta*(w) is evaluated the ONLY correct way available in this codebase: a real inner-problem
# solve at that fixed outer point, via the existing production run_fn with a SHORT outer budget
# (the outer loop barely gets to move, so the returned best_feasible.Delta is, to within that short
# budget's own outer movement, Delta* at essentially the seed point itself -- this does not
# reimplement or approximate the inner KNITRO solve, it just gives the outer loop almost no time to
# leave the seed). Per CLAUDE.md, this does NOT assume linearity in the path parameter t --
# `bisect_delta_star` explicitly re-evaluates Delta* at each candidate t and only narrows the
# bracket if the sign of (Delta*(t) - target) is consistent with a monotone bracket; it gives up
# (returns the best point found, not a guessed one) if the bracket is not monotone within
# `max_iters`, rather than extrapolating.
#
# Usage: julia --project=. delta_star_path_bisection.jl <family> <direction> <lo_report_jls>
#   <hi_report_jls> <target_delta_star> <eval_budget_s> <output_report_jls> <max_iters=12>
# Requires the full driver chain + continuation_polish_orchestrator.jl + continuation_polish_run_fn.jl
# already include-d (same convention as continuation_campaign_cell_driver.jl).

lp(xs...) = (println(xs...); flush(stdout))

function path_point(w_lo::Vector{Float64}, w_hi::Vector{Float64}, t::Float64)
    length(w_lo) == length(w_hi) ||
        error("path_point: endpoint length mismatch ($(length(w_lo)) vs $(length(w_hi))) -- not the same family/layout")
    return (1 - t) .* w_lo .+ t .* w_hi
end

"""
    eval_delta_star(family, direction, w, budget_s, ckpt_dir) -> (Delta_star, GT, w_out)

One short production_run_fn call; returns the resulting best_feasible's Delta*/GT/w. Errors loudly
(does not silently return NaN) if no feasible point is found at all -- a short budget failing to
find ANY feasible point at a candidate t is itself useful information, not something to paper over.
"""
function eval_delta_star(family::String, direction::Symbol, w::Vector{Float64}, budget_s::Float64, ckpt_dir::String;
                          eval_delta_ceiling::Float64 = 10.0)
    # production_run_fn reads its own `delta_in`/`delta` kwarg from the ACTIVE_TARGET_DELTA[] Ref
    # (continuation_polish_run_fn.jl), NOT from an argument of this function -- the campaign cell
    # driver sets this Ref itself before calling run_target_cell!. This module has no target delta
    # of its own (it's evaluating a point's OWN Delta*, not solving to a caller-given budget), so it
    # sets a generous, non-binding ceiling here so the outer NLP's Delta<=delta_in constraint never
    # actually binds -- with a short budget_s the outer loop barely moves, so the returned
    # best_feasible.Delta is (to within that short budget's own outer movement) the seed's own
    # Delta*. Leaving this Ref at its default NaN (as this module originally did, 2026-08-04) is a
    # real bug, not a benign default: KNITRO's KN_add_eval_callback fails outright ("upper bound
    # specified for constraint index 0 is undefined", code -515) because the Delta<=delta_in
    # constraint's own upper bound is NaN -- confirmed live by the first version of this script.
    ACTIVE_TARGET_DELTA[] = eval_delta_ceiling
    raw = production_run_fn(family, w, EXPLORE_DIRECT_SR1, budget_s, ckpt_dir; find_smallest = direction_is_upper(direction))
    cand = normalize_result(family, raw)
    cand.w === nothing && error("eval_delta_star: no feasible point found at this candidate t within budget_s=$budget_s -- widen budget, do not guess.")
    return (cand.Delta_star, cand.GT, cand.w)
end

"""
    bisect_delta_star(family, direction, w_lo, w_hi, Ds_lo, Ds_hi, target, budget_s, ckpt_dir; max_iters)

`Ds_lo`/`Ds_hi` are the ALREADY-KNOWN verified Delta* of the two endpoints (no need to re-evaluate
t=0/t=1). Requires target to lie between Ds_lo and Ds_hi (monotone bracket assumption checked, not
asserted) -- if it doesn't bracket, returns the closer endpoint untouched rather than extrapolating.
"""
function bisect_delta_star(family::String, direction::Symbol, w_lo::Vector{Float64}, w_hi::Vector{Float64},
                            Ds_lo::Float64, Ds_hi::Float64, target::Float64, budget_s::Float64, ckpt_dir::String;
                            max_iters::Int = 12, tol::Float64 = 0.002)
    if !((Ds_lo <= target <= Ds_hi) || (Ds_hi <= target <= Ds_lo))
        lp("bisect_delta_star: target=", target, " NOT bracketed by Ds_lo=", Ds_lo, "/Ds_hi=", Ds_hi,
           " -- returning the closer endpoint verbatim, not extrapolating.")
        return abs(Ds_lo - target) <= abs(Ds_hi - target) ? (0.0, Ds_lo, w_lo) : (1.0, Ds_hi, w_hi)
    end
    t_lo, t_hi = 0.0, 1.0
    best = abs(Ds_lo - target) <= abs(Ds_hi - target) ? (t_lo, Ds_lo, w_lo) : (t_hi, Ds_hi, w_hi)
    d_lo, d_hi = Ds_lo, Ds_hi
    for i in 1:max_iters
        t_mid = 0.5 * (t_lo + t_hi)
        w_mid = path_point(w_lo, w_hi, t_mid)
        Ds_mid, GT_mid, w_verified = eval_delta_star(family, direction, w_mid, budget_s,
                                                       joinpath(ckpt_dir, "bisect_iter$(i)"))
        lp("  iter=", i, " t=", round(t_mid, digits = 4), " Delta*=", Ds_mid, " GT=", GT_mid)
        if abs(Ds_mid - target) < abs(best[2] - target)
            best = (t_mid, Ds_mid, w_verified)
        end
        abs(Ds_mid - target) <= tol && return best
        # Only narrow the bracket if this preserves monotone sign-bracketing (does NOT assume
        # linearity beyond that -- if Delta* is not monotone in t between the current bracket, we
        # stop narrowing and just keep the best point found so far over the remaining iterations).
        if (d_lo <= target <= Ds_mid) || (Ds_mid <= target <= d_lo)
            t_hi, d_hi = t_mid, Ds_mid
        elseif (Ds_mid <= target <= d_hi) || (d_hi <= target <= Ds_mid)
            t_lo, d_lo = t_mid, Ds_mid
        else
            lp("  non-monotone at iter=", i, " -- bracket narrowing stopped; continuing to sample midpoints only for best-so-far.")
            t_lo, t_hi = max(0.0, t_mid - 0.1), min(1.0, t_mid + 0.1)
        end
    end
    return best
end

if abspath(PROGRAM_FILE) == @__FILE__
    family = ARGS[1]
    direction = Symbol(ARGS[2])
    lo_path, hi_path = ARGS[3], ARGS[4]
    target = parse(Float64, ARGS[5])
    budget_s = parse(Float64, ARGS[6])
    out_path = ARGS[7]
    max_iters = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 12

    state_lo = load_run_state(lo_path); r_lo = state_lo.report
    state_hi = load_run_state(hi_path); r_hi = state_hi.report
    lp("Endpoint lo: delta=", r_lo.target_delta, " GT=", r_lo.final_GT, " Delta*=", r_lo.final_Delta_star)
    lp("Endpoint hi: delta=", r_hi.target_delta, " GT=", r_hi.final_GT, " Delta*=", r_hi.final_Delta_star)

    ckpt_root = dirname(out_path)
    isdir(ckpt_root) || mkpath(ckpt_root)
    t_star, Ds_star, w_star = bisect_delta_star(family, direction, r_lo.final_w, r_hi.final_w,
        r_lo.final_Delta_star, r_hi.final_Delta_star, target, budget_s, ckpt_root; max_iters = max_iters)

    lp("=== BISECTION RESULT: t*=", t_star, " Delta*=", Ds_star, " (target=", target, ") ===")
    result = (t_star = t_star, Delta_star = Ds_star, w = w_star, target = target,
              lo_path = lo_path, hi_path = hi_path, family = family, direction = direction)
    serialize(out_path, result)
    lp("Saved bisection result to ", out_path)
end
