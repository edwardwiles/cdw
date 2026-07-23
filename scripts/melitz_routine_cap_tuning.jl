# Phase I.7 (screening-session continuation): empirical successful-inner-solve time/
# iteration distribution, and candidate routine :budget_check caps tested against it.
#
# Methodology: `on_inner_result` (finite_delta_outer.jl/inner_screening.jl) fires exactly
# once per ATTEMPTED inner solve (screen rejection OR a real KNITRO call) -- never on an
# exact-point cache hit. For a REAL KNITRO attempt, `CounterfactualSensitivity.INNER_ITERS_TOTAL[]` (cc_algo,
# accumulated across the whole process) advances by exactly that call's own iteration count
# (screen rejections never touch it, so the delta between two consecutive `on_result` calls
# correctly attributes iterations to whichever one just ran a real solve). This gives a
# per-call (elapsed_s, iters) pair for every attempted inner solve without any cc_algo
# changes.
#
# Usage: julia --project=. scripts/melitz_routine_cap_tuning.jl

using Printf, Statistics
include(joinpath(@__DIR__, "melitz_finite_delta_campaign.jl"))

const MAXIT_CANDIDATES = [50, 100, 150, 250]
const ELAPSED_CANDIDATES = [0.10, 0.25, 0.50, 1.00]

function collect_successful_solve_distribution(; D=4, W=20_000, seed=29,
                                                  deltas=[1e-3, 1e-2], directions=(:upper, :lower),
                                                  theta_box=0.10, cutoff_constraint_backend=:linear)
    data = generate_fake_melitz_data(; D=D, sigma=2.5, theta_star=6.8, target_country=1, seed=seed, W=W)
    inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")
    outer_opt = joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt")
    obj, theta0 = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt)
    ctx = obj.γ

    rows = NamedTuple[]   # (delta, direction, elapsed_s, iters) for InnerSolved attempts only
    for delta in deltas, direction in directions
        iters_prev = Ref(CounterfactualSensitivity.INNER_ITERS_TOTAL[])
        function collector(theta, result)
            iters_now = CounterfactualSensitivity.INNER_ITERS_TOTAL[]
            d_iters = iters_now - iters_prev[]
            iters_prev[] = iters_now
            if result isa InnerSolved
                # elapsed time for THIS attempt is recorded by melitz_record_seconds_outcome!
                # under :inner_solve_warm_success -- but that's process-wide too. We instead
                # time the call ourselves at the point granularity we need: re-derive elapsed
                # from the most recent :inner_solve_warm_success sample (appended in the same
                # order attempts occur, so the LAST sample belongs to THIS call).
                st = get(MELITZ_PROF.stats, :inner_solve_warm_success, nothing)
                elapsed = st === nothing || isempty(st.samples) ? NaN : st.samples[end] / 1e9
                push!(rows, (delta=delta, direction=direction, elapsed_s=elapsed, iters=d_iters))
            end
            return nothing
        end
        @printf("running delta=%.1e direction=%s ...\n", delta, direction)
        t0 = time()
        res = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=delta, direction=direction,
            theta_box=theta_box, inner_loop_opt=inner_opt, outer_loop_opt=outer_opt,
            cutoff_constraint_backend=cutoff_constraint_backend, on_inner_result=collector)
        @printf("  wall=%.1fs inner_solved=%d numerical_fail=%d\n", time() - t0, res.n_inner_solved,
            res.n_numerical_failure_reject)
    end
    return rows
end

function pctile(v::Vector{<:Real}, p::Real)
    isempty(v) && return NaN
    s = sort(v)
    idx = clamp(ceil(Int, p * length(s)), 1, length(s))
    return s[idx]
end

function report_distribution(rows::Vector{<:NamedTuple})
    elapsed = [r.elapsed_s for r in rows if isfinite(r.elapsed_s)]
    iters = Float64.([r.iters for r in rows])
    println("\n", "="^100)
    @printf("Successful inner-solve distribution: n=%d\n", length(rows))
    println("="^100)
    for (name, v) in (("elapsed_s", elapsed), ("iters", iters))
        @printf("%-10s p50=%.4f p90=%.4f p95=%.4f p99=%.4f max=%.4f\n",
            name, pctile(v, 0.5), pctile(v, 0.9), pctile(v, 0.95), pctile(v, 0.99), maximum(v))
    end

    println("\n-- candidate maxit caps: successful solves that would be LOST (iters > cap) --")
    for m in MAXIT_CANDIDATES
        n_lost = count(r -> r.iters > m, rows)
        @printf("  maxit=%-6d n_lost=%d / %d (%.2f%%)\n", m, n_lost, length(rows), 100 * n_lost / length(rows))
    end

    println("\n-- candidate elapsed caps: successful solves that would be LOST (elapsed_s > cap) --")
    for c in ELAPSED_CANDIDATES
        n_lost = count(r -> isfinite(r.elapsed_s) && r.elapsed_s > c, rows)
        @printf("  cap=%-6.2f n_lost=%d / %d (%.2f%%)\n", c, n_lost, length(rows), 100 * n_lost / length(rows))
    end

    p99_iters = pctile(iters, 0.99)
    p99_elapsed = pctile(elapsed, 0.99)
    @printf("\nRecommended cap (just above empirical p99): maxit >= %d, elapsed_cap >= %.3fs\n",
        ceil(Int, p99_iters), p99_elapsed)
    return (elapsed=elapsed, iters=iters, p99_iters=p99_iters, p99_elapsed=p99_elapsed)
end

if abspath(PROGRAM_FILE) == @__FILE__
    MELITZ_PROFILE[] = true
    rows = collect_successful_solve_distribution()
    report_distribution(rows)
end
