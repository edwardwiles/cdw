# ADDENDUM Section 2: verify the "no-rescue" conjecture -- that warm vs. cold
# initialization affects inner-solve SPEED but never SUCCESS vs. FAILURE -- on a
# representative sample of thetas that produced `NumericalFailure` during a real
# finite-delta outer trajectory. If confirmed, this justifies removing the routine cold
# retry entirely (addendum Section 1), rather than keeping it "just in case" at the cost
# of doubling every genuine failure's wall time.
#
# Usage: julia --project=. scripts/melitz_no_rescue_benchmark.jl

using Printf, Random

const MELITZ_DIR = joinpath(@__DIR__, "..", "src", "melitz")
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(MELITZ_DIR, "profiling.jl"))
include(joinpath(MELITZ_DIR, "types.jl"))
include(joinpath(MELITZ_DIR, "pareto.jl"))
include(joinpath(MELITZ_DIR, "firm_quantities.jl"))
include(joinpath(MELITZ_DIR, "equilibrium.jl"))
include(joinpath(MELITZ_DIR, "moments.jl"))
include(joinpath(MELITZ_DIR, "delta_star.jl"))
include(joinpath(MELITZ_DIR, "affine_cutoff.jl"))
include(joinpath(MELITZ_DIR, "log_cutoff_param.jl"))
include(joinpath(MELITZ_DIR, "fake_data.jl"))
include(joinpath(MELITZ_DIR, "gradient_lab.jl"))
include(joinpath(MELITZ_DIR, "outer_solve.jl"))
include(joinpath(MELITZ_DIR, "inner_screening.jl"))
include(joinpath(MELITZ_DIR, "origin_block_screen.jl"))
include(joinpath(MELITZ_DIR, "finite_delta_outer.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity
const CS = CounterfactualSensitivity

"""
Collects a sample of `(theta, x_warm)` pairs where `x_warm` is the LAST VERIFIED dual
vector the single-slot warm-start mechanism (`obj.use_cached_x`/`obj.x`) actually held at
the moment this `theta` was attempted -- NOT `obj.x` read post-hoc (by the time a
`NumericalFailure` classification fires, `CounterfactualSensitivity.inner_loop_internal`
has already overwritten `obj.x` with `NaN` on that same failing call, per
`inner_loop_functions.jl`'s `PsiObjectiveBundleImplicit` branch -- reading it afterward
would silently just reproduce the cold-start condition). Tracking the last VERIFIED `x`
separately reconstructs the true historical warm-start condition.

Also surfaces a genuine, reportable structural finding along the way (see this session's
report Section "no-rescue benchmark"): because a failure poisons `obj.x` to `NaN`, and
`inner_loop_initial_values`'s own `norm(obj.x) < 1e6` guard silently falls back to a COLD
start whenever `obj.x` is `NaN`, every attempt IMMEDIATELY FOLLOWING a failure is already
effectively cold under the current single-slot design, regardless of `obj.use_cached_x` --
only the FIRST attempt after a SUCCESS is genuinely warm.
"""
function collect_numerical_failures(; D=4, W=20_000, seed=29, delta=1e-2, direction=:lower,
                                      theta_box=0.10, max_points=8)
    data = generate_fake_melitz_data(; D=D, sigma=2.5, theta_star=6.8, target_country=1, seed=seed, W=W)
    inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")
    outer_opt = joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt")
    obj, theta0 = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt)
    ctx = obj.γ

    failures = NamedTuple[]
    last_verified_x = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    n_immediately_after_failure = Ref(0)
    n_immediately_after_success = Ref(0)
    was_last_a_failure = Ref(false)
    function collector(theta, result)
        if result isa NumericalFailure
            if length(failures) < max_points
                push!(failures, (theta=collect(Float64.(theta)),
                    x_warm=last_verified_x[] === nothing ? nothing : copy(last_verified_x[])))
            end
            was_last_a_failure[] ? (n_immediately_after_failure[] += 1) : (n_immediately_after_success[] += 1)
            was_last_a_failure[] = true
        elseif result isa InnerSolved
            last_verified_x[] = copy(result.x)
            was_last_a_failure[] = false
        end
        return nothing
    end

    res = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=delta, direction=direction,
        theta_box=theta_box, inner_loop_opt=inner_opt, outer_loop_opt=outer_opt,
        on_inner_result=collector)
    @printf("collected %d NumericalFailure thetas (of %d numerical-failure rejections total, %d moment-infeasible, %d budget-infeasible, %d solved)\n",
        length(failures), res.n_numerical_failure_reject, res.n_moment_infeasible_reject,
        res.n_budget_infeasible_reject, res.n_inner_solved)
    @printf("of the %d numerical failures: %d occurred immediately after ANOTHER failure (already cold-poisoned per the note above), %d occurred immediately after a success (genuinely warm)\n",
        res.n_numerical_failure_reject, n_immediately_after_failure[], n_immediately_after_success[])
    return obj, ctx, failures
end

"One probe: try a given (obj.use_cached_x, obj.x) initial dual, time it, report outcome."
function probe_one(obj, theta, x_init; use_cached::Bool)
    was_cached, was_x = obj.use_cached_x, copy(obj.x)
    obj.use_cached_x = use_cached
    if use_cached
        obj.x .= x_init
    end
    t0 = time_ns()
    objSol, x, nStatus = CS.inner_loop_internal(obj, theta)
    elapsed = (time_ns() - t0) / 1e9
    ok = nStatus in (0, -100, -101, -103)
    obj.use_cached_x, obj.x[:] = was_cached, was_x
    return (ok=ok, nStatus=Int(nStatus), elapsed=elapsed, x=collect(Float64.(x)))
end

function run_no_rescue_benchmark(; D=4, W=20_000, seed=29, delta=1e-2, direction=:lower,
                                   n_random_starts=3, rng=MersenneTwister(1))
    println("="^100)
    println("Addendum Section 2: warm/cold no-rescue benchmark")
    println("="^100)
    obj, ctx, failures = collect_numerical_failures(; D=D, W=W, seed=seed, delta=delta, direction=direction)
    if isempty(failures)
        println("No NumericalFailure points observed in this trajectory -- nothing to benchmark.")
        return NamedTuple[]
    end

    n = ctx.moment_layout.num_moments
    rows = NamedTuple[]
    for (i, pt) in enumerate(failures)
        theta = pt.theta
        println("\n-- point $i / $(length(failures)) --")
        if pt.x_warm === nothing
            println("  warm (last verified x):   n/a (no verified solve yet at this point in the trajectory)")
            r_warm = (ok=false, nStatus=0, elapsed=0.0, x=Float64[])
        else
            r_warm = probe_one(obj, theta, pt.x_warm; use_cached=true)
            @printf("  warm (last verified x):   ok=%s nStatus=%-5d t=%.3fs\n", r_warm.ok, r_warm.nStatus, r_warm.elapsed)
        end

        r_cold = probe_one(obj, theta, zeros(n + 1); use_cached=false)
        @printf("  cold (neutral zero dual): ok=%s nStatus=%-5d t=%.3fs\n", r_cold.ok, r_cold.nStatus, r_cold.elapsed)

        random_results = []
        for k in 1:n_random_starts
            x_rand = randn(rng, n + 1) .* 0.5
            r_rand = probe_one(obj, theta, x_rand; use_cached=true)
            @printf("  random start %d:           ok=%s nStatus=%-5d t=%.3fs\n", k, r_rand.ok, r_rand.nStatus, r_rand.elapsed)
            push!(random_results, r_rand)
        end

        any_rescue = r_cold.ok || any(r.ok for r in random_results)
        push!(rows, (point=i, warm_ok=r_warm.ok, warm_t=r_warm.elapsed, cold_ok=r_cold.ok, cold_t=r_cold.elapsed,
            n_random_ok=count(r -> r.ok, random_results), any_rescue=any_rescue))
    end

    println("\n" * "="^100)
    println("Summary")
    println("="^100)
    @printf("%-6s %-8s %-10s %-8s %-10s %-12s %-10s\n", "point", "warm_ok", "warm_t", "cold_ok", "cold_t", "n_rand_ok", "any_rescue")
    for r in rows
        @printf("%-6d %-8s %-10.3f %-8s %-10.3f %-12d %-10s\n", r.point, r.warm_ok, r.warm_t, r.cold_ok, r.cold_t, r.n_random_ok, r.any_rescue)
    end
    n_rescued = count(r -> r.any_rescue, rows)
    @printf("\n%d / %d NumericalFailure points were rescued by SOME alternative dual start (cold or random).\n", n_rescued, length(rows))
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_no_rescue_benchmark()
end
