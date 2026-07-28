# 2026-07-28 step-control/robustness session, Phase 0: reproduce the six required
# configurations on the CURRENT (post-Phase-1-edit) code -- the new `objective_scale=:auto`
# default plus `allow_unscaled_objective` fail-safe wiring in
# src/melitz/finite_delta_outer.jl. Confirms the new API reproduces the SAME economic
# conclusions the 2026-07-28 gamma-profile session's own hand-rolled A/B/C test found,
# without silently changing behavior for any existing caller.
#
# Usage: julia --project=. -t 20 scripts/melitz_phase0_reproduce_2026-07-28.jl

using Pkg
Pkg.activate(dirname(@__DIR__))
using Printf, DelimitedFiles, LinearAlgebra
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(@__DIR__, "melitz_regression_fixtures_2026-07-28.jl"))
using KNITRO

const REPO = dirname(@__DIR__)
const S_G = 1e-4
const CAP = 10.0

gamma_only_box(n, g_radius) = (b = zeros(n); b[1] = g_radius; b)

function run_one(label, ctx, obj, theta0, delta, direction, g_radius, inner_opt, outer_opt;
                  var_scale, objective_scale, allow_unscaled_objective=false)
    n = length(theta0)
    box = gamma_only_box(n, g_radius)
    vc = var_scale === nothing ? nothing : collect(Float64.(theta0))
    local res
    wall = @elapsed begin
        res = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=delta, direction=direction,
            delta_evaluation_cap=CAP, gradient_backend=:B_direct_argument_sorted_serial, h=1e-4,
            theta_box=box, cutoff_constraint_backend=:linear,
            inner_loop_opt=inner_opt, outer_loop_opt=outer_opt,
            var_scale=var_scale, var_center=vc,
            backend=:matrix_free, forbid_dense_fallback=true,
            objective_scale=objective_scale, allow_unscaled_objective=allow_unscaled_objective)
    end
    cv = res.cold_verified_incumbent
    best_g = cv === nothing ? theta0[1] : cv.eval.theta_free[1]
    best_Delta = cv === nothing ? NaN : cv.eval.Delta
    moved = abs(best_g - theta0[1]) > 1e-8
    @printf("  [%-28s] nStatus=%5d wall=%6.2fs obj_scale_resolved=%s  g0=%.6f -> best_g=%.6f (moved=%s)  Delta=%s\n",
        label, res.nStatus, wall, string(res.objective_scale_resolved),
        theta0[1], best_g, moved, isnan(best_Delta) ? "NA" : @sprintf("%.4e", best_Delta))
    flush(stdout)
    return (config=label, nStatus=res.nStatus, wall=wall, moved=moved, g0=theta0[1], best_g=best_g,
            best_Delta=best_Delta, objective_scale_resolved=res.objective_scale_resolved)
end

function main()
    BLAS.set_num_threads(1)
    outer_active = joinpath(REPO, "melitz_outer_finite_delta_alg_active_2026-07-27.opt")
    results = NamedTuple[]

    println("="^100); println("D=4 fixed-A/f profile point (direct inner solve, no outer KNITRO)"); println("="^100)
    d4 = build_d4_fixture()
    inner_opt4 = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
    r_mid = melitz_classified_inner_solve(d4.obj, (t = copy(d4.theta0); t[1] -= 0.10; t), d4.ctx; delta_evaluation_cap=50.0, bank=MelitzDualBank(4))
    @printf("  D4 g=%.6f -> %s\n", d4.theta0[1] - 0.10, r_mid isa FiniteSolved ? (@sprintf("FiniteSolved Delta=%.4e", r_mid.Delta)) : string(typeof(r_mid)))

    println("\n" * "="^100); println("real-D20 fixed-A/f profile point (direct inner solve, no outer KNITRO)"); println("="^100)
    d20 = build_realD20_fixture()
    r_mid20 = melitz_classified_inner_solve(d20.obj, (t = copy(d20.theta0); t[1] -= 0.05; t), d20.ctx; delta_evaluation_cap=50.0, bank=MelitzDualBank(4))
    @printf("  D20 g=%.6f -> %s\n", d20.theta0[1] - 0.05, r_mid20 isa FiniteSolved ? (@sprintf("FiniteSolved Delta=%.4e", r_mid20.Delta)) : string(typeof(r_mid20)))

    println("\n" * "="^100); println("D=4 gamma-only: unscaled / scaled-no-objscale(override) / scaled-auto-objscale"); println("="^100)
    n4 = length(d4.theta0)
    vs4 = ones(n4); vs4[1] = S_G
    push!(results, run_one("D4_unscaled", d4.ctx, d4.obj, d4.theta0, 1e-2, :upper, 0.3, inner_opt4, outer_active;
        var_scale=nothing, objective_scale=:auto))
    push!(results, run_one("D4_scaled_no_objscale_OVERRIDE", d4.ctx, d4.obj, d4.theta0, 1e-2, :upper, 0.3, inner_opt4, outer_active;
        var_scale=vs4, objective_scale=nothing, allow_unscaled_objective=true))
    push!(results, run_one("D4_scaled_auto_objscale", d4.ctx, d4.obj, d4.theta0, 1e-2, :upper, 0.3, inner_opt4, outer_active;
        var_scale=vs4, objective_scale=:auto))

    println("\n" * "="^100); println("real-D20 gamma-only: unscaled / scaled-no-objscale(override) / scaled-auto-objscale"); println("="^100)
    BLAS.set_num_threads(20)
    inner_opt20 = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
    n20 = length(d20.theta0)
    vs20 = ones(n20); vs20[1] = S_G
    push!(results, run_one("D20_unscaled", d20.ctx, d20.obj, d20.theta0, 1.0, :upper, 0.15, inner_opt20, outer_active;
        var_scale=nothing, objective_scale=:auto))
    push!(results, run_one("D20_scaled_no_objscale_OVERRIDE", d20.ctx, d20.obj, d20.theta0, 1.0, :upper, 0.15, inner_opt20, outer_active;
        var_scale=vs20, objective_scale=nothing, allow_unscaled_objective=true))
    push!(results, run_one("D20_scaled_auto_objscale", d20.ctx, d20.obj, d20.theta0, 1.0, :upper, 0.15, inner_opt20, outer_active;
        var_scale=vs20, objective_scale=:auto))
    BLAS.set_num_threads(1)

    # The fail-fast guard itself: this MUST throw (var_scale set, objective_scale=nothing, no override).
    threw = false
    try
        run_one("D4_scaled_no_objscale_UNGUARDED", d4.ctx, d4.obj, d4.theta0, 1e-2, :upper, 0.3, inner_opt4, outer_active;
            var_scale=vs4, objective_scale=nothing)
    catch e
        threw = e isa ArgumentError
        println("  Fail-fast guard fired as expected: ", sprint(showerror, e))
    end
    @assert threw "the fail-fast guard did NOT throw for var_scale set + objective_scale=nothing + no override -- REGRESSION"

    outfile = joinpath(REPO, "docs", "key_results", "melitz_phase0_reproduce_2026-07-28.csv")
    open(outfile, "w") do io
        cols = keys(results[1])
        println(io, join(cols, ","))
        for r in results
            println(io, join([r[c] for c in cols], ","))
        end
    end
    println("\nDONE. CSV written to ", outfile)
end

main()
