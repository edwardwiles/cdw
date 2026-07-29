# q-bandwidth convergence campaign (2026-07-29), Phase 12: bounded D4 outer-solver smoke
# test. Compares production (A,f) backend vs the experimental (A,q) backend (Phase 11) from
# an IDENTICAL starting economic state, identical KNITRO settings, identical evaluation cap.
#
# DISCLOSED: D4's fixed-A/f corridor tops out at Delta~0.57 (Phase 3) -- budgets tested are
# 0.1 and 0.5 (both reachable), NOT 2 (D4 cannot reach it; this is the same established
# corridor limit, not a new gap). This is a SHORT SOLVER SMOKE TEST, not a convergence proof.

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles

const OUTDIR = joinpath(REPO, "docs", "key_results")
inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
outer_opt = joinpath(REPO, "melitz_outer_finite_delta_alg_direct_2026-07-27.opt")
println("Julia threads: ", Threads.nthreads()); flush(stdout)

data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)

results = NamedTuple[]

function run_backend(label::AbstractString, gradient_backend::Symbol, outer_parameterization::Symbol, delta::Real, direction::Symbol)
    obj, theta0 = build_melitz_psi_bundle(data; outer_parameterization=outer_parameterization,
        policy=CappedEvaluation(10.0), backend=:matrix_free, forbid_dense_fallback=true)
    ctx = obj.γ
    t0 = time()
    res = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=delta, direction=direction,
        gradient_backend=gradient_backend, backend=:matrix_free,
        policy=CappedEvaluation(10.0), inner_loop_opt=inner_opt, outer_loop_opt=outer_opt)
    wall = time() - t0
    inc = res.cold_verified_incumbent
    g_best = inc === nothing ? theta0[1] : inc.eval.theta_free[1]
    kappa_best = kappa_ratio_of_g(g_best, ctx)
    delta_best = inc === nothing ? NaN : inc.eval.Delta
    @printf("[%-30s] delta=%.1f dir=%s  nStatus=%d  wall=%.1fs  inner_solves=%d (infeas=%d, evalfail=%d)\n",
            label, delta, direction, res.nStatus, wall, res.inner_solve_count, res.inner_infeas_count, res.inner_eval_failures)
    @printf("    best incumbent: g=%.6f  kappa(GT%%)=%.4f%%  Delta=%.6e  within_budget=%s  classification=%s\n",
            g_best, 100*kappa_best, delta_best, string(!isnan(delta_best) && delta_best <= delta), string(inc === nothing ? :none : inc.source))
    flush(stdout)
    push!(results, (label=label, gradient_backend=gradient_backend, delta=delta, direction=direction,
                     nStatus=res.nStatus, wall_s=wall, inner_solve_count=res.inner_solve_count,
                     inner_infeas_count=res.inner_infeas_count, inner_eval_failures=res.inner_eval_failures,
                     g_best=g_best, kappa_best_pct=100*kappa_best, delta_best=delta_best,
                     within_budget=(!isnan(delta_best) && delta_best <= delta),
                     incumbent_source=string(inc === nothing ? :none : inc.source)))
end

for delta in (0.1, 0.5), direction in (:upper, :lower)
    run_backend("production_(A,f)", :auto, :logf, delta, direction)
    run_backend("experimental_(A,q)", :B_direct_argument_aq_experimental, :logcutoff, delta, direction)
end

open(joinpath(OUTDIR, "melitz_qbw_phase12_solver_smoke_2026-07-29.csv"), "w") do io
    println(io, "label,gradient_backend,delta,direction,nStatus,wall_s,inner_solve_count,inner_infeas_count,inner_eval_failures,g_best,kappa_best_pct,delta_best,within_budget,incumbent_source")
    for r in results
        println(io, join([r.label, r.gradient_backend, r.delta, r.direction, r.nStatus, r.wall_s,
                           r.inner_solve_count, r.inner_infeas_count, r.inner_eval_failures, r.g_best,
                           r.kappa_best_pct, r.delta_best, r.within_budget, r.incumbent_source], ","))
    end
end
println("\nPhase 12 complete. Rows: ", length(results))
flush(stdout)
