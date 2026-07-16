# ============================================================================
# Stage A: validate the BlackBoxOptim global-search logic (penalty design, log-Acol
# parameterization, hyperparameters) cheaply at D=4 synthetic data before spending D=20
# real-data compute on it. Both bounds, delta=1, W=8000 -- the SAME setting
# derivative_diagnostics/full_d2_correction_report.md section 5 already validated with the
# LOCAL (KNITRO, gradient-based) search, giving a direct comparison point:
#   lower bound: kappa_full = 0.004062  (fixed-A comparator 0.005297)
#   upper bound: kappa_full = 0.172109  (fixed-A comparator 0.159080)
#
#   julia --project=. -t 4 sequential_gravity/global_opt/run_bbo_d4_synthetic.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["DELTA_GRID"] = get(ENV, "DELTA_GRID", "1.0")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using BlackBoxOptim, Printf, Dates
using BlackBoxOptim: f_calls
include(joinpath(@__DIR__, "bbo_common.jl"))

const DELTA_VAL = parse(Float64, get(ENV, "BBO_DELTA", "1.0"))
# Per-eval cost at D=4/W=8000 is ~5s once warm (measured directly) -- 900s/bound at PopulationSize=24
# gives roughly 7-8 generations of adaptive DE, enough to see real descent for validation purposes.
const MAXTIME = parse(Float64, get(ENV, "BBO_MAXTIME", "900"))       # seconds, per bound
const POPSIZE = parse(Int, get(ENV, "BBO_POPSIZE", "24"))
const MAXIT = parse(Int, get(ENV, "BBO_MAXIT", "100"))

function run_bound(name::Symbol, maximize_gp::Bool)
    @printf("\n===== BBO global search: %s bound, D=%d W=%d delta=%.4g maxtime=%.0fs pop=%d =====\n",
            name, D, W, DELTA_VAL, MAXTIME, POPSIZE)
    fitness = make_fitness(; maximize_gp = maximize_gp, δbudget = DELTA_VAL, maxit = MAXIT)
    x0 = x_from_theta(θr0)   # seed the initial population with A* itself (known-feasible)
    t0 = time()
    res = bboptimize(fitness;
        SearchRange = search_range(), NumDimensions = D + 1,
        Method = :adaptive_de_rand_1_bin_radiuslimited,
        PopulationSize = POPSIZE, MaxTime = MAXTIME,
        TraceMode = :compact, TraceInterval = 15.0)
    wall = time() - t0
    xstar = best_candidate(res)
    diag = eval_candidate(xstar; δbudget = DELTA_VAL, maxit = MAXIT)
    @printf("  BBO done: wall=%.1fs fevals=%d best_fitness=%.6g\n", wall, f_calls(res), best_fitness(res))
    @printf("  best candidate: gp=%.6f kappa=%.6f R=%.3e div(p)=%.4f feasible=%s\n",
            diag.gp, diag.κ, diag.R, diag.div_p, diag.feasible)
    relΔA = norm(diag.θ[4:3+D] .- Acol_star) / norm(Acol_star)
    @printf("  rel‖ΔA‖ vs A* = %.3f\n", relΔA)
    (name = name, wall = wall, xstar = xstar, diag = diag, relΔA = relΔA, fevals = f_calls(res))
end

results = Dict{Symbol,Any}()
results[:lower] = run_bound(:lower, true)    # maximize gp -> minimize kappa
results[:upper] = run_bound(:upper, false)   # minimize gp -> maximize kappa

println("\n" * "="^78)
println(">>> SUMMARY vs Stage-A local-search reference (full_d2_correction_report.md section 5)")
println("="^78)
@printf("%-8s %12s %12s %10s %10s\n", "bound", "kappa_BBO", "kappa_local", "feasible", "rel‖ΔA‖")
@printf("%-8s %12.6f %12.6f %10s %9.1f%%\n", "lower", results[:lower].diag.κ, 0.004062, results[:lower].diag.feasible, 100*results[:lower].relΔA)
@printf("%-8s %12.6f %12.6f %10s %9.1f%%\n", "upper", results[:upper].diag.κ, 0.172109, results[:upper].diag.feasible, 100*results[:upper].relΔA)

println("\nBBO_D4_SYNTHETIC_VALIDATION_DONE")
