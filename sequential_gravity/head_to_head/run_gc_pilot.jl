# ============================================================================
# GC PILOT: a single, short, cheap solve to verify the LOGBOUND=3.0 +
# population-seeded-at-A* fix actually lets GC find a feasible point at D=20 --
# BEFORE committing the full 9-solve/90min-per-solve overnight budget to it.
#
# The one prior real GC run (run_bbo_d20_real.jl, LOGBOUND=6.0 default, unseeded
# population) found 0/34 feasible evaluations in ~25 minutes. This pilot targets
# delta=1.0 (the middle target) with a SHORT cap (default 12 min) and reports
# plainly: did it find ANY feasible point, and how many evaluations did that take.
#
#   FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true \
#     GC_PILOT_MAXTIME=720 \
#     julia -t 19 --project=. sequential_gravity/head_to_head/run_gc_pilot.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
ENV["LOGBOUND"] = get(ENV, "LOGBOUND", "3.0")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
include(joinpath(@__DIR__, "common.jl"))
using BlackBoxOptim, Dates
using BlackBoxOptim: num_func_evals, f_calls
include(joinpath(@__DIR__, "..", "global_opt", "bbo_common.jl"))

@assert D == 20 && W == 80000 "this driver is D=20/W=80000-specific"

const PILOT_MAXTIME = parse(Float64, get(ENV, "GC_PILOT_MAXTIME", "720"))   # 12 min default
const PILOT_POPSIZE = parse(Int, get(ENV, "BBO_POPSIZE", "16"))
const PILOT_DELTA = parse(Float64, get(ENV, "GC_PILOT_DELTA", "1.0"))

Acol_star_h2h = θr0[4:3+D]
gp0 = θr0[3]

println("\n" * "="^78)
@printf(">>> GC PILOT: delta_budget=%.4g  maxtime=%.0fs  LOGBOUND=%.2f  PopulationSize=%d  seeded-at-A*\n",
        PILOT_DELTA, PILOT_MAXTIME, LOGBOUND, PILOT_POPSIZE)
println("="^78)
flush(stdout)

fitness = make_fitness(; maximize_gp = false, δbudget = PILOT_DELTA, maxit = 100)
x0 = vcat(gp0, zeros(D))   # seed the population with A*, gp0 -- KNOWN feasible at delta=Inf

n_feasible_seen = Ref(0)
n_evals_seen = Ref(0)
first_feasible_eval = Ref(-1)

function pilot_callback(oc)
    x = best_candidate(oc); f = best_fitness(oc)
    diag = eval_candidate(x; δbudget = PILOT_DELTA, maxit = 100)
    n_evals_seen[] = num_func_evals(oc)
    if diag.feasible
        n_feasible_seen[] += 1
        first_feasible_eval[] == -1 && (first_feasible_eval[] = num_func_evals(oc))
    end
    @printf("[pilot checkpoint %s] fevals=%d best_fitness=%.6g gp=%.6f kappa=%.6f feasible=%s (first_feasible_eval=%d)\n",
            Dates.format(Dates.now(), "HH:MM:SS"), num_func_evals(oc), f, diag.gp, diag.κ, diag.feasible, first_feasible_eval[])
    flush(stdout)
end

t0 = time()
res = bboptimize(fitness, x0;
    SearchRange = search_range(), NumDimensions = D + 1,
    Method = :adaptive_de_rand_1_bin_radiuslimited,
    PopulationSize = PILOT_POPSIZE, MaxTime = PILOT_MAXTIME,
    TraceMode = :compact, TraceInterval = 15.0,
    CallbackFunction = pilot_callback, CallbackInterval = 0.0)
wall = time() - t0

xstar = best_candidate(res)
diag = eval_candidate(xstar; δbudget = PILOT_DELTA, maxit = 100)

println("\n" * "="^78); println(">>> GC PILOT RESULT"); println("="^78)
@printf("wall=%.1fs total_evals=%d\n", wall, f_calls(res))
@printf("first feasible point found at eval #%d\n", first_feasible_eval[])
@printf("final best: feasible=%s gp=%.6f kappa=%.6f R=%.3e div_p=%.4f\n", diag.feasible, diag.gp, diag.κ, diag.R, diag.div_p)
if diag.feasible
    println("\n=> PILOT PASSED: GC (with LOGBOUND=3.0 + A*-seeded population) DOES find feasible points at D=20. Safe to launch the full 9-solve overnight run.")
else
    println("\n=> PILOT FAILED: still 0 feasible points found even after the fix. DO NOT launch GC's full overnight budget blind -- investigate further (e.g. tighter LOGBOUND, more seeded individuals, or drop GC from this comparison) before proceeding.")
end
println("\nGC_PILOT DONE")
