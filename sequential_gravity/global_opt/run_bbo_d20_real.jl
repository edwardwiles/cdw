# ============================================================================
# Stage C: D=20 real-data global (gradient-free) outer search via BlackBoxOptim, UPPER bound
# (maximizes kappa), delta=1, W=80000 -- same setting already tried with the LOCAL KNITRO
# search (see derivative_diagnostics/full_d2_correction_report.md section 7/7.2):
#   fixed-A* baseline:              kappa = 0.076693  (A pinned at point estimate)
#   local search, unscaled:         kappa = 0.076693  (A never moved -- gradient-scale bug)
#   local search, KNITRO var-scaled: kappa = 0.081518  (A moved 522%, but messy convergence)
#
# This is a SERIAL population loop -- each candidate evaluated one at a time, each using the
# full 19-thread destination-inversion parallelism via PARALLEL_INVERSION (option 1 in the
# task spec: cheapest to build; option 2, Distributed.jl worker-process parallelism ACROSS the
# population, was explicitly scoped out of this session for time). At ~10-80s/eval a population
# of BBO_POPSIZE=16 needs many hours to get through a meaningful number of generations --
# MaxTime-bounded (not MaxFuncEvals-bounded) so it can be launched detached and left running,
# with periodic checkpointing so partial progress is never lost.
#
#   FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true \
#     REAL_DATA_DIR=<repo>/real_data/noah_D20 \
#     BBO_MAXTIME=10800 BBO_POPSIZE=16 \
#     julia -t 19 --project=. sequential_gravity/global_opt/run_bbo_d20_real.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using BlackBoxOptim, Printf, Dates, JLD2
using BlackBoxOptim: num_func_evals, f_calls
include(joinpath(@__DIR__, "bbo_common.jl"))

const DELTA_VAL = parse(Float64, get(ENV, "BBO_DELTA", "1.0"))
const MAXTIME = parse(Float64, get(ENV, "BBO_MAXTIME", "10800"))     # default 3h
const POPSIZE = parse(Int, get(ENV, "BBO_POPSIZE", "16"))
const MAXIT = parse(Int, get(ENV, "BBO_MAXIT", "100"))
const MAXIMIZE_GP = lowercase(get(ENV, "BBO_MAXIMIZE_GP", "false")) in ("1", "true", "yes")  # false=upper bound (default)
const CKPT_DIR = get(ENV, "BBO_CKPT_DIR", joinpath(@__DIR__, "d20_checkpoints"))
isdir(CKPT_DIR) || mkpath(CKPT_DIR)
const CKPT_PATH = joinpath(CKPT_DIR, "bbo_d20_$(MAXIMIZE_GP ? "lower" : "upper")_delta$(DELTA_VAL).jld2")

@printf("\n===== BBO D=20 real-data global search: %s bound, W=%d delta=%.4g maxtime=%.0fs pop=%d maxit=%d LOGBOUND=%.2f =====\n",
        MAXIMIZE_GP ? "lower" : "upper", W, DELTA_VAL, MAXTIME, POPSIZE, MAXIT, LOGBOUND)
@printf("fixed-A* baseline kappa (point estimate) = %.6f\n", KAPPA_POINT_EST)
flush(stdout)

fitness = make_fitness(; maximize_gp = MAXIMIZE_GP, δbudget = DELTA_VAL, maxit = MAXIT)

n_eval = Ref(0)
function checkpoint_callback(oc)
    n_eval[] += 1
    x = best_candidate(oc); f = best_fitness(oc)
    diag = eval_candidate(x; δbudget = DELTA_VAL, maxit = MAXIT)
    relΔA = norm(diag.θ[4:3+D] .- Acol_star) / norm(Acol_star)
    @printf("[checkpoint %s] fevals=%d best_fitness=%.6g  gp=%.6f kappa=%.6f feasible=%s relΔA=%.3f\n",
            Dates.format(Dates.now(), "HH:MM:SS"), num_func_evals(oc), f, diag.gp, diag.κ, diag.feasible, relΔA)
    flush(stdout)
    try
        JLD2.save(CKPT_PATH, Dict(
            "x_best" => x, "fitness_best" => f, "num_evals" => num_func_evals(oc),
            "theta_star" => diag.θ, "gamma_p" => diag.gp, "kappa" => diag.κ,
            "R" => diag.R, "div_p" => diag.div_p, "feasible" => diag.feasible, "relDeltaA" => relΔA,
            "delta_budget" => DELTA_VAL, "D" => D, "W" => W, "maximize_gp" => MAXIMIZE_GP,
            "kappa_point_estimate" => KAPPA_POINT_EST, "timestamp" => string(Dates.now())))
    catch e
        @warn "checkpoint save failed" exception=e
    end
end

t0 = time()
res = bboptimize(fitness;
    SearchRange = search_range(), NumDimensions = D + 1,
    Method = :adaptive_de_rand_1_bin_radiuslimited,
    PopulationSize = POPSIZE, MaxTime = MAXTIME,
    TraceMode = :compact, TraceInterval = 30.0,
    CallbackFunction = checkpoint_callback, CallbackInterval = 0.0)
wall = time() - t0

xstar = best_candidate(res)
diag = eval_candidate(xstar; δbudget = DELTA_VAL, maxit = MAXIT)
relΔA = norm(diag.θ[4:3+D] .- Acol_star) / norm(Acol_star)

@printf("\n===== BBO D=20 FINAL: wall=%.1fs fevals=%d =====\n", wall, f_calls(res))
@printf("  best candidate: gp=%.6f kappa=%.6f R=%.3e div(p)=%.4f feasible=%s relΔA=%.3f\n",
        diag.gp, diag.κ, diag.R, diag.div_p, diag.feasible, relΔA)
@printf("  fixed-A* baseline kappa    = %.6f\n", KAPPA_POINT_EST)
@printf("  local unscaled search kappa= 0.076693\n")
@printf("  local var-scaled kappa     = 0.081518\n")
@printf("  BBO global search kappa    = %.6f\n", diag.κ)

JLD2.save(CKPT_PATH, Dict(
    "x_best" => xstar, "fitness_best" => best_fitness(res), "num_evals" => num_func_evals(res),
    "theta_star" => diag.θ, "gamma_p" => diag.gp, "kappa" => diag.κ,
    "R" => diag.R, "div_p" => diag.div_p, "feasible" => diag.feasible, "relDeltaA" => relΔA,
    "delta_budget" => DELTA_VAL, "D" => D, "W" => W, "maximize_gp" => MAXIMIZE_GP,
    "kappa_point_estimate" => KAPPA_POINT_EST, "wall" => wall, "done" => true,
    "timestamp" => string(Dates.now())))

println("\nBBO_D20_REAL_DONE")
