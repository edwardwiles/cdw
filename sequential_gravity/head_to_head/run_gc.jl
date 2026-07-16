# ============================================================================
# GC ("global-constrained"): extremize gamma'_focal s.t. delta*<=budget, via BlackBoxOptim
# (gradient-free, population-based DE), joint search over (gamma'_focal, A_od) in
# (gp, logratio) space, using bbo_common.jl's smooth infeasibility-penalty fitness.
#
# "Revived" per HEAD_TO_HEAD_PROMPT.md: the one prior real run (run_bbo_d20_real.jl,
# global_opt/logs/bbo_d20_upper_delta1_W80000_run.log) found 0/34 feasible evaluations,
# relΔA up to 116x -- because (a) LOGBOUND defaulted to 6.0 (bbo_common.jl), calibrated from
# a single-coordinate reading of multistart screening, wildly too loose applied jointly
# across 20 independent coordinates, and (b) the initial population was NOT seeded near the
# known-feasible A*, so DE had to find its way back from an essentially random 21-D start.
# Both are fixed here: LOGBOUND=3.0 (matching the already-validated GU driver's own choice,
# global_optimizer_report.md section 4.2), and the population is seeded with ONE individual
# at the actual starting point for each solve (BlackBoxOptim's own `bboptimize(fitness, x0,
# params...)` two-arg form -- confirmed by reading BlackBoxOptim's own source,
# bboptimize.jl:70-88, this calls `set_candidate!`, inserting x0 as a real population member,
# not just an initial-guess hint).
#
# Because GC has never yet been shown to find a feasible point at D=20, this driver is
# PILOTED ALONE (one cheap/short solve) before committing the full 9-solve overnight budget
# to it -- see run_gc_pilot.jl.
#
# 9 solves (3 targets x 3 starts), sequential, each capped at 90 min via BBO's own MaxTime.
#
#   FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true \
#     julia -t 19 --project=. sequential_gravity/head_to_head/run_gc.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
ENV["LOGBOUND"] = get(ENV, "LOGBOUND", "3.0")   # fix for the 0/34-feasible bug, see header
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
include(joinpath(@__DIR__, "common.jl"))
using BlackBoxOptim, Dates
using BlackBoxOptim: num_func_evals, f_calls
include(joinpath(@__DIR__, "..", "global_opt", "bbo_common.jl"))

@assert D == 20 && W == 80000 "this driver is D=20/W=80000-specific"

const POPSIZE = parse(Int, get(ENV, "BBO_POPSIZE", "16"))
const MAXIT_GC = parse(Int, get(ENV, "BBO_MAXIT", "100"))

const OUT_DIR_GC = joinpath(H2H_DIR, "out_gc")
isdir(OUT_DIR_GC) || mkpath(OUT_DIR_GC)
result_path_gc(tname, sname) = joinpath(OUT_DIR_GC, "gc_$(tname)_$(sname).jld2")

starts = load_shared_starts()
const Acol_star_h2h = θr0[4:3+D]
const gp0 = θr0[3]   # Frechet point estimate -- the initial gp guess for every cold start (matches LC's own θr0 convention)

println("\n" * "="^78)
@printf(">>> GC (global-constrained) head-to-head: D=%d W=%d cap=%.0fs/solve LOGBOUND=%.2f PopulationSize=%d\n",
        D, W, CAP_SECONDS, LOGBOUND, POPSIZE)
println("="^78)
flush(stdout)

"""
    gc_solve(tname, sname, δval, x0)

`x0` is in bbo_common.jl's (gp, logratio) representation (see `x_from_theta`). Seeds the BBO
population with x0 as a real individual (not just a hint), runs adaptive DE for up to
CAP_SECONDS, checkpoints every optimizer step, and re-verifies the final best candidate via
`eval_candidate` (an independent re-solve, not just trusting the BBO-internal fitness cache).
"""
function gc_solve(tname::String, sname::String, δval::Float64, x0::Vector{Float64})
    path = result_path_gc(tname, sname)
    existing = load_done(path)
    if existing !== nothing
        @printf("[GC %s/%s] ALREADY DONE (resume) -- feasible=%s kappa=%.6f\n",
                tname, sname, existing["feasible"], existing["kappa"])
        flush(stdout)
        return existing
    end
    @printf("\n[GC %s/%s] delta_budget=%.4g -- starting solve (cap=%.0fs, seeded at x0)\n", tname, sname, δval, CAP_SECONDS)
    flush(stdout)

    fitness = make_fitness(; maximize_gp = false, δbudget = δval, maxit = MAXIT_GC)   # upper bound: MINIMIZE gp

    function checkpoint_callback(oc)
        x = best_candidate(oc); f = best_fitness(oc)
        diag = eval_candidate(x; δbudget = δval, maxit = MAXIT_GC)
        relΔA = norm(diag.θ[4:3+D] .- Acol_star_h2h) / norm(Acol_star_h2h)
        @printf("[GC %s/%s checkpoint %s] fevals=%d best_fitness=%.6g gp=%.6f kappa=%.6f feasible=%s relΔA=%.3f\n",
                tname, sname, Dates.format(Dates.now(), "HH:MM:SS"), num_func_evals(oc), f, diag.gp, diag.κ, diag.feasible, relΔA)
        flush(stdout)
        try
            JLD2.save(path, Dict(
                "method" => "GC", "target" => tname, "start" => sname, "delta_budget" => δval,
                "x_best" => x, "fitness_best" => f, "num_evals" => num_func_evals(oc),
                "theta_star" => diag.θ, "gamma_p" => diag.gp, "kappa" => diag.κ,
                "R" => diag.R, "div_p" => diag.div_p, "feasible" => diag.feasible, "relDeltaA" => relΔA,
                "cap_seconds" => CAP_SECONDS, "LOGBOUND" => LOGBOUND,
                "done" => false, "timestamp" => string(Dates.now())))
        catch e
            @warn "GC checkpoint save failed" exception = e
        end
    end

    t0 = time()
    res = bboptimize(fitness, x0;
        SearchRange = search_range(), NumDimensions = D + 1,
        Method = :adaptive_de_rand_1_bin_radiuslimited,
        PopulationSize = POPSIZE, MaxTime = CAP_SECONDS,
        TraceMode = :compact, TraceInterval = 30.0,
        CallbackFunction = checkpoint_callback, CallbackInterval = 0.0)
    wall = time() - t0

    xstar = best_candidate(res)
    diag = eval_candidate(xstar; δbudget = δval, maxit = MAXIT_GC)
    relΔA = norm(diag.θ[4:3+D] .- Acol_star_h2h) / norm(Acol_star_h2h)

    audited_δstar = NaN; audited_ok = false
    if diag.feasible
        audit = exact_inner_divergence_at(diag.θ)
        audited_δstar = audit.δ_star
        audited_ok = audit.gravity_ok && !istop_delta(audited_δstar)
    end

    d = Dict(
        "method" => "GC", "target" => tname, "start" => sname, "delta_budget" => δval,
        "x_best" => xstar, "fitness_best" => best_fitness(res), "num_evals" => f_calls(res),
        "theta_star" => diag.θ, "gamma_p" => diag.gp, "kappa" => diag.κ,
        "R" => diag.R, "div_p" => diag.div_p, "feasible" => diag.feasible, "relDeltaA" => relΔA,
        "audited_delta_star" => audited_δstar, "audited_gravity_ok" => audited_ok,
        "wall" => wall, "cap_seconds" => CAP_SECONDS, "LOGBOUND" => LOGBOUND,
        "done" => true, "timestamp" => string(Dates.now()))
    JLD2.save(path, d)
    @printf("[GC %s/%s] DONE: feasible=%s gp=%.6f kappa=%.6f audited_delta*=%.6g wall=%.1fs nevals=%d\n",
            tname, sname, diag.feasible, diag.gp, diag.κ, audited_δstar, wall, f_calls(res))
    flush(stdout)
    d
end

"""
    gc_target_winner(rows)

Among a target's 3 completed solves, picks the FEASIBLE candidate with the HIGHEST kappa
(re-verified via `eval_candidate`, not the raw BBO fitness). `nothing` if none of the 3
starts found a feasible point.
"""
function gc_target_winner(rows::Vector{Dict{String,Any}})
    feas = filter(r -> r["feasible"] === true, rows)
    isempty(feas) && return nothing
    feas[argmax([r["kappa"] for r in feas])]
end

results_by_target = Dict{String,Vector{Dict{String,Any}}}()
prev_winner_theta = nothing

for (i, tgt) in enumerate(TARGETS)
    tname = tgt.name; δval = tgt.delta
    println("\n" * "-"^78)
    @printf("[GC] TARGET %s: delta_budget=%.4g (reference gp=%.6f kappa=%.6f)\n", tname, δval, tgt.gp, tgt.kappa)
    println("-"^78)
    flush(stdout)

    x0_Astar = vcat(gp0, zeros(D))
    x0_rand1 = vcat(gp0, log.(starts.rand1 ./ Acol_star_h2h))

    rows = Dict{String,Any}[]
    push!(rows, Dict{String,Any}(gc_solve(tname, "Astar", δval, x0_Astar)))
    push!(rows, Dict{String,Any}(gc_solve(tname, "rand1", δval, x0_rand1)))

    if i == 1
        x0_rand2 = vcat(gp0, log.(starts.rand2 ./ Acol_star_h2h))
        push!(rows, Dict{String,Any}(gc_solve(tname, "rand2", δval, x0_rand2)))
    else
        if prev_winner_theta === nothing
            @printf("[GC %s] WARNING: no feasible point found at any start of the previous target -- falling back to A* for the 'warm' start slot too.\n", tname)
            x0_warm = x0_Astar
        else
            x0_warm = x_from_theta(prev_winner_theta)
        end
        push!(rows, Dict{String,Any}(gc_solve(tname, "warm", δval, x0_warm)))
    end

    results_by_target[tname] = rows
    winner = gc_target_winner(rows)
    if winner === nothing
        @printf("[GC %s] NO FEASIBLE POINT FOUND at any of the 3 starts.\n", tname)
        global prev_winner_theta = nothing
    else
        @printf("[GC %s] WINNER: start=%s kappa=%.6f gp=%.6f audited_delta*=%.6g\n",
                tname, winner["start"], winner["kappa"], winner["gamma_p"], winner["audited_delta_star"])
        global prev_winner_theta = Float64.(winner["theta_star"])
    end
    flush(stdout)
end

println("\n" * "="^78); println(">>> GC HEAD-TO-HEAD SUMMARY"); println("="^78)
for tgt in TARGETS
    rows = results_by_target[tgt.name]
    winner = gc_target_winner(rows)
    if winner === nothing
        @printf("%s (delta_budget=%.4g): NO FEASIBLE RESULT\n", tgt.name, tgt.delta)
    else
        @printf("%s (delta_budget=%.4g): winner=%-6s gp=%.6f kappa=%.6f audited_delta*=%.6g relΔA=%.3f wall=%.1fs nevals=%d\n",
                tgt.name, tgt.delta, winner["start"], winner["gamma_p"], winner["kappa"],
                winner["audited_delta_star"], winner["relDeltaA"], winner["wall"], winner["num_evals"])
    end
end

open(joinpath(H2H_DIR, "gc_ALLDONE"), "w") do io
    println(io, string(Dates.now()))
end
println("\nGC_HEAD_TO_HEAD DONE")
