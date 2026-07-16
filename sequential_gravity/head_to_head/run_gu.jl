# ============================================================================
# GU ("global-unconstrained/profiled"): fix gamma'_focal at a target GT, minimize
# delta*(A_od) alone via BlackBoxOptim (adaptive DE) over logratio = log(A_od/A_od*),
# LOGBOUND=3.0. Generalizes global_opt/run_bbo_d20_profiled_deltastar.jl (which hardcoded
# GT to the single delta=1.0 saved LC result, and only had 2 fixed starts) to the 3-target x
# 3-start design. Already-validated machinery reused as-is: per-BBO-step JLD2 checkpointing,
# best-feasible-by-construction fitness (a huge penalty for any gravity-infeasible or
# delta_star>=MAX_SANE_DELTA_STAR candidate, so BlackBoxOptim's own best_candidate/
# best_fitness IS the best feasible point once any feasible point exists -- see
# global_optimizer_report.md section 4.2 and bbo_common.jl's sibling pattern).
#
# ADDS population-seeding at the actual starting point for each solve (the original driver's
# 6h production run did not need this since it only ever started cold at A* or the
# constrained-search endpoint; this driver's 3rd/warm start at T2/T3 needs it) via
# BlackBoxOptim's own bboptimize(fitness, x0, params...) two-arg form.
#
# 9 solves (3 targets x 3 starts), sequential, each capped at 90 min via BBO's own MaxTime
# (much shorter than the 6h production run this method's own report used -- see the
# fair-comparison caveat in the final report: GU is the method most likely to be
# budget-starved by the uniform 90-min cap, since its own single 6h run was STILL improving
# at cutoff).
#
#   FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true \
#     julia -t 19 --project=. sequential_gravity/head_to_head/run_gu.jl
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

const POPSIZE_GU = parse(Int, get(ENV, "BBO_POPSIZE", "12"))   # matches the validated 6h production run's own choice

const OUT_DIR_GU = joinpath(H2H_DIR, "out_gu")
isdir(OUT_DIR_GU) || mkpath(OUT_DIR_GU)
result_path_gu(tname, sname) = joinpath(OUT_DIR_GU, "gu_$(tname)_$(sname).jld2")

starts = load_shared_starts()
const Acol_star_h2h = θr0[4:3+D]

println("\n" * "="^78)
@printf(">>> GU (global-unconstrained/profiled) head-to-head: D=%d W=%d cap=%.0fs/solve LOGBOUND=%.2f PopulationSize=%d\n",
        D, W, CAP_SECONDS, LOGBOUND, POPSIZE_GU)
println("="^78)
flush(stdout)

function build_theta_profiled_gu(logratio::AbstractVector, GT::Float64)
    θ = copy(θr0)
    θ[3] = GT
    θ[4:3+D] .= Acol_star_h2h .* exp.(logratio)
    θ
end

function make_profiled_fitness(GT::Float64)
    function profiled_fitness(logratio::AbstractVector)
        θ = build_theta_profiled_gu(logratio, GT)
        local r
        try
            r = exact_inner_divergence_at(θ)
        catch
            return 1.0e4 + 1.0e-3 * norm(logratio)
        end
        (r.gravity_ok && !istop_delta(r.δ_star)) || return 1.0e4 + 1.0e-3 * norm(logratio)
        r.δ_star
    end
    profiled_fitness
end

logratio_range() = [(-LOGBOUND, LOGBOUND) for _ in 1:D]

"""
    gu_solve(tname, sname, GT, x0)

`x0` is a logratio vector (D-dim). Seeds the population at x0, runs adaptive DE up to
CAP_SECONDS minimizing delta*(A_od) at the fixed target GT, checkpointing every step.
"""
function gu_solve(tname::String, sname::String, GT::Float64, x0::Vector{Float64})
    path = result_path_gu(tname, sname)
    existing = load_done(path)
    if existing !== nothing
        @printf("[GU %s/%s] ALREADY DONE (resume) -- delta_star_best=%.6g\n",
                tname, sname, existing["delta_star_best"])
        flush(stdout)
        return existing
    end
    @printf("\n[GU %s/%s] GT=%.9f -- starting solve (cap=%.0fs, seeded at x0)\n", tname, sname, GT, CAP_SECONDS)
    flush(stdout)

    fitness = make_profiled_fitness(GT)

    function checkpoint_callback(oc)
        x = best_candidate(oc); f = best_fitness(oc)
        Acol_best = Acol_star_h2h .* exp.(x)
        relΔA = norm(Acol_best .- Acol_star_h2h) / norm(Acol_star_h2h)
        @printf("[GU %s/%s checkpoint %s] fevals=%d best_delta_star=%.6g relΔA=%.3f\n",
                tname, sname, Dates.format(Dates.now(), "HH:MM:SS"), num_func_evals(oc), f, relΔA)
        flush(stdout)
        try
            JLD2.save(path, Dict(
                "method" => "GU", "target" => tname, "start" => sname, "gammap_target" => GT,
                "logratio_best" => x, "delta_star_best" => f, "num_evals" => num_func_evals(oc),
                "Acol_best" => Acol_best, "relDeltaA" => relΔA,
                "cap_seconds" => CAP_SECONDS, "LOGBOUND" => LOGBOUND,
                "done" => false, "timestamp" => string(Dates.now())))
        catch e
            @warn "GU checkpoint save failed" exception = e
        end
    end

    t0 = time()
    res = bboptimize(fitness, x0;
        SearchRange = logratio_range(), NumDimensions = D,
        Method = :adaptive_de_rand_1_bin_radiuslimited,
        PopulationSize = POPSIZE_GU, MaxTime = CAP_SECONDS,
        TraceMode = :compact, TraceInterval = 30.0,
        CallbackFunction = checkpoint_callback, CallbackInterval = 0.0)
    wall = time() - t0

    xbest = best_candidate(res); δbest = best_fitness(res)
    Acol_best = Acol_star_h2h .* exp.(xbest)
    relΔA = norm(Acol_best .- Acol_star_h2h) / norm(Acol_star_h2h)

    d = Dict(
        "method" => "GU", "target" => tname, "start" => sname, "gammap_target" => GT,
        "logratio_best" => xbest, "delta_star_best" => δbest, "num_evals" => f_calls(res),
        "Acol_best" => Acol_best, "relDeltaA" => relΔA,
        "wall" => wall, "cap_seconds" => CAP_SECONDS, "LOGBOUND" => LOGBOUND,
        "done" => true, "timestamp" => string(Dates.now()))
    JLD2.save(path, d)
    @printf("[GU %s/%s] DONE: delta_star_best=%.6g relΔA=%.3f wall=%.1fs nevals=%d\n",
            tname, sname, δbest, relΔA, wall, f_calls(res))
    flush(stdout)
    d
end

"""
    gu_target_winner(rows)

Lowest delta_star_best across the target's 3 starts (gp is fixed by construction, so
"best" = cheapest way to reach it), same convention as LU. `nothing` if every start's best
is still an infeasible-penalty value (>= 1e4, well above any legitimate delta*).
"""
function gu_target_winner(rows::Vector{Dict{String,Any}})
    feas = filter(r -> r["delta_star_best"] < 1.0e3, rows)
    isempty(feas) && return nothing
    feas[argmin([r["delta_star_best"] for r in feas])]
end

results_by_target = Dict{String,Vector{Dict{String,Any}}}()
prev_winner_logratio = nothing

for (i, tgt) in enumerate(TARGETS)
    tname = tgt.name; GT = tgt.gp
    println("\n" * "-"^78)
    @printf("[GU] TARGET %s: GT=gamma'_focal=%.9f (kappa=%.6f, reference delta_budget=%.4g)\n", tname, GT, tgt.kappa, tgt.delta)
    println("-"^78)
    flush(stdout)

    x0_Astar = zeros(D)
    x0_rand1 = log.(starts.rand1 ./ Acol_star_h2h)

    rows = Dict{String,Any}[]
    push!(rows, Dict{String,Any}(gu_solve(tname, "Astar", GT, x0_Astar)))
    push!(rows, Dict{String,Any}(gu_solve(tname, "rand1", GT, x0_rand1)))

    if i == 1
        x0_rand2 = log.(starts.rand2 ./ Acol_star_h2h)
        push!(rows, Dict{String,Any}(gu_solve(tname, "rand2", GT, x0_rand2)))
    else
        if prev_winner_logratio === nothing
            @printf("[GU %s] WARNING: no feasible point found at any start of the previous target -- falling back to A* for the 'warm' start slot too.\n", tname)
            x0_warm = x0_Astar
        else
            x0_warm = prev_winner_logratio
        end
        push!(rows, Dict{String,Any}(gu_solve(tname, "warm", GT, x0_warm)))
    end

    results_by_target[tname] = rows
    winner = gu_target_winner(rows)
    if winner === nothing
        @printf("[GU %s] NO FEASIBLE POINT FOUND at any of the 3 starts.\n", tname)
        global prev_winner_logratio = nothing
    else
        @printf("[GU %s] WINNER: start=%s delta_star=%.6g relΔA=%.3f\n",
                tname, winner["start"], winner["delta_star_best"], winner["relDeltaA"])
        global prev_winner_logratio = Float64.(winner["logratio_best"])
    end
    flush(stdout)
end

println("\n" * "="^78); println(">>> GU HEAD-TO-HEAD SUMMARY"); println("="^78)
for tgt in TARGETS
    rows = results_by_target[tgt.name]
    winner = gu_target_winner(rows)
    if winner === nothing
        @printf("%s (GT=%.6f): NO FEASIBLE RESULT\n", tgt.name, tgt.gp)
    else
        @printf("%s (GT=%.6f, kappa=%.6f): winner=%-6s delta_star=%.6g relΔA=%.3f nevals=%d wall=%.1fs\n",
                tgt.name, tgt.gp, tgt.kappa, winner["start"], winner["delta_star_best"],
                winner["relDeltaA"], winner["num_evals"], winner["wall"])
    end
end

open(joinpath(H2H_DIR, "gu_ALLDONE"), "w") do io
    println(io, string(Dates.now()))
end
println("\nGU_HEAD_TO_HEAD DONE")
