# ============================================================================
# LC ("local-constrained"): extremize gamma'_focal s.t. delta*<=budget, via the corrected
# KNITRO gradient-based outer search (gradient_method=:fixed_dual_fd_full,
# use_var_scaling=true, scaling_power=1.0 -- the setting that actually produced the D=20
# scaled delta-grid results, see D20_METHOD_WRITEUP.md and the scaling_power-bug fix in
# run_profiled_production.jl's own comments; NOT the mislabeled "scaling_power=0.5" the
# batch_out folder name suggests -- that label was wrong, the values were computed at 1.0).
#
# 9 solves (3 targets x 3 starts), sequential (no intra-job parallelism -- see
# HEAD_TO_HEAD_PROMPT.md rationale: KNITRO's floating license is only validated to 4-way
# concurrency, already used up by the 4 concurrent method-jobs). Each solve capped at 90 min
# via full_aod_diag/csw_outer_90min.opt (maxtime_real=5400). Bound: upper only
# (find_smallest=true, maximizes kappa).
#
# Reuses run_profiled_production.jl's own `run_one_bound(name, fs, δval, θinit)` UNCHANGED --
# it already accepts an arbitrary θinit (not just the module's default θr0) and already
# implements best-feasible tracking (make_stateful_moments) + a fresh re-verification of the
# best point before returning (NOT the bug HEAD_TO_HEAD_PROMPT.md warns about -- that's
# already fixed here). This driver's only job is the 9-solve loop, warm-start bookkeeping,
# per-solve JLD2 checkpointing (skip-if-done, so a crash only loses the IN-FLIGHT solve), and
# the post-hoc exact-delta* audit at each target's winning point.
#
#   FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true \
#     julia -t 19 --project=. sequential_gravity/head_to_head/run_lc.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
# The corrected, validated D=20 setting (D20_METHOD_WRITEUP.md) -- fixed for this whole
# comparison, not a free choice per solve.
ENV["GRADIENT_METHOD"] = "fixed_dual_fd_full"
ENV["USE_VAR_SCALING"] = "true"
ENV["SCALING_POWER"] = "1.0"
# 90-min uniform per-solve wall-clock cap (see common.jl CAP_SECONDS=5400 -- the two must
# match; this env var is what actually enforces it, common.jl's constant is just documentation
# + used by other methods' BBO_MAXTIME). Env-overridable for smoke-testing with a short-cap
# opt file.
ENV["OUTER_OPT_FILE"] = get(ENV, "OUTER_OPT_FILE", joinpath(@__DIR__, "..", "..", "full_aod_diag", "csw_outer_90min.opt"))

include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
include(joinpath(@__DIR__, "common.jl"))

@assert D == 20 && W == 80000 "this driver is D=20/W=80000-specific"
@assert isfile(ENV["OUTER_OPT_FILE"]) "capped opt file missing: $(ENV["OUTER_OPT_FILE"])"

const OUT_DIR_LC = joinpath(H2H_DIR, "out_lc")
isdir(OUT_DIR_LC) || mkpath(OUT_DIR_LC)
result_path_lc(tname, sname) = joinpath(OUT_DIR_LC, "lc_$(tname)_$(sname).jld2")

starts = load_shared_starts()
const Acol_star_h2h = θr0[4:3+D]

println("\n" * "="^78)
@printf(">>> LC (local-constrained) head-to-head: D=%d W=%d cap=%.0fs/solve\n", D, W, CAP_SECONDS)
println("="^78)
flush(stdout)

"""
    lc_solve(tname, sname, δval, θinit)

Runs (or, if already done, loads) one LC solve. Saves a per-solve JLD2 with the same
core schema as run_profiled_production.jl's own batch loop, PLUS a post-hoc exact-delta*
audit at the verified best-feasible endpoint (exact_inner_divergence_at -- the "true delta*
achieved" HEAD_TO_HEAD_PROMPT.md's Final Comparison section requires, not just the nominal
budget the search was given).
"""
function lc_solve(tname::String, sname::String, δval::Float64, θinit::Vector{Float64})
    path = result_path_lc(tname, sname)
    existing = load_done(path)
    if existing !== nothing
        @printf("[LC %s/%s] ALREADY DONE (resume) -- best_feasible_gp=%.6f kappa=%.6f\n",
                tname, sname, existing["best_feasible_gp"], existing["best_feasible_kappa"])
        flush(stdout)
        return existing
    end
    @printf("\n[LC %s/%s] delta_budget=%.4g -- starting solve (cap=%.0fs)\n", tname, sname, δval, CAP_SECONDS)
    flush(stdout)
    r = run_one_bound(Symbol("LC_$(tname)_$(sname)"), true, δval, θinit)

    audited_δstar = NaN; audited_ok = false
    if r.best_θ !== nothing
        t0 = time()
        audit = exact_inner_divergence_at(r.best_θ)
        audited_δstar = audit.δ_star
        audited_ok = audit.gravity_ok && !istop_delta(audited_δstar)
        @printf("[LC %s/%s] post-hoc audit: exact delta* at best-feasible endpoint = %.6f (nominal budget=%.4g)  wall=%.1fs\n",
                tname, sname, audited_δstar, δval, time() - t0)
    end

    d = Dict(
        "method" => "LC", "target" => tname, "start" => sname, "delta_budget" => δval,
        "theta_init" => θinit, "theta_star" => r.θstar, "gamma_p" => r.gp, "kappa" => r.κ,
        "nStatus" => r.nStatus, "R_mean_at_solution" => r.R, "gravity_feasible" => r.ok,
        "best_feasible_kappa" => r.best_κ, "best_feasible_gp" => r.best_gp,
        "best_feasible_theta" => r.best_θ, "best_feasible_gravity_ok" => r.best_ok,
        "audited_delta_star" => audited_δstar, "audited_gravity_ok" => audited_ok,
        "wall" => r.wall, "inner_solves" => r.cache.n_inner_solve, "grad_computations" => r.cache.n_grad_compute,
        "gradient_method" => "fixed_dual_fd_full", "use_var_scaling" => true, "scaling_power" => 1.0,
        "cap_seconds" => CAP_SECONDS, "done" => true, "timestamp" => string(Dates.now()))
    JLD2.save(path, d)
    @printf("[LC %s/%s] DONE: best_feasible_gp=%.6f kappa=%.6f audited_delta*=%.6f wall=%.1fs\n",
            tname, sname, r.best_gp, r.best_κ, audited_δstar, r.wall)
    flush(stdout)
    d
end

"""
    lc_target_winner(rows)

Among a target's 3 completed solves, picks the one with the HIGHEST best_feasible_kappa
(upper bound => maximize kappa). Returns `nothing` if NONE of the 3 starts found any
feasible point at all (a legitimate, must-be-reported-honestly outcome, not an error).
"""
function lc_target_winner(rows::Vector{Dict{String,Any}})
    feas = filter(r -> r["best_feasible_theta"] !== nothing, rows)
    isempty(feas) && return nothing
    feas[argmax([r["best_feasible_kappa"] for r in feas])]
end

results_by_target = Dict{String,Vector{Dict{String,Any}}}()
prev_winner = nothing   # full theta (mu,sigma,gp,Acol) of the previous target's winner

for (i, tgt) in enumerate(TARGETS)
    tname = tgt.name; δval = tgt.delta
    println("\n" * "-"^78)
    @printf("[LC] TARGET %s: delta_budget=%.4g (reference gp=%.6f kappa=%.6f)\n", tname, δval, tgt.gp, tgt.kappa)
    println("-"^78)
    flush(stdout)

    θ_Astar = copy(θr0)
    θ_rand1 = copy(θr0); θ_rand1[4:3+D] .= starts.rand1

    rows = Dict{String,Any}[]
    push!(rows, Dict{String,Any}(lc_solve(tname, "Astar", δval, θ_Astar)))
    push!(rows, Dict{String,Any}(lc_solve(tname, "rand1", δval, θ_rand1)))

    if i == 1
        θ_rand2 = copy(θr0); θ_rand2[4:3+D] .= starts.rand2
        push!(rows, Dict{String,Any}(lc_solve(tname, "rand2", δval, θ_rand2)))
    else
        if prev_winner === nothing
            @printf("[LC %s] WARNING: no feasible point found at ANY start of the previous target -- falling back to A* for the 'warm' start slot too.\n", tname)
            θ_warm = copy(θr0)
        else
            θ_warm = copy(prev_winner)
        end
        push!(rows, Dict{String,Any}(lc_solve(tname, "warm", δval, θ_warm)))
    end

    results_by_target[tname] = rows
    winner = lc_target_winner(rows)
    if winner === nothing
        @printf("[LC %s] NO FEASIBLE POINT FOUND at any of the 3 starts.\n", tname)
        global prev_winner = nothing
    else
        @printf("[LC %s] WINNER: start=%s kappa=%.6f gp=%.6f audited_delta*=%.6f\n",
                tname, winner["start"], winner["best_feasible_kappa"], winner["best_feasible_gp"], winner["audited_delta_star"])
        global prev_winner = Float64.(winner["best_feasible_theta"])
    end
    flush(stdout)
end

println("\n" * "="^78); println(">>> LC HEAD-TO-HEAD SUMMARY"); println("="^78)
for tgt in TARGETS
    rows = results_by_target[tgt.name]
    winner = lc_target_winner(rows)
    if winner === nothing
        @printf("%s (delta_budget=%.4g): NO FEASIBLE RESULT\n", tgt.name, tgt.delta)
    else
        relΔA = norm(winner["best_feasible_theta"][4:3+D] .- Acol_star_h2h) / norm(Acol_star_h2h)
        @printf("%s (delta_budget=%.4g): winner=%-6s gp=%.6f kappa=%.6f audited_delta*=%.6f relΔA=%.3f wall=%.1fs\n",
                tgt.name, tgt.delta, winner["start"], winner["best_feasible_gp"], winner["best_feasible_kappa"],
                winner["audited_delta_star"], relΔA, winner["wall"])
    end
end

open(joinpath(H2H_DIR, "lc_ALLDONE"), "w") do io
    println(io, string(Dates.now()))
end
println("\nLC_HEAD_TO_HEAD DONE")
