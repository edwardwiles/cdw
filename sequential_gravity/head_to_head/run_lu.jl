# ============================================================================
# LU ("local-unconstrained/profiled"): fix gamma'_focal at a target GT, minimize
# delta*(A_od) alone via KNITRO (existing corrected fixed-dual-FD gradient trick), no outer
# divergence constraint at all. Generalizes
# derivative_diagnostics/run_profiled_delta_star_min_knitro_d20_real.jl (which hardcoded
# GT/starts to the single delta=1.0 saved LC result) to the 3-target x 3-start design, and
# ADDS the two disciplines that script did not have:
#   1. best-feasible tracking across ALL evaluations (the raw KNITRO endpoint is NOT
#      reliable -- see HEAD_TO_HEAD_PROMPT.md's known-bugs list; the original script trusted
#      it directly).
#   2. periodic + final JLD2 checkpointing (the original script wrote exactly once, at the
#      very end -- an interrupted run lost everything).
# Also explicitly treats any delta_star>=MAX_SANE_DELTA_STAR (the observed 1e10 sentinel
# failure mode) as an infeasible evaluation, not a real number.
#
# 9 solves (3 targets x 3 starts), sequential. Each solve capped at 90 min via KNITRO's own
# maxtime_real (full_aod_diag/csw_outer_90min.opt).
#
#   FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true \
#     julia -t 19 --project=. sequential_gravity/head_to_head/run_lu.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
include(joinpath(@__DIR__, "common.jl"))
using KNITRO

@assert D == 20 && W == 80000 "this driver is D=20/W=80000-specific"

const KNITRO_OPT_FILE_H2H = get(ENV, "KNITRO_OPT_FILE", joinpath(@__DIR__, "..", "..", "full_aod_diag", "csw_outer_90min.opt"))
@assert isfile(KNITRO_OPT_FILE_H2H) "capped opt file missing: $KNITRO_OPT_FILE_H2H"
const H_FD_H2H = parse(Float64, get(ENV, "H_FD", "0.1"))

const OUT_DIR_LU = joinpath(H2H_DIR, "out_lu")
isdir(OUT_DIR_LU) || mkpath(OUT_DIR_LU)
result_path_lu(tname, sname) = joinpath(OUT_DIR_LU, "lu_$(tname)_$(sname).jld2")

starts = load_shared_starts()
const Acol_star_h2h = θr0[4:3+D]
const mu0 = θr0[1]
const sigma0 = θr0[2]
const lo_h2h = θ_lo[4:3+D]
const hi_h2h = θ_hi[4:3+D]

println("\n" * "="^78)
@printf(">>> LU (local-unconstrained/profiled) head-to-head: D=%d W=%d cap=%.0fs/solve\n", D, W, CAP_SECONDS)
println("="^78)
flush(stdout)

"""
    minimize_delta_star_h2h(x0, GT; tname, sname, ckpt_path)

KNITRO local search minimizing delta*(A_od) at FIXED gamma'_focal=GT, starting from x0.
Tracks the best GRAVITY-FEASIBLE, non-sentinel evaluation seen across the whole search (not
the raw KNITRO endpoint), and checkpoints that running best to `ckpt_path` every 5
evaluations plus once more at the very end (done=true). Same eval_fcga/FieldError/absolute-
opt-file-path fixes as the original run_profiled_delta_star_min_knitro_d20_real.jl.
"""
function minimize_delta_star_h2h(x0::Vector{Float64}, GT::Float64; tname::String, sname::String, ckpt_path::String)
    @printf("\n  --- LU %s/%s: KNITRO minimizing delta*(A_od), GT=%.9f ---\n", tname, sname, GT)
    flush(stdout)
    last_x = Ref(fill(NaN, D)); last_delta = Ref(Inf)
    last_frozen = Ref{Any}(nothing); last_xstar = Ref{Vector{Float64}}(Float64[])
    last_theta = Ref{Vector{Float64}}(Float64[])
    best_delta = Ref(Inf); best_x = Ref(copy(x0))
    neval = Ref(0)
    t_start = time()

    function save_ckpt(done::Bool, raw_status::Integer)
        relΔA = norm(best_x[] .- Acol_star_h2h) / norm(Acol_star_h2h)
        JLD2.save(ckpt_path, Dict(
            "method" => "LU", "target" => tname, "start" => sname, "gammap_target" => GT,
            "Aod_init" => x0, "nStatus" => raw_status,
            "best_feasible_delta_star" => best_delta[], "best_feasible_Aod" => best_x[],
            "num_evals" => neval[], "relDeltaA" => relΔA, "wall" => time() - t_start,
            "cap_seconds" => CAP_SECONDS, "done" => done, "timestamp" => string(Dates.now())))
        relΔA
    end

    function ensure!(x_free::Vector{Float64})
        x_free == last_x[] && return
        θ = vcat(mu0, sigma0, GT, x_free)
        res = exact_inner_divergence_at(θ)
        last_x[] = copy(x_free); last_theta[] = θ
        δ_eff = istop_delta(res.δ_star) ? Inf : res.δ_star
        last_delta[] = δ_eff
        if isfinite(res.δ_star) && hasproperty(res, :frozen)
            last_frozen[] = res.frozen; last_xstar[] = collect(res.x_star)
        else
            last_frozen[] = nothing; last_xstar[] = Float64[]
        end
        if res.gravity_ok && isfinite(δ_eff) && δ_eff < best_delta[]
            best_delta[] = δ_eff; best_x[] = copy(x_free)
        end
        neval[] += 1
        @printf("    [eval %d, %.1fs] delta*=%.8g gravity_ok=%s best_feasible_so_far=%.8g\n",
                neval[], time() - t_start, res.δ_star, res.gravity_ok, best_delta[])
        flush(stdout)
        neval[] % 5 == 0 && save_ckpt(false, -1)
    end

    function fill_grad!(g, x_free)
        frozen = last_frozen[]
        if !isfinite(last_delta[]) || frozen === nothing || !frozen.ok
            g .= 0.0
            return
        end
        frozen_moments = make_frozen_gravity_moments(EK_moments_focal_norm_directgp!, D, frozen.θ, frozen.Rcol, frozen.gcol, frozen.dRdθ)
        θ = last_theta[]
        grad_log = fixed_dual_fd_gradient(θ, γ, U, frozen_moments, D + 2, last_xstar[], H_FD_H2H;
            l = length(θ), Acol_offset = 3, find_smallest = true)
        g .= grad_log ./ x_free
    end

    function cb_FG!(kc2, cb, evalRequest, evalResult, userParams)
        x_free = evalRequest.x
        ensure!(x_free)
        evalResult.obj[1] = isfinite(last_delta[]) ? last_delta[] : 1e10
        fill_grad!(evalResult.objGrad, x_free)
        return 0
    end
    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        ensure!(evalRequest.x)
        evalResult.obj[1] = isfinite(last_delta[]) ? last_delta[] : 1e10
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        x_free = evalRequest.x
        ensure!(x_free)
        fill_grad!(evalResult.objGrad, x_free)
        return 0
    end

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, KNITRO_OPT_FILE_H2H)
    KNITRO.KN_add_vars(kc, D)
    KNITRO.KN_set_var_lobnds_all(kc, lo_h2h)
    KNITRO.KN_set_var_upbnds_all(kc, hi_h2h)
    KNITRO.KN_set_var_primal_init_values_all(kc, x0)
    if KNITRO.KN_get_int_param(kc, "eval_fcga") == 1
        KNITRO.KN_add_eval_callback(kc, true, Int32[], cb_FG!)
    else
        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], cb_F!)
        KNITRO.KN_set_cb_grad(kc, cb, cb_G!)
    end

    KNITRO.KN_solve(kc)
    nStatus, objv, x_min, lambda_ = KNITRO.KN_get_solution(kc)
    wall = time() - t_start
    KNITRO.KN_free(kc)

    relΔA = save_ckpt(true, nStatus)
    @printf("  [LU %s/%s] KNITRO raw status=%d raw_endpoint_delta*=%.6g  BEST-FEASIBLE delta*=%.8g  nevals=%d wall=%.1fs relΔA=%.4f\n",
            tname, sname, nStatus, objv, best_delta[], neval[], wall, relΔA)
    flush(stdout)
    (nStatus = nStatus, raw_delta = objv, best_delta = best_delta[], best_x = best_x[],
     nevals = neval[], wall = wall, relΔA = relΔA)
end

function lu_solve(tname::String, sname::String, GT::Float64, Aod_init::Vector{Float64})
    path = result_path_lu(tname, sname)
    existing = load_done(path)
    if existing !== nothing
        @printf("[LU %s/%s] ALREADY DONE (resume) -- best_feasible_delta_star=%.6g\n",
                tname, sname, existing["best_feasible_delta_star"])
        flush(stdout)
        return existing
    end
    r = minimize_delta_star_h2h(copy(Aod_init), GT; tname = tname, sname = sname, ckpt_path = path)
    load_done(path)   # minimize_delta_star_h2h's own save_ckpt(true,...) already wrote the final schema
end

"""
    lu_target_winner(rows)

Among a target's 3 completed solves, picks the LOWEST best_feasible_delta_star (cheapest way
to reach the FIXED gp target -- kappa/gp don't vary across LU's 3 starts by construction, so
"best" here means lowest divergence cost, not highest kappa). `nothing` if none of the 3
starts ever found a gravity-feasible point.
"""
function lu_target_winner(rows::Vector{Dict{String,Any}})
    feas = filter(r -> isfinite(r["best_feasible_delta_star"]), rows)
    isempty(feas) && return nothing
    feas[argmin([r["best_feasible_delta_star"] for r in feas])]
end

# SKIP_MAIN_LOOP (additive, off by default): lets a sibling driver (e.g. an expanded
# multistart script) `include` this file purely for its setup/machinery (minimize_delta_star_h2h,
# lu_solve, lu_target_winner, Acol_star_h2h, mu0/sigma0, KNITRO_OPT_FILE_H2H, etc.) without
# re-running the official 9-solve loop below.
if lowercase(get(ENV, "SKIP_MAIN_LOOP", "false")) != "true"

results_by_target = Dict{String,Vector{Dict{String,Any}}}()
prev_winner_Aod = nothing

for (i, tgt) in enumerate(TARGETS)
    tname = tgt.name; GT = tgt.gp
    println("\n" * "-"^78)
    @printf("[LU] TARGET %s: GT=gamma'_focal=%.9f (kappa=%.6f, reference delta_budget=%.4g)\n", tname, GT, tgt.kappa, tgt.delta)
    println("-"^78)
    flush(stdout)

    rows = Dict{String,Any}[]
    push!(rows, Dict{String,Any}(lu_solve(tname, "Astar", GT, Acol_star_h2h)))
    push!(rows, Dict{String,Any}(lu_solve(tname, "rand1", GT, starts.rand1)))

    if i == 1
        push!(rows, Dict{String,Any}(lu_solve(tname, "rand2", GT, starts.rand2)))
    else
        Aod_warm = prev_winner_Aod === nothing ? Acol_star_h2h : prev_winner_Aod
        if prev_winner_Aod === nothing
            @printf("[LU %s] WARNING: no feasible point found at any start of the previous target -- falling back to A* for the 'warm' start slot too.\n", tname)
        end
        push!(rows, Dict{String,Any}(lu_solve(tname, "warm", GT, Aod_warm)))
    end

    results_by_target[tname] = rows
    winner = lu_target_winner(rows)
    if winner === nothing
        @printf("[LU %s] NO FEASIBLE POINT FOUND at any of the 3 starts.\n", tname)
        global prev_winner_Aod = nothing
    else
        @printf("[LU %s] WINNER: start=%s delta_star=%.6g relΔA=%.3f\n",
                tname, winner["start"], winner["best_feasible_delta_star"], winner["relDeltaA"])
        global prev_winner_Aod = Float64.(winner["best_feasible_Aod"])
    end
    flush(stdout)
end

println("\n" * "="^78); println(">>> LU HEAD-TO-HEAD SUMMARY"); println("="^78)
for tgt in TARGETS
    rows = results_by_target[tgt.name]
    winner = lu_target_winner(rows)
    if winner === nothing
        @printf("%s (GT=%.6f): NO FEASIBLE RESULT\n", tgt.name, tgt.gp)
    else
        @printf("%s (GT=%.6f, kappa=%.6f): winner=%-6s delta_star=%.6g relΔA=%.3f nevals=%d wall=%.1fs\n",
                tgt.name, tgt.gp, tgt.kappa, winner["start"], winner["best_feasible_delta_star"],
                winner["relDeltaA"], winner["num_evals"], winner["wall"])
    end
end

open(joinpath(H2H_DIR, "lu_ALLDONE"), "w") do io
    println(io, string(Dates.now()))
end
println("\nLU_HEAD_TO_HEAD DONE")
end # SKIP_MAIN_LOOP guard
