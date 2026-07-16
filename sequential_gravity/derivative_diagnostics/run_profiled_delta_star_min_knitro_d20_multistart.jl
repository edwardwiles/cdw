# ============================================================================
# D=20 REAL-DATA multi-start extension of the KNITRO profiled delta*(A_od)
# minimization test (run_profiled_delta_star_min_knitro_d20_real.jl). That
# script's 2-point result found genuine local-optima multiplicity: starting
# from A* converges to delta*=1.083 (a WORSE local optimum, still above the
# delta=1 budget), while starting from the scaled constrained search's own
# A_od finds that point IS ALREADY a genuine local optimum (delta*=1.00004).
#
# This script adds 3 NEW starting points (log-normal multiplicative
# perturbations of Acol, matching the precedent in multistart_screening_d20.jl
# -- that screening found 20/20 such perturbations gravity-feasible even out
# to ~20x relative distance from A*), to see whether EITHER already-found
# basin is robust to nearby starts, or whether a THIRD, better basin exists:
#   3. sigma=0.3 around A*                         (mild, near the WORSE basin)
#   4. sigma=0.3 around the scaled-search endpoint  (mild, near the BETTER basin)
#   5. sigma=1.0 around A*                          (more distant, tests a new basin)
# The 2 already-computed points (A* and the scaled-search endpoint) are
# reused from the saved JLD2, NOT re-run (identical setup, deterministic
# KNITRO given a fixed starting point, no need to re-pay ~7-400s each).
#
#   FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true \
#     julia -t 19 --project=. sequential_gravity/derivative_diagnostics/run_profiled_delta_star_min_knitro_d20_multistart.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))

using KNITRO, Printf, JLD2, LinearAlgebra, Random

const PRIOR_RESULT_PATH = joinpath(@__DIR__, "profiled_delta_star_min_knitro_D20_W80000_delta1.0_upper_real.jld2")
const KNITRO_OPT_FILE = get(ENV, "KNITRO_OPT_FILE", joinpath(@__DIR__, "..", "..", "full_aod_diag", "csw_outer_1000.opt"))
const H_FD = parse(Float64, get(ENV, "H_FD", "0.1"))
const SEED = parse(Int, get(ENV, "SEED", "20260715"))   # same seed as multistart_screening_d20.jl for continuity

@assert D == 20 && W == 80000
@assert isfile(KNITRO_OPT_FILE) "KNITRO_OPT_FILE does not exist: $KNITRO_OPT_FILE"
@assert isfile(PRIOR_RESULT_PATH) "prior 2-point result not found -- run run_profiled_delta_star_min_knitro_d20_real.jl first"

prior = JLD2.load(PRIOR_RESULT_PATH)
gammap_star = prior["gammap_star"]
κ_saved = prior["kappa_saved"]
Acol_star = prior["Acol_star"]
Aod_constrained = prior["Aod_constrained"]
@assert isapprox(θr0[4:3+D], Acol_star; rtol=1e-8) "this run's A* does not match the prior result's A* -- real-data setup mismatch"

println("="^78)
println(">>> D=20 REAL DATA multi-start: KNITRO profiled delta*(A_od), gamma'=$(gammap_star) (kappa=$(κ_saved))")
println("="^78)

const mu0 = θr0[1]
const sigma0 = θr0[2]
const lo = θ_lo[4:3+D]
const hi = θ_hi[4:3+D]

function minimize_delta_star_knitro(x0::Vector{Float64}; label::String)
    println("\n  --- KNITRO minimizing delta*(A_od) from $label ---")
    flush(stdout)
    last_x = Ref(fill(NaN, D))
    last_delta = Ref(Inf)
    last_frozen = Ref{Any}(nothing)
    last_xstar = Ref{Vector{Float64}}(Float64[])
    last_theta = Ref{Vector{Float64}}(Float64[])
    neval = Ref(0)
    t_start = time()

    function ensure!(x_free::Vector{Float64})
        x_free == last_x[] && return
        θ = vcat(mu0, sigma0, gammap_star, x_free)
        res = exact_inner_divergence_at(θ)
        last_x[] = copy(x_free)
        last_delta[] = res.δ_star
        last_theta[] = θ
        if isfinite(res.δ_star) && hasproperty(res, :frozen)
            last_frozen[] = res.frozen
            last_xstar[] = collect(res.x_star)
        else
            last_frozen[] = nothing
            last_xstar[] = Float64[]
        end
        neval[] += 1
        @printf("    [eval %d, %.1fs] delta*=%.8f gravity_ok=%s\n", neval[], time() - t_start, res.δ_star, res.gravity_ok)
        flush(stdout)
    end

    function fill_grad!(g, x_free)
        frozen = last_frozen[]
        if !isfinite(last_delta[]) || frozen === nothing || !frozen.ok
            g .= 0.0
            return
        end
        frozen_moments = make_frozen_gravity_moments(EK_moments_focal_norm_directgp!, D, frozen.θ, frozen.Rcol, frozen.gcol, frozen.dRdθ)
        θ = last_theta[]
        grad_log = fixed_dual_fd_gradient(θ, γ, U, frozen_moments, D + 2, last_xstar[], H_FD;
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
    KNITRO.KN_load_param_file(kc, KNITRO_OPT_FILE)
    KNITRO.KN_add_vars(kc, D)
    KNITRO.KN_set_var_lobnds_all(kc, lo)
    KNITRO.KN_set_var_upbnds_all(kc, hi)
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

    relΔA = norm(x_min .- Acol_star) / norm(Acol_star)
    @printf("  [%s] KNITRO status=%d  delta*_min=%.8f  nevals=%d  wall=%.1fs  rel‖A_od_min-A*‖=%.4f\n",
        label, nStatus, objv, neval[], wall, relΔA)
    flush(stdout)
    (label = label, x0 = x0, nStatus = nStatus, δmin = objv, xmin = collect(x_min), nevals = neval[], wall = wall, relΔA = relΔA)
end

# ---- generate 3 new starting points (log-normal Acol perturbations, matching
# multistart_screening_d20.jl's own scheme) ----
Random.seed!(SEED)
x0_pert_Astar_mild = Acol_star .* exp.(0.3 .* randn(D))
x0_pert_constrained_mild = Aod_constrained .* exp.(0.3 .* randn(D))
x0_pert_Astar_wide = Acol_star .* exp.(1.0 .* randn(D))

@printf("\nNew starting points (seed=%d):\n", SEED)
@printf("  3. sigma=0.3 around A*                        relΔA-from-A*=%.4f\n", norm(x0_pert_Astar_mild .- Acol_star)/norm(Acol_star))
@printf("  4. sigma=0.3 around scaled-search endpoint     relΔA-from-A*=%.4f\n", norm(x0_pert_constrained_mild .- Acol_star)/norm(Acol_star))
@printf("  5. sigma=1.0 around A*                         relΔA-from-A*=%.4f\n", norm(x0_pert_Astar_wide .- Acol_star)/norm(Acol_star))
flush(stdout)

res3 = minimize_delta_star_knitro(x0_pert_Astar_mild; label = "3. sigma=0.3 around A*")
res4 = minimize_delta_star_knitro(x0_pert_constrained_mild; label = "4. sigma=0.3 around scaled-search endpoint")
res5 = minimize_delta_star_knitro(x0_pert_Astar_wide; label = "5. sigma=1.0 around A*")

all_results = [
    (label = "1. A_od = A*", δmin = prior["res_fromAstar"].δmin, nStatus = prior["res_fromAstar"].nStatus, nevals = prior["res_fromAstar"].nevals, wall = prior["res_fromAstar"].wall, relΔA = prior["res_fromAstar"].relΔA),
    (label = "2. A_od = scaled-search endpoint", δmin = prior["res_fromConstrained"].δmin, nStatus = prior["res_fromConstrained"].nStatus, nevals = prior["res_fromConstrained"].nevals, wall = prior["res_fromConstrained"].wall, relΔA = prior["res_fromConstrained"].relΔA),
    res3, res4, res5,
]

println("\n" * "="^78); println(">>> MULTI-START SUMMARY (D=20 real data, 5 starting points)"); println("="^78)
@printf("%-45s %10s %8s %8s %8s %10s\n", "start", "delta*_min", "status", "nevals", "wall(s)", "relΔA-A*")
for r in all_results
    @printf("%-45s %10.6f %8d %8d %8.1f %10.4f\n", r.label, r.δmin, r.nStatus, r.nevals, r.wall, r.relΔA)
end
δbest = minimum(r.δmin for r in all_results)
best_label = all_results[argmin([r.δmin for r in all_results])].label
@printf("\nBEST across all 5 starts: delta*_min=%.8f (%s)\n", δbest, best_label)
@printf("Budget was 1.0 -> gap = %.6f\n", 1.0 - δbest)

out_path = joinpath(@__DIR__, "profiled_delta_star_min_knitro_D20_multistart5.jld2")
JLD2.save(out_path, Dict("all_results" => all_results, "gammap_star" => gammap_star, "kappa_saved" => κ_saved, "seed" => SEED))
println("\nSaved: $out_path")
println("\nKNITRO D=20 MULTISTART TEST DONE")
