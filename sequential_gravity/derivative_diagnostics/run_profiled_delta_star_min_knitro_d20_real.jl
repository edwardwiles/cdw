# ============================================================================
# D=20 REAL-DATA, KNITRO-based (not derivative-free) version of the profiled
# delta*(A_od) minimization test. See run_profiled_delta_star_min_knitro.jl
# (D=4) for the full rationale/design and run_profiled_delta_star_min_d20_real.jl
# (the BlackBoxOptim D=20 version, a separate track) for the GT-reuse pattern.
#
# Per explicit user request: use LITERALLY the same optimizer (KNITRO local
# SQP/interior-point) as the existing constrained outer loop -- NOT a global/
# derivative-free method (that's a separate track, another Claude session).
# Also per explicit user request: do NOT re-run the constrained full-A search
# here (it took ~46 min with KNITRO variable scaling -- report section 7.2).
# Reuse EXACTLY the already-computed result from that run, saved at
#   sequential_gravity/batch_out_realD20_W80000_fixeddualfdfull_scaled05/seq_upper_delta1.0.jld2
# D=20, W=80000, REAL data, bound=upper, delta budget=1.0,
# gradient_method=fixed_dual_fd_full, use_var_scaling=true (scaling_power=0.5).
# best_feasible_gp = 0.950259956422648 -> kappa = 0.081518 is the "GT".
#
# Objective VALUE: exact_inner_divergence_at(theta).delta_star (real fresh
# inner solve each genuinely-new A_od). GRADIENT: the cheap fixed-dual-
# criterion FD trick (fixed_dual_fd_gradient), NOT fresh FD of the expensive
# true objective -- see run_profiled_delta_star_min_knitro.jl's docstring for
# why (would need D or 2D extra real inner solves per outer iteration,
# prohibitive at D=20 real data).
#
# Two bugs already found and fixed (D=4 validation) that this script inherits
# the fix for:
#   1. `eval_fcga=yes` (set in this project's own .opt files) requires a
#      COMBINED value+gradient callback -- registering separate cb_F!/cb_G!
#      leaves KNITRO calling ONLY cb_F!, silently never populating the
#      gradient, causing instant false "already optimal" convergence with
#      zero real search. Fixed via cb_FG! (mirrors outer_loop_cached.jl's own
#      eval_fcga branch).
#   2. exact_inner_divergence_at returns a SHORTER NamedTuple (no x_star/
#      frozen fields) when gravity-infeasible -- must guard before accessing
#      those fields, or KNITRO's C callback boundary silently swallows the
#      resulting Julia FieldError (as a "puts callback exception" warning),
#      corrupting the search.
# A THIRD bug found separately: passing KNITRO_OPT_FILE as a bare RELATIVE
# path can silently fail to load (a known cwd-resolution issue in this repo,
# see memory / prior D=20 sessions) -- KNITRO just prints
# "ERROR: Knitro could not open file ... for input." and falls back to its
# own defaults WITHOUT throwing a Julia exception or halting the script. This
# script defaults KNITRO_OPT_FILE to an ABSOLUTE path to avoid that trap.
#
#   MAXTIME_UNUSED (KNITRO's own opt-file maxit governs iteration budget, not
#   a wall-clock cap) -- just run it:
#   FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true \
#     julia -t 19 --project=. sequential_gravity/derivative_diagnostics/run_profiled_delta_star_min_knitro_d20_real.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))

using KNITRO, Printf, JLD2, LinearAlgebra

const SAVED_PATH = joinpath(@__DIR__, "..", "batch_out_realD20_W80000_fixeddualfdfull_scaled05", "seq_upper_delta1.0.jld2")
const H_FD = parse(Float64, get(ENV, "H_FD", "0.1"))
# ABSOLUTE path default -- a bare relative path (e.g. "full_aod_diag/csw_outer_1000.opt") can
# silently fail to load depending on the process's working directory at the time KN_load_param_file
# runs (see module docstring, bug 3). joinpath(@__DIR__, ...) is always absolute.
const KNITRO_OPT_FILE = get(ENV, "KNITRO_OPT_FILE", joinpath(@__DIR__, "..", "..", "full_aod_diag", "csw_outer_1000.opt"))

@assert D == 20 "this driver is D=20-specific"
@assert W == 80000 "this driver is W=80000-specific (matches the saved scaled run)"
@assert isfile(KNITRO_OPT_FILE) "KNITRO_OPT_FILE does not exist: $KNITRO_OPT_FILE"

println("="^78)
println(">>> KNITRO-based profiled delta*(A_od) minimization -- D=20 REAL DATA, W=80000")
println("    opt file: $KNITRO_OPT_FILE")
println("="^78)

saved = JLD2.load(SAVED_PATH)
@assert saved["D"] == 20 && saved["delta"] == 1.0 && saved["bound"] == "upper" &&
        saved["gradient_method"] == "fixed_dual_fd_full" && saved["use_var_scaling"] == true "saved jld2 spec mismatch -- not the section-7.2 scaled run"

gammap_star = saved["best_feasible_gp"]
κ_saved = saved["best_feasible_kappa"]
θ_saved_best = Float64.(saved["best_feasible_theta"])
Aod_constrained = θ_saved_best[4:3+D]

@printf("  sanity check: this run's θr0[1:2] = %s\n", θr0[1:2])
@printf("  sanity check: saved best_feasible_theta[1:2] = %s\n", θ_saved_best[1:2])
@assert isapprox(θr0[1], θ_saved_best[1]; rtol = 1e-8) && isapprox(θr0[2], θ_saved_best[2]; rtol = 1e-8) "MISMATCH: this driver's real-data setup does not match the saved D=20 scaled run -- do not trust results below"
println("  sanity check PASSED: mu/sigma match the saved run exactly.")
flush(stdout)

@printf("\nReusing saved GT (NOT re-solving the constrained search): gamma'_focal* = %.12f -> kappa = %.6f\n", gammap_star, κ_saved)
@printf("  (from %s)\n\n", SAVED_PATH)
flush(stdout)

const mu0 = θr0[1]
const sigma0 = θr0[2]
const Acol_star = θr0[4:3+D]
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

t0 = time()
δ_at_Astar = exact_inner_divergence_at(vcat(mu0, sigma0, gammap_star, Acol_star)).δ_star
@printf("  delta*(A_od = A*)                    = %.8f  (%.1fs)\n", δ_at_Astar, time() - t0)
flush(stdout)
t0 = time()
δ_at_constrained = exact_inner_divergence_at(vcat(mu0, sigma0, gammap_star, Aod_constrained)).δ_star
@printf("  delta*(A_od = scaled-search endpoint) = %.8f  (%.1fs)\n", δ_at_constrained, time() - t0)
flush(stdout)

res_fromAstar = minimize_delta_star_knitro(copy(Acol_star); label = "A_od = A*")
res_fromConstrained = minimize_delta_star_knitro(copy(Aod_constrained); label = "A_od = scaled-constrained-search endpoint")

println("\n" * "="^78); println(">>> SUMMARY (D=20 real data, KNITRO local)"); println("="^78)
@printf("  gamma'_focal* (GT, reused from saved scaled run)   = %.12f (kappa=%.6f)\n", gammap_star, κ_saved)
@printf("  original delta BUDGET used by that constrained run  = 1.000000\n")
@printf("  exact delta*(A*, same gamma')                       = %.8f\n", δ_at_Astar)
@printf("  exact delta*(scaled-search A_od, same gamma')       = %.8f\n", δ_at_constrained)
@printf("  KNITRO-minimized delta*, start=A*                          = %.8f  (status=%d, nevals=%d, wall=%.1fs)\n",
    res_fromAstar.δmin, res_fromAstar.nStatus, res_fromAstar.nevals, res_fromAstar.wall)
@printf("  KNITRO-minimized delta*, start=scaled-search A_od          = %.8f  (status=%d, nevals=%d, wall=%.1fs)\n",
    res_fromConstrained.δmin, res_fromConstrained.nStatus, res_fromConstrained.nevals, res_fromConstrained.wall)
δmin = min(res_fromAstar.δmin, res_fromConstrained.δmin)
gap = 1.0 - δmin
@printf("\n  gap = budget - min(delta*_min over both starts) = 1.000000 - %.6f = %.6f\n", δmin, gap)
if gap > 0.02
    println("  => MEANINGFULLY LOWER than the budget: real headroom left on the table even after KNITRO-scaling.")
elseif gap < -0.02
    println("  => the reformulation's minimum EXCEEDS the budget -- unexpected, needs investigation.")
else
    println("  => approximately EQUAL to the budget: the scaled constrained search already found (close to) the true optimum.")
end

out_path = joinpath(@__DIR__, "profiled_delta_star_min_knitro_D20_W80000_delta1.0_upper_real.jld2")
JLD2.save(out_path, Dict(
    "D" => D, "W" => W, "delta_budget" => 1.0, "bound" => "upper",
    "gammap_star" => gammap_star, "kappa_saved" => κ_saved, "source_jld2" => SAVED_PATH,
    "Acol_star" => Acol_star, "Aod_constrained" => Aod_constrained,
    "delta_at_Astar" => δ_at_Astar, "delta_at_constrained" => δ_at_constrained,
    "res_fromAstar" => res_fromAstar, "res_fromConstrained" => res_fromConstrained,
))
println("\nSaved: $out_path")
println("\nKNITRO PROFILED DELTA* MINIMIZATION TEST (D=20 REAL DATA) DONE")
