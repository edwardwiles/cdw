# ============================================================================
# KNITRO-based (not derivative-free) version of the profiled delta*(A_od)
# minimization test. Per explicit user request: use LITERALLY the same
# optimizer as the existing constrained outer loop (KNITRO local SQP/interior-
# point), not a global/derivative-free method -- the BlackBoxOptim version
# (run_profiled_delta_star_min.jl) is a separate track another Claude session
# is running; this script is the strict apples-to-apples comparison the user
# asked for: same solver, same style of gradient, only the PROBLEM SHAPE
# changes (unconstrained box-bounded minimization of delta*(A_od), gamma'_focal
# fixed, vs the original budget-constrained joint search over (gamma',A_od)).
#
# Objective VALUE: exact_inner_divergence_at(theta).delta_star -- the same
# already-validated genuine profiled delta* evaluator used everywhere else in
# this diagnostics suite (fixed_A_incumbent.jl), a real fresh gravity-freeze +
# real KNITRO inner-dual solve at every genuinely NEW A_od.
#
# Objective GRADIENT: NOT a fresh finite-difference of the expensive true
# objective (which would need D or 2D extra real inner solves per outer
# iteration -- prohibitively expensive at D=20 real data). Instead, reuses the
# EXACT SAME cheap trick production's own gradient_method=:fixed_dual_fd_full
# already uses (full_gradient_method_wiring.jl / fixed_dual_fd.jl): freeze the
# dual solution x* and the gravity linearization AT the current A_od (one real
# solve, already paid for by the value computation), then take CENTRAL finite
# differences of the cheap closed-form fixed-dual criterion Q(theta,x*) --
# exact at theta=theta_k (Part 7 identity in full_d2_correction_report.md) and
# validated to track the true re-solved profile derivative to 0.05%-2% nearby.
# `fixed_dual_fd_gradient(...; Acol_offset=3)` already differentiates ONLY the
# Acol block (indices 4:3+D of a length-(3+D) theta) -- exactly the free-set
# this problem needs, since gamma'_focal (index 3) is genuinely pinned here,
# not merely held out of one particular gradient block as in the original
# wiring.
#
#   DVAL=4 WVAL=8000 DELTA=1.0 BOUND=lower \
#     julia --project=. sequential_gravity/derivative_diagnostics/run_profiled_delta_star_min_knitro.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))

using KNITRO, Printf, JLD2, LinearAlgebra

const DELTA = parse(Float64, get(ENV, "DELTA", "1.0"))
const BOUND = get(ENV, "BOUND", "lower")
const FIND_SMALLEST = BOUND == "upper"
const GRAD_METHOD = Symbol(get(ENV, "GRADIENT_METHOD", "fixed_dual_fd_full"))
const H_FD = parse(Float64, get(ENV, "H_FD", "0.1"))
const KNITRO_OPT_FILE = get(ENV, "KNITRO_OPT_FILE", OUTER_OPT_FILE)

println("="^78)
println(">>> KNITRO-based profiled delta*(A_od) minimization -- D=$D, W=$W, delta budget=$DELTA, bound=$BOUND")
println("="^78)

# ---- Step 1: reproduce the EXISTING constrained full-A search to get gamma'_focal* AND
# the constrained search's own A_od endpoint (second starting guess). ----
t0 = time()
fA = outer_solve_fixedA(FIND_SMALLEST, copy(θr0); δ = DELTA)
fA.best_θ === nothing && error("fixed-A* search found no gravity-feasible point -- cannot proceed")
θinit_full = copy(θr0); θinit_full[3] = fA.best_θ[3]
gp_full_raw, θfull_raw, stfull, bθfull, bκfull, bwarmfull, cachefull = outer_solve_nested_cached(
    FIND_SMALLEST, θinit_full; use_exact_grad = true, δ = DELTA, gradient_method = GRAD_METHOD)
bθfull === nothing && error("constrained full-A search found no gravity-feasible point -- cannot proceed")
gammap_star = bθfull[3]
Aod_constrained = bθfull[4:3+D]
κ_full = gp2kappa(gammap_star)
@printf("  [constrained search] gamma'_focal* = %.9f -> kappa = %.6f  (wall=%.1fs)\n", gammap_star, κ_full, time() - t0)

const mu0 = θr0[1]
const sigma0 = θr0[2]
const Acol_star = θr0[4:3+D]
const lo = θ_lo[4:3+D]
const hi = θ_hi[4:3+D]

"""
    minimize_delta_star_knitro(x0) -> (nStatus, δmin, x_min, nevals)

Unconstrained (box-bounded only) KNITRO minimization of delta*(A_od) at fixed
gamma'_focal=gammap_star, starting from x0. Value = exact_inner_divergence_at
(real, fresh inner solve each genuinely-new point). Gradient = the cheap
fixed-dual-criterion FD trick (no extra real inner solves).
"""
function minimize_delta_star_knitro(x0::Vector{Float64}; label::String)
    println("\n  --- KNITRO minimizing delta*(A_od) from $label ---")
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
        # exact_inner_divergence_at returns a SHORTER NamedTuple (no x_star/frozen fields) when
        # gravity-infeasible (frozen.ok==false) -- guard against that shape before touching them,
        # rather than letting KNITRO's C callback boundary swallow the FieldError (it did, silently,
        # as a "puts callback exception" warning, corrupting the search with stale/zero gradients).
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
        g .= grad_log ./ x_free   # log-space -> level-space (d/dlevel = d/dlog / level)
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

    function cb_FG!(kc2, cb, evalRequest, evalResult, userParams)
        x_free = evalRequest.x
        ensure!(x_free)
        evalResult.obj[1] = isfinite(last_delta[]) ? last_delta[] : 1e10
        fill_grad!(evalResult.objGrad, x_free)
        return 0
    end

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, KNITRO_OPT_FILE)
    KNITRO.KN_add_vars(kc, D)
    KNITRO.KN_set_var_lobnds_all(kc, lo)
    KNITRO.KN_set_var_upbnds_all(kc, hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, x0)
    # `eval_fcga` (set in this project's own .opt files, e.g. csw_outer_25.opt/csw_outer_1000.opt):
    # when enabled, KNITRO expects ONE combined callback to fill both value and gradient per call --
    # registering separate cb_F!/cb_G! (the naive pattern) leaves KNITRO calling ONLY cb_F!, so
    # objGrad is never populated (silently stays at its default, ~0), producing instant false
    # "already optimal" convergence at whatever the starting point happens to be. Exactly the same
    # branch outer_loop_cached.jl already has to handle this -- caught here by a toy 2-variable
    # KNITRO sanity check (cb_G! was registered correctly but never actually invoked; matched
    # eval_fcga=yes in both opt files this script uses).
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
    (label = label, x0 = x0, nStatus = nStatus, δmin = objv, xmin = collect(x_min), nevals = neval[], wall = wall, relΔA = relΔA)
end

t0 = time()
δ_at_Astar = exact_inner_divergence_at(vcat(mu0, sigma0, gammap_star, Acol_star)).δ_star
@printf("  delta*(A_od = A*)                    = %.8f  (%.1fs)\n", δ_at_Astar, time() - t0)
t0 = time()
δ_at_constrained = exact_inner_divergence_at(vcat(mu0, sigma0, gammap_star, Aod_constrained)).δ_star
@printf("  delta*(A_od = constrained-search)    = %.8f  (%.1fs)\n", δ_at_constrained, time() - t0)

res_fromAstar = minimize_delta_star_knitro(copy(Acol_star); label = "A_od = A*")
res_fromConstrained = minimize_delta_star_knitro(copy(Aod_constrained); label = "A_od = constrained-search endpoint")

println("\n" * "="^78); println(">>> SUMMARY (KNITRO local, unconstrained reformulation)"); println("="^78)
@printf("  original delta BUDGET used by constrained search        = %.6f\n", DELTA)
@printf("  gamma'_focal* reached by constrained search              = %.9f (kappa=%.6f)\n", gammap_star, κ_full)
@printf("  exact delta*(A*, same gamma')                            = %.8f\n", δ_at_Astar)
@printf("  exact delta*(constrained-search A_od, same gamma')       = %.8f\n", δ_at_constrained)
@printf("  KNITRO-minimized delta*, start=A*                        = %.8f  (status=%d, nevals=%d, wall=%.1fs)\n",
    res_fromAstar.δmin, res_fromAstar.nStatus, res_fromAstar.nevals, res_fromAstar.wall)
@printf("  KNITRO-minimized delta*, start=constrained-search A_od   = %.8f  (status=%d, nevals=%d, wall=%.1fs)\n",
    res_fromConstrained.δmin, res_fromConstrained.nStatus, res_fromConstrained.nevals, res_fromConstrained.wall)
δmin = min(res_fromAstar.δmin, res_fromConstrained.δmin)
gap = DELTA - δmin
@printf("\n  gap = budget - min(delta*_min over both starts) = %.6f - %.6f = %.6f\n", DELTA, δmin, gap)
if gap > 0.02 * DELTA
    println("  => MEANINGFULLY LOWER than the budget: the constrained search left real headroom on the table.")
elseif gap < -0.02 * DELTA
    println("  => the reformulation's minimum EXCEEDS the budget -- unexpected, needs investigation.")
else
    println("  => approximately EQUAL to the budget: the constrained search already found (close to) the true optimum.")
end

out_path = joinpath(@__DIR__, "profiled_delta_star_min_knitro_D$(D)_W$(W)_delta$(DELTA)_$(BOUND).jld2")
JLD2.save(out_path, Dict(
    "D" => D, "W" => W, "delta_budget" => DELTA, "bound" => BOUND, "gradient_method" => String(GRAD_METHOD),
    "gammap_star" => gammap_star, "kappa_full" => κ_full,
    "Acol_star" => Acol_star, "Aod_constrained" => Aod_constrained,
    "delta_at_Astar" => δ_at_Astar, "delta_at_constrained" => δ_at_constrained,
    "res_fromAstar" => res_fromAstar, "res_fromConstrained" => res_fromConstrained,
))
println("\nSaved: $out_path")
println("\nKNITRO PROFILED DELTA* MINIMIZATION TEST DONE")
