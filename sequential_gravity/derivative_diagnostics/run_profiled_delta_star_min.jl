# ============================================================================
# Test of a "profiled" reformulation of the CC outer loop (see
# full_d2_correction_report.md section 7/7.1/7.2 for the motivating problem).
#
# EXISTING formulation: extremize gamma'_focal over (gamma'_focal, A_od[1:D])
# subject to delta*(gamma'_focal, A_od) <= delta (a divergence-budget
# constraint), via KNITRO SQP/interior-point, initialized at A_od=A*. That
# search fights a severe gamma'-vs-A_od gradient-scale mismatch (report
# section 7: ~1e4-1e5x at D=20 real data, ~160x even at D=4).
#
# REFORMULATION tested here: FIX gamma'_focal at the value the existing
# constrained search already reached, and MINIMIZE delta*(A_od) over A_od
# alone -- a genuinely unconstrained (box-bounded) scalar minimization with
# NO gamma'-vs-A_od scale mismatch, since gamma'_focal is no longer a free
# variable in this formulation at all.
#
# Logic: if the original constrained search found the TRUE constrained
# optimum, minimizing delta* at the resulting gamma'* should recover
# (approximately) the SAME delta used as the original budget. If the
# reformulation finds a LOWER delta*, that is direct evidence the original
# search left real headroom on the table, independent of any KNITRO-scaling
# issue.
#
# Uses the EXISTING, already-validated `exact_inner_divergence_at` (Part 10's
# genuine profiled delta* evaluator -- fixed_A_incumbent.jl, included by
# run_profiled_production.jl) as the objective function -- no new inner-dual
# machinery. Optimizer: BlackBoxOptim.jl (already a project dependency,
# derivative-free -- explicitly permitted by the task for this first
# validation pass), method=:generating_set_search (a local direct-search
# method well suited to a smooth-ish low-dimensional problem with a good
# starting point, unlike population-based DE methods which need a much
# tighter box or many more evals to be efficient here).
#
# Two starting points are tried per the task's optional step 4:
#   (a) A_od = A* (the calibrated Frechet baseline)
#   (b) A_od = the constrained search's OWN A_od endpoint
# If the constrained search really found a local optimum in A_od, both should
# converge to the same delta*.
#
#   DVAL=4 WVAL=8000 DELTA=1.0 MAXEVALS=300 \
#     julia --project=. sequential_gravity/derivative_diagnostics/run_profiled_delta_star_min.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))

using BlackBoxOptim, Printf, JLD2, LinearAlgebra, Statistics

const DELTA = parse(Float64, get(ENV, "DELTA", "1.0"))
const MAXEVALS = parse(Int, get(ENV, "MAXEVALS", "300"))
const GRAD_METHOD = Symbol(get(ENV, "GRADIENT_METHOD", "fixed_dual_fd_full"))
const BOUND = get(ENV, "BOUND", "lower")   # "lower" (find_smallest=false) or "upper" (find_smallest=true)
const FIND_SMALLEST = BOUND == "upper"

println("="^78)
println(">>> Profiled delta*(A_od) minimization test -- D=$D, W=$W, delta budget=$DELTA, bound=$BOUND")
println("="^78)

# ---- Step 1: reproduce the EXISTING constrained full-A search (report's most-validated
# D=4 setting) to get gamma'_focal* AND the constrained search's own A_od endpoint
# (needed for starting point (b) below). ----
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

# ---- Step 2: the profiled objective, gamma'_focal held FIXED at gammap_star ----
const mu0 = θr0[1]
const sigma0 = θr0[2]
const Acol_star = θr0[4:3+D]

neval = Ref(0)
function delta_star_of_Aod(Aod::AbstractVector{Float64})
    neval[] += 1
    θ = vcat(mu0, sigma0, gammap_star, Aod)
    exact_inner_divergence_at(θ).δ_star
end

search_range = [(θ_lo[3+i], θ_hi[3+i]) for i in 1:D]

# ---- Baseline checks: objective value AT each starting point, before optimizing ----
t0 = time()
δ_at_Astar = delta_star_of_Aod(Acol_star)
@printf("  delta*(A_od = A*)                = %.8f  (%.1fs)\n", δ_at_Astar, time() - t0)
t0 = time()
δ_at_constrained = delta_star_of_Aod(Aod_constrained)
@printf("  delta*(A_od = constrained-search) = %.8f  (%.1fs)\n", δ_at_constrained, time() - t0)

# ---- Step 3/4: minimize delta*(A_od) from both starting points ----
function run_minimization(label::String, x0::Vector{Float64})
    println("\n  --- minimizing delta*(A_od) from $label ---")
    t0 = time()
    neval[] = 0
    res = bboptimize(delta_star_of_Aod, x0; SearchRange = search_range,
        Method = :generating_set_search, MaxFuncEvals = MAXEVALS, TraceMode = :silent)
    wall = time() - t0
    xbest = best_candidate(res); δbest = best_fitness(res)
    @printf("  [%s] delta*_min = %.8f  (nevals=%d, wall=%.1fs)\n", label, δbest, neval[], wall)
    relΔA = norm(xbest .- Acol_star) / norm(Acol_star)
    @printf("  [%s] rel‖A_od_min - A*‖ = %.4f\n", label, relΔA)
    (label = label, x0 = x0, xbest = xbest, δbest = δbest, nevals = neval[], wall = wall, relΔA = relΔA)
end

res_fromAstar = run_minimization("A_od = A*", copy(Acol_star))
res_fromConstrained = run_minimization("A_od = constrained-search endpoint", copy(Aod_constrained))

# ---- Summary ----
println("\n" * "="^78); println(">>> SUMMARY"); println("="^78)
@printf("  original delta BUDGET used by constrained search        = %.6f\n", DELTA)
@printf("  gamma'_focal* reached by constrained search              = %.9f (kappa=%.6f)\n", gammap_star, κ_full)
@printf("  exact delta*(A*, same gamma')                            = %.8f\n", δ_at_Astar)
@printf("  exact delta*(constrained-search A_od, same gamma')       = %.8f\n", δ_at_constrained)
@printf("  minimized delta*, start=A*                               = %.8f  (nevals=%d, wall=%.1fs)\n",
    res_fromAstar.δbest, res_fromAstar.nevals, res_fromAstar.wall)
@printf("  minimized delta*, start=constrained-search A_od          = %.8f  (nevals=%d, wall=%.1fs)\n",
    res_fromConstrained.δbest, res_fromConstrained.nevals, res_fromConstrained.wall)
δmin = min(res_fromAstar.δbest, res_fromConstrained.δbest)
gap = DELTA - δmin
@printf("\n  gap = budget - min(delta*_min over both starts) = %.6f - %.6f = %.6f\n", DELTA, δmin, gap)
if gap > 0.02 * DELTA
    println("  => MEANINGFULLY LOWER than the budget: the constrained search left real headroom on the table.")
elseif gap < -0.02 * DELTA
    println("  => the reformulation's minimum EXCEEDS the original budget -- unexpected, needs investigation")
    println("     (the constrained-search point is itself delta*<=DELTA-feasible by construction, so the")
    println("     minimizer starting AT that point should never do worse -- check optimizer convergence).")
else
    println("  => approximately EQUAL to the budget: the constrained search already found (close to) the true optimum.")
end

out_path = joinpath(@__DIR__, "profiled_delta_star_min_D$(D)_W$(W)_delta$(DELTA)_$(BOUND).jld2")
JLD2.save(out_path, Dict(
    "D" => D, "W" => W, "delta_budget" => DELTA, "bound" => BOUND, "gradient_method" => String(GRAD_METHOD),
    "gammap_star" => gammap_star, "kappa_full" => κ_full,
    "Acol_star" => Acol_star, "Aod_constrained" => Aod_constrained,
    "delta_at_Astar" => δ_at_Astar, "delta_at_constrained" => δ_at_constrained,
    "res_fromAstar" => res_fromAstar, "res_fromConstrained" => res_fromConstrained,
))
println("\nSaved: $out_path")
println("\nPROFILED DELTA* MINIMIZATION TEST DONE")
