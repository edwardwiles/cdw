# ============================================================================
# Part 8/9/10 driver (Stage A, D=4): the REAL outer-loop test.
#   1. Solve fixed-A* first (Part 9) -> gamma'_fixed, feasible incumbent.
#   2. Initialize the full-A search at (gamma'_fixed, A*) with
#      gradient_method=:fixed_dual_fd_full (the corrected method).
#   3. Sanity check: kappa_full weakly dominates kappa_fixed in the direction
#      implied by the bound (Part 9's required condition).
#   4. Exact post-solve audit (Part 10): at the moved-A endpoint's gamma',
#      compare the exact re-solved moved-A divergence against the exact
#      fixed-A* divergence AT THE SAME gamma' -- Delta_delta(gamma').
#
# IMPORTANT (caught while building this driver): the raw KNITRO outer
# endpoint (theta_min_full/gp) is NOT always gravity-feasible -- KNITRO's own
# divergence-budget constraint is evaluated using make_stateful_moments's
# placeholder INFCOL gravity column whenever seq_gravcol fails to converge at
# a trial theta, so the outer search can wander into (and even terminate at)
# a gravity-infeasible point while appearing "converged" to KNITRO. Both
# `outer_solve_fixedA` and `outer_solve_nested_cached` already track
# best_theta/best_kappa (the best GRAVITY-FEASIBLE point seen during the
# search, verified via a real seq_gravcol call at the time -- see
# run_profiled_production.jl::make_stateful_moments). This driver uses THAT,
# not the raw KNITRO endpoint, for every economic comparison -- exactly what
# production's own run_one_bound already does, reporting "best-feasible"
# alongside (and preferred over) the raw KNITRO result.
#
#   DVAL=4 WVAL=8000 DELTA=1 BOUND=both julia --project=. sequential_gravity/derivative_diagnostics/run_full_d2_outer_loop_test.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))

using Printf, JLD2, LinearAlgebra

const DELTA = parse(Float64, get(ENV, "DELTA", "1.0"))
const BOUND = get(ENV, "BOUND", "both")
const METHOD = Symbol(get(ENV, "METHOD", "fixed_dual_fd_full"))

function run_bound_test(name::Symbol, find_smallest::Bool)
    println("\n" * "="^78); println(">>> $name bound, delta=$DELTA, method=$METHOD"); println("="^78)

    # ---- Step 1: fixed-A* incumbent (Part 9) -- use the GRAVITY-FEASIBLE best point, not the raw KNITRO endpoint ----
    t0 = time()
    fA = outer_solve_fixedA(find_smallest, copy(θr0); δ=DELTA)
    @printf("  [fixed-A*]  raw KNITRO endpoint: gamma'=%.6f status=%d\n", fA.gp, fA.nStatus)
    fA.best_θ === nothing && error("fixed-A* search found NO gravity-feasible point at all -- cannot proceed")
    # NOTE: make_stateful_moments's `best_κ` is a MISNOMER inherited from production -- it actually
    # stores the raw K[1]=gamma'_focal target (`κθ = Ktmp[1]` in run_profiled_production.jl), not the
    # real kappa=1-gamma'^(sigma/(sigma-1)). Since kappa is a strictly DECREASING function of gamma',
    # using the raw gamma' value directly as if it were kappa silently flips every comparison's sense
    # -- caught here because it produced a spurious Part-9 "FAIL" that vanished once corrected via
    # gp2kappa. Always convert explicitly.
    θ_fixed_best = fA.best_θ; gp_fixed_best = θ_fixed_best[3]; κ_fixed = gp2kappa(gp_fixed_best)
    @printf("  [fixed-A*]  BEST FEASIBLE: gamma'=%.6f -> kappa=%.6f  wall=%.1fs\n", gp_fixed_best, κ_fixed, time()-t0)

    # ---- Step 2: full-A search, initialized at (gamma'_fixed, A*), corrected gradient ----
    θinit_full = copy(θr0); θinit_full[3] = gp_fixed_best
    t0 = time()
    gp_full_raw, θfull_raw, stfull, bθfull, bκfull, bwarmfull, cachefull = outer_solve_nested_cached(
        find_smallest, θinit_full; use_exact_grad=true, δ=DELTA, gradient_method=METHOD)
    @printf("  [full-A, %s]  raw KNITRO endpoint: gamma'=%.6f status=%d  wall=%.1fs\n", METHOD, gp_full_raw, stfull, time()-t0)
    bθfull === nothing && error("full-A search found NO gravity-feasible point at all -- cannot proceed")
    θfull = bθfull; gp_full = θfull[3]; κ_full = gp2kappa(gp_full)   # see note above -- bκfull is raw gamma', not kappa
    @printf("  [full-A, %s]  BEST FEASIBLE: gamma'=%.6f -> kappa=%.6f\n", METHOD, gp_full, κ_full)
    relΔA = norm(θfull[4:3+D] .- θr0[4:3+D]) / norm(θr0[4:3+D])
    @printf("  rel‖A_full - A*‖ (best-feasible point) = %.4f\n", relΔA)

    # ---- Step 3: Part 9 sanity check (on best-feasible points, the only economically valid comparison) ----
    # lower bound (find_smallest=false) maximizes gamma' => minimizes kappa: full-A must reach kappa_full <= kappa_fixed.
    # upper bound (find_smallest=true) minimizes gamma' => maximizes kappa: full-A must reach kappa_full >= kappa_fixed.
    sanity_ok = find_smallest ? (κ_full >= κ_fixed - 1e-6) : (κ_full <= κ_fixed + 1e-6)
    @printf("  Part-9 sanity: kappa_full=%.6f  kappa_fixed=%.6f  (full-A must weakly beat fixed-A) -> %s\n",
        κ_full, κ_fixed, sanity_ok ? "PASS" : "FAIL -- flag as non-economic result")

    # ---- Step 4: Part 10 exact post-solve audit ----
    println("  --- Part 10 exact post-solve audit ---")
    audit_moved = exact_inner_divergence_at(θfull)
    audit_fixed_same_gp = exact_fixedA_divergence_at(gp_full, θr0)
    Δδ = audit_fixed_same_gp.δ_star - audit_moved.δ_star
    @printf("  exact delta*_movedA(gamma'=%.6f)  = %.8f  (gravity_ok=%s, nStatus=%d)\n", gp_full, audit_moved.δ_star, audit_moved.gravity_ok, audit_moved.nStatus)
    @printf("  exact delta*_fixedA(gamma'=%.6f)  = %.8f  (gravity_ok=%s, nStatus=%d)\n", gp_full, audit_fixed_same_gp.δ_star, audit_fixed_same_gp.gravity_ok, audit_fixed_same_gp.nStatus)
    @printf("  Delta_delta(gamma') = fixedA - movedA = %.8f  (positive => moving A genuinely helped reach this gamma' more cheaply)\n", Δδ)
    weakly_improved = Δδ >= -1e-6
    @printf("  fixed-A incumbent weakly improved by moved-A search: %s\n", weakly_improved)

    return (name=name, find_smallest=find_smallest, gp_fixed=gp_fixed_best, κ_fixed=κ_fixed,
            gp_full=gp_full, κ_full=κ_full, status_full=stfull, relΔA=relΔA,
            sanity_ok=sanity_ok, δ_movedA=audit_moved.δ_star, δ_fixedA_samegp=audit_fixed_same_gp.δ_star,
            Δδ=Δδ, weakly_improved=weakly_improved)
end

results = NamedTuple[]
BOUND in ("both", "lower") && push!(results, run_bound_test(:lower, false))
BOUND in ("both", "upper") && push!(results, run_bound_test(:upper, true))

println("\n" * "="^78); println(">>> SUMMARY"); println("="^78)
@printf("%-6s %10s %10s %8s %10s %10s %10s %8s %10s\n", "bound", "kappa_fix", "kappa_full", "sanity", "relΔA", "d_movedA", "d_fixedA", "Δδ", "improved")
for r in results
    @printf("%-6s %10.6f %10.6f %8s %10.4f %10.6f %10.6f %8.4f %10s\n",
        r.name, r.κ_fixed, r.κ_full, r.sanity_ok ? "PASS" : "FAIL", r.relΔA, r.δ_movedA, r.δ_fixedA_samegp, r.Δδ, r.weakly_improved ? "yes" : "no")
end

JLD2.save(joinpath(@__DIR__, "part8_9_10_outer_loop_test_D$(D)_W$(W)_delta$(DELTA)_$(METHOD).jld2"),
    Dict("results" => results, "D" => D, "W" => W, "delta" => DELTA, "method" => String(METHOD)))
println("\nPART 8/9/10 outer-loop test DONE")
