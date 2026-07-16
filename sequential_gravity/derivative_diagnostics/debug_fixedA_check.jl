ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf, LinearAlgebra

find_smallest = false   # lower bound
δ = 1.0
fA = outer_solve_fixedA(find_smallest, copy(θr0); δ=δ)
@printf("fixed-A KNITRO result: gamma'=%.6f kappa=%.6f status=%d\n", fA.gp, gp2kappa(fA.gp), fA.nStatus)
@printf("theta_min_full == theta_r0 except index 3? %s\n", fA.θ_min_full[[1,2,4,5,6,7]] == θr0[[1,2,4,5,6,7]])

# Exact re-solve check: is the EXACT full-(D+2) divergence at this (gamma', A*) point <= delta budget?
audit = exact_inner_divergence_at(fA.θ_min_full)
@printf("exact delta*(gamma'=%.6f, A=A*) = %.6f  (budget=%.2f)  gravity_ok=%s  nStatus=%d\n",
    fA.θ_min_full[3], audit.δ_star, δ, audit.gravity_ok, audit.nStatus)
@printf("WITHIN BUDGET? %s\n", audit.δ_star <= δ*(1+1e-4))
