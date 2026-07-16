ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf, LinearAlgebra

find_smallest = false   # lower bound
δ = 1.0
t0 = time()
scaling_power = parse(Float64, get(ENV, "SCALING_POWER", "1.0"))
gp, θfull, st, bθ, bκ, bwarm, cache = outer_solve_nested_cached(find_smallest, copy(θr0);
    use_exact_grad=true, δ=δ, gradient_method=:fixed_dual_fd_full, use_var_scaling=true, scaling_power=scaling_power)
@printf("SCALED: gamma'=%.6f kappa=%.6f status=%d wall=%.1fs\n", gp, gp2kappa(gp), st, time()-t0)
bθ === nothing || @printf("best-feasible: gamma'=%.6f kappa=%.6f  rel‖ΔA‖=%.4f\n",
    bθ[3], gp2kappa(bθ[3]), norm(bθ[4:end] .- θr0[4:end])/norm(θr0[4:end]))
