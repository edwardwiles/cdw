ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf, LinearAlgebra

find_smallest = false   # lower bound
δ = 1.0
t0 = time()
gp, θfull, st, bθ, bκ, bwarm, cache = outer_solve_nested_cached(find_smallest, copy(θr0); use_exact_grad=true, δ=δ, gradient_method=:fixed_dual_fd_full)
@printf("gamma'=%.6f kappa=%.6f status=%d wall=%.1fs\n", gp, gp2kappa(gp), st, time()-t0)
@printf("rel‖ΔA‖ = %.4f\n", norm(θfull[4:3+D] .- θr0[4:3+D]) / norm(θr0[4:3+D]))
@printf("n_inner_solve=%d n_grad_compute=%d trace_len=%d\n", cache.n_inner_solve, cache.n_grad_compute, length(cache.trace))
_, R, _, _, _, ok = seq_gravcol(θfull; δ=δ)
@printf("gravity-feasible at endpoint: %s R=%.3e\n", ok, R)
