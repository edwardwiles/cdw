# Empirical check (not derived from reading code alone -- too easy to get this backwards
# silently): does CS.outer_loop(obj, ...; find_smallest=true) with the plain generic
# calculate_grad_k_autodiff! path move kappa toward the known UPPER (~0.1725) or LOWER (~0.0044)
# anchor from candidate_registry.jl? Baseline (no CM) context, short budget, both directions.
include(joinpath(@__DIR__, "context.jl"))
using Printf

for fs in (true, false)
    ctx = d4_exact_setup(δ = 1.0, find_smallest = fs, needs_outer_moment_jacobian = true)
    println(">>> find_smallest=$fs  starting kappa (calibration) = ", 1 - ctx.θ0_up[3+ctx.D]^(ctx.σ/(ctx.σ-1)))
    t0 = time()
    γp_min, θ_min, nStatus, lambda_ = CS.outer_loop(ctx.obj, ctx.θ_lo, ctx.θ_hi, ctx.θ0_up)
    κ = 1 - γp_min^(ctx.σ / (ctx.σ - 1))
    @printf("find_smallest=%s -> status=%d  gamma'=%.6f  kappa=%.6f  wall=%.1fs\n", fs, nStatus, γp_min, κ, time()-t0)
end
