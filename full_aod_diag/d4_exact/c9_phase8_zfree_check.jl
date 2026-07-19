# Definitive check: what does zfree0_natural (the pivot-reduced z-vector at "A = natural theta")
# actually look like? norm, extrema, first few values.
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
using Statistics, LinearAlgebra, Printf

ctx = d20_real_setup(W = 80000, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
@printf("Aod_theta_natural: length=%d min=%.6g max=%.6g mean=%.6g\n", length(Aod_theta_natural),
        minimum(Aod_theta_natural), maximum(Aod_theta_natural), mean(Aod_theta_natural))
println("Aod_theta_natural[1:10] = ", Aod_theta_natural[1:10])
z0 = log.(Aod_theta_natural)
@printf("z0 = log(Aod_theta_natural): length=%d min=%.6g max=%.6g norm=%.6g\n",
        length(z0), minimum(z0), maximum(z0), norm(z0))
zfree0 = pivot_reduce(reshape(z0, D, D), pe)
@printf("zfree0 (pivot-reduced): length=%d min=%.6g max=%.6g norm=%.6g\n",
        length(zfree0), minimum(zfree0), maximum(zfree0), norm(zfree0))
println("zfree0[1:10] = ", zfree0[1:10])
