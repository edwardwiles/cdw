include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
using Printf, Statistics

ctx = d20_real_setup(W = 80000, δ = 1.0, find_smallest = true)
D = ctx.D
x_free_calib = ctx.θ0_up[ctx.free_idx]
gp_calib = x_free_calib[1]
A_calib = x_free_calib[2:end]

@printf "length(x_free_calib)=%d  D^2=%d\n" length(x_free_calib) D^2
@printf "A_calib: min=%.15f max=%.15f  all==1.0? %s  n!=1.0: %d\n" minimum(A_calib) maximum(A_calib) all(A_calib .== 1.0) count(A_calib .!= 1.0)
if any(A_calib .!= 1.0)
    bad = findall(A_calib .!= 1.0)
    println("  entries != 1.0 (first 10): ", [(i, A_calib[i]) for i in bad[1:min(10,end)]])
end

pe = build_pivot_elimination(ctx)
gravity0 = gravity_offset(ctx)
@printf "gravity_offset (gravity value AT A_od=1 everywhere, z=0): %.6e\n" gravity0

snaps = nested_grid_sequence([10, 20, 50])
pcx = build_cm_production_context(ctx, CS; L = 10, contrasts = :anchored, probs = snaps[10])

println("\n=== eval at x_free_calib (real calibration A_od) ===")
try
    K, base = cm_production_value(x_free_calib, pcx)
    @printf "  SUCCESS  nStatus=%d  Delta=%.6f\n" base.inner_status (-base.ζstar)
catch e
    println("  FAILED: ", sprint(showerror, e)[1:min(150,end)])
end

println("=== eval at [gp_calib; ones(D^2)] ===")
x_free_ones = vcat(gp_calib, ones(D^2))
@printf "  max|x_free_calib - x_free_ones| = %.3e\n" maximum(abs.(x_free_calib .- x_free_ones))
try
    K, base = cm_production_value(x_free_ones, pcx)
    @printf "  SUCCESS  nStatus=%d  Delta=%.6f\n" base.inner_status (-base.ζstar)
catch e
    println("  FAILED: ", sprint(showerror, e)[1:min(150,end)])
end
println("DONE")
