# Diagnostic: isolate why the production bundle's cold inner solve (use_archB_moments=true) is
# much slower than c15_d20_cm_smoke_test.jl's raw Architecture-C timing (which uses Architecture
# A's dense-appended moments, not Architecture B).
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
using Printf

W = 80000; DELTA = 1.0
ctx = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
x_free_calib = ctx.θ0_up[ctx.free_idx]
L = 50

println("=== use_archB_moments=false (matches c15's own ArchC timing exactly) ===")
t1 = @elapsed pcxA = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, use_archB_moments = false)
t2 = @elapsed (K1, base1) = cm_production_value(x_free_calib, pcxA)
t3 = @elapsed (K1b, base1b) = cm_production_value(x_free_calib, pcxA)
@printf "  setup=%.2fs  1st solve=%.3fs  2nd solve=%.3fs  nStatus=%d\n" t1 t2 t3 base1.inner_status

println("=== use_archB_moments=true (production default) ===")
t4 = @elapsed pcxB = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, use_archB_moments = true)
t5 = @elapsed (K2, base2) = cm_production_value(x_free_calib, pcxB)
t6 = @elapsed (K2b, base2b) = cm_production_value(x_free_calib, pcxB)
@printf "  setup=%.2fs  1st solve=%.3fs  2nd solve=%.3fs  nStatus=%d\n" t4 t5 t6 base2.inner_status

println()
println("=== isolate: time JUST the moments! call for each variant ===")
θ_full0 = CS.reconstruct_full(x_free_calib, ctx.m)
K = zeros(W)
GA = Matrix{Float64}(undef, W, pcxA.ctx_cm.obj.d)
t_momA1 = @elapsed pcxA.ctx_cm.obj.moments!(K, GA, θ_full0, pcxA.ctx_cm.obj.U, pcxA.ctx_cm.obj)
t_momA2 = @elapsed pcxA.ctx_cm.obj.moments!(K, GA, θ_full0, pcxA.ctx_cm.obj.U, pcxA.ctx_cm.obj)
@printf "  ArchA-moments (dense-appended precomputed CM) call: 1st=%.3fs  2nd=%.3fs\n" t_momA1 t_momA2

GB = Matrix{Float64}(undef, W, pcxB.ctx_cm.obj.d)
t_momB1 = @elapsed pcxB.ctx_cm.obj.moments!(K, GB, θ_full0, pcxB.ctx_cm.obj.U, pcxB.ctx_cm.obj)
t_momB2 = @elapsed pcxB.ctx_cm.obj.moments!(K, GB, θ_full0, pcxB.ctx_cm.obj.U, pcxB.ctx_cm.obj)
@printf "  ArchB-moments (bin-based fill_cm_columns_from_bins!) call: 1st=%.3fs  2nd=%.3fs\n" t_momB1 t_momB2
@printf "  max|GA-GB| (core+grav cols only, cols 1:%d and end) = %.3e\n" (pcxA.aug.ncore-1) max(maximum(abs.(GA[:,1:pcxA.aug.ncore-1] .- GB[:,1:pcxA.aug.ncore-1])), abs(maximum(GA[:,end].-GB[:,end])))
println("DONE")
