# Diagnostic: is the nStatus=-300 (unbounded) at real D=20/exclude_row/small-W the calibration
# point ALSO hit by flexible-CM (no fixed-Frechet restriction), or is it specific to the new
# Frechet code? Cheap, fast (W=4000, small L).
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "draw_design.jl"))
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
include(joinpath(@__DIR__, "cm_screen_bridge.jl"))
include(joinpath(@__DIR__, "knitro_status.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
using Printf, LinearAlgebra

ctx = d20_real_setup_design(W = 4000, δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false,
    destination_sample = :exclude_row)
println("ctx.D=$(ctx.D) ctx.D_dest=$(ctx.D_dest)")
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full0 = CS.reconstruct_full(x_free_calib, ctx.m)

println("=== Bare core (no CM) inner solve at calibration point ===")
K0, x0, ns0, nfg0, nh0 = inner_loop_internal_archgeneric(ctx.obj, θ_full0; hess_cb_builder = archA_hess_cb_builder)
println("bare core: nStatus=$ns0  category=$(decode_knitro_status(ns0).category)")

println("=== Flexible-CM (L=5, orthonormal) inner solve at calibration point ===")
pcx = build_cm_production_context(ctx, CS; L = 5, contrasts = :orthonormal)
base_cm = archC_base_state(x_free_calib, pcx.ctx_cm, pcx.cctx)
println("flexible-CM: nStatus=$(base_cm.nStatus)  category=$(decode_knitro_status(base_cm.nStatus).category)")
