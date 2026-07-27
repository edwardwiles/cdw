# ============================================================================
# Shared outer-A-gradient task (2026-07-27), task §8/§9: correctness gate for
# the ZC-only (origin-ZC) family's shared-backend wiring
# (cm_originzc_production_gradient's gradient_backend=:shared_inplace_pooled
# default vs :legacy_unbuffered reference), at D=4. Structure mirrors
# test_cm_originzc_cplus_equivalence.jl's own established gate pattern.
# ============================================================================
using Test, Printf, LinearAlgebra

include(joinpath(@__DIR__, "context.jl"))
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
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "cm_originzc_target_layout.jl"))
include(joinpath(@__DIR__, "cm_originzc_moments.jl"))
include(joinpath(@__DIR__, "cm_originzc_production.jl"))

println("=== D=4 origin-ZC shared-backend wiring gate ===")
ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)
x_free_calib = ctx.θ0_up[ctx.free_idx]
D = ctx.D; D2 = D^2

nu0_origin(K::Int, D::Int) = vcat([fill(Float64(factorial(k)), D) for k in 1:K]...)
const CONFIGS = [(1, 0, "K1_mean_only"), (1, 1, "K1_mean_zc"), (2, 0, "K2_mean_only")]

all_pass = true
@testset "origin-ZC: :shared_inplace_pooled (default) vs :legacy_unbuffered full-gradient equivalence" begin
    for (K_mean, K_pair, label) in CONFIGS
        layout = OriginByPowerLayout(D, K_mean, K_pair)
        pcx = build_originzc_production_context(ctx, CS, layout)

        ν0 = nu0_origin(K_mean, D)
        base0, verify0 = archOZ_verified_state(x_free_calib, ν0, pcx.ctx_cm)
        @test is_verified_success(verify0)

        g_shared, meta_shared = cm_originzc_production_gradient(x_free_calib, ν0, pcx, ctx, pe; base = base0, verify = verify0,
            threaded = false, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())  # default backend
        g_legacy, meta_legacy = cm_originzc_production_gradient(x_free_calib, ν0, pcx, ctx, pe; base = base0, verify = verify0,
            gradient_backend = :legacy_unbuffered, threaded = false, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())

        n_eta_here = n_eta(layout)
        @test length(g_shared) == D2 + n_eta_here
        @test length(g_legacy) == D2 + n_eta_here
        ok = g_shared == g_legacy
        global all_pass &= ok
        maxdiff = maximum(abs.(g_shared .- g_legacy))
        @printf("  %-14s n_eta=%d  max|Δg|=%.3e  bit-identical=%s\n", label, n_eta_here, maxdiff, ok)
        @test ok
    end
end

println("\nALL D=4 origin-ZC shared-backend gates: ", all_pass ? "PASS" : "FAIL")
all_pass || error("origin-ZC shared-backend D=4 gate FAILED")
