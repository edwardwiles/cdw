# ============================================================================
# Regression test formalizing c14_verify_lfix_buffer_fix.jl (a print-based
# script) into a real @testset: price_and_pTsigma_cell! (the in-place,
# buffer-reuse variant wired into build_lfix_base_cache) must be bit-for-bit
# identical to the original allocating price_and_pTsigma_cell, at real D=20
# points. Backfilled for the final-production-merge consolidation (brief §6):
# lfix_incremental.jl's own docstring cited this testset before it existed --
# see the corrected citation there.
# ============================================================================
using Test, Printf, LinearAlgebra, Random

include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))

println("Building real D=20 context (W=80000, delta=1.0) for lfix buffer-fix test...")
ctx = d20_real_setup(W = 80000, δ = 1.0, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2

x_free_from_w2(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, D), pe)
gp0 = ctx.θ0_up[3+D]
xf_calib = x_free_from_w2(vcat(gp0 * 1.01, zfree0))

Random.seed!(777)
dir = randn(length(zfree0)); dir ./= sqrt(sum(abs2, dir))
xf_near = x_free_from_w2(vcat(gp0 * 1.01, zfree0 .+ 0.05 .* dir))

points = [("calibration", xf_calib), ("nearby_perturbed", xf_near)]

@testset "price_and_pTsigma_cell! bit-for-bit identical to allocating version" begin
    for (label, xf) in points
        θ_full = CS.reconstruct_full(xf, ctx.m)
        W = size(ctx.obj.U, 1)
        price_ref = Array{Float64}(undef, W, D, D); pTσ_ref = Array{Float64}(undef, W, D, D)
        price_new = Array{Float64}(undef, W, D, D); pTσ_new = Array{Float64}(undef, W, D, D)
        for d in 1:D, o in 1:D
            p, ps = price_and_pTsigma_cell(θ_full, ctx, o, d)
            price_ref[:, o, d] .= p; pTσ_ref[:, o, d] .= ps
            price_and_pTsigma_cell!(@view(price_new[:, o, d]), @view(pTσ_new[:, o, d]), θ_full, ctx, o, d)
        end
        @testset "$label" begin
            @test price_ref == price_new
            @test pTσ_ref == pTσ_new
        end
    end

    @testset "build_lfix_base_cache: end-to-end + allocation reduction" begin
        xf = xf_calib
        base = solve_base_state(xf, ctx)
        GC.gc()
        stats = @timed build_lfix_base_cache(xf, ctx, base; validate_dense = false)
        cache = stats.value
        mb = stats.bytes / 2^20
        @printf("  build_lfix_base_cache: wall=%.3fs  bytes=%.3e (%.1f MB)  [pre-fix baseline ~1078MB]\n", stats.time, stats.bytes, mb)
        @test all(isfinite, cache.q0)
        @test mb < 750.0   # pre-fix baseline was ~1078MB; fix measured ~590-650MB -- generous margin for server noise

        # validate_dense=true's own independent dense rebuild must still agree -- the strongest
        # available correctness gate for the cache contents themselves, reused not reinvented.
        @test build_lfix_base_cache(xf, ctx, base; validate_dense = true) !== nothing
    end
end

println("All lfix-buffer-fix tests passed.")
