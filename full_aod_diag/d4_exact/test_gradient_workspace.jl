# ============================================================================
# Regression test for gradient_workspace.jl (allocation/cache cleanup task §7):
# composite_gradient_at_fast_pooled must be bit-for-bit identical to the existing
# composite_gradient_at_fast_buffered, at real D=20/W=80000 points, both serial and
# threaded, and must allocate substantially less per call.
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
include(joinpath(@__DIR__, "bandwidth_quantile.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))   # gamma_component_analytic, select_bandwidth
include(joinpath(@__DIR__, "gradient_workspace.jl"))   # includes lfix_buffer_reuse.jl transitively

println("Building real D=20 context (W=80000, delta=1.0) for gradient-workspace test...")
ctx = d20_real_setup(W = 80000, δ = 1.0, find_smallest = true)
pe = build_pivot_elimination(ctx)
# BUGFIX (shared-FG-verification-and-A-gradient release, 2026-07-27): this previously hardcoded
# D2=D^2 and reshape(D,D), assuming a SQUARE A_od block -- correct only for destination_sample=
# :all_legacy. `d20_real_setup`'s own default is destination_sample=:exclude_row (D=20/Ddest=19,
# real production default), under which this test never even reached the functions it exists to
# regression-test (it errored at setup, BoundsError on the D^2-sized slice, before ever calling
# composite_gradient_at_fast_pooled/_buffered) -- this is the "permanent D=20 omit-ROW regression
# test" gap task item 5 asks to close, not just the two production functions' own bug.
D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D; D2 = D * Ddest; W = ctx.W

x_free_from_w2(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, Ddest), pe)
gp0 = ctx.θ0_up[3+D]
xf_calib = x_free_from_w2(vcat(gp0 * 1.01, zfree0))

Random.seed!(778)
dir = randn(length(zfree0)); dir ./= sqrt(sum(abs2, dir))
xf_near = x_free_from_w2(vcat(gp0 * 1.01, zfree0 .+ 0.05 .* dir))

points = [("calibration", xf_calib), ("nearby_perturbed", xf_near)]

@testset "composite_gradient_at_fast_pooled bit-identical to composite_gradient_at_fast_buffered" begin
    pool = build_grad_workspace_pool(W)
    for (label, xf) in points
        base = solve_base_state(xf, ctx)
        bwc_ref = Dict{Int,Float64}()
        bwc_new = Dict{Int,Float64}()
        g_ref, meta_ref = composite_gradient_at_fast_buffered(xf, ctx, pe; base = base, threaded = false,
            h_mode = :cached, bandwidth_cache = bwc_ref)
        g_new, meta_new = composite_gradient_at_fast_pooled(xf, ctx, pe, pool; base = base, threaded = false,
            h_mode = :cached, bandwidth_cache = bwc_new)
        @testset "$label (serial)" begin
            @test g_ref == g_new
            @test meta_ref.h_used == meta_new.h_used
        end

        bwc_ref_t = Dict{Int,Float64}()
        bwc_new_t = Dict{Int,Float64}()
        g_ref_t, _ = composite_gradient_at_fast_buffered(xf, ctx, pe; base = base, threaded = true,
            h_mode = :cached, bandwidth_cache = bwc_ref_t)
        g_new_t, _ = composite_gradient_at_fast_pooled(xf, ctx, pe, pool; base = base, threaded = true,
            h_mode = :cached, bandwidth_cache = bwc_new_t)
        @testset "$label (threaded, nthreads=$(Threads.nthreads()))" begin
            @test g_ref_t == g_new_t
            @test g_ref_t == g_ref   # threaded must also match serial
        end
    end

    @testset "pool reuse: second call on a warm pool allocates less than a fresh q_bufs/psi_bufs build" begin
        base = solve_base_state(xf_calib, ctx)
        bwc = Dict{Int,Float64}()
        # warm
        composite_gradient_at_fast_pooled(xf_calib, ctx, pe, pool; base = base, threaded = false, h_mode = :fixed, h0 = 0.01)
        GC.gc()
        b_pooled = @allocated composite_gradient_at_fast_pooled(xf_calib, ctx, pe, pool; base = base, threaded = false, h_mode = :fixed, h0 = 0.01)
        GC.gc()
        b_buffered = @allocated composite_gradient_at_fast_buffered(xf_calib, ctx, pe; base = base, threaded = false, h_mode = :fixed, h0 = 0.01)
        @printf("  pooled:   %.1f MB   buffered: %.1f MB   (ratio %.2fx)\n", b_pooled/1e6, b_buffered/1e6, b_buffered/b_pooled)
        @test b_pooled < b_buffered / 3   # expect a large, not marginal, reduction
    end
end

println("All gradient-workspace tests passed.")
