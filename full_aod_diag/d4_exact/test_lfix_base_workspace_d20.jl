# ============================================================================
# Permanent D=20 omit-ROW regression test for the rectangular generalization of
# LFixBaseWorkspace/build_lfix_base_cache!/composite_gradient_at_Aplus (shared-FG-verification-
# and-A-gradient release, 2026-07-27). Previously this persistent-cache backend was hard
# square-only (`Ddest_here == D || error(...)`) -- this test is the first time it has ever been
# exercised under the real D=20 production default (destination_sample=:exclude_row, D=20/
# Ddest=19). Structure mirrors test_gradient_workspace.jl's own real-D20 gate.
#
# Usage: julia --project=. full_aod_diag/d4_exact/test_lfix_base_workspace_d20.jl
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
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "gradient_workspace.jl"))
include(joinpath(@__DIR__, "lfix_base_workspace_pooled.jl"))

println("Building real D=20 context (W=80000, delta=1.0) for LFixBaseWorkspace rectangular gate...")
ctx = d20_real_setup(W = 80000, δ = 1.0, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D; D2 = D * Ddest; W = ctx.W
@assert Ddest != D "sanity: this gate is meaningless unless the context is genuinely rectangular"

x_free_from_w2(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, Ddest), pe)
gp0 = ctx.θ0_up[3+D]
xf_calib = x_free_from_w2(vcat(gp0 * 1.01, zfree0))

Random.seed!(778)
dir = randn(length(zfree0)); dir ./= sqrt(sum(abs2, dir))
xf_near = x_free_from_w2(vcat(gp0 * 1.01, zfree0 .+ 0.05 .* dir))

points = [("calibration", xf_calib), ("nearby_perturbed", xf_near)]

@testset "D=20 rectangular (Ddest=$Ddest != D=$D): build_lfix_base_cache! matches allocating reference, composite_gradient_at_Aplus matches composite_gradient_at_fast_pooled" begin
    ws_ref = Ref(build_lfix_base_workspace(D, Ddest, W))
    pool = build_grad_workspace_pool(W)
    for (label, xf) in points
        base = solve_base_state(xf, ctx)

        # 1) cache-level equivalence: persistent-workspace cache vs allocating reference, every field
        cache_ref = build_lfix_base_cache(xf, ctx, base; validate_dense = true)
        ws = ensure_lfix_workspace!(ws_ref, D, Ddest, W)
        cache_ws = build_lfix_base_cache!(ws, xf, ctx, base; validate_dense = true)
        @testset "$label: cache fields bit-identical" begin
            @test cache_ws.D == cache_ref.D && cache_ws.Ddest == cache_ref.Ddest
            @test cache_ws.price0 == cache_ref.price0
            @test cache_ws.pTσ0 == cache_ref.pTσ0
            @test cache_ws.winner0 == cache_ref.winner0
            @test cache_ws.contrib0 == cache_ref.contrib0
            @test cache_ws.CONST_d == cache_ref.CONST_d
            @test cache_ws.q0 == cache_ref.q0
        end

        # 2) full-gradient equivalence: Backend A+ (persistent cache + pooled coordinate buffers) vs
        #    the already-D20-gated composite_gradient_at_fast_pooled (fresh cache each call)
        bwc_ref = Dict{Int,Float64}(); bwc_aplus = Dict{Int,Float64}()
        g_ref, _ = composite_gradient_at_fast_pooled(xf, ctx, pe, pool; base = base, threaded = false, h_mode = :cached, bandwidth_cache = bwc_ref)
        g_aplus, _ = composite_gradient_at_Aplus(xf, ctx, pe, pool, ws; base = base, threaded = false, h_mode = :cached, bandwidth_cache = bwc_aplus)
        @test length(g_aplus) == D2
        ok = g_ref == g_aplus
        maxdiff = maximum(abs.(g_ref .- g_aplus))
        @printf("  [%s] A+ vs pooled: max|Δg|=%.3e (must be 0), bit-identical=%s\n", label, maxdiff, ok)
        @test ok

        # 3) second point through the SAME workspace: catches stale-array bugs a single-point test
        #    (e.g. always building fresh) could hide.
    end

    println("\n  Confirming a THIRD, freshly-drawn point through the SAME warm workspace (stale-data check)...")
    Random.seed!(991)
    dir2 = randn(length(zfree0)); dir2 ./= sqrt(sum(abs2, dir2))
    xf_third = x_free_from_w2(vcat(gp0 * 0.99, zfree0 .+ 0.03 .* dir2))
    base3 = solve_base_state(xf_third, ctx)
    ws = ensure_lfix_workspace!(ws_ref, D, Ddest, W)
    cache_ref3 = build_lfix_base_cache(xf_third, ctx, base3; validate_dense = true)
    cache_ws3 = build_lfix_base_cache!(ws, xf_third, ctx, base3; validate_dense = true)
    @test cache_ws3.q0 == cache_ref3.q0
    @printf("  third point through warm workspace: q0 bit-identical=%s\n", cache_ws3.q0 == cache_ref3.q0)
end

println("All D=20 rectangular LFixBaseWorkspace tests passed.")
