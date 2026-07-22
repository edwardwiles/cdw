# Backend A+ quick correctness check (production-gate addendum): composite_gradient_at_Aplus
# must be bit-identical to composite_gradient_at_fast_pooled (the existing, already-validated
# pooled path) -- the only difference is where `cache`'s arrays come from (persistent workspace
# vs fresh allocation), never what's computed.
using Test, Random, Printf
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "bandwidth_quantile.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "lfix_buffer_reuse.jl"))
include(joinpath(@__DIR__, "lfix_base_workspace_pooled.jl"))

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2; W = size(ctx.obj.U, 1)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
w_low = [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385]

@testset "Backend A+ bit-identical to composite_gradient_at_fast_pooled" begin
    ws_ref = Ref(build_lfix_base_workspace(D, W))
    pool = build_grad_workspace_pool(W)
    for (lbl, w0) in (("upper40", w_up40), ("lower", w_low))
        xf0 = x_free_from_w(w0)
        base = solve_base_state(xf0, ctx)
        ws = ensure_lfix_workspace!(ws_ref, D, W)

        bwc_ref = Dict{Int,Float64}(); bwc_aplus = Dict{Int,Float64}()
        g_ref, _ = composite_gradient_at_fast_pooled(xf0, ctx, pe, pool; base = base, threaded = false, h_mode = :cached, bandwidth_cache = bwc_ref)
        g_aplus, _ = composite_gradient_at_Aplus(xf0, ctx, pe, pool, ws; base = base, threaded = false, h_mode = :cached, bandwidth_cache = bwc_aplus)
        @test g_ref == g_aplus

        bwc_aplus_t = Dict{Int,Float64}()
        g_aplus_t, _ = composite_gradient_at_Aplus(xf0, ctx, pe, pool, ws; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bwc_aplus_t)
        @test g_aplus_t == g_aplus
        @printf("  [%s] A+ vs pooled maxabsdiff=%.3e (must be 0), A+ serial vs threaded maxabsdiff=%.3e (must be 0)\n",
            lbl, maximum(abs.(g_ref .- g_aplus)), maximum(abs.(g_aplus .- g_aplus_t)))
    end
end
println("All Backend A+ tests passed.")
