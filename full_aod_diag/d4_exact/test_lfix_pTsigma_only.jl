# ============================================================================
# Backend B correctness suite (addendum "test persistent preallocation and
# eliminate redundant full price tensors", Step 4 + part of Step 7).
#
# Confirms `lfix_pTsigma_only.jl` (pTσ0-only, price0 NEVER computed) reproduces
# the allocating REFERENCE (`lfix_incremental.jl`) bit-for-bit:
#   (A) winner/runner-up/third identities + pTσ values, fixed + randomized D=4 points.
#   (B) adversarial exact price tie -> TiedWinnerError from BOTH, same n_tied_pairs/examples.
#   (C) full L_fix values across ALL D^2-1 coordinates' +-h probes, all 3 tiers
#       (:block_local, :incremental, :incremental_o1), both multi_method modes.
#   (D) complete composite_gradient_at_B gradient == composite_gradient_at (Reference), bit-for-bit.
#   (E) count_winner_flips*_B / select_bandwidth_B agree with the Reference's own.
# ============================================================================
using Test, Random, Printf
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "bandwidth_quantile.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "winner_certificate.jl"))       # constCons_matrix, for the tie fixture
include(joinpath(@__DIR__, "lfix_pTsigma_only.jl"))

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2; W = size(ctx.obj.U, 1)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
w_low = [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385]

rng = MersenneTwister(20260721)
points = [("upper40", w_up40), ("lower", w_low)]
for i in 1:4
    base_w = i <= 2 ? w_up40 : w_low
    dir = randn(rng, D2); dir ./= sqrt(sum(abs2, dir))
    push!(points, ("rand$i", base_w .+ (0.005 * i) .* dir))
end

@testset "Backend B: pTσ0-only reproduces the Reference bit-for-bit" begin

    @testset "(A) winner/runner-up/third + pTσ identities" begin
        for (lbl, w0) in points
            xf0 = x_free_from_w(w0)
            local base
            try
                base = solve_base_state(xf0, ctx)
            catch
                @printf("  SKIP %s: inner solve infeasible\n", lbl)
                continue
            end
            cache_ref = build_lfix_base_cache(xf0, ctx, base)
            cache_B = build_lfix_base_cache_B(xf0, ctx, base)

            @test cache_ref.winner0 == cache_B.winner0
            @test cache_ref.runnerup0 == cache_B.runnerup0
            @test cache_ref.third0 == cache_B.third0
            @test cache_ref.third_pTσ0 == cache_B.third_pTσ0
            @test cache_ref.contrib0 == cache_B.contrib0
            @test cache_ref.q0 == cache_B.q0
            # winner/runnerup pTσ VALUE must equal pTσ0 indexed at the winner/runnerup origin
            @test cache_ref.pTσ0[CartesianIndex.(1:W, cache_ref.winner0[:, 1], 1)] == cache_B.winner_pTσ0[:, 1]
        end
    end

    @testset "(B) adversarial exact price tie -> TiedWinnerError from BOTH" begin
        xf0 = x_free_from_w(w_up40)
        θ0 = CS.reconstruct_full(xf0, ctx.m)
        cc, _, _ = constCons_matrix(θ0, ctx)
        μ = θ0[1]; s0 = 7; d0 = 1
        prices = [cc[o, d0] / (ctx.U[s0, o]^(-μ)) for o in 1:D]
        w = argmin(prices); pmin = prices[w]; o2 = (w == 1 ? 2 : 1)
        Utie = copy(ctx.U); Utie[s0, o2] = (cc[o2, d0] / pmin)^(-1 / μ)
        # Uσ = U.^(1-σ) EXACTLY, computed once at context-build time
        # (prepare_cc/createUDerivatives!.jl:18) and stored SEPARATELY in ctx.γ.Uσ -- patching
        # only ctx.U (as test_winner_certificate.jl's own tie-injection recipe does) leaves
        # ctx.γ.Uσ stale/inconsistent, which does NOT propagate the tie into pTσ-space. Must
        # patch both for an internally-consistent tied context (this dependency, and the exact
        # Uσ=U^(1-σ) relationship, is confirmed in docs/fullA_price_tensor_audit.md).
        Uσtie = copy(ctx.γ.Uσ); Uσtie[s0, o2] = Utie[s0, o2]^(1 - ctx.σ)
        γ_tie = merge(ctx.γ, (Uσ = Uσtie,))
        ctx_tie = merge(ctx, (U = Utie, γ = γ_tie))
        base = solve_base_state(xf0, ctx_tie)

        threw_ref = false; ex_ref = nothing
        try
            build_lfix_base_cache(xf0, ctx_tie, base)
        catch e
            threw_ref = e isa TiedWinnerError; ex_ref = e
        end
        threw_B = false; ex_B = nothing
        try
            build_lfix_base_cache_B(xf0, ctx_tie, base)
        catch e
            threw_B = e isa TiedWinnerError; ex_B = e
        end
        @test threw_ref
        @test threw_B
        if threw_ref && threw_B
            @test ex_ref.n_tied_pairs == ex_B.n_tied_pairs
            @test ex_ref.examples == ex_B.examples
        end
    end

    @testset "(C) full L_fix value across every coordinate, all tiers" begin
        for (lbl, w0) in points[1:2]   # the two verified-feasible fixed points
            xf0 = x_free_from_w(w0)
            base = solve_base_state(xf0, ctx)
            cache_ref = build_lfix_base_cache(xf0, ctx, base)
            cache_B = build_lfix_base_cache_B(xf0, ctx, base)
            z0 = log.(reshape(xf0[2:end], D, D))
            w0r = vcat(xf0[1], pivot_reduce(z0, pe))

            for tier in (:block_local, :incremental, :incremental_o1)
                worst = 0.0
                for k in 2:D2, h in (0.01, 0.1)
                    Lref = lfix_incremental_at(cache_ref, ctx, pe, w0r, k, w0r[k] + h; tier = tier)
                    LB = lfix_incremental_at_B(cache_B, ctx, pe, w0r, k, w0r[k] + h; tier = tier)
                    worst = max(worst, abs(Lref - LB))
                end
                @test worst == 0.0
                worst == 0.0 || @printf("  [%s tier=%s] worst abs diff = %.3e\n", lbl, tier, worst)
            end
        end
    end

    @testset "(D) composite_gradient_at_B == composite_gradient_at (full gradient, bit-for-bit)" begin
        for (lbl, w0) in points[1:2]
            xf0 = x_free_from_w(w0)
            base = solve_base_state(xf0, ctx)
            g_ref, _ = composite_gradient_at(xf0, ctx, pe; base = base)
            g_B, _ = composite_gradient_at_B(xf0, ctx, pe; base = base)
            @test g_ref == g_B
            g_ref == g_B || @printf("  [%s] gradient mismatch, maxabsdiff=%.3e\n", lbl, maximum(abs.(g_ref .- g_B)))
        end
    end

    @testset "(E) count_winner_flips_B / select_bandwidth_B agree with Reference" begin
        for (lbl, w0) in points[1:2]
            xf0 = x_free_from_w(w0)
            base = solve_base_state(xf0, ctx)
            cache_ref = build_lfix_base_cache(xf0, ctx, base)
            cache_B = build_lfix_base_cache_B(xf0, ctx, base)
            z0 = log.(reshape(xf0[2:end], D, D))
            w0r = vcat(xf0[1], pivot_reduce(z0, pe))
            for k in 2:D2
                h_ref, m_ref, _ = select_bandwidth(cache_ref, ctx, pe, w0r, k)
                h_B, m_B, _ = select_bandwidth_B(cache_B, ctx, pe, w0r, k)
                @test h_ref == h_B
                @test m_ref == m_B
            end
        end
    end
end

println("All Backend B (pTσ0-only) tests passed.")
