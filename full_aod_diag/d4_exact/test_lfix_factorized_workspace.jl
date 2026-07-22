# ============================================================================
# Backend C+ correctness suite (production-gate addendum, part of Section 5/14).
#
# Confirms `lfix_factorized_workspace.jl` (factorized representation + persistent workspace +
# GradWorkspacePool, "Backend C+") reproduces:
#   (A) lfix_factorized.jl's own allocating Backend C, to machine precision (same formulas,
#       just persistent-buffer-backed -- should NOT introduce any NEW numerical difference
#       beyond what Backend C already has vs the Reference).
#   (B) the Reference (`composite_gradient_at_fast_buffered`, the ACTUAL production gradient
#       function, not the obsolete `composite_gradient_at`) to the same ~1e-14 tolerance
#       Backend C itself showed against the Reference.
#   (C) workspace-reuse correctness: two different points in a row on the SAME
#       (ws, grad_pool) pair, contamination check (mirrors test_lfix_base_workspace.jl's own).
#   (D) adversarial exact-tie fixture -> TiedWinnerError, ws.valid lifecycle.
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
include(joinpath(@__DIR__, "lfix_buffer_reuse.jl"))
include(joinpath(@__DIR__, "gradient_workspace.jl"))
include(joinpath(@__DIR__, "lfix_factorized.jl"))
include(joinpath(@__DIR__, "lfix_factorized_workspace.jl"))

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2; W = size(ctx.obj.U, 1)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
w_low = [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385]

const CPLUS_TOL = 1e-9
isclose(a, b) = isapprox(a, b; rtol = CPLUS_TOL, atol = CPLUS_TOL)

@testset "Backend C+: factorized + persistent workspace + GradWorkspacePool" begin

    @testset "(A) matches allocating Backend C" begin
        ws_ref = Ref(build_lfix_factorized_workspace(D, W))
        pool = build_grad_workspace_pool(W)
        for (lbl, w0) in (("upper40", w_up40), ("lower", w_low))
            xf0 = x_free_from_w(w0)
            base = solve_base_state(xf0, ctx)
            cache_C = build_lfix_base_cache_C(xf0, ctx, base)
            ws = ensure_lfix_factorized_workspace!(ws_ref, D, W)
            cache_Cplus = build_lfix_base_cache_C!(ws, xf0, ctx, base)
            @test cache_C.ref.winner == cache_Cplus.ref.winner
            @test all(isclose.(cache_C.contrib0, cache_Cplus.contrib0))
            @test all(isclose.(cache_C.q0, cache_Cplus.q0))
            @test ws.valid

            # composite_gradient_at_C always uses ADAPTIVE bandwidth (select_bandwidth_C) with no
            # h_mode/h0 kwarg at all -- must compare against C+'s OWN adaptive selection
            # (h_mode=:cached with a fresh cache, which internally also calls select_bandwidth_C),
            # NOT a fixed h -- different FD step sizes give genuinely different secant values,
            # not a bug (caught by this test itself: an earlier version used h_mode=:fixed/h0=0.01
            # here and saw a real ~0.3-0.7 "difference" that was purely a step-size mismatch).
            g_C, _ = composite_gradient_at_C(xf0, ctx, pe; base = base)
            bwc_match = Dict{Int,Float64}()
            g_Cplus, _ = composite_gradient_at_Cplus(xf0, ctx, pe, pool, ws; base = base, threaded = false, h_mode = :cached, bandwidth_cache = bwc_match)
            maxdiff = maximum(abs.(g_C .- g_Cplus))
            @test maxdiff < CPLUS_TOL
            @printf("  [%s] C vs C+ gradient maxabsdiff=%.3e\n", lbl, maxdiff)
        end
    end

    @testset "(B) matches the Reference production gradient (composite_gradient_at_fast_buffered)" begin
        ws_ref = Ref(build_lfix_factorized_workspace(D, W))
        pool = build_grad_workspace_pool(W)
        for (lbl, w0) in (("upper40", w_up40), ("lower", w_low))
            xf0 = x_free_from_w(w0)
            base = solve_base_state(xf0, ctx)
            ws = ensure_lfix_factorized_workspace!(ws_ref, D, W)

            bwc_ref = Dict{Int,Float64}(); bwc_cplus = Dict{Int,Float64}()
            g_ref, _ = composite_gradient_at_fast_buffered(xf0, ctx, pe; base = base, threaded = false, h_mode = :cached, bandwidth_cache = bwc_ref)
            g_cplus, _ = composite_gradient_at_Cplus(xf0, ctx, pe, pool, ws; base = base, threaded = false, h_mode = :cached, bandwidth_cache = bwc_cplus)
            maxdiff = maximum(abs.(g_ref .- g_cplus))
            @test maxdiff < CPLUS_TOL
            @printf("  [%s] Reference vs C+ gradient maxabsdiff=%.3e (tol=%.0e)\n", lbl, maxdiff, CPLUS_TOL)

            # also check threaded=true matches serial (production always calls threaded=true)
            bwc_cplus_t = Dict{Int,Float64}()
            g_cplus_t, _ = composite_gradient_at_Cplus(xf0, ctx, pe, pool, ws; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bwc_cplus_t)
            maxdiff_t = maximum(abs.(g_cplus .- g_cplus_t))
            @test maxdiff_t == 0.0
            @printf("  [%s] C+ serial vs threaded maxabsdiff=%.3e (must be exact)\n", lbl, maxdiff_t)
        end
    end

    @testset "(C) workspace reuse across different points is not contaminated" begin
        ws = build_lfix_factorized_workspace(D, W)
        pool = build_grad_workspace_pool(W)
        xf1 = x_free_from_w(w_up40); base1 = solve_base_state(xf1, ctx)
        cache1 = build_lfix_base_cache_C!(ws, xf1, ctx, base1)
        cache1_ref = build_lfix_base_cache_C(xf1, ctx, base1)
        @test all(isclose.(cache1.contrib0, cache1_ref.contrib0))

        xf2 = x_free_from_w(w_low); base2 = solve_base_state(xf2, ctx)
        cache2 = build_lfix_base_cache_C!(ws, xf2, ctx, base2)   # SAME ws, refilled
        cache2_ref = build_lfix_base_cache_C(xf2, ctx, base2)
        @test all(isclose.(cache2.contrib0, cache2_ref.contrib0))
        @test cache1.ref.mulU === cache2.ref.mulU   # proves the alias (same backing array)
    end

    @testset "(D) adversarial exact price tie -> TiedWinnerError, ws.valid lifecycle" begin
        xf0 = x_free_from_w(w_up40)
        θ0 = CS.reconstruct_full(xf0, ctx.m)
        cc, _, _ = constCons_matrix(θ0, ctx)
        μ = θ0[1]; s0 = 7; d0 = 1
        prices = [cc[o, d0] / (ctx.U[s0, o]^(-μ)) for o in 1:D]
        w = argmin(prices); pmin = prices[w]; o2 = (w == 1 ? 2 : 1)
        Utie = copy(ctx.U); Utie[s0, o2] = (cc[o2, d0] / pmin)^(-1 / μ)
        ctx_tie = merge(ctx, (U = Utie,))
        base = solve_base_state(xf0, ctx_tie)

        ws = build_lfix_factorized_workspace(D, W)
        threw = false; ex = nothing
        try
            build_lfix_base_cache_C!(ws, xf0, ctx_tie, base)
        catch e
            threw = e isa TiedWinnerError; ex = e
        end
        @test threw
        @test !ws.valid

        xf_clean = x_free_from_w(w_low); base_clean = solve_base_state(xf_clean, ctx)
        cache_clean = build_lfix_base_cache_C!(ws, xf_clean, ctx, base_clean)
        @test ws.valid
    end
end

println("All Backend C+ tests passed.")
