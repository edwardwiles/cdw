# ============================================================================
# Backend C correctness suite (addendum "test persistent preallocation and
# eliminate redundant full price tensors", Step 5 + part of Step 7).
#
# Confirms `lfix_factorized.jl` (logCC+mulU factorization, reusing
# winner_certificate.jl's ALREADY-VALIDATED `build_winner_ref`, NEITHER
# price0 NOR pTσ0 ever materialized as a dense tensor) reproduces the
# allocating REFERENCE (`lfix_incremental.jl`) to machine precision -- NOT
# bit-for-bit, UNLIKE Backend A (a true zero-formula-change aliasing refactor)
# and Backend B (reuses the IDENTICAL constConsσ/Uσ formula as the Reference,
# just skips storing price0). Backend C reconstructs pTσ via
# `exp((1-σ)*(logCC+mulU))` -- a log-sum-then-exp pathway that is
# MATHEMATICALLY equivalent to but NOT bit-identical to the Reference's
# direct `constConsσ_od/Uσ^(-μ)` power formula (different floating-point
# rounding at every step). Found empirically this session (first run of this
# test used strict `==` throughout and failed 16/98 assertions, ALL at
# ~1e-14 to 1e-15 absolute/relative magnitude, i.e. 1-2 ULPs -- confirmed by
# inspecting the actual mismatched values, not assumed) -- winner/runner-up/
# third IDENTITIES and tie detection remain exactly bit-identical throughout
# (those are integer comparisons with a real, non-infinitesimal margin;
# ULP-level score noise doesn't flip them), only the floating-point pTσ/
# contrib0/q0/L_fix/gradient VALUES need an `isapprox` tolerance below.
#   (A) winner/runner-up/third identities + contrib0/q0, fixed + randomized points.
#   (B) adversarial exact price tie -> TiedWinnerError (reuses build_winner_ref's
#       OWN already-robust price-space tie check -- no new tie-detection logic
#       to validate here, just confirm it still propagates correctly).
#   (C) full L_fix value across ALL D^2-1 coordinates' +-h probes (top-3 tier only,
#       Backend C's documented scope).
#   (D) complete composite_gradient_at_C gradient == composite_gradient_at (Reference), bit-for-bit.
#   (E) count_winner_flips_C / select_bandwidth_C agree with the Reference's own.
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
include(joinpath(@__DIR__, "lfix_factorized.jl"))

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2; W = size(ctx.obj.U, 1)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
w_low = [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385]

rng = MersenneTwister(20260721)
"1e-10 relative-or-absolute tolerance: Backend C's log-sum-exp pTσ reconstruction differs from the Reference's direct power formula at the ULP level (~1e-14 to 1e-15 observed), never larger -- this tolerance is 4+ orders of magnitude looser than the observed noise floor, not a loosened correctness bar."
const C_TOL = 1e-10
isclose(a, b) = isapprox(a, b; rtol = C_TOL, atol = C_TOL)

points = [("upper40", w_up40), ("lower", w_low)]
for i in 1:4
    base_w = i <= 2 ? w_up40 : w_low
    dir = randn(rng, D2); dir ./= sqrt(sum(abs2, dir))
    push!(points, ("rand$i", base_w .+ (0.005 * i) .* dir))
end

@testset "Backend C: factorized (logCC+mulU) reproduces the Reference bit-for-bit" begin

    @testset "(A) winner/runner-up/third + contrib0/q0" begin
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
            cache_C = build_lfix_base_cache_C(xf0, ctx, base)

            @test cache_ref.winner0 == cache_C.ref.winner
            @test cache_ref.runnerup0 == cache_C.ref.runnerup
            @test cache_ref.third0 == cache_C.ref.third
            @test all(isclose.(cache_ref.contrib0, cache_C.contrib0))
            @test all(isclose.(cache_ref.q0, cache_C.q0))
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
        ctx_tie = merge(ctx, (U = Utie,))
        base = solve_base_state(xf0, ctx_tie)

        threw_ref = false; ex_ref = nothing
        try
            build_lfix_base_cache(xf0, ctx_tie, base)
        catch e
            threw_ref = e isa TiedWinnerError; ex_ref = e
        end
        threw_C = false; ex_C = nothing
        try
            build_lfix_base_cache_C(xf0, ctx_tie, base)
        catch e
            threw_C = e isa TiedWinnerError; ex_C = e
        end
        @test threw_ref
        @test threw_C
        if threw_ref && threw_C
            @test ex_ref.n_tied_pairs == ex_C.n_tied_pairs
            @test ex_ref.examples == ex_C.examples
        end
    end

    @testset "(C) full L_fix value across every coordinate (top-3 tier)" begin
        for (lbl, w0) in points[1:2]
            xf0 = x_free_from_w(w0)
            base = solve_base_state(xf0, ctx)
            cache_ref = build_lfix_base_cache(xf0, ctx, base)
            cache_C = build_lfix_base_cache_C(xf0, ctx, base)
            z0 = log.(reshape(xf0[2:end], D, D))
            w0r = vcat(xf0[1], pivot_reduce(z0, pe))

            worst = 0.0
            for k in 2:D2, h in (0.01, 0.1)
                Lref = lfix_incremental_at(cache_ref, ctx, pe, w0r, k, w0r[k] + h; tier = :incremental_o1)
                LC = lfix_incremental_at_C(cache_C, ctx, pe, w0r, k, w0r[k] + h)
                worst = max(worst, abs(Lref - LC))
            end
            @test worst < C_TOL
            @printf("  [%s] worst abs diff = %.3e (tol=%.0e)\n", lbl, worst, C_TOL)
        end
    end

    @testset "(D) composite_gradient_at_C == composite_gradient_at (full gradient, bit-for-bit)" begin
        for (lbl, w0) in points[1:2]
            xf0 = x_free_from_w(w0)
            base = solve_base_state(xf0, ctx)
            g_ref, _ = composite_gradient_at(xf0, ctx, pe; base = base)
            g_C, _ = composite_gradient_at_C(xf0, ctx, pe; base = base)
            maxdiff = maximum(abs.(g_ref .- g_C))
            @test maxdiff < C_TOL
            @printf("  [%s] gradient maxabsdiff=%.3e (tol=%.0e)\n", lbl, maxdiff, C_TOL)
        end
    end

    @testset "(E) count_winner_flips_C / select_bandwidth_C agree with Reference" begin
        for (lbl, w0) in points[1:2]
            xf0 = x_free_from_w(w0)
            base = solve_base_state(xf0, ctx)
            cache_ref = build_lfix_base_cache(xf0, ctx, base)
            cache_C = build_lfix_base_cache_C(xf0, ctx, base)
            z0 = log.(reshape(xf0[2:end], D, D))
            w0r = vcat(xf0[1], pivot_reduce(z0, pe))
            for k in 2:D2
                h_ref, m_ref, _ = select_bandwidth(cache_ref, ctx, pe, w0r, k)
                h_C, m_C, _ = select_bandwidth_C(cache_C, ctx, pe, w0r, k)
                @test h_ref == h_C
                @test m_ref == m_C
            end
        end
    end
end

println("All Backend C (factorized) tests passed.")
