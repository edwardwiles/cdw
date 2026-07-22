# ============================================================================
# Backend :kbplus correctness suite (finalization task Phase 5, 2026-07-22).
#
# Directly mirrors test_lfix_factorized.jl's (Backend C+) own test structure and REUSES its
# exact fixed points (w_up40, w_low, rand1-4 built from the same RNG seed) -- attributed here,
# not silently re-derived, per this repo's own established preference for reused-code
# attribution. Confirms `lfix_kbplus.jl` (ratio-based `constConsσ/USigmaPow` reconstruction,
# NO exp/log/pow call anywhere in a coordinate probe) reproduces:
#   (1) the allocating REFERENCE (`lfix_incremental.jl`) -- expected close to bit-identical
#       (same power-formula shape as the Reference's own `constConsσ_od/Uσ^(-μ)`, just computed
#       via a persistent USigmaPow rather than a dense W×D×D tensor -- should have LESS
#       rounding drift than C+'s log-exp path, not more; measured, not assumed).
#   (2) Backend C+ (`lfix_factorized.jl`) -- both are mathematically equivalent reconstructions
#       of the same pTσ value via different floating-point paths; agreement here is a THIRD,
#       independent cross-check beyond either one's own agreement with the Reference.
#   (A) winner/runner-up/third identities + contrib0/q0, fixed + randomized points.
#   (B) adversarial exact price tie -> TiedWinnerError (reuses build_winner_ref's own check).
#   (C) full L_fix value across every coordinate's +-h probe (top-3 tier).
#   (D) complete composite_gradient_at_KB gradient vs Reference AND vs C+.
#   (E) count_winner_flips_KB / select_bandwidth_KB agree with the Reference's own.
#   (F) extreme-range A_od probes (h up to +-20) stay finite and agree with the Reference.
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
include(joinpath(@__DIR__, "lfix_kbplus.jl"))

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2; W = size(ctx.obj.U, 1)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

# Same fixed points as test_lfix_factorized.jl (Backend C+'s own suite) -- reused verbatim.
w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
w_low = [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385]

rng = MersenneTwister(20260721)
const KB_TOL = 1e-10
isclose(a, b) = isapprox(a, b; rtol = KB_TOL, atol = KB_TOL)

points = [("upper40", w_up40), ("lower", w_low)]
for i in 1:4
    base_w = i <= 2 ? w_up40 : w_low
    dir = randn(rng, D2); dir ./= sqrt(sum(abs2, dir))
    push!(points, ("rand$i", base_w .+ (0.005 * i) .* dir))
end

n_pass_extra = 0; n_fail_extra = 0
function xcheck(name, cond)
    global n_pass_extra, n_fail_extra
    if cond
        n_pass_extra += 1; println("  PASS: ", name)
    else
        n_fail_extra += 1; println("  FAIL: ", name)
    end
end

@testset "Backend :kbplus (ratio, no W-scale exp) reproduces the Reference and C+" begin

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
            cache_KB = build_lfix_base_cache_KB(xf0, ctx, base)

            @test cache_ref.winner0 == cache_KB.ref.winner
            @test cache_ref.runnerup0 == cache_KB.ref.runnerup
            @test cache_ref.third0 == cache_KB.ref.third
            @test cache_C.ref.winner == cache_KB.ref.winner   # cross-check vs C+ too
            @test all(isclose.(cache_ref.contrib0, cache_KB.contrib0))
            @test all(isclose.(cache_ref.q0, cache_KB.q0))
            @test all(isclose.(cache_C.contrib0, cache_KB.contrib0))
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
        threw_KB = false; ex_KB = nothing
        try
            build_lfix_base_cache_KB(xf0, ctx_tie, base)
        catch e
            threw_KB = e isa TiedWinnerError; ex_KB = e
        end
        @test threw_ref
        @test threw_KB
        if threw_ref && threw_KB
            @test ex_ref.n_tied_pairs == ex_KB.n_tied_pairs
            @test ex_ref.examples == ex_KB.examples
        end
    end

    @testset "(C) full L_fix value across every coordinate (top-3 tier)" begin
        for (lbl, w0) in points[1:2]
            xf0 = x_free_from_w(w0)
            base = solve_base_state(xf0, ctx)
            cache_ref = build_lfix_base_cache(xf0, ctx, base)
            cache_KB = build_lfix_base_cache_KB(xf0, ctx, base)
            z0 = log.(reshape(xf0[2:end], D, D))
            w0r = vcat(xf0[1], pivot_reduce(z0, pe))

            worst = 0.0
            for k in 2:D2, h in (0.01, 0.1)
                Lref = lfix_incremental_at(cache_ref, ctx, pe, w0r, k, w0r[k] + h; tier = :incremental_o1)
                LKB = lfix_incremental_at_KB(cache_KB, ctx, pe, w0r, k, w0r[k] + h)
                worst = max(worst, abs(Lref - LKB))
            end
            @test worst < KB_TOL
            @printf("  [%s] worst abs diff vs Reference = %.3e (tol=%.0e)\n", lbl, worst, KB_TOL)
        end
    end

    @testset "(D) composite_gradient_at_KB vs Reference AND vs C+ (full gradient)" begin
        for (lbl, w0) in points[1:2]
            xf0 = x_free_from_w(w0)
            base = solve_base_state(xf0, ctx)
            g_ref, _ = composite_gradient_at(xf0, ctx, pe; base = base)
            g_C, _ = composite_gradient_at_C(xf0, ctx, pe; base = base)
            g_KB, _ = composite_gradient_at_KB(xf0, ctx, pe; base = base)
            maxdiff_ref = maximum(abs.(g_ref .- g_KB))
            maxdiff_C = maximum(abs.(g_C .- g_KB))
            @test maxdiff_ref < KB_TOL
            @test maxdiff_C < KB_TOL
            @printf("  [%s] gradient maxabsdiff vs Reference=%.3e vs C+=%.3e (tol=%.0e)\n", lbl, maxdiff_ref, maxdiff_C, KB_TOL)
        end
    end

    @testset "(E) count_winner_flips_KB / select_bandwidth_KB agree with Reference" begin
        for (lbl, w0) in points[1:2]
            xf0 = x_free_from_w(w0)
            base = solve_base_state(xf0, ctx)
            cache_ref = build_lfix_base_cache(xf0, ctx, base)
            cache_KB = build_lfix_base_cache_KB(xf0, ctx, base)
            z0 = log.(reshape(xf0[2:end], D, D))
            w0r = vcat(xf0[1], pivot_reduce(z0, pe))
            for k in 2:D2
                h_ref, m_ref, _ = select_bandwidth(cache_ref, ctx, pe, w0r, k)
                h_KB, m_KB, _ = select_bandwidth_KB(cache_KB, ctx, pe, w0r, k)
                @test h_ref == h_KB
                @test m_ref == m_KB
            end
        end
    end
end

println("\n== (F) extreme-range A_od probes (h up to +-20), :kbplus vs Reference ==")
for (lbl, w0) in points[1:2]
    xf0 = x_free_from_w(w0)
    base = solve_base_state(xf0, ctx)
    cache_ref = build_lfix_base_cache(xf0, ctx, base)
    cache_KB = build_lfix_base_cache_KB(xf0, ctx, base)
    z0 = log.(reshape(xf0[2:end], D, D))
    w0r = vcat(xf0[1], pivot_reduce(z0, pe))
    worst_abs = 0.0; worst_rel = 0.0; n_nonfinite_ref = 0; n_nonfinite_KB = 0; n_mismatch_finiteness = 0
    for k in 2:D2, h in (-20.0, -10.0, -1.0, 1.0, 10.0, 20.0)
        Lref = try
            lfix_incremental_at(cache_ref, ctx, pe, w0r, k, w0r[k] + h; tier = :incremental_o1)
        catch
            NaN
        end
        LKB = try
            lfix_incremental_at_KB(cache_KB, ctx, pe, w0r, k, w0r[k] + h)
        catch
            NaN
        end
        !isfinite(Lref) && (n_nonfinite_ref += 1)
        !isfinite(LKB) && (n_nonfinite_KB += 1)
        if isfinite(Lref) != isfinite(LKB)
            n_mismatch_finiteness += 1
        elseif isfinite(Lref)
            worst_abs = max(worst_abs, abs(Lref - LKB))
            worst_rel = max(worst_rel, abs(Lref - LKB) / max(abs(Lref), 1.0))
        end
    end
    # extreme steps (h=+-20, i.e. Aod_theta up to e^20~4.85e8) legitimately produce large-magnitude
    # L values, so an ABSOLUTE-only 1e-10 bound (appropriate near h=0.01-0.1) is too strict here --
    # use the same relative-or-absolute standard as every other check in this file (isclose).
    xcheck("[$lbl] extreme-range: no finiteness mismatch between Reference and :kbplus", n_mismatch_finiteness == 0)
    xcheck("[$lbl] extreme-range: finite-value agreement, rel-or-abs < $KB_TOL (worst_abs=$worst_abs worst_rel=$worst_rel)",
        worst_abs < KB_TOL || worst_rel < KB_TOL)
    println("  [$lbl] n_nonfinite_ref=$n_nonfinite_ref n_nonfinite_KB=$n_nonfinite_KB worst_abs=$worst_abs worst_rel=$worst_rel")
end

println("\n============================================================")
println("Extra (non-@testset) checks: $n_pass_extra passed, $n_fail_extra failed")
n_fail_extra == 0 || error("$n_fail_extra extra check(s) failed")
println("All Backend :kbplus tests passed.")
