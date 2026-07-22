# ============================================================================
# Backend A correctness suite (addendum "test persistent preallocation and
# eliminate redundant full price tensors", Step 3 + part of Step 7).
#
# Confirms `build_lfix_base_cache!` (lfix_base_workspace.jl, workspace-backed)
# is BIT-FOR-BIT identical to the allocating reference `build_lfix_base_cache`
# (lfix_incremental.jl) on:
#   (A) several D=4 synthetic points, including a randomized fuzz sweep of
#       nearby perturbations (not just the two fixed w_up40/w_low vectors this
#       file's sibling tests use -- the addendum explicitly asks for
#       "randomized outer points").
#   (B) an adversarial exact-price-tie fixture -- both builders must throw
#       TiedWinnerError identically (same n_tied_pairs/examples).
#   (C) workspace REUSE across two different points in a row (the actual
#       steady-state usage pattern) -- second build must be correct and not
#       contaminated by the first point's stale contents; also checks
#       `ws.valid` is false immediately after a tie-triggered exception, and
#       that a subsequent successful build sets it back to true.
#   (D) that the full composite_gradient_at-style consumption (dest_contrib_*,
#       count_winner_flips, select_bandwidth, bandwidth_quantile) produces
#       IDENTICAL results whether `cache` came from the allocating or the
#       workspace-backed builder -- i.e. nothing downstream cares.
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
include(joinpath(@__DIR__, "winner_certificate.jl"))       # constCons_matrix, reused for the tie fixture
include(joinpath(@__DIR__, "lfix_base_workspace.jl"))

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2; W = size(ctx.obj.U, 1)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
w_low = [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385]

"Deep field-by-field bit-identity check between two LFixBaseCache instances (== on the struct itself isn't defined; compare fields explicitly)."
function caches_identical(c1::LFixBaseCache, c2::LFixBaseCache)
    for f in fieldnames(LFixBaseCache)
        v1 = getfield(c1, f); v2 = getfield(c2, f)
        if v1 isa AbstractArray
            v1 == v2 || return (false, f)
        else
            v1 === v2 || v1 == v2 || return (false, f)
        end
    end
    return (true, :none)
end

@testset "Backend A: build_lfix_base_cache! bit-identical to build_lfix_base_cache" begin

    @testset "(A) fixed + randomized points" begin
        rng = MersenneTwister(20260721)
        ws_ref = Ref(build_lfix_base_workspace(D, W))
        points = [("upper40", w_up40), ("lower", w_low)]
        for i in 1:8
            base_w = i <= 2 ? (i == 1 ? w_up40 : w_low) : w_low
            dir = randn(rng, D2); dir ./= sqrt(sum(abs2, dir))
            # small perturbations only -- the point is to fuzz NEARBY feasible points, not to
            # wander into the infeasible region (that's a separate, pre-existing inner-solve
            # concern unrelated to this workspace's own correctness, see nStatus=-300 handling
            # below).
            push!(points, ("rand$i", base_w .+ (0.005 * i) .* dir))
        end
        n_skipped = 0
        for (lbl, w0) in points
            xf0 = x_free_from_w(w0)
            local base
            try
                base = solve_base_state(xf0, ctx)
            catch e
                # inner dual solve infeasible/unbounded at this randomly-perturbed point --
                # unrelated to Backend A (fails before build_lfix_base_cache/! is ever called);
                # skip this draw rather than fail the whole testset on a bad random point.
                n_skipped += 1
                @printf("  SKIP point %s: solve_base_state failed (%s)\n", lbl, sprint(showerror, e))
                continue
            end
            cache_ref = build_lfix_base_cache(xf0, ctx, base)
            ws = ensure_lfix_workspace!(ws_ref, D, W)
            cache_ws = build_lfix_base_cache!(ws, xf0, ctx, base)
            ok, badfield = caches_identical(cache_ref, cache_ws)
            @test ok
            ok || @printf("  MISMATCH at point %s, field %s\n", lbl, badfield)
            @test ws.valid
        end
        @test n_skipped < length(points)   # sanity: not EVERY point was infeasible
    end

    @testset "(B) adversarial exact price tie -> TiedWinnerError from BOTH builders" begin
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
        @test threw_ref

        ws = build_lfix_base_workspace(D, W)
        threw_ws = false; ex_ws = nothing
        try
            build_lfix_base_cache!(ws, xf0, ctx_tie, base)
        catch e
            threw_ws = e isa TiedWinnerError; ex_ws = e
        end
        @test threw_ws
        @test !ws.valid   # AUD-style discipline: a failed build must leave the workspace invalid
        if threw_ref && threw_ws
            @test ex_ref.n_tied_pairs == ex_ws.n_tied_pairs
            @test ex_ref.examples == ex_ws.examples
        end

        # workspace must recover cleanly on the NEXT (non-tied) build
        xf_clean = x_free_from_w(w_low)
        base_clean = solve_base_state(xf_clean, ctx)
        cache_clean_ws = build_lfix_base_cache!(ws, xf_clean, ctx, base_clean)
        @test ws.valid
        cache_clean_ref = build_lfix_base_cache(xf_clean, ctx, base_clean)
        ok, badfield = caches_identical(cache_clean_ref, cache_clean_ws)
        @test ok
        ok || @printf("  post-tie-recovery MISMATCH at field %s\n", badfield)
    end

    @testset "(C) workspace reuse across DIFFERENT consecutive points is not contaminated" begin
        ws = build_lfix_base_workspace(D, W)
        xf1 = x_free_from_w(w_up40); base1 = solve_base_state(xf1, ctx)
        cache1_ws = build_lfix_base_cache!(ws, xf1, ctx, base1)
        cache1_ref = build_lfix_base_cache(xf1, ctx, base1)
        ok1, bf1 = caches_identical(cache1_ref, cache1_ws)
        @test ok1

        xf2 = x_free_from_w(w_low); base2 = solve_base_state(xf2, ctx)
        cache2_ws = build_lfix_base_cache!(ws, xf2, ctx, base2)   # SAME ws object, refilled
        cache2_ref = build_lfix_base_cache(xf2, ctx, base2)
        ok2, bf2 = caches_identical(cache2_ref, cache2_ws)
        @test ok2
        ok2 || @printf("  reuse-contamination MISMATCH at field %s\n", bf2)

        # cache1_ws's array fields now ALIAS ws's buffers, which have since been overwritten by
        # point 2 -- this is the documented, expected aliasing behavior (see lfix_base_workspace.jl's
        # own docstring: callers must not use a returned LFixBaseCache after the NEXT
        # build_lfix_base_cache! call on the same workspace). Confirm this IS what happens (a
        # canary, not a desired property) so the hazard is demonstrated, not just asserted in prose.
        @test cache1_ws.price0 === cache2_ws.price0   # same backing array (proves the alias, not a copy)
        @test cache1_ws.price0 != cache1_ref.price0 || xf1 == xf2   # point 1's OWN copy is now stale/wrong
    end

    @testset "(D) full downstream consumption identical (dest_contrib_*, gradient) via workspace-backed cache" begin
        ws = build_lfix_base_workspace(D, W)
        for (lbl, w0) in (("upper40", w_up40), ("lower", w_low))
            xf0 = x_free_from_w(w0)
            base = solve_base_state(xf0, ctx)
            cache_ref = build_lfix_base_cache(xf0, ctx, base)
            cache_ws = build_lfix_base_cache!(ws, xf0, ctx, base)

            z0 = log.(reshape(xf0[2:end], D, D))
            w0r = vcat(xf0[1], pivot_reduce(z0, pe))
            for k in 2:D2
                h = 0.01
                Lp_ref = lfix_incremental_at(cache_ref, ctx, pe, w0r, k, w0r[k] + h; tier = :incremental_o1)
                Lp_ws = lfix_incremental_at(cache_ws, ctx, pe, w0r, k, w0r[k] + h; tier = :incremental_o1)
                @test Lp_ref == Lp_ws
            end

            g_ref, _ = composite_gradient_at(xf0, ctx, pe; base = base)
            # composite_gradient_at rebuilds its OWN (allocating) cache internally; cross-check
            # its gradient equals one recomputed by hand from the workspace-backed cache.
            g_ws = zeros(D2)
            g_ws[1] = gamma_component_analytic(cache_ws, base, w0r[1])
            for k in 2:D2
                h, _, _ = select_bandwidth(cache_ws, ctx, pe, w0r, k)
                g_ws[k] = a_block_fd_component(cache_ws, ctx, pe, w0r, k, h)
            end
            @test g_ref == g_ws
        end
    end
end

println("All Backend A (LFixBaseWorkspace) tests passed.")
