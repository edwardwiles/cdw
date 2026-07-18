# ============================================================================
# Section 6: threading interaction for the winner-certificate schemes.
#
# Context A -- ORDINARY exact value evaluation: draw-level threading over the
#   certificate screen/rescan may help. Measure single vs threaded, and confirm
#   the threaded result is bit-identical + deterministic.
#
# Context B -- COORDINATE-PARALLEL L_fix gradient: the outer loop over the D^2
#   coordinates is threaded (each coordinate is an independent FD probe). The
#   INNER winner update must stay single-threaded to avoid oversubscription /
#   nested Threads.@threads. Demonstrate: (a) outer-threaded + inner-serial
#   (correct discipline) beats (b) outer-serial + inner-threaded (nesting) at
#   equal thread budget.
# ============================================================================
include(joinpath(@__DIR__, "context_scaled.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))   # -> affected_cells
include(joinpath(@__DIR__, "winner_certificate.jl"))
using BenchmarkTools, Random, Printf, LinearAlgebra, Statistics

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.5
BenchmarkTools.DEFAULT_PARAMETERS.samples = 300
med_ms(b) = median(b).time / 1e6

nT = Threads.nthreads()
println("="^96)
println("SECTION 6: THREADING INTERACTION  (threads=$nT)")
println("="^96)

# ---- Context A: ordinary value eval, draw-level threading ----
println("\nContext A -- ordinary exact value eval (draw-level threading over the certificate):")
@printf("%-6s %-8s %-14s %-14s %-9s %-10s\n", "D", "W", "single_ms", "threaded_ms", "speedup", "exact?")
for (D, W) in ((4, 8000), (4, 80000), (10, 8000), (10, 80000))
    ctx = try
        (D == 4 && W == 8000) ? d4_exact_setup(find_smallest = true) : d_exact_setup_scaled(D = D, W = W, find_smallest = true)
    catch e
        @printf("%-6d %-8d ctx FAILED\n", D, W); continue
    end
    pe = build_pivot_elimination(ctx)
    x0 = vcat(ctx.θ0_up[3+ctx.D], vec(exp.(pivot_expand(zeros(ctx.D^2 - 1), pe))))  # feasible reduced coord
    ref = build_winner_ref(x0, ctx)
    rng = MersenneTwister(5)
    x1 = copy(x0); x1[2:end] .*= exp.(0.05 .* randn(rng, length(x0) - 1))    # continuation-size step
    w_s, _ = certified_winner_update(ref, ctx, x1)
    w_t, _ = certified_winner_update_threaded(ref, ctx, x1)
    exact = maximum(abs.(w_s .- w_t)) == 0
    bs = @benchmark certified_winner_update($ref, $ctx, $x1)
    bt = @benchmark certified_winner_update_threaded($ref, $ctx, $x1)
    @printf("%-6d %-8d %-14.4f %-14.4f %-9.2fx %-10s\n", D, W, med_ms(bs), med_ms(bt), med_ms(bs)/med_ms(bt), exact ? "yes" : "NO")
end

# ---- Context B: coordinate-parallel gradient, inner winner update serial ----
println("\nContext B -- coordinate-parallel L_fix-style sweep (D^2 coord FD probes):")
println("  (a) outer-threaded + inner-serial (CORRECT discipline)")
println("  (b) outer-serial   + inner-threaded (nested draw-threading, oversubscription-prone)")
let
    ctx = d4_exact_setup(find_smallest = true)
    D = ctx.D; W = size(ctx.U, 1)
    pe = build_pivot_elimination(ctx)
    x0 = vcat(ctx.θ0_up[3+ctx.D], vec(exp.(pivot_expand(zeros(D^2 - 1), pe))))
    ref = build_winner_ref(x0, ctx)
    ncoord = D^2
    h = 1e-6
    # per-coordinate FD probe: keep BOTH x_free (for the threaded cert) and theta_full (for coord update).
    function probe_for(coord)
        xp = copy(x0)
        if coord == 1
            xp[1] += h
        else
            xp[coord] *= exp(h)
        end
        return xp, CS.reconstruct_full(xp, ctx.m), affected_cells(pe, coord)
    end
    probes = [probe_for(c) for c in 1:ncoord]
    outs = [Matrix{Int}(undef, W, D) for _ in 1:ncoord]

    # (a) outer-threaded over coordinates, inner coord update SINGLE-threaded
    function sweep_a()
        Threads.@threads for c in 1:ncoord
            _, θp, cells = probes[c]
            coord_winner_update!(outs[c], ref, ctx, θp, cells)
        end
        return outs
    end
    # (b) outer serial, inner draw-threaded certificate (the nesting-prone pattern)
    function sweep_b()
        for c in 1:ncoord
            xf, _, _ = probes[c]
            w, _ = certified_winner_update_threaded(ref, ctx, xf)
            outs[c] .= w
        end
        return outs
    end
    # exactness cross-check (both vs single-threaded coord update)
    ok = true
    for c in 1:ncoord
        _, θp, cells = probes[c]
        ref_out = Matrix{Int}(undef, W, D)
        coord_winner_update!(ref_out, ref, ctx, θp, cells)
        wa = Matrix{Int}(undef, W, D); coord_winner_update!(wa, ref, ctx, θp, cells)
        wb, _ = certified_winner_update_threaded(ref, ctx, probes[c][1])
        (maximum(abs.(wa .- ref_out)) == 0 && maximum(abs.(wb .- ref_out)) == 0) || (ok = false)
    end
    ba = @benchmark $sweep_a()
    bb = @benchmark $sweep_b()
    @printf("  (a) outer-threaded + inner-serial : %.3f ms for %d coords\n", med_ms(ba), ncoord)
    @printf("  (b) outer-serial + inner-threaded : %.3f ms for %d coords  => (a) is %.2fx faster\n",
            med_ms(bb), ncoord, med_ms(bb)/med_ms(ba))
    @printf("  both exact vs single-threaded coord update: %s\n", ok ? "yes" : "NO")
end
println("="^96)
