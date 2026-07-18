# ============================================================================
# Equivalence test for winner_certificate.jl.
#   (A) build_winner_ref's winner/runnerup match compute_winners_fast exactly.
#   (B) certified_winner_update(ref, x') == compute_winners_fast(theta') winner,
#       bit-for-bit, across many nearby points (accepted-size, line-search-size,
#       continuation-size steps), and via the tol_far full-scan fallback path.
#   (C) coord_winner_update! (1- and 2-changed-cell coordinate steps) == full scan.
#   (D) winners_from_certificate's wval == compressed build_compressed_factual wval.
#   (E) tie injection -> TiedWinnerError (reused prior art).
# The TRUSTED reference is winners_v2.jl::compute_winners_fast throughout.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "winner_certificate.jl"))
using Random, Printf, LinearAlgebra, Statistics

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; W = size(ctx.U, 1)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
w_low = [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385]

trusted_winner(θfull) = compute_winners_fast(θfull, ctx)[1]

all_ok = true
println("="^96)
println("WINNER-CERTIFICATE EQUIVALENCE vs trusted compute_winners_fast (D=$D, W=$W)")
println("="^96)

# ---- (A) reference build matches the trusted scan ----
for (lbl, w0) in (("upper40", w_up40), ("lower", w_low))
    xf0 = x_free_from_w(w0)
    ref = build_winner_ref(xf0, ctx)
    θ0 = CS.reconstruct_full(xf0, ctx.m)
    wt = trusted_winner(θ0)
    errA = maximum(abs.(ref.winner .- wt))
    ok = errA == 0
    global all_ok &= ok
    @printf("(A) ref.winner vs trusted  [%-8s]  maxabsdiff=%d  %s\n", lbl, errA, ok ? "PASS" : "FAIL")
end

# ---- (B) certificate == full scan across step sizes / directions ----
println("-"^96)
rng = MersenneTwister(20260718)
for (lbl, w0) in (("upper40", w_up40), ("lower", w_low))
    xf0 = x_free_from_w(w0)
    ref = build_winner_ref(xf0, ctx)
    for (sz_lbl, sz) in (("accept 1e-3", 1e-3), ("linesrch 5e-2", 5e-2), ("continuation 2e-1", 2e-1), ("far 1.0", 1.0))
        worst = 0; agg_cert = 0; agg_cells = 0
        for rep in 1:6
            w′ = copy(w0); w′ .+= sz .* randn(rng, length(w0))
            xf′ = x_free_from_w(w′)
            θ′ = CS.reconstruct_full(xf′, ctx.m)
            wt = trusted_winner(θ′)
            wc, st = certified_winner_update(ref, ctx, xf′)
            worst = max(worst, maximum(abs.(wc .- wt)))
            agg_cert += st.n_certified; agg_cells += st.n_cells
        end
        ok = worst == 0
        global all_ok &= ok
        @printf("(B) cert==full  [%-8s %-18s]  maxabsdiff=%d  certified=%.1f%%  %s\n",
                lbl, sz_lbl, worst, 100 * agg_cert / agg_cells, ok ? "PASS" : "FAIL")
    end
    # tol_far fallback path must also be exact
    w′ = copy(w0); w′ .+= 0.3 .* randn(rng, length(w0))
    xf′ = x_free_from_w(w′)
    wt = trusted_winner(CS.reconstruct_full(xf′, ctx.m))
    wc, st = certified_winner_update(ref, ctx, xf′; tol_far = 1e-6)
    okf = (maximum(abs.(wc .- wt)) == 0) && st.fell_back_full
    global all_ok &= okf
    @printf("(B) tol_far fallback [%-8s]  exact=%s  fell_back=%s  %s\n",
            lbl, maximum(abs.(wc .- wt)) == 0, st.fell_back_full, okf ? "PASS" : "FAIL")
end

# ---- (C) coordinate updates (1- and 2-changed-cell) == full scan ----
println("-"^96)
for (lbl, w0) in (("upper40", w_up40), ("lower", w_low))
    xf0 = x_free_from_w(w0)
    ref = build_winner_ref(xf0, ctx)
    wout = Matrix{Int}(undef, W, D)
    worst = 0
    for coord in 1:length(w0)            # reduced coords: 1 = gamma' (no A cell), 2..D^2 = z_free
        for hh in (1e-2, 8e-2, 3e-1)
            w′ = copy(w0); w′[coord] += hh
            xf′ = x_free_from_w(w′)
            θ′ = CS.reconstruct_full(xf′, ctx.m)
            wt = trusted_winner(θ′)
            cells = affected_cells(pe, coord)     # (o,d) list; empty for coord 1
            coord_winner_update!(wout, ref, ctx, θ′, cells)
            worst = max(worst, maximum(abs.(wout .- wt)))
        end
    end
    ok = worst == 0
    global all_ok &= ok
    @printf("(C) coord_update==full  [%-8s]  maxabsdiff over all coords/steps=%d  %s\n",
            lbl, worst, ok ? "PASS" : "FAIL")
end

# ---- (D) winners_from_certificate wval == compressed build wval ----
println("-"^96)
for (lbl, w0) in (("upper40", w_up40), ("lower", w_low))
    xf0 = x_free_from_w(w0)
    ref = build_winner_ref(xf0, ctx)
    w′ = copy(w0); w′ .+= 3e-2 .* randn(rng, length(w0))
    xf′ = x_free_from_w(w′)
    θ′ = CS.reconstruct_full(xf′, ctx.m)
    _, wval_c, _ = winners_from_certificate(ref, ctx, xf′)
    cf = build_compressed_factual(θ′, ctx)
    errD = maximum(abs.(wval_c .- cf.wval))
    ok = errD < 1e-12
    global all_ok &= ok
    @printf("(D) wval(cert) vs wval(compressed) [%-8s]  maxabsdiff=%.2e  %s\n", lbl, errD, ok ? "PASS" : "FAIL")
end

# ---- (E) tie injection -> TiedWinnerError ----
println("-"^96)
let
    xf0 = x_free_from_w(w_up40)
    θ0 = CS.reconstruct_full(xf0, ctx.m)
    cc, _, _ = constCons_matrix(θ0, ctx)
    μ = θ0[1]; s0 = 7; d0 = 1
    prices = [cc[o, d0] / (ctx.U[s0, o]^(-μ)) for o in 1:D]
    w = argmin(prices); pmin = prices[w]; o2 = (w == 1 ? 2 : 1)
    Utie = copy(ctx.U); Utie[s0, o2] = (cc[o2, d0] / pmin)^(-1 / μ)
    ctx_tie = merge(ctx, (U = Utie,))
    threw = false
    try
        build_winner_ref(xf0, ctx_tie)
    catch e
        threw = e isa TiedWinnerError
        threw && @printf("(E) tie injection -> TiedWinnerError (n=%d, ex=%s)  PASS\n", e.n_tied_pairs, e.examples)
    end
    threw || println("(E) tie injection: FAIL (no TiedWinnerError)")
    global all_ok &= threw
end

println("="^96)
if all_ok
    println("ALL WINNER-CERTIFICATE EQUIVALENCE TESTS PASSED")
else
    println("SOME WINNER-CERTIFICATE EQUIVALENCE TESTS FAILED"); exit(1)
end
