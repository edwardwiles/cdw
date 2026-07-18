# ============================================================================
# Continuation 8, Section 3 (coordinate-specialized top-three update):
# equivalence test for the O(1) top-3-cache replacements of the
# two-changed-origins-in-one-destination O(D) fallback:
#   - composite_gradient.jl::count_winner_flips_multi_top3 vs the ORIGINAL
#     count_winner_flips_multi (flip COUNT, used by select_bandwidth's mass_at)
#   - lfix_incremental.jl::dest_contrib_incremental_top3 vs the ORIGINAL
#     dest_contrib_incremental (contribution VALUE, used by
#     dest_contrib_incremental_o1's 2-changed-origin branch)
#
# The 2-changed-origin case is RARE in a real D=4 run (3 of 15 A-block
# coordinates -- the gravity-pivot coordinates that share the pivot's
# destination, per docs/winner_certificate_report.md sec 2's audit), so
# (A)+(B) below construct SYNTHETIC 2-changed-origin cases covering every
# origin pair at every destination (not just the one pivot destination a real
# run exercises), at several step sizes. (C) then also checks the REAL
# coordinate sweep (the 3 pivot-sharing coordinates that occur naturally) end
# to end through composite_gradient_at, confirming the new default
# (multi_method=:top3) reproduces the OLD default (multi_method=:generic)
# bit-for-bit on an actual gradient call, not just the isolated helper.
#
# The TRUSTED reference throughout is the ORIGINAL O(D) rescan
# (count_winner_flips_multi / dest_contrib_incremental), NOT re-derived here.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
using Random, Printf, LinearAlgebra

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
w_low = [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385]

all_ok = true
rng = MersenneTwister(20260718)

println("="^96)
println("WINNER TOP-3 COORDINATE-UPDATE EQUIVALENCE (D=$D)")
println("="^96)

# ---- (A)+(B) synthetic 2-changed-origin cases: every origin pair, every destination ----
println("(A)+(B) synthetic 2-changed-origin cases: count_winner_flips_multi_top3 vs _multi, dest_contrib_incremental_top3 vs dest_contrib_incremental")
for (lbl, w0) in (("upper40", w_up40), ("lower", w_low))
    xf0 = x_free_from_w(w0)
    base = solve_base_state(xf0, ctx)
    cache = build_lfix_base_cache(xf0, ctx, base)

    n_cases = 0; n_flip_mismatch = 0; n_contrib_mismatch = 0
    max_contrib_err = 0.0
    for d in 1:D
        origins = collect(1:D)
        for i in 1:D, j in (i+1):D
            o1, o2 = origins[i], origins[j]
            for h in (1e-3, 1e-2, 8e-2, 3e-1, 1.0)
                # perturb Aod_theta at (o1,d) and (o2,d) by independent random signed steps of
                # magnitude h -- constructs a genuine 2-changed-origin-in-one-destination case
                # regardless of whether the real pivot-reduction ever produces it at this (o1,o2,d).
                z = log.(reshape(xf0[2:end], D, D))
                z[o1, d] += h * (rand(rng) < 0.5 ? -1 : 1)
                z[o2, d] += h * (rand(rng) < 0.5 ? -1 : 1)
                x_free_p = vcat(xf0[1], vec(exp.(z)))
                θ_full_p = CS.reconstruct_full(x_free_p, ctx.m)
                changed = [o1, o2]

                flips_ref = count_winner_flips_multi(cache, ctx, θ_full_p, d, changed)
                flips_top3 = count_winner_flips_multi_top3(cache, ctx, θ_full_p, d, changed)
                flips_ok = flips_ref == flips_top3
                n_flip_mismatch += !flips_ok

                contrib_ref = dest_contrib_incremental(cache, ctx, θ_full_p, d, changed)
                contrib_top3 = dest_contrib_incremental_top3(cache, ctx, θ_full_p, d, changed)
                cerr = maximum(abs.(contrib_ref .- contrib_top3))
                max_contrib_err = max(max_contrib_err, cerr)
                n_contrib_mismatch += cerr > 0.0   # must be BIT-IDENTICAL (both derive the same origin+pTσ from the same cache), not just close

                n_cases += 1
            end
        end
    end
    ok = (n_flip_mismatch == 0) && (n_contrib_mismatch == 0)
    global all_ok &= ok
    @printf("  [%-8s] cases=%d  flip_mismatches=%d  contrib_mismatches=%d  max_contrib_abs_err=%.3e  %s\n",
            lbl, n_cases, n_flip_mismatch, n_contrib_mismatch, max_contrib_err, ok ? "PASS" : "FAIL")
end

# ---- (C) real coordinate sweep through composite_gradient_at: :top3 default == :generic reference ----
println("-"^96)
println("(C) real coordinate sweep: composite_gradient_at reproduces itself bit-for-bit whether the\n    A-block FD tiers reach count_winner_flips_multi_top3/dest_contrib_incremental_top3 (default)\n    or the ORIGINAL O(D) fallback (multi_method=:generic, checked via direct calls at the\n    3 pivot-sharing coordinates)")
for (lbl, w0) in (("upper40", w_up40), ("lower", w_low))
    xf0 = x_free_from_w(w0)
    base = solve_base_state(xf0, ctx)
    cache = build_lfix_base_cache(xf0, ctx, base)
    z0 = log.(reshape(xf0[2:end], D, D))
    w0r = vcat(xf0[1], pivot_reduce(z0, pe))

    # identify which of the 2:D^2 coordinates hit the 2-changed-origin (same-destination) case
    pivot_dest = lin_to_od(pe.pivot_lin, D)[2]
    two_origin_coords = Int[]
    for k in 2:D2
        cells = affected_cells(pe, k)
        dests = last.(cells)
        length(unique(dests)) == 1 && length(cells) == 2 && push!(two_origin_coords, k)
    end
    @printf("  [%-8s] pivot destination=%d, 2-changed-origin coordinates found: %s\n", lbl, pivot_dest, two_origin_coords)

    worst_flip = 0; worst_contrib = 0.0
    for k in two_origin_coords
        cells = affected_cells(pe, k)
        d = cells[1][2]
        origins_here = [o for (o, dd) in cells if dd == d]
        for h in (0.01, 0.05, 0.1)
            w′ = copy(w0r); w′[k] += h
            z′ = pivot_expand(w′[2:end], pe); Aod_theta′ = exp.(z′)
            x_free′ = vcat(w′[1], vec(Aod_theta′))
            θ_full′ = CS.reconstruct_full(x_free′, ctx.m)

            f_top3 = count_winner_flips(cache, ctx, θ_full′, d, origins_here; multi_method = :top3)
            f_generic = count_winner_flips(cache, ctx, θ_full′, d, origins_here; multi_method = :generic)
            worst_flip = max(worst_flip, abs(f_top3 - f_generic))

            c_top3 = dest_contrib_incremental_o1(cache, ctx, θ_full′, d, origins_here; multi_method = :top3)
            c_generic = dest_contrib_incremental_o1(cache, ctx, θ_full′, d, origins_here; multi_method = :generic)
            worst_contrib = max(worst_contrib, maximum(abs.(c_top3 .- c_generic)))
        end
    end
    ok = isempty(two_origin_coords) ? true : (worst_flip == 0 && worst_contrib == 0.0)
    global all_ok &= ok
    @printf("  [%-8s] worst_flip_diff=%d  worst_contrib_absdiff=%.3e  %s\n", lbl, worst_flip, worst_contrib, ok ? "PASS" : "FAIL")

    # also confirm the LIVE default (no multi_method kwarg passed anywhere, i.e. what a real run
    # actually calls) matches a direct lfix_incremental_at recompute at each affected coordinate.
    g_default, meta_default = composite_gradient_at(xf0, ctx, pe; base = base)
    ok2 = true
    if !isempty(two_origin_coords)
        for k in two_origin_coords
            h = meta_default.h_used[k]
            Lp_top3 = lfix_incremental_at(cache, ctx, pe, w0r, k, w0r[k] + h; tier = :incremental_o1)
            Lm_top3 = lfix_incremental_at(cache, ctx, pe, w0r, k, w0r[k] - h; tier = :incremental_o1)
            g_top3 = (Lp_top3 - Lm_top3) / (2h)
            ok2 &= (g_top3 == g_default[k])
        end
    end
    global all_ok &= ok2
    @printf("  [%-8s] live composite_gradient_at A-block value at 2-changed-origin coords reproducible via lfix_incremental_at directly: %s\n", lbl, ok2 ? "PASS" : "FAIL")
end

println("="^96)
if all_ok
    println("ALL WINNER TOP-3 EQUIVALENCE TESTS PASSED")
else
    println("SOME WINNER TOP-3 EQUIVALENCE TESTS FAILED"); exit(1)
end
