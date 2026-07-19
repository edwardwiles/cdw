# ============================================================================
# Phase 2 equivalence test: lfix_incremental_at (both :block_local and
# :incremental tiers) MUST match the trusted full-rebuild fixed_dual_L
# (three_way_derivatives.jl) EXACTLY (near-machine-precision) across every
# coordinate, both signs, several candidate points, and several h values,
# before either tier is trusted for optimization or timing.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
using Random

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

function run_point_test(label, w0; hs = (0.02, 0.01, 0.005, 0.001))
    xf0 = x_free_from_w(w0)
    base = solve_base_state(xf0, ctx)
    # Continuation 9, Phase 3.2: build_lfix_base_cache's dense self-validation is now opt-in
    # (validate_dense=false by default) -- this test explicitly requests it since its whole
    # point is to confirm the closed-form derivation, not just the incremental-vs-fixed_dual_L
    # equivalence checked below.
    cache = build_lfix_base_cache(xf0, ctx, base; validate_dense = true)
    println("  base cache self-validation PASSED for $label (see build_lfix_base_cache internal check)")

    ok = true
    maxdiff_bl = 0.0; maxdiff_inc = 0.0; maxdiff_o1 = 0.0
    for coord in 1:length(w0), sign in (+1, -1), h in hs
        new_val = w0[coord] + sign * h
        # trusted ground truth
        w_true = copy(w0); w_true[coord] = new_val
        xf_true = x_free_from_w(w_true)
        L_true = fixed_dual_L(xf_true, ctx, base)

        L_bl = lfix_incremental_at(cache, ctx, pe, w0, coord, new_val; tier = :block_local)
        L_inc = lfix_incremental_at(cache, ctx, pe, w0, coord, new_val; tier = :incremental)
        L_o1 = lfix_incremental_at(cache, ctx, pe, w0, coord, new_val; tier = :incremental_o1)

        d_bl = abs(L_bl - L_true); d_inc = abs(L_inc - L_true); d_o1 = abs(L_o1 - L_true)
        maxdiff_bl = max(maxdiff_bl, d_bl); maxdiff_inc = max(maxdiff_inc, d_inc); maxdiff_o1 = max(maxdiff_o1, d_o1)
        if d_bl > 1e-8 || d_inc > 1e-8 || d_o1 > 1e-8
            println("    MISMATCH coord=$coord sign=$sign h=$h  L_true=$L_true  L_blocklocal=$L_bl (diff=$d_bl)  L_incremental=$L_inc (diff=$d_inc)  L_incremental_o1=$L_o1 (diff=$d_o1)")
            ok = false
        end
    end
    println(rpad(label, 34), " n_checks=", length(w0)*2*length(hs), "  max|block_local-true|=", maxdiff_bl,
            "  max|incremental-true|=", maxdiff_inc, "  max|incremental_o1-true|=", maxdiff_o1, "  ", ok ? "PASS" : "FAIL")
    return ok
end

all_ok = true
println("="^78); println("PHASE 2 EQUIVALENCE: block_local & incremental L_fix vs trusted full-rebuild fixed_dual_L"); println("="^78)

w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
global all_ok &= run_point_test("upper_maxit40", w_up40)

w_low = [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385]
global all_ok &= run_point_test("lower_stalled", w_low)

# random feasible points -- skip (not fail) points where the BASE solve itself is infeasible
# (a separate, already-documented phenomenon -- random-start infeasibility certified genuine by
# Phase F's independent LP, docs/fullA_d4_final_report.md sec 5 -- not a bug in this file)
rng = MersenneTwister(9001)
n_skipped = 0
for trial in 1:8
    w = w_up40 .+ 0.02 .* randn(rng, length(w_up40))
    local ok_point = true
    try
        ok_point = run_point_test("random_point_$trial", w; hs = (0.01, 0.001))
    catch e
        if occursin("inner solve failed", sprint(showerror, e))
            println("  random_point_$trial: SKIPPED (base solve infeasible, nStatus!=0 -- not a code bug)")
            global n_skipped += 1
            continue
        else
            rethrow()
        end
    end
    global all_ok &= ok_point
end
println("  ($n_skipped/8 random points skipped for base infeasibility)")

println("\n" * "="^78)
println(all_ok ? "ALL PHASE 2 EQUIVALENCE TESTS PASSED (block_local AND incremental tiers both exact)" : "SOME TESTS FAILED")
println("="^78)
all_ok || error("test_lfix_incremental.jl: equivalence check failed")
