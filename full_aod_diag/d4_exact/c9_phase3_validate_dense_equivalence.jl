# ============================================================================
# Continuation 9, Phase 3.2: correctness check for build_lfix_base_cache's new
# validate_dense kwarg (default false, skips the dense self-validation
# rebuild). Confirms the claim in that function's docstring directly rather
# than assuming it from the report: a full L_fix gradient computed with
# validate_dense=true vs validate_dense=false must be BIT-IDENTICAL (the
# dense rebuild is a pure self-check on q0 -- both branches now compute q0
# from the SAME cache-pieces formula; validate_dense=true only ADDS an
# assertion, it does not change what gets returned).
#
# Run at D=4 (cheap, canonical point) AND D=8 (gated pilot scale) per the
# standing brief's "confirm this claim directly ... at a D=4 or D=8 point"
# instruction -- both, since both are cheap.
# ============================================================================
include(joinpath(@__DIR__, "context_scaled.jl"))   # -> d4_exact_setup AND d_exact_setup_scaled; includes context.jl exactly once
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
using Printf, LinearAlgebra

function check_one(label, ctx)
    pe = build_pivot_elimination(ctx)
    D = ctx.D
    xf0 = ctx.θ0_up[ctx.free_idx]
    base = solve_base_state(xf0, ctx)

    g_true, meta_true = composite_gradient_at_fast(xf0, ctx, pe; base = base, threaded = true,
                                                     h_mode = :adaptive, multi_method = :top3, validate_dense = true)
    g_false, meta_false = composite_gradient_at_fast(xf0, ctx, pe; base = base, threaded = true,
                                                       h_mode = :adaptive, multi_method = :top3, validate_dense = false)

    maxdiff = maximum(abs.(g_true .- g_false))
    bit_identical = g_true == g_false
    @printf("  [%s] D=%d, n_free=%d: bit_identical=%s  max|diff|=%.3e  ||g_true||=%.6e\n",
            label, D, length(g_true), bit_identical, maxdiff, norm(g_true))

    # also directly confirm the cache CONTENTS (not just the downstream gradient) match --
    # a stronger check than the gradient alone, since the gradient only reads a subset of
    # the cache's fields (q0, contrib0, cf_contrib0 indirectly via lfix_incremental_at).
    cache_true = build_lfix_base_cache(xf0, ctx, base; validate_dense = true)
    cache_false = build_lfix_base_cache(xf0, ctx, base; validate_dense = false)
    q0_diff = maximum(abs.(cache_true.q0 .- cache_false.q0))
    contrib0_diff = maximum(abs.(cache_true.contrib0 .- cache_false.contrib0))
    @printf("  [%s] cache.q0 max|diff|=%.3e  cache.contrib0 max|diff|=%.3e  q0 bit_identical=%s\n",
            label, q0_diff, contrib0_diff, cache_true.q0 == cache_false.q0)

    return bit_identical, maxdiff
end

println("="^90)
println("D=4 canonical point")
println("="^90)
ctx4 = d4_exact_setup(find_smallest = true)
ok4, md4 = check_one("D=4", ctx4)

println("\n", "="^90)
println("D=8 gated-pilot scale")
println("="^90)
ctx8 = d_exact_setup_scaled(D = 8, W = 8000, find_smallest = true)
ok8, md8 = check_one("D=8", ctx8)

println("\n", "="^90)
allok = ok4 && ok8
println(allok ? "OVERALL: PASS -- validate_dense=true/false give identical gradients at D=4 and D=8" :
                "OVERALL: FAIL -- see max|diff| above")
