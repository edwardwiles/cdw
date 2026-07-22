# ============================================================================
# Regression test for cross_delta_cache.jl (allocation/cache cleanup task §12):
# a CrossDeltaExactCache entry stored at one δ must be servable, byte-identical on every
# δ-independent field, to a caller at a DIFFERENT δ (same x_free/find_smallest), with
# Delta_minus_delta correctly reflecting the NEW δ, not the one that populated the entry.
# ============================================================================
using Test

include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "cross_delta_cache.jl"))
using Random

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

DELTA_INDEPENDENT_FIELDS = (:gamma_focal_prime, :K_hard, :Delta_dual, :Delta_primal,
    :gravity_raw, :gravity_value, :gravity_R_sum, :gravity_R_mean, :gravity_R_beta,
    :max_abs_moment_resid, :zeta, :lambda, :benchmark_unweighted_moment_mean, :m_mean, :m_min, :m_max,
    :weight_norm_resid, :mean_m_resid, :max_abs_moment_kkt_resid, :primal_dual_gap,
    :winner_hash, :inner_status, :θ_full, :logA)

@testset "CrossDeltaExactCache: store at one δ, hit at another" begin
    calib_x = ctx.θ0_up[ctx.free_idx]
    Random.seed!(9001)
    dir = randn(length(calib_x)); dir ./= sqrt(sum(abs2, dir))
    xf = calib_x .+ 0.02 .* dir

    cache = CrossDeltaExactCache()

    ctx.obj.δ = 2.0
    r_store, _ = evaluate_fullA_fast(xf, ctx; cache = cache, use_cache = true, warm = false)
    @test r_store.cache_hit == false
    @test length(cache) == 1

    # look up at a DIFFERENT delta (the staged-continuation scenario, task §12's own example)
    ctx.obj.δ = 5.0
    key = FullAEvalKey(collect(xf), ctx.obj.δ, ctx.obj.find_smallest, ctx.obj.inner_loop_opt, :hard, context_fingerprint(ctx))
    hit = _cache_lookup(cache, key)
    @test hit !== nothing
    @test hit.Delta_minus_delta == hit.Delta_dual - 5.0
    @test hit.Delta_minus_delta != r_store.Delta_minus_delta   # sanity: really did change

    # ground truth: a FRESH cold solve at δ=5, same x_free/find_smallest, must match on every
    # delta-independent field (confirms delta genuinely does not affect the inner solve, and
    # that the cache hit is not silently returning something stale)
    ctx.obj.x .= NaN
    r_fresh5, _ = evaluate_fullA_fast(xf, ctx; cache = nothing, use_cache = false, warm = false)
    @test r_fresh5.inner_status in (0, -100, -101, -103)

    for f in DELTA_INDEPENDENT_FIELDS
        va = getfield(hit, f); vb = getfield(r_fresh5, f)
        @testset "field $f matches fresh delta=5 solve" begin
            if va isa AbstractFloat
                @test (isnan(va) && isnan(vb)) || va == vb
            else
                @test va == vb
            end
        end
    end
    @test hit.Delta_minus_delta == r_fresh5.Delta_minus_delta   # both now reflect delta=5

    # a THIRD delta (3.0) also hits, still delta-independent-consistent, and stays a single entry
    ctx.obj.δ = 3.0
    key3 = FullAEvalKey(collect(xf), ctx.obj.δ, ctx.obj.find_smallest, ctx.obj.inner_loop_opt, :hard, context_fingerprint(ctx))
    hit3 = _cache_lookup(cache, key3)
    @test hit3 !== nothing
    @test hit3.Delta_dual == r_store.Delta_dual
    @test hit3.Delta_minus_delta == hit3.Delta_dual - 3.0
    @test length(cache) == 1   # still one inner-key entry -- delta never fragmented it

    # a genuinely different x_free must still miss (the cache must not become trivially permissive)
    xf2 = calib_x .+ 0.03 .* dir
    key_miss = FullAEvalKey(collect(xf2), 3.0, ctx.obj.find_smallest, ctx.obj.inner_loop_opt, :hard, context_fingerprint(ctx))
    @test _cache_lookup(cache, key_miss) === nothing
end

println("All cross-delta-cache tests passed.")
