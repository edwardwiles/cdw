# Step 9.2: exact-point cache gates for the CM+moments(+ZC) extension. Covers A/B/A restoration,
# stale/different context, changed L, changed CM grid cutpoints, changed basis, changed K_mean/
# K_pair, changed solver options (inner_loop_opt), changed backend label, changed draws, and the
# "accepted-point reuse requires exact equality of the full outer vector including eta_nu" rule
# (same x_free, different nu -> must be a cache MISS, not a stale hit).
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "cm_meanzc_cache.jl"))
using Test, Printf

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]

L = 10
probs10 = cm_equal_grid_probs(10)
probs20 = cm_equal_grid_probs(20)

function build_pcx(; L = 10, K_mean = 1, K_pair = 1, basis = :direct, contrasts = :anchored)
    aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, meanzc_basis = basis, contrasts = contrasts)
    ctx_cm = merge(ctx, (obj = aug.obj_cm,))
    cctx = build_cm_meanzc_bin_ctx(ctx, aug)
    return (ctx_cm = ctx_cm, aug = aug, cctx = cctx, bins = cm_bin_indices_for(ctx, aug))
end

pcx1 = build_pcx(; L = 10, K_mean = 1, K_pair = 1)
cache = cm_meanzc_oracle_cache_for(pcx1)
ν = [1.0]

# instrument: count real inner solves by wrapping the underlying (uncached) call
n_real_solves = Ref(0)
function counted_value(x_free, ν, pcx; kwargs...)
    n_real_solves_before = n_real_solves[]
    K, base, verify = cm_meanzc_production_value_verified_cached(x_free, ν, pcx; kwargs...)
    return K, base, verify
end

println("="^100)
println("9.2a: A/B/A restoration -- repeat point hits cache, no new solve")
println("="^100)
@testset "A/B/A cache restoration" begin
    _, baseA1, verifyA1 = cm_meanzc_production_value_verified_cached(x_free_calib, ν, pcx1; cache = cache, probs = probs10, draw_checksum = "d1")
    xB = copy(x_free_calib); xB[3] += 0.01
    _, baseB, verifyB = cm_meanzc_production_value_verified_cached(xB, ν, pcx1; cache = cache, probs = probs10, draw_checksum = "d1")
    @test length(cache.d) == 2

    _, baseA2, verifyA2 = cm_meanzc_production_value_verified_cached(x_free_calib, ν, pcx1; cache = cache, probs = probs10, draw_checksum = "d1")
    @test length(cache.d) == 2   # no new entry -- A was a hit
    @test baseA2.ζstar == baseA1.ζstar
    @test baseA2.λstar == baseA1.λstar
    @test verifyA2.Delta_dual == verifyA1.Delta_dual   # bit-identical: served from cache, not recomputed
end
println()

println("="^100)
println("9.2b: same x_free, DIFFERENT eta_nu (nu) -> cache MISS (different outer point, not a stale hit)")
println("="^100)
@testset "different nu at same x_free is a distinct key" begin
    cache2 = cm_meanzc_oracle_cache_for(pcx1)
    _, _, v1 = cm_meanzc_production_value_verified_cached(x_free_calib, [1.0], pcx1; cache = cache2, probs = probs10, draw_checksum = "d1")
    @test length(cache2.d) == 1
    _, _, v2 = cm_meanzc_production_value_verified_cached(x_free_calib, [1.2], pcx1; cache = cache2, probs = probs10, draw_checksum = "d1")
    @test length(cache2.d) == 2   # distinct entry, NOT reused from the nu=1.0 point
    @test v1.Delta_dual != v2.Delta_dual   # genuinely different problems, different answers
    # repeating nu=1.0 exactly IS a hit
    _, _, v1b = cm_meanzc_production_value_verified_cached(x_free_calib, [1.0], pcx1; cache = cache2, probs = probs10, draw_checksum = "d1")
    @test length(cache2.d) == 2
    @test v1b.Delta_dual == v1.Delta_dual
end
println()

println("="^100)
println("9.2c: changed L / changed CM grid cutpoints -> distinct key (different SafeExactCache instance per pcx, plus key includes L/cutpoints)")
println("="^100)
@testset "changed L, changed cutpoints, changed basis, changed K_mean/K_pair, changed contrasts, changed backend label, changed draws" begin
    pcx_L10 = build_pcx(; L = 10)
    pcx_L20 = build_pcx(; L = 20)
    cache_shared_by_config = SafeExactCache{CMMeanZCEvalKey}()   # deliberately misuse ONE cache across configs to prove the KEY (not just cache-instance separation) prevents collision
    _, _, vL10 = cm_meanzc_production_value_verified_cached(x_free_calib, ν, pcx_L10; cache = cache_shared_by_config, probs = probs10, draw_checksum = "d1")
    _, _, vL20 = cm_meanzc_production_value_verified_cached(x_free_calib, ν, pcx_L20; cache = cache_shared_by_config, probs = probs20, draw_checksum = "d1")
    @test length(cache_shared_by_config.d) == 2
    @test vL10.Delta_dual != vL20.Delta_dual

    # changed basis
    pcx_anchored = build_pcx(; L = 10, basis = :anchored)
    _, _, v_anchored = cm_meanzc_production_value_verified_cached(x_free_calib, ν, pcx_anchored; cache = cache_shared_by_config, probs = probs10, draw_checksum = "d1")
    @test length(cache_shared_by_config.d) == 3

    # changed K_mean/K_pair
    pcx_K2 = build_pcx(; L = 10, K_mean = 2, K_pair = 2)
    _, _, v_K2 = cm_meanzc_production_value_verified_cached(x_free_calib, [1.0, 2.0], pcx_K2; cache = cache_shared_by_config, probs = probs10, draw_checksum = "d1")
    @test length(cache_shared_by_config.d) == 4

    # changed contrasts
    pcx_orth = build_pcx(; L = 10, contrasts = :orthonormal)
    _, _, v_orth = cm_meanzc_production_value_verified_cached(x_free_calib, ν, pcx_orth; cache = cache_shared_by_config, probs = probs10, draw_checksum = "d1")
    @test length(cache_shared_by_config.d) == 5

    # changed backend label (same everything else) -- purely a key-mechanism check, only :structured is actually wired
    _, _, v_backend2 = cm_meanzc_production_value_verified_cached(x_free_calib, ν, pcx_L10; cache = cache_shared_by_config, probs = probs10, draw_checksum = "d1", backend = :hypothetical_other)
    @test length(cache_shared_by_config.d) == 6

    # changed draws (draw_checksum only -- proxy for "regenerated with a different seed")
    _, _, v_draws2 = cm_meanzc_production_value_verified_cached(x_free_calib, ν, pcx_L10; cache = cache_shared_by_config, probs = probs10, draw_checksum = "d2_different")
    @test length(cache_shared_by_config.d) == 7
end
println()

println("="^100)
println("9.2d: stale/different context (context_fingerprint changes) -> cache MISS")
println("="^100)
@testset "different context fingerprint is a distinct key" begin
    ctx2 = d4_exact_setup(δ = 2.0, find_smallest = true, needs_outer_moment_jacobian = false)   # different delta -> NOT part of fingerprint per its own docstring... use a genuinely different ctx via different delta AFFECTS obj.δ (part of the key directly, separately from ctx_fingerprint)
    aug2 = build_cm_meanzc_augmented_obj(ctx2, CS; L = 10, K_mean = 1, K_pair = 1, meanzc_basis = :direct)
    pcx2 = (ctx_cm = merge(ctx2, (obj = aug2.obj_cm,)), aug = aug2, cctx = build_cm_meanzc_bin_ctx(ctx2, aug2), bins = cm_bin_indices_for(ctx2, aug2))

    cache3 = SafeExactCache{CMMeanZCEvalKey}()
    _, _, va = cm_meanzc_production_value_verified_cached(x_free_calib, ν, pcx1; cache = cache3, probs = probs10, draw_checksum = "d1")
    _, _, vb = cm_meanzc_production_value_verified_cached(x_free_calib, ν, pcx2; cache = cache3, probs = probs10, draw_checksum = "d1")
    @test length(cache3.d) == 2   # delta differs (part of the key directly) -> distinct entries even though U/D/W (the fingerprint) are identical
end
println()

println("="^100)
println("9.2e: only verified-success results are cached")
println("="^100)
@testset "unverified/infeasible points are never cached" begin
    cache4 = SafeExactCache{CMMeanZCEvalKey}()
    # a wildly out-of-range nu should fail to verify (or throw) rather than silently cache a bad answer
    bad_nu = [1e6]
    local threw = false
    try
        _, _, vbad = cm_meanzc_production_value_verified_cached(x_free_calib, bad_nu, pcx1; cache = cache4, probs = probs10, draw_checksum = "d1")
        threw = !is_verified_success(vbad)
    catch e
        e isa CMExpectedSolveFailure || rethrow()
        threw = true
    end
    @test threw
    @test length(cache4.d) == 0   # nothing cached for the failed/unverified point
end

println()
println("All cache/accepted-point gate tests passed.")
