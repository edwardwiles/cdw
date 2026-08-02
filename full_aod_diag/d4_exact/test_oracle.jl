# Tests for evaluate_fullA (task §7 requirements): determinism, cache
# hit/miss correctness, warm-vs-cold VALUE agreement (runtime may differ),
# and internal consistency (primal-dual gap, inner KKT residual, gravity
# value cross-checked against the independently-validated gravity_tariff.jl).
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))

ctx = d4_exact_setup()
x0 = CS.pack_free(ctx.θ0_up, ctx.m)

println("="^78); println("TEST 1: single evaluation sanity"); println("="^78)
r1 = evaluate_fullA(x0, ctx; cache = nothing)
println("inner_status=", r1.inner_status, "  K_hard=", r1.K_hard, "  Delta_dual=", r1.Delta_dual,
        "  Delta_primal=", r1.Delta_primal, "  primal_dual_gap=", r1.primal_dual_gap)
println("mean_m_resid=", r1.mean_m_resid, "  max_abs_moment_kkt_resid=", r1.max_abs_moment_kkt_resid)
println("gravity_value=", r1.gravity_value, "  gravity_R_mean=", r1.gravity_R_mean)
@assert r1.inner_status in (0, -100, -101, -103)
@assert r1.error_reason === nothing

println("\n---- CHECK: gravity_value matches gravity_tariff.jl's own gravity_value() bit-for-bit ----")
Aod_θ = reshape(r1.θ_full[ctx.Aod_offset+1:ctx.Aod_offset+ctx.D^2], ctx.D, ctx.D)
μh = r1.θ_full[1]
lambda_g = reshape(ctx.γ.P, (ctx.D, ctx.D))'
Aod_lvl = Aod_θ .* ctx.γ.cHat .* (((ctx.γ.wHat .* ctx.τ) ./ (ctx.γ.wHat[1,1] .* ctx.τ[1,:]')) .^ (1/μh)) .* (lambda_g ./ lambda_g[1,:]')
AodPow_check = (Aod_lvl ./ ctx.γ.cHat) .^ (-μh)
g_direct = gravity_value(ctx.τ, AodPow_check, ctx.q_tilde, ctx.N_obs; exclude_diagonal=get(ctx, :exclude_diagonal_gravity, false), exclude_cells=get(ctx, :gravity_exclude_cells, Tuple{Int,Int}[]))
println("oracle gravity_value = ", r1.gravity_value, "   direct recompute = ", g_direct, "   diff = ", abs(r1.gravity_value - g_direct))
@assert r1.gravity_value == g_direct

println("\n---- CHECK: inner KKT residuals are small at a converged solve ----")
println("mean(m)-1 residual: ", r1.mean_m_resid, "  (expect ~1e-10 or smaller at opttol=1e-12)")
println("max |mean(m.*G_j)| residual: ", r1.max_abs_moment_kkt_resid)
@assert r1.mean_m_resid < 1e-6
@assert r1.max_abs_moment_kkt_resid < 1e-6

println("\n---- CHECK: primal-dual gap is small (LFD recovers a distribution matching delta*) ----")
println("primal_dual_gap = ", r1.primal_dual_gap)
@assert r1.primal_dual_gap < 1e-6

println("\n---- CHECK: recovered weights are a valid probability distribution ----")
println("weight_norm_resid (|sum(p)-1|, by construction of p=m/sum(m)) = ", r1.weight_norm_resid)
println("m_min=", r1.m_min, " (should be > 0 for this Psi'(exp-branch)) m_max=", r1.m_max)
@assert r1.m_min > 0
@assert r1.weight_norm_resid < 1e-12

println("\n" * "="^78); println("TEST 2: determinism -- repeated exact calls, cache DISABLED"); println("="^78)
r2 = evaluate_fullA(x0, ctx; cache = nothing)
for f in (:K_hard, :Delta_dual, :Delta_primal, :gravity_value, :zeta, :mean_m_resid)
    v1 = getfield(r1, f); v2 = getfield(r2, f)
    println("  ", f, ": call1=", v1, "  call2=", v2, "  equal=", v1 == v2)
    @assert v1 == v2 "determinism FAILED on field $f: $v1 != $v2"
end
@assert r1.winner_hash == r2.winner_hash "winner_hash differs across repeated calls -- nondeterministic winner recovery"
println("Repeated calls at the same x_free are BIT-IDENTICAL across every checked field: PASS")

println("\n" * "="^78); println("TEST 3: cache hit/miss correctness"); println("="^78)
cache = oracle_cache_for(ctx)
r3a = evaluate_fullA(x0, ctx; cache = cache)
println("first call: cache_hit=", r3a.cache_hit, " (expect false)")
@assert r3a.cache_hit == false
r3b = evaluate_fullA(x0, ctx; cache = cache)
println("second call, same x: cache_hit=", r3b.cache_hit, " (expect true)")
@assert r3b.cache_hit == true
@assert r3b.Delta_dual == r3a.Delta_dual
x1 = copy(x0); x1[2] *= 1.001   # perturb one A_od entry
r3c = evaluate_fullA(x1, ctx; cache = cache)
println("third call, DIFFERENT x: cache_hit=", r3c.cache_hit, " (expect false)")
@assert r3c.cache_hit == false
@assert length(cache) == 2
println("Cache hit/miss behavior correct, no tolerance-bucket reuse: PASS")

println("\n" * "="^78); println("TEST 4: warm vs cold start -- VALUE must agree, runtime need not"); println("="^78)
obj_state_backup = copy(ctx.obj.x)
r4_warm = evaluate_fullA(x0, ctx; cache = nothing, warm = true)
r4_cold = evaluate_fullA(x0, ctx; cache = nothing, warm = false)
println("warm: Delta_dual=", r4_warm.Delta_dual, "  elapsed=", r4_warm.elapsed.inner, "s")
println("cold: Delta_dual=", r4_cold.Delta_dual, "  elapsed=", r4_cold.elapsed.inner, "s")
println("value diff = ", abs(r4_warm.Delta_dual - r4_cold.Delta_dual))
@assert abs(r4_warm.Delta_dual - r4_cold.Delta_dual) < 1e-8 "warm/cold start changed the accepted VALUE -- should only change runtime"
println("Warm vs cold start give the SAME value (diff < 1e-8): PASS")

println("\n" * "="^78)
println("ALL ORACLE TESTS PASSED")
println("="^78)
