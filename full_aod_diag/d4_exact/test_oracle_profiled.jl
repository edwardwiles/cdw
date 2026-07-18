# Equivalence test: evaluate_fullA_profiled must be bit-identical to evaluate_fullA
# (oracle.jl) before it is trusted for any timing conclusion.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_profiled.jl"))

ctx = d4_exact_setup()
x0 = CS.pack_free(ctx.θ0_up, ctx.m)

println("="^78); println("EQUIVALENCE TEST: evaluate_fullA_profiled vs evaluate_fullA"); println("="^78)

for (label, warm) in (("warm", true), ("cold", false))
    r_ref = evaluate_fullA(x0, ctx; cache = nothing, warm = warm)
    r_prof, meta = evaluate_fullA_profiled(x0, ctx; cache = nothing, warm = warm)
    fields_to_check = (:gamma_focal_prime, :K_hard, :Delta_dual, :Delta_primal, :Delta_minus_delta,
                        :gravity_value, :gravity_R_mean, :zeta, :mean_m_resid,
                        :max_abs_moment_kkt_resid, :winner_hash, :inner_status)
    all_match = true
    for f in fields_to_check
        a = getfield(r_ref, f); b = getfield(r_prof, f)
        ok = a == b || (a isa Real && b isa Real && (isnan(a) && isnan(b) || a === b))
        ok || (all_match = false; println("  MISMATCH on $f: ref=$a profiled=$b"))
    end
    println("[$label] all fields match: ", all_match, "  (prof_meta: $meta)")
    all_match || error("evaluate_fullA_profiled diverges from evaluate_fullA -- DO NOT trust timing results until fixed")
end

# also check a cache-enabled path (cache_lookup / cache_materialize instrumentation)
cache = oracle_cache_for(ctx)
r1, meta1 = evaluate_fullA_profiled(x0, ctx; cache = cache, warm = true)
r2, meta2 = evaluate_fullA_profiled(x0, ctx; cache = cache, warm = true)
println("\ncache test: first call cache_hit=", r1.cache_hit, " (expect false), second call cache_hit=", r2.cache_hit, " (expect true)")
@assert !r1.cache_hit && r2.cache_hit "cache instrumentation broke cache semantics"
@assert meta2.n_inner_solves == 0 "a cache HIT must not trigger a real inner solve -- got $(meta2.n_inner_solves)"
println("PASS: cache semantics preserved, cache-hit correctly shows n_inner_solves=0")

println("\n" * "="^78); println("ALL EQUIVALENCE CHECKS PASSED -- evaluate_fullA_profiled is trustworthy for timing"); println("="^78)
