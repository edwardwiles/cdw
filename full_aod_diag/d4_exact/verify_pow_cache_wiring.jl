# ============================================================================
# Verification that enable_pow_cache! (moments_fast.jl) wires the fixed-mu,sigma
# MuSigmaPowCache into the LIVE oracle path (evaluate_fullA / evaluate_fullA_fast)
# with ZERO change to results. Compares a cache-free ctx against a cache-enabled
# ctx, full evaluate_fullA field-for-field, at calibration + both candidates +
# random feasible perturbations. Because enable_pow_cache! only swaps the moment
# build for a byte-identical implementation, the ENTIRE downstream result
# (inner solve, Delta, gravity, winners, KKT residual) must be bit-identical.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "moments_fast.jl"))
using Random, Printf

# ---- build two independent contexts: one plain, one with the pow cache wired in ----
ctx_ref   = d4_exact_setup(find_smallest = true)
ctx_cache = d4_exact_setup(find_smallest = true)
pow_cache = enable_pow_cache!(ctx_cache)

pe = build_pivot_elimination(ctx_ref)
D = ctx_ref.D; D2 = D^2
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

points = Dict(
    "calibration" => vcat(ctx_ref.θ0_up[3+D], pivot_reduce(zeros(D, D), pe)),
    "upper_lfixcomposite_sr1_60s" => [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181, 0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515, 1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252, 0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845],
    "lower_stalled" => [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385],
)
rng = MersenneTwister(20260718)
for i in 1:5
    points["random_$i"] = points["upper_lfixcomposite_sr1_60s"] .+ 0.05 .* randn(rng, D2)
end

fields_scalar = (:gamma_focal_prime, :K_hard, :Delta_dual, :Delta_primal, :Delta_minus_delta,
                 :gravity_value, :gravity_R_mean, :gravity_R_sum, :gravity_R_beta, :zeta,
                 :m_mean, :m_min, :m_max, :mean_m_resid, :max_abs_moment_kkt_resid,
                 :max_abs_moment_resid, :primal_dual_gap, :weight_norm_resid, :K_hard)
fields_int = (:inner_status, :winner_hash)

function compare(rr, rc, label; tol = 1e-13)
    ok = true
    worst = 0.0; worst_f = :none
    for f in fields_scalar
        a = getfield(rr, f); b = getfield(rc, f)
        if a === nothing && b === nothing; continue; end
        if isnan(a) && isnan(b); continue; end
        d = abs(a - b)
        if d > worst; worst = d; worst_f = f; end
        d > tol && (ok = false)
    end
    for f in fields_int
        getfield(rr, f) == getfield(rc, f) || (ok = false)
    end
    # vector fields
    for f in (:lambda, :benchmark_unweighted_moment_mean, :logA)
        a = vec(collect(getfield(rr, f))); b = vec(collect(getfield(rc, f)))
        if length(a) == length(b) && !isempty(a)
            d = maximum(abs.(a .- b))
            d > worst && (worst = d; worst_f = f)
            d > tol && (ok = false)
        elseif length(a) != length(b)
            ok = false
        end
    end
    @printf("  %-30s worst|Δ|=%.3e (%s)  %s\n", label, worst, worst_f, ok ? "PASS" : "FAIL")
    return ok
end

all_pass = true
println("="^78)
println("LIVE-PATH EQUIVALENCE: evaluate_fullA, cache-free ctx vs pow-cache-wired ctx")
println("="^78)
for (label, w) in points
    xf = x_free_from_w(w)
    rr = evaluate_fullA(xf, ctx_ref;   cache = nothing, warm = false)
    rc = evaluate_fullA(xf, ctx_cache; cache = nothing, warm = false)
    global all_pass &= compare(rr, rc, label)
end

println("\n" * "="^78)
println("LIVE-PATH EQUIVALENCE: evaluate_fullA_fast, cache-free ctx vs pow-cache-wired ctx")
println("="^78)
# fresh contexts so evaluate_fullA_fast's own warm-start state is clean
ctx_ref2   = d4_exact_setup(find_smallest = true)
ctx_cache2 = d4_exact_setup(find_smallest = true)
pow_cache2 = enable_pow_cache!(ctx_cache2)
for (label, w) in points
    xf = x_free_from_w(w)
    rr, _ = evaluate_fullA_fast(xf, ctx_ref2;   cache = nothing, warm = false)
    rc, _ = evaluate_fullA_fast(xf, ctx_cache2; cache = nothing, warm = false)
    global all_pass &= compare(rr, rc, label)
end

println("\npow_cache stats (evaluate_fullA path):      n_recompute=$(pow_cache.n_recompute)  n_reuse=$(pow_cache.n_reuse)")
println("pow_cache2 stats (evaluate_fullA_fast path): n_recompute=$(pow_cache2.n_recompute)  n_reuse=$(pow_cache2.n_reuse)")
cache_ok = pow_cache.n_recompute == 1 && pow_cache.n_reuse >= 1 &&
           pow_cache2.n_recompute == 1 && pow_cache2.n_reuse >= 1
println(cache_ok ? "CACHE BEHAVIOR: PASS (recomputed exactly once per ctx, reused thereafter)" :
                   "CACHE BEHAVIOR: FAIL")
all_pass &= cache_ok

println("\n" * "="^78)
println(all_pass ? "ALL POW-CACHE WIRING EQUIVALENCE CHECKS PASSED" :
                   "SOME CHECKS FAILED -- do not trust the wiring")
println("="^78)
all_pass || error("verify_pow_cache_wiring.jl: equivalence checks failed")
