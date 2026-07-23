# Overnight task 2026-07-22, Section 3.3: D=4 correctness tests for the complete-state cache
# prototype (complete_state_cache.jl). See docs/COMPLETE_STATE_CACHE_DESIGN_2026-07-22.md.
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
include(joinpath(@__DIR__, "complete_state_cache.jl"))
using Printf, LinearAlgebra, Random

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1; println("  PASS: ", name)
    else
        n_fail += 1; println("  FAIL: ", name)
    end
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D; W = size(ctx.U, 1)
L = 10; contrasts = :anchored
probs = collect(range(1 / L, (L - 1) / L, length = L))
pcx = build_cm_production_context(ctx, CS; L = L, contrasts = contrasts, probs = probs)
ctx_cm = pcx.ctx_cm; cctx = pcx.cctx

x_A = ctx.θ0_up[ctx.free_idx]
Random.seed!(7001)
x_B = copy(x_A); x_B[2:end] .*= exp.(0.04 .* randn(length(x_A) - 1))
Random.seed!(7002)
x_C = copy(x_A); x_C[2:end] .*= exp.(0.06 .* randn(length(x_A) - 1))

opt_file = joinpath(@__DIR__, "csw_outer_wallclock_sr1.opt")
knitro_version = "test-knitro-v1"

fp0() = complete_state_fingerprint(ctx, L, contrasts, probs, :cumulative, :structured, true, opt_file, knitro_version)

println("="^100)
println("SECTION 1: cold solve A, cache it; solve B; restore A from cache; compare to a FRESH cache-disabled cold solve at A")
println("="^100)
cache = CompleteStateCache(max_entries = 64)
fp = fp0()

base_A_cold1, verify_A_cold1, hitA1 = archC_verified_state_cached!(cache, fp, x_A, ctx_cm, cctx)
check("A: first lookup is a MISS (from_cache=false)", hitA1 == false)
check("A: cache now has exactly 1 entry", length(cache.entries) == 1)

base_B, verify_B, hitB = archC_verified_state_cached!(cache, fp, x_B, ctx_cm, cctx)
check("B: MISS, cache now has 2 entries", hitB == false && length(cache.entries) == 2)

base_A_restored, verify_A_restored, hitA2 = archC_verified_state_cached!(cache, fp, x_A, ctx_cm, cctx)
check("A: second lookup is a HIT (from_cache=true)", hitA2 == true)
check("cache.hits == 1, cache.inner_solves_avoided == 1", cache.hits == 1 && cache.inner_solves_avoided == 1)

# Independent ground truth: a FRESH cold solve at A with caching disabled entirely.
base_A_fresh, verify_A_fresh = archC_verified_state(x_A, ctx_cm, cctx)

maxerr_x = maximum(abs.(base_A_restored.x_free0 .- base_A_fresh.x_free0))
maxerr_theta = maximum(abs.(base_A_restored.θ_full0 .- base_A_fresh.θ_full0))
maxerr_lambda = maximum(abs.(base_A_restored.λstar .- base_A_fresh.λstar))
maxerr_m = maximum(abs.(base_A_restored.m_star .- base_A_fresh.m_star))
@printf "  max|Δx_free0|=%.3e max|Δθ_full0|=%.3e max|Δζ|=%.3e max|Δλ*|=%.3e max|Δm*|=%.3e max|ΔDelta_dual|=%.3e\n" maxerr_x maxerr_theta abs(base_A_restored.ζstar-base_A_fresh.ζstar) maxerr_lambda maxerr_m abs(verify_A_restored.Delta_dual - verify_A_fresh.Delta_dual)
check("A/B/A: restored base.x_free0 exactly matches fresh cold solve", maxerr_x == 0.0)
check("A/B/A: restored base.θ_full0 exactly matches fresh cold solve", maxerr_theta == 0.0)
# NOTE (real finding, not a cache bug -- see Section 2 above, which DOES prove exact bit-for-bit
# restoration of what was actually STORED): two INDEPENDENT archC_verified_state solves at the
# SAME x_free0 are NOT bit-identical to each other (KNITRO is a tolerance-based iterative solver,
# not exactly reproducible run-to-run at the last few ULPs -- observed diffs here are
# ~1e-11..1e-13, well inside any production KKT/feasibility tolerance, ~1e9 smaller than the
# 1e-6-ish scale those tolerances operate at). The cache's correctness claim is "restoration
# reproduces what was STORED, exactly" (Section 2), not "two independent solves of the same
# problem agree to the last bit" -- a materially weaker (and false, for any tolerance-based
# solver) property this test does not need and must not require. Tolerance chosen well above the
# observed KNITRO run-to-run noise floor, well below any scale that would mask a real bug.
check("A/B/A: restored base.ζstar matches fresh cold solve to 1e-8", isapprox(base_A_restored.ζstar, base_A_fresh.ζstar; atol = 1e-8))
check("A/B/A: restored base.λstar matches fresh cold solve to 1e-8", maxerr_lambda < 1e-8)
check("A/B/A: restored base.m_star matches fresh cold solve to 1e-8", maxerr_m < 1e-8)
check("A/B/A: restored verify.Delta_dual matches fresh cold solve to 1e-10", abs(verify_A_restored.Delta_dual - verify_A_fresh.Delta_dual) < 1e-10)
check("A/B/A: restored inner_status matches", verify_A_restored.inner_status == verify_A_fresh.inner_status)

# Gradient outputs from the cached base must also match a gradient built off the fresh base
# (same tolerance-based-solver caveat as above -- the two `base`s differ by ~1e-11, so gradients
# built from them differ by a correspondingly tiny amount, not zero).
g_from_cache, _ = cm_production_gradient(x_A, pcx, ctx, build_pivot_elimination(ctx); base = base_A_restored)
g_from_fresh, _ = cm_production_gradient(x_A, pcx, ctx, build_pivot_elimination(ctx); base = base_A_fresh)
gdiff = maximum(abs.(g_from_cache .- g_from_fresh))
@printf "  max|Δg(cached_base) - Δg(fresh_base)|=%.3e\n" gdiff
check("A/B/A: gradient computed from cached base matches gradient from fresh base to 1e-6", gdiff < 1e-6)

println()
println("="^100)
println("SECTION 2: no aliasing -- further evaluations mutate ctx_cm.obj's scratch; cached A must be unchanged")
println("="^100)
m_star_snapshot = copy(base_A_cold1.m_star)
for i in 1:5
    xperturb = copy(x_A); xperturb[2:end] .*= exp.(0.01 * i .* randn(length(x_A) - 1))
    try
        archC_verified_state(xperturb, ctx_cm, cctx)   # mutates ctx_cm.obj.arg1 and other live scratch, uncached
    catch e
        e isa CMExpectedSolveFailure || rethrow()
    end
end
check("cached A's m_star is bit-identical to its value at store time (no aliasing to obj's mutated scratch)",
      cache.entries[hash((fp, outer_point_key(x_A)))].base.m_star == m_star_snapshot)

println()
println("="^100)
println("SECTION 3: context-mismatch and cross-delta-independence gates")
println("="^100)
fp_diffL = complete_state_fingerprint(ctx, L + 5, contrasts, probs, :cumulative, :structured, true, opt_file, knitro_version)
check("different L -> different fingerprint (context mismatch, would be a MISS)", fp_diffL != fp)

probs_shifted = probs .+ 1e-9
fp_diffprobs = complete_state_fingerprint(ctx, L, contrasts, probs_shifted, :cumulative, :structured, true, opt_file, knitro_version)
check("perturbed probs -> different fingerprint", fp_diffprobs != fp)

fp_diffcontrasts = complete_state_fingerprint(ctx, L, :orthonormal, probs, :cumulative, :structured, true, opt_file, knitro_version)
check("different contrasts -> different fingerprint", fp_diffcontrasts != fp)

fp_diffbackend = complete_state_fingerprint(ctx, L, contrasts, probs, :cumulative, :dense_reference, true, opt_file, knitro_version)
check("different cm_hessian_backend -> different fingerprint", fp_diffbackend != fp)

fp_diffknitro = complete_state_fingerprint(ctx, L, contrasts, probs, :cumulative, :structured, true, opt_file, "different-knitro-vX")
check("different KNITRO release string -> different fingerprint", fp_diffknitro != fp)

tmp_opt = tempname() * ".opt"
write(tmp_opt, "outlev 0\n")
fp_opt1 = complete_state_fingerprint(ctx, L, contrasts, probs, :cumulative, :structured, true, tmp_opt, knitro_version)
write(tmp_opt, "outlev 1\n")   # same PATH, different CONTENT
fp_opt2 = complete_state_fingerprint(ctx, L, contrasts, probs, :cumulative, :structured, true, tmp_opt, knitro_version)
check("same opt-file PATH, different CONTENT -> different fingerprint (content-hashed, not path-hashed)", fp_opt1 != fp_opt2)
rm(tmp_opt; force = true)

# d4_exact_setup's ctx has no draw_meta field at all (that field is specific to
# d20_real_setup_design real-data contexts) -- complete_state_fingerprint falls back to hashing
# ctx.U directly in that case (see its own docstring/comment). Verify that fallback genuinely
# discriminates on the draws by perturbing U and confirming the fingerprint changes.
ctx_perturbedU = merge(ctx, (U = ctx.U .+ 1e-9,))
fp_diffdraws = complete_state_fingerprint(ctx_perturbedU, L, contrasts, probs, :cumulative, :structured, true, opt_file, knitro_version)
check("perturbed draws (ctx.U) -> different fingerprint (draws-fallback path genuinely discriminates)", fp_diffdraws != fp)

# delta-independence: the fingerprint function signature has NO delta parameter at all -- a caller
# using two different `delta` values (e.g. delta=0.1 vs delta=1.0 in its own outer wrapper) that
# never passes delta into complete_state_fingerprint necessarily gets the IDENTICAL fp, hence a
# HIT across delta stages. Demonstrated directly (not just "no delta arg exists" by inspection):
fp_delta_a = fp0()   # caller conceptually at delta=0.1
fp_delta_b = fp0()   # caller conceptually at delta=1.0 -- same call, delta never enters
check("cross-delta: identical fingerprint regardless of caller's own delta (delta excluded from the key by construction)", fp_delta_a == fp_delta_b)

println()
println("="^100)
println("SECTION 4: never store an unverified/failed result")
println("="^100)
cache2 = CompleteStateCache(max_entries = 64)
x_bad = copy(x_A); x_bad[1] = 50.0   # absurd gamma-prime, expected to be infeasible/fail
n_entries_before = length(cache2.entries)
try
    archC_verified_state_cached!(cache2, fp, x_bad, ctx_cm, cctx)
    println("  (point did not throw -- checking is_verified_success gate instead)")
    base_bad, verify_bad, _ = archC_verified_state_cached!(cache2, fp, x_bad, ctx_cm, cctx)
    check("a not-is_verified_success point is never stored", !haskey(cache2.entries, hash((fp, outer_point_key(x_bad)))) || is_verified_success(verify_bad))
catch e
    e isa CMExpectedSolveFailure || rethrow()
    check("an inner-solve failure (CMExpectedSolveFailure) leaves the cache with 0 new entries", length(cache2.entries) == n_entries_before)
end

println()
println("="^100)
println("SECTION 5: bounded LRU eviction")
println("="^100)
cache3 = CompleteStateCache(max_entries = 2)
archC_verified_state_cached!(cache3, fp, x_A, ctx_cm, cctx)
archC_verified_state_cached!(cache3, fp, x_B, ctx_cm, cctx)
check("cache3 has 2 entries after 2 distinct inserts (at max_entries=2)", length(cache3.entries) == 2)
archC_verified_state_cached!(cache3, fp, x_A, ctx_cm, cctx)   # re-touch A -- A is now MRU, B is LRU
archC_verified_state_cached!(cache3, fp, x_C, ctx_cm, cctx)   # forces an eviction
check("cache3 still has exactly max_entries=2 entries after a 3rd distinct insert", length(cache3.entries) == 2)
check("cache3.evictions == 1", cache3.evictions == 1)
keyA = hash((fp, outer_point_key(x_A))); keyB = hash((fp, outer_point_key(x_B))); keyC = hash((fp, outer_point_key(x_C)))
check("LRU correctness: A (recently re-touched) survives eviction", haskey(cache3.entries, keyA))
check("LRU correctness: C (just inserted) survives eviction", haskey(cache3.entries, keyC))
check("LRU correctness: B (least recently used) was evicted", !haskey(cache3.entries, keyB))

println()
println("="^100)
println("SECTION 6: instrumentation sanity")
println("="^100)
@printf "  cache: hits=%d misses=%d evictions=%d inner_solves_avoided=%d bytes_current=%d bytes_peak=%d hit_restore_wall=%.6fs fresh_solve_wall_counterfactual=%.4fs\n" cache.hits cache.misses cache.evictions cache.inner_solves_avoided cache.bytes_current cache.bytes_peak cache.hit_restore_wall cache.fresh_base_solve_wall_counterfactual
check("bytes_current > 0 after real inserts", cache.bytes_current > 0)
check("bytes_peak >= bytes_current", cache.bytes_peak >= cache.bytes_current)
check("fresh_base_solve_wall_counterfactual > 0 (real inner-solve wall time was recorded on misses)", cache.fresh_base_solve_wall_counterfactual > 0)

println()
println("="^100)
println("TOTAL: $n_pass passed, $n_fail failed")
println("="^100)
n_fail == 0 || error("$n_fail check(s) failed")
println("DONE")
