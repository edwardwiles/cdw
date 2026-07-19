# ============================================================================
# Continuation 10, Phase 7 (QMC investigation), step 0: validate
# cc_algo/rhalton.jl's scrambled Halton generator BEFORE trusting it for
# anything downstream. This generator (ported from Art Owen's R code) is
# currently only wired into cc_algo/boot.jl's bootstrap routine -- it has
# NEVER been used to drive F* draw generation, so its correctness for THIS
# purpose is not established by prior use. Standalone, lightweight (no KNITRO,
# no production context) -- just Random.jl + the file itself.
#
# Checks: shape/domain, per-dimension first/second moments vs Uniform(0,1),
# reproducibility (same seed -> identical points), independence across
# replicate seeds, cross-dimension correlation (the real target use is
# D=20 dimensions -- the UoModel=1 branch draws U[:, 1:D], not U[:, 1:D^2],
# confirmed by inspecting winners.jl/drawU.jl -- so we validate at d=20,
# small primes only, not the D^2=400 pathological-large-prime regime we
# originally worried about), a simple discrepancy proxy vs matched-size
# pseudorandom, and generation wall-time at production scale (W=80000, d=20).
# ============================================================================
using Random, Statistics, Printf
const D4X_ROOT2 = dirname(dirname(@__DIR__))
include(joinpath(D4X_ROOT2, "cc_algo", "rhalton.jl"))

println("="^80); println("Phase 7 step 0: rhalton.jl validation"); println("="^80)

# ---- 1. Shape / domain ----
n, d = 5000, 20
x = rhalton(n, d; singleseed = 12345)
@assert size(x) == (n, d)
@assert all(0.0 .<= x .< 1.0) "rhalton produced values outside [0,1)"
println("[1] shape=", size(x), " min=", minimum(x), " max=", maximum(x), " -- PASS (domain [0,1))")

# ---- 2. Per-dimension moments vs Uniform(0,1): mean=0.5, var=1/12=0.08333 ----
means = vec(mean(x, dims = 1)); vars = vec(var(x, dims = 1))
println("[2] per-dim mean: range=[", round(minimum(means), digits=4), ",", round(maximum(means), digits=4),
        "] (target 0.5), per-dim var: range=[", round(minimum(vars), digits=4), ",", round(maximum(vars), digits=4),
        "] (target 0.0833)")
mean_ok = all(abs.(means .- 0.5) .< 0.05)
var_ok = all(abs.(vars .- 1/12) .< 0.02)
println("    mean_ok=", mean_ok, " var_ok=", var_ok, mean_ok && var_ok ? " -- PASS" : " -- FAIL")

# ---- 3. Reproducibility: same seed -> identical points ----
x2 = rhalton(n, d; singleseed = 12345)
same_seed_identical = x == x2
println("[3] same-seed reproducibility: identical=", same_seed_identical, same_seed_identical ? " -- PASS" : " -- FAIL")

# ---- 4. Independent scrambles differ, but both remain well-equidistributed ----
x3 = rhalton(n, d; singleseed = 67890)
diff_seed_differs = x != x3
max_pointwise_diff = maximum(abs.(x .- x3))
println("[4] different-seed differs=", diff_seed_differs, " max_pointwise_diff=", round(max_pointwise_diff, digits=4),
        diff_seed_differs ? " -- PASS" : " -- FAIL")

# ---- 5. Cross-dimension linear correlation (should be ~0, no gross degeneracy) ----
pairs_to_check = [(1,2), (1,10), (5,15), (19,20), (1,20)]
println("[5] cross-dimension correlations (target ~0):")
worst_corr = 0.0
for (i,j) in pairs_to_check
    c = cor(x[:,i], x[:,j])
    global worst_corr = max(worst_corr, abs(c))
    @printf("    dims (%d,%d): corr=%.4f\n", i, j, c)
end
corr_ok = worst_corr < 0.1
println("    worst |corr|=", round(worst_corr, digits=4), corr_ok ? " -- PASS (<0.1)" : " -- MARGINAL/FAIL")

# ---- 6. Discrepancy proxy: coarse-bin chi-square uniformity, Halton vs pseudorandom, 1D ----
function binvar(u::AbstractVector, nbins::Int)
    counts = zeros(Int, nbins)
    for v in u
        b = clamp(floor(Int, v * nbins) + 1, 1, nbins)
        counts[b] += 1
    end
    return var(counts)
end
nbins = 50
Random.seed!(999)
u_rand = rand(n)
bv_halton = binvar(x[:,1], nbins)
bv_rand = binvar(u_rand, nbins)
expected_var_rand = n / nbins * (1 - 1/nbins)
println("[6] 1D bin-count variance (n=", n, ", bins=", nbins, "): halton=", round(bv_halton, digits=2),
        " pseudorandom=", round(bv_rand, digits=2), " (iid-theory=", round(expected_var_rand, digits=2), ")")
discrepancy_improves = bv_halton < bv_rand
println("    halton < pseudorandom bin-variance: ", discrepancy_improves, discrepancy_improves ? " -- PASS (expected QMC behavior)" : " -- did not show improvement here")

# ---- 7. x=0 pathology check (plain, unscrambled Halton famously starts at exactly 0) ----
x_idx0 = rhalton(1, d; n0 = 0, singleseed = 42)
println("[7] first-index (n0=0) point (scrambled, should generally NOT be exactly 0): ", x_idx0[1, 1:min(5,d)])
println("    all exactly zero? ", all(x_idx0 .== 0.0), " (expect false -- scrambling should avoid the classic index-0-at-origin degeneracy)")

# ---- 8. Production-scale timing: W=80000, d=20 ----
GC.gc()
t0 = time()
x_full = rhalton(80000, 20; singleseed = 111)
t_full = time() - t0
println("[8] production-scale generation: W=80000, d=20 -> wall=", round(t_full, digits = 3), "s, size=", size(x_full))
@assert size(x_full) == (80000, 20)
@assert all(0.0 .<= x_full .< 1.0)

println()
println("="^80)
println("SUMMARY: domain_ok=", all(0.0 .<= x .< 1.0), " mean_ok=", mean_ok, " var_ok=", var_ok,
        " reproducible=", same_seed_identical, " differs_across_seeds=", diff_seed_differs,
        " corr_ok=", corr_ok, " discrepancy_improves=", discrepancy_improves,
        " prod_scale_wall_s=", round(t_full, digits=3))
println("="^80)
