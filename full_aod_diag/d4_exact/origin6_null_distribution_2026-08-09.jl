# Is origin 6 genuinely special, or is it an ordinary draw of the "worst of 20"? (2026-08-09)
#
# CONTEXT / correction of a bad prior test: an earlier version of this check tried to vary the draws
# via `d20_real_setup_design(draw_seed=...)` under `draw_design=:pseudorandom`. That is INERT --
# under :pseudorandom, U is drawn INSIDE `d20_real_setup` by `master_prepare_cc`'s own internal
# `Random.seed!(AD_PARAMS.seedU)` (seedU=888, a fixed constant), which overrides the outer
# `Random.seed!(draw_seed)` that d20_real_setup_design sets (draw_design.jl lines 170-185 document
# this deliberately). Three "different" seeds produced BIT-IDENTICAL discrepancies, which is what
# gave that away. Only :sobol_randomized/:halton_scrambled/:precomputed actually consume draw_seed.
#
# So this script tests the question directly and cheaply, WITHOUT any ctx rebuild or KNITRO solve:
# the mean-block discrepancy is a pure function of the raw draw matrix U and mu, via
# frechet_power_feature (z = U^{-mu*k}). We therefore:
#
#  (1) Take the PRODUCTION U (from one real ctx build) and record origin 6's z-scores + the identity
#      of the worst origin at each level.
#  (2) Check whether the production U's column 6 is anomalous AS A DRAW: its mean/quantiles vs the
#      other 19 columns, and (critically) its MINIMUM, since z = U^{-mu*k} blows up as U -> 0, so a
#      single unusually small U value dominates a whole column's high-power mean.
#  (3) Simulate NREP independent fresh Exp(1) W x D matrices (genuinely different RNG draws) and,
#      for each, record which origin is worst and the max|z|. This gives the NULL DISTRIBUTION for
#      "max |z| over 20 origins" and "is the worst origin the same at k=1,2,3".
#
# The key statistical subtlety (i.e. why the earlier "origin 6 is worst at EVERY level" observation
# was never evidence of anything): within ONE draw matrix, z^1, z^2, z^3 are all monotone functions
# of the SAME column, so whichever column has the heaviest low-U tail is worst at all three levels
# SIMULTANEOUSLY, by construction. Consistency across k within a fixed draw set is expected under
# pure iid noise, not evidence of a code asymmetry. (3) quantifies exactly that.
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "country_resolve.jl"]
    include(joinpath(D4X, f))
end
using Statistics, LinearAlgebra, Printf, Random
using SpecialFunctions: gamma

const W = 100_000
const K_mean = 3
const NREP = 200

mean_se(μ, k, W) = sqrt((gamma(1 - 2μ*k) - gamma(1 - μ*k)^2) / W)

println("Building ONE production D20 context (W=$W) to get the real U...")
flush(stdout)
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
D = ctx.D
μ = ctx.μHat
U = ctx.U
println("ctx built. D=$D muHat=$μ size(U)=$(size(U))")
flush(stdout)

# ---------- (1) production U: z-scores per origin ----------
println("\n================ (1) PRODUCTION U: per-origin z-scores ================")
for k in 1:K_mean
    Zk = frechet_power_feature(U, k, μ)
    tgt = gamma(1 - μ*k)
    se = mean_se(μ, k, W)
    z = (vec(mean(Zk, dims = 1)) .- tgt) ./ se
    jw = argmax(abs.(z))
    @printf("  k=%d: worst origin=%d (z=%+.2f) | z(origin6)=%+.2f | mean|z|=%.2f (theory E|Z|=0.798)\n",
            k, jw, z[jw], z[6], mean(abs.(z)))
    if k == 1
        println("      all z: ", join([@sprintf("%d:%+.2f", o, z[o]) for o in 1:D], " "))
    end
end
flush(stdout)

# ---------- (2) is production U's column 6 anomalous AS A DRAW? ----------
println("\n================ (2) production U column stats (is col 6 an odd Exp(1) sample?) ================")
colmeans = vec(mean(U, dims = 1))
colmins  = vec(minimum(U, dims = 1))
@printf("  Exp(1) theory: mean=1.0, E[min of %d]=%.3e\n", W, 1/W)
@printf("  col means: min=%.5f (origin %d)  max=%.5f (origin %d)  | origin6=%.5f  (SE=%.5f)\n",
        minimum(colmeans), argmin(colmeans), maximum(colmeans), argmax(colmeans), colmeans[6], 1/sqrt(W))
@printf("  col MINIMA (drives high-power means, z=U^-mu*k blows up as U->0):\n")
ord = sortperm(colmins)
for r in 1:5
    o = ord[r]
    @printf("      rank %d smallest-min: origin %2d  min(U)=%.4e   -> U^(-mu*3)=%.2f\n", r, o, colmins[o], colmins[o]^(-μ*3))
end
@printf("      origin 6: min(U)=%.4e (rank %d of %d)  -> U^(-mu*3)=%.2f\n",
        colmins[6], findfirst(==(6), ord), D, colmins[6]^(-μ*3))
flush(stdout)

# ---------- (3) null distribution over genuinely-fresh draws ----------
println("\n================ (3) NULL DISTRIBUTION: $NREP fresh iid Exp(1) draw matrices ================")
println("(genuinely different RNG draws -- NOT via the inert :pseudorandom draw_seed path)")
flush(stdout)
worst_counts = zeros(Int, D)
maxz_all = Float64[]
same_worst_all3 = 0
for rep in 1:NREP
    rng = MersenneTwister(1_000_000 + rep)
    Usim = -log.(rand(rng, W, D))          # Exp(1) via inverse CDF
    worst_k = Int[]
    maxz_rep = 0.0
    for k in 1:K_mean
        Zk = Usim .^ (-μ*k)
        tgt = gamma(1 - μ*k); se = mean_se(μ, k, W)
        z = (vec(mean(Zk, dims = 1)) .- tgt) ./ se
        jw = argmax(abs.(z))
        push!(worst_k, jw)
        maxz_rep = max(maxz_rep, abs(z[jw]))
    end
    worst_counts[worst_k[1]] += 1
    push!(maxz_all, maxz_rep)
    # `global` required: rebinding assignment to an outer-scope variable from INSIDE a top-level
    # `for` loop, in a script run via `julia file.jl`/include (see this repo's own memory
    # julia-toplevel-catch-scoping-gotcha). worst_counts/maxz_all above need no `global` -- they are
    # MUTATED (setindex!/push!), not rebound.
    global same_worst_all3
    all(==(worst_k[1]), worst_k) && (same_worst_all3 += 1)
end
@printf("  max|z| over 20 origins, across %d fresh draw sets: mean=%.2f  median=%.2f  p05=%.2f  p95=%.2f\n",
        NREP, mean(maxz_all), median(maxz_all), quantile(maxz_all, 0.05), quantile(maxz_all, 0.95))
@printf("  PRODUCTION max|z| was ~2.5 -> percentile in this null: %.1f%%\n",
        100 * mean(maxz_all .<= 2.52))
@printf("  fraction of fresh draw sets where the SAME origin is worst at k=1,2,3: %.1f%% (%d/%d)\n",
        100 * same_worst_all3 / NREP, same_worst_all3, NREP)
@printf("  how often each origin is 'worst at k=1' (should be ~%.1f%% each if exchangeable):\n", 100/D)
println("      ", join([@sprintf("%d:%d", o, worst_counts[o]) for o in 1:D], " "))
@printf("  origin 6 was worst in %d/%d = %.1f%% of fresh draw sets\n", worst_counts[6], NREP, 100*worst_counts[6]/NREP)

println("\n================ VERDICT ================")
println("If (a) production max|z| sits in the bulk of the null, (b) 'same worst origin at all 3 levels'")
println("is COMMON in fresh draws, and (c) origin 6 wins ~1/20 of fresh draw sets, then origin 6 is")
println("simply this ONE draw matrix's unlucky column -- no code/data asymmetry, nothing to fix.")
