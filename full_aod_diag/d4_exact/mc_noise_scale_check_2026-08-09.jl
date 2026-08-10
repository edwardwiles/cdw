# Is the raw moment discrepancy at the calibration point LARGE, or is it just Monte Carlo noise?
# (2026-08-09, user question: "there should be literally nothing different about the origins in
# terms of the raw draws, right? So how can origin 6 be systematically different?")
#
# The prior raw-MC script reported discrepancies (max ~1.2e-2 at mean level k=3, ~3.9e-2 at pair
# (3,3)) and flagged origin 6 as worst everywhere -- but NEVER compared them against the MC standard
# error they should have. That was the missing step. This script supplies it, three ways:
#
# (A) THEORETICAL MC standard error, closed form. z_o(w) = U_o(w)^{-mu*k}, U ~ Exp(1) iid, so
#       E[z^k]   = Gamma(1 - mu*k)                       (the nu0 target itself)
#       E[z^2k]  = Gamma(1 - 2*mu*k)                     (finite iff 1-2*mu*k > 0)
#       Var(z^k) = Gamma(1-2mu k) - Gamma(1-mu k)^2,  SE = sqrt(Var/W)
#     and for a pair feature z_o^k1 * z_p^k2 (o != p, INDEPENDENT columns):
#       E    = Gamma(1-mu k1) * Gamma(1-mu k2)
#       E[.^2] = Gamma(1-2mu k1) * Gamma(1-2mu k2)
#     If observed |disc| ~ 1-3 SE, the moments are exactly as close to target as iid draws allow,
#     and NOTHING is wrong with the calibration or the targets -- the nonzero Delta* is finite-W
#     noise, not misspecification.
# (B) The z-score of EVERY origin/pair, and where origin 6 ranks. Under iid draws the max |z| over
#     20 origins should be ~2-2.6 typically (max of 20 standard normals), NOT ~0. "Worst of 20" is
#     not evidence of anything by itself.
# (C) SEED SENSITIVITY -- the decisive test for the user's actual question. If origin 6 is worst
#     because of a genuine code/data asymmetry, it stays worst under a DIFFERENT draw seed. If it's
#     luck, a different seed makes some other origin worst. Nothing in the model treats origins
#     asymmetrically at the draw level, so the latter is expected; this test settles it empirically.
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "country_resolve.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl"]
    include(joinpath(D4X, f))
end
using Statistics, LinearAlgebra, Printf
using SpecialFunctions: gamma

const W = 100_000
const K_mean = 3
const K_pair = 3
const SEEDS = [20260719, 20260720, 20260721]   # production seed + two alternates

# ---- (A) closed-form MC standard errors ----
function mean_se(μ, k, W)
    1 - 2μ*k > 0 || return NaN   # infinite variance otherwise
    v = gamma(1 - 2μ*k) - gamma(1 - μ*k)^2
    return sqrt(v / W)
end
function pair_se(μ, k1, k2, W)
    (1 - 2μ*k1 > 0 && 1 - 2μ*k2 > 0) || return NaN
    m = gamma(1 - μ*k1) * gamma(1 - μ*k2)
    e2 = gamma(1 - 2μ*k1) * gamma(1 - 2μ*k2)
    return sqrt((e2 - m^2) / W)
end

results = Dict{Int,Any}()

for (si, seed) in enumerate(SEEDS)
    println("\n############ SEED $seed ############")
    flush(stdout)
    ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
        draw_design = :pseudorandom, draw_seed = seed, destination_sample = :exclude_row,
        exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
        σHat = 3.0, inner_lower_limit = -10.0)
    D = ctx.D
    μ = ctx.μHat
    if si == 1
        println("D=$D  muHat=$μ  W=$W")
        println("Finite-variance check (need 1-2*mu*k > 0): " *
                join(["k=$k: $(round(1-2μ*k, digits=4))" for k in 1:K_mean], "  "))
    end
    flush(stdout)

    layout = OriginByPowerLayout(D, K_mean, K_pair)
    cross_layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
    νfull0 = vcat([fill(gamma(1 - μ * k), D) for k in 1:K_mean]...)
    Zraw_all, _ = build_raw_mean_pair_matrix_levels(ctx.U, K_mean, 0; μ = μ)
    Zpairraw_all = build_raw_cross_pair_matrix_levels(Zraw_all, K_pair)
    pairs = packed_pair_index(D)

    println("\n-- MEAN block: observed disc vs theoretical MC standard error --")
    worst_origins = Int[]
    for k in 1:K_mean
        tgt = mean_targets(layout, νfull0, k, D)
        raw = vec(mean(Zraw_all[k], dims = 1))
        disc = raw .- tgt
        se = mean_se(μ, k, W)
        z = disc ./ se
        jw = argmax(abs.(z))
        push!(worst_origins, jw)
        @printf("  k=%d: SE=%.4e | max|disc|=%.4e  max|z|=%.2f (origin %d) | mean|z|=%.2f | z(origin6)=%+.2f\n",
                k, se, maximum(abs.(disc)), maximum(abs.(z)), jw, mean(abs.(z)), z[6])
    end
    flush(stdout)

    println("\n-- CROSS-PAIR block: observed disc vs theoretical MC standard error --")
    levels = cross_pair_level_index(K_pair)
    worst_pairs = Tuple{Int,Int}[]
    for (klin, (k1, k2)) in enumerate(levels)
        tgt = pair_targets(cross_layout, νfull0, klin, D)
        raw = vec(mean(Zpairraw_all[klin], dims = 1))
        disc = raw .- tgt
        se = pair_se(μ, k1, k2, W)
        z = disc ./ se
        jw = argmax(abs.(z))
        push!(worst_pairs, pairs[jw])
        @printf("  (k1=%d,k2=%d): SE=%.4e | max|disc|=%.4e  max|z|=%.2f (pair %s) | mean|z|=%.2f\n",
                k1, k2, se, maximum(abs.(disc)), maximum(abs.(z)), string(pairs[jw]), mean(abs.(z)))
    end
    flush(stdout)

    # How often does origin 6 appear in the worst pair, vs any other origin?
    origin_counts = zeros(Int, D)
    for (o, p) in worst_pairs
        origin_counts[o] += 1; origin_counts[p] += 1
    end
    top = sortperm(origin_counts, rev = true)[1:5]
    println("\n-- which origins dominate the worst CROSS-PAIR cells (count over the 9 blocks) --")
    println("  ", join(["origin $o: $(origin_counts[o])" for o in top], "  |  "))
    println("  worst MEAN-block origin at k=1,2,3: ", worst_origins)
    results[seed] = (worst_origins = copy(worst_origins), worst_pairs = copy(worst_pairs), counts = copy(origin_counts))
    flush(stdout)
end

println("\n\n################ SEED-SENSITIVITY VERDICT ################")
for seed in SEEDS
    r = results[seed]
    println("  seed $seed: worst mean-block origin at k=1,2,3 = $(r.worst_origins);  top cross-pair origin = $(argmax(r.counts))")
end
println("\nIf the worst origin CHANGES across seeds, origin 6 was luck-of-the-draw (expected: nothing")
println("in the model distinguishes origins at the draw level). If it stays 6 under every seed,")
println("there IS a genuine asymmetry and it needs a root-cause hunt.")
