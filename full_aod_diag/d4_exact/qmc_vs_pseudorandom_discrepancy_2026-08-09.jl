# Does Sobol QMC shrink the raw moment discrepancies (and hence Delta*)? (2026-08-09, user question:
# "is this Delta*=0.07 from pseudorandom or Sobol QMC? This is the kind of thing that I'd expect
# Sobol QMC to be well suited for.")
#
# ANSWER TO THE FACTUAL PART: every Delta* reported this session used draw_design=:pseudorandom
# (set explicitly in every script). So 0.0667 is a pseudorandom number.
#
# WHY QMC SHOULD HELP, and what this script measures: Delta* at the calibration point is essentially
# a quadratic form in the moment discrepancy vector g (obs mean - target). Under plain Monte Carlo,
# |g| ~ W^{-1/2}, so Delta* ~ |g|^2 ~ 1/W -- which is exactly the scaling already observed
# (base: 0.00949 at W=100k -> 0.00441 at W=200k, ratio 2.15 for a 2x W). Under scrambled Sobol,
# smooth-enough integrands give |g| ~ W^{-1} (up to log^d factors), hence Delta* ~ 1/W^2. If that
# holds here it is a far bigger win than raising W.
#
# CAVEAT worth measuring rather than assuming: these integrands are z^k = U^{-mu*k} with U ~ Exp(1),
# which are UNBOUNDED as U -> 0 (integrable, since 1-2*mu*k > 0 for k<=3 at mu=0.1335, but with a
# genuine singularity at the corner of the unit cube after the inverse-CDF transform). QMC's
# advantage degrades for unbounded/singular integrands, so the realized rate may fall between
# W^{-1/2} and W^{-1}. That is exactly what this script measures empirically.
#
# This part needs NO KNITRO solve -- the discrepancies are pure functions of the draws -- so it is
# fast and settles the mechanism before spending wall-clock on a real Delta* solve.
#
# Usage: julia --project=. -t 4 full_aod_diag/d4_exact/qmc_vs_pseudorandom_discrepancy_2026-08-09.jl
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
using Statistics, LinearAlgebra, Printf, Random
using SpecialFunctions: gamma

const K_mean = 3
const K_pair = 3
const D = D20_REAL
const MU = 0.1335220518360467          # ctx.muHat at the production config (sigma=3.0), fixed here
                                        # so no ctx build is needed for the pure-draw statistics.
const WS = [25_000, 50_000, 100_000, 200_000, 400_000]

mk = [gamma(1 - MU*k) for k in 1:K_mean]
mean_se(k, W) = sqrt((gamma(1 - 2MU*k) - gamma(1 - MU*k)^2) / W)
pair_se(k1, k2, W) = sqrt((gamma(1-2MU*k1)*gamma(1-2MU*k2) - (gamma(1-MU*k1)*gamma(1-MU*k2))^2) / W)

"Summed squared standardized discrepancy over mean+cross-pair blocks -- a Delta*-like scalar that
 needs no solve. Reported both raw (sum g^2) and standardized (sum (g/SE)^2 = chi2-like)."
function discrepancy_stats(U::AbstractMatrix{Float64})
    W = size(U, 1)
    Zraw = [U .^ (-MU*k) for k in 1:K_mean]
    levels = cross_pair_level_index(K_pair)
    pairs = packed_pair_index(D)
    sum_g2_mean = 0.0; sum_z2_mean = 0.0; max_abs_mean = 0.0
    for k in 1:K_mean
        g = vec(mean(Zraw[k], dims = 1)) .- mk[k]
        sum_g2_mean += sum(abs2, g)
        sum_z2_mean += sum(abs2, g ./ mean_se(k, W))
        max_abs_mean = max(max_abs_mean, maximum(abs.(g)))
    end
    sum_g2_pair = 0.0; sum_z2_pair = 0.0; max_abs_pair = 0.0
    for (klin, (k1, k2)) in enumerate(levels)
        tgt = mk[k1] * mk[k2]
        gg = Vector{Float64}(undef, length(pairs))
        @inbounds for (j, (o, p)) in enumerate(pairs)
            gg[j] = mean(@views Zraw[k1][:, o] .* Zraw[k2][:, p]) - tgt
        end
        sum_g2_pair += sum(abs2, gg)
        sum_z2_pair += sum(abs2, gg ./ pair_se(k1, k2, W))
        max_abs_pair = max(max_abs_pair, maximum(abs.(gg)))
    end
    return (rms_mean = sqrt(sum_g2_mean / (K_mean*D)), rms_pair = sqrt(sum_g2_pair / (length(levels)*length(pairs))),
            chi2_mean = sum_z2_mean, chi2_pair = sum_z2_pair,
            max_mean = max_abs_mean, max_pair = max_abs_pair)
end

println("D=$D  mu=$MU  K_mean=$K_mean K_pair=$K_pair (cross grid $(K_pair^2) blocks)")
println("NOTE: every Delta* reported this session used :pseudorandom. This compares draw designs.\n")
flush(stdout)

@printf("%-18s %9s | %11s %11s | %11s %11s | %10s %10s\n",
        "design", "W", "rms_g mean", "rms_g pair", "max|g| mean", "max|g| pair", "chi2 mean", "chi2 pair")
println("-"^115)
results = Dict{Tuple{Symbol,Int},Any}()
for W in WS
    # --- plain pseudorandom Exp(1), matching production's own inverse-CDF construction ---
    rngp = MersenneTwister(888)
    Up = -log.(rand(rngp, W, D))
    sp = discrepancy_stats(Up)
    results[(:pseudorandom, W)] = sp
    @printf("%-18s %9d | %11.3e %11.3e | %11.3e %11.3e | %10.1f %10.1f\n",
            "pseudorandom", W, sp.rms_mean, sp.rms_pair, sp.max_mean, sp.max_pair, sp.chi2_mean, sp.chi2_pair)
    flush(stdout)

    # --- scrambled Sobol via the repo's OWN generator (generate_randoms!/resolve_draw_design,
    #     draw_design.jl) so this matches what draw_design=:sobol_randomized would actually feed in ---
    design = resolve_draw_design(:sobol_randomized, 20260719)   # signature is (design::Symbol, seed::Int); W/D come from the preallocated buffer
    Us = Matrix{Float64}(undef, W, D)
    generate_randoms!(Us, design)          # fills with Exp(1)-transformed draws, exactly as the :sobol_randomized path in d20_real_setup_design does
    ss = discrepancy_stats(Us)
    results[(:sobol_randomized, W)] = ss
    @printf("%-18s %9d | %11.3e %11.3e | %11.3e %11.3e | %10.1f %10.1f\n",
            "sobol_randomized", W, ss.rms_mean, ss.rms_pair, ss.max_mean, ss.max_pair, ss.chi2_mean, ss.chi2_pair)
    println()
    flush(stdout)
end

println("\n================ CONVERGENCE RATES (fit rms_g ~ C * W^(-r)) ================")
for design in (:pseudorandom, :sobol_randomized)
    for fld in (:rms_mean, :rms_pair)
        xs = [log(W) for W in WS]
        ys = [log(getfield(results[(design, W)], fld)) for W in WS]
        n = length(xs); mx = mean(xs); my = mean(ys)
        r = -sum((xs .- mx) .* (ys .- my)) / sum(abs2, xs .- mx)
        @printf("  %-18s %-9s: rate r = %.3f   (MC theory 0.5, QMC theory ~1.0)\n", design, fld, r)
    end
end

println("\n================ SOBOL / PSEUDORANDOM RATIO (lower = QMC wins) ================")
for W in WS
    p = results[(:pseudorandom, W)]; s = results[(:sobol_randomized, W)]
    @printf("  W=%7d: rms_g mean %.3fx | rms_g pair %.3fx | chi2 mean %.3fx | chi2 pair %.3fx\n",
            W, s.rms_mean/p.rms_mean, s.rms_pair/p.rms_pair, s.chi2_mean/p.chi2_mean, s.chi2_pair/p.chi2_pair)
end
println("\nIf Delta* ~ quadratic in g, the expected Delta* reduction factor is roughly the SQUARE of")
println("the rms_g ratio -- e.g. a 10x smaller g implies ~100x smaller Delta*.")
