# 2026-08-05: cheap, solver-free diagnostic (user-suggested) -- at the raw calibration point,
# under the RAW (uniform-weight) Monte Carlo measure, the CM moment matrix's column means should
# be near zero if the moments are correctly specified and the calibration is correct (same logic
# as a GMM moment condition holding at the true parameter). Checks BOTH families directly from
# `aug.CM`, no KNITRO solve involved at all.
const D4X = @__DIR__
cd(D4X)
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl"]
    include(joinpath(D4X, f))
end
using Statistics, Printf
lp(xs...) = (println(xs...); flush(stdout))

for W in (20_000, 100_000)
    lp("="^100)
    lp("W = $W")
    lp("="^100)
    ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
    L = 10
    probs_ = collect(range(1 / L, (L - 1) / L, length = L))
    aug = build_cm_augmented_obj(ctx, CS; L = L, include_truncated_moment = true, contrasts = :anchored, probs = probs_)
    CM = aug.CM
    ncm_cdf = aug.ncm_cdf
    cdf_block = @view CM[:, 1:ncm_cdf]
    pow_block = @view CM[:, ncm_cdf+1:end]
    cdf_means = vec(mean(cdf_block, dims = 1))
    pow_means = vec(mean(pow_block, dims = 1))
    @printf("  sigma=%.3f muHat=%.4f L=%d ncm_cdf=%d\n", ctx.σ, ctx.μHat, L, ncm_cdf)
    @printf("  CDF block: mean|col mean|=%.6g  max|col mean|=%.6g\n", mean(abs.(cdf_means)), maximum(abs.(cdf_means)))
    @printf("  POW block: mean|col mean|=%.6g  max|col mean|=%.6g\n", mean(abs.(pow_means)), maximum(abs.(pow_means)))
    # for scale reference, compare against the typical magnitude of the raw (non-centered) pow feature
    Pow = frechet_power_feature(ctx.U, ctx.σ - 1, Float64(ctx.μHat))
    @printf("  raw z^(sigma-1) column means (scale reference): mean=%.6g range=[%.6g, %.6g]\n",
            mean(mean(Pow, dims=1)), minimum(mean(Pow, dims=1)), maximum(mean(Pow, dims=1)))
end
