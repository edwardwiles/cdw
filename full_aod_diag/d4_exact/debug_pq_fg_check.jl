const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "operator_psi_bundle.jl",
          "cm_callback_health.jl", "compressed_factual_buffer_reuse.jl",
          "pairwise_quantile_cutoff_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_verification.jl",
          "pairwise_quantile_production.jl"]
    include(joinpath(D4X, f))
end
using Random, LinearAlgebra

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
layout = PairwiseQuantileCutoffLayout(ctx.D)
aug = build_pairwise_quantile_augmented_obj(ctx, layout)
bin_state = PairwiseQuantileBinState(size(ctx.U,1), ctx.D)

function quantile_naive(v::AbstractVector{Float64}, p::Float64)
    s = sort(v); n = length(s)
    return s[clamp(round(Int, p*n), 1, n)]
end
raw_cutoffs = zeros(n_raw(layout))
for o in 1:ctx.D
    base = raw_index(layout, o, 1)
    Uo = @view ctx.U[:, o]
    q = [quantile_naive(Uo, r/5) for r in 1:4]
    raw_cutoffs[base] = log(q[1])
    for k in 2:4
        gap = log(q[k]) - log(q[k-1])
        raw_cutoffs[base+k-1] = gap > 0 ? log(expm1(gap)) : -5.0
    end
end

obj = aug.obj_pq
θ_econ0 = CS.reconstruct_full(x_free_calib, ctx.m)
prime_operator!(obj, θ_econ0, ctx, aug.core_cf_ref)

ncore1 = obj.outer_constr_index - 1 - n_total_rows(ctx.D)
println("ncore1 = ", ncore1, "  n_total_rows = ", n_total_rows(ctx.D), "  outer_constr_index = ", obj.outer_constr_index)

st = PairwiseQuantileOperatorState(obj, ncore1, aug.op, ctx.U, bin_state, aug.core_cf_ref)
reset_for_solve!(st, raw_cutoffs, layout)

Random.seed!(3)
n = obj.outer_constr_index
x0 = 0.01 .* randn(n)   # a random-but-small dual point (not necessarily optimal)

g = zeros(n)
f0 = st(x0, g)
println("f0 = ", f0, "  norm(g) = ", norm(g))

# finite-difference check of a handful of coordinates (full n=130 would be slow-ish but let's do all)
h = 1e-6
maxerr = 0.0
worst = 0
for i in 1:n
    xp = copy(x0); xp[i] += h
    xm = copy(x0); xm[i] -= h
    fp = st(xp, Float64[])
    fm = st(xm, Float64[])
    gfd = (fp - fm) / (2h)
    err = abs(gfd - g[i])
    if err > maxerr
        global maxerr = err
        global worst = i
    end
end
println("max |g_analytic - g_fd| over all $n coords = ", maxerr, "  at i=", worst, "  (g[i]=", g[worst], ")")
println(maxerr < 1e-4 ? "FG GRADIENT CHECK: PASS" : "FG GRADIENT CHECK: FAIL")
