# Real D4 KNITRO solve + independent verifier check for the pairwise-quantile-independence restriction.
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
using LinearAlgebra

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
layout = PairwiseQuantileCutoffLayout(ctx.D)
aug = build_pairwise_quantile_augmented_obj(ctx, layout)
bin_state = PairwiseQuantileBinState(size(ctx.U, 1), ctx.D)
hess_ctx = PairwiseQuantileCoreHessCtx(aug.ncore_econ, aug.op, bin_state, aug.core_cf_ref)
ctx_cm = merge(ctx, (obj = aug.obj_pq, pq_op = aug.op, pq_bin_state = bin_state,
                      pq_core_cf_ref = aug.core_cf_ref, pq_hess_ctx = hess_ctx))

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

println("=== running REAL KNITRO inner solve ===")
nStatus, x, obj, n_fg, n_hess = archPQ_base_state(x_free_calib, raw_cutoffs, ctx, ctx_cm, layout)
println("nStatus = ", nStatus, "  n_fg = ", n_fg, "  n_hess = ", n_hess)
check("real KNITRO solve feasible", nStatus in (0, -100, -101, -103))

println("\n=== running independent verifier on the solved point ===")
ncore1 = obj.outer_constr_index - 1 - n_total_rows(ctx.D)
zeta = x[1]
lambda = x[2:end]
cf = aug.core_cf_ref[]
econ_ws = economic_operator_workspace(cf)

verify = verify_inner_solution_operator_pairwisequantile!(zeta, lambda, cf, aug.op, bin_state, aug.op.W,
    economic_forward!, economic_transpose!, econ_ws, obj.Psi!, obj.dPsi!, ncore1)

println("kkt_resid = ", verify.kkt_resid)
println("kkt_resid_E = ", verify.kkt_resid_E, "  kkt_resid_marginalbin = ", verify.kkt_resid_marginalbin,
        "  kkt_resid_pairindep = ", verify.kkt_resid_pairindep)
check("verifier KKT residual small (< 1e-4)", verify.kkt_resid < 1e-4)

println("\nmarginal_prob sample (origin 1, all 5 bins): ", round.(verify.marginal_prob[1, :], digits=4))
println("max_marginal_cumulative_residual = ", verify.max_marginal_cumulative_residual)
println("max_cumulative_residual (joint) = ", verify.max_cumulative_residual)
check("marginal probabilities sum to ~1 per origin", all(o -> abs(sum(verify.marginal_prob[o, :]) - 1.0) < 1e-9, 1:ctx.D))

println()
println(ALL_PASS[] ? "ALL VERIFIER CHECKS PASSED" : "SOME CHECKS FAILED")
