# D20/W=<ARG> real KNITRO solve + independent verifier, AFTER the Hessian-optimization pass
# (T3/T4 dedup+threading, dense-packing removal, cross-block scratch reuse). Confirms the real
# production path is still correct at D20 scale, not just D4 -- the T3/T4 dedup overcounting bug
# this pass found+fixed was only caught because of exactly this kind of end-to-end real-context
# check (a synthetic brute-force check alone caught the bug; this confirms the fix holds at the
# real production scale too).
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
          "cm_callback_health.jl", "compressed_factual_buffer_reuse.jl", "draw_design.jl",
          "pairwise_quantile_cutoff_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_verification.jl",
          "pairwise_quantile_production.jl"]
    include(joinpath(D4X, f))
end
using LinearAlgebra, Printf

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

W = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : error("usage: julia test_pairwise_quantile_d20_verifier_after_fixes.jl <W> <L>")
PQ_L = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : error("usage: julia test_pairwise_quantile_d20_verifier_after_fixes.jl <W> <L>")

t_ctx = @elapsed begin
    global ctx = d20_real_setup_design(; W = W, δ = 1.0, find_smallest = true,
        draw_design = :pseudorandom, draw_seed = 20260719,
        destination_sample = :exclude_row, σHat = 3.0, inner_lower_limit = -10.0)
end
println("context build: ", round(t_ctx, digits=2), "s  D=", ctx.D)
flush(stdout)

x_free_calib = ctx.θ0_up[ctx.free_idx]
layout = PairwiseQuantileCutoffLayout(ctx.D, PQ_L)
aug = build_pairwise_quantile_augmented_obj(ctx, layout)
bin_state = PairwiseQuantileBinState(size(ctx.U, 1), ctx.D, PQ_L)
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
    q = [quantile_naive(Uo, r/PQ_L) for r in 1:PQ_L-1]
    raw_cutoffs[base] = log(q[1])
    for k in 2:PQ_L-1
        gap = log(q[k]) - log(q[k-1])
        raw_cutoffs[base+k-1] = gap > 0 ? log(expm1(gap)) : -5.0
    end
end

println("\n=== running REAL KNITRO inner solve (post-fix production path) ===")
flush(stdout)
t_solve = @elapsed begin
    global nStatus, x, obj, n_fg, n_hess = archPQ_base_state(x_free_calib, raw_cutoffs, ctx, ctx_cm, layout)
end
println("solve: ", round(t_solve, digits=2), "s  nStatus=", nStatus, "  n_fg=", n_fg, "  n_hess=", n_hess)
check("real KNITRO solve feasible", nStatus in (0, -100, -101, -103))
flush(stdout)

println("\n=== running independent verifier on the solved point ===")
ncore1 = obj.outer_constr_index - 1 - n_total_rows(ctx.D, PQ_L)
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
check("marginal probabilities sum to ~1 per origin", all(o -> abs(sum(verify.marginal_prob[o, :]) - 1.0) < 1e-9, 1:ctx.D))

println()
println(ALL_PASS[] ? "ALL POST-FIX D20 CHECKS PASSED" : "SOME CHECKS FAILED")
