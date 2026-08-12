# D20/W=<ARG> controlled A/B: dense-Hessian inner solve (current production path, hessopt=exact)
# vs the new HVP inner solve (hessopt=5/product, algorithm=cg) for the pairwise-quantile-
# independence restriction, at REAL scale. Builds the context ONCE (expensive: ~112s at W=100k),
# runs both solves against it sequentially (each call rebuilds its own fresh
# PairwiseQuantileOperatorState/primes cf/resets bins, so no cross-contamination), and passes BOTH
# converged points through the SAME independent verifier. Reports total wall-clock, FG calls,
# Hessian(-vector) calls, nStatus, and verifier KKT residual for both -- the handover doc's own
# explicit correctness+performance bar, not per-callback cost alone.
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
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_verification.jl",
          "pairwise_quantile_production.jl", "pairwise_quantile_hvp.jl"]
    include(joinpath(D4X, f))
end
using LinearAlgebra, Printf

W = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : error("usage: julia profile_pairwise_quantile_d20_hvp_ab.jl <W> <L>")
PQ_L = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : error("usage: julia profile_pairwise_quantile_d20_hvp_ab.jl <W> <L>")
println("=== D20 dense-vs-HVP A/B, W=$W, L=$PQ_L ===")
flush(stdout)

t_ctx = @elapsed begin
    global ctx = d20_real_setup_design(; W = W, δ = 1.0, find_smallest = true,
        draw_design = :pseudorandom, draw_seed = 20260719,
        destination_sample = :exclude_row, σHat = 3.0, inner_lower_limit = -10.0)
end
println("context build: ", round(t_ctx, digits=2), "s  D=", ctx.D, "  size(U)=", size(ctx.U))
flush(stdout)

x_free_calib = ctx.θ0_up[ctx.free_idx]
layout = PairwiseQuantileMassLayout(ctx.D, PQ_L)

t_aug = @elapsed begin
    global Zfeat = pairwise_quantile_frechet_features(ctx.U, ctx.μHat)   # restriction is on the Frechet z
    global aug = build_pairwise_quantile_augmented_obj(ctx, layout, Zfeat,
        pairwise_quantile_fixed_cutoffs(Zfeat, PQ_L; cutoff_source = CUTOFF_SOURCE, mu_frechet = ctx.μHat))
end
println("augmented obj + PairwiseQuantileOperator build (incl. presort): ", round(t_aug, digits=2), "s")
flush(stdout)

mass_state = PairwiseQuantileMassState(ctx.D, PQ_L)
hess_ctx = PairwiseQuantileCoreHessCtx(aug.ncore_econ, aug.op, mass_state, aug.core_cf_ref)
ctx_cm = merge(ctx, (obj = aug.obj_pq, pq_op = aug.op, pq_mass_state = mass_state,
                      pq_core_cf_ref = aug.core_cf_ref, pq_hess_ctx = hess_ctx))

# VERSION B: the outer restriction coordinates are BIN MASSES on the simplex; the canonical
# starting point is mu = 1/L (uniform_mass_raw) -- exactly version A's fixed target. The
# cutoffs are no longer outer coordinates at all: they are FIXED at context-build time from
# CUTOFF_SOURCE (see pairwise_quantile_fixed_cutoffs).
raw_masses = uniform_mass_raw(layout)

DEFAULT_OPT = ctx_cm.obj.inner_loop_opt
HVP_OPT = joinpath(D4X, "ek_inner_hvp.opt")

function run_verifier(x, obj)
    ncore1 = obj.outer_constr_index - 1 - n_total_rows(ctx.D, PQ_L)
    zeta = x[1]; lambda = x[2:end]
    cf = aug.core_cf_ref[]
    econ_ws = economic_operator_workspace(cf)
    return verify_inner_solution_operator_pairwisequantile!(zeta, lambda, cf, aug.op, mass_state, aug.op.W,
        economic_forward!, economic_transpose!, econ_ws, obj.Psi!, obj.dPsi!, ncore1)
end

println("\n=== [A] dense-Hessian inner solve (hessopt=exact, current production path) ===")
flush(stdout)
ctx_cm.obj.inner_loop_opt = DEFAULT_OPT
t_A = @elapsed begin
    global nStatus_A, x_A, obj_A, n_fg_A, n_hess_A = archPQ_base_state(x_free_calib, raw_masses, ctx, ctx_cm, layout)
end
println("[A] solve: ", round(t_A, digits=2), "s  nStatus=", nStatus_A, "  n_fg=", n_fg_A, "  n_hess=", n_hess_A)
flush(stdout)
verify_A = run_verifier(x_A, obj_A)
println("[A] kkt_resid = ", verify_A.kkt_resid)
flush(stdout)

println("\n=== [B] HVP inner solve (hessopt=5/product, algorithm=cg) ===")
flush(stdout)
ctx_cm.obj.inner_loop_opt = HVP_OPT
t_B = @elapsed begin
    global nStatus_B, x_B, obj_B, n_fg_B, n_hess_B = archPQ_base_state_hvp(x_free_calib, raw_masses, ctx, ctx_cm, layout)
end
println("[B] solve: ", round(t_B, digits=2), "s  nStatus=", nStatus_B, "  n_fg=", n_fg_B, "  n_hess=", n_hess_B)
flush(stdout)
verify_B = run_verifier(x_B, obj_B)
println("[B] kkt_resid = ", verify_B.kkt_resid)
flush(stdout)

ctx_cm.obj.inner_loop_opt = DEFAULT_OPT

println("\n=== A vs B agreement ===")
dzeta = abs(x_A[1] - x_B[1])
dlambda = maximum(abs.(x_A[2:end] .- x_B[2:end]))
println("|zeta_A - zeta_B| = ", dzeta, "   max|lambda_A - lambda_B| = ", dlambda)

@printf("\n%-8s %-10s %-8s %-8s %-12s %-14s\n", "variant", "nStatus", "n_fg", "n_hess", "wall(s)", "kkt_resid")
@printf("%-8s %-10d %-8d %-8d %-12.2f %-14.3e\n", "dense", nStatus_A, n_fg_A, n_hess_A, t_A, verify_A.kkt_resid)
@printf("%-8s %-10d %-8d %-8d %-12.2f %-14.3e\n", "hvp", nStatus_B, n_fg_B, n_hess_B, t_B, verify_B.kkt_resid)
@printf("\nspeedup (dense wall / hvp wall) = %.2fx\n", t_A / t_B)

println("\nDONE")
