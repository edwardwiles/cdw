# D4 A/B: current dense-Hessian inner solve (hessopt=exact, ek_inner.opt) vs the new HVP inner
# solve (hessopt=5/product, algorithm=cg, ek_inner_hvp.opt) for the pairwise-quantile-independence
# restriction. Both run a REAL KN_solve() against the real d4_exact_setup context, and both
# converged points are passed through the SAME independent verifier
# (verify_inner_solution_operator_pairwisequantile!) -- correctness bar per the handover doc:
# "pass the SAME converged point through the verifier for BOTH and confirm the KKT residual is
# small for both (not just that one is faster)."
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
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_verification.jl",
          "pairwise_quantile_production.jl", "pairwise_quantile_hvp.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

const PQ_L = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5   # test-file convenience only; production requires explicit L
# Version B needs an EXPLICIT cutoff source (no default anywhere in production); these
# diagnostic scripts pin :empirical_quantile because that is the setting under which
# mu = 1/L reproduces version A's moment matrix exactly.
const CUTOFF_SOURCE = :empirical_quantile
ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
layout = PairwiseQuantileMassLayout(ctx.D, PQ_L)
aug = build_pairwise_quantile_augmented_obj(ctx, layout, pairwise_quantile_fixed_cutoffs(ctx.U, PQ_L; cutoff_source = CUTOFF_SOURCE))
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

println("=== [A] dense-Hessian inner solve (hessopt=exact, ", DEFAULT_OPT, ") ===")
ctx_cm.obj.inner_loop_opt = DEFAULT_OPT
t0 = time()
nStatus_A, x_A, obj_A, n_fg_A, n_hess_A = archPQ_base_state(x_free_calib, raw_masses, ctx, ctx_cm, layout)
t_A = time() - t0
println("nStatus=", nStatus_A, "  n_fg=", n_fg_A, "  n_hess=", n_hess_A, "  wall=", round(t_A, digits=3), "s")
check("[A] dense: feasible", nStatus_A in (0, -100, -101, -103))
verify_A = run_verifier(x_A, obj_A)
println("[A] kkt_resid = ", verify_A.kkt_resid)
check("[A] dense: verifier KKT residual small (< 1e-4)", verify_A.kkt_resid < 1e-4)

println("\n=== [B] HVP inner solve (hessopt=5/product, algorithm=cg, ", HVP_OPT, ") ===")
ctx_cm.obj.inner_loop_opt = HVP_OPT
t0 = time()
nStatus_B, x_B, obj_B, n_fg_B, n_hess_B = archPQ_base_state_hvp(x_free_calib, raw_masses, ctx, ctx_cm, layout)
t_B = time() - t0
println("nStatus=", nStatus_B, "  n_fg=", n_fg_B, "  n_hess=", n_hess_B, "  wall=", round(t_B, digits=3), "s")
check("[B] hvp: feasible", nStatus_B in (0, -100, -101, -103))
verify_B = run_verifier(x_B, obj_B)
println("[B] kkt_resid = ", verify_B.kkt_resid)
check("[B] hvp: verifier KKT residual small (< 1e-4)", verify_B.kkt_resid < 1e-4)

ctx_cm.obj.inner_loop_opt = DEFAULT_OPT

println("\n=== A vs B agreement (same convex problem -> same optimum) ===")
dzeta = abs(x_A[1] - x_B[1])
dlambda = maximum(abs.(x_A[2:end] .- x_B[2:end]))
println("|zeta_A - zeta_B| = ", dzeta, "   max|lambda_A - lambda_B| = ", dlambda)
check("A/B duals agree to < 1e-4", dzeta < 1e-4 && dlambda < 1e-4)

@printf("\n%-8s %-8s %-8s %-8s %-10s %-14s\n", "variant", "nStatus", "n_fg", "n_hess", "wall(s)", "kkt_resid")
@printf("%-8s %-8d %-8d %-8d %-10.3f %-14.3e\n", "dense", nStatus_A, n_fg_A, n_hess_A, t_A, verify_A.kkt_resid)
@printf("%-8s %-8d %-8d %-8d %-10.3f %-14.3e\n", "hvp", nStatus_B, n_fg_B, n_hess_B, t_B, verify_B.kkt_resid)

println()
println(ALL_PASS[] ? "ALL D4 A/B CHECKS PASSED" : "SOME CHECKS FAILED")
