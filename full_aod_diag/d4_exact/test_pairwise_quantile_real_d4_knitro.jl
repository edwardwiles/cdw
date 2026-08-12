# Real D4 KNITRO smoke test for the pairwise-quantile-independence restriction, using the ACTUAL
# production economic context (d4_exact_setup) and a REAL KN_solve, not a synthetic standalone
# script. Mirrors test_zc_centered_cache_d4.jl's include chain/style.
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
          "pairwise_quantile_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random

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
println("=== building ctx via d4_exact_setup ===")
ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
println("ctx.D = ", ctx.D, "  size(ctx.U) = ", size(ctx.U), "  L = ", PQ_L)

layout = PairwiseQuantileMassLayout(ctx.D, PQ_L)
Zfeat = pairwise_quantile_frechet_features(ctx.U, ctx.μHat)   # the restriction is on the Frechet z
aug = build_pairwise_quantile_augmented_obj(ctx, layout, Zfeat,
    pairwise_quantile_fixed_cutoffs(Zfeat, PQ_L; cutoff_source = CUTOFF_SOURCE, mu_frechet = ctx.μHat))
println("obj_pq.outer_constr_index = ", aug.obj_pq.outer_constr_index, "  ncore_econ = ", aug.ncore_econ,
        "  n_total_rows(D) = ", n_total_rows(ctx.D, PQ_L))
check("outer_constr_index == ncore_econ + n_total_rows(D)", aug.obj_pq.outer_constr_index == aug.ncore_econ + n_total_rows(ctx.D, PQ_L))

mass_state = PairwiseQuantileMassState(ctx.D, PQ_L)
hess_ctx = PairwiseQuantileCoreHessCtx(aug.ncore_econ, aug.op, mass_state, aug.core_cf_ref)

ctx_cm = merge(ctx, (obj = aug.obj_pq, pq_op = aug.op, pq_mass_state = mass_state,
                      pq_core_cf_ref = aug.core_cf_ref, pq_hess_ctx = hess_ctx))

# VERSION B: the outer restriction coordinates are BIN MASSES on the simplex; the canonical
# starting point is mu = 1/L (uniform_mass_raw) -- exactly version A's fixed target. The
# cutoffs are no longer outer coordinates at all: they are FIXED at context-build time from
# CUTOFF_SOURCE (see pairwise_quantile_fixed_cutoffs).
raw_masses = uniform_mass_raw(layout)

println("\n=== running REAL KNITRO inner solve (pairwise-quantile-independence restriction) ===")
nStatus, x, obj, n_fg, n_hess = archPQ_base_state(x_free_calib, raw_masses, ctx, ctx_cm, layout)
println("nStatus = ", nStatus, "  n_fg = ", n_fg, "  n_hess = ", n_hess, "  length(x) = ", length(x))
check("REAL KNITRO inner solve feasible", nStatus in (0, -100, -101, -103))

println()
println(ALL_PASS[] ? "ALL REAL-D4-KNITRO CHECKS PASSED" : "SOME CHECKS FAILED")
