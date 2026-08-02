# Integration continuation (2026-08-02): verification gate for common Fréchet's genuine, matrix-free
# reduced+CM-grid+level operator FG (profiled_reduced_frechet_lookup_kernels_2026-08-02.jl). Same
# three independent checks as flexible CM's own gate.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "no_dense_g_counters.jl", "economic_operator.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl",
          "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_checkpoint.jl",
          "operator_hessian_weights.jl", "operator_psi_bundle.jl", "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl",
          "compressed_factual_buffer_reuse.jl",
          "profiled_reduced_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_frechet_lookup_kernels_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
Random.seed!(2026)

spec = build_anchor_spec_from_ctx(ctx)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
assert_no_factual_price_index_moment(layout)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
L = 10; contrasts = :anchored

aug_reduced_f = build_cm_frechet_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0, profiled_layout = layout)
level_targets = aug_reduced_f.level_targets

# =====================================================================================
# Check 1: FD gradient check on the new operator FG, at a random (non-optimal) point
# =====================================================================================
println("="^90); println("Check 1: FD gradient check on ReducedCMFrechetLookupState's functor"); println("="^90)
cctx_probe = build_cm_bin_ctx(ctx, aug_reduced_f; profiled_layout = layout, inner_fg_backend = :dense_reference,
                               threaded_bins = false, core_hessian_backend = :dense_reference, cm_cross_hessian_backend = :winner_bin)
obj_probe, st_probe = build_reduced_frechet_operator_bundle(ctx, θ_full_calib, layout, cctx_probe, level_targets)
prime_operator!(obj_probe, θ_full_calib, ctx, cctx_probe.core_cf_ref; restriction_state = cctx_probe)
cctx_probe.profiled_theta_ref[] = copy(θ_full_calib)

n = obj_probe.outer_constr_index
Random.seed!(2027)
x0 = 0.01 .* randn(n)
g_analytic = zeros(n)
f0 = st_probe(x0, g_analytic)
check("Check1: f0 finite", isfinite(f0))

h = 1e-6
g_fd = zeros(n)
for i in 1:n
    xp = copy(x0); xp[i] += h
    xm = copy(x0); xm[i] -= h
    fp = st_probe(xp, Float64[])
    fm = st_probe(xm, Float64[])
    g_fd[i] = (fp - fm) / (2h)
end
err = maximum(abs.(g_analytic .- g_fd))
relerr = err / maximum(abs.(g_fd))
@printf("  max_abs_err=%.3e  max_rel_err=%.3e  (n=%d: 1 zeta + %d econ + %d CM + %d level, no gravity)\n",
    err, relerr, n, layout.total_reduced_economic_moments, cctx_probe.ncm - cctx_probe.L, cctx_probe.L)
check("Check1: analytic gradient matches central-FD (<1e-6 abs, generous for h=1e-6)", err < 1e-6)

# =====================================================================================
# Check 2: real KNITRO solve via the new operator path
# =====================================================================================
println("="^90); println("Check 2: real KNITRO solve via the new (gravity-free) operator path"); println("="^90)
cctx_reduced_new = build_cm_bin_ctx(ctx, aug_reduced_f; profiled_layout = layout, inner_fg_backend = :dense_reference,
                                     threaded_bins = false, core_hessian_backend = :dense_reference, cm_cross_hessian_backend = :winner_bin)
before = deepcopy(NO_DENSE_G_COUNTERS[])
base_new = reduced_frechet_base_state(x_free_calib, ctx, layout, cctx_reduced_new, level_targets)
after = NO_DENSE_G_COUNTERS[]
@printf("  NEW (operator, zero-dense-G, no gravity)  inner_status=%d  zeta*=%.12f\n", base_new.inner_status, base_new.ζstar)
check("Check2: NEW operator-path reduced solve reaches optimality (nStatus==0)", base_new.inner_status == 0)

# =====================================================================================
# Check 3: zero dense-G materialization across the ENTIRE new-path solve
# =====================================================================================
println("="^90); println("Check 3: zero dense-G materialization (NO_DENSE_G_COUNTERS delta across the new-path solve)"); println("="^90)
d_econ = after.dense_economic_G_materializations - before.dense_economic_G_materializations
d_cm = after.dense_CM_G_materializations - before.dense_CM_G_materializations
@printf("  delta dense_economic_G_materializations=%d  delta dense_CM_G_materializations=%d  (n_fg_calls this solve=%d)\n",
    d_econ, d_cm, base_new.n_fg)
check("Check3: zero dense economic G materializations during the new-path solve", d_econ == 0)
check("Check3: zero dense CM-grid G materializations during the new-path solve", d_cm == 0)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
