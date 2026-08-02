# Phase 9/production-dims gate (integration/profiled-all-five-production-closeout, 2026-08-02):
# GENUINE-COLD CM+ZC ("mean ZC") solve at PRODUCTION dimensions (D=20, Ddest=19, L=50, K_mean=3,
# K_pair=3, W=100,000 -- NOT the smaller K_mean=1/K_pair=0/L=2-3 toy config prior sessions
# certified), via the zero-dense reduced/operator inner-solve path (`reduced_meanzc_base_state`,
# widened-core H_EM Hessian). Must be launched as a FRESH `julia` process -- see
# run_coldsolve_flexcm_w100k_2026-08-02.jl's own header for why.
const D4X = @__DIR__
t0_total = time()
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl",
          "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_meanzc_lookup_kernels.jl", "cm_meanzc_lookup_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_checkpoint.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl",
          "recover_full_a_2026-07-31.jl", "reduced_recovery_from_lfd_2026-08-01.jl",
          "profiled_reduced_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random
flush(stdout)
println("PID=", getpid(), "  include done at t=", round(time() - t0_total, digits = 1), "s"); flush(stdout)

const W_VAL = 100_000
const D_VAL, DDEST_VAL, L_VAL, K_MEAN, K_PAIR = 20, 19, 50, 3, 3

println("="^90); println("GENUINE-COLD CM+ZC solve, PRODUCTION DIMS: D=$D_VAL Ddest=$DDEST_VAL L=$L_VAL K_mean=$K_MEAN K_pair=$K_PAIR W=$W_VAL")
println("="^90); flush(stdout)

t_ctx = @elapsed ctx = d20_real_setup(W = W_VAL, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
@printf("context build: %.2fs\n", t_ctx); flush(stdout)
D = ctx.D; Ddest = ctx.D_dest
@assert D == D_VAL && Ddest == DDEST_VAL "expected D=$D_VAL/Ddest=$DDEST_VAL, got D=$D/Ddest=$Ddest"
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
W = size(ctx.U, 1)

korea_idx = 14; brazil_idx = 3
t_layout = @elapsed begin
    spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
    cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
    has_france = cf_probe.cf_col > 0
    layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
    reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
end
@printf("layout build: %.2fs  total_reduced_econ=%d  france=%s\n", t_layout, layout.total_reduced_economic_moments, has_france)
flush(stdout)

νvec0 = fill(1.0, K_MEAN)
t_build = @elapsed begin
    aug_reduced = build_cm_meanzc_augmented_obj(ctx, CS; L = L_VAL, K_mean = K_MEAN, K_pair = K_PAIR,
        base_obj = reduced_obj0, profiled_layout = layout)
    cctx_reduced = build_cm_meanzc_bin_ctx(ctx, aug_reduced; core_hessian_backend = :exact_winner_pair_parallel,
        zc_cross_hessian_backend = :winner_bin, threaded_bins = false, inner_fg_backend = :dense_reference,
        profiled_layout = layout)
end
@printf("reduced object/cctx build: %.2fs  ncore_core=%d  NCORE=%d  ncm=%d\n",
    t_build, cctx_reduced.ncore_core, cctx_reduced.NCORE, cctx_reduced.ncm)
flush(stdout)

reset_no_dense_g_counters!()
println("Starting COLD KNITRO inner solve ..."); flush(stdout)
t_solve = @elapsed base = reduced_meanzc_base_state(x_free_calib, νvec0, ctx, layout, cctx_reduced)
@printf("SOLVE: wall=%.2fs  nStatus=%d  zeta*=%.10f  n_fg=%d  n_hess=%d\n",
    t_solve, base.inner_status, base.ζstar, base.n_fg, base.n_hess)
flush(stdout)

# Snapshot as PLAIN INTEGERS/booleans immediately after the solve -- NO_DENSE_G_COUNTERS[] returns
# the live mutable counters struct (not a copy), and brute_force_verify below deliberately calls
# the dense obj.moments! directly for an independent cross-check, which would legitimately bump
# these same counters if re-read afterward -- snapshotting now isolates the SOLVE's own claim.
c_disp = NO_DENSE_G_COUNTERS[]
check_disp = c_disp.winner_cross_hessian_calls > 0 && c_disp.dense_cross_hessian_calls == 0
n_dense_econ_after_solve = c_disp.dense_economic_G_materializations
n_dense_cm_after_solve = c_disp.dense_CM_G_materializations
@printf("winner_cross_hessian_calls=%d  dense_cross_hessian_calls=%d  dense_economic_G_materializations=%d  dense_CM_G_materializations=%d\n",
    c_disp.winner_cross_hessian_calls, c_disp.dense_cross_hessian_calls,
    n_dense_econ_after_solve, n_dense_cm_after_solve)
flush(stdout)

function brute_force_verify(obj, ζ::Float64, λ::AbstractVector{Float64}, θ_ext, U, W)
    K = Vector{Float64}(undef, W)
    G = Matrix{Float64}(undef, W, obj.d - 1)
    obj.moments!(K, G, θ_ext, U, obj)
    r = fill(-ζ, W) .- G * λ
    Psi_r = similar(r); obj.Psi!(Psi_r, r)
    f = sum(Psi_r) / W + ζ
    dPsi_r = similar(r); obj.dPsi!(dPsi_r, r)
    g = -(G' * dPsi_r) ./ W
    return (r = r, f = f, g = g, m_weights = dPsi_r, Delta_dual = -f, Delta_primal = primal_divergence(dPsi_r),
        kkt_resid = maximum(abs, g))
end

θ_ext_calib = vcat(collect(θ_full_calib), νvec0)
t_verify = @elapsed ov = brute_force_verify(aug_reduced.obj_cm, base.ζstar, base.λstar, θ_ext_calib, ctx.U, W)
@printf("VERIFY: wall=%.2fs  Delta_dual=%.10g  Delta_primal=%.10g  kkt_resid=%.4g\n",
    t_verify, ov.Delta_dual, ov.Delta_primal, ov.kkt_resid)
flush(stdout)

ok = base.inner_status == 0 && ov.kkt_resid < 1e-3 && check_disp &&
     n_dense_econ_after_solve == 0 && n_dense_cm_after_solve == 0
@printf("\nTOTAL WALL: %.2fs\n", time() - t0_total)
println("CMZC_W100K_COLD_SOLVE_RESULT: ", ok ? "PASS" : "FAIL")
ok || exit(1)
