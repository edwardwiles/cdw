# Phase 14 final-gates driver (integration/phase14-final-gates-2026-08-02): parametrized-W
# genuine-cold flexible-CM solve at PRODUCTION dimensions (D=20, Ddest=19, L=50), reusing
# run_coldsolve_flexcm_w100k_2026-08-02.jl's exact template verbatim, only parametrizing W via
# ARGS[1] (defaults to 100_000) and adding a post-solve allocation measurement of the hot-path
# reduced FG functor (`base.st(x_state, g)`) and Hessian callback (`archC_hess_cb_builder`), so this
# single real cold-solve process serves BOTH the W100k/W500k allocation audit (item 1) and the
# W500k public-entry smoke (item 2). Must be launched as a FRESH `julia` process. Usage:
#   julia run_prodscale_flexcm_2026-08-02.jl 500000
const D4X = @__DIR__
t0_total = time()
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "no_dense_g_counters.jl", "economic_operator.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl",
          "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
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
          "operator_verification.jl",
          "profiled_reduced_lookup_kernels_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random
flush(stdout)
println("PID=", getpid(), "  include done at t=", round(time() - t0_total, digits = 1), "s"); flush(stdout)

const W_VAL = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
const D_VAL, DDEST_VAL, L_VAL = 20, 19, 50

println("="^90); println("GENUINE-COLD flexible-CM solve, PRODUCTION DIMS: D=$D_VAL Ddest=$DDEST_VAL L=$L_VAL W=$W_VAL")
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

t_build = @elapsed begin
    aug_reduced = build_cm_augmented_obj_archB(ctx, CS; L = L_VAL, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
    # threaded_bins=true (profiled-inner-readiness-2026-08-03): see run_coldsolve_flexcm_w100k_2026-08-02.jl's
    # own comment -- gate-proven at D4 to match serial to ~1e-14 at production's 10-thread policy.
    cctx_reduced = build_cm_bin_ctx(ctx, aug_reduced; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = true)
end
@printf("reduced object/cctx build: %.2fs  NCORE=%d  ncm=%d\n", t_build, cctx_reduced.NCORE, cctx_reduced.ncm)
flush(stdout)

reset_no_dense_g_counters!()
println("Starting COLD KNITRO inner solve ..."); flush(stdout)
t_solve = @elapsed base = reduced_cm_base_state(x_free_calib, ctx, layout, cctx_reduced)
@printf("SOLVE: wall=%.2fs  nStatus=%d  zeta*=%.10f  n_fg=%d  n_hess=%d\n",
    t_solve, base.inner_status, base.ζstar, base.n_fg, base.n_hess)
flush(stdout)

c_disp = NO_DENSE_G_COUNTERS[]
n_dense_econ_after_solve = c_disp.dense_economic_G_materializations
n_dense_cm_after_solve = c_disp.dense_CM_G_materializations
@printf("dense_economic_G_materializations=%d  dense_CM_G_materializations=%d (0 expected)\n",
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
t_verify = @elapsed ov = brute_force_verify(aug_reduced.obj_cm, base.ζstar, base.λstar, θ_full_calib, ctx.U, W)
@printf("VERIFY: wall=%.2fs  Delta_dual=%.10g  Delta_primal=%.10g  kkt_resid=%.4g\n",
    t_verify, ov.Delta_dual, ov.Delta_primal, ov.kkt_resid)
flush(stdout)

# ---- Phase 14 item 1: allocation audit at THIS W (hot-path reduced FG functor + Hessian cb) ----
x_state = vcat(base.ζstar, base.λstar); n_dual = length(x_state)
g_buf = zeros(n_dual)
base.st(x_state, g_buf)  # JIT warm-up
b_fg = @allocated base.st(x_state, g_buf)
@printf("ALLOC_FG: bytes_per_call=%d  n_dual=%d  W=%d\n", b_fg, n_dual, W_VAL)
flush(stdout)

b_hess = try
    cb = archC_hess_cb_builder(cctx_reduced)
    hlen = n_dual * (n_dual + 1) ÷ 2
    h = Vector{Float64}(undef, hlen); fake_req = (x = x_state,); fake_res = (hess = h,)
    cb(nothing, nothing, fake_req, fake_res, aug_reduced.obj_cm)  # warm-up
    @allocated cb(nothing, nothing, fake_req, fake_res, aug_reduced.obj_cm)
catch e
    println("ALLOC_HESS: measurement failed (", typeof(e), "), skipping -- FG allocation above is unaffected.")
    -1
end
b_hess >= 0 && @printf("ALLOC_HESS: bytes_per_call=%d  n_dual=%d  W=%d\n", b_hess, n_dual, W_VAL)
flush(stdout)

ok = base.inner_status == 0 && ov.kkt_resid < 1e-3 &&
     n_dense_econ_after_solve == 0 && n_dense_cm_after_solve == 0
@printf("\nTOTAL WALL: %.2fs\n", time() - t0_total)
println("FLEXCM_PRODSCALE_W$(W_VAL)_RESULT: ", ok ? "PASS" : "FAIL")
ok || exit(1)
