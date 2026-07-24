# Gate B (exclude-ROW-destination production release, 2026-07-24): genuine CM+ZC K_mean=1/K_pair=1
# and origin-specific-ZC K_mean=1/K_pair=1 real-point checks under destination_sample=:exclude_row,
# real D=20/W=80000. lfix_cplus_exclude_row_validation.jl already covers plain CM (:cm_only) and
# origin-ZC at K_pair=0 with the full 5-part battery (incremental exact-tier, threaded/serial,
# finite-difference); this script covers the K_pair=1 gap the task brief explicitly calls out
# ("the prior sustained origin-specific campaign may have used K_pair=0 -- this release must
# explicitly test K_pair=1") with the one-real-point-per-family bar the brief also sets, reusing
# the SAME core algorithm the 5-part battery already validated (K_pair mainly changes moment
# construction/dimension, not lfix_incremental_at_Cplus! itself).
#
# Run: JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#        full_aod_diag/d4_exact/test_exclude_row_gateB_meanzc_originzc_k1.jl
const D4X = @__DIR__
cd(D4X)
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_screen_bridge.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_outer_driver.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl",
          "cm_meanzc_production.jl", "cm_meanzc_cplus.jl", "cm_checkpoint.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl",
          "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(D4X, f))
end
println("=== includes OK ==="); flush(stdout)
using Printf, LinearAlgebra

W = 80000
ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

println("--- building ctx (destination_sample=:exclude_row, W=$W) ---"); flush(stdout)
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :pseudorandom,
                             draw_seed = 20260719, destination_sample = :exclude_row)
pe = build_pivot_elimination(ctx)
D = ctx.D; Ddest = ctx.D_dest
println("ctx.D=", D, " D_dest=", Ddest, " row_idx=", ctx.row_idx)
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest]
zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, Ddest), pe)
gp0 = ctx.θ0_up[3+D]
w0 = vcat(gp0 * 1.01, zfree0)
xfc = vcat(w0[1], vec(exp.(pivot_expand(w0[2:end], pe))))

snaps10 = nested_grid_sequence([10])[10]
cplus_pool = build_grad_workspace_pool(W)
cplus_ws = build_lfix_factorized_workspace(D, Ddest, W)

println("\n" * "="^78); println("PART A: genuine CM+mean-ZC, K_mean=1, K_pair=1 (real point)"); println("="^78)
flush(stdout)
pcx_mz = build_cm_meanzc_production_context(ctx, CS; L = 10, K_mean = 1, K_pair = 1,
                                             contrasts = :anchored, meanzc_basis = :direct, probs = snaps10)
pcx_mz = with_screen_counters(pcx_mz)
nu_bounds_mz = meanzc_default_nu_bounds(ctx, 1)
νvec_mz = [1.0]   # E_F[z^1] level start, matches cm_production_stage_runner.jl's own MEANZC_ETA_NU0 default convention

_, base_mz, verify_mz = cm_meanzc_production_value_verified_screened(xfc, νvec_mz, pcx_mz; counters = pcx_mz.screen_counters)
cls_mz = classify_inner_result(verify_mz)
ok_mz_verified = is_verified_success(verify_mz)
check("CM+ZC K=1/K_pair=1: VerifiedSolved (class=$cls_mz, inner_status=$(verify_mz.inner_status))", ok_mz_verified)
check("CM+ZC K=1/K_pair=1: Delta_dual finite ($(verify_mz.Delta_dual))", isfinite(verify_mz.Delta_dual))
check("CM+ZC K=1/K_pair=1: primal_dual_gap finite ($(verify_mz.primal_dual_gap))", isfinite(verify_mz.primal_dual_gap))
check("CM+ZC K=1/K_pair=1: kkt_resid finite ($(verify_mz.max_abs_moment_kkt_resid))", isfinite(verify_mz.max_abs_moment_kkt_resid))
check("CM+ZC K=1/K_pair=1: threshold_state active (not Inf)", isfinite(pcx_mz.ctx_cm.obj.threshold_state.threshold))
check("CM+ZC K=1/K_pair=1: screen_counters attached", hasproperty(pcx_mz, :screen_counters))

t0 = time()
g_mz_ref, _ = cm_meanzc_production_gradient(xfc, νvec_mz, pcx_mz, ctx, pe; base = base_mz, verify = verify_mz, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
t_mz_ref = time() - t0
t0 = time()
g_mz_cp, _ = cm_meanzc_production_gradient_cplus(xfc, νvec_mz, pcx_mz, ctx, pe, cplus_pool, cplus_ws; base = base_mz, verify = verify_mz, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
t_mz_cp = time() - t0
d_mz = maximum(abs.(g_mz_ref .- g_mz_cp))
cosang_mz = dot(g_mz_ref, g_mz_cp) / (norm(g_mz_ref) * norm(g_mz_cp) + 1e-300)
check("CM+ZC K=1/K_pair=1: C+ vs Reference full gradient agree (max|diff|=$d_mz, cosine=$cosang_mz)", d_mz < 1e-8)
@printf("  CM+ZC K=1/K_pair=1  len=%d  max|diff|=%.3e  cosine=%.12f  t_ref=%.2fs  t_cplus=%.2fs  speedup=%.2fx\n",
        length(g_mz_ref), d_mz, cosang_mz, t_mz_ref, t_mz_cp, t_mz_ref / t_mz_cp)
flush(stdout)

println("\n" * "="^78); println("PART B: genuine origin-specific-ZC, K_mean=1, K_pair=1 (real point)"); println("="^78)
flush(stdout)
cfg_oz1 = OriginZCConfig(distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = 1, K_pair = 1,
                          power_target_layout = :origin_by_power, meanzc_basis = :direct)
layout_oz1 = originzc_make_layout(cfg_oz1, D)
pcx_oz1 = build_originzc_production_context(ctx, CS, layout_oz1)
pcx_oz1 = with_screen_counters(pcx_oz1)
nu0_oz1 = ones(n_eta(layout_oz1))

_, base_oz1, verify_oz1 = cm_originzc_production_value_verified_screened(xfc, nu0_oz1, pcx_oz1; counters = pcx_oz1.screen_counters)
cls_oz1 = classify_inner_result(verify_oz1)
ok_oz1_verified = is_verified_success(verify_oz1)
check("originZC K=1/K_pair=1: VerifiedSolved (class=$cls_oz1, inner_status=$(verify_oz1.inner_status))", ok_oz1_verified)
check("originZC K=1/K_pair=1: Delta_dual finite ($(verify_oz1.Delta_dual))", isfinite(verify_oz1.Delta_dual))
check("originZC K=1/K_pair=1: primal_dual_gap finite ($(verify_oz1.primal_dual_gap))", isfinite(verify_oz1.primal_dual_gap))
check("originZC K=1/K_pair=1: kkt_resid finite ($(verify_oz1.max_abs_moment_kkt_resid))", isfinite(verify_oz1.max_abs_moment_kkt_resid))
check("originZC K=1/K_pair=1: threshold_state active (not Inf)", isfinite(pcx_oz1.ctx_cm.obj.threshold_state.threshold))
check("originZC K=1/K_pair=1: screen_counters attached", hasproperty(pcx_oz1, :screen_counters))

t0 = time()
g_oz1_ref, _ = cm_originzc_production_gradient(xfc, nu0_oz1, pcx_oz1, ctx, pe; base = base_oz1, verify = verify_oz1, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
t_oz1_ref = time() - t0
t0 = time()
g_oz1_cp, _ = cm_originzc_production_gradient_cplus(xfc, nu0_oz1, pcx_oz1, ctx, pe, cplus_pool, cplus_ws; base = base_oz1, verify = verify_oz1, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
t_oz1_cp = time() - t0
d_oz1 = maximum(abs.(g_oz1_ref .- g_oz1_cp))
cosang_oz1 = dot(g_oz1_ref, g_oz1_cp) / (norm(g_oz1_ref) * norm(g_oz1_cp) + 1e-300)
check("originZC K=1/K_pair=1: C+ vs Reference full gradient agree (max|diff|=$d_oz1, cosine=$cosang_oz1)", d_oz1 < 1e-8)
check("originZC K=1/K_pair=1: gradient length == D*Ddest + n_eta(layout) ($(D*Ddest + n_eta(layout_oz1)))",
      length(g_oz1_cp) == D * Ddest + n_eta(layout_oz1))
@printf("  originZC K=1/K_pair=1  len=%d  max|diff|=%.3e  cosine=%.12f  t_ref=%.2fs  t_cplus=%.2fs  speedup=%.2fx\n",
        length(g_oz1_ref), d_oz1, cosang_oz1, t_oz1_ref, t_oz1_cp, t_oz1_ref / t_oz1_cp)
flush(stdout)

println()
if ALL_PASS[]
    println(">>> RESULT: ALL PASS")
else
    println(">>> RESULT: SOME FAILURES -- see above")
    exit(1)
end
