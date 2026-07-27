# port/finish-operator-stack-no-dense-G-and-CM-basis-diagnosis-2026-07-26, Phase B item 9 (scoped):
# one real inner solve per family AT ITS CURRENT PRODUCTION DEFAULT config, reporting
# no_dense_g_report() after each -- the cheapest real evidence for "ordinary FG-callback paths take
# the operator, not dense" across all five families as they stand after today's Phase A work.
#
# Usage: julia --project=. -t 4 full_aod_diag/d4_exact/smoke_no_dense_g_five_families.jl
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "operator_verification.jl"]
    include(joinpath(D4X, f))
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
D = ctx.D
nu0_origin(K::Int, D::Int) = vcat([fill(Float64(factorial(k)), D) for k in 1:K]...)

function report(label)
    r = no_dense_g_report()
    println("[$label]  operator_FG_calls=$(r.operator_FG_calls)  operator_forward=$(r.operator_forward_calls)  operator_transpose=$(r.operator_transpose_calls)  dense_economic_G=$(r.dense_economic_G_materializations)  dense_CM_G=$(r.dense_CM_G_materializations)  dense_ZC_G=$(r.dense_ZC_G_materializations)  generic_dense_FG=$(r.generic_dense_FG_calls)  full_G=$(r.full_G_materializations)")
end

println("=== unrestricted (Addendum Part A compressed FG -- allocation-free by construction, no dense/operator branch to select) ===")
reset_no_dense_g_counters!()
# unrestricted's default production FG (compressed_cc_inner.jl) doesn't route through
# no_dense_g_counters.jl (it predates this branch's instrumentation and has no dense-fallback
# branch at all -- it's unconditionally the compressed operator). Recorded as N/A here, documented
# in the final doc rather than faked with a report call that would show all-zeros misleadingly.
println("[unrestricted] N/A -- no dense/operator branch exists (single compressed-only FG since Addendum Part A)")

println("\n=== flexible CM (production default: cm_lookup) ===")
reset_no_dense_g_counters!()
pcx = build_cm_production_context(ctx, CS; L = 20, contrasts = :anchored)   # inner_fg_backend defaults to CM_INNER_FG_BACKEND_DEFAULT[]
archC_verified_state(x_free_calib, pcx.ctx_cm, pcx.cctx)
report("flexible_cm default=$(CM_INNER_FG_BACKEND_DEFAULT[])")

println("\n=== common Frechet (production default: dense_reference, pending this session's own flip decision) ===")
reset_no_dense_g_counters!()
pcx_f = build_cm_frechet_production_context(ctx, CS; L = 20, contrasts = :anchored, cm_hessian_backend = :structured)
archC_frechet_base_state(x_free_calib, pcx_f.ctx_cm, pcx_f.cctx, pcx_f.aug.level_targets)
report("common_frechet default=$(CM_FRECHET_INNER_FG_BACKEND_DEFAULT[])")

println("\n=== CM+ZC (production default: operator) ===")
reset_no_dense_g_counters!()
νvec0 = [1.0]
pcx_z = build_cm_meanzc_production_context(ctx, CS; L = 20, K_mean = 1, K_pair = 0, contrasts = :anchored)
cm_meanzc_production_value_verified(x_free_calib, νvec0, pcx_z)
report("cm_plus_zc default=$(CM_MEANZC_INNER_FG_BACKEND_DEFAULT[])")

println("\n=== origin-ZC / ZC-only (production default: operator) ===")
reset_no_dense_g_counters!()
layout = OriginByPowerLayout(D, 1, 0)
νfull0 = nu0_origin(1, D)
pcx_o = build_originzc_production_context(ctx, CS, layout)
cm_originzc_production_value_verified(x_free_calib, νfull0, pcx_o)
report("zc_only default=$(ORIGINZC_FG_BACKEND_DEFAULT[])")
