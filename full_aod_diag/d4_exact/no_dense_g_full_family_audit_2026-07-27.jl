# agent/dense-g-audit-2026-07-27, task §7 Part C: real, instrumented, full-counter-snapshot
# invariant check across every restricted family, one real inner solve each AT ITS CURRENT
# PRODUCTION-DEFAULT config (i.e. whatever CM_INNER_FG_BACKEND_DEFAULT[]/
# CM_FRECHET_INNER_FG_BACKEND_DEFAULT[]/ORIGINZC_FG_BACKEND_DEFAULT[]/
# CM_MEANZC_INNER_FG_BACKEND_DEFAULT[]/CM_CROSS_HESSIAN_BACKEND_DEFAULT[] resolve to on THIS
# branch, right now -- not some future/aspirational state).
#
# ATTRIBUTION: this script's include list, context setup, and per-family call pattern are copied
# directly from the pre-existing `smoke_no_dense_g_five_families.jl` (commit ccc2bbb, "Phase B
# item 9 (scoped): five-family no-dense-G runtime smoke test", already on this branch's history
# before this audit session started) -- that file's own `report()` only prints a SUBSET of
# `no_dense_g_report()`'s fields (the 2026-07-26-era counters). This script is a full-report
# variant, added new rather than editing the original, that prints EVERY field
# `no_dense_g_report()` now returns (including the 2026-07-27 winner-aware-H_ER cross-Hessian
# counters ccc2bbb predates) plus an explicit PASS/FAIL line against task §7's five required-zero
# invariant counters. No claim of originality on the shared setup/include-list/call pattern.
#
# Usage: julia --project=. -t 4 full_aod_diag/d4_exact/no_dense_g_full_family_audit_2026-07-27.jl
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

const REQUIRED_ZERO_FIELDS = (:full_G_materializations, :dense_economic_G_materializations,
                               :generic_dense_FG_calls, :dense_reference_verification_calls,
                               :dense_cross_hessian_calls)

function full_report(label)
    r = no_dense_g_report()
    println("[$label] FULL COUNTER SNAPSHOT:")
    for k in fieldnames(typeof(r))
        println("    $k = $(getfield(r, k))")
    end
    bad = [k for k in REQUIRED_ZERO_FIELDS if getfield(r, k) != 0]
    if isempty(bad)
        println("  >>> INVARIANT (5 required-zero counters) HOLDS for $label")
    else
        println("  >>> INVARIANT VIOLATED for $label -- nonzero: $(join(["$k=$(getfield(r,k))" for k in bad], ", "))")
    end
    flush(stdout)
    return r
end

println("=== unrestricted (Addendum Part A compressed FG -- allocation-free by construction, no dense/operator branch to select) ===")
reset_no_dense_g_counters!()
println("[unrestricted] N/A -- no dense/operator branch exists (single compressed-only FG since Addendum Part A); no_dense_g_counters.jl was never wired to this family's call sites because there is no fallback branch for it to distinguish. Not measured by this script's counters; documented, not measured-as-zero.")
flush(stdout)

println("\n=== flexible CM (production default: FG=$(CM_INNER_FG_BACKEND_DEFAULT[]), cross-Hessian=$(CM_CROSS_HESSIAN_BACKEND_DEFAULT[])) ===")
reset_no_dense_g_counters!()
pcx = build_cm_production_context(ctx, CS; L = 20, contrasts = :anchored)
archC_verified_state(x_free_calib, pcx.ctx_cm, pcx.cctx)
full_report("flexible_cm")

println("\n=== common Frechet (production default: FG=$(CM_FRECHET_INNER_FG_BACKEND_DEFAULT[])) ===")
reset_no_dense_g_counters!()
pcx_f = build_cm_frechet_production_context(ctx, CS; L = 20, contrasts = :anchored, cm_hessian_backend = :structured)
archC_frechet_base_state(x_free_calib, pcx_f.ctx_cm, pcx_f.cctx, pcx_f.aug.level_targets)
full_report("common_frechet")

println("\n=== CM+ZC (production default: FG=$(CM_MEANZC_INNER_FG_BACKEND_DEFAULT[])) ===")
reset_no_dense_g_counters!()
νvec0 = [1.0]
pcx_z = build_cm_meanzc_production_context(ctx, CS; L = 20, K_mean = 1, K_pair = 0, contrasts = :anchored)
cm_meanzc_production_value_verified(x_free_calib, νvec0, pcx_z)
full_report("cm_plus_zc")

println("\n=== origin-ZC / ZC-only (production default: FG=$(ORIGINZC_FG_BACKEND_DEFAULT[])) ===")
reset_no_dense_g_counters!()
layout = OriginByPowerLayout(D, 1, 0)
νfull0 = nu0_origin(1, D)
pcx_o = build_originzc_production_context(ctx, CS, layout)
cm_originzc_production_value_verified(x_free_calib, νfull0, pcx_o)
full_report("zc_only")

println("\n=== DONE ===")
flush(stdout)
