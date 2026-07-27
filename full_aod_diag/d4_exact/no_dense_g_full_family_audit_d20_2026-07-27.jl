# agent/dense-g-audit-2026-07-27, task §7 Part C: real D=20/W=80,000 counterpart to
# no_dense_g_full_family_audit_2026-07-27.jl (D=4). Same attribution note applies: include list and
# per-family call pattern adapted from `smoke_no_dense_g_five_families.jl` (D=4, commit ccc2bbb) and
# `test_flexible_cm_winner_bin_her_wiring_d20.jl` (real-D20 setup recipe: `d20_real_setup`,
# `destination_sample=:exclude_row`), combined here to cover the same 5 families at real scale.
#
# Usage: julia --project=. -t 4 full_aod_diag/d4_exact/no_dense_g_full_family_audit_d20_2026-07-27.jl
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl", "instrumentation.jl",
          "oracle_fast.jl", "gravity_elimination.jl", "structured_moment_build.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "operator_verification.jl"]
    include(joinpath(D4X, f))
end
using Random

println("Building real D=20 context (W=80000, delta=1.0, destination_sample=:exclude_row)..."); flush(stdout)
ctx = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]
D = ctx.D
Random.seed!(2026)
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

println("\n=== flexible CM (production default: FG=$(CM_INNER_FG_BACKEND_DEFAULT[]), cross-Hessian=$(CM_CROSS_HESSIAN_BACKEND_DEFAULT[])) ==="); flush(stdout)
reset_no_dense_g_counters!()
pcx = build_cm_production_context(ctx, CS; L = 50, contrasts = :anchored)
archC_verified_state(x_free_calib, pcx.ctx_cm, pcx.cctx)
full_report("flexible_cm_d20")

println("\n=== common Frechet (production default: FG=$(CM_FRECHET_INNER_FG_BACKEND_DEFAULT[])) ==="); flush(stdout)
reset_no_dense_g_counters!()
pcx_f = build_cm_frechet_production_context(ctx, CS; L = 50, contrasts = :anchored, cm_hessian_backend = :structured)
archC_frechet_base_state(x_free_calib, pcx_f.ctx_cm, pcx_f.cctx, pcx_f.aug.level_targets)
full_report("common_frechet_d20")

println("\n=== CM+ZC (production default: FG=$(CM_MEANZC_INNER_FG_BACKEND_DEFAULT[])) ==="); flush(stdout)
reset_no_dense_g_counters!()
νvec0 = [1.0]
pcx_z = build_cm_meanzc_production_context(ctx, CS; L = 50, K_mean = 1, K_pair = 0, contrasts = :anchored)
cm_meanzc_production_value_verified(x_free_calib, νvec0, pcx_z)
full_report("cm_plus_zc_d20")

println("\n=== origin-ZC / ZC-only (production default: FG=$(ORIGINZC_FG_BACKEND_DEFAULT[])) ==="); flush(stdout)
reset_no_dense_g_counters!()
layout = OriginByPowerLayout(D, 1, 0)
νfull0 = nu0_origin(1, D)
pcx_o = build_originzc_production_context(ctx, CS, layout)
cm_originzc_production_value_verified(x_free_calib, νfull0, pcx_o)
full_report("zc_only_d20")

println("\n=== DONE ===")
flush(stdout)
