# Default-flips task (2026-07-27), Task C gate: MOMENT_REPRESENTATION explicit dispatch
# (:operator | :dense_reference) for the generic once-per-inner-solve composite-G SETUP call
# (`obj.moments!(@view(H[:,1]), select_G_from_H(obj,H), theta, obj.U, obj)` in
# `inner_loop_internal_archgeneric`/family moments! closures) -- NOT the per-Newton-iterate FG
# callback, which was already correctly operator-vs-dense forked before this task.
#
# Verifies, using the EXISTING no_dense_g_counters.jl counters (no new counter names invented):
#   1. Flexible CM, at production defaults (CM_INNER_FG_BACKEND_DEFAULT[]=:cm_lookup,
#      CM_CROSS_HESSIAN_BACKEND_DEFAULT[]=:winner_bin) + MOMENT_REPRESENTATION[]=:operator (default):
#      generic_dense_FG_calls==0, dense_CM_G_materializations==0, full_G_materializations==0.
#   2. The SAME flexible-CM context with MOMENT_REPRESENTATION[] forced to :dense_reference:
#      dense_CM_G_materializations becomes NONZERO -- proving the selector genuinely controls
#      behavior (not just "always zero regardless").
#   3. Common-Fréchet, at production defaults + MOMENT_REPRESENTATION[]=:operator: dense_Frechet_G_
#      materializations is NONZERO regardless (deliberately NOT wired to this selector -- the
#      Hessian callback genuinely needs those columns, see MOMENT_REPRESENTATION's own docstring and
#      the nStatus=-400 regression this exact skip caused once before,
#      docs/COMMON_FRECHET_FG_D20_FINAL_GATE_2026-07-27.md). Confirms this family's own solve is
#      UNCHANGED, not silently broken, by this task's selector.
#   4. Unrestricted, at MOMENT_REPRESENTATION[]=:operator: generic_dense_FG_calls==0 and
#      full_G_materializations==0 -- trivially true (this family never calls
#      inner_loop_internal_archgeneric at all; its own bespoke inner_loop_internal_profiled,
#      oracle_fast.jl, is untouched by this task), documented as structural, not new work.
#
# Usage: julia --project=. -t 4 full_aod_diag/d4_exact/test_moment_representation_default_2026-07-27.jl
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl",
          "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "draw_design.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl",
          "operator_verification.jl"]
    include(joinpath(D4X, f))
end

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1; println("  PASS: ", name)
    else
        n_fail += 1; println("  FAIL: ", name)
    end
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]

println("MOMENT_REPRESENTATION[] initial default: ", MOMENT_REPRESENTATION[])
check("MOMENT_REPRESENTATION[] defaults to :operator", MOMENT_REPRESENTATION[] == :operator)
check("CM_INNER_FG_BACKEND_DEFAULT[] is :cm_lookup (precondition for skip to engage)", CM_INNER_FG_BACKEND_DEFAULT[] == :cm_lookup)
check("CM_CROSS_HESSIAN_BACKEND_DEFAULT[] is :winner_bin (precondition for skip to engage)", CM_CROSS_HESSIAN_BACKEND_DEFAULT[] == :winner_bin)

println("\n=== 1. Flexible CM, MOMENT_REPRESENTATION[]=:operator (default) ===")
MOMENT_REPRESENTATION[] = :operator
reset_no_dense_g_counters!()
pcx = build_cm_production_context(ctx, CS; L = 20, contrasts = :anchored)
base1 = archC_base_state(x_free_calib, pcx.ctx_cm, pcx.cctx)
r1 = no_dense_g_report()
println("  generic_dense_FG_calls=", r1.generic_dense_FG_calls, " dense_CM_G_materializations=", r1.dense_CM_G_materializations,
        " full_G_materializations=", r1.full_G_materializations, " nStatus=", base1.inner_status)
check("flexible CM :operator -- generic_dense_FG_calls==0", r1.generic_dense_FG_calls == 0)
check("flexible CM :operator -- dense_CM_G_materializations==0", r1.dense_CM_G_materializations == 0)
check("flexible CM :operator -- full_G_materializations==0", r1.full_G_materializations == 0)
check("flexible CM :operator -- inner solve feasible", base1.inner_status in (0, -100, -101, -103))

println("\n=== 2. Flexible CM, MOMENT_REPRESENTATION[] forced to :dense_reference (selector control-check) ===")
MOMENT_REPRESENTATION[] = :dense_reference
reset_no_dense_g_counters!()
base2 = archC_base_state(x_free_calib, pcx.ctx_cm, pcx.cctx)
r2 = no_dense_g_report()
println("  generic_dense_FG_calls=", r2.generic_dense_FG_calls, " dense_CM_G_materializations=", r2.dense_CM_G_materializations,
        " nStatus=", base2.inner_status)
check("flexible CM :dense_reference -- dense_CM_G_materializations>0 (selector genuinely controls behavior)", r2.dense_CM_G_materializations > 0)
check("flexible CM :dense_reference -- inner solve still feasible", base2.inner_status in (0, -100, -101, -103))
check("flexible CM -- zeta* unaffected by MOMENT_REPRESENTATION (same answer both ways)", abs(base1.ζstar - base2.ζstar) < 1e-9)
MOMENT_REPRESENTATION[] = :operator   # restore default before the remaining sections

println("\n=== 3. Common-Fréchet, MOMENT_REPRESENTATION[]=:operator (deliberately NOT wired -- must be unaffected) ===")
reset_no_dense_g_counters!()
pcx_f = build_cm_frechet_production_context(ctx, CS; L = 20, contrasts = :anchored, cm_hessian_backend = :structured)
base3 = archC_frechet_base_state(x_free_calib, pcx_f.ctx_cm, pcx_f.cctx, pcx_f.aug.level_targets)
r3 = no_dense_g_report()
println("  dense_Frechet_G_materializations=", r3.dense_Frechet_G_materializations, " nStatus=", base3.inner_status)
check("common-Fréchet :operator -- dense_Frechet_G_materializations>0 (deliberately still fills, safety-preserved)", r3.dense_Frechet_G_materializations > 0)
check("common-Fréchet :operator -- inner solve feasible", base3.inner_status in (0, -100, -101, -103))

println("\n=== 4. Unrestricted, MOMENT_REPRESENTATION[]=:operator (structural, not new work) ===")
reset_no_dense_g_counters!()
result, prof_meta = evaluate_fullA_fast(x_free_calib, ctx)
r4 = no_dense_g_report()
println("  generic_dense_FG_calls=", r4.generic_dense_FG_calls, " full_G_materializations=", r4.full_G_materializations,
        " status=", result.inner_status)
check("unrestricted :operator -- generic_dense_FG_calls==0 (own bespoke dispatcher, untouched)", r4.generic_dense_FG_calls == 0)
check("unrestricted :operator -- full_G_materializations==0", r4.full_G_materializations == 0)
check("unrestricted -- inner solve feasible", result.inner_status in (0, -100, -101, -103))

println("\n============================================================")
println("TOTAL: $n_pass passed, $n_fail failed")
n_fail == 0 || error("$n_fail check(s) failed")
