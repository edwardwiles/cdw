# Gap-closing test (2026-07-26), same motivation as test_phaseE_meanzc_end_to_end.jl: confirm
# common-Frechet ALSO genuinely works end-to-end through the real B1/C/D/E wiring (dense-reference
# FG dispatch tag, exact cache, dual bank, compressed-core workspace), not just "structurally
# similar to flexible-CM's already-tested chain" -- exactly the gap that let the CM+ZC CMBinHessCtx
# bug go uncaught until Phase E's own benchmark work.
#
# SCOPE NOTE: an equivalent origin-ZC end-to-end test was attempted in the same session but
# abandoned after 5 include-chain/config attempts (each failure was a load-time UndefVarError from
# an incomplete ad-hoc include list for this test harness, or a config-validation constraint on
# K_pair -- NOT a runtime code defect the way the CM+ZC CMBinHessCtx bug was). Origin-ZC's own B1/
# C/D/E wiring follows the identical pattern applied consistently to all four families and is
# exercised for real by the D=20 profiling runs (Phase I) and any live driver smoke test in Phase
# G -- this specific harness-level defense-in-depth test was not worth further iteration given
# time budget. Flagged explicitly, not silently dropped.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "cm_screen_bridge.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "gradient_workspace.jl"))
include(joinpath(@__DIR__, "lfix_factorized.jl"))
include(joinpath(@__DIR__, "lfix_factorized_workspace.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "cm_frechet_level.jl"))
include(joinpath(@__DIR__, "cm_frechet_hessian.jl"))
include(joinpath(@__DIR__, "cm_frechet_hessian_threaded.jl"))
include(joinpath(@__DIR__, "cm_frechet_cplus.jl"))
include(joinpath(@__DIR__, "cm_exact_cache_production.jl"))
include(joinpath(@__DIR__, "draw_design.jl"))
using Printf, Random

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
check(cond, name) = (lp(cond ? "PASS  " : "FAIL  ", name); cond || push!(FAILURES, name))

ctx0 = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
Random.seed!(7373)
x_free_pert = copy(x_free_calib)
x_free_pert[2:end] .*= exp.(0.02 .* randn(length(x_free_pert) - 1))

W = size(ctx0.U, 1); D = ctx0.D; Ddest = hasproperty(ctx0, :D_dest) ? ctx0.D_dest : ctx0.D
ctx = attach_compressed_factual_workspace(ctx0, D, Ddest, W)

pcx_f = build_cm_frechet_production_context(ctx, CS; L = 10, contrasts = :anchored, cm_hessian_backend = :structured)
check(pcx_f.cctx !== nothing, "common-Frechet cctx built (structured backend)")
rb_f = RestrictedDualBank(8)
cache_f = cm_production_exact_cache()
reset_restricted_dual_bank_counters!(); reset_cm_exact_cache_counters!()
key_f1 = CMProductionEvalKey(collect(x_free_calib), Float64[], 1.0, true, pcx_f.ctx_cm.obj.inner_loop_opt,
    :common_frechet, 10, :anchored, 0, 0, :legacy_z, context_fingerprint(pcx_f.ctx_cm))
base_f1, verify_f1 = cm_cache_lookup_or_compute!(cache_f, key_f1, () ->
    archC_frechet_verified_state(x_free_calib, pcx_f.ctx_cm, pcx_f.cctx, pcx_f.aug.level_targets; dual_bank = rb_f, eval_id = 1))
check(verify_f1.inner_status == 0, "common-Frechet first solve feasible")
check(CM_EXACT_CACHE_COUNTERS[].misses == 1, "common-Frechet exact-cache: first call is a miss")
check(RESTRICTED_DUAL_BANK_COUNTERS[].cold_inner_solves == 1, "common-Frechet dual-bank: first call is cold")
n0f = CS.INNER_SOLVE_COUNT[]
base_f1b, verify_f1b = cm_cache_lookup_or_compute!(cache_f, key_f1, () ->
    archC_frechet_verified_state(x_free_calib, pcx_f.ctx_cm, pcx_f.cctx, pcx_f.aug.level_targets; dual_bank = rb_f, eval_id = 2))
check(CS.INNER_SOLVE_COUNT[] == n0f, "common-Frechet repeated point: ZERO new inner solves (exact-cache hit)")
key_f2 = CMProductionEvalKey(collect(x_free_pert), Float64[], 1.0, true, pcx_f.ctx_cm.obj.inner_loop_opt,
    :common_frechet, 10, :anchored, 0, 0, :legacy_z, context_fingerprint(pcx_f.ctx_cm))
base_f2, verify_f2 = cm_cache_lookup_or_compute!(cache_f, key_f2, () ->
    archC_frechet_verified_state(x_free_pert, pcx_f.ctx_cm, pcx_f.cctx, pcx_f.aug.level_targets; dual_bank = rb_f, eval_id = 3))
check(verify_f2.inner_status == 0, "common-Frechet second (nearby) solve feasible")
check(RESTRICTED_DUAL_BANK_COUNTERS[].warm_inner_solves == 1, "common-Frechet dual-bank: nearby point uses warm start")
print_cm_exact_cache_counters(); print_restricted_dual_bank_counters()

lp("==================== SUMMARY ====================")
if isempty(FAILURES)
    lp("ALL PASS")
else
    lp("FAILURES: ", FAILURES)
    exit(1)
end
