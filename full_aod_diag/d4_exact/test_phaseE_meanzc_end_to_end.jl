# Post-hoc gap-closing test (found during Phase E's D=20 benchmark, 2026-07-26): CM+ZC
# (marginal_restriction=:common_flexible, cm_extension=:cm_plus_equal_means_zero_covariance) was
# NEVER actually exercised end-to-end by any of this remediation's Phase B1/C/D/E gates so far --
# all of those gates only built/solved a plain flexible-CM pcx directly. This surfaced a real bug:
# build_cm_meanzc_bin_ctx's own CMBinHessCtx(...) constructor call was missing the new
# inner_fg_backend field Phase B1 added to the struct (a SECOND, separate construction site from
# the one Phase B1 actually edited in build_cm_bin_ctx) -- MethodError on every CM+ZC build since
# that commit, only caught now because Phase E's benchmark script was the first thing in this
# whole remediation to actually construct a real CM+ZC pcx. Fixed in cm_meanzc_production.jl.
#
# This test exercises the FULL chain (dense-reference FG + exact cache + dual bank + compressed-
# core workspace, i.e. everything Phases B1/C/D/E touch) through archC_meanzc_verified_state
# directly, at D=4, to confirm CM+ZC genuinely works end-to-end now, not just "looks structurally
# similar to flexible-CM's own already-tested chain."
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
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "cm_exact_cache_production.jl"))
include(joinpath(@__DIR__, "draw_design.jl"))
using Printf, Random

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
check(cond, name) = (lp(cond ? "PASS  " : "FAIL  ", name); cond || push!(FAILURES, name))

ctx0 = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
Random.seed!(5151)
x_free_pert = copy(x_free_calib)
x_free_pert[2:end] .*= exp.(0.02 .* randn(length(x_free_pert) - 1))
νvec = [1.0]

W = size(ctx0.U, 1); D = ctx0.D; Ddest = hasproperty(ctx0, :D_dest) ? ctx0.D_dest : ctx0.D
ctx = attach_compressed_factual_workspace(ctx0, D, Ddest, W)   # Phase E wiring, exercised here too

pcx = build_cm_meanzc_production_context(ctx, CS; L = 10, K_mean = 1, K_pair = 1, contrasts = :anchored)
check(pcx.cctx.inner_fg_backend == :dense_reference, "CM+ZC cctx correctly tagged :dense_reference (the only supported value)")

rb = RestrictedDualBank(8)
cache = cm_production_exact_cache()
reset_restricted_dual_bank_counters!()
reset_cm_exact_cache_counters!()

key1 = CMProductionEvalKey(collect(x_free_calib), collect(νvec), 1.0, true, pcx.ctx_cm.obj.inner_loop_opt,
    :cm_meanzc, 10, :anchored, 1, 1, :legacy_z, context_fingerprint(pcx.ctx_cm))

base1, verify1 = cm_cache_lookup_or_compute!(cache, key1, () ->
    archC_meanzc_verified_state(x_free_calib, νvec, pcx.ctx_cm, pcx.cctx; dual_bank = rb, eval_id = 1))
check(verify1.inner_status == 0, "CM+ZC first solve (calib point) feasible")
check(CM_EXACT_CACHE_COUNTERS[].misses == 1, "CM+ZC exact-cache: first call is a miss")
check(RESTRICTED_DUAL_BANK_COUNTERS[].cold_inner_solves == 1, "CM+ZC dual-bank: first call is cold")

# repeat identical point -- must be an exact-cache hit, zero new inner solves.
n0 = CS.INNER_SOLVE_COUNT[]
base1b, verify1b = cm_cache_lookup_or_compute!(cache, key1, () ->
    archC_meanzc_verified_state(x_free_calib, νvec, pcx.ctx_cm, pcx.cctx; dual_bank = rb, eval_id = 2))
n1 = CS.INNER_SOLVE_COUNT[]
check(n1 == n0, "CM+ZC repeated identical point: ZERO new inner solves (exact-cache hit)")
check(CM_EXACT_CACHE_COUNTERS[].hits == 1, "CM+ZC exact-cache: repeat is a hit")

# nearby point -- new inner solve, should be warm (bank has one entry).
key2 = CMProductionEvalKey(collect(x_free_pert), collect(νvec), 1.0, true, pcx.ctx_cm.obj.inner_loop_opt,
    :cm_meanzc, 10, :anchored, 1, 1, :legacy_z, context_fingerprint(pcx.ctx_cm))
base2, verify2 = cm_cache_lookup_or_compute!(cache, key2, () ->
    archC_meanzc_verified_state(x_free_pert, νvec, pcx.ctx_cm, pcx.cctx; dual_bank = rb, eval_id = 3))
check(verify2.inner_status == 0, "CM+ZC second (nearby) solve feasible")
check(CM_EXACT_CACHE_COUNTERS[].misses == 2, "CM+ZC exact-cache: distinct point is a miss")
check(RESTRICTED_DUAL_BANK_COUNTERS[].warm_inner_solves == 1, "CM+ZC dual-bank: nearby point uses a warm start")

print_cm_exact_cache_counters()
print_restricted_dual_bank_counters()

lp("==================== SUMMARY ====================")
if isempty(FAILURES)
    lp("ALL PASS")
else
    lp("FAILURES: ", FAILURES)
    exit(1)
end
