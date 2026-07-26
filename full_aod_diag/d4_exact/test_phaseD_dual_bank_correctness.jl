# Phase D remediation (production-audit continuation, 2026-07-26) correctness gate: dual-bank
# warm starts (RestrictedDualBank/select_warm_start_restricted) for the restricted-family drivers.
# Tests the same archC_verified_state call path cb_F! now uses.
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
include(joinpath(@__DIR__, "draw_design.jl"))
using Printf, Random, Statistics, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
check(cond, name) = (lp(cond ? "PASS  " : "FAIL  ", name); cond || push!(FAILURES, name))

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
Random.seed!(4242)

pcx = build_cm_production_context(ctx, CS; L = 10, contrasts = :anchored)
rb = RestrictedDualBank(8)
reset_restricted_dual_bank_counters!()

# 1. First call: bank empty, obj.x not yet valid (fresh obj) -- must fall back to neutral.
base1, verify1 = archC_verified_state(x_free_calib, pcx.ctx_cm, pcx.cctx; dual_bank = rb, eval_id = 1)
check(verify1.inner_status == 0, "first (bank-empty) call solves feasibly")
check(RESTRICTED_DUAL_BANK_COUNTERS[].cold_inner_solves == 1, "first call recorded as cold (neutral start)")
check(length(rb.xfree_history) == 1, "bank has one entry after first successful solve")

# 2. Second call, a NEARBY perturbed point -- bank now has one entry, should be selected (warm).
x_free_near = copy(x_free_calib)
x_free_near[2:end] .*= exp.(0.01 .* randn(length(x_free_near) - 1))
base2, verify2 = archC_verified_state(x_free_near, pcx.ctx_cm, pcx.cctx; dual_bank = rb, eval_id = 2)
check(verify2.inner_status == 0, "second (nearby) call solves feasibly")
check(RESTRICTED_DUAL_BANK_COUNTERS[].warm_inner_solves == 1, "second call recorded as warm (bank entry selected)")
check(length(rb.xfree_history) == 2, "bank has two entries after second successful solve")

# 3. A FRESH pcx (no dual_bank at all) at the SAME two points must give isapprox-equal Delta_dual
# to the bank-assisted run above -- warm-starting must never change the CONVERGED answer, only how
# fast/from-where KNITRO gets there.
pcx_nobank = build_cm_production_context(ctx, CS; L = 10, contrasts = :anchored)
base1_nb, verify1_nb = archC_verified_state(x_free_calib, pcx_nobank.ctx_cm, pcx_nobank.cctx)
base2_nb, verify2_nb = archC_verified_state(x_free_near, pcx_nobank.ctx_cm, pcx_nobank.cctx)
check(isapprox(verify1.Delta_dual, verify1_nb.Delta_dual; atol = 1e-8, rtol = 1e-8),
      "bank-on vs bank-off Delta_dual agree at point 1 (diff=$(abs(verify1.Delta_dual-verify1_nb.Delta_dual)))")
check(isapprox(verify2.Delta_dual, verify2_nb.Delta_dual; atol = 1e-8, rtol = 1e-8),
      "bank-on vs bank-off Delta_dual agree at point 2 (diff=$(abs(verify2.Delta_dual-verify2_nb.Delta_dual)))")

print_restricted_dual_bank_counters()

lp("==================== SUMMARY ====================")
if isempty(FAILURES)
    lp("ALL PASS")
else
    lp("FAILURES: ", FAILURES)
    exit(1)
end
