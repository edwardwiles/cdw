# 2026-07-24 release: regression test for a bug found during this release's own audit (not in
# the prior 2026-07-23 session): every restricted-family objective-bundle rebuild
# (build_cm_augmented_obj / cm_production_bundle.jl's archB rebuild / cm_hessian_architectures.jl
# / common_marginals_interval.jl / c12b_interval_common_marginals_moments.jl / cm_meanzc_moments.jl
# / cm_originzc_moments.jl) constructs a FRESH `PsiObjectiveBundleImplicit` without forwarding
# `threshold_state`, which defaults to `ThresholdAbortState()` (threshold=Inf) -- silently
# DISABLING Part C's threshold-10 early-abort for every CM / CM+mean-ZC / origin-ZC production
# entry point, even though the base unrestricted `ctx.obj` had a finite threshold configured by
# `d20_real_setup`. Fixed by forwarding `threshold_state = obj0.threshold_state` (or
# `obj_cm.threshold_state` for the second, already-CM rebuild in cm_production_bundle.jl) at each
# site. This test proves the fix: builds each restricted-family production context at
# delta=1 (where task step 10/11 requires the active threshold to be exactly 10.0) and checks
# `pcx.ctx_cm.obj.threshold_state.threshold == 10.0`, NOT `Inf`. Construction-only -- no KNITRO
# inner solve is run, so this is cheap relative to a full campaign test.
include(joinpath(@__DIR__, "draw_design.jl"))
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
include(joinpath(@__DIR__, "gradient_workspace.jl"))
include(joinpath(@__DIR__, "lfix_factorized.jl"))
include(joinpath(@__DIR__, "lfix_factorized_workspace.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "cm_meanzc_cplus.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
include(joinpath(@__DIR__, "cm_originzc_target_layout.jl"))
include(joinpath(@__DIR__, "cm_originzc_moments.jl"))
include(joinpath(@__DIR__, "cm_originzc_production.jl"))
include(joinpath(@__DIR__, "cm_originzc_cplus.jl"))
include(joinpath(@__DIR__, "cm_originzc_config.jl"))
include(joinpath(@__DIR__, "cm_originzc_checkpoint.jl"))
using Random

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
end

println("="^78)
println("Threshold-propagation regression: delta=1 must resolve active threshold=10.0")
println("in EVERY restricted-family production objective bundle, not just the base ctx.obj")
println("="^78)

Random.seed!(20260719)
ctx = d20_real_setup(W = 80000, δ = 1.0, find_smallest = true)
check("base ctx.obj.threshold_state.threshold == 10.0 (sanity check on the base context)",
      ctx.obj.threshold_state.threshold == 10.0)

L = 50
snaps = nested_grid_sequence([10, 20, 50])
probs = snaps[L]

println("\n>>> Flexible CM (build_cm_production_context, archB rebuild path)")
pcx_cm = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = probs)
check("CM: threshold_state.threshold == 10.0 (was Inf pre-fix)",
      pcx_cm.ctx_cm.obj.threshold_state.threshold == 10.0)

println("\n>>> CM + mean/ZC (build_cm_meanzc_production_context)")
pcx_mz = build_cm_meanzc_production_context(ctx, CS; L = L, K_mean = 2, K_pair = 0,
    contrasts = :orthonormal, meanzc_basis = :direct, probs = probs)
check("CM+meanZC: threshold_state.threshold == 10.0 (was Inf pre-fix)",
      pcx_mz.ctx_cm.obj.threshold_state.threshold == 10.0)

println("\n>>> Origin-specific ZC (build_originzc_production_context)")
layout = OriginByPowerLayout(ctx.D, 2, 0)
pcx_oz = build_originzc_production_context(ctx, CS, layout)
check("origin-ZC: threshold_state.threshold == 10.0 (was Inf pre-fix)",
      pcx_oz.ctx_cm.obj.threshold_state.threshold == 10.0)

println("\n>>> resolve_threshold_for_delta compatibility rule applies identically across families")
check("delta=9 disables (Inf) per the safety-margin rule", CS.resolve_threshold_for_delta(9.0) == Inf)
check("delta=1 resolves to 10.0", CS.resolve_threshold_for_delta(1.0) == 10.0)

println()
n_fail = length(FAILURES)
println("="^78)
println(">>> RESULT: ", n_fail == 0 ? "ALL PASS" : "$(n_fail) FAILURE(S): $(join(FAILURES, ", "))")
println("="^78)
n_fail == 0 || error("test_threshold_propagation_regression.jl: $(n_fail) assertion(s) FAILED")
