# Small diagnostic: c14_allocation_audit.jl's compressed-path value eval at the calibration point
# (gp0*1.01/zfree0 convention) completed in 0.63s, allocating 324MB, and produced NONE of
# evaluate_fullA_screened_ranged's own compressed-inner-solve @prof labels -- meaning it took a
# pre-solve SCREEN shortcut rather than a real KNITRO solve. But the SAME call convention in
# timing_harness.jl (already-committed, historically validated on this branch) reports ~20-21s for
# the "cold value eval, complete callback" at what looks like the identical point. This script
# isolates exactly which screen fired and why, to resolve the discrepancy honestly rather than
# report a number without understanding it.
# Include order matches c10_d20_production_driver.jl EXACTLY (not the ad-hoc subset used by
# earlier c14 scripts) -- see this file's own investigation note below for why that ad-hoc subset
# was silently wrong (missing compressed_cc_inner.jl/dual_bank.jl and several others produced a
# real, reproducible bug: Delta_dual computed as -0.0 instead of the correct 0.2308841490034606,
# with no error thrown -- confirmed by cross-checking against timing_harness.jl, unmodified, which
# DOES include the full c10_d20_production_driver.jl chain and correctly reproduces the historical
# value).
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "lfix_buffer_reuse.jl"))
include(joinpath(@__DIR__, "bandwidth_cache_policy.jl"))
include(joinpath(@__DIR__, "fast_range_screen.jl"))
include(joinpath(@__DIR__, "dual_bank.jl"))
using Printf, Random, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))
Random.seed!(20260719)
t0 = time()
ctx = d20_real_setup(W = 80000, δ = 1.0, find_smallest = true)
pe = build_pivot_elimination(ctx)
rsc = build_ranged_screen_context(ctx)
D = ctx.D; D2 = D^2
lp(@sprintf(">>> ctx built in %.1fs", time()-t0))

x_free_from_w2(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, D), pe)
gp0 = ctx.θ0_up[3+D]
xf0 = x_free_from_w2(vcat(gp0 * 1.01, zfree0))

lp(">>> call 1 (matches timing_harness.jl's OWN first call, cache=nothing, warm=false):")
t0 = time()
r1, m1 = evaluate_fullA_screened_ranged(xf0, ctx, rsc; moment_representation = :compressed,
    cache = nothing, use_cache = false, warm = false, tag = "", pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
lp(@sprintf("  wall=%.4fs  screen_status=%s  inner_status=%s  Delta_dual=%s  cache_hit=%s",
    time()-t0, string(get(m1, :screen_status, missing)), string(r1.inner_status), string(r1.Delta_dual), string(get(r1, :cache_hit, missing))))
lp("  full meta = ", m1)
